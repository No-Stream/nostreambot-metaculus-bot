import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Literal, NamedTuple, get_args

from forecasting_tools import ForecastReport, MetaculusApi

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.api_preflight import verify_api_identity, verify_metaculus_api_identity
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    MANTIC_API_BASE_URL,
    MANTIC_TOURNAMENT_END_DATE,
    MANTIC_TOURNAMENT_ID,
    METACULUS_CUP_ID,
    PERSIST_RESEARCH_ENABLED_ENV,
    PLATFORM_MANTIC,
    PLATFORM_METACULUS,
    TEST_QUESTIONS_OVERRIDE_ENV,
    TOURNAMENT_ID,
    check_fall_cup_reminder,
    check_tournament_dates,
    credit_alerts_active,
    donated_openrouter_key_enabled,
    env_flag_enabled,
)
from metaculus_bot.credit_telemetry import (
    CreditTelemetry,
    drain_litellm_callbacks,
    get_probed_donated_key_state,
    install_role_spend_tracker,
    log_role_spend,
    log_run_summary,
)
from metaculus_bot.fallback_openrouter import (
    check_deprecation_alerts_and_exit,
    get_credit_key_fallback_count,
    get_donated_404_fallback_count,
    get_generic_key_fallback_count,
    has_deprecation_alerts,
)
from metaculus_bot.fetch_hardening import apply_fetch_hardening
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    PARSER_LLM,
    RESEARCHER_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
)
from metaculus_bot.mantic import (
    build_mantic_client,
    get_post_drop_count,
    preflight_mantic_tournaments,
    reset_post_drop_count,
)
from metaculus_bot.publish_hardening import apply_publish_hardening
from metaculus_bot.research.persistence import ResearchPersistenceWriter

logger = logging.getLogger(__name__)


RunMode = Literal["tournament", "minibench", "quarterly_cup", "metaculus_cup", "test_questions", "mantic"]


class CliArgs(NamedTuple):
    """What argv decides: the run mode, and the optional ``--only-posts`` narrowing of it."""

    run_mode: RunMode
    only_posts: frozenset[int] | None


def _assert_personal_keys_only() -> None:
    """Fail shut unless the Metaculus-donated OpenRouter key is switched off for this process.

    Metaculus donated ``OAI_ANTH_OPENROUTER_KEY`` for its own tournaments, so a run that
    forecasts for Mantic may spend only the operator's personal keys. The switch has to be an
    environment variable set BEFORE the process starts rather than something this function
    could flip: the roster's module-level ``GeneralLlm`` objects (``llm_configs``) freeze their
    api_key at import, and ``main.py`` imports them before ``main`` runs. So the only safe
    thing to do when it still reads on is to stop, before any fetch or spend.
    """
    if donated_openrouter_key_enabled():
        raise RuntimeError(
            f"A Mantic run may spend only personal API keys, but {DONATED_OPENROUTER_KEY_ENABLED_ENV} does "
            "not read false. Set it to false in the environment before starting the process: the roster "
            "freezes its OpenRouter key at import, so the donated key cannot be switched off from here."
        )


def _configure_process(run_mode: RunMode) -> None:
    """Set up logging levels and install the client hardening / identity preflight.

    Done here (the runtime entry point) rather than at module import so test imports and
    library consumers don't inherit these global mutations. The run mode decides which
    platform host the identity preflight vets, and mantic mode fails shut on the
    donated-key switch first.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Per-question tracing at DEBUG; openai-agents is noisy at INFO. See docs/architecture.md "CLI startup wiring".
    logging.getLogger("metaculus_bot.forecaster").setLevel(logging.DEBUG)
    logging.getLogger("openai.agents").setLevel(logging.ERROR)
    # Its only warnings are "hidden param cost 0.0 vs response object cost" (litellm has no price map for the
    # current roster, so every call over 5 cents fires one) and a callback-registration notice; it books the
    # larger cost regardless, and spend is accounted by credit_telemetry's CREDIT_ROLE_SPEND ledger.
    logging.getLogger("forecasting_tools.ai_models.resource_managers.monetary_cost_manager").setLevel(logging.ERROR)

    # A single hung publish POST would block the whole batch. See docs/architecture.md "CLI startup wiring".
    apply_publish_hardening()

    # One transient 403/429/5xx would otherwise kill the run. See docs/architecture.md "CLI startup wiring".
    apply_fetch_hardening()

    # Reset here, not in forecast_questions: that fetch runs first. See docs/architecture.md "CLI startup wiring".
    reset_post_drop_count()

    # One-shot and unauthenticated, so no token reaches a hijacked host. See docs/architecture.md "CLI startup wiring".
    if run_mode == "mantic":
        _assert_personal_keys_only()
        verify_api_identity(MANTIC_API_BASE_URL)
    else:
        verify_metaculus_api_identity()


def _parse_post_ids(text: str) -> frozenset[int]:
    """The ``--only-posts`` value: comma-separated post ids, ``650`` or ``650,651``."""
    try:
        return frozenset(int(token) for token in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integer post ids, got {text!r}") from exc


def _parse_cli_args() -> CliArgs:
    """Read ``--mode`` and the optional ``--only-posts`` filter off argv.

    ``--only-posts`` narrows a tournament-shaped mode to the listed post ids (the one-question
    paid smoke run). It is refused with ``test_questions``, whose evergreen set is not a
    tournament's open questions: a filter that silently did nothing on a paid run would be
    worse than a usage error.
    """
    parser = argparse.ArgumentParser(description="Run the Q1TemplateBot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(get_args(RunMode)),
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--only-posts",
        type=_parse_post_ids,
        default=None,
        metavar="POST_IDS",
        help=(
            "Comma-separated post ids: forecast only these of the tournament's open questions "
            "(the one-question smoke run). Tournament-shaped modes only."
        ),
    )
    args = parser.parse_args()
    run_mode: RunMode = args.mode
    only_posts: frozenset[int] | None = args.only_posts
    if only_posts is not None and run_mode == "test_questions":
        parser.error("--only-posts narrows a tournament's open questions and does not apply to --mode test_questions")
    return CliArgs(run_mode=run_mode, only_posts=only_posts)


async def _forecast_with_callback_drain(start_forecast: Callable[[], Awaitable[list[Any]]]) -> list[Any]:
    """Run one forecast coroutine, then drain litellm's success callbacks on the SAME loop.

    The ``CREDIT_ROLE_SPEND`` ledger is fed by a litellm callback that the logging worker
    delivers a tick after each completion, on the loop the completion ran on; the worker's
    queue is bound to that loop and ``asyncio.run`` tears it down on return, so the drain
    cannot sit beside ``log_role_spend`` in ``main``'s ``finally`` — it has to happen here.
    Takes a factory rather than a coroutine so the coroutine is created inside the loop it
    runs on (and so a test stub that closes this wrapper unrun leaves nothing un-awaited).
    """
    try:
        return await start_forecast()
    finally:
        await drain_litellm_callbacks()


def _test_questions_source(template_bot: TemplateForecaster) -> Callable[[], Awaitable[list[Any]]]:
    """Forecast-factory over the evergreen example set, or a TEST_QUESTIONS_OVERRIDE list.

    Resolving the URLs is a synchronous Metaculus fetch and stays OUTSIDE the event loop,
    where it has always been; only the forecast itself is deferred into the factory.
    """
    EXAMPLE_QUESTIONS = [
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
        # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice  # noqa: ERA001  # parked test question, kept for one-line re-enable
        "https://www.metaculus.com/questions/20683/which-ai-world/",  # Scott Aaronson's five AI worlds
        "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
    ]
    template_bot.skip_previously_forecasted_questions = (
        False  # obviously, we need to rerun test q predictions to test them :)
    )
    # Unset falls back to EXAMPLE_QUESTIONS. See docs/operations.md "The one-question smoke test".
    override_urls = os.environ.get(TEST_QUESTIONS_OVERRIDE_ENV, "").replace(",", " ").split()
    question_urls = override_urls or EXAMPLE_QUESTIONS
    if override_urls:
        logger.info(
            "TEST_QUESTIONS_OVERRIDE set: forecasting %d override question(s) instead of the evergreen set",
            len(override_urls),
        )
    questions = [MetaculusApi.get_question_by_url(url) for url in question_urls]
    return lambda: template_bot.forecast_questions(questions, return_exceptions=True)


async def _forecast_nothing() -> list[Any]:
    """The factory for an ``--only-posts`` filter that matched none of the open questions."""
    return []


def _post_ids_csv(post_ids: Iterable[int | None]) -> str:
    """Sorted comma-separated ids for the ``ONLY_POSTS`` marker; ``none`` for an empty list."""
    known_ids = sorted(post_id for post_id in post_ids if post_id is not None)
    return ",".join(str(post_id) for post_id in known_ids) or "none"


def _tournament_source(
    template_bot: TemplateForecaster,
    tournament_id: int | str,
    only_posts: frozenset[int] | None,
) -> Callable[[], Awaitable[list[Any]]]:
    """Forecast-factory over one tournament's open questions, narrowed to ``only_posts`` when set.

    Unfiltered, this is the framework's own ``forecast_on_tournament``, untouched. Filtered, the
    question-list fetch it makes internally runs here instead, on the same injected client (the
    ``ManticClient`` in mantic mode), and only the posts asked for reach ``forecast_questions``;
    a filter that matches nothing forecasts nothing rather than the whole tournament. Like the
    URL resolves in ``_test_questions_source`` the fetch is synchronous and stays outside the
    event loop, with only the forecast deferred into the factory. The ``ONLY_POSTS`` marker
    records the request against what the tournament held open, so a smoke run's log says
    which question it spent on.
    """
    if only_posts is None:
        return lambda: template_bot.forecast_on_tournament(tournament_id, return_exceptions=True)
    open_questions = template_bot.metaculus_client.get_all_open_questions_from_tournament(tournament_id)
    matched = [question for question in open_questions if question.id_of_post in only_posts]
    logger.info(
        f"ONLY_POSTS: requested={_post_ids_csv(only_posts)} "
        f"matched={_post_ids_csv(question.id_of_post for question in matched)} "
        f"dropped={len(open_questions) - len(matched)}"
    )
    if not matched:
        logger.warning(
            "--only-posts matched none of the %d open question(s) in tournament %s; forecasting nothing",
            len(open_questions),
            tournament_id,
        )
        return _forecast_nothing
    return lambda: template_bot.forecast_questions(matched, return_exceptions=True)


def _question_source(
    template_bot: TemplateForecaster, run_mode: RunMode, *, only_posts: frozenset[int] | None
) -> Callable[[], Awaitable[list[Any]]]:
    """Resolve one run mode to the factory that forecasts its questions.

    Returns a factory rather than forecasting here so that every mode goes through the
    single ``asyncio.run`` + callback drain in ``_run_forecasts``. The drain has to happen
    on the loop the completions ran on, and stating it once means a mode added here cannot
    silently report no per-role spend by forgetting to wrap itself.

    Every tournament-shaped mode pins ``skip_previously_forecasted_questions`` on so a
    re-run can't re-spend on questions already forecast, and honours ``only_posts``
    (``_tournament_source``); the parser refuses that filter for ``test_questions``.
    """
    if run_mode == "tournament":
        # to not risk explosive spend, we won't update preds.
        template_bot.skip_previously_forecasted_questions = True
        return _tournament_source(template_bot, TOURNAMENT_ID, only_posts)
    if run_mode == "minibench":
        # to not risk explosive spend, we won't update preds.
        template_bot.skip_previously_forecasted_questions = True
        return _tournament_source(template_bot, MetaculusApi.CURRENT_MINIBENCH_ID, only_posts)
    if run_mode in ("quarterly_cup", "metaculus_cup"):
        # Regularly open questions; to not risk explosive spend, we won't update preds.
        template_bot.skip_previously_forecasted_questions = True
        return _tournament_source(template_bot, METACULUS_CUP_ID, only_posts)
    if run_mode == "mantic":
        # Mantic's platform via the ManticClient main injects; to not risk explosive spend, we won't update preds.
        template_bot.skip_previously_forecasted_questions = True
        return _tournament_source(template_bot, MANTIC_TOURNAMENT_ID, only_posts)
    if run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        return _test_questions_source(template_bot)
    raise ValueError(f"Invalid run mode: {run_mode}")


def persisted_platform(run_mode: RunMode) -> str:
    """The question platform this run's research records are archived under.

    Additive next to ``tournament_id``. Archive filenames are not namespaced by platform, and
    ``scripts/download_research.build_archive`` groups records on the bare qid, so this field
    tells the two platforms apart inside a group and is what an analysis keyed on bare post
    ids across both platforms has to filter on; it cannot stop a Mantic id and a Metaculus id
    that meet from merging into one ``by_qid`` / ``latest`` / manifest entry. The margin is not
    "hundreds versus tens of thousands": the evergreen ``test_questions`` set puts Metaculus
    ids 578, 14333 and 20683 in the archive, so with the open Mantic posts in the 650s the next
    Metaculus key above them is 14333, about 13,700 Mantic posts away.
    """
    return PLATFORM_MANTIC if run_mode == "mantic" else PLATFORM_METACULUS


def persisted_tournament_id(run_mode: RunMode) -> str:
    """The tournament label this run's research records are archived under.

    Pure, and keyed on the run mode rather than pinned to ``TOURNAMENT_ID``, because residual
    analysis buckets and joins on this label: a cup run labelled with the BOT tournament's slug
    is silent data corruption. ``test_questions`` deliberately keeps ``TOURNAMENT_ID``, and an
    unknown mode raises rather than mislabel a whole run's archive. Reasoning:
    docs/architecture.md "The research-archive label".
    """
    if run_mode in ("tournament", "test_questions"):
        return TOURNAMENT_ID
    if run_mode == "minibench":
        return str(MetaculusApi.CURRENT_MINIBENCH_ID)
    if run_mode in ("quarterly_cup", "metaculus_cup"):
        return METACULUS_CUP_ID
    if run_mode == "mantic":
        return MANTIC_TOURNAMENT_ID
    raise ValueError(f"Invalid run mode: {run_mode}")


def _run_forecasts(
    template_bot: TemplateForecaster, run_mode: RunMode, *, only_posts: frozenset[int] | None = None
) -> list[Any]:
    """Forecast one run mode's questions, on one event loop, with the callback drain.

    The only ``asyncio.run`` in the module: the loop is created here and torn down on
    return, and ``_forecast_with_callback_drain`` drains litellm's success callbacks
    inside it while the queue bound to it is still alive.
    """
    source = _question_source(template_bot, run_mode, only_posts=only_posts)
    return asyncio.run(_forecast_with_callback_drain(source))


def _check_tournament_dates(run_mode: RunMode) -> bool:
    """Run the mode's stale-slug check at startup; True only when the MANTIC slug is past its end date.

    ``check_tournament_dates`` (constants.py) warns from the UTC day after a slug's end date (the
    constant names the last open day) and raises at the hard stop two weeks later, for the
    Metaculus bot tournament and for Mantic alike. In between, the Metaculus tournament stays
    advisory: its questions are open for weeks and a fortnight of warnings costs nothing. On
    Mantic that fortnight is a silent forfeit. A zero-question run is green, Series 2 opens under
    a slug that does not exist yet, and every hourly run would keep fetching the ended preseason
    and exit 0, about seventy questions at Series 1's rate (edge review item 5). So in mantic mode
    the verdict is held and reddens the run after publishing, the same shape as the fall-cup
    reminder; the shared hard stop is untouched. The cup and minibench slugs carry no end date and
    are not checked.
    """
    if run_mode == "tournament":
        check_tournament_dates(logger)
        return False
    if run_mode == "mantic":
        return check_tournament_dates(
            logger, tournament_id=MANTIC_TOURNAMENT_ID, end_date_str=MANTIC_TOURNAMENT_END_DATE
        )
    return False


def main() -> None:
    """Command-line entry-point for running the TemplateForecaster.

    main.py delegates here, so GitHub Actions invoking ``python main.py`` and a direct
    ``python -m metaculus_bot.cli`` behave identically. Order matters: the mode is parsed
    first, then ``_configure_process`` installs the hardening patches and runs the
    fail-shut and identity checks, and only then is any platform token read.
    """
    run_mode, only_posts = _parse_cli_args()
    _configure_process(run_mode)

    # ERROR now, red exit after publishing, every run mode. See docs/operations.md "the exit ladder".
    fall_cup_reminder = check_fall_cup_reminder(logger)
    mantic_tournament_stale = _check_tournament_dates(run_mode)

    # Wire research persistence if enabled (production GHA runs set this env var)
    research_writer = None
    research_sink = None
    if env_flag_enabled(PERSIST_RESEARCH_ENABLED_ENV):
        research_writer = ResearchPersistenceWriter(
            run_mode=run_mode,
            platform=persisted_platform(run_mode),
            tournament_id=persisted_tournament_id(run_mode),
            run_id=os.environ.get("GITHUB_RUN_ID", "local"),
        )
        research_sink = research_writer.record

    # dict[str, Any] because "forecasters" holds a list the parent's invariant annotation cannot express.
    llms: dict[str, Any] = {
        "forecasters": FORECASTER_LLMS,
        "stacker": STACKER_LLM,
        "analyzer": DISAGREEMENT_ANALYZER_LLM,
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }
    # Built after _configure_process, so the key check and identity preflight pass before the token is read.
    metaculus_client = build_mantic_client() if run_mode == "mantic" else None
    if metaculus_client is not None:
        # Two authenticated GETs, still before any spend. See docs/operations.md "Startup checks and robustness rules".
        preflight_mantic_tournaments(metaculus_client, MANTIC_TOURNAMENT_ID)
    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Ignored when 'forecasters' present
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        research_sink=research_sink,
        llms=llms,
        metaculus_client=metaculus_client,
    )

    # Installed before the first completion. See docs/operations.md "Credit telemetry and the refill floor".
    install_role_spend_tracker()
    credit_telemetry = CreditTelemetry()
    credit_telemetry.log_start()
    donated_below_floor = False
    # Empty until the forecasts return, so a crashed run's summary reads its money against zero questions.
    forecast_reports: list[Any] = []
    try:
        forecast_reports = _run_forecasts(template_bot, run_mode, only_posts=only_posts)
    finally:
        donated_below_floor = credit_telemetry.log_end_and_check_floor()
        log_role_spend()
        log_run_summary(n_questions=sum(isinstance(report, ForecastReport) for report in forecast_reports))
        # Records accumulate in memory all run: without this flush a crash archives nothing.
        if research_writer is not None:
            research_writer.flush()

    # Emit-then-raise: the breakdown must be recorded before this propagates. See docs/operations.md "the exit ladder".
    report_summary_error: Exception | None = None
    try:
        TemplateForecaster.log_report_summary(forecast_reports)
    # Boundary: holding ANY summary error, deliberately unnarrowed, is the point.
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # held until the breakdown is emitted, then re-raised
        report_summary_error = exc

    _report_degradation_and_exit(
        template_bot,
        report_summary_error=report_summary_error,
        donated_below_floor=donated_below_floor,
        fall_cup_reminder=fall_cup_reminder,
        mantic_tournament_stale=mantic_tournament_stale,
    )


def _report_degradation_and_exit(
    template_bot: TemplateForecaster,
    *,
    report_summary_error: Exception | None,
    donated_below_floor: bool,
    fall_cup_reminder: bool,
    mantic_tournament_stale: bool,
) -> None:
    """Emit the one-line degradation breakdown and decide the process exit status.

    Alert on degraded runs. Publication has already happened inside
    forecast_on_tournament / forecast_questions before this is called, so every Q that met
    MIN_FORECASTERS_TO_PUBLISH is on Metaculus regardless of exit status. Non-zero exit
    here just triggers the GitHub Actions red-check alert so the operator knows to
    investigate (forecaster drops, stacker fallback usage, research provider failures,
    etc. — see ``forecaster.py`` ``alertable_count``).

    EVERY non-zero exit path lives in this function, which is what lets the ``run_clean``
    predicate below be their exact complement.
    """
    bot_alertable = template_bot.alertable_count
    # Only the all-causes total enters alertable; its subsets double-count. See docs/operations.md "the exit ladder".
    alerts_active = credit_alerts_active()
    generic_fallback = get_generic_key_fallback_count()
    donated_404 = get_donated_404_fallback_count()
    credit_fallback = get_credit_key_fallback_count()
    suppressed_credit_fallback = 0 if alerts_active else credit_fallback
    # Only this process-global counter turns a forfeited post red. See docs/operations.md "Parse drops are counted".
    mantic_post_drops = get_post_drop_count()
    alertable = bot_alertable + generic_fallback - suppressed_credit_fallback + mantic_post_drops

    suppression_note = (
        ""
        if alerts_active
        else f" with {suppressed_credit_fallback} credit event(s) suppressed until "
        f"{CREDIT_ALERT_RESUME_DATE.isoformat()}"
    )
    # Omitted when nothing probed the donated key: "unknown" would read as a failed probe.
    probed_donated_key_state = get_probed_donated_key_state()
    donated_key_note = "" if probed_donated_key_state is None else f", donated_key={probed_donated_key_state.value}"
    # Rendered only when a post dropped, so the term explaining a non-zero alertable appears when it applies.
    mantic_drops_note = "" if mantic_post_drops == 0 else f", mantic_post_drops={mantic_post_drops}"
    # Emitted on EVERY path, and the "clean" phrase marks a clean run. See docs/operations.md "the exit ladder".
    run_clean = (
        report_summary_error is None
        and alertable <= 0
        and generic_fallback <= 0
        and not (donated_below_floor and alerts_active)
        and not fall_cup_reminder
        and not mantic_tournament_stale
        and not has_deprecation_alerts()
    )
    completion_phrase = "Run completed clean with" if run_clean else "Run completed with"
    breakdown = (
        f"{completion_phrase} {alertable} alertable degradation event(s) "
        f"(bot={bot_alertable}, personal_key_fallback={generic_fallback} of which "
        f"donated_404={donated_404}, credit={credit_fallback}{suppression_note}{donated_key_note}{mantic_drops_note});"
    )
    if report_summary_error is not None:
        # Re-raise rather than sys.exit so the traceback survives; it outranks the alertable exit below.
        logger.warning("%s re-raising the forecasting failure so CI marks this run red.", breakdown)
        raise report_summary_error
    if alertable > 0:
        logger.warning("%s exiting non-zero so CI marks this run red.", breakdown)
        sys.exit(1)
    if generic_fallback > 0:
        # Reachable only under suppression with every fallback credit-caused, so the line says so.
        logger.info("%s every fallback was a suppressed credit event, so this run stays green.", breakdown)
    elif run_clean:
        # The all-clear census line (see ``run_clean`` above).
        logger.info("%s nothing degraded, so this run stays green.", breakdown)
    else:
        # Counters are quiet but a red condition below still decides the exit, so no green claim here.
        logger.info("%s a post-summary check below decides the exit status.", breakdown)

    # Published normally; this is only the ask-Metaculus-for-a-top-up signal. See docs/operations.md "the exit ladder".
    if donated_below_floor:
        if alerts_active:
            sys.exit(1)
        logger.info(
            "Donated-key credit floor breached, but credit alerting is suppressed until %s, so this run exits zero.",
            CREDIT_ALERT_RESUME_DATE.isoformat(),
        )

    # Reminder signal only, retired by flipping FALL_CUP_CONFIGURED. See docs/operations.md "the exit ladder".
    if fall_cup_reminder:
        sys.exit(1)

    # A zero-question run is otherwise green. See docs/operations.md "Stale slug goes red".
    if mantic_tournament_stale:
        logger.error(
            "Mantic tournament %s is past MANTIC_TOURNAMENT_END_DATE (%s); re-point MANTIC_TOURNAMENT_ID and "
            "MANTIC_TOURNAMENT_END_DATE in constants.py. Exiting non-zero so CI marks this run red.",
            MANTIC_TOURNAMENT_ID,
            MANTIC_TOURNAMENT_END_DATE,
        )
        sys.exit(1)

    # Runs LAST so submission completed and other exits fire first. See docs/operations.md "the exit ladder".
    check_deprecation_alerts_and_exit()


if __name__ == "__main__":
    main()
