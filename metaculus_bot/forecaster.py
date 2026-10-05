import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import ForecastReport, ResearchWithPredictions
from forecasting_tools.data_models.questions import ConditionalQuestion, DateQuestion
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
)
from metaculus_bot.close_margin import format_close_margin_marker
from metaculus_bot.comment.formatting import (
    build_unified_explanation,
    format_forecaster_rationales_section,
    format_main_research_section,
    format_research_summary_with_models,
)
from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
    DEFAULT_MAX_CONCURRENT_RESEARCH,
    FORECASTER_SOFT_DEADLINE,
    MIN_FORECASTERS_TO_PUBLISH,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    TIME_BUDGET_MIN_VIABLE_S,
    TS_ANCHOR_CHART_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.degradation_counters import (
    DegradationSnapshot,
    alertable_total,
    format_conditional_stacking_summary,
    format_degradation_summary,
)
from metaculus_bot.drop_telemetry import (
    DROP_CAUSE_TIMEOUT_SOFT_DEADLINE,
    DROP_CAUSE_TIMEOUT_WALL_CLOCK,
    ForecasterDrop,
    classify_raised_drop_cause,
    emit_drop_telemetry,
)
from metaculus_bot.extreme_call import format_extreme_call_markers
from metaculus_bot.forecaster_runners import (
    run_binary_forecast,
    run_date_forecast,
    run_mc_forecast,
    run_numeric_forecast,
)
from metaculus_bot.llm_setup import prepare_llm_config
from metaculus_bot.member_forecast import NUMERIC_COMBINE_METHOD_UNRECORDED, format_numeric_aggregate_marker
from metaculus_bot.numeric.date_axis import numeric_qtype, numeric_view
from metaculus_bot.numeric.out_of_range_floor import floor_published_tails
from metaculus_bot.numeric.pchip_processing import log_pchip_summary, reset_pchip_stats
from metaculus_bot.performance_analysis.parsing import extract_model_display_name_from_reasoning
from metaculus_bot.publish_gate import (
    publish_skipped_closed_count,
    record_publish_skipped_closed,
    reset_publish_skipped_closed,
)
from metaculus_bot.publish_hardening import publish_attempt_failures, reset_publish_attempt_failures
from metaculus_bot.question_platform import question_platform
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.providers import (
    ResearchCallable,
)
from metaculus_bot.research.timeseries_anchor import _session_charts
from metaculus_bot.run_status import RUN_STATUS
from metaculus_bot.stacking_route import route_after_forecasts
from metaculus_bot.time_budget import (
    QuestionTimeBudget,
    build_question_time_budget,
    format_time_budget_marker,
)
from metaculus_bot.time_utils import _as_utc
from metaculus_bot.tool_runner import run_tools_for_forecaster
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot

logger = logging.getLogger(__name__)

# Sort sentinel for a missing close_time, so the tightest-close-first sort never compares None to a datetime.
_CLOSE_TIME_MAX = datetime.max.replace(tzinfo=UTC)

load_environment()


def _forecast_history_is_readable(question: MetaculusQuestion) -> bool:
    """True when the payload carries the ``my_forecasts`` field ``already_forecasted`` is derived from."""
    question_json = question.api_json.get("question") or {}
    return question_json.get("my_forecasts") is not None


def _drop_questions_with_unreadable_forecast_history(questions: Sequence[MetaculusQuestion]) -> list[MetaculusQuestion]:
    """The skip guard's fail-shut leg: a question whose ``my_forecasts`` field is unreadable is not eligible.

    The framework derives ``already_forecasted`` inside a blanket except that answers False, so a
    payload without the field (a list GET without ``with_cp=true``, an unauthenticated Mantic read,
    an API change) would read as never forecast and re-publish every question on every hourly run.
    One WARNING marker per dropped question keeps the drop visible in the telemetry archive.
    """
    readable: list[MetaculusQuestion] = []
    for question in questions:
        if _forecast_history_is_readable(question):
            readable.append(question)
            continue
        logger.warning(
            "SKIP_GUARD_UNREADABLE: question=%s post_id=%s platform=%s reason=my_forecasts_missing",
            question.id_of_question,
            question.id_of_post,
            question_platform(question),
        )
    if len(readable) != len(questions):
        logger.warning(
            "Dropped %d question(s) with no readable my_forecasts field; the skip guard fails shut",
            len(questions) - len(readable),
        )
    return readable


class TemplateForecaster(CompactLoggingForecastBot):
    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN,
        research_provider: ResearchCallable | None = None,
        max_questions_per_run: int | None = 10,
        is_benchmarking: bool = False,
        max_concurrent_research: int = DEFAULT_MAX_CONCURRENT_RESEARCH,
        allow_research_fallback: bool = True,
        research_cache: dict[int, str] | None = None,
        stacking_fallback_on_failure: bool = True,
        stacking_randomize_order: bool = True,
        stacking_spread_thresholds: dict[str, float] | None = None,
        min_forecasters_to_publish: int | None = None,
        research_sink: Any | None = None,
        metaculus_client: MetaculusClient | None = None,
    ) -> None:
        setup = prepare_llm_config(
            llms=llms,
            aggregation_strategy=aggregation_strategy,
            predictions_per_report=predictions_per_research_report,
        )

        self._forecaster_llms: list[GeneralLlm] = setup.forecaster_llms
        self._analyzer_llm: GeneralLlm | None = setup.analyzer_llm
        normalized_llms: dict[str, str | GeneralLlm] = setup.normalized_llms
        predictions_per_research_report = setup.predictions_per_report

        self._custom_research_provider: ResearchCallable | None = research_provider
        self.research_provider: ResearchCallable | None = research_provider  # For framework config access
        if max_questions_per_run is not None and max_questions_per_run <= 0:
            raise ValueError("max_questions_per_run must be a positive integer if provided")
        self.max_questions_per_run: int | None = max_questions_per_run
        self.is_benchmarking: bool = is_benchmarking
        self.allow_research_fallback: bool = allow_research_fallback
        self.research_cache: dict[int, str] | None = research_cache
        # Tests and harnesses override the prod constant; a 2-model ensemble would otherwise always fail the guard.
        self.min_forecasters_to_publish: int = (
            min_forecasters_to_publish if min_forecasters_to_publish is not None else MIN_FORECASTERS_TO_PUBLISH
        )
        if self.min_forecasters_to_publish <= 0:
            raise ValueError(
                f"min_forecasters_to_publish must be a positive integer, got {self.min_forecasters_to_publish}"
            )
        if self.min_forecasters_to_publish > len(self._forecaster_llms):
            logger.warning(
                "min_forecasters_to_publish=%d exceeds configured forecaster count=%d; "
                "every question will fail the guard and skip publication",
                self.min_forecasters_to_publish,
                len(self._forecaster_llms),
            )
        # Conditional stacking thresholds (overridable per question type)
        _valid_threshold_keys = {"binary", "mc", "numeric"}
        if stacking_spread_thresholds is not None:
            unknown_keys = set(stacking_spread_thresholds) - _valid_threshold_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown stacking_spread_thresholds keys: {unknown_keys}. Valid keys: {_valid_threshold_keys}"
                )
        stacking_spread_thresholds_by_type: dict[str, float] = {
            "binary": CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
            "mc": CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
            "numeric": CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
        } | (stacking_spread_thresholds or {})

        # Aggregation pipeline owns stacking state, counters, and dispatch
        self._pipeline = AggregationPipeline(
            strategy=aggregation_strategy,
            stacker_llm=setup.stacker_llm,
            parser_llm=GeneralLlm(model="placeholder"),  # replaced after super().__init__
            stacking_fallback_on_failure=stacking_fallback_on_failure,
            stacking_randomize_order=stacking_randomize_order,
            stacking_spread_thresholds=stacking_spread_thresholds_by_type,
        )

        self._init_alerting_counters()

        if max_concurrent_research <= 0:
            raise ValueError("max_concurrent_research must be a positive integer")
        # Persist for framework config introspection and logging
        self.max_concurrent_research: int = max_concurrent_research
        # Instance-level semaphore to avoid cross-instance throttling
        self._concurrency_limiter: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_research)

        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=normalized_llms,  # type: ignore[arg-type]  # dict value type lacks None but parent expects Optional
            # Off so our guard is the sole arbiter of a degraded publish; see docs/architecture.md "4. Min-forecasters guard".
            required_successful_predictions=0.0,
            # None keeps the framework's default Metaculus client; mantic mode injects a ManticClient.
            metaculus_client=metaculus_client,
        )

        # Declared so the harnesses' display-name tag is statically known; see docs/architecture.md "Harness seams".
        self.name: str = getattr(self, "name", type(self).__name__)

        # Now that super().__init__ has run, resolve the parser LLM.
        self._pipeline.parser_llm = self.get_llm("parser", "llm")

        self._research = ResearchOrchestrator(
            default_llm=self.get_llm("default", "llm"),
            summarizer_llm=self.get_llm("summarizer", "llm"),
            custom_provider=research_provider,
            research_cache=research_cache,
            is_benchmarking=is_benchmarking,
            allow_research_fallback=allow_research_fallback,
            max_concurrent_research=max_concurrent_research,
            research_sink=research_sink,
        )

        self._log_ensemble_configuration()

    def _init_alerting_counters(self) -> None:
        """Zero the run-level counters cli.py reads to decide the process exit status.

        One bot instance == one run, so these are per-run totals rather than per-question.
        """
        self._forecasters_dropped_count: int = 0
        # _record_forecaster_drop is the single write path that keeps this list and the scalar above in lockstep.
        self._forecaster_drops: list[ForecasterDrop] = []
        self._questions_failed_to_publish: int = 0
        # A fast-path publish is a degraded publish and reddens CI; see docs/architecture.md "Run-level counters".
        self._time_budget_fast_path_count: int = 0
        # qid -> FORECASTERS_USED contributor count; a stacked publish collapses predictions to one, so count it here.
        self._contributing_forecasters: defaultdict[int, int] = defaultdict(int)

    @property
    def aggregation_strategy(self) -> AggregationStrategy:
        return self._pipeline.strategy

    @aggregation_strategy.setter
    def aggregation_strategy(self, value: AggregationStrategy) -> None:
        self._pipeline.strategy = value

    @property
    def stacking_fallback_on_failure(self) -> bool:
        return self._pipeline.stacking_fallback_on_failure

    @stacking_fallback_on_failure.setter
    def stacking_fallback_on_failure(self, value: bool) -> None:
        self._pipeline.stacking_fallback_on_failure = value

    @property
    def stacking_randomize_order(self) -> bool:
        return self._pipeline.stacking_randomize_order

    @stacking_randomize_order.setter
    def stacking_randomize_order(self, value: bool) -> None:
        self._pipeline.stacking_randomize_order = value

    def _log_ensemble_configuration(self) -> None:
        """Log the ensemble + aggregation configuration once on init."""
        num_models = len(self._forecaster_llms) if self._forecaster_llms else 1
        logger.info(
            "Ensemble configured: %s model(s) | Aggregation: %s",
            num_models,
            self.aggregation_strategy.value,
        )
        if self.aggregation_strategy not in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
            return

        stacker_name = self._pipeline.stacker_llm.model if self._pipeline.stacker_llm else "<missing>"
        base_models = [m.model for m in self._forecaster_llms]
        shown_models = base_models[:6]  # HARNESS-SCAN-EXEMPT-subsampling: log-line display truncation, no computation
        short_list = base_models if len(base_models) <= 6 else [*shown_models, "..."]

        if self.aggregation_strategy == AggregationStrategy.STACKING:
            logger.info(
                "STACKING config | stacker=%s | base_forecasters(%d)=%s | final_outputs_per_question=1",
                stacker_name,
                len(base_models),
                short_list,
            )
            return

        analyzer_name = self._analyzer_llm.model if self._analyzer_llm else "<missing>"
        logger.info(
            "CONDITIONAL_STACKING config | stacker=%s | analyzer=%s | base_forecasters(%d)=%s | thresholds=%s",
            stacker_name,
            analyzer_name,
            len(base_models),
            short_list,
            self._pipeline.stacking_spread_thresholds,
        )

    async def forecast_questions(  # pyright: ignore[reportIncompatibleMethodOverride]  # matches base's broadest @overload; base declares Literal overloads we deliberately don't replicate
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        """Filter, sort and cap the fetched questions, then hand the survivors to the framework fan-out.

        The single chokepoint every entry path funnels through (forecast_on_tournament and
        forecast_question both call it): the unsupported-type guard drops the ConditionalQuestion
        the 0.2.92 tournament fetch can return, the skip guard fails shut on an unreadable
        ``my_forecasts``, and the tightest-close-first sort decides who wins the shared research
        semaphore and the per-run cap. DiscreteQuestion is a NumericQuestion and stays; DateQuestion
        runs through the numeric pipeline on its epoch-seconds axis.
        """
        # A loud WARNING here beats a per-question exception in the fan-out (_make_prediction has no runner).
        supported_types = (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, DateQuestion)
        supported_questions = [q for q in questions if isinstance(q, supported_types)]
        if len(supported_questions) != len(questions):
            dropped_type_names = sorted({type(q).__name__ for q in questions if not isinstance(q, supported_types)})
            logger.warning(
                "Skipping %d unsupported question(s) this bot cannot forecast (types: %s)",
                len(questions) - len(supported_questions),
                dropped_type_names,
            )
            questions = supported_questions

        # Apply skip filter first (mirrors base class behavior) so we cap unforecasted items
        if self.skip_previously_forecasted_questions:
            questions = _drop_questions_with_unreadable_forecast_history(questions)
            unforecasted_questions = [q for q in questions if not q.already_forecasted]
            if len(questions) != len(unforecasted_questions):
                logger.info(f"Skipping {len(questions) - len(unforecasted_questions)} previously forecasted questions")
            questions = unforecasted_questions

        # Stable, so questions sharing a close time keep fetch order; a missing close_time sorts LAST (no urgency).
        questions = sorted(
            questions,
            key=lambda q: (
                q.close_time is None,
                _as_utc(q.close_time) if q.close_time is not None else _CLOSE_TIME_MAX,
            ),
        )

        # A registered WARNING marker names the forfeited posts; an unharvestable forfeit is gone at the 90-day log expiry.
        if self.max_questions_per_run is not None and len(questions) > self.max_questions_per_run:
            dropped = list(questions)[self.max_questions_per_run :]
            logger.warning(
                "QUESTION_CAP_FORFEIT: platform=%s cap=%d total=%d dropped=%d posts=%s",
                question_platform(dropped[0]),
                self.max_questions_per_run,
                len(questions),
                len(dropped),
                ",".join(str(q.id_of_post) for q in dropped),
            )
            questions = list(questions)[: self.max_questions_per_run]

        if questions:
            bot_name = getattr(self, "name", "Bot")
            logger.info(f"📊 {bot_name}: Processing {len(questions)} questions...")

        reset_pchip_stats()
        RUN_STATUS.reset(stage="runtime")
        self._research.reset_run_degradation_counters()
        # The publish wrapper and the close gate have no handle back to the bot, so their counters are module-scoped.
        reset_publish_attempt_failures()
        reset_publish_skipped_closed()

        results = await super().forecast_questions(questions, return_exceptions)

        log_pchip_summary()

        degradation = self._degradation_snapshot()
        RUN_STATUS.record_degradation_snapshot(degradation)
        if self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING:
            logger.info(format_conditional_stacking_summary(degradation))

        # Any non-zero counter reddens CI via cli.py, after every publishable question has already published.
        logger.info(format_degradation_summary(degradation))
        self._emit_forecaster_drop_telemetry()
        # Emitted even at zero, so "no provider degraded" is a recorded fact rather than an absent line.
        self._research.log_provider_degradation_summary()

        return results

    @property
    def _research_provider_failure_count(self) -> int:
        return self._research.provider_failure_count

    @_research_provider_failure_count.setter
    def _research_provider_failure_count(self, value: int) -> None:
        self._research.provider_failure_count = value

    @property
    def _summarizer_failure_count(self) -> int:
        return self._research.summarizer_failure_count

    @_summarizer_failure_count.setter
    def _summarizer_failure_count(self, value: int) -> None:
        self._research.summarizer_failure_count = value

    @property
    def _prediction_market_degraded_count(self) -> int:
        return self._research.prediction_market_degraded_count

    @property
    def _prediction_market_source_loss_count(self) -> int:
        return self._research.prediction_market_source_loss_count

    @property
    def _provider_degradation_count(self) -> int:
        return self._research.provider_degradation_count

    @property
    def _publish_attempt_failures(self) -> int:
        return publish_attempt_failures()

    @property
    def _publish_skipped_closed_count(self) -> int:
        return publish_skipped_closed_count()

    @property
    def _gap_fill_v2_error_count(self) -> int:
        return self._research.gap_fill_v2_error_count

    @_gap_fill_v2_error_count.setter
    def _gap_fill_v2_error_count(self, value: int) -> None:
        self._research.gap_fill_v2_error_count = value

    @property
    def _gap_fill_v1_error_count(self) -> int:
        return self._research.gap_fill_v1_error_count

    @_gap_fill_v1_error_count.setter
    def _gap_fill_v1_error_count(self, value: int) -> None:
        self._research.gap_fill_v1_error_count = value

    @property
    def alertable_count(self) -> int:
        """Sum of counters whose non-zero value should page us (see degradation_counters)."""
        return alertable_total(self._degradation_snapshot())

    def _degradation_snapshot(self) -> DegradationSnapshot:
        """Read the current counters into one immutable, point-in-time value.

        This is deliberately uncached: ``cli.py`` reads ``alertable_count`` after
        forecasting, while the end-of-run log reads happen earlier in the lifecycle.
        Each read must see any counters changed since the previous one.
        """
        aggregation_counters = self._pipeline.counters
        return DegradationSnapshot(
            forecasters_dropped=self._forecasters_dropped_count,
            questions_failed_to_publish=self._questions_failed_to_publish,
            stacker_primary_failed=aggregation_counters.stacker_primary_failed_count,
            stacker_fallback_used=aggregation_counters.stacker_fallback_used_count,
            stacker_fallback_failed=aggregation_counters.stacker_fallback_failed_count,
            research_provider_failures=self._research_provider_failure_count,
            summarizer_failures=self._summarizer_failure_count,
            gap_fill_v1_errors=self._gap_fill_v1_error_count,
            gap_fill_v2_errors=self._gap_fill_v2_error_count,
            prediction_market_degraded=self._prediction_market_degraded_count,
            prediction_market_source_losses=self._prediction_market_source_loss_count,
            provider_degradation=self._provider_degradation_count,
            publish_attempt_failures=self._publish_attempt_failures,
            publish_skipped_closed=self._publish_skipped_closed_count,
            time_budget_fast_path=self._time_budget_fast_path_count,
            research_budget_cuts=self._research.research_budget_cut_count,
            conditional_stacking_triggered=aggregation_counters.conditional_stacking_triggered_count,
            conditional_stacking_skipped=aggregation_counters.conditional_stacking_skipped_count,
            conditional_stacking_skipped_single_forecaster=(
                aggregation_counters.conditional_stacking_skipped_single_forecaster_count
            ),
            conditional_stacking_crux_failures=aggregation_counters.conditional_stacking_crux_failures,
            conditional_stacking_search_failures=aggregation_counters.conditional_stacking_search_failures,
        )

    def _record_forecaster_drop(self, *, model: str, qid: int | None, cause: str) -> None:
        """Record ONE dropped ensemble member with attribution, bumping the scalar.

        The single write path for both the attributed drops list (which model,
        which question, why) and the legacy ``_forecasters_dropped_count`` scalar,
        so the two can never drift. The scalar stays a plain settable int for
        continuity — the telemetry archive and ``alertable_count`` both read it.
        """
        self._forecaster_drops.append(ForecasterDrop(model=model, qid=qid, cause=cause))
        self._forecasters_dropped_count += 1
        RUN_STATUS.record_model_drop(cause)

    def _emit_forecaster_drop_telemetry(self) -> None:
        """Emit this run's per-model drop attribution (see drop_telemetry)."""
        emit_drop_telemetry(self._forecaster_drops)

    async def run_research(self, question: MetaculusQuestion, time_budget: QuestionTimeBudget | None = None) -> str:
        return await self._research.run_research(question, time_budget=time_budget)

    def _select_research_providers(self) -> list[tuple[ResearchCallable, str]]:
        return self._research._select_research_providers()

    def _build_time_budget(self, question: MetaculusQuestion) -> QuestionTimeBudget:
        """Grant this question its wall-clock budget (see time_budget.py).

        Close-derived only when we intend to publish: backtests and ablations
        forecast RESOLVED questions whose close time is in the past, and deriving a
        budget from that would hand every one a negative budget and skip the run.
        ``PER_QUESTION_WALL_CLOCK_DEADLINE`` is read here, from this module's
        globals, so it stays the single knob tests monkeypatch.
        """
        return build_question_time_budget(
            question,
            close_aware=self.publish_reports_to_metaculus,
            static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE,
        )

    async def _gather_predictions_with_wall_clock(
        self,
        coros: list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
        qid_for_log: int,
        time_budget: QuestionTimeBudget,
    ) -> tuple[list[ReasonedPrediction[PredictionTypes]], list[str], ExceptionGroup | None]:
        """Run forecaster coroutines concurrently with a wall-clock cap.

        Differs from the parent ``_gather_results_and_exceptions`` in two ways:
        - Pending tasks at deadline are cancelled (parent's ``asyncio.gather``
          can't cancel mid-flight).
        - Drops counter is bumped on cancellation so end-of-run alerting
          surfaces the abort.

        Mirrors ``_gather_results_and_exceptions`` return shape so callers
        treat it identically. Tests can patch this method directly to inject
        a synthetic prediction list without spinning up real tasks.
        """
        # coros are built from self._forecaster_llms in order, so idx names the model; "unknown" only if they desync.
        tasks: list[asyncio.Task[Any]] = []
        task_model: dict[asyncio.Task[Any], str] = {}
        for idx, coro in enumerate(coros):
            task = asyncio.create_task(coro, name=f"forecaster:{idx}:q{qid_for_log}")
            tasks.append(task)
            task_model[task] = self._forecaster_llms[idx].model if idx < len(self._forecaster_llms) else "unknown"
        n_total = len(tasks)
        RUN_STATUS.record_model_attempt(n_total)
        remaining = time_budget.remaining_s()
        wait_timeout = max(0.0, remaining)
        done_set, pending_set = await asyncio.wait(tasks, timeout=wait_timeout, return_when=asyncio.ALL_COMPLETED)
        if pending_set:
            for pending in pending_set:
                pending.cancel()
            # Give cancelled tasks a chance to clean up so we don't leak warnings.
            await asyncio.wait(pending_set, timeout=2.0)
            for pending in pending_set:
                self._record_forecaster_drop(
                    model=task_model.get(pending, "unknown"),
                    qid=qid_for_log,
                    cause=DROP_CAUSE_TIMEOUT_WALL_CLOCK,
                )
            logger.warning(
                "WALLCLOCK_ABORT: qid=%s elapsed=%.1fs forecasters_completed=%d/%d cancelled=%d remaining_budget=%.1fs",
                qid_for_log,
                time_budget.elapsed_s(),
                len(done_set),
                n_total,
                len(pending_set),
                remaining,
            )

        # Sort by task name (stable across runs) since asyncio.wait returns sets.
        done_sorted = sorted(done_set, key=lambda t: t.get_name())
        valid_predictions: list[ReasonedPrediction[PredictionTypes]] = []
        errors: list[str] = []
        exceptions: list[BaseException] = []
        for task in done_sorted:
            exc = task.exception()
            if exc is None:
                valid_predictions.append(cast(ReasonedPrediction[PredictionTypes], task.result()))
                RUN_STATUS.record_model_success()
            else:
                errors.append(f"{type(exc).__name__}: {exc}")
                exceptions.append(exc)
                # A raised forecaster is a drop; soft-deadline timeouts were already counted at their raise site.
                if not isinstance(exc, asyncio.TimeoutError):
                    self._record_forecaster_drop(
                        model=task_model.get(task, "unknown"),
                        qid=qid_for_log,
                        cause=classify_raised_drop_cause(exc),
                    )
        exception_group: ExceptionGroup | None = (
            ExceptionGroup(f"Errors: {errors}", cast(list[Exception], exceptions)) if exceptions else None
        )
        return valid_predictions, errors, exception_group

    async def _run_individual_question(self, question: MetaculusQuestion) -> ForecastReport:
        """Run the base per-question pipeline, then log the close-margin marker on submit.

        The base method publishes the report to Metaculus at its tail (when
        ``publish_reports_to_metaculus`` is set), so the moment it returns is the
        submission time. We gate the marker on that flag: backtests/benchmarks don't
        submit and run resolved (past-close) questions, so a margin there would be a
        meaningless negative — the marker is a *submission*-latency watch signal.
        """
        report = await super()._run_individual_question(question)
        if self.publish_reports_to_metaculus:
            marker = format_close_margin_marker(question, datetime.now(UTC))
            if marker is not None:
                logger.info(marker)
        return report

    async def _research_and_make_predictions(
        self,
        question: MetaculusQuestion,
    ) -> ResearchWithPredictions[PredictionTypes]:
        """Research once and fan out over the forecaster roster; the framework's own path when no roster is set."""
        if not self._forecaster_llms:
            return await super()._research_and_make_predictions(question)

        assert question.id_of_question is not None, "id_of_question must not be None for stacking state-dict keying"

        # Granted before any spend so research alone cannot overshoot the shared per-question budget.
        time_budget = self._build_time_budget(question)
        logger.info(format_time_budget_marker(question, time_budget))

        # Unpublishable before any spend (the q45085 shape); see docs/architecture.md "0. Close-derived time budget".
        if time_budget.is_exhausted:
            # The close gate's counter, not the min-forecasters floor's: latency losses have one home however early.
            record_publish_skipped_closed()
            msg = (
                f"Q {question.id_of_question} has no viable time budget "
                f"(close_time={time_budget.close_time}, budget={time_budget.total_s:.0f}s, "
                f"minimum viable {TIME_BUDGET_MIN_VIABLE_S}s on a close-limited window); "
                "skipping before any research or forecaster spend."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if time_budget.fast_path:
            self._time_budget_fast_path_count += 1
            logger.warning(
                "TIME_BUDGET_FAST_PATH: qid=%s budget=%.0fs close_time=%s; "
                "dropping the slow search providers and gap-fill to protect the prediction POST",
                question.id_of_question,
                time_budget.total_s,
                time_budget.close_time,
            )

        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question, time_budget=time_budget)

        # Withheld from every prompt; the router re-appends it to the comment-bound research only.
        diagnostics_block = self._research.pop_provider_diagnostics(question.id_of_question)

        # The stacker never receives chart_b64, so the time-series chart reaches the base models only.
        chart_b64 = self._pull_research_chart(question.id_of_question)

        # A stub: the framework renders both fields, so the full text here doubled the comment past its cap.
        summary_report = "_Full research in the RESEARCH section below._"

        qid_for_log = question.id_of_question
        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [
                self._forecaster_with_soft_deadline(
                    question, research, llm_instance, qid=qid_for_log, chart_b64=chart_b64
                )
                for llm_instance in self._forecaster_llms
            ],
        )
        (
            valid_predictions,
            errors,
            exception_group,
        ) = await self._gather_predictions_with_wall_clock(tasks, qid_for_log, time_budget)
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")

        # Min-forecasters guard: raising skips this question alone; the counter reddens CI at the end of the run.
        n_valid = len(valid_predictions)
        if n_valid < self.min_forecasters_to_publish:
            self._questions_failed_to_publish += 1
            msg = (
                f"Only {n_valid}/{len(self._forecaster_llms)} forecasters succeeded for Q {qid_for_log} "
                f"(need >= {self.min_forecasters_to_publish}); skipping publication."
            )
            logger.error(msg)
            if exception_group is not None:
                self._reraise_exception_with_prepended_message(exception_group, msg)
            raise RuntimeError(msg)

        # Accumulated, not assigned, so several research reports still match the per-model bullet count.
        self._contributing_forecasters[qid_for_log] += n_valid

        # Read off each prediction's Model: prefix, not the roster, so a degraded run is never relabelled as full.
        survivor_names = [extract_model_display_name_from_reasoning(pred.reasoning) for pred in valid_predictions]
        survivor_models = sorted(filter(None, survivor_names))
        logger.info(
            "FORECASTERS_SURVIVED: question=%s survived=%d/%d models=%s",
            qid_for_log,
            n_valid,
            len(self._forecaster_llms),
            ",".join(survivor_models) if survivor_models else "unknown",
        )

        # Binary members are floats by runner dispatch, so the cast is exact; an isinstance filter could drop one silently.
        if isinstance(question, BinaryQuestion):
            for marker in format_extreme_call_markers(
                qid_for_log,
                [
                    (name, cast(float, pred.prediction_value))
                    for name, pred in zip(survivor_names, valid_predictions, strict=True)
                ],
            ):
                logger.info(marker)

        return await route_after_forecasts(
            self._pipeline,
            analyzer_llm=self._analyzer_llm,
            is_benchmarking=self.is_benchmarking,
            research_reports_per_question=self.research_reports_per_question,
            question=question,
            qid=qid_for_log,
            valid_predictions=valid_predictions,
            errors=errors,
            research=research,
            summary_report=summary_report,
            diagnostics_block=diagnostics_block,
            time_budget=time_budget,
        )

    @classmethod
    def _format_and_expand_research_summary(
        cls,
        report_number: int,
        report_type: type[ForecastReport],
        predicted_research: ResearchWithPredictions,
    ) -> str:
        text = super()._format_and_expand_research_summary(report_number, report_type, predicted_research)
        return format_research_summary_with_models(text, predicted_research.predictions, report_number)

    @classmethod
    def _format_main_research(
        cls,
        report_number: int,
        predicted_research: ResearchWithPredictions,
    ) -> str:
        text = super()._format_main_research(report_number, predicted_research)
        return format_main_research_section(text, report_number)

    def _format_forecaster_rationales(
        self,
        report_number: int,
        researched_predictions: ResearchWithPredictions,
    ) -> str:
        text = super()._format_forecaster_rationales(report_number, researched_predictions).lstrip()
        return format_forecaster_rationales_section(text, report_number)

    # The PLR0917 noqa is permanent: forecasting-tools calls this override POSITIONALLY, so nothing can go keyword-only.
    def _create_unified_explanation(  # noqa: PLR0917  # signature is fixed by the ft base class, see comment above
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: PredictionTypes,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        base_text = super()._create_unified_explanation(
            question,
            research_prediction_collections,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        qid = question.id_of_question
        stacker_outcome = self._pipeline.outcomes.pop(qid, None) if qid is not None else None
        stacker_skip_reason = self._pipeline.skip_reasons.pop(qid, None) if qid is not None else None
        # Fan-out count, not the collections' (a stacked publish holds one); docs/architecture.md "Ensemble-size disclosure".
        recorded_used = self._contributing_forecasters.pop(qid, None) if qid is not None else None
        n_used = (
            recorded_used
            if recorded_used is not None
            else sum(len(collection.predictions) for collection in research_prediction_collections)
        )
        n_configured = len(self._forecaster_llms) or self.predictions_per_research_report
        return build_unified_explanation(
            base_text,
            question,
            self.aggregation_strategy,
            stacker_outcome,
            skip_reason=stacker_skip_reason,
            n_used=n_used,
            n_configured=n_configured,
        )

    async def _forecaster_with_soft_deadline(
        self,
        question: MetaculusQuestion,
        research: str,
        llm: GeneralLlm,
        *,
        qid: int | None,
        chart_b64: str | None = None,
    ) -> ReasonedPrediction[PredictionTypes]:
        """Run a single forecaster with FORECASTER_SOFT_DEADLINE.

        Why the deadline: a single stuck forecaster used to be able to hold a
        question for REASONING_MODEL_CONFIG's litellm timeout times its
        allowed_tries. This caps each forecaster at FORECASTER_SOFT_DEADLINE.

        On timeout: bumps _forecasters_dropped_count, logs a loud WARNING
        identifying the model + question, and re-raises TimeoutError so the
        caller's _gather_results_and_exceptions treats it like any other failed
        forecaster (dropped from the ensemble; the other models in the ensemble
        carry the question).

        ``chart_b64`` is the optional time-series-anchor chart image forwarded to
        the base runners; None (the default) when the chart flag is off.
        """
        start = time.monotonic()
        try:
            return await asyncio.wait_for(
                self._make_prediction(question, research, llm, chart_b64),
                timeout=FORECASTER_SOFT_DEADLINE,
            )
        except TimeoutError:
            elapsed = time.monotonic() - start
            self._record_forecaster_drop(model=llm.model, qid=qid, cause=DROP_CAUSE_TIMEOUT_SOFT_DEADLINE)
            logger.warning(
                "SOFT_DEADLINE: forecaster model=%s qid=%s exceeded %ds (elapsed=%.0fs); dropping this forecaster",
                llm.model,
                qid,
                FORECASTER_SOFT_DEADLINE,
                elapsed,
            )
            raise

    async def _make_prediction(
        self,
        question: MetaculusQuestion,
        research: str,
        llm_to_use: GeneralLlm | None = None,
        chart_b64: str | None = None,
    ) -> ReasonedPrediction[PredictionTypes]:
        notepad = await self._get_notepad(question)
        notepad.total_predictions_attempted += 1

        actual_llm = llm_to_use if llm_to_use else self.get_llm("default", "llm")

        forecast_function: Callable[..., Coroutine[Any, Any, ReasonedPrediction[Any]]]
        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_binary(q, r, llm, chart_b64)  # noqa: E731
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_multiple_choice(q, r, llm, chart_b64)  # noqa: E731
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_numeric(q, r, llm, chart_b64)  # noqa: E731
        elif isinstance(question, DateQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_date(q, r, llm, chart_b64)  # noqa: E731
        elif isinstance(question, ConditionalQuestion):
            # forecast_questions filters these out; this backstops a caller that reaches _make_prediction directly.
            raise NotImplementedError(f"{type(question).__name__} is not supported by this bot")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        prediction = await forecast_function(question, research, actual_llm)
        # Load-bearing: performance_analysis.parsing pulls per-model attribution from this "Model:" prefix.
        prediction.reasoning = f"Model: {actual_llm.model}\n\n{prediction.reasoning}"

        # No-ops when PROBABILISTIC_TOOLS_ENABLED is off (prod) or no block was emitted, so callers do not gate.
        computed_md = run_tools_for_forecaster(
            question=question,
            rationale=prediction.reasoning,
            forecaster_id=actual_llm.model,
        )
        if computed_md:
            prediction.reasoning = f"{prediction.reasoning}\n\n## Computed quantities\n{computed_md}"

        return prediction  # type: ignore[return-value]  # each branch narrows T; the framework carries the same ignore

    # The PLR0917 noqa is permanent: forecasting-tools calls the first two params POSITIONALLY; the rest are ours.
    async def _aggregate_predictions(  # noqa: PLR0917  # signature is fixed by the ft base class, see comment above
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
        research: str | None = None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None = None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        if self.aggregation_strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
            if reasoned_predictions is None and research is None:
                aggregated = self._pipeline.base_combine(predictions, question)
            else:
                aggregated = await self._pipeline.stack_predictions(
                    predictions,
                    question,
                    research=research,
                    reasoned_predictions=reasoned_predictions,
                    aggregated_tool_output=aggregated_tool_output,
                )
        else:
            aggregated = self._pipeline.simple_combine(predictions, question)
        # The one seam every aggregation path returns through, so the tail floor and marker see the published CDF.
        if isinstance(aggregated, NumericDistribution):
            floored = floor_published_tails(aggregated, question)
            method = NUMERIC_COMBINE_METHOD_UNRECORDED
            if question.id_of_question is not None:
                method = self._pipeline.numeric_combine_methods.pop(question.id_of_question, method)
            logger.info(
                format_numeric_aggregate_marker(
                    question_id=question.id_of_question,
                    qtype=numeric_qtype(numeric_view(question)),
                    cdf_size=floored.cdf_size,
                    out_of_range=floored.published,
                    out_of_range_raw=floored.raw,
                    tail_floor=floored.floor,
                    method=method,
                )
            )
            return floored.distribution
        return aggregated

    def _pull_research_chart(self, qid: int | None) -> str | None:
        """Pop the time-series-anchor chart image for this qid from the provider's
        per-session cache and return the base64 PNG (or None).

        Popping keeps the provider cache from growing across a batch; the returned
        value is handed straight down the fan-out. The chart-flag gate short-circuits
        the read when the feature is off (the default in prod), so a stale entry from
        an earlier flag-on run is never attached. matplotlib is NOT on the import path
        the module-level ``_session_charts`` import creates: it lives behind
        ``ts_chart``, which ``timeseries_anchor`` imports inside its own render guard,
        so a prod ``uv sync --no-dev`` install imports this fine.
        """
        if qid is None or not env_flag_enabled(TS_ANCHOR_CHART_ENABLED_ENV):
            return None
        return _session_charts.pop(qid, None)

    async def _run_forecast_on_binary(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra params: ensemble fan-out passes a specific LLM + optional chart per call
        self, question: BinaryQuestion, research: str, llm_to_use: GeneralLlm, chart_b64: str | None = None
    ) -> ReasonedPrediction[float]:
        return await run_binary_forecast(
            question, research, llm_to_use, self.get_llm("parser", "llm"), chart_b64=chart_b64
        )

    async def _run_forecast_on_multiple_choice(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra params: ensemble fan-out passes a specific LLM + optional chart per call
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm, chart_b64: str | None = None
    ) -> ReasonedPrediction[PredictedOptionList]:
        return await run_mc_forecast(question, research, llm_to_use, self.get_llm("parser", "llm"), chart_b64=chart_b64)

    async def _run_forecast_on_numeric(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra params: ensemble fan-out passes a specific LLM + optional chart per call
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm, chart_b64: str | None = None
    ) -> ReasonedPrediction[NumericDistribution]:
        prediction, discrete_vote = await run_numeric_forecast(
            question, research, llm_to_use, self.get_llm("parser", "llm"), chart_b64=chart_b64
        )
        qid = question.id_of_question
        if qid is not None and discrete_vote is not None:
            self._pipeline.discrete_integer_votes[qid].append(discrete_vote)
        return prediction

    async def _run_forecast_on_date(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra params: ensemble fan-out passes a specific LLM + optional chart per call
        self, question: DateQuestion, research: str, llm_to_use: GeneralLlm, chart_b64: str | None = None
    ) -> ReasonedPrediction[NumericDistribution]:
        """The date runner on the epoch-seconds axis; no discrete-integer vote, since snapping there is meaningless."""
        return await run_date_forecast(
            question, research, llm_to_use, self.get_llm("parser", "llm"), chart_b64=chart_b64
        )
