"""The Mantic end-to-end harness: one full ``--mode mantic`` run with the HTTP transport and the LLM faked.

``run_mantic_mode`` drives the real chain (``TemplateForecaster.forecast_on_tournament`` -> the
framework's ``forecast_questions`` -> ``_run_individual_question`` ->
``publish_report_to_metaculus(metaculus_client=...)``) with a ``ManticClient`` injected exactly as
``cli.main`` injects it, both class-level hardening patch sets installed first the way
``cli._configure_process`` installs them, and yields a ``ManticRun`` holding the reports, every
prepared request, every LLM call and the ``metaculus_bot`` log records.

Only two things are faked. The HTTP transport (``HTTPAdapter.send``) answers the Mantic API shape, so
the real ``requests.Session`` still prepares every request and the headers, query string and body
that would go on the wire are observable; and ``GeneralLlm.invoke`` returns canned rationales
carrying the fenced ```json STRUCTURED FORECAST block the extraction ladder reads, routed by question
title, so an LLM call that is not a base-forecaster prompt (a parser salvage, a stacker) raises
instead of being quietly answered. Research is stubbed at ``run_research``.

The posts are the recorded preseason payload plus post 643 (``tests/mantic_fakes.py``), re-dated into
the future because the intake time budget and the publish gate both read the real clock. The canned
declarations are the constants below: percentile blocks for the binary, multiple-choice and 450-bin
questions, per-bin ``pmf`` blocks for the 12-bin date question (three members agreeing on one day, 0
on the weekend days) and for post 643 (three members each certain of a different cell).
``tests/test_mantic_e2e.py`` holds the fixtures and the assertions.

Not named ``test_*`` on purpose: pytest imports it without collecting it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.helpers import metaculus_client as ft_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from requests.adapters import HTTPAdapter
from scipy.stats import norm

from metaculus_bot import fetch_hardening, publish_hardening
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    MANTIC_API_BASE_URL,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_ID,
    PMF_ABOVE_RANGE_KEY,
)
from metaculus_bot.mantic import ManticClient
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.time_budget import QuestionTimeBudget
from tests.http_fakes import json_response
from tests.mantic_fakes import (
    BINARY_POST_ID,
    COARSE_DISCRETE_POST_ID,
    DATE_POST_ID,
    DISCRETE_POST_ID,
    MULTIPLE_CHOICE_POST_ID,
    load_coarse_discrete_post,
    load_preseason_posts,
)
from tests.pipeline_test_helpers import make_e2e_bot

FAKE_TOKEN = "m" * 40
EXPECTED_AUTH = f"Token {FAKE_TOKEN}"

# The second run: the bot's forecast already stands on the binary question, leaving four fresh ones.
PRIOR_FORECAST_POST_ID = BINARY_POST_ID
STILL_FRESH_POST_IDS = frozenset({MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID, COARSE_DISCRETE_POST_ID})
ALL_POST_IDS = (BINARY_POST_ID, MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID, COARSE_DISCRETE_POST_ID)

# Derived from the base URL, so the routing paths and the asserted prefixes cannot disagree.
POSTS_URL = f"{MANTIC_API_BASE_URL}/posts/"
_FORECAST_URL = f"{MANTIC_API_BASE_URL}/questions/forecast/"
_COMMENT_URL = f"{MANTIC_API_BASE_URL}/comments/create/"
_POSTS_PATH = urlparse(POSTS_URL).path
_FORECAST_PATH = urlparse(_FORECAST_URL).path
_COMMENT_PATH = urlparse(_COMMENT_URL).path

# Far enough ahead that the time budget is the static one and the publish gate a no-op, forever.
_CLOSE_OFFSET = timedelta(days=30)
_OPEN_OFFSET = timedelta(days=5)

_RESEARCH_TEXT = (
    "## Research Summary\n\nStubbed research for the Mantic end-to-end test: no provider ran, and "
    "no forecast in this run depends on the contents of this section.\n"
)

# Per-forecaster declarations; every triple's spread sits below its stacking threshold, so the median publishes.
_BINARY_MEMBER_PROBS: tuple[float, ...] = (0.20, 0.22, 0.25)
BINARY_MEDIAN = 0.22
# What the platform reports as standing after the previous run published that median: ``[1 - p, p]``.
BINARY_PRIOR_FORECAST_VALUES = (1.0 - BINARY_MEDIAN, BINARY_MEDIAN)
# In the fixture's option order: a hike above 25bp, a 25bp hike, a hold, a cut.
_MC_MEMBER_PROBS: tuple[tuple[float, ...], ...] = (
    (0.05, 0.10, 0.55, 0.30),
    (0.05, 0.12, 0.52, 0.31),
    (0.05, 0.08, 0.58, 0.29),
)
MC_MODAL_OPTION_INDEX = 2
_BITCOIN_MEMBER_NORMALS: tuple[tuple[float, float], ...] = (
    (78_000.0, 6_000.0),
    (78_800.0, 6_200.0),
    (77_300.0, 5_800.0),
)
# The third run: every percentile below post 650's range (from 54,950), so 97.6% of the aggregate lies past its open lower bound.
BITCOIN_MEMBER_NORMALS_BELOW_THE_RANGE: tuple[tuple[float, float], ...] = (
    (51_000.0, 2_000.0),
    (50_500.0, 2_200.0),
    (51_500.0, 1_800.0),
)
BITCOIN_MEDIAN_RANGE = (77_000.0, 79_000.0)
# The per-bin date declarations: most of the mass on bin 8, the rest over the other trading days, 0 on weekends.
DATE_CERTAIN_DAY = datetime(2026, 9, 16, tzinfo=UTC)
DATE_LABELS: tuple[str, ...] = tuple(f"2026-09-{day:02d}" for day in range(8, 20))
DATE_WEEKEND_LABELS: frozenset[str] = frozenset({"2026-09-12", "2026-09-13", "2026-09-19"})
DATE_WEEKEND_BINS = (4, 5, 11)
DATE_CERTAIN_LABEL = "2026-09-16"
DATE_CERTAIN_BIN = 8
DATE_MEMBER_CERTAINTIES: tuple[float, ...] = (0.90, 0.92, 0.88)
# The per-bin count declarations for post 643: three members each certain of one cell, so the pool is exact.
COUNT_LABELS: tuple[str, ...] = tuple(str(count) for count in range(21))
_COUNT_MEMBER_BELIEFS: tuple[str, ...] = ("1", "3", PMF_ABOVE_RANGE_KEY)
COUNT_BELIEVED_BINS = (1, 3)

FORECASTS_PER_QUESTION = 3
FORECAST_QUESTION_COUNT = 5


# ---------------------------------------------------------------------------
# The fixture, re-dated into the future
# ---------------------------------------------------------------------------


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def future_dated_posts() -> list[dict[str, Any]]:
    """The probe's four posts plus the coarse count post, their open and close timestamps moved around today.

    Both levels are rewritten: the framework reads the close and resolve times off the QUESTION
    json, while the post-level copies are what a reader of this fixture would compare against.
    """
    posts = [*load_preseason_posts(), load_coarse_discrete_post()]
    now = datetime.now(UTC)
    close_iso = iso(now + _CLOSE_OFFSET)
    open_iso = iso(now - _OPEN_OFFSET)
    for post in posts:
        for holder in (post, post["question"]):
            for key in ("scheduled_close_time", "scheduled_resolve_time", "cp_reveal_time"):
                if key in holder:
                    holder[key] = close_iso
            if "open_time" in holder:
                holder["open_time"] = open_iso
    return posts


# ---------------------------------------------------------------------------
# Canned forecaster rationales
# ---------------------------------------------------------------------------


def _percentile_key(percentile: float) -> str:
    return f"{percentile:g}"


def _structured_block(payload: dict[str, Any]) -> str:
    return f"```json\n{json.dumps(payload)}\n```\n"


def _binary_reasoning(posterior_prob: float) -> str:
    return (
        "## Analysis\n\nSeptember is the climatological peak of the Atlantic season, and the "
        "current basin state is the dominant term.\n\n"
        f"{_structured_block({'question_type': 'binary', 'posterior_prob': posterior_prob})}"
    )


def _mc_reasoning(options: Sequence[str], probs: Sequence[float]) -> str:
    option_probs = dict(zip(options, probs, strict=True))
    return (
        "## Analysis\n\nThe futures-implied path and the last statement both point at a hold, with "
        "a cut as the live alternative.\n\n"
        f"{_structured_block({'question_type': 'multiple_choice', 'option_probs': option_probs})}"
    )


def _numeric_reasoning(mean: float, sd: float) -> str:
    percentiles = {_percentile_key(p): round(float(norm.ppf(p, loc=mean, scale=sd)), 2) for p in STANDARD_PERCENTILES}
    payload = {
        "question_type": "numeric",
        "declared_percentiles": percentiles,
        "outcome_type": "continuous",
    }
    return (
        "## Analysis\n\nSpot, realized volatility over the window, and the eleven daily "
        "resolutions this question averages.\n\n"
        f"{_structured_block(payload)}"
    )


def _date_reasoning(certainty: float) -> str:
    """A per-bin forecaster with ``certainty`` on 2026-09-16, the rest over the other trading days, 0 on weekends.

    The block is the ``PmfStructured`` shape: one probability per calendar-day label, both bounds
    closed so no reserved key. The three members differ only in how sure they are.
    """
    trading_days = [label for label in DATE_LABELS if label not in DATE_WEEKEND_LABELS]
    remainder = (1.0 - certainty) / (len(trading_days) - 1)
    bin_probs = dict.fromkeys(DATE_LABELS, 0.0)
    for label in trading_days:
        bin_probs[label] = round(remainder, 6)
    bin_probs[DATE_CERTAIN_LABEL] = certainty
    payload = {"question_type": "pmf", "bin_probs": bin_probs}
    return (
        "## Analysis\n\nRealised volatility clusters around the CPI print and the FOMC decision; the "
        "session after the FOMC statement is the modal largest move, and the three weekend days cannot "
        "resolve.\n\n"
        f"{_structured_block(payload)}"
    )


def _count_reasoning(believed_key: str) -> str:
    """A per-bin forecaster certain of one cell of post 643's grid: 1.0 on ``believed_key``, 0 elsewhere."""
    bin_probs = dict.fromkeys((*COUNT_LABELS, PMF_ABOVE_RANGE_KEY), 0.0)
    bin_probs[believed_key] = 1.0
    payload = {"question_type": "pmf", "bin_probs": bin_probs}
    return (
        "## Analysis\n\nThe release tempo over the window is set by one operational decision, and the "
        "count follows from it.\n\n"
        f"{_structured_block(payload)}"
    )


def _canned_responses(
    posts: list[dict[str, Any]],
    *,
    discrete_normals: tuple[tuple[float, float], ...] = _BITCOIN_MEMBER_NORMALS,
) -> dict[str, list[str]]:
    """One rationale per forecaster per question, keyed by the question title the prompt carries."""
    questions = {post["id"]: post["question"] for post in posts}
    mc_options: list[str] = questions[MULTIPLE_CHOICE_POST_ID]["options"]
    return {
        questions[BINARY_POST_ID]["title"]: [_binary_reasoning(prob) for prob in _BINARY_MEMBER_PROBS],
        questions[MULTIPLE_CHOICE_POST_ID]["title"]: [_mc_reasoning(mc_options, probs) for probs in _MC_MEMBER_PROBS],
        questions[DISCRETE_POST_ID]["title"]: [_numeric_reasoning(mean, sd) for mean, sd in discrete_normals],
        questions[DATE_POST_ID]["title"]: [_date_reasoning(certainty) for certainty in DATE_MEMBER_CERTAINTIES],
        questions[COARSE_DISCRETE_POST_ID]["title"]: [_count_reasoning(key) for key in _COUNT_MEMBER_BELIEFS],
    }


@dataclass
class _LlmCallLog:
    """Which canned rationale each ``GeneralLlm.invoke`` served, and anything unrouted."""

    forecaster_calls: list[str] = field(default_factory=list)
    unexpected_prompts: list[str] = field(default_factory=list)


def _install_llm_stub(mp: pytest.MonkeyPatch, responses_by_title: dict[str, list[str]]) -> _LlmCallLog:
    """Serve the canned rationales off ``GeneralLlm.invoke``, and refuse any other LLM call.

    Routing is by question title, so an invocation that is NOT a base-forecaster prompt (a parser
    fallback because the ladder's block rung failed, or a stacker call the spread should have
    skipped) has nowhere to go and raises instead of being quietly answered.
    """
    log = _LlmCallLog()
    served: dict[str, int] = dict.fromkeys(responses_by_title, 0)

    async def invoke(self: GeneralLlm, prompt: Any, system_prompt: str | None = None) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        for title, responses in responses_by_title.items():
            if title in text:
                index = served[title]
                served[title] = index + 1
                log.forecaster_calls.append(title)
                return responses[index % len(responses)]
        log.unexpected_prompts.append(text[:400])
        raise AssertionError("unrouted LLM call: not a base-forecaster prompt for any preseason question")

    mp.setattr(GeneralLlm, "invoke", invoke)
    return log


# ---------------------------------------------------------------------------
# The fake Mantic transport
# ---------------------------------------------------------------------------

# Mantic's ``forecast_type`` vocabulary, honoured by the fake so asserting the parameter's absence has teeth.
_MANTIC_FILTER_TYPE = {
    "binary": "binary",
    "multiple_choice": "multiple_choice",
    "date": "date",
    "discrete": "quantitative",
    "quantitative": "quantitative",
}


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    url: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any
    send_kwargs: dict[str, Any]


def _visible_posts(posts: list[dict[str, Any]], query: dict[str, list[str]]) -> list[dict[str, Any]]:
    """The page of ``posts`` a real list endpoint would answer: type-filtered, then ``offset``/``limit``-sliced.

    The Mantic client walks offsets until an EMPTY page (its ceiling makes the framework keep
    paging, since Mantic's ``next`` link is unreliable), so the fake must run out of posts the
    way the platform does or the framework reads the same four posts twice and refuses them as
    duplicates.
    """
    requested_types = query.get("forecast_type")
    if requested_types:
        posts = [post for post in posts if _MANTIC_FILTER_TYPE[post["question"]["type"]] in requested_types]
    offset = int(query.get("offset", ["0"])[0])
    limit = int(query.get("limit", [str(len(posts))])[0])
    return posts[offset : offset + limit]


def install_fake_transport(mp: pytest.MonkeyPatch, posts: list[dict[str, Any]]) -> list[RecordedRequest]:
    """Answer the three Mantic endpoints at the transport, recording every prepared request."""
    # Every framework request sleeps 3.5-4.5s first to rate-limit itself against the real API.
    mp.setattr(MetaculusClient, "_sleep_between_requests", lambda self: None)
    recorded: list[RecordedRequest] = []

    def fake_send(self: HTTPAdapter, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        parsed = urlparse(request.url or "")
        query = parse_qs(parsed.query)
        body = json.loads(request.body) if request.body else None
        recorded.append(
            RecordedRequest(
                method=request.method or "",
                url=request.url or "",
                path=parsed.path,
                query=query,
                headers=dict(request.headers),
                body=body,
                send_kwargs=dict(kwargs),
            )
        )
        if request.method == "GET" and parsed.path == _POSTS_PATH:
            results = _visible_posts(posts, query)
            return json_response({"next": None, "previous": None, "results": results}, request=request)
        if request.method == "POST" and parsed.path == _FORECAST_PATH:
            return json_response([], status=201, request=request)
        if request.method == "POST" and parsed.path == _COMMENT_PATH:
            return json_response({}, status=201, request=request)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    mp.setattr(HTTPAdapter, "send", fake_send)
    return recorded


# ---------------------------------------------------------------------------
# Process-global state: hardening and log capture
# ---------------------------------------------------------------------------


def _apply_hardening_with_restore(mp: pytest.MonkeyPatch) -> None:
    """Install both hardening patch sets the way ``cli._configure_process`` does, reversibly.

    Every patched name is snapshotted through monkeypatch first (the idiom in
    tests/test_ft_upgrade_seams.py), because these mutate the framework's classes and its
    ``requests`` module for the whole process: left installed, they would change what the seam
    tests' own negative controls observe. Both sentinels are cleared so this run always exercises
    a fresh install rather than silently inheriting an earlier test's.
    """
    for method_name in (*publish_hardening._PATCHED_METHODS, *fetch_hardening._PATCHED_METHODS):
        mp.setattr(MetaculusClient, method_name, MetaculusClient.__dict__[method_name])
    for report_type in publish_hardening._PATCHED_REPORT_TYPES:
        publish_method = publish_hardening._PUBLISH_METHOD
        mp.setattr(report_type, publish_method, report_type.__dict__[publish_method])
    # Setting each sentinel before deleting it is what makes the deletion reversible.
    for sentinel in (publish_hardening._SENTINEL, fetch_hardening._SENTINEL):
        mp.setattr(MetaculusClient, sentinel, False, raising=False)
        delattr(MetaculusClient, sentinel)
    # Both hardening layers replace the module-level requests.post / requests.get permanently.
    mp.setattr(ft_client.requests, "post", ft_client.requests.post)
    mp.setattr(ft_client.requests, "get", ft_client.requests.get)

    publish_hardening.apply_publish_hardening()
    fetch_hardening.apply_fetch_hardening()


class _RecordingHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.DEBUG)
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


@contextmanager
def _captured_bot_logs() -> Iterator[list[logging.LogRecord]]:
    """Capture ``metaculus_bot`` records for the whole run (caplog is function-scoped)."""
    logger = logging.getLogger("metaculus_bot")
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManticRun:
    reports: Sequence[ForecastReport | BaseException]
    requests: list[RecordedRequest]
    llm_calls: _LlmCallLog
    records: list[logging.LogRecord]
    posts: list[dict[str, Any]]

    @property
    def log_text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)

    def posts_requests(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == _POSTS_PATH]

    def forecast_posts(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == _FORECAST_PATH]

    def comment_posts(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == _COMMENT_PATH]

    def posted_question_ids(self) -> set[int]:
        return {r.body[0]["question"] for r in self.forecast_posts()}

    def commented_post_ids(self) -> set[int]:
        return {r.body["on_post"] for r in self.comment_posts()}

    def mantic_question_marker_lines(self) -> list[str]:
        return self.marker_lines("MANTIC_QUESTION:")

    def marker_lines(self, prefix: str) -> list[str]:
        return [line for line in self.log_text.splitlines() if line.startswith(prefix)]

    def forecast_payload(self, question_id: int) -> dict[str, Any]:
        payloads = [r.body[0] for r in self.forecast_posts() if r.body[0]["question"] == question_id]
        assert len(payloads) == 1, f"expected exactly one forecast POST for question {question_id}, got {payloads}"
        return payloads[0]

    def comment_payload(self, post_id: int) -> dict[str, Any]:
        payloads = [r.body for r in self.comment_posts() if r.body["on_post"] == post_id]
        assert len(payloads) == 1, f"expected exactly one comment POST for post {post_id}, got {len(payloads)}"
        return payloads[0]

    def report_for(self, post_id: int) -> ForecastReport:
        matches = [r for r in self.reports if isinstance(r, ForecastReport) and r.question.id_of_post == post_id]
        assert len(matches) == 1, f"expected one report for post {post_id}, got {matches}"
        return matches[0]


async def _stub_research(question: MetaculusQuestion, time_budget: QuestionTimeBudget | None = None) -> str:
    """Stand in for ``TemplateForecaster.run_research`` with its real signature: no provider runs."""
    return _RESEARCH_TEXT


def run_mantic_mode(
    posts: list[dict[str, Any]],
    *,
    discrete_normals: tuple[tuple[float, float], ...] = _BITCOIN_MEMBER_NORMALS,
) -> Iterator[ManticRun]:
    """One full mantic-mode run over ``posts``; the module-scoped fixtures below each wrap one.

    Synchronous so each fixture runs the pipeline a single time (and installs and removes the
    process-global patches once). The framework's tournament fetch calls ``asyncio.run`` inside the
    running loop, which works because forecasting-tools applies nest_asyncio at import — the same
    nesting production relies on.

    The bot is built in ``cli.main``'s shape for a mantic run, with two deliberate notes:
    ``is_benchmarking=False`` keeps the publish path real, and ``min_forecasters_to_publish`` is the
    full roster (production's floor is 1) so a forecaster lost to a stub defect fails the test.
    """
    with pytest.MonkeyPatch.context() as mp:
        # The environment a Mantic run requires, though the client below is handed its token.
        mp.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
        mp.setenv(MANTIC_TOKEN_ENV, FAKE_TOKEN)
        _apply_hardening_with_restore(mp)
        recorded = install_fake_transport(mp, posts)
        llm_calls = _install_llm_stub(mp, _canned_responses(posts, discrete_normals=discrete_normals))

        bot = make_e2e_bot(
            AggregationStrategy.CONDITIONAL_STACKING,
            n_forecasters=FORECASTS_PER_QUESTION,
            publish_reports_to_metaculus=True,
            is_benchmarking=False,
            skip_previously_forecasted_questions=True,
            min_forecasters_to_publish=FORECASTS_PER_QUESTION,
            metaculus_client=ManticClient(token=FAKE_TOKEN),
        )
        mp.setattr(bot, "run_research", _stub_research)

        with _captured_bot_logs() as records:
            reports = asyncio.run(bot.forecast_on_tournament(MANTIC_TOURNAMENT_ID, return_exceptions=True))

        yield ManticRun(
            reports=list(reports),
            requests=recorded,
            llm_calls=llm_calls,
            records=records,
            posts=posts,
        )
