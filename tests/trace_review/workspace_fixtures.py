"""Fixture workspace builders for the trace-review tests: a round file, telemetry, a research archive and fetchers.

The round records are produced by the real collector (``records_from_posts``) from posts shaped
like the Metaculus posts list and the twelve real comments in
``tests/data/performance_comments_mini.jsonl``, then tagged by ``tag_fresh_records``, so a change
to either shows up here. Telemetry lines copy the archive's real shapes (one line per marker per
question, ``qid`` plus ``qid_kind``). Nothing opens a socket: the fetchers are doubles that count
their calls, and ``api_get`` raises if anything reaches for it.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from metaculus_bot import api_preflight
from metaculus_bot.constants import (
    MANTIC_API_BASE_URL,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_ID,
    METACULUS_TOKEN_ENV,
    PLATFORM_MANTIC,
    TOURNAMENT_ID,
)
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.round_dataset import tag_fresh_records
from metaculus_bot.trace_review.build import BuildConfig, BuildResult

DATA_DIR = Path(__file__).parent.parent / "data"
MINI_COMMENTS = {
    row["post_id"]: row
    for row in (json.loads(line) for line in (DATA_DIR / "performance_comments_mini.jsonl").read_text().splitlines())
}
MANTIC_DISCRETE_POST = json.loads((DATA_DIR / "mantic_series1_discrete_post_643_2026_09_08.json").read_text())[
    "results"
][0]

ROUND_LABEL = "2026-11-04"
NUMERIC_CDF_POINTS = 201

# The id collision the join tests use: post 41871 holds question 41609, and question 41871 sits on post 42108.
COLLIDING_NUMERIC = (41609, 41871)  # (question_id, post_id)
COLLIDING_BINARY = (41871, 42108)
EARLY_BINARY = (41261, 41518)
NUMERIC_BACKFILLED = (41278, 41537)
DISCRETE = (41577, 41841)
NAMED_MC = (41260, 41517)
STACKED_MC = (43053, 43053)
STACKED_NUMERIC = (43059, 43059)
NEW_SINCE_ROUND = (45790, 46001)
DATE_QUESTION = (45791, 46002)
ANNULLED = (45792, 46003)
PRACTICE = (45793, 46004)
# Mantic's ids share the integer space: this Mantic question id equals a Metaculus question id above.
MANTIC_DISCRETE = (41261, 41261)
MANTIC_MULTI_RESOLUTION = (41262, 41262)

RUN_EARLY = "41000000001"  # before the first extreme_call line anywhere
RUN_EXTREME = "41000000002"  # the first extreme_call line
RUN_LATE = "41000000003"  # after it, with no extreme_call line of its own
RUN_MANTIC = "41000000009"
RUN_DATES = {
    RUN_EARLY: "2026-01-11T13:05:00Z",
    RUN_EXTREME: "2026-02-03T22:15:00Z",
    RUN_LATE: "2026-02-16T03:30:00Z",
    RUN_MANTIC: "2026-09-10T12:00:00Z",
}


def forecast_key(platform: str, ids: tuple[int, int]) -> str:
    return f"{platform}:{ids[0]}"


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------


def _numeric_cdf(range_min: float, range_max: float, points: list[tuple[float, float]], n_points: int) -> list[float]:
    """A monotone CDF on the question's grid through the declared (probability, value) points."""
    grid = np.linspace(range_min, range_max, n_points)
    probabilities, values = zip(*points, strict=True)
    cdf = np.interp(grid, values, probabilities, left=0.001, right=0.999)
    return [float(v) for v in np.maximum.accumulate(cdf)]


@dataclass(frozen=True)
class PostSpec:
    ids: tuple[int, int]
    question_type: str
    resolution: str
    forecast_values: list[float]
    scaling: dict = field(default_factory=dict)
    options: list[str] | None = None
    nr_forecasters: int = 40
    title: str | None = None
    open_upper_bound: bool = False
    comment_from: int | None = None


def _binary(ids: tuple[int, int], prob_yes: float, resolution: str, **extra) -> PostSpec:
    return PostSpec(ids, "binary", resolution, [1.0 - prob_yes, prob_yes], **extra)


def _continuous(ids, question_type, resolution, range_min, range_max, points, n_points=NUMERIC_CDF_POINTS) -> PostSpec:
    scaling = {
        "range_min": range_min,
        "range_max": range_max,
        "zero_point": None,
        "inbound_outcome_count": n_points - 1,
    }
    return PostSpec(ids, question_type, resolution, _numeric_cdf(range_min, range_max, points, n_points), scaling)


MC_41260_OPTIONS = ["0", "1", "2", "3+"]
MC_43053_OPTIONS = [
    "Ismaïl Omar Guelleh officially declared winner",
    "Another candidate officially declared winner",
    "Vote held but no final official result before May 1",
    "Election postponed or cancelled",
]

ROUND_POSTS: list[PostSpec] = [
    _binary(EARLY_BINARY, 0.20, "no", nr_forecasters=0),
    _binary(COLLIDING_BINARY, 0.38, "no"),
    PostSpec(NAMED_MC, "multiple_choice", "2", [0.0521, 0.3438, 0.3958, 0.2083], options=MC_41260_OPTIONS),
    PostSpec(STACKED_MC, "multiple_choice", MC_43053_OPTIONS[0], [0.92, 0.02, 0.02, 0.04], options=MC_43053_OPTIONS),
    _continuous(
        NUMERIC_BACKFILLED, "numeric", "81000000000", 65e9, 95e9, [(0.025, 70e9), (0.394, 80e9), (0.975, 90e9)]
    ),
    _continuous(
        COLLIDING_NUMERIC, "numeric", "7100000000", 5.5e9, 8.5e9, [(0.025, 6.1e9), (0.5, 7.2e9), (0.975, 7.97e9)]
    ),
    _continuous(STACKED_NUMERIC, "numeric", "150", 0.0, 250.0, [(0.025, 71.0), (0.5, 146.0), (0.975, 200.0)]),
    _continuous(DISCRETE, "discrete", "63", 49.5, 75.5, [(0.025, 54.5), (0.406, 62.5), (0.975, 70.5)], n_points=27),
]

FRESH_ONLY_POSTS: list[PostSpec] = [
    _binary(NEW_SINCE_ROUND, 0.20, "yes", comment_from=EARLY_BINARY[1]),
    PostSpec(DATE_QUESTION, "date", "2026-10-01", [0.1, 0.5, 0.9]),
    _binary(ANNULLED, 0.6, "annulled"),
    _binary(PRACTICE, 0.6, "yes", title="[PRACTICE] Will the practice question resolve yes?"),
]


def post_json(spec: PostSpec) -> dict:
    """One post in the posts-list shape: the fields the collector reads, with real values."""
    question_id, post_id = spec.ids
    title = spec.title or f"Fixture question {question_id}"
    question = {
        "id": question_id,
        "title": title,
        "type": spec.question_type,
        "resolution": spec.resolution,
        "multi_resolution": False,
        "question_weight": 1.0,
        "options": spec.options,
        "scaling": spec.scaling,
        "open_lower_bound": False,
        "open_upper_bound": spec.open_upper_bound,
        "open_time": "2026-01-05T12:00:00Z",
        "scheduled_resolve_time": "2026-03-01T12:00:00Z",
        "actual_resolve_time": "2026-03-02T12:00:00Z",
        "resolution_set_time": "2026-03-02T12:05:00Z",
        "my_forecasts": {
            "history": [],
            "latest": {"forecast_values": spec.forecast_values},
            "score_data": {
                "spot_peer_score": 3.5,
                "peer_score": 2.0,
                "spot_baseline_score": 10.0,
                "baseline_score": 8.0,
                "coverage": 1.0,
            },
        },
    }
    return {
        "id": post_id,
        "title": title,
        "nr_forecasters": spec.nr_forecasters,
        "projects": {"category": [{"name": "Economy & Business"}]},
        "question": question,
    }


def comment_json(spec: PostSpec, comment_id: int) -> dict | None:
    source_post = spec.comment_from or spec.ids[1]
    mini = MINI_COMMENTS.get(source_post)
    if mini is None:
        return None
    return {
        "id": comment_id,
        "on_post": spec.ids[1],
        "text": mini["comment_text"],
        "created_at": mini["bot_comment_created_at"],
        "is_private": False,
    }


def mantic_post(ids: tuple[int, int], *, multi_resolution: bool) -> dict:
    """The recorded series-1 discrete post, renumbered, resolved above its upper bound, with our forecast."""
    post = copy.deepcopy(MANTIC_DISCRETE_POST)
    question = post["question"]
    question_id, post_id = ids
    post["id"], question["id"], question["post_id"] = post_id, question_id, post_id
    n_points = question["scaling"]["inbound_outcome_count"] + 1
    question.update(
        {
            "status": "resolved",
            "resolution": "above_upper_bound",
            "multi_resolution": multi_resolution,
            "actual_resolve_time": "2026-09-12T07:00:00Z",
            "my_forecasts": {
                "history": [],
                "latest": {"forecast_values": [0.02 + 0.6 * i / (n_points - 1) for i in range(n_points)]},
                "score_data": {"spot_baseline_score": 12.0, "baseline_score": 11.0},
            },
        }
    )
    return post


# ---------------------------------------------------------------------------
# Telemetry and research archive
# ---------------------------------------------------------------------------


def telemetry_line(
    marker: str, run_id: str, ids: tuple[int, int] | None, *, kind: str = "question_id", **payload
) -> dict:
    line = {
        "marker": marker,
        "run_id": run_id,
        "workflow": "unknown",
        "artifact": f"research-{run_id}",
        "run_date": RUN_DATES[run_id],
        "log_file": f"run_{run_id}.log",
        "seq": 0,
        "line_ts": RUN_DATES[run_id].removesuffix("Z"),
        **payload,
    }
    if ids is not None:
        line |= {"question": str(ids[0]), "qid": ids[0] if kind == "question_id" else ids[1], "qid_kind": kind}
    return line


def telemetry_lines() -> list[dict]:
    binary_raw = {"gpt-5.2": ("0.37", "0.37"), "gpt-5.1": ("0.42", "0.42"), "gemini-3-pro-preview": ("0.551", "0.55")}
    return [
        # The late run covers both halves of the collision and records members for the binary one.
        *(
            telemetry_line(
                "member_forecast",
                RUN_LATE,
                COLLIDING_BINARY,
                model=f"openrouter/x/{model}",
                role="member",
                qtype="binary",
                raw=raw,
                published=published,
                oor_low=None,
                oor_high=None,
                elicitation=None,
            )
            for model, (raw, published) in binary_raw.items()
        ),
        telemetry_line(
            "extraction_rung",
            RUN_LATE,
            COLLIDING_BINARY,
            model="openrouter/openai/gpt-5.2",
            qtype="binary",
            rung="block",
            block_present=True,
        ),
        telemetry_line(
            "extraction_rung",
            RUN_LATE,
            COLLIDING_BINARY,
            model="openrouter/openai/gpt-5.1",
            qtype="binary",
            rung="regex_fallback",
            block_present=False,
        ),
        telemetry_line("gap_fill_v2", RUN_LATE, COLLIDING_NUMERIC, kind="post_id", tool_calls=17, steps=6),
        telemetry_line(
            "close_margin",
            RUN_LATE,
            COLLIDING_BINARY,
            close_time="2026-02-16T06:00:00+00:00",
            margin_s=9000,
            margin_frac=0.83,
        ),
        telemetry_line("time_budget", RUN_LATE, COLLIDING_BINARY, budget_s=3400, fast_path=False, close_limited=False),
        telemetry_line(
            "forecasters_survived", RUN_LATE, COLLIDING_BINARY, survived=5, configured=5, models="a,b,c,d,e"
        ),
        telemetry_line(
            "numeric_aggregate", RUN_LATE, COLLIDING_NUMERIC, qtype="numeric", cdf_size=201, method="median"
        ),
        telemetry_line(
            "ghost_forecast_json",
            RUN_LATE,
            COLLIDING_BINARY,
            kind="post_id",
            forecast_json=json.dumps({"qtype": "binary", "prob": 0.30}),
        ),
        telemetry_line("credit_run_summary", RUN_LATE, None, n_questions=2, usd_per_question=0.41),
        telemetry_line("forecaster_drops", RUN_LATE, None, total=0, systematic=None, detail="{}"),
        # The extreme-call run, and an early run dated before any extreme_call line exists.
        telemetry_line(
            "extreme_call",
            RUN_EXTREME,
            DISCRETE,
            model="gemini-3-pro-preview",
            p=0.02,
            side="low",
            lone=False,
            survivors=5,
        ),
        telemetry_line(
            "extraction_rung",
            RUN_EXTREME,
            DISCRETE,
            model="openrouter/openai/gpt-5.2",
            qtype="discrete",
            rung="block",
            block_present=True,
        ),
        telemetry_line(
            "extraction_rung",
            RUN_EARLY,
            EARLY_BINARY,
            model="openrouter/openai/gpt-5.2",
            qtype="binary",
            rung="block",
            block_present=True,
        ),
        # Mantic: its run names the tournament, and its qid equals the early Metaculus binary's question id.
        telemetry_line(
            "mantic_tournaments",
            RUN_MANTIC,
            None,
            ongoing=MANTIC_TOURNAMENT_ID,
            configured=MANTIC_TOURNAMENT_ID,
            new=None,
        ),
        telemetry_line(
            "extraction_rung",
            RUN_MANTIC,
            MANTIC_DISCRETE,
            model="openrouter/openai/gpt-6.1-sol",
            qtype="discrete",
            rung="block",
            block_present=True,
        ),
        telemetry_line(
            "member_forecast",
            RUN_MANTIC,
            MANTIC_DISCRETE,
            model="openrouter/openai/gpt-6.1-sol",
            role="member",
            qtype="discrete",
            raw=json.dumps(MANTIC_PER_BIN_MASSES),
            published=json.dumps(MANTIC_PER_BIN_MASSES),
            oor_low=0.0,
            oor_high=MANTIC_PER_BIN_MASSES[-1],
            elicitation="pmf",
        ),
    ]


# [below, one mass per bin, above] on the 21-bin grid of the recorded discrete post.
MANTIC_PER_BIN_MASSES: list[float] = [0.0, *([0.03] * 20), 0.01, 0.39]


def archive_records() -> dict[int, dict]:
    """``latest/<id>.json`` contents, one per record class plus a Mantic record on a colliding id."""
    numeric_qid, numeric_post = COLLIDING_NUMERIC
    return {
        numeric_qid: {
            "schema_version": 2,
            "qid": numeric_qid,
            "post_id": numeric_post,
            "page_url": f"https://www.metaculus.com/questions/{numeric_post}",
            "research_text": "## Web Research (Native Search)\nThe series rose 4% year on year.\n",
            "providers_attempted": ["native_search", "gemini_search", "prediction_market"],
            "providers_succeeded": ["native_search", "prediction_market"],
            "provider_results": [
                {"name": "native_search", "status": "ok", "chars": 1200, "latency_ms": 9000, "error_type": None},
                {
                    "name": "gemini_search",
                    "status": "timeout",
                    "chars": 0,
                    "latency_ms": 60000,
                    "error_type": "Timeout",
                },
                {"name": "prediction_market", "status": "ok", "chars": 300, "latency_ms": 800, "error_type": None},
            ],
            "gap_fill_used": True,
            "gap_fill_v2": {"findings": [{"claim": "Q4 print due 2026-02-20"}]},
            "provider_diagnostics_block": "native_search ok; gemini_search timeout",
            "run_mode": "tournament",
            "platform": "metaculus",
            "run_id": RUN_LATE,
            "source": "artifact",
        },
        NUMERIC_BACKFILLED[0]: {
            "schema_version": 1,
            "qid": NUMERIC_BACKFILLED[0],
            "on_post": NUMERIC_BACKFILLED[1],
            "page_url": f"https://www.metaculus.com/questions/{NUMERIC_BACKFILLED[0]}/",
            "research_text": "[research recovered from the comment]",
            "providers_used": ["native_search"],
            "run_mode": "tournament",
            "run_id": "comment-900001",
            "is_trimmed": True,
            "source": "comment_backfill",
        },
        EARLY_BINARY[1]: {
            "schema_version": 1,
            "qid": EARLY_BINARY[1],
            "page_url": f"https://www.metaculus.com/questions/{EARLY_BINARY[1]}/",
            "research_text": "[research parsed from a run log]",
            "run_id": "log-1",
            "source": "log_backfill",
        },
        MANTIC_DISCRETE[0]: {
            "schema_version": 2,
            "qid": MANTIC_DISCRETE[0],
            "post_id": MANTIC_DISCRETE[1],
            "page_url": f"https://competitions.mantic.com/questions/{MANTIC_DISCRETE[1]}/",
            "research_text": "## Web Research (Native Search)\nCENTCOM posted 23 releases in the window.\n",
            "providers_attempted": ["native_search"],
            "providers_succeeded": ["native_search"],
            "provider_results": [
                {"name": "native_search", "status": "ok", "chars": 900, "latency_ms": 7000, "error_type": None}
            ],
            "run_mode": "mantic",
            "platform": PLATFORM_MANTIC,
            "run_id": RUN_MANTIC,
            "source": "artifact",
        },
    }


# ---------------------------------------------------------------------------
# Fetcher doubles and the workspace
# ---------------------------------------------------------------------------


def metaculus_fresh_posts() -> list[dict]:
    """The current-season listing now: the round's posts (one re-resolved) plus what resolved since."""
    posts = [post_json(spec) for spec in ROUND_POSTS]
    for post in posts:
        if post["id"] == COLLIDING_BINARY[1]:
            post["question"]["resolution"] = "yes"
    return posts + [post_json(spec) for spec in FRESH_ONLY_POSTS]


def bot_comments() -> list[dict]:
    specs = [*ROUND_POSTS, *FRESH_ONLY_POSTS]
    return [comment for index, spec in enumerate(specs) if (comment := comment_json(spec, 900_000 + index))]


@dataclass
class FetcherDoubles:
    """Counts every call; ``fail`` makes the posts fetch raise, as a dropped connection would."""

    calls: Counter = field(default_factory=Counter)
    fail: bool = False

    def fetch_resolved_questions(
        self, tournament: str, token: str, *, base_url: str = collector.BASE_URL
    ) -> list[dict]:
        self.calls[("posts", base_url, tournament)] += 1
        if self.fail:
            raise ConnectionError("fixture: the posts listing dropped the connection")
        if base_url == MANTIC_API_BASE_URL:
            if tournament != MANTIC_TOURNAMENT_ID:
                return []
            return [
                mantic_post(MANTIC_DISCRETE, multi_resolution=False),
                mantic_post(MANTIC_MULTI_RESOLUTION, multi_resolution=True),
            ]
        return metaculus_fresh_posts() if tournament == TOURNAMENT_ID else []

    def fetch_bot_comments(self, author_id: int, token: str, *, base_url: str = collector.BASE_URL) -> list[dict]:
        self.calls[("comments", base_url)] += 1
        if base_url == MANTIC_API_BASE_URL:
            return [
                {
                    "id": 700_001,
                    "on_post": MANTIC_DISCRETE[1],
                    "text": "*Forecaster 1*: Probability distribution:\n- 50.00% chance of value below 10.0\n",
                    "created_at": "2026-09-10T12:01:00Z",
                    "is_private": False,
                }
            ]
        return bot_comments()

    def total(self, kind: str) -> int:
        return sum(n for key, n in self.calls.items() if key[0] == kind)


def install_doubles(monkeypatch: pytest.MonkeyPatch, doubles: FetcherDoubles) -> list[str]:
    preflights: list[str] = []

    def no_network(*args, **kwargs):
        raise AssertionError(f"api_get reached: {args}")

    monkeypatch.setattr(collector, "fetch_resolved_questions", doubles.fetch_resolved_questions)
    monkeypatch.setattr(collector, "fetch_bot_comments", doubles.fetch_bot_comments)
    monkeypatch.setattr(collector, "api_get", no_network)
    monkeypatch.setattr(api_preflight, "verify_api_identity", lambda base_url, **_options: preflights.append(base_url))
    monkeypatch.setenv(METACULUS_TOKEN_ENV, "fixture-metaculus-token")
    monkeypatch.setenv(MANTIC_TOKEN_ENV, "fixture-mantic-token")
    return preflights


@dataclass(frozen=True)
class Workspace:
    root: Path
    scratch_dir: Path
    telemetry_dir: Path
    research_latest_dir: Path
    research_raw_dir: Path

    def config(self, output_dir: Path, **overrides) -> BuildConfig:
        return BuildConfig(
            output_dir=output_dir,
            scratch_dir=self.scratch_dir,
            telemetry_dir=self.telemetry_dir,
            research_latest_dir=self.research_latest_dir,
            research_raw_dir=self.research_raw_dir,
            **overrides,
        )


def write_workspace(root: Path) -> Workspace:
    workspace = Workspace(root, root / "scratch", root / "telemetry", root / "latest", root / "raw")
    workspace.telemetry_dir.mkdir(parents=True)
    by_marker: dict[str, list[str]] = {}
    for line in telemetry_lines():
        by_marker.setdefault(line["marker"], []).append(json.dumps(line))
    for marker, lines in by_marker.items():
        (workspace.telemetry_dir / f"{marker}.jsonl").write_text("\n".join(lines) + "\n")
    workspace.research_latest_dir.mkdir(parents=True)
    for archive_id, record in archive_records().items():
        (workspace.research_latest_dir / f"{archive_id}.json").write_text(json.dumps(record))
    workspace.research_raw_dir.mkdir(parents=True)
    (workspace.research_raw_dir / f"{RUN_LATE}.jsonl").write_text("{}\n")

    round_posts = [post_json(spec) for spec in ROUND_POSTS]
    comments = [c for index, spec in enumerate(ROUND_POSTS) if (c := comment_json(spec, 900_000 + index))]
    records = collector.records_from_posts(
        round_posts, comments, "fixture-token", research_archive_dir=workspace.research_latest_dir
    )
    for record in records:
        record.update(source_tournament=TOURNAMENT_ID, source_is_fresh_pull=True, record_provenance="fixture")
    tag_fresh_records(
        records,
        prior_records=[],
        prior_label="none",
        weights={},
        weighted_slug=TOURNAMENT_ID,
        telemetry_dir=workspace.telemetry_dir,
    )
    round_dir = workspace.scratch_dir / f"residual_{ROUND_LABEL}"
    round_dir.mkdir(parents=True)
    (round_dir / "perf_all_tagged.json").write_text(json.dumps(records))
    return workspace


@dataclass(frozen=True)
class BuiltBundle:
    workspace: Workspace
    result: BuildResult
    doubles: FetcherDoubles
    preflights: list[str]

    @property
    def output_dir(self) -> Path:
        return self.result.output_dir
