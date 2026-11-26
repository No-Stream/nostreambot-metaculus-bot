"""The section-strip bench (``scripts/probes/section_strip_bench``) without a network.

The bench is a PAID probe on the operator's personal OpenRouter key, so what these tests pin is
everything that decides whether and how money moves: the stripping is exact (each arm loses one
whole section and nothing else), the question each pair is scored as is the one production would
build, the prompt is the production template for the type with the clock anchored to the archived
run date, the scoring plumbing turns a canned reply into the platform score through the real
extraction ladder and CDF build, the spend gates refuse and abort where the estimate says they must,
the meter fails shut on a reply with no charge, and the donated key is unreachable on the paid path.
The model call is a fake returning canned replies with real ``STRUCTURED FORECAST`` blocks;
``tests/conftest.py``'s egress guard has nothing to block.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, OutOfBoundsResolution

from metaculus_bot import prompts as prompts_module
from metaculus_bot import value_extraction
from metaculus_bot.credit_telemetry import PERSONAL_KEY_ALIAS, TokenCounts, record_llm_call_spend, reset_role_spend
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.scoring_common import binary_log_score, mc_log_score
from scripts.probes.section_strip_bench import bundle, cli, plan, report, run

pytestmark = pytest.mark.usefixtures("_clean_ledger")


@pytest.fixture
def _clean_ledger() -> None:
    reset_role_spend()


RUN_DAY = "2026-07-23"
WINDOW = (
    f"Today: {RUN_DAY}\n"
    "Question opened: 2026-07-01 (22 days ago)\n"
    "Scheduled to resolve: 2026-08-31 (39 days from now)\n"
    "Forecasting window: open date → resolution date. Events occurring BEFORE the open date do NOT resolve this "
    "question YES unless the resolution criteria explicitly say they count. If the question uses forward-looking "
    "language ('will X occur by DATE'), interpret it as asking about the open→resolution window, not all of history."
)
FIRST_PASS = "## News Articles (AskNews)\nFirst-pass evidence.\n\n---\n\n## Native Search\nMore evidence."
V1_BODY = "### Gap 1: What is the latest count?\n*Why it matters:* it decides the bin.\n\n- 4,173 cases as of July 24."
V2_BODY = '### ⚠ Corrections to the briefing\nClaim: none.\n\n### Latest count\n- [fetched] 4,173 cases. "quote"'
OPTIONS = ["Texas", "Arizona", "Nevada", "Other"]
METADATA = {"open_time": "2026-07-01T12:00:00Z", "scheduled_resolve_time": "2026-08-31T16:00:00Z"}
SEP, V1H, V2H = bundle.SECTION_SEPARATOR, bundle.V1_SECTION_HEADER, bundle.V2_SECTION_HEADER


def bundle_text(first: str = FIRST_PASS, v1: str = V1_BODY, v2: str = V2_BODY) -> str:
    """A bundle as gap_fill_stages appends it: separator, header, blank line, body, twice."""
    return f"{first}{SEP}{V1H}\n\n{v1}{SEP}{V2H}\n\n{v2}"


def _header(qtype: str, title: str) -> str:
    if qtype == "binary":
        return f"{title}\n\nType: binary (probability of YES)"
    if qtype == "multiple_choice":
        return f"{title}\n\nType: multiple choice\nOptions: {', '.join(OPTIONS)}"
    return f"{title}\n\nType: numeric\nUnits: Cases\nDisplayed range: [7000.0, 14000.0] (lower bound open, upper bound open)"


def make_pair(qid: int, qtype: str, *, resolution: Any, published: dict[str, Any] | None, title: str) -> dict[str, Any]:
    resolved = resolution is not None
    return {
        "question_id": qid,
        "post_id": qid + 100,
        "title": title,
        "type": qtype,
        "resolution_criteria": "Resolves per the CDC count.",
        "fine_print": None,
        "question_header": _header(qtype, title),
        "forecasting_window": WINDOW,
        "first_pass_research": FIRST_PASS,
        "resolution_raw": (str(resolution).lower() if resolved else None),
        "resolution_parsed": resolution,
        "published_forecast": published,
        "spot_peer_score": 12.5 if resolved else None,
    }


def make_perf(qid: int, qtype: str, *, inbound: int = 200) -> dict[str, Any]:
    scaling = {
        "range_min": 7000,
        "range_max": 14000,
        "nominal_min": 7000,
        "nominal_max": 14000,
        "zero_point": None,
        "inbound_outcome_count": inbound,
    }
    return {
        "question_id": qid,
        "type": qtype,
        "metadata": METADATA,
        "scaling": scaling if qtype in ("numeric", "discrete") else None,
        "open_lower_bound": True,
        "open_upper_bound": True,
        "options": OPTIONS if qtype == "multiple_choice" else None,
    }


def _cdf_201() -> list[float]:
    """A legal open-bound 201-point CDF climbing evenly from 0.02 to 0.98."""
    return [0.02 + 0.96 * i / 200 for i in range(201)]


BINARY_YES = make_pair(101, "binary", resolution=True, published={"prob_yes": 0.7}, title="Will the count pass 10k?")
BINARY_NO = make_pair(
    102, "binary", resolution=False, published={"prob_yes": 0.3}, title="Will the strait stay closed?"
)
MC = make_pair(
    103,
    "multiple_choice",
    resolution="Arizona",
    published={"forecast_values": [0.1, 0.6, 0.2, 0.1], "options": OPTIONS},
    title="Which state hosts the plant?",
)
NUMERIC = make_pair(
    104, "numeric", resolution=9500.0, published={"forecast_values": _cdf_201()}, title="How many cases by September?"
)
DISCRETE = make_pair(
    105,
    "discrete",
    resolution="above_upper_bound",
    published={"forecast_values": [0.01 + 0.98 * i / 81 for i in range(82)]},
    title="How many points will the lead be?",
)
UNRESOLVED = make_pair(106, "binary", resolution=None, published=None, title="Will humans go extinct?")
ALL_PAIRS = [BINARY_YES, BINARY_NO, MC, NUMERIC, DISCRETE, UNRESOLVED]
RESOLVED_IDS = [101, 102, 103, 104, 105]


@pytest.fixture
def dataset(tmp_path: Path) -> dict[str, Path]:
    """pairs.jsonl, perf.json and an archive dir carrying every pair's bundle, under tmp_path."""
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in ALL_PAIRS), encoding="utf-8")
    perf_rows = [
        make_perf(p["question_id"], p["type"], inbound=81 if p["type"] == "discrete" else 200) for p in ALL_PAIRS
    ]
    perf_path = tmp_path / "perf.json"
    perf_path.write_text(json.dumps(perf_rows), encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    for pair in ALL_PAIRS:
        record = {"qid": pair["question_id"], "source": "artifact", "research_text": bundle_text()}
        (archive / f"{pair['question_id']}.json").write_text(json.dumps(record), encoding="utf-8")
    return {"pairs": pairs_path, "perf": perf_path, "archive": archive, "out": tmp_path / "out"}


def _dataset_args(dataset: dict[str, Path]) -> list[str]:
    return [
        "--pairs",
        str(dataset["pairs"]),
        "--perf-json",
        str(dataset["perf"]),
        "--archive-dir",
        str(dataset["archive"]),
        "--out-root",
        str(dataset["out"]),
    ]


def _load(dataset: dict[str, Path], **kwargs: Any) -> list[bundle.BenchQuestion]:
    return bundle.load_bench_questions(dataset["pairs"], dataset["perf"], dataset["archive"], **kwargs)


BINARY_REPLY = 'Reasoning.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.22}\n```\n'
MC_REPLY = (
    "Reasoning.\n\n```json\n"
    '{"question_type": "multiple_choice", "option_probs": {"Texas": 0.1, "Arizona": 0.5, "Nevada": 0.3, "Other": 0.1}}'
    "\n```\n"
)


def numeric_reply(values: list[float]) -> str:
    pairs = ", ".join(f'"{p}": {v}' for p, v in zip(STANDARD_PERCENTILES, values, strict=True))
    return (
        "Reasoning.\n\n```json\n"
        f'{{"question_type": "numeric", "declared_percentiles": {{{pairs}}}, "outcome_type": "continuous"}}'
        "\n```\n"
    )


NUMERIC_REPLY = numeric_reply([7500 + 500 * i for i in range(len(STANDARD_PERCENTILES))])
NO_BLOCK_REPLY = "I think about 22 percent but I forgot the block."
REPLY_BY_TITLE = {
    BINARY_YES["title"]: BINARY_REPLY,
    BINARY_NO["title"]: BINARY_REPLY,
    MC["title"]: MC_REPLY,
    NUMERIC["title"]: NUMERIC_REPLY,
    DISCRETE["title"]: NUMERIC_REPLY,
}
FAKE_TOKENS = TokenCounts(prompt=1000, completion=300, cached=50, reasoning=200)


class FakeModel:
    """A ``ModelCall`` answering from the question title it finds in the prompt, at a fixed charge per call."""

    def __init__(self, replies: dict[str, str] | None = None, *, charged_usd: float | None = 0.001) -> None:
        self.replies = replies or REPLY_BY_TITLE
        self.charged_usd = charged_usd
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> run.ModelReply:
        await asyncio.sleep(0)
        self.prompts.append(prompt)
        text = next(reply for title, reply in self.replies.items() if title in prompt)
        return run.ModelReply(text=text, tokens=FAKE_TOKENS, charged_usd=self.charged_usd)


def parser_stub() -> MagicMock:
    """A parser the ladder never reaches while the block parses; its ``model`` is what the salvage rung would read."""
    stub = MagicMock()
    stub.model = "openrouter/test/parser"
    return stub


def _run(
    plan_items: list[plan.PlanItem], call: Any, *, cap_usd: float = 10.0, concurrency: int = 2
) -> list[report.CallRow]:
    return asyncio.run(
        run.run_plan(
            plan_items,
            call=call,
            parser_llm=parser_stub(),
            model_name="fake",
            meter=run.SpendMeter(cap_usd=cap_usd),
            concurrency=concurrency,
            on_row=lambda _row: None,
        )
    )


def _completion(usage: dict[str, Any], text: str = "hello") -> litellm.ModelResponse:
    return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": text}}], usage=usage)


def _awaited(awaitable: Awaitable[run.ModelReply]) -> run.ModelReply:
    async def _wait() -> run.ModelReply:
        return await awaitable

    return asyncio.run(_wait())


class TestSectionStripping:
    def test_each_arm_loses_exactly_the_named_section(self) -> None:
        sections = bundle.split_bundle(bundle_text())
        arms = bundle.arm_texts(sections)

        assert arms["full"] == bundle_text()
        assert arms["minus_both"] == FIRST_PASS
        assert arms["minus_v1"] == FIRST_PASS + sections.v2_block
        assert arms["minus_v2"] == FIRST_PASS + sections.v1_block
        assert V1H not in arms["minus_v1"]
        assert V1_BODY not in arms["minus_v1"]
        assert V2H not in arms["minus_v2"]
        assert V2_BODY not in arms["minus_v2"]
        # The separator also sits between first-pass providers, so the cut keys on the headers, never on it.
        assert SEP in arms["minus_both"]
        assert list(arms) == list(bundle.ARMS)

    def test_the_blocks_carry_their_own_separator_and_header(self) -> None:
        sections = bundle.split_bundle(bundle_text())

        assert sections.v1_block.startswith(SEP + V1H)
        assert sections.v2_block.startswith(SEP + V2H)
        assert sections.v2_block.endswith(V2_BODY)

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            (bundle_text(v1=V1_BODY + "\n" + V2H), "exactly one"),
            (FIRST_PASS + SEP + V2H + "\n\n" + V2_BODY, "exactly one"),
            (bundle_text().replace(SEP + V1H, "\n" + V1H), "separator"),
            (f"{FIRST_PASS}{SEP}{V2H}\n\n{V2_BODY}{SEP}{V1H}\n\n{V1_BODY}", "precede"),
        ],
        ids=["duplicate-v2-header", "missing-v1", "v1-without-separator", "v2-before-v1"],
    )
    def test_a_malformed_bundle_is_refused(self, text: str, reason: str) -> None:
        with pytest.raises(ValueError, match=reason):
            bundle.split_bundle(text)


class TestQuestionConstruction:
    def test_binary(self) -> None:
        question = bundle.build_question(BINARY_YES, make_perf(101, "binary"))

        assert isinstance(question, BinaryQuestion)
        assert (question.id_of_question, question.id_of_post) == (101, 201)
        assert question.page_url == "https://www.metaculus.com/questions/201/"
        assert question.open_time == datetime(2026, 7, 1, 12, tzinfo=UTC)
        assert question.fine_print == ""

    def test_multiple_choice_carries_the_tagged_options(self) -> None:
        question = bundle.build_question(MC, make_perf(103, "multiple_choice"))

        assert isinstance(question, MultipleChoiceQuestion)
        assert question.options == OPTIONS

    def test_numeric_reads_bounds_grid_and_unit(self) -> None:
        question = bundle.build_question(NUMERIC, make_perf(104, "numeric"))

        assert isinstance(question, NumericQuestion)
        assert not isinstance(question, DiscreteQuestion)
        assert (question.lower_bound, question.upper_bound) == (7000.0, 14000.0)
        assert (question.open_lower_bound, question.open_upper_bound) == (True, True)
        assert question.cdf_size == 201
        assert question.unit_of_measure == "Cases"
        assert question.nominal_upper_bound == 14000

    def test_discrete_is_a_discrete_question_on_its_own_grid(self) -> None:
        question = bundle.build_question(DISCRETE, make_perf(105, "discrete", inbound=81))

        assert isinstance(question, DiscreteQuestion)
        assert question.cdf_size == 82

    def test_the_unitless_placeholder_reads_back_as_none(self) -> None:
        pair = {
            **NUMERIC,
            "question_header": NUMERIC["question_header"].replace("Cases", "unspecified (assume unitless)"),
        }
        question = bundle.build_question(pair, make_perf(104, "numeric"))

        assert question.unit_of_measure is None

    def test_resolutions_per_type(self) -> None:
        binary = bundle.build_question(BINARY_NO, make_perf(102, "binary"))
        mc = bundle.build_question(MC, make_perf(103, "multiple_choice"))
        numeric = bundle.build_question(NUMERIC, make_perf(104, "numeric"))
        discrete = bundle.build_question(DISCRETE, make_perf(105, "discrete", inbound=81))

        # A NO is False, and stays a scoreable resolution: the pair file's own caveat.
        assert bundle.typed_resolution(BINARY_NO, binary) is False
        assert bundle.typed_resolution(MC, mc) == "Arizona"
        assert bundle.typed_resolution(NUMERIC, numeric) == 9500.0
        assert bundle.typed_resolution(DISCRETE, discrete) is OutOfBoundsResolution.ABOVE_UPPER_BOUND

    def test_a_resolution_off_the_option_list_is_refused(self) -> None:
        mc = bundle.build_question(MC, make_perf(103, "multiple_choice"))
        with pytest.raises(ValueError, match="not one of"):
            bundle.typed_resolution({**MC, "resolution_parsed": "Utah"}, mc)

    def test_load_keeps_every_resolved_pair_including_a_no_and_skips_the_unresolved(
        self, dataset: dict[str, Path]
    ) -> None:
        by_id = {q.question_id: q for q in _load(dataset)}

        assert list(by_id) == RESOLVED_IDS
        assert by_id[102].resolution is False
        assert by_id[101].today == RUN_DAY
        assert by_id[101].arms["minus_both"] == FIRST_PASS
        assert by_id[101].published_score == pytest.approx(binary_log_score(0.7, True))
        assert by_id[103].published_score == pytest.approx(mc_log_score([0.1, 0.6, 0.2, 0.1], 1))
        numeric_published = by_id[104].published_score
        assert numeric_published is not None
        assert math.isfinite(numeric_published)

    def test_load_narrows_to_the_requested_ids_and_names_a_miss(self, dataset: dict[str, Path]) -> None:
        assert [q.question_id for q in _load(dataset, only={101, 104})] == [101, 104]

        with pytest.raises(ValueError, match=r"\[106, 999\]"):
            _load(dataset, only={101, 106, 999})

    def test_load_refuses_an_archive_that_disagrees_with_the_pair(self, dataset: dict[str, Path]) -> None:
        record = {"qid": 101, "source": "artifact", "research_text": bundle_text(first="## Something else entirely")}
        (dataset["archive"] / "101.json").write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(ValueError, match="disagrees"):
            _load(dataset)


class TestPromptSelection:
    def test_binary_prompt_is_anchored_to_the_archived_run_date(self, dataset: dict[str, Path]) -> None:
        question = {q.question_id: q for q in _load(dataset)}[101]
        with plan.anchored_clock(question.forecasting_window, question.today):
            prompt = plan.render_prompt(question.question, question.arms["full"])

        assert WINDOW in prompt
        assert f"as of {RUN_DAY}" in prompt
        assert datetime.now(UTC).strftime("%Y-%m-%d") not in prompt
        assert V1_BODY in prompt
        assert V2_BODY in prompt
        assert "posterior_prob" in prompt

    def test_the_clock_is_restored_after_the_build(self) -> None:
        original_window, original_today = prompts_module._forecasting_window_str, prompts_module._today_str
        with plan.anchored_clock(WINDOW, RUN_DAY):
            assert prompts_module._today_str() == RUN_DAY
        assert prompts_module._forecasting_window_str is original_window
        assert prompts_module._today_str is original_today

    def test_multiple_choice_and_numeric_pick_their_own_templates(self, dataset: dict[str, Path]) -> None:
        questions = {q.question_id: q for q in _load(dataset)}
        with plan.anchored_clock(WINDOW, RUN_DAY):
            mc_prompt = plan.render_prompt(questions[103].question, questions[103].arms["minus_v1"])
            numeric_prompt = plan.render_prompt(questions[104].question, questions[104].arms["minus_both"])
            discrete_prompt = plan.render_prompt(questions[105].question, questions[105].arms["full"])

        assert "option_probs" in mc_prompt
        assert all(option in mc_prompt for option in OPTIONS)
        assert V1_BODY not in mc_prompt
        assert V2_BODY in mc_prompt
        assert "declared_percentiles" in numeric_prompt
        assert "The upper bound is open" in numeric_prompt
        assert "Cases" in numeric_prompt
        assert V1_BODY not in numeric_prompt
        assert V2_BODY not in numeric_prompt
        assert "declared_percentiles" in discrete_prompt

    def test_the_plan_has_one_prompt_per_question_arm_replicate(self, dataset: dict[str, Path]) -> None:
        items = plan.build_plan(_load(dataset), ["full", "minus_v1"], 2, model="fake/model")

        assert len(items) == 5 * 2 * 2
        first, second = (item for item in items if item.question.question_id == 101 and item.arm == "full")
        assert first.prompt.endswith(plan.replicate_nonce(1))
        assert second.prompt.endswith(plan.replicate_nonce(2))
        assert first.prompt.removesuffix(plan.replicate_nonce(1)) == second.prompt.removesuffix(plan.replicate_nonce(2))
        assert first.prompt_tokens == plan.count_tokens("fake/model", first.prompt)
        minus = next(i for i in items if i.question.question_id == 101 and i.arm == "minus_v1")
        assert first.prompt_tokens > minus.prompt_tokens > 0


class TestScoringPlumbing:
    def _plan(self, dataset: dict[str, Path], arms: list[str], seeds: int = 1) -> list[plan.PlanItem]:
        return plan.build_plan(_load(dataset), arms, seeds, model="fake/model")

    def test_canned_replies_score_through_the_ladder_and_the_cdf_build(self, dataset: dict[str, Path]) -> None:
        rows = _run(self._plan(dataset, ["full", "minus_v1"]), FakeModel())

        assert {row.status for row in rows} == {report.STATUS_SCORED}
        by_key = {(row.question_id, row.arm): row for row in rows}
        yes = by_key[(101, "full")]
        assert yes.score == pytest.approx(binary_log_score(0.22, True))
        assert (yes.rung, yes.block_present, yes.forecast) == ("block", True, 0.22)
        assert by_key[(102, "full")].score == pytest.approx(binary_log_score(0.22, False))
        mc = by_key[(103, "full")]
        assert mc.forecast == pytest.approx([0.1, 0.5, 0.3, 0.1])
        assert mc.score == pytest.approx(mc_log_score([0.1, 0.5, 0.3, 0.1], OPTIONS.index("Arizona")))
        numeric = by_key[(104, "full")]
        assert numeric.score is not None
        assert math.isfinite(numeric.score)
        assert numeric.forecast is not None
        assert len(numeric.forecast) == len(STANDARD_PERCENTILES)
        discrete_score = by_key[(105, "full")].score
        assert discrete_score is not None
        assert math.isfinite(discrete_score)
        # The same reply in two arms scores identically: the arm text changes the prompt, never the scorer.
        assert by_key[(104, "minus_v1")].score == numeric.score
        for row in rows:
            assert (row.prompt_tokens, row.completion_tokens, row.reasoning_tokens, row.cached_tokens) == (
                1000,
                300,
                200,
                50,
            )
            assert row.cost_usd == 0.001
            assert row.rationale

    def test_a_binary_value_outside_the_clamp_is_recorded_as_published(self, dataset: dict[str, Path]) -> None:
        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 101]
        extreme = 'Sure.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.005}\n```\n'
        rows = _run(items, FakeModel({BINARY_YES["title"]: extreme}))

        assert rows[0].forecast == 0.02
        assert rows[0].score == pytest.approx(binary_log_score(0.02, True))

    def test_a_reply_without_a_block_is_an_extraction_failure_row(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _no_salvage(*_args: Any, **_kwargs: Any) -> Any:
            await asyncio.sleep(0)
            raise RuntimeError("no parser LLM in tests")

        monkeypatch.setattr(value_extraction, "parse_structured", _no_salvage)
        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 101]
        rows = _run(items, FakeModel({BINARY_YES["title"]: NO_BLOCK_REPLY}))

        assert [row.status for row in rows] == [report.STATUS_EXTRACTION_FAILED]
        assert rows[0].score is None
        assert rows[0].error is not None
        assert "no parser LLM" in rows[0].error
        assert rows[0].rationale == NO_BLOCK_REPLY
        assert rows[0].cost_usd == 0.001

    def test_a_scale_error_trips_the_unit_mismatch_guard(self, dataset: dict[str, Path]) -> None:
        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 104]
        tiny = numeric_reply([0.0075 + 0.0005 * i for i in range(len(STANDARD_PERCENTILES))])
        rows = _run(items, FakeModel({NUMERIC["title"]: tiny}))

        assert [row.status for row in rows] == [report.STATUS_UNIT_MISMATCH]
        assert rows[0].error is not None
        assert "Unit mismatch" in rows[0].error

    def test_an_account_refusal_stops_the_run_after_the_first_call(self, dataset: dict[str, Path]) -> None:
        """The shape the first live reachability call returned: a 403 gate on the account (the 18+ attestation),
        which every later call would hit identically, so the run stops with the body on the console."""
        body = '{"error":{"message":"This model requires you to complete the following before use: 18+ age confirmation."}}'

        async def _refused(_prompt: str) -> run.ModelReply:
            await asyncio.sleep(0)
            raise litellm.exceptions.APIError(status_code=403, message=body, llm_provider="openrouter", model="m")

        items = plan.build_plan(_load(dataset), ["full"], 1, model="fake/model")
        rows = _run(items, _refused, concurrency=1)

        statuses = [row.status for row in rows]
        assert statuses.count(report.STATUS_API_ERROR) == 1
        assert statuses.count(report.STATUS_SKIPPED_SPEND_CAP) == len(items) - 1
        skipped = next(row for row in rows if row.status == report.STATUS_SKIPPED_SPEND_CAP)
        assert skipped.error is not None
        assert "refused" in skipped.error
        assert "18+ age confirmation" in skipped.error

    def test_a_provider_error_is_a_row_not_a_crash(self, dataset: dict[str, Path]) -> None:
        async def _timeout(_prompt: str) -> run.ModelReply:
            await asyncio.sleep(0)
            raise litellm.exceptions.Timeout(message="slow upstream", model="m", llm_provider="openrouter")

        items = [item for item in self._plan(dataset, ["full"]) if item.question.question_id == 101]
        rows = _run(items, _timeout)

        assert [row.status for row in rows] == [report.STATUS_API_ERROR]
        assert rows[0].error is not None
        assert rows[0].error.startswith("Timeout")
        assert rows[0].cost_usd is None


class TestTheModelCall:
    def test_the_request_is_the_prompt_alone_on_the_personal_key_with_no_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(
            return_value=_completion(
                {
                    "prompt_tokens": 12,
                    "completion_tokens": 9,
                    "total_tokens": 21,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                    "cost": 0.00042,
                }
            )
        )
        monkeypatch.setattr(run.litellm, "acompletion", acompletion)

        reply = _awaited(run.build_model_call("vendor/cheap", "sk-personal", reasoning_effort="low")("the prompt"))

        assert acompletion.await_args is not None
        kwargs = acompletion.await_args.kwargs
        assert kwargs["model"] == "openrouter/vendor/cheap"
        assert kwargs["api_key"] == "sk-personal"
        assert kwargs["messages"] == [{"role": "user", "content": "the prompt"}]
        assert "tools" not in kwargs
        assert kwargs["metadata"] == {"role": run.FORECASTER_ROLE, "key_alias": PERSONAL_KEY_ALIAS}
        assert (kwargs["max_tokens"], kwargs["timeout"], kwargs["num_retries"]) == (64_000, 480.0, 2)
        assert kwargs["reasoning_effort"] == "low"
        assert reply == run.ModelReply(
            text="hello", tokens=TokenCounts(prompt=12, completion=9, cached=0, reasoning=4), charged_usd=0.00042
        )

    def test_an_unset_effort_is_not_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acompletion = AsyncMock(return_value=_completion({"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}))
        monkeypatch.setattr(run.litellm, "acompletion", acompletion)

        _awaited(run.build_model_call("vendor/cheap", "sk-personal", reasoning_effort=None)("p"))

        assert acompletion.await_args is not None
        assert "reasoning_effort" not in acompletion.await_args.kwargs

    def test_the_charge_follows_the_ledger_rule(self) -> None:
        """Off BYOK the upstream figure is an echo of the same money; on BYOK it is the real bill."""
        echoed = _completion(
            {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.3, "cost_details": {"upstream_inference_cost": 0.3}}
        )
        byok = _completion(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost": 0.0,
                "is_byok": True,
                "cost_details": {"upstream_inference_cost": 0.5},
            }
        )
        unpriced = _completion({"prompt_tokens": 1, "completion_tokens": 1})

        assert run.reply_from_response(echoed).charged_usd == pytest.approx(0.3)
        assert run.reply_from_response(byok).charged_usd == pytest.approx(0.5)
        assert run.reply_from_response(unpriced).charged_usd is None


class TestSpendGates:
    def test_the_meter_stops_the_run_once_measured_spend_reaches_the_cap(self, dataset: dict[str, Path]) -> None:
        items = plan.build_plan(_load(dataset), ["full"], 2, model="fake/model")
        model = FakeModel(charged_usd=3.0)
        rows = _run(items, model, cap_usd=5.0, concurrency=1)

        statuses = [row.status for row in rows]
        # Two calls ($6) pass the $5 cap; every later call is skipped before it is made.
        assert statuses.count(report.STATUS_SCORED) == 2
        assert statuses.count(report.STATUS_SKIPPED_SPEND_CAP) == len(items) - 2
        assert len(model.prompts) == 2
        skipped = next(row for row in rows if row.status == report.STATUS_SKIPPED_SPEND_CAP)
        assert skipped.error is not None
        assert "reached the cap" in skipped.error

    def test_a_reply_with_no_charge_stops_the_run_instead_of_reading_as_free(self, dataset: dict[str, Path]) -> None:
        items = plan.build_plan(_load(dataset), ["full"], 1, model="fake/model")
        model = FakeModel(charged_usd=None)
        rows = _run(items, model, cap_usd=10.0, concurrency=1)

        statuses = [row.status for row in rows]
        assert statuses.count(report.STATUS_SCORED) == 1
        assert statuses.count(report.STATUS_SKIPPED_SPEND_CAP) == len(items) - 1
        assert len(model.prompts) == 1
        skipped = next(row for row in rows if row.status == report.STATUS_SKIPPED_SPEND_CAP)
        assert skipped.error is not None
        assert "no charge" in skipped.error

    def test_the_meter_counts_the_parser_ledger_too(self) -> None:
        meter = run.SpendMeter(cap_usd=1.0)
        record_llm_call_spend(run.PARSER_ROLE, PERSONAL_KEY_ALIAS, cost_usd=0.75, byok_upstream_usd=0.75)
        meter.add(0.2)
        assert meter.stop_reason is None
        meter.add(0.1)
        # The ledger's ``charged_usd`` is read, so the echoed upstream figure is not double counted.
        assert meter.measured_usd == pytest.approx(1.05)
        assert meter.stop_reason is not None

    def test_a_bare_invocation_prints_the_estimate_and_refuses(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "run_bench", lambda *_a, **_k: pytest.fail("the refusal path started a run"))

        assert cli.main(_dataset_args(dataset)) == 2
        out = capsys.readouterr().out
        assert "Estimated cost of this run" in out
        assert "--i-accept-spend" in out
        assert "calls: 60;" in out  # 5 questions x 4 arms x 3 replicates
        assert not dataset["out"].exists()

    def test_dry_run_prints_the_plan_and_calls_nothing(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "run_bench", lambda *_a, **_k: pytest.fail("a dry run started a run"))

        assert cli.main(["--dry-run", "--seeds", "2", "--arms", "full", "minus_v1", *_dataset_args(dataset)]) == 0
        out = capsys.readouterr().out
        assert "Plan: 5 resolved questions" in out
        assert "calls: 20;" in out
        assert "Estimated cost of this run" in out
        assert "replicates of the full 5 x 2 set" in out
        assert "salvage call to the parser model" in out

    def test_an_estimate_over_the_cap_aborts_before_the_first_call(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "run_bench", lambda *_a, **_k: pytest.fail("an over-cap estimate started a run"))

        code = cli.main(["--i-accept-spend", "--max-spend-usd", "0.000001", *_dataset_args(dataset)])

        assert code == 2
        assert "Aborting before the first call" in capsys.readouterr().out
        assert not dataset["out"].exists()

    def test_the_estimate_arithmetic_on_known_counts(self) -> None:
        estimate = plan.Estimate(
            n_questions=2,
            n_arms=2,
            n_seeds=1,
            n_calls=4,
            prompt_tokens_total=40_000,
            median_prompt_tokens=10_000,
            output_tokens_per_call=800,
            price_in_usd_per_m=0.10,
            price_out_usd_per_m=0.20,
        )
        # 40,000 prompt tokens at $0.10/M is $0.004; 4 x 800 completion tokens at $0.20/M is $0.00064.
        assert estimate.prompt_usd == pytest.approx(0.004)
        assert estimate.completion_usd == pytest.approx(0.00064)
        assert estimate.total_usd == pytest.approx(0.00464)
        assert estimate.usd_per_call == pytest.approx(0.00116)
        assert estimate.calls_under(10.0) == 8620
        assert estimate.seeds_under(10.0) == 2155

    def test_the_estimate_reads_the_plan(self, dataset: dict[str, Path]) -> None:
        items = plan.build_plan(_load(dataset), list(bundle.ARMS), 3, model="fake/model")
        estimate = plan.estimate_spend(items, output_tokens=800, price_in=0.10, price_out=0.20)

        assert (estimate.n_questions, estimate.n_arms, estimate.n_seeds, estimate.n_calls) == (5, 4, 3, 60)
        assert estimate.prompt_tokens_total == sum(item.prompt_tokens for item in items)
        assert estimate.median_prompt_tokens > 0

    @pytest.mark.parametrize(
        "argv",
        [
            ["--dry-run", "--arms", "minus_v1"],
            ["--dry-run", "--max-spend-usd", "nan"],
            ["--dry-run", "--max-spend-usd", "0"],
            ["--dry-run", "--seeds", "0"],
            ["--dry-run", "--concurrency", "0"],
            ["--dry-run", "--bootstrap-seed", "-1"],
        ],
        ids=["no-full-arm", "nan-cap", "zero-cap", "zero-seeds", "zero-concurrency", "negative-bootstrap-seed"],
    )
    def test_arguments_that_would_disable_or_wedge_the_run_are_refused(
        self, dataset: dict[str, Path], argv: list[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args([*argv, *_dataset_args(dataset)])

    @pytest.mark.parametrize(
        ("argv", "missing"),
        [
            (["--dry-run"], "--pairs and --perf-json"),
            (["--dry-run", "--pairs", "p.jsonl"], "--perf-json"),
            (["--dry-run", "--perf-json", "perf.json"], "--pairs"),
        ],
        ids=["neither", "no-perf-json", "no-pairs"],
    )
    def test_the_round_dataset_has_to_be_passed(self, argv: list[str], missing: str, capsys) -> None:
        """``--pairs`` and ``--perf-json`` used to default into ``scratch/cost_pass_2026-09-09/``
        and ``scratch/residual_2026-09-09/``. Those directories are gitignored, so a fresh clone
        got a confusing missing-file error, and on the operator's machine they keep resolving
        after the round is superseded, which benches a stale pairing under a current-looking run."""
        with pytest.raises(SystemExit) as exit_info:
            cli.parse_args(argv)

        assert exit_info.value.code == 2
        assert missing in capsys.readouterr().err

    def test_rescore_needs_no_round_dataset(self, tmp_path: Path) -> None:
        """``--rescore`` rebuilds results from a finished run dir and reads neither file."""
        args = cli.parse_args(["--rescore", str(tmp_path)])

        assert (args.rescore, args.pairs, args.perf_json) == (tmp_path, None, None)


class TestPersonalKeyOnly:
    def test_the_donated_key_is_unreachable_on_the_paid_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.probes.bench_common.paid_call.load_environment", lambda: None)
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "sk-donated")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-personal")
        monkeypatch.setenv("DONATED_OPENROUTER_KEY_ENABLED", "true")

        assert run.personal_key_only_environment() == "sk-personal"
        assert "OAI_ANTH_OPENROUTER_KEY" not in os.environ
        assert os.environ["DONATED_OPENROUTER_KEY_ENABLED"] == "false"

    def test_no_personal_key_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.probes.bench_common.paid_call.load_environment", lambda: None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            run.personal_key_only_environment()


def _row(qid: int, qtype: str, arm: str, seed: int, score: float, *, n_options: int | None = None) -> report.CallRow:
    return report.CallRow(
        question_id=qid, qtype=qtype, n_options=n_options, arm=arm, seed=seed, status=report.STATUS_SCORED, score=score
    )


SUMMARIES = [
    {
        "question_id": 1,
        "qtype": "binary",
        "title": "b",
        "resolution": "True",
        "n_options": None,
        "published_score": 40.0,
        "spot_peer_score": 5.0,
    },
    {
        "question_id": 2,
        "qtype": "multiple_choice",
        "title": "m",
        "resolution": "A",
        "n_options": 3,
        "published_score": 10.0,
        "spot_peer_score": 1.0,
    },
    {
        "question_id": 3,
        "qtype": "numeric",
        "title": "n",
        "resolution": "9.0",
        "n_options": None,
        "published_score": 20.0,
        "spot_peer_score": 2.0,
    },
]


class TestAggregation:
    def test_paired_deltas_average_replicates_then_pair_by_question(self) -> None:
        rows = [
            _row(1, "binary", "full", 1, 60.0),
            _row(1, "binary", "full", 2, 40.0),
            _row(1, "binary", "minus_v1", 1, 30.0),
            _row(2, "multiple_choice", "full", 1, 20.0, n_options=3),
            _row(2, "multiple_choice", "minus_v1", 1, 30.0, n_options=3),
            _row(3, "numeric", "full", 1, 15.0),
            _row(3, "numeric", "minus_v1", 1, 5.0),
            report.CallRow(
                question_id=3,
                qtype="numeric",
                n_options=None,
                arm="minus_v1",
                seed=2,
                status=report.STATUS_EXTRACTION_FAILED,
            ),
        ]
        results = report.aggregate(rows, SUMMARIES, arms=["full", "minus_v1"], bootstrap_seed=7)

        deltas = results["paired_deltas"]["minus_v1"]
        assert deltas["binary"]["mean_delta"] == pytest.approx(50.0 - 30.0)
        assert deltas["multiple_choice"]["mean_delta"] == pytest.approx(-10.0)
        assert deltas["numeric"]["mean_delta"] == pytest.approx(10.0)
        peer = deltas["all_peer_points"]
        assert peer["n"] == 3
        assert peer["mean_delta"] == pytest.approx((20.0 * math.log(2) - 10.0 * math.log(3) + 10.0) / 3)
        assert (peer["n_full_better"], peer["n_arm_better"]) == (2, 1)
        assert 0.0 <= peer["sign_test_p"] <= 1.0
        # Too few pairs for a bootstrap interval, so the CI collapses onto the mean rather than fabricating one.
        assert peer["ci95_low"] == peer["ci95_high"] == peer["mean_delta"]

        summary = results["arm_summary"]
        assert summary["full"]["binary"] == {"n": 1, "mean": 50.0, "median": 50.0}
        assert "all" not in summary["full"]
        assert summary["published"]["binary"]["mean"] == 40.0
        assert results["replicate_spread"] == {"full": {"n_questions": 1, "mean_std": pytest.approx(math.sqrt(200.0))}}
        assert results["call_status_by_arm"]["minus_v1"] == {
            report.STATUS_SCORED: 3,
            report.STATUS_EXTRACTION_FAILED: 1,
        }
        per_question = {row["question_id"]: row for row in results["per_question"]}
        assert per_question[1]["deltas"] == {"minus_v1": pytest.approx(20.0)}
        assert per_question[1]["arm_scores"] == {"full": 50.0, "minus_v1": 30.0}

    def test_the_markdown_leads_with_the_deltas(self) -> None:
        rows = [_row(1, "binary", "full", 1, 60.0), _row(1, "binary", "minus_v1", 1, 30.0)]
        results = report.aggregate(rows, SUMMARIES[:1], arms=["full", "minus_v1"], bootstrap_seed=0)
        text = report.render_markdown(results, model="fake/model", seeds=1)

        assert text.startswith("# Section-strip bench")
        assert "## Paired deltas, full minus arm" in text
        assert "| minus_v1 | binary | 1 | +30.00 |" in text
        assert "| published | binary | 1 | 40.00 | 40.00 |" in text
        assert "reference level" in text
        assert "fewer than two scored replicates" in text


class TestPaidPathWithFakes:
    def test_the_whole_run_writes_every_artifact_and_rescores_identically(
        self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model = FakeModel()
        order: list[str] = []

        def _record(name: str, value: Any) -> Any:
            def _factory(*_a: Any, **_k: Any) -> Any:
                order.append(name)
                return value

            return _factory

        async def _drained() -> None:
            await asyncio.sleep(0)

        monkeypatch.setattr(cli, "personal_key_only_environment", _record("environment", "sk-personal"))
        monkeypatch.setattr(cli, "install_role_spend_tracker", lambda: None)
        monkeypatch.setattr(cli, "drain_litellm_callbacks", _drained)
        monkeypatch.setattr(cli, "build_model_call", _record("model", model))
        monkeypatch.setattr(cli, "build_parser_llm", _record("parser", parser_stub()))

        code = cli.main(["--i-accept-spend", "--seeds", "1", "--concurrency", "3", *_dataset_args(dataset)])

        assert code == 0
        # The donated key is scrubbed before any client exists, which is what makes it unreachable.
        assert order[0] == "environment"
        assert set(order[1:]) == {"model", "parser"}
        (run_dir,) = list(dataset["out"].iterdir())
        rows = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 5 * 4 == len(model.prompts)
        assert {row["status"] for row in rows} == {report.STATUS_SCORED}
        run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_meta["measured_usd"] == pytest.approx(20 * 0.001)
        assert run_meta["stop_reason"] is None
        assert [q["question_id"] for q in run_meta["questions"]] == RESOLVED_IDS
        results_text = (run_dir / "results.json").read_text(encoding="utf-8")
        assert set(json.loads(results_text)["paired_deltas"]) == set(bundle.STRIPPED_ARMS)
        assert json.loads(results_text)["spend"]["forecaster_usd"] == pytest.approx(0.02)
        assert (run_dir / "SUMMARY.md").read_text(encoding="utf-8").startswith("# Section-strip bench")
        assert (run_dir / "bench.log").exists()

        assert cli.main(["--rescore", str(run_dir)]) == 0
        assert (run_dir / "results.json").read_text(encoding="utf-8") == results_text
        assert "Rescored 20 calls" in capsys.readouterr().out
