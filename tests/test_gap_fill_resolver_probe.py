"""The gap-fill resolver probe: archive parsing, grid parsing, the spend gate, the ledger join, the write-up.

``scripts/probes/gap_fill_resolver_probe.py`` sits in AGENTS.md's PAID list: it replays one
question's archived gap-fill v1 gaps through the production resolver at several models and search
context sizes, gaps x cells billed calls on the operator's personal OpenRouter key. Between a bare
invocation and that spend there is one guard, the ``--i-accept-spend`` refusal, so the first class
pins that the refusal exits before the Metaculus read and before any LLM is built, and that the
accepted path makes exactly gaps x cells calls through ``build_native_search_llm`` with the cell's
model, the cell's context size and the production reasoning effort, on the personal key only.

Nothing here touches the network: the archive is a temp directory, the Metaculus read and the LLM
builder are replaced with fakes, and the fake LLM books its call on the production
``CREDIT_ROLE_SPEND`` ledger the way litellm's success callback would, so the probe's usage join is
tested against the real ledger rather than a stub of it.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from metaculus_bot.constants import (
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    GAP_FILL_RESOLVER_MODEL,
    GAP_FILL_RESOLVER_REASONING_EFFORT,
    NATIVE_SEARCH_CONTEXT_SIZE,
)
from metaculus_bot.credit_telemetry import TokenCounts, record_llm_call_spend, reset_role_spend
from metaculus_bot.research.providers import build_native_search_llm
from scripts.probes import gap_fill_resolver_probe as probe

_QUESTION_ID = 44267
_POST_ID = 44256
_RUN_ID = "28674047443"

_RESEARCH_TEXT = """## News Articles (AskNews)

Some briefing.

## Web Research (Native Search)

### Current official July 2026 data points found

More text.

---

## Targeted Gap-Fill (second pass)

### Gap 1: Retrieve the official MND entries for July 1-3, 2026 to confirm the count field used for resolution.

_Why it matters: The first pass may be mixing total aircraft with ADIZ-entering aircraft._

The July 3 entry lists 30 sorties, 26 of which entered the zone ([MND](https://example.test/mnd)).

### Gap 2: Clarify whether the question resolves on the headline total or the ADIZ subset.

_Why it matters: The two readings differ by four aircraft on July 3._

The resolving metric is the headline total, per a sister question ([Metaculus](https://example.test/q)).

### Gap 4: Verify the Han Kuang schedule.

No why line on this one; the answer starts immediately.

## Provider Diagnostics

- asknews: ok | 12670 chars | 39905 ms
"""


def _archive(tmp_path: Path, *, source: str = "artifact", with_raw: bool = False) -> tuple[Path, Path]:
    """Write one ``latest/<qid>.json`` record (and optionally the run's raw log) under ``tmp_path``."""
    latest_dir = tmp_path / "latest"
    raw_dir = tmp_path / "raw"
    latest_dir.mkdir()
    raw_dir.mkdir()
    record = {
        "schema_version": 2,
        "qid": _QUESTION_ID,
        "post_id": _POST_ID,
        "run_id": _RUN_ID,
        "timestamp": "2026-07-03T17:07:13+00:00",
        "question_text": "What will be the highest daily number of PLA aircraft ...?",
        "research_text": _RESEARCH_TEXT,
        "source": source,
    }
    (latest_dir / f"{_QUESTION_ID}.json").write_text(json.dumps(record), encoding="utf-8")
    if with_raw:
        payload = {
            "gaps": [
                {"gap": "Retrieve the official MND entries.", "search_query": "MND regional dynamic list July 3 2026"},
                {
                    "gap": "Headline total or ADIZ subset?",
                    "search_query": "PLA aircraft ADIZ subset definition",
                    "why_matters": "Four aircraft apart.",
                },
                {"gap": "A gap whose resolver failed in production.", "search_query": "failed gap query"},
                {"gap": "Verify the Han Kuang schedule.", "search_query": ""},
            ],
            "results": ["...", "...", "TimeoutError()", "..."],
        }
        other = {"qid": 1, "provider": "gap_fill", "payload": {"gaps": [{"gap": "other question"}], "results": []}}
        mine = {"qid": _QUESTION_ID, "provider": "gap_fill", "payload": payload}
        asknews = {"qid": _QUESTION_ID, "provider": "asknews", "phase": "hot", "payload": []}
        (raw_dir / f"{_RUN_ID}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in (other, asknews, mine)) + "\n", encoding="utf-8"
        )
    return latest_dir, raw_dir


class TestParseArchivedGaps:
    def test_reads_index_text_why_and_answer_per_gap(self) -> None:
        gaps = probe.parse_archived_gaps(_RESEARCH_TEXT)

        assert [gap.index for gap in gaps] == [1, 2, 4], "the index is the analyzer's own, holes included"
        assert gaps[0].gap.startswith("Retrieve the official MND entries")
        assert gaps[0].why_matters == "The first pass may be mixing total aircraft with ADIZ-entering aircraft."
        assert gaps[0].archived_answer.startswith("The July 3 entry lists 30 sorties")
        assert gaps[1].archived_answer.endswith("([Metaculus](https://example.test/q)).")
        # The rendered section carries no search query, so the gap text stands in, as the analyzer's own default does.
        assert all(gap.search_query == gap.gap for gap in gaps)

    def test_a_gap_without_a_why_line_keeps_its_whole_body_as_the_answer(self) -> None:
        gaps = probe.parse_archived_gaps(_RESEARCH_TEXT)

        assert gaps[2].why_matters == ""
        assert gaps[2].archived_answer == "No why line on this one; the answer starts immediately."

    def test_the_section_stops_at_the_next_top_level_heading(self) -> None:
        gaps = probe.parse_archived_gaps(_RESEARCH_TEXT)

        assert "Provider Diagnostics" not in gaps[-1].archived_answer
        assert "asknews" not in gaps[-1].archived_answer

    def test_no_gap_fill_section_means_no_gaps(self) -> None:
        assert probe.parse_archived_gaps("## News Articles (AskNews)\n\nnothing else\n") == []


class TestLoadArchivedQuestion:
    def test_rendered_gaps_when_the_run_has_no_raw_log(self, tmp_path: Path) -> None:
        latest_dir, raw_dir = _archive(tmp_path)

        question = probe.load_archived_question(_QUESTION_ID, latest_dir=latest_dir, raw_dir=raw_dir)

        assert (question.question_id, question.post_id, question.run_id) == (_QUESTION_ID, _POST_ID, _RUN_ID)
        assert [gap.index for gap in question.gaps] == [1, 2, 4]
        assert question.gaps[1].search_query == question.gaps[1].gap

    def test_the_raw_analyzer_record_is_authoritative_when_present(self, tmp_path: Path) -> None:
        latest_dir, raw_dir = _archive(tmp_path, with_raw=True)

        question = probe.load_archived_question(_QUESTION_ID, latest_dir=latest_dir, raw_dir=raw_dir)

        assert [gap.index for gap in question.gaps] == [1, 2, 3, 4], "every analyzer gap, including the one that failed"
        assert question.gaps[0].search_query == "MND regional dynamic list July 3 2026"
        assert question.gaps[1].why_matters == "Four aircraft apart."
        # Production's answer joins by index from the rendered section; a failed gap has none.
        assert question.gaps[1].archived_answer.startswith("The resolving metric is the headline total")
        assert question.gaps[2].archived_answer == ""
        # An empty raw search_query falls back to the gap text, as the analyzer parser does.
        assert question.gaps[3].search_query == "Verify the Han Kuang schedule."

    def test_only_an_artifact_record_is_replayed(self, tmp_path: Path) -> None:
        latest_dir, raw_dir = _archive(tmp_path, source="comment_backfill")

        with pytest.raises(ValueError, match="not 'artifact'"):
            probe.load_archived_question(_QUESTION_ID, latest_dir=latest_dir, raw_dir=raw_dir)

    def test_a_question_missing_from_the_archive_raises(self, tmp_path: Path) -> None:
        latest_dir, raw_dir = _archive(tmp_path)

        with pytest.raises(FileNotFoundError):
            probe.load_archived_question(999, latest_dir=latest_dir, raw_dir=raw_dir)


class TestSelectGaps:
    def test_none_keeps_every_gap_and_a_selection_narrows_in_the_given_order(self) -> None:
        gaps = probe.parse_archived_gaps(_RESEARCH_TEXT)

        assert probe.select_gaps(gaps, None) == gaps
        assert [gap.index for gap in probe.select_gaps(gaps, "4,1")] == [4, 1]

    def test_an_unknown_index_is_an_error_naming_what_exists(self) -> None:
        gaps = probe.parse_archived_gaps(_RESEARCH_TEXT)

        with pytest.raises(ValueError, match=r"\[3\].*\[1, 2, 4\]"):
            probe.select_gaps(gaps, "1,3")


class TestParseGrid:
    def test_the_default_grid_is_the_two_models_at_the_three_sizes(self) -> None:
        cells = probe.parse_grid(probe.DEFAULT_GRID)

        assert [cell.label for cell in cells] == [
            "current:high",
            "current:medium",
            "current:low",
            "luna:high",
            "luna:medium",
            "luna:low",
        ]
        assert {cell.model_slug for cell in cells if cell.model_alias == "current"} == {GAP_FILL_RESOLVER_MODEL}
        assert {cell.model_slug for cell in cells if cell.model_alias == "luna"} == {probe.CANDIDATE_MODEL}

    def test_a_bare_openrouter_slug_is_accepted_as_its_own_alias(self) -> None:
        (cell,) = probe.parse_grid(["openai/gpt-5.6-sol:medium"])

        assert cell == probe.GridCell("openai/gpt-5.6-sol", "openai/gpt-5.6-sol", "medium")

    def test_a_third_field_is_the_reasoning_effort_and_defaults_to_production(self) -> None:
        with_effort, default = probe.parse_grid(["luna:high:medium", "luna:high"])

        assert with_effort == probe.GridCell("luna", probe.CANDIDATE_MODEL, "high", "medium")
        assert (with_effort.label, with_effort.model_label) == ("luna:high:medium", "luna@medium")
        assert default.reasoning_effort == GAP_FILL_RESOLVER_REASONING_EFFORT
        assert (default.label, default.model_label) == ("luna:high", "luna")

    def test_a_slug_carrying_a_colon_still_parses(self) -> None:
        (cell,) = probe.parse_grid(["openai/gpt-5.6-luna:batch:low"])

        assert cell.model_slug == "openai/gpt-5.6-luna:batch"
        assert (cell.search_context_size, cell.reasoning_effort) == ("low", GAP_FILL_RESOLVER_REASONING_EFFORT)

    @pytest.mark.parametrize("spec", ["current", "current:huge", "sol:high", ":high", "current:high:turbo"])
    def test_malformed_cells_are_rejected(self, spec: str) -> None:
        with pytest.raises(ValueError, match="grid cell"):
            probe.parse_grid([spec])

    def test_a_repeated_cell_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            probe.parse_grid(["current:high", "current:high"])


class TestEstimate:
    def test_the_ceiling_is_gaps_times_cells_times_the_per_call_ceiling(self) -> None:
        assert probe.estimate_ceiling_usd(5, 6) == pytest.approx(6.0)
        assert probe.estimate_ceiling_usd(1, 1) == pytest.approx(probe.PER_CALL_CEILING_USD)

    def test_the_printed_estimate_names_the_call_count_the_ceiling_and_the_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        latest_dir, raw_dir = _archive(tmp_path)
        question = probe.load_archived_question(_QUESTION_ID, latest_dir=latest_dir, raw_dir=raw_dir)

        probe.print_cost_estimate(question, question.gaps, probe.parse_grid(probe.DEFAULT_GRID))

        out = capsys.readouterr().out
        assert "Estimated cost of this run" in out
        assert "3 x 6 = 18 resolver calls" in out
        assert "Ceiling: $3.60" in out
        assert "PERSONAL OpenRouter key" in out


def _wording() -> probe.QuestionWording:
    return probe.QuestionWording(
        question_text="What will be the highest daily number of PLA aircraft ...?",
        resolution_criteria="Resolves as the highest number detected in Taiwan's de facto ADIZ.",
        fine_print="Synced with an original identical question.",
    )


@pytest.fixture
def probe_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the probe's archive and output paths at a temp dir; yields the output dir."""
    latest_dir, raw_dir = _archive(tmp_path)
    monkeypatch.setattr(probe, "ARCHIVE_LATEST_DIR", latest_dir)
    monkeypatch.setattr(probe, "ARCHIVE_RAW_DIR", raw_dir)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(probe, "OUTPUT_DIR", output_dir)
    return output_dir


@pytest.fixture
def clean_ledger() -> Iterator[None]:
    reset_role_spend()
    yield
    reset_role_spend()


class TestRefusesWithoutTheFlag:
    """A bare invocation must cost nothing: no Metaculus read, no LLM built, exit 2 after the estimate."""

    def test_it_exits_two_before_the_metaculus_read_and_before_any_llm(
        self, probe_archive: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def _no_read(post_id: int) -> probe.QuestionWording:
            pytest.fail(f"Metaculus was read for post {post_id}")

        def _no_llm(model_slug: str, **kwargs: Any) -> None:
            pytest.fail(f"an LLM was built for {model_slug} with {kwargs}")

        monkeypatch.setattr(probe, "fetch_question_wording", _no_read)
        monkeypatch.setattr(probe, "build_native_search_llm", _no_llm)
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)

        with pytest.raises(SystemExit) as exc:
            probe.main(["--question", str(_QUESTION_ID)])

        assert exc.value.code == 2
        out = capsys.readouterr().out
        assert "Estimated cost of this run" in out
        assert "18 resolver calls" in out
        assert "--i-accept-spend" in out
        # The key routing is forced only on the spending path, so the refusal leaves the environment alone.
        assert DONATED_OPENROUTER_KEY_ENABLED_ENV not in os.environ
        assert not probe_archive.exists(), "the refusal path wrote an output"


class _FakeResolverLlm:
    """Stands in for the GeneralLlm ``build_native_search_llm`` returns.

    ``invoke`` books the call on the production role ledger exactly as litellm's success
    callback would, so the probe's join by role is exercised against the real ledger. The
    current model books ten times the cost of the candidate, so the ratios are checkable.
    """

    def __init__(self, model_slug: str, role: str, search_context_size: str, calls: list[dict[str, Any]]) -> None:
        self.model = f"openrouter/{model_slug}"
        self._role = role
        self._size = search_context_size
        self._calls = calls

    async def invoke(self, prompt: str) -> str:
        await asyncio.sleep(0)  # a real yield point, so the fake schedules like the completion it replaces
        self._calls.append({"role": self._role, "prompt": prompt})
        if self._size == "low" and "luna" in self._role:
            raise RuntimeError("provider returned nothing")
        cost = 0.10 if self._role.split(":")[2] == "current" else 0.01
        record_llm_call_spend(
            self._role,
            "personal",
            cost_usd=cost,
            byok_upstream_usd=None,
            tokens=TokenCounts(prompt=80_000, completion=600, cached=0, reasoning=200),
        )
        return f"answer from {self._role}"


class TestAcceptedPath:
    """With the flag, exactly gaps x cells calls through the production builder, then both files."""

    @pytest.fixture
    def run(
        self, probe_archive: Path, clean_ledger: None, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
        del clean_ledger
        builds: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []

        def _fake_builder(model_slug: str, **kwargs: Any) -> _FakeResolverLlm:
            builds.append({"model_slug": model_slug, **kwargs})
            return _FakeResolverLlm(model_slug, kwargs["role"], kwargs["search_context_size"], calls)

        async def _no_drain() -> None:
            await asyncio.sleep(0)  # a real yield point, so the stub schedules like the drain it replaces

        monkeypatch.setattr(probe, "fetch_question_wording", lambda post_id: _wording())
        monkeypatch.setattr(probe, "build_native_search_llm", _fake_builder)
        monkeypatch.setattr(probe, "install_role_spend_tracker", lambda: None)
        monkeypatch.setattr(probe, "drain_litellm_callbacks", _no_drain)
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)

        probe.main(
            [
                "--question",
                str(_QUESTION_ID),
                "--i-accept-spend",
                "--grid",
                "current:high",
                "current:low",
                "luna:high",
                "luna:low",
            ]
        )
        return builds, calls, probe_archive

    def test_gaps_times_cells_calls_through_the_production_builder(
        self, run: tuple[list[dict[str, Any]], list[dict[str, Any]], Path]
    ) -> None:
        builds, calls, _ = run

        assert len(builds) == len(calls) == 3 * 4, "three archived gaps at four cells"
        assert {build["model_slug"] for build in builds} == {GAP_FILL_RESOLVER_MODEL, probe.CANDIDATE_MODEL}
        assert {build["search_context_size"] for build in builds} == {"high", "low"}
        assert {build["reasoning_effort"] for build in builds} == {GAP_FILL_RESOLVER_REASONING_EFFORT}
        assert len({build["role"] for build in builds}) == 12, "one role per call, so the ledger is per call"
        assert all(build["role"].startswith("probe:gap") for build in builds)

    def test_the_prompt_carries_the_gap_the_query_and_the_criteria(
        self, run: tuple[list[dict[str, Any]], list[dict[str, Any]], Path]
    ) -> None:
        _, calls, _ = run
        prompt = next(call["prompt"] for call in calls if call["role"].startswith("probe:gap2:"))

        assert "Clarify whether the question resolves on the headline total or the ADIZ subset." in prompt
        assert "Resolves as the highest number detected in Taiwan's de facto ADIZ." in prompt
        assert "Synced with an original identical question." in prompt

    def test_the_donated_key_is_forced_off_before_the_calls(
        self, run: tuple[list[dict[str, Any]], list[dict[str, Any]], Path]
    ) -> None:
        del run
        assert os.environ[DONATED_OPENROUTER_KEY_ENABLED_ENV] == "false"

    def test_both_files_are_written_with_the_ledger_join_and_the_ratios(
        self, run: tuple[list[dict[str, Any]], list[dict[str, Any]], Path]
    ) -> None:
        _, _, output_dir = run
        (json_path,) = output_dir.glob("gap_fill_resolver_probe_44267_*.json")
        (md_path,) = output_dir.glob("gap_fill_resolver_probe_44267_*.md")
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        assert payload["key"] == "personal"
        assert payload["wording"]["resolution_criteria"].startswith("Resolves as the highest number")
        by_role = {result["role"]: result for result in payload["results"]}
        current_high = by_role["probe:gap1:current:high"]
        assert current_high["cost_usd"] == pytest.approx(0.10)
        assert current_high["tokens"] == {"prompt": 80_000, "completion": 600, "cached": 0, "reasoning": 200}
        assert current_high["calls"] == 1
        assert current_high["byok_calls"] == 0, "a BYOK call would mean the donated key served the probe"
        assert current_high["answer"] == "answer from probe:gap1:current:high"
        luna_low = by_role["probe:gap1:luna:low"]
        assert luna_low["answer"] is None
        assert "RuntimeError" in luna_low["error"]
        assert luna_low["cost_usd"] is None, "a failed call never reached the ledger, so it is uncosted, not zero"
        # Totals over the three gaps: current 0.30 and luna 0.03 at high, current 0.30 at low, luna failed at low.
        assert payload["ratios"]["current low/high"] == pytest.approx(1.0)
        assert payload["ratios"]["luna/current at high"] == pytest.approx(0.1)
        assert payload["ratios"]["luna low/high"] is None
        assert payload["ratios"]["luna/current at low"] is None

        markdown = md_path.read_text(encoding="utf-8")
        assert "# Gap-fill resolver probe: question 44267 (post 44256)" in markdown
        assert (
            "| current:high | openai/gpt-6-sol | high | low | 3 | 0 | $0.3000 | $0.1000 | 240000 | 1800 | 0 | 600 |"
            in markdown
        )
        assert "| luna:low | openai/gpt-6-luna | low | low | 0 | 3 | n/a | n/a | 0 | 0 | 0 | 0 |" in markdown
        assert "- luna/current at high: 0.10x" in markdown
        assert "- luna low/high: n/a" in markdown
        assert "## Gap 2: Clarify whether the question resolves on the headline total or the ADIZ subset." in markdown
        assert "### Archived production answer (run 28674047443)" in markdown
        assert "The resolving metric is the headline total" in markdown
        assert "ERROR: RuntimeError('provider returned nothing')" in markdown


class TestCostRatios:
    def test_an_effort_cell_is_compared_to_its_own_high_and_to_production_at_its_size(self) -> None:
        cells = probe.parse_grid(["current:high", "current:low", "luna:high:medium", "luna:medium:medium"])
        costs = {"current:high": 0.20, "current:low": 0.15, "luna:high:medium": 0.05, "luna:medium:medium": 0.04}
        results = [
            probe.CellResult(
                gap_index=1,
                cell=cell,
                role=probe.probe_role(1, cell),
                answer="ok",
                error=None,
                wall_s=1.0,
                calls=1,
                cost_usd=costs[cell.label],
            )
            for cell in cells
        ]

        ratios = dict(probe.cost_ratios(probe.summarize_cells(cells, results)))

        assert ratios["current low/high"] == pytest.approx(0.75)
        assert ratios["luna@medium medium/high"] == pytest.approx(0.8)
        assert ratios["luna@medium/current at high"] == pytest.approx(0.25)
        assert ratios["luna@medium/current at medium"] is None, "no production cell at medium in this grid"
        assert "luna@medium/current at low" not in ratios, "no luna@medium cell at low, so no ratio is offered for it"


class TestRenderMarkdown:
    """The write-up is what the operator reads, so its shape is pinned on synthetic results."""

    def test_one_section_per_gap_with_every_cell_under_it(self) -> None:
        cells = probe.parse_grid(["current:high", "luna:medium"])
        gaps = probe.parse_archived_gaps(_RESEARCH_TEXT)
        question = probe.ArchivedQuestion(
            _QUESTION_ID, _POST_ID, _RUN_ID, "2026-07-03T17:07:13+00:00", "title", tuple(gaps)
        )
        results = [
            probe.CellResult(
                gap_index=gap.index,
                cell=cell,
                role=probe.probe_role(gap.index, cell),
                answer=f"{cell.label} says 26" if cell.model_alias == "luna" else f"{cell.label} says 30",
                error=None,
                wall_s=12.5,
                calls=1,
                cost_usd=0.19 if cell.model_alias == "current" else 0.02,
                tokens=TokenCounts(prompt=85_000, completion=700, cached=1_000, reasoning=300),
            )
            for gap in gaps
            for cell in cells
        ]

        run = probe.ProbeRun(
            question, _wording(), tuple(cells), tuple(results), datetime(2026, 9, 9, 23, 0, tzinfo=UTC)
        )

        markdown = probe.render_markdown(run)

        gap_headings = [line for line in markdown.splitlines() if line.startswith("## Gap ")]
        assert gap_headings == [f"## Gap {gap.index}: {gap.gap}" for gap in gaps]
        cell_headings = [
            line for line in markdown.splitlines() if line.startswith(("### current:high", "### luna:medium"))
        ]
        assert len(cell_headings) == len(gaps) * len(cells)
        assert (
            "### luna:medium (openai/gpt-6-luna, context medium, effort low): $0.0200, 85000 prompt / 700 completion tokens (1000 cached, 300 reasoning), 12.5 s, 1 billed call(s)"
            in markdown
        )
        assert "luna:medium says 26" in markdown
        assert "- luna/current at medium: n/a" in markdown, "no current:medium cell in this grid"
        assert "- luna medium/high: n/a" in markdown, "no luna:high cell in this grid"
        assert "3 gap(s) x 2 cell(s) = 6 resolver calls" in markdown


class TestSearchContextSizeOverride:
    """``build_native_search_llm`` takes the per-call context size the probe needs, defaulting to the constant."""

    def _web_search_options(self, **overrides: Any) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class _CaptureLlm:
            def __init__(self, model: str, **kwargs: Any) -> None:
                captured.update(kwargs)
                self.model = model

        with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", _CaptureLlm):
            build_native_search_llm("openai/gpt-5.6-terra", reasoning_effort="low", **overrides)
        return captured["web_search_options"]

    def test_an_explicit_size_wins(self) -> None:
        assert self._web_search_options(search_context_size="low") == {"search_context_size": "low"}

    def test_none_keeps_the_production_constant(self) -> None:
        assert self._web_search_options() == {"search_context_size": NATIVE_SEARCH_CONTEXT_SIZE}
