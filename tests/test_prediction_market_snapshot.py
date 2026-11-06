"""The prediction-market pipeline end to end, through the seam's own `fetch_market_snapshot`.

The four stages wired together against a fake aiohttp session and both LLM stages stubbed: what
the pool sees, what the ranker's picks do to the render, every way the ranking can fail open, the
`as_of` leakage filter and its cache key, the raw-vs-stripped query split, and the ranking
telemetry line. The last class here pins the port's biggest silent-failure risk — that retrieval
width and the render cap are independent numbers.

Fakes, payload fixtures and the `handlers()` baseline live in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import litellm.exceptions
import pytest

from metaculus_bot.prompts import binary_prompt
from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.market_retrieval import generation
from metaculus_bot.research.market_retrieval.queries import dedupe_queries, strip_dates_and_numbers
from metaculus_bot.research.market_retrieval.ranking import (
    DEGRADED_RANKING_MARKER,
    RENDER_BUDGET,
    cap_stale_top_tier,
)
from metaculus_bot.research.prediction_market import format_snapshot_for_research
from metaculus_bot.research.section_format import provider_header
from scripts.telemetry.markers import MARKER_SPECS
from tests import market_retrieval_fakes as _fakes
from tests.market_retrieval_fakes import AUTHOR_JSON as _AUTHOR_JSON
from tests.market_retrieval_fakes import CANDIDATE_LINE_RE as _CANDIDATE_LINE_RE
from tests.market_retrieval_fakes import KALSHI_EVENTS_URL as _KALSHI_EVENTS_URL
from tests.market_retrieval_fakes import MANIFOLD_SEARCH_URL as _MANIFOLD_SEARCH_URL
from tests.market_retrieval_fakes import OFF_TOPIC_KALSHI_EVENT as _OFF_TOPIC_KALSHI_EVENT
from tests.market_retrieval_fakes import POLY_URL as _POLY_URL
from tests.market_retrieval_fakes import PREDICTIT_URL as _PREDICTIT_URL
from tests.market_retrieval_fakes import RANKER_CUE as _RANKER_CUE
from tests.market_retrieval_fakes import FakeResponse, FakeSession
from tests.market_retrieval_fakes import fetch_snapshot as _fetch
from tests.market_retrieval_fakes import handlers as _handlers
from tests.market_retrieval_fakes import market_llm as _market_llm
from tests.market_retrieval_fakes import market_row as _row
from tests.market_retrieval_fakes import rank_one_per_venue as _rank_one_per_venue

# Bound by assignment rather than imported — see the note in tests/test_prediction_market_transport.py.
reset_provider_caches = _fakes.reset_provider_caches
mock_question = _fakes.mock_question
polymarket_payload = _fakes.polymarket_payload
manifold_payload = _fakes.manifold_payload
kalshi_events_payload = _fakes.kalshi_events_payload
predictit_payload = _fakes.predictit_payload

RETRIEVAL_WIDTH_KALSHI = generation.RETRIEVAL_WIDTH["kalshi"]


def _rank_first_as(*, tier: str, why: str) -> Callable[[str], str]:
    """`rank_one_per_venue` with the tier and phrase chosen by the caller.

    Indices are read back out of the prompt for the reason the shared helper documents: a
    hard-coded array would silently render fewer rows than the test means, and a test that then
    asserts on rows passes for the wrong reason.
    """

    def _rank(prompt: str) -> str:
        picks = json.loads(_rank_one_per_venue(prompt))
        return json.dumps([{**pick, "tier": tier, "why": why} for pick in picks])

    return _rank


def _tier_capped_fields(line: str) -> dict[str, Any]:
    """One MARKET_TIER_CAPPED line's fields, read the way the telemetry harvester reads them.

    Through the marker's own ``MarkerSpec`` regex rather than substring checks, so the field
    ORDER is pinned next to the emitter: the regex spells the sequence out literally, and a
    reordered emitter would harvest zero records. That failure is invisible from the parser
    side for this marker in particular — it fires on none of the 102 archived snapshots, so an
    empty harvest is indistinguishable from a broken regex.
    """
    spec = next(spec for spec in MARKER_SPECS if spec.name == "market_tier_capped")
    match = spec.regex.search(line)
    assert match is not None, f"line does not match the market_tier_capped MarkerSpec regex: {line!r}"
    return match.groupdict()


def _table_row_by_platform(rendered: str, platform: str) -> dict[str, str]:
    """One rendered parent row as a header-keyed dict.

    By header NAME rather than by position, matching the rendering suite: a column addition would
    otherwise shift every assertion onto its neighbour.
    """
    lines = [line for line in rendered.split("\n") if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("| ").split("|")]
    rows = [dict(zip(header, [cell.strip() for cell in line.strip("| ").split("|")], strict=False)) for line in lines]
    return next(row for row in rows if row["platform"] == platform)


class TestFetchMarketSnapshot:
    @pytest.mark.asyncio
    async def test_the_pool_sees_all_four_venues_and_the_render_is_the_models_order(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload, predictit_payload
    ):
        """Generation is venue-complete; SELECTION is the ranker's, and the render is its order
        verbatim.

        This replaces an older assertion that all four platforms appear in the output, which was
        the venue-fairness assumption in test form — the exact defect this pipeline fixes (a
        round-robin fairness pass evicted 43 of 58 wanted rows). A venue can now legitimately be
        absent from the render, so what is asserted is that the POOL saw every venue and that
        the rows come back in the order the model asked for.
        """
        handlers = _handlers(
            **{
                _POLY_URL: FakeResponse(200, polymarket_payload),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _PREDICTIT_URL: FakeResponse(200, predictit_payload),
            }
        )
        captured: list[str] = []

        def _rank_reversed(prompt: str) -> str:
            """Pick one candidate per venue, then INVERT the order. A renderer that sorted by
            venue, by index, or by anything else would produce the pool's order instead."""
            captured.append(prompt)
            picks = json.loads(_rank_one_per_venue(prompt))
            return json.dumps(list(reversed(picks)))

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_reversed)

        assert captured, "the ranker was never called"
        pool_venues = {match.group(2) for match in _CANDIDATE_LINE_RE.finditer(captured[0])}
        assert pool_venues == {"kalshi", "polymarket", "manifold", "predictit"}

        expected = [pick["i"] for pick in reversed(json.loads(_rank_one_per_venue(captured[0])))]
        assert [row.rank for row in snapshot.matches] == list(range(len(snapshot.matches)))
        assert [row.platform for row in snapshot.matches] == [
            _CANDIDATE_LINE_RE.findall(captured[0])[index][1] for index in expected
        ]
        assert all(row.relation_tier == "same_quantity_other_cut" for row in snapshot.matches)
        assert all(row.relevance_label for row in snapshot.matches)

    @pytest.mark.asyncio
    async def test_a_multi_outcome_markets_prices_survive_the_whole_pipeline_into_the_render(
        self, mock_question, predictit_payload
    ):
        """The plumbing test for the sub-row expansion: a venue parser builds the outcomes, the pool
        carries them, the ranker never sees them, and the renderer emits one row each.

        Worth an end-to-end test rather than only unit coverage because the outcomes cross four
        stages and the ranker layer deliberately ignores them — a stage that rebuilt rows from the
        ranker's own view, or a pool truncation that dropped the field, would leave every
        multi-outcome market priceless again with the venue unit tests still green. PredictIt is the
        venue that proves it, since its market level has never quoted a price at all.
        """
        handlers = _handlers(**{_PREDICTIT_URL: FakeResponse(200, predictit_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        rendered = format_snapshot_for_research(snapshot)
        predictit_rows = [row for row in snapshot.matches if row.platform == "predictit"]
        assert predictit_rows, "the ranker must have picked the PredictIt row"
        assert predictit_rows[0].implied_prob_yes is None, "a ballot has no single probability"
        assert [(child.title, child.implied_prob_yes) for child in predictit_rows[0].children] == [("Yes", 0.58)]
        assert "| ↳|Yes|0.58|" in rendered
        assert _RANKER_CUE not in rendered

    @pytest.mark.asyncio
    async def test_a_market_closed_long_before_the_question_opened_is_capped_and_disclosed(
        self, mock_question, manifold_payload, caplog: pytest.LogCaptureFixture
    ):
        """The staleness pair end to end: the deterministic tier cap and the row-level disclosure.

        The Manifold fixture is the measured shape — it closes 2026-06-22 and carries
        `isResolved: false`, exactly like q45163's rank-0 row, which closed five months before the
        forecast and still rendered `status=open` because Manifold's close dates are soft. Worth a
        pipeline test rather than only unit coverage because the two halves are computed in different
        stages off different clocks: the cap reads the QUESTION's open time in the ranking stage, and
        the disclosure reads the SNAPSHOT's own `forecast_time` in the renderer. A wiring that read
        one of them off the wrong object would leave both unit suites green.

        The `(Nd ago)` figure is matched as a pattern, not a number: `forecast_time` is the fetch
        instant on the provider path, so the age grows by one every day this fixture ages.
        """
        mock_question.open_time = datetime(2026, 12, 1, tzinfo=UTC)
        handlers = _handlers(**{_MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload)})

        with caplog.at_level(logging.INFO):
            snapshot = await _fetch(
                mock_question, handlers, ranking=_rank_first_as(tier="same_quantity_same_date", why="same window")
            )

        row = next(match for match in snapshot.matches if match.platform == "manifold")
        assert row.relation_tier == "same_quantity_other_cut", "the top tier is refused, one rung only"
        assert row.tier_cap_note == "demoted from same-date: closed 162d before the question opened"
        assert row.relevance_label == "same window", "the ranker's own phrase is left verbatim"

        rendered = format_snapshot_for_research(snapshot)
        cells = _table_row_by_platform(rendered, "manifold")
        assert cells["relation"] == "same_quantity_other_cut"
        assert cells["why"].startswith("demoted from same-date: closed 162d before the question opened")
        # The forecaster-facing half of the note: the legend has to define the shape the cell holds.
        assert "`demoted from same-date:`" in rendered
        assert re.fullmatch(r"2026-06-22 \(\d+d ago\)", cells["close"]), cells["close"]
        assert cells["status"] == "open", "the soft-close row still reads open — the cue is the close cell"

        capped = [message for message in caplog.messages if message.startswith("MARKET_TIER_CAPPED:")]
        assert len(capped) == 1
        assert row.rank is not None, "the ranker must have stamped a rank for the capped= entry to name"
        assert _tier_capped_fields(capped[0]) == {
            "question": str(mock_question.id_of_question),
            "rows": "1",
            "capped": f"manifold@{row.rank}",
        }

    @pytest.mark.asyncio
    async def test_a_market_closing_after_the_question_opened_keeps_its_grade_and_logs_nothing(
        self, mock_question, manifold_payload, caplog: pytest.LogCaptureFixture
    ):
        """The other side of the cap, and the reason its log line is conditional: the overwhelmingly
        common case must leave the ranker's grade alone and say nothing at all, so a
        `MARKET_TIER_CAPPED` line in a run log means something happened."""
        mock_question.open_time = datetime(2026, 5, 1, tzinfo=UTC)
        handlers = _handlers(**{_MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload)})

        with caplog.at_level(logging.INFO):
            snapshot = await _fetch(
                mock_question, handlers, ranking=_rank_first_as(tier="same_quantity_same_date", why="same window")
            )

        row = next(match for match in snapshot.matches if match.platform == "manifold")
        assert row.relation_tier == "same_quantity_same_date"
        assert row.tier_cap_note == ""
        assert not [message for message in caplog.messages if message.startswith("MARKET_TIER_CAPPED:")]

    def test_two_capped_rows_are_listed_in_one_spaceless_comma_joined_field(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`capped=` is a `venue@rank` list the harvester takes whole with `\\S+`, so a space
        anywhere in the join would truncate the field and silently drop every entry past the
        first. Only a multi-row cap can show that, and the fixture pipeline caps at most one row
        (the cap fires on nothing in the archived corpus at all), so the real staleness pass and
        the real emitter are driven directly over two tier-1 rows that closed before the question
        opened."""
        stale_rows = [
            replace(
                _row("Manifold market", platform="manifold", tier="same_quantity_same_date"),
                rank=0,
                close_time=datetime(2026, 6, 22, tzinfo=UTC),
            ),
            replace(
                _row("Kalshi market", platform="kalshi", tier="same_quantity_same_date"),
                rank=2,
                close_time=datetime(2026, 5, 1, tzinfo=UTC),
            ),
        ]
        capped_rows = cap_stale_top_tier(stale_rows, question_open_time=datetime(2026, 12, 1, tzinfo=UTC))
        assert all(row.tier_cap_note for row in capped_rows), "both rows must actually be capped"

        with caplog.at_level(logging.INFO):
            pmp._log_tier_caps(45163, capped_rows)

        (line,) = [message for message in caplog.messages if message.startswith("MARKET_TIER_CAPPED:")]
        assert _tier_capped_fields(line) == {
            "question": "45163",
            "rows": "2",
            "capped": "manifold@0,kalshi@2",
        }

    @pytest.mark.asyncio
    async def test_an_empty_ranking_is_a_valid_answer_and_not_a_degradation(self, mock_question, kalshi_events_payload):
        """`[]` means "nothing here bears on the question", which is the whole adaptive-width
        mechanism. Conflating it with a failure would delete that mechanism AND redden CI on a
        correct answer. The forecaster is TOLD about the deliberate empty answer rather than
        shown a vanished section (the q45200 outage-lookalike), and the notice quotes the pool
        the ranker reviewed."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking="[]")

        assert snapshot.matches == []
        assert snapshot.sources["ranking"] == "ok(0)"
        assert pmp.prediction_market_source_losses() == 0
        assert snapshot.pool_size > 0
        rendered = format_snapshot_for_research(snapshot)
        assert f"No sufficiently relevant market among {snapshot.pool_size} candidates" in rendered
        assert "not a provider outage" in rendered

    @pytest.mark.asyncio
    async def test_an_unreadable_ranking_fails_open_to_retrieval_order(self, mock_question, kalshi_events_payload):
        """Output that cannot be read as a JSON array at all renders the pool-order top rows —
        literally the head of what the model was shown, so a fail-open is a truncation of its own
        input rather than a different pipeline. It is marked as such, because the preamble and
        legend both claim evidential order and a silently-wrong ordering claim is worse than a
        visibly degraded one."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking="I could not decide.")

        assert snapshot.matches, "a fail-open must still render the deterministic slate"
        assert len(snapshot.matches) <= RENDER_BUDGET
        assert all(row.relation_tier == "" for row in snapshot.matches)
        assert snapshot.sources["ranking"].startswith("error(")
        assert pmp.prediction_market_source_losses() == 1
        assert DEGRADED_RANKING_MARKER in format_snapshot_for_research(snapshot)

    @pytest.mark.asyncio
    async def test_a_shape_regression_fails_open_and_is_named_in_the_telemetry(
        self, mock_question, kalshi_events_payload, caplog: pytest.LogCaptureFixture
    ):
        """A well-formed array of objects with a RENAMED index key: readable JSON, no usable row.

        This used to render as `ranking: ok(0)` with zero rows — indistinguishable from the
        model's considered "nothing here bears on the question", and since the deliberate-empty
        sentence shipped it published that sentence's affirmative claim ("prediction markets were
        retrieved and reviewed… none was judged to bear on it") over output whose shape we had
        failed to parse. It now fails open like any other unreadable ranking, and the WARN names
        `reason=shape_regression` so the archive can tell a broken parser contract from a model
        that emitted prose.
        """
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        with caplog.at_level(logging.WARNING):
            snapshot = await _fetch(mock_question, handlers, ranking='[{"index": 0}, {"index": 1}]')

        assert snapshot.matches, "a fail-open must still render the deterministic slate"
        assert all(row.relation_tier == "" for row in snapshot.matches)
        assert snapshot.sources["ranking"] == "error(RankingShapeRegression)"
        assert pmp.prediction_market_source_losses() == 1
        rendered = format_snapshot_for_research(snapshot)
        assert DEGRADED_RANKING_MARKER in rendered
        assert "No sufficiently relevant market" not in rendered
        degraded = [m for m in caplog.messages if m.startswith("MARKET_RANKING_DEGRADED:")]
        assert len(degraded) == 1
        assert "reason=shape_regression" in degraded[0]
        assert f"pool={snapshot.pool_size}" in degraded[0]

    @pytest.mark.asyncio
    async def test_an_unreadable_ranking_is_labelled_unreadable_not_a_shape_regression(
        self, mock_question, kalshi_events_payload, caplog: pytest.LogCaptureFixture
    ):
        """The sibling of the case above, and the reason the WARN carries a `reason=` at all: one
        of these two says the model answered badly, the other says our own prompt/parser contract
        broke, and only the second is a defect to go fix."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        with caplog.at_level(logging.WARNING):
            await _fetch(mock_question, handlers, ranking="I could not decide.")

        degraded = [m for m in caplog.messages if m.startswith("MARKET_RANKING_DEGRADED:")]
        assert len(degraded) == 1
        assert "reason=unreadable" in degraded[0]

    @pytest.mark.asyncio
    async def test_a_deliberate_empty_ranking_emits_no_degraded_marker(
        self, mock_question, kalshi_events_payload, caplog: pytest.LogCaptureFixture
    ):
        """The control for the marker: `[]` is the mechanism working, so it must not appear in a
        channel an operator is meant to read as "the ranker is broken"."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        with caplog.at_level(logging.WARNING):
            snapshot = await _fetch(mock_question, handlers, ranking="[]")

        assert snapshot.sources["ranking"] == "ok(0)"
        assert not [m for m in caplog.messages if m.startswith("MARKET_RANKING_DEGRADED:")]

    @pytest.mark.parametrize(
        "exception_factory",
        [
            pytest.param(
                lambda: litellm.exceptions.Timeout(message="stalled", model="m", llm_provider="openrouter"),
                id="timeout",
            ),
            pytest.param(
                lambda: litellm.exceptions.RateLimitError(message="429", model="m", llm_provider="openrouter"),
                id="rate_limit",
            ),
            pytest.param(
                lambda: litellm.exceptions.APIConnectionError(message="reset", model="m", llm_provider="openrouter"),
                id="connection",
            ),
            pytest.param(
                lambda: litellm.exceptions.InternalServerError(message="500", model="m", llm_provider="openrouter"),
                id="internal_server",
            ),
            pytest.param(
                lambda: litellm.exceptions.ServiceUnavailableError(message="503", model="m", llm_provider="openrouter"),
                id="service_unavailable",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_transient_ranker_failure_degrades_one_stage_not_the_whole_snapshot(
        self, mock_question, kalshi_events_payload, exception_factory
    ):
        """A routine provider blip on the ranking call must fail OPEN, not discard the snapshot.

        Every litellm transport exception except a bare `APIError` subclasses `openai.APIError`
        and NOT `litellm.exceptions.APIError`, so a catch written against the latter let all five
        of these escape `_invoke_market_llm`, sail past `_rank_pool`'s `except RankingUnusable`
        (which wraps only the parse), and land on the snapshot-level net — returning zero rows
        with `sources={'snapshot': 'error(...)'}` where the pool-order slate was due. A
        string-only unusable-ranking test cannot catch this class, which is why these are raised
        from `invoke` at the real seam.
        """

        class RaisingLlm:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def invoke(self, prompt: str) -> str:
                if _RANKER_CUE in prompt:
                    raise exception_factory()
                return _AUTHOR_JSON

        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", RaisingLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert "snapshot" not in snapshot.sources, "a ranker blip took the whole snapshot down"
        assert snapshot.sources["ranking"].startswith("error("), snapshot.sources
        assert snapshot.matches, "the fail-open slate must still render the pool-order top rows"
        assert snapshot.sources["kalshi"].startswith("ok(")

    @pytest.mark.asyncio
    async def test_an_empty_pool_skips_the_ranking_call_entirely(self, mock_question):
        """With nothing to rank there is no LLM call to make, so the stage reports a benign
        `none` rather than a loss: an empty pool is the venues' story to tell through their own
        tokens, not a ranking failure."""
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, {"events": [], "cursor": ""}),
                _PREDICTIT_URL: FakeResponse(200, {"markets": []}),
            }
        )
        ranker_prompts: list[str] = []

        def _record(prompt: str) -> str:
            ranker_prompts.append(prompt)
            return "[]"

        snapshot = await _fetch(mock_question, handlers, ranking=_record)

        assert ranker_prompts == []
        assert snapshot.matches == []
        assert snapshot.sources["ranking"] == "none"
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_the_settlement_join_reaches_a_market_the_question_shares_no_words_with(self, mock_question):
        """The structural recall channel, end to end through the seam's own text assembly.

        The event's title shares nothing with the question; what connects them is that both name
        the same publisher. It is joined via `resolution_criteria` + `fine_print` CONCATENATED —
        the fine print is where a question often names the actual release page, so a pipeline
        that passed only the resolution criteria would miss this class entirely, which is what
        this test pins.
        """
        mock_question.title = "What will the U-3 unemployment rate be in July 2026?"
        mock_question.question_text = mock_question.title
        mock_question.resolution_criteria = "Per the official release."
        mock_question.fine_print = "Resolution uses https://data.bls.gov/timeseries/LNS14000000"

        joined_event = {
            "event_ticker": "KXJOBLESS-26JUL",
            "title": "Jobless rate, July",
            "settlement_sources": [{"name": "BLS", "url": "https://www.bls.gov/news.release/empsit.nr0.htm"}],
            "markets": [{"ticker": "KXJOBLESS-26JUL-T4", "status": "active", "close_time": "2026-08-07T00:00:00Z"}],
        }
        handlers = _handlers(
            **{_KALSHI_EVENTS_URL: FakeResponse(200, {"events": [joined_event, _OFF_TOPIC_KALSHI_EVENT], "cursor": ""})}
        )
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        # The join is the first channel, and channel order IS the pool order, so the joined event
        # leads the Kalshi block ahead of the alphabetically-earlier off-topic one.
        assert "[0] (kalshi) Jobless rate, July" in captured[0]
        assert "settles via: BLS" in captured[0]
        joined = next(row for row in snapshot.matches if row.venue_market_id == "KXJOBLESS-26JUL")
        assert joined.retrieval_channel == "settlement_join"

    @pytest.mark.asyncio
    async def test_timeout_returns_an_empty_snapshot_soft_fail(self, mock_question):
        """A per-question timeout must NOT raise: soft-fail with an empty snapshot, a loss token
        and a counter bump. With an empty `sources` map the diagnostics line renders no suffix at
        all, so a dead snapshot would be indistinguishable from one nobody asked for."""

        class StalledLlm:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(10)
                return "..."

        with patch.object(pmp, "build_llm_with_openrouter_fallback", StalledLlm):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=0.05)

        assert snapshot.matches == []
        assert snapshot.sources == {"snapshot": "error(timeout)"}
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_one_dead_venue_does_not_take_the_others_down(self, mock_question, manifold_payload, caplog):
        """Soft-fail isolation at venue granularity: a venue that raises outright is that
        venue's loss, and every other venue's rows still reach the pool."""

        def _boom(_params: dict[str, Any]) -> FakeResponse:
            raise RuntimeError("connection refused")

        handlers = _handlers(
            **{
                _POLY_URL: _boom,
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _KALSHI_EVENTS_URL: _boom,
                _PREDICTIT_URL: _boom,
            }
        )

        with caplog.at_level(logging.WARNING):
            snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert "manifold" in {row.platform for row in snapshot.matches}
        assert "polymarket" not in {row.platform for row in snapshot.matches}
        assert snapshot.sources["manifold"].startswith("ok(")
        assert "degraded (alertable)" in "\n".join(rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_the_snapshot_cache_key_includes_as_of(self, mock_question, kalshi_events_payload):
        """One entry per (qid, as_of), so a backtest at as_of=A cannot reuse a snapshot computed
        at as_of=B. The provider path passes None, which is finally what makes this cache
        hittable — the old `datetime.now(utc)` default changed the key on every call."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        as_of_a = datetime(2026, 5, 1, tzinfo=UTC)
        as_of_b = datetime(2026, 6, 1, tzinfo=UTC)

        await _fetch(mock_question, handlers, as_of=as_of_a)
        await _fetch(mock_question, handlers, as_of=as_of_b)
        await _fetch(mock_question, handlers, as_of=None)

        assert (mock_question.id_of_question, as_of_a.isoformat()) in pmp._SNAPSHOT_CACHE
        assert (mock_question.id_of_question, as_of_b.isoformat()) in pmp._SNAPSHOT_CACHE
        assert (mock_question.id_of_question, "none") in pmp._SNAPSHOT_CACHE

    def test_one_instant_spelled_three_ways_is_one_cache_key(self) -> None:
        """The key normalizes through `time_utils._as_utc`, so equal instants cannot miss.

        Naive-UTC, explicit `+00:00` and an offset-aware spelling of the SAME instant are one
        cache entry, and a genuinely different instant is a different one. Worth pinning
        because the naive-vs-aware branch used to be respelled here rather than shared with
        the copy `assemble_pool` applies to the same value; two spellings drifting apart
        would silently re-fetch (a wasted pull) or, if they drifted the other way, serve a
        snapshot computed at the wrong `as_of`.
        """
        naive = datetime(2026, 8, 4, 12, 0, 0)
        utc = datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC)
        offset = datetime(2026, 8, 4, 5, 0, 0, tzinfo=timezone(timedelta(hours=-7)))

        keys = {pmp._as_of_cache_key(moment) for moment in (naive, utc, offset)}
        assert len(keys) == 1, f"one instant must yield one key; got {keys}"
        assert pmp._as_of_cache_key(None) == "none"
        assert pmp._as_of_cache_key(utc) != pmp._as_of_cache_key(datetime(2026, 8, 4, 13, tzinfo=UTC))

    @pytest.mark.asyncio
    async def test_an_explicit_as_of_filters_the_pool_before_the_ranker_sees_it(self, mock_question):
        """The leakage filter survives for explicit callers, and it filters the POOL.

        Filtering the pool rather than the rendered rows is what keeps the ranker from reading a
        market that already resolved, and keeps the candidate indices the model is given
        identical to the ones its picks are applied against. The provider path passes None — that
        decision is asserted separately in TestProviderFactory.
        """
        closed_event = {
            "event_ticker": "KXSTAR-PAST",
            "title": "Will SpaceX Starship reach orbit in 2026?",
            "markets": [
                {
                    "ticker": "KXSTAR-PAST-YES",
                    "rules_primary": "orbital velocity",
                    "status": "settled",
                    "close_time": "2026-04-01T00:00:00Z",
                }
            ],
        }
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, {"events": [closed_event], "cursor": ""})})
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture, as_of=datetime(2026, 5, 1, tzinfo=UTC))

        assert "kalshi" not in {row.platform for row in snapshot.matches}
        assert not any("KXSTAR-PAST" in prompt or "reach orbit in 2026" in prompt for prompt in captured)

    @pytest.mark.asyncio
    async def test_the_conjunctive_venues_get_stripped_queries_and_the_pool_gets_raw_ones(
        self, mock_question, kalshi_events_payload
    ):
        """The digit-strip split, pinned end to end, because both inversions are silent in prod.

        Both halves are load-bearing in opposite directions. The RAW set is what the enumerable
        catalogues are fuzzy-scored against, where a year is real signal against a corpus of dated
        market titles — stripping it measurably reorders Kalshi's top rows (48.7/45.0/43.2 →
        44.7/44.3/44.2 over the live fixture). The STRIPPED set is what Manifold's `term` needs,
        because it is a strict conjunction that one date token no market's text carries zeroes
        outright — the cliff that hid the Manifold breakage for 17+ days.

        Neither inversion raises, logs, or changes a source token, and the two call sites sit ~20
        lines apart with near-identical names, so an accidental swap is the realistic refactor.
        Both directions passed the full suite before this test existed; the only adjacent guard is
        skip-gated out of CI. The expectation is DERIVED from the observed raw set rather than
        probed for absence of digits, so a partial strip or a wrong dedupe fails too.
        """
        dispatched: dict[str, list[str]] = {"manifold": [], "polymarket": []}

        def _recorder(venue: str, param: str, payload: Any):
            def _handler(params: dict[str, Any]) -> FakeResponse:
                dispatched[venue].append(params.get(param) or "")
                return FakeResponse(200, payload)

            return _handler

        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _MANIFOLD_SEARCH_URL: _recorder("manifold", "term", []),
                _POLY_URL: _recorder("polymarket", "q", {"events": [], "markets": []}),
            }
        )
        pool_queries: list[list[str]] = []
        real_build_pool = generation.build_pool

        async def _spy(*, queries, **kwargs: Any):
            pool_queries.append(list(queries))
            return await real_build_pool(queries=queries, **kwargs)

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=_rank_one_per_venue)),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
            patch.object(pmp.generation, "build_pool", _spy),
        ):
            await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert len(pool_queries) == 1, pool_queries
        raw = pool_queries[0]
        assert any(any(char.isdigit() for char in query) for query in raw), (
            "pool assembly must receive the RAW query set — the year is what the catalogues score on"
        )
        expected_stripped = dedupe_queries([strip_dates_and_numbers(query) for query in raw])
        for venue, terms in dispatched.items():
            assert terms == expected_stripped, venue

    @pytest.mark.asyncio
    async def test_the_ranking_telemetry_names_every_rendered_rows_pool_index(
        self, mock_question, kalshi_events_payload, caplog
    ):
        """One INFO line per question, carrying `(venue, pool_index, rank)` per rendered row.

        The pool index is the point: it is the post-ship instrument for the two questions this
        port deliberately left open — whether ranker attention decays down a ~400-candidate
        prompt, and whether Manifold detail enrichment changes which rows get picked. Both then
        answer themselves from prod logs instead of another bake-off.

        So the expectation is the EXACT index string, derived from the prompt the ranker was
        actually shown rather than pattern-matched as `\\d+`: an index that is merely well-formed
        answers neither question. `_pool_positions` recovers the index by the venue-native id
        after `apply_picks` has already copied the rows, and every way that recovery can go
        wrong — keying on the title so two same-titled rows collide, or missing entirely and
        yielding the `-1` sentinel — produces a well-formed line that a lax pattern accepts.
        """
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        prompts: list[str] = []

        def _capture(prompt: str) -> str:
            prompts.append(prompt)
            return _rank_one_per_venue(prompt)

        with caplog.at_level(logging.INFO):
            snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        lines = [rec.getMessage() for rec in caplog.records if "MARKET_RANKING:" in rec.getMessage()]
        assert len(lines) == 1, lines
        line = lines[0]
        assert f"question={mock_question.id_of_question}" in line
        assert "outcome=ranked" in line
        assert f"rows={len(snapshot.matches)}" in line
        assert re.search(r"pool=[1-9]\d*", line), line
        assert re.search(r"prompt_chars=[1-9]\d*", line), line

        # The prompt's candidate indices ARE the pool indices, and the stub picks the first
        # candidate of each venue block in block order, which is the rank order `apply_picks`
        # stamps. So the whole `rendered=` field is predictable from the prompt alone.
        first_of: dict[str, int] = {}
        for match in _CANDIDATE_LINE_RE.finditer(prompts[0]):
            first_of.setdefault(match.group(2), int(match.group(1)))
        expected = ",".join(f"{venue}:{index}@{rank}" for rank, (venue, index) in enumerate(first_of.items()))

        assert f"rendered={expected}" in line, line
        assert len(first_of) > 1
        assert any(index > 0 for index in first_of.values()), (
            f"the fixture must span several venue blocks so a nonzero index is under test; got {first_of}"
        )

    def test_the_pool_index_recovery_falls_back_to_the_title_then_to_a_minus_one_sentinel(self):
        """The recovery's two edge cases, driven directly because no live payload can reach them.

        `_pool_positions` recovers an index by venue-native id, falling back to the title for a
        row a venue shipped without one. Every venue in the fixtures ships an id, so the fallback
        and the `-1` miss are unreachable end to end — and a silent `-1` in prod would read as a
        real index to anyone eyeballing the line, so both halves need pinning here.
        """
        in_pool = _row("In the pool", platform="kalshi")
        in_pool.venue_market_id = "KX-1"
        id_less = _row("No id anywhere", platform="manifold")
        pool = generation.PoolResult(candidates=(in_pool, id_less))

        orphan = _row("Never assembled", platform="polymarket")
        orphan.venue_market_id = "not-in-pool"
        positions = pmp._pool_positions(pool, [in_pool, id_less, orphan])

        assert [index for index, _ in positions] == [0, 1, -1]


class TestRetrievalWidthIsNotTheRenderCap:
    """The port's biggest silent-failure risk, pinned end to end.

    In the old design the render cap WAS the retrieval width (`top_k = max_matches_per_platform
    + 2`), so a ranked pipeline that inherited those call sites would hand the ranker a
    5-candidate pool and measure nothing while looking finished. The pool has to be wide, and
    the render cap has to be independent of it.
    """

    @pytest.mark.asyncio
    async def test_a_large_catalogue_reaches_the_ranker_at_full_width(self, mock_question):
        big = {
            "events": [
                {
                    "event_ticker": f"KX{i:04d}",
                    "title": f"Will SpaceX Starship reach orbit, threshold {i}?",
                    "markets": [{"ticker": f"KX{i:04d}-Y", "status": "active"}],
                }
                for i in range(RETRIEVAL_WIDTH_KALSHI + 50)
            ],
            "cursor": "",
        }
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, big)})
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        kalshi_lines = [line for line in captured[0].splitlines() if line.startswith("[") and "(kalshi)" in line]
        assert len(kalshi_lines) == RETRIEVAL_WIDTH_KALSHI
        # And the RENDER stays the ranker's choice, bounded by the budget rather than by the width.
        assert len(snapshot.matches) <= RENDER_BUDGET


class TestAFamilyReachesTheForecasterWhole:
    """The pipeline-visible half of the 2026-08-25 render change, end to end into a forecaster prompt.

    Every other test in this area stops at one seam: the venue parsers pin what a strike ladder parses
    to, the renderer tests pin the table, the formatter tests pin the marker. None of them shows that a
    real Kalshi ladder survives catalogue parse -> pool -> ranker -> render -> the string a forecaster
    is actually handed, and that is exactly where q45189 was lost: every stage was individually correct
    and the model still saw one bracket of a twelve-bracket distribution.

    The fixture is that shape — a diesel-price bracket ladder with more outcomes than the render's full-row
    allowance, one strike carrying an empty book (`yes_bid 0.0000` / `yes_ask 1.0000`, whose midpoint is
    the synthetic $0.50 nobody quoted). Marked `e2e` because it is the pipeline-level guard for a
    forecast-affecting change; it opens no socket and bills no key (fake aiohttp session, both LLM stages
    stubbed).
    """

    # A real Kalshi ladder's shape: mutually-exclusive brackets whose open prices sum to ~0.98, plus one
    # far-tail strike nobody is quoting. Labels are distinct and none is a substring of another, so
    # "every outcome reached the prompt" is a real check rather than an accidental one.
    _BRACKETS: tuple[tuple[str, float | None], ...] = (
        ("Below $4.00", 0.06),
        ("$4.00 - $4.19", 0.09),
        ("$4.20 - $4.39", 0.14),
        ("$4.40 - $4.59", 0.19),
        ("$4.60 - $4.79", 0.17),
        ("$4.80 - $4.99", 0.12),
        ("$5.00 - $5.19", 0.08),
        ("$5.20 - $5.39", 0.05),
        ("$5.40 - $5.59", 0.03),
        ("$5.60 - $5.79", 0.02),
        ("$5.80 - $5.99", 0.02),
        ("Above $6.00", None),
    )

    def _diesel_question(self, mock_question: Any) -> Any:
        """The seam's question stub, re-pointed at the ladder's own subject.

        The Kalshi channel scores the catalogue against the question's own text, so an event about a
        different subject would make the pool membership itself the thing under test.
        """
        mock_question.question_text = "What will the US average diesel price be at the end of 2026?"
        mock_question.title = mock_question.question_text
        mock_question.short_title = "diesel price end of 2026"
        mock_question.resolution_criteria = "Resolves to the EIA weekly on-highway diesel average for 2026-12-28."
        mock_question.background_info = "EIA reports a weekly national average."
        return mock_question

    def _ladder_event(self) -> dict[str, Any]:
        markets = []
        for index, (label, price) in enumerate(self._BRACKETS):
            bid, ask = (0.0, 1.0) if price is None else (round(price - 0.01, 2), round(price + 0.01, 2))
            markets.append(
                {
                    "ticker": f"KXDIESEL-26DEC-{index:02d}",
                    "yes_sub_title": label,
                    "status": "active",
                    "yes_bid_dollars": f"{bid:.4f}",
                    "yes_ask_dollars": f"{ask:.4f}",
                    "notional_value_dollars": "1.0000",
                    "volume_fp": "42000.00",
                    "open_interest_fp": "9000.00",
                    "close_time": "2026-12-28T23:59:59Z",
                    "rules_primary": "Settles on the EIA weekly on-highway diesel average.",
                }
            )
        return {
            "event_ticker": "KXDIESEL-26DEC",
            "title": "What will the US average diesel price be at the end of 2026?",
            "sub_title": "EIA weekly average, 2026-12-28",
            "markets": markets,
        }

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_every_bracket_and_no_fabricated_price_reaches_the_forecaster_prompt(
        self, monkeypatch, mock_question, caplog
    ):
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        question = self._diesel_question(mock_question)
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, {"events": [self._ladder_event()], "cursor": ""}),
                # The baseline's PredictIt market is emptied out so the ladder is the slate's ONLY family
                # and the marker's outcome accounting is exact rather than a lower bound.
                _PREDICTIT_URL: FakeResponse(200, {"markets": []}),
            }
        )

        with (
            caplog.at_level(logging.INFO, logger="metaculus_bot.research.prediction_market"),
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=_rank_one_per_venue)),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            research = await pmp.prediction_market_provider()(question)

        assert "| ↳|" in research, "the ranker must have picked the ladder, or nothing below is under test"
        # Completeness: every bracket reaches the text, whether on a full sub-row or in the ladder row.
        for label, _ in self._BRACKETS:
            assert label in research, label
        assert "[remaining " in research, "the fixture must exceed the full-row allowance"

        # No fabricated price: the empty-book strike is named and NOT priced. Its midpoint would be a
        # synthetic 0.50, which is the class of number this change exists to keep out of the table.
        unquoted_label = self._BRACKETS[-1][0]
        assert f"{unquoted_label} -" in research, "an unquoted outcome is named with a dash in the ladder row"
        assert f"{unquoted_label} 0.50" not in research
        assert "| 0.50 |" not in research

        # The seam's telemetry, keyed to the question the provider was handed: `withheld` counts the
        # refusal, and the completeness invariant holds on the EMITTED line rather than only in the renderer.
        marker = next(rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("MARKET_CHILD_RENDER:"))
        fields = dict(field.split("=", 1) for field in marker.removeprefix("MARKET_CHILD_RENDER: ").split(" "))
        assert fields["question"] == str(question.id_of_question)
        assert int(fields["outcomes"]) == len(self._BRACKETS)
        assert int(fields["named"]) + int(fields["collapsed"]) == int(fields["outcomes"])
        assert fields["withheld"] == "1"

        # And the whole ladder plus the instruction for reading it land in ONE forecaster prompt. The
        # render half is what makes the distribution available; the prompt sentence is what stops a model
        # reading one bracket as an equality constraint on a tail (the q45189 failure). The provider
        # returns the bare snapshot; the orchestrator labels it with the section header on the way to the
        # forecaster (`assemble_provider_sections`), and the prompt's market clause is gated on that header,
        # so the prompt is built from the section as the forecaster actually receives it.
        question.open_time = datetime.now(UTC) - timedelta(days=30)
        question.scheduled_resolution_time = datetime.now(UTC) + timedelta(days=120)
        prompt = binary_prompt(question, research=f"{provider_header('prediction_market')}\n{research}")

        for label, _ in self._BRACKETS:
            assert label in prompt, label
        assert "[remaining " in prompt
        assert "is a DISTRIBUTION over that market's own question" in prompt
        assert "Never treat one outcome's price as an equality constraint" in prompt
