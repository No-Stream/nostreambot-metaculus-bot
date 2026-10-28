"""End-to-end path for the time-series anchor: route -> fetch seam -> render -> bundle -> prompt.

Every other anchor test stops at one seam. `test_ts_routing` ends at the `_Route`,
`test_timeseries_anchor_provider` ends at the provider's return string, and the pipeline e2e
suites patch `run_research` wholesale, so nothing checked that a routed question's anchor
actually SURVIVES the research bundle and reaches the forecaster. Three links sit in that gap
and each is a live regression risk:

- the orchestrator has to select the provider off `TS_ANCHOR_ENABLED` and stamp the section
  with `provider_header("timeseries_anchor")`;
- that header string is `prompts.TS_ANCHOR_SECTION_HEADER`, and the numeric prompt gates its
  anchor-evidence clause on finding that exact substring in the research — so a header reword
  on either side silently drops the guidance while every unit test stays green;
- `_demote_inner_headings` runs over every provider body, so the anchor's own text has to come
  through it intact.

Only the network is mocked: `timeseries_anchor.fetch_series` is replaced with a synthetic
series (the module's real HTTP seam is covered in `test_ts_fetch`), the autouse egress guard in
`conftest` blocks anything else, and no LLM is reachable because the only enabled provider
makes no model call.

The 24/7 (crypto) shape is deliberate. It is the series whose horizon and volatility basis the
calendar-vs-row-count fix changed, and it is reachable in prod through any Yahoo ticker URL, so
the e2e assertion doubles as the pipeline-level receipt for that fix.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import GeneralLlm, NumericQuestion

from metaculus_bot.prompts import TS_ANCHOR_SECTION_HEADER, numeric_prompt
from metaculus_bot.research import timeseries_anchor as ts
from metaculus_bot.research.orchestrator import ResearchOrchestrator

# Horizon of the question under test. Fixed so the rendered band line names a known step count:
# on a 24/7 series one step IS one calendar day, so h == this number.
_HORIZON_CALENDAR_DAYS = 90

# Flags for every optional provider except the anchor, plus both gap-fill passes. Set to "false"
# rather than deleted so a value inherited from the developer's shell can't switch one on.
_OTHER_PROVIDER_FLAGS = (
    "NATIVE_SEARCH_ENABLED",
    "GEMINI_SEARCH_ENABLED",
    "FINANCIAL_DATA_ENABLED",
    "PREDICTION_MARKETS_ENABLED",
    "RESOLUTION_SOURCE_ENABLED",
    "GAP_FILL_ENABLED",
    "GAP_FILL_V2_ENABLED",
    "RAW_RESEARCH_LOG_ENABLED",
)

# AskNews credentials are deleted so the anchor is the only provider in the bundle and the
# assertions read the anchor's own text rather than a stub's.
_PRIMARY_PROVIDER_KEYS = (
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
)


def _twenty_four_seven_series(end: pd.Timestamp, *, years: int = 6) -> pd.Series:
    """A strictly-positive series with a bar every CALENDAR day — the crypto shape, whose
    observed density (1.0 rows/day) puts it on the 365 annualization basis."""
    index = pd.date_range(end=end, periods=years * 365, freq="D")
    rng = np.random.default_rng(20260825)
    walk = 3_000.0 + np.cumsum(rng.normal(0.0, 20.0, len(index)))
    return pd.Series(np.abs(walk) + 500.0, index=index, name="BTC-USD")


def _btc_question(*, resolution_criteria: str, qid: int = 46001) -> NumericQuestion:
    """A real NumericQuestion (not a mock): the provider's isinstance gate admits only
    NumericQuestion, and `numeric_prompt` reads bounds/units off the real model. Bounds are wide
    open so the provider's magnitude backstop is a no-op — the backstop has its own tests, and
    letting it fire here would mask a render regression as a bounds mismatch."""
    now = datetime.now(UTC)
    return NumericQuestion(
        question_text="What will the price of Bitcoin be at the end of the window?",
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}",
        lower_bound=0.0,
        upper_bound=1_000_000.0,
        open_lower_bound=True,
        open_upper_bound=True,
        zero_point=None,
        open_time=now - timedelta(days=10),
        scheduled_resolution_time=now + timedelta(days=_HORIZON_CALENDAR_DAYS),
        resolution_criteria=resolution_criteria,
        background_info="",
        fine_print="",
        unit_of_measure="USD",
    )


@pytest.fixture
def anchor_only_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anchor on, every other provider off, no primary-provider credentials.

    Deleting the credentials matters and is not paranoia: ``metaculus_bot.config`` calls
    ``load_dotenv()`` at import, so a developer's real AskNews credentials are in
    ``os.environ`` by the time any test runs. Left in place, the primary provider would be
    selected here, try to reach a real host, get refused by conftest's egress guard, and land in
    the bundle as a soft-fail banner — the assertions below would still pass while reading a
    bundle that isn't the one under test. ``assert_anchor_is_the_only_provider`` is the check
    that makes that impossible rather than merely unlikely."""
    monkeypatch.setenv("TS_ANCHOR_ENABLED", "true")
    monkeypatch.delenv("TS_ANCHOR_CHART_ENABLED", raising=False)
    for flag in _OTHER_PROVIDER_FLAGS:
        monkeypatch.setenv(flag, "false")
    for key in _PRIMARY_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_anchor_caches() -> Iterator[None]:
    # The provider memoizes rendered sections per (qid, as_of) and the fetch layer caches parsed
    # series; both would bleed across these two tests.
    ts._reset_session_caches()
    yield
    ts._reset_session_caches()


@pytest.fixture
def orchestrator() -> ResearchOrchestrator:
    llm = GeneralLlm(model="test/model", temperature=0.0)
    return ResearchOrchestrator(default_llm=llm, summarizer_llm=llm)


def assert_anchor_is_the_only_provider(orchestrator: ResearchOrchestrator) -> None:
    """Read the selection the orchestrator would make, so the bundle under assertion is known to
    be the anchor's own text and not a stub's or a soft-failed provider's banner."""
    assert [name for _callable, name in orchestrator._select_research_providers()] == ["timeseries_anchor"]


class TestAnchorRidesTheResearchBundleIntoThePrompt:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_a_routed_numeric_question_renders_an_anchor_the_prompt_can_see(
        self, orchestrator: ResearchOrchestrator, anchor_only_env: None, monkeypatch: pytest.MonkeyPatch
    ):
        """The whole path on one question: a Yahoo ticker URL in the resolution criteria routes,
        the fetch seam hands back a 24/7 series, the render produces a calendar-step band, the
        orchestrator stamps the section header, and the numeric prompt's anchor clause fires off
        that header. Break any link — routing, header string, prompt gate — and this fails."""
        assert_anchor_is_the_only_provider(orchestrator)
        series = _twenty_four_seven_series(pd.Timestamp(datetime.now(UTC).date()))
        monkeypatch.setattr(ts, "fetch_series", lambda *_a, **_k: series)
        question = _btc_question(
            resolution_criteria="Resolves per the close at https://finance.yahoo.com/quote/BTC-USD/history/."
        )

        research = await orchestrator.run_research(question)

        # The orchestrator's own header, which is the substring the prompt gates on, and the whole
        # bundle is the anchor's one section (no other provider ran).
        assert research.startswith(TS_ANCHOR_SECTION_HEADER)
        # The anchor's quantitative payload survived the bundle's heading demotion intact...
        assert "P10 / P50 / P90 →" in research
        # ...on the series' own clock: one step is one calendar day for a 24/7 series, so the
        # band is the full 90-day window and says so (the pre-fix render read 62 trading days).
        assert f"all {_HORIZON_CALENDAR_DAYS}-calendar-day change windows" in research
        assert "trading-day" not in research
        assert "annualized realized volatility" in research

        prompt = numeric_prompt(question, research, "Lower bound is open.", "Upper bound is open.")

        assert "purely-statistical extrapolation of the resolution series' own history" in prompt
        assert TS_ANCHOR_SECTION_HEADER in prompt

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_an_unroutable_question_leaves_no_anchor_and_arms_no_clause(
        self, orchestrator: ResearchOrchestrator, anchor_only_env: None, monkeypatch: pytest.MonkeyPatch
    ):
        """The negative half, and the reason the prompt gate exists: on a question the router
        skips, the anchor contributes nothing and the numeric prompt must NOT carry guidance
        pointing at a section that isn't there. Without this, the positive test above would pass
        just as happily if the clause were unconditional."""
        assert_anchor_is_the_only_provider(orchestrator)
        fetch_calls: list[object] = []
        monkeypatch.setattr(ts, "fetch_series", lambda *a, **_k: fetch_calls.append(a) or pd.Series(dtype="float64"))
        question = _btc_question(
            resolution_criteria="Resolves per the consensus of election forecasters, with no series cited.",
            qid=46002,
        )
        question.question_text = "Who will win the 2028 presidential election?"

        research = await orchestrator.run_research(question)

        assert TS_ANCHOR_SECTION_HEADER not in research
        assert not fetch_calls, "an unroutable question must not reach the fetch layer at all"

        prompt = numeric_prompt(question, research, "Lower bound is open.", "Upper bound is open.")

        assert "purely-statistical extrapolation of the resolution series' own history" not in prompt
