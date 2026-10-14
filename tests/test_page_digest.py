"""The LLM-extractive page digest (``research/page_digest.py``): grounding, fallback and the wall.

Every test runs against a scripted ``invoke`` double patched in at the module's builder seam, so
no test builds a real client or leaves the process (the autouse egress guard in ``conftest.py``
would raise if one did). The synthetic page is long enough to cross the pre-filter cap and carries
its resolving figure in the TAIL, which is the exact shape head-first truncation lost.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import litellm.exceptions
import pytest

import metaculus_bot.research.page_digest as page_digest_module
from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_DIGEST_WINDOW_CHARS,
    PAGE_DIGEST_EXTRACTOR_EFFORT,
    PAGE_DIGEST_EXTRACTOR_MODEL,
    PAGE_DIGEST_EXTRACTOR_TIMEOUT_S,
    PAGE_DIGEST_MIN_CALL_BUDGET_S,
    PAGE_DIGEST_PREFILTER_MAX_CHARS,
    PAGE_DIGEST_WALL_MARGIN_S,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.research.page_digest import (
    DIGEST_METHOD_BM25,
    DIGEST_METHOD_LLM_EXTRACTIVE,
    PAGE_DIGEST_EXTRACTOR_PROMPT,
    PAGE_DIGEST_ROLE,
    PageDigest,
    PageDigestPassages,
    digest_page,
)

QUERY = "What was the civilian unemployment rate for August 2026?\nResolves on the BLS Employment Situation figure."

OPENING_SENTENCE = "Employment Situation Summary. The Bureau of Labor Statistics released the August 2026 report."
MIDDLE_SENTENCE = "Health care added 31,000 jobs in August, in line with the average monthly gain over the prior year."
TAIL_TABLE = (
    "Table A-1. Employment status of the civilian population.\n"
    "| Measure | July 2026 | August 2026 |\n"
    "| Civilian unemployment rate | 4.2 | 4.3 |\n"
    "| Labor force participation rate | 62.3 | 62.2 |"
)
FILLER_SENTENCE = (
    "Nonfarm payroll employment changed little over the month, and the household survey measures were "
    "essentially unchanged from the prior release across every major worker group."
)


def _synthetic_page() -> str:
    """A page well past the pre-filter cap whose resolving table sits at the very end."""
    filler_paragraphs = [f"Paragraph {i}. {FILLER_SENTENCE}" for i in range(140)]
    body = "\n\n".join([OPENING_SENTENCE, *filler_paragraphs[:70], MIDDLE_SENTENCE, *filler_paragraphs[70:]])
    page = f"{body}\n\n{TAIL_TABLE}"
    assert len(page) > PAGE_DIGEST_PREFILTER_MAX_CHARS
    return page


PAGE = _synthetic_page()


class ScriptedLlm:
    """An ``invoke`` double: returns the scripted string, raises the scripted exception, or hangs."""

    def __init__(self, result: str | BaseException | None = None, *, hang_seconds: float = 0.0) -> None:
        self._result = result
        self._hang_seconds = hang_seconds
        self.prompts: list[str] = []

    async def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._hang_seconds:
            await asyncio.sleep(self._hang_seconds)
        if isinstance(self._result, BaseException):
            raise self._result
        assert isinstance(self._result, str)
        return self._result


@pytest.fixture
def builder_calls() -> list[dict[str, Any]]:
    """Every builder call's kwargs, filled in by ``_install``; the test decides which double the builder hands back."""
    return []


def _install(monkeypatch: pytest.MonkeyPatch, llm: ScriptedLlm, calls: list[dict[str, Any]]) -> None:
    def fake_builder(**kwargs: Any) -> ScriptedLlm:
        calls.append(kwargs)
        return llm

    monkeypatch.setattr(page_digest_module, "build_llm_with_openrouter_fallback", fake_builder)


def _passages_json(*passages: str) -> str:
    return PageDigestPassages(passages=list(passages)).model_dump_json()


def _norm(text: str) -> str:
    return " ".join(text.split())


class TestGrounding:
    """A passage is kept only when it is a literal substring of the page after whitespace and quote-glyph normalisation."""

    async def test_mixed_returns_keep_only_the_literal_passages(self, monkeypatch, builder_calls) -> None:
        reflowed_tail = TAIL_TABLE.replace("\n", " ").replace("  ", " ")  # same text, whitespace reflowed
        paraphrase = "The jobless rate in August 2026 came in at 4.3 percent."
        fabrication = "Table A-2 shows the rate fell to 3.9 percent in August 2026."
        llm = ScriptedLlm(_passages_json(reflowed_tail, paraphrase, MIDDLE_SENTENCE, fabrication, "   "))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert isinstance(digest, PageDigest)
        assert digest.passages_returned == 5
        assert digest.passages_grounded == 2
        assert digest.fallback_used is False
        assert digest.method == DIGEST_METHOD_LLM_EXTRACTIVE
        # The opening passage leads, then the grounded passages in the model's rank order.
        assert digest.passages[0].startswith(OPENING_SENTENCE)
        assert [_norm(p) for p in digest.passages[1:]] == [_norm(TAIL_TABLE), _norm(MIDDLE_SENTENCE)]
        assert not any(paraphrase in p or fabrication in p for p in digest.passages)

    async def test_the_kept_passage_is_the_models_text_not_a_collapsed_copy(self, monkeypatch, builder_calls) -> None:
        """Whitespace is normalised for the CHECK only: a table keeps its rows."""
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.passages[1] == TAIL_TABLE

    async def test_duplicates_and_opening_repeats_are_dropped_but_counted_grounded(
        self, monkeypatch, builder_calls
    ) -> None:
        llm = ScriptedLlm(_passages_json(TAIL_TABLE, TAIL_TABLE, OPENING_SENTENCE))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.passages_returned == 3
        assert digest.passages_grounded == 3
        assert len(digest.passages) == 2  # opening + one distinct grounded passage
        assert digest.fallback_used is False

    async def test_nothing_grounded_falls_back_to_bm25_and_keeps_the_counts(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json("Not on the page at all.", "Neither is this sentence."))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.fallback_used is True
        assert digest.method == DIGEST_METHOD_BM25
        assert digest.passages_returned == 2
        assert digest.passages_grounded == 0
        assert digest.passages[0].startswith(OPENING_SENTENCE)
        # BM25 on the same query still finds the resolving table in the tail.
        assert any("Civilian unemployment rate | 4.2 | 4.3" in p for p in digest.passages[1:])
        assert len(digest.passages) <= 1 + DOCUMENT_DIGEST_TOP_K

    async def test_an_empty_answer_list_is_a_valid_no_match(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json())
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.fallback_used is False
        assert digest.outcome == "valid_empty"
        assert digest.passages == []
        assert digest.passages_returned == 0
        assert digest.passages_grounded == 0

    async def test_grounded_passages_inside_the_opening_are_a_success_not_a_fallback(
        self, monkeypatch, builder_calls
    ) -> None:
        """The lede case: the model found the answer in the page's first window, so the opening alone is the digest."""
        llm = ScriptedLlm(_passages_json(OPENING_SENTENCE))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.method == DIGEST_METHOD_LLM_EXTRACTIVE
        assert digest.fallback_used is False
        assert digest.passages_returned == 1
        assert digest.passages_grounded == 1
        assert len(digest.passages) == 1
        assert digest.passages[0].startswith(OPENING_SENTENCE)

    async def test_the_llm_path_serves_at_most_top_k_passages_after_the_opening(
        self, monkeypatch, builder_calls
    ) -> None:
        """The same bound as the BM25 path; the counters still report the full answer."""
        many = [f"Paragraph {i}. {FILLER_SENTENCE}" for i in range(10, 30)]
        llm = ScriptedLlm(_passages_json(*many))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.passages_returned == 20
        assert digest.passages_grounded == 20
        assert len(digest.passages) == 1 + DOCUMENT_DIGEST_TOP_K
        assert digest.passages[1:] == many[:DOCUMENT_DIGEST_TOP_K]

    async def test_a_curly_quoted_sentence_retyped_with_straight_quotes_still_grounds(
        self, monkeypatch, builder_calls
    ) -> None:
        quoted = "The commissioner said “the civilian unemployment rate for August 2026 was 4.3 percent” today."
        page = "\n\n".join([OPENING_SENTENCE, *(f"Paragraph {i}. {FILLER_SENTENCE}" for i in range(4)), quoted])
        assert len(page) > DOCUMENT_DIGEST_WINDOW_CHARS  # so the quoted sentence sits outside the opening
        retyped = quoted.replace("“", '"').replace("”", '"')
        llm = ScriptedLlm(_passages_json(retyped))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(page, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.passages_grounded == 1
        assert digest.fallback_used is False
        assert digest.passages[1] == quoted

    async def test_a_passage_spliced_across_a_real_cut_is_not_grounded(self, monkeypatch, builder_calls) -> None:
        """The one fabrication shape the pre-filter itself creates: text joined across a `[...]` cut."""
        probe = ScriptedLlm(_passages_json())
        _install(monkeypatch, probe, builder_calls)
        await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)
        page_text = probe.prompts[0].split("Page text:\n", 1)[1]
        cut = page_text.index("\n[...]\n")
        spliced = page_text[cut - 60 : cut + len("\n[...]\n") + 60]

        llm = ScriptedLlm(_passages_json(spliced))
        _install(monkeypatch, llm, builder_calls)
        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.passages_returned == 1
        assert digest.passages_grounded == 0
        assert digest.fallback_used is True
        assert digest.outcome == "ungrounded"

    async def test_an_unmatched_page_keeps_the_opening_on_a_transient_digest_failure(
        self, monkeypatch, builder_calls
    ) -> None:
        llm = ScriptedLlm(TimeoutError("provider timeout"))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, "zymurgy quokka", budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.outcome == "timeout"
        assert digest.fallback_used is True
        assert digest.passages == [page_digest_module._opening_passage(PAGE)]
        assert digest.evidence_passages == ()


class TestPrefilter:
    """A page past the cap reaches the model as its best BM25 windows, in page order; a short page goes whole."""

    async def test_long_page_is_cut_to_the_cap_and_the_tail_survives(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        (prompt,) = llm.prompts
        page_text = prompt.split("Page text:\n", 1)[1]
        assert len(page_text) <= PAGE_DIGEST_PREFILTER_MAX_CHARS + 64 * len("\n[...]\n")
        assert "Civilian unemployment rate | 4.2 | 4.3" in page_text
        assert "Paragraph 139." not in page_text or "[...]" in page_text  # filler was cut, the gap is marked
        assert QUERY in prompt
        assert str(DOCUMENT_DIGEST_TOP_K) in prompt

    async def test_windows_are_presented_in_page_order(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        (prompt,) = llm.prompts
        page_text = prompt.split("Page text:\n", 1)[1]
        assert page_text.index(OPENING_SENTENCE[:40]) < page_text.index("Table A-1.")

    async def test_short_page_goes_to_the_model_whole(self, monkeypatch, builder_calls) -> None:
        short_page = f"{OPENING_SENTENCE}\n\n{MIDDLE_SENTENCE}\n\n{TAIL_TABLE}"
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        await digest_page(short_page, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        (prompt,) = llm.prompts
        assert prompt.endswith(short_page)
        assert "[...]" not in prompt

    async def test_long_page_with_no_query_term_sends_its_head(self, monkeypatch, builder_calls) -> None:
        """BM25 selects nothing when no query token occurs; the head is what a reader saw before."""
        llm = ScriptedLlm(_passages_json())
        _install(monkeypatch, llm, builder_calls)

        await digest_page(PAGE, "zymurgy quokka", budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        (prompt,) = llm.prompts
        page_text = prompt.split("Page text:\n", 1)[1]
        assert page_text == PAGE[:PAGE_DIGEST_PREFILTER_MAX_CHARS]

    async def test_a_table_spanning_several_windows_reaches_the_model_unbroken(
        self, monkeypatch, builder_calls
    ) -> None:
        """Abutting windows are spliced back together; `[...]` marks only a real cut."""
        table = "\n".join(
            f"| Row {n} | Civilian unemployment rate, August 2026 | 4.{n % 10} percent |" for n in range(40)
        )
        filler = "\n\n".join(f"Paragraph {i}. {FILLER_SENTENCE}" for i in range(100))
        page = f"{OPENING_SENTENCE}\n\n{filler}\n\n{table}"
        assert len(page) > PAGE_DIGEST_PREFILTER_MAX_CHARS
        assert len(table) > 3 * DOCUMENT_DIGEST_WINDOW_CHARS  # the table alone spans several windows
        llm = ScriptedLlm(_passages_json())
        _install(monkeypatch, llm, builder_calls)

        await digest_page(page, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        (prompt,) = llm.prompts
        page_text = prompt.split("Page text:\n", 1)[1]
        assert table in page_text
        assert "[...]" in page_text  # the filler between the opening and the table was cut


class TestWallBudget:
    """The call is bounded by min(per-call timeout, remaining budget less the margin) and never attempted below the floor."""

    async def test_below_the_floor_no_call_is_made(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)
        budget = PAGE_DIGEST_WALL_MARGIN_S + PAGE_DIGEST_MIN_CALL_BUDGET_S - 0.01

        digest = await digest_page(PAGE, QUERY, budget_seconds=budget)

        assert builder_calls == []
        assert llm.prompts == []
        assert digest.fallback_used is True
        assert digest.method == DIGEST_METHOD_BM25
        assert digest.passages_returned == 0
        assert digest.passages_grounded == 0
        assert digest.passages[0].startswith(OPENING_SENTENCE)

    async def test_just_above_the_floor_the_call_is_made(self, monkeypatch, builder_calls) -> None:
        """The BM25 hop's own milliseconds are charged first, so the floor is tested with a little slack."""
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(
            PAGE, QUERY, budget_seconds=PAGE_DIGEST_WALL_MARGIN_S + PAGE_DIGEST_MIN_CALL_BUDGET_S + 0.5
        )

        assert len(llm.prompts) == 1
        assert digest.fallback_used is False

    async def test_a_slow_bm25_hop_is_charged_against_the_budget(self, monkeypatch, builder_calls) -> None:
        """CPU spent before the call counts: a hop that eats the budget leaves no room for a call."""
        monkeypatch.setattr(page_digest_module, "PAGE_DIGEST_WALL_MARGIN_S", 0.0)
        monkeypatch.setattr(page_digest_module, "PAGE_DIGEST_MIN_CALL_BUDGET_S", 1.0)

        def slow_select_passages(*args: Any, **kwargs: Any) -> list[Any]:
            time.sleep(0.3)
            return []

        monkeypatch.setattr(page_digest_module, "select_passages", slow_select_passages)
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=1.2)

        assert llm.prompts == []
        assert digest.fallback_used is True

    async def test_a_hung_call_is_cut_at_the_per_call_timeout(self, monkeypatch, builder_calls) -> None:
        monkeypatch.setattr(page_digest_module, "PAGE_DIGEST_EXTRACTOR_TIMEOUT_S", 0.05)
        monkeypatch.setattr(page_digest_module, "PAGE_DIGEST_MIN_CALL_BUDGET_S", 0.0)
        llm = ScriptedLlm(_passages_json(TAIL_TABLE), hang_seconds=5.0)
        _install(monkeypatch, llm, builder_calls)

        started = time.monotonic()
        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert time.monotonic() - started < 1.0
        assert len(llm.prompts) == 1
        assert digest.fallback_used is True
        assert digest.method == DIGEST_METHOD_BM25
        assert digest.passages_returned == 0

    async def test_a_hung_call_is_cut_at_the_remaining_wall_when_that_is_smaller(self, monkeypatch, builder_calls):
        """The budget, not the 20 s constant, wins when the caller has less wall left."""
        monkeypatch.setattr(page_digest_module, "PAGE_DIGEST_WALL_MARGIN_S", 0.0)
        monkeypatch.setattr(page_digest_module, "PAGE_DIGEST_MIN_CALL_BUDGET_S", 0.0)
        llm = ScriptedLlm(_passages_json(TAIL_TABLE), hang_seconds=5.0)
        _install(monkeypatch, llm, builder_calls)

        started = time.monotonic()
        digest = await digest_page(PAGE, QUERY, budget_seconds=0.05)

        assert time.monotonic() - started < 1.0
        assert digest.fallback_used is True

    async def test_empty_page_makes_no_call(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json())
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page("   \n ", QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert llm.prompts == []
        assert digest.passages == []
        assert digest.fallback_used is False
        assert digest.outcome == "empty_page"


class TestExpectedFailures:
    """Only the enumerated failure classes degrade to BM25; anything else propagates."""

    @pytest.mark.parametrize(
        "exc",
        [
            litellm.exceptions.Timeout("slow", model="m", llm_provider="openrouter"),
            litellm.exceptions.APIConnectionError("reset", model="m", llm_provider="openrouter"),
            litellm.exceptions.InternalServerError("500", model="m", llm_provider="openrouter"),
            litellm.exceptions.ServiceUnavailableError("503", model="m", llm_provider="openrouter"),
            litellm.exceptions.RateLimitError("429", model="m", llm_provider="openrouter"),
            litellm.exceptions.AuthenticationError("401", model="m", llm_provider="openrouter"),
            litellm.exceptions.BadRequestError("400", model="m", llm_provider="openrouter"),
            litellm.exceptions.ContextWindowExceededError("too long", model="m", llm_provider="openrouter"),
            litellm.exceptions.APIError(403, "OpenrouterException - forbidden", llm_provider="openrouter", model="m"),
        ],
        ids=lambda exc: type(exc).__name__,
    )
    async def test_each_provider_error_falls_back(self, monkeypatch, builder_calls, exc) -> None:
        llm = ScriptedLlm(exc)
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.fallback_used is True
        assert digest.method == DIGEST_METHOD_BM25
        assert digest.passages_returned == 0

    @pytest.mark.parametrize("raw", ["not json at all", '{"passages": "one string, not a list"}', '{"other": []}'])
    async def test_an_off_schema_answer_falls_back(self, monkeypatch, builder_calls, raw) -> None:
        llm = ScriptedLlm(raw)
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.fallback_used is True
        assert digest.passages_returned == 0

    async def test_an_empty_completion_falls_back(self, monkeypatch, builder_calls) -> None:
        """forecasting-tools' bare RuntimeError for an empty answer, the repo's most-seen zero-output failure."""
        llm = ScriptedLlm(RuntimeError("LLM answer is an empty string. The model was m and the prompt was: p"))
        _install(monkeypatch, llm, builder_calls)

        digest = await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        assert digest.fallback_used is True
        assert digest.method == DIGEST_METHOD_BM25
        assert digest.passages_returned == 0

    @pytest.mark.parametrize(
        "exc",
        [
            ZeroDivisionError("a bug in the call path"),
            RuntimeError("some other runtime failure"),
            AssertionError("Answer is not a string and is of type: <class 'NoneType'>. Answer: None"),
        ],
        ids=lambda exc: type(exc).__name__,
    )
    async def test_an_unexpected_error_propagates(self, monkeypatch, builder_calls, exc) -> None:
        """Only the empty-completion RuntimeError is expected; the None-content assert and everything else surface."""
        llm = ScriptedLlm(exc)
        _install(monkeypatch, llm, builder_calls)

        with pytest.raises(type(exc)):
            await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)


class TestClientConstruction:
    """The client is built the way every other support role builds one, with the role tagged for the ledger."""

    async def test_builder_kwargs(self, monkeypatch, builder_calls) -> None:
        llm = ScriptedLlm(_passages_json(TAIL_TABLE))
        _install(monkeypatch, llm, builder_calls)

        await digest_page(PAGE, QUERY, budget_seconds=RESOLUTION_SOURCE_WALL_TIMEOUT)

        (kwargs,) = builder_calls
        assert kwargs["model"] == PAGE_DIGEST_EXTRACTOR_MODEL
        assert kwargs["role"] == PAGE_DIGEST_ROLE == "page_digest_extractor"
        assert kwargs["reasoning"] == {"effort": PAGE_DIGEST_EXTRACTOR_EFFORT}
        assert kwargs["timeout"] == PAGE_DIGEST_EXTRACTOR_TIMEOUT_S
        assert kwargs["allowed_tries"] == 1
        assert kwargs["response_format"] is PageDigestPassages
        assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}

    def test_the_real_builder_tags_the_role_on_the_personal_key(self, monkeypatch) -> None:
        """Construction only, no call: the litellm metadata the ledger callback reads back carries the role."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.setattr(
            page_digest_module, "build_llm_with_openrouter_fallback", build_llm_with_openrouter_fallback
        )

        llm = page_digest_module._build_extractor()

        assert llm.model == PAGE_DIGEST_EXTRACTOR_MODEL
        metadata = llm.litellm_kwargs["metadata"]
        assert metadata["role"] == "page_digest_extractor"
        assert metadata["key_alias"] == "personal"
        assert llm.litellm_kwargs["response_format"] is PageDigestPassages


class TestOpeningPassage:
    """The opening window is cut at whitespace of any kind, so a table or list head never ends mid-token."""

    def test_a_newline_separated_head_is_cut_on_a_line_boundary(self) -> None:
        rows = [f"row_{n}_value_{n * 7}" for n in range(200)]
        text = "Table 1\n" + "\n".join(rows)
        assert len(text) > DOCUMENT_DIGEST_WINDOW_CHARS

        opening = page_digest_module._opening_passage(text)

        assert opening.startswith("Table 1\n")
        assert len(opening) > DOCUMENT_DIGEST_WINDOW_CHARS // 2  # not collapsed to the first token
        assert opening.split("\n")[-1] in rows  # the last line is a whole row

    def test_a_short_page_is_its_own_opening(self) -> None:
        assert page_digest_module._opening_passage("  one short page  ") == "one short page"


class TestConstants:
    """The knobs, pinned to the receipts in docs/constants.md."""

    def test_model_is_an_openrouter_slug_so_the_builder_routes_its_key(self) -> None:
        """A bare ``openai/`` slug would dial OpenAI directly on a key this repo does not carry."""
        assert PAGE_DIGEST_EXTRACTOR_MODEL.startswith("openrouter/openai/")

    def test_effort_is_medium_by_operator_decision(self) -> None:
        assert PAGE_DIGEST_EXTRACTOR_EFFORT == "medium"

    def test_the_call_fits_inside_the_fetch_wall(self) -> None:
        assert PAGE_DIGEST_EXTRACTOR_TIMEOUT_S + PAGE_DIGEST_WALL_MARGIN_S < RESOLUTION_SOURCE_WALL_TIMEOUT
        assert 0 < PAGE_DIGEST_MIN_CALL_BUDGET_S <= PAGE_DIGEST_EXTRACTOR_TIMEOUT_S

    def test_the_prefilter_holds_more_windows_than_the_digest_returns(self) -> None:
        assert PAGE_DIGEST_PREFILTER_MAX_CHARS // DOCUMENT_DIGEST_WINDOW_CHARS >= DOCUMENT_DIGEST_TOP_K


class TestPromptPins:
    """The extractor prompt's load-bearing clauses: verbatim only, ranked, no paraphrase."""

    def test_verbatim_clause(self) -> None:
        assert "VERBATIM" in PAGE_DIGEST_EXTRACTOR_PROMPT

    def test_no_paraphrase_clause(self) -> None:
        assert "Do not paraphrase" in PAGE_DIGEST_EXTRACTOR_PROMPT

    def test_ranked_clause(self) -> None:
        assert "most relevant first" in PAGE_DIGEST_EXTRACTOR_PROMPT

    def test_template_slots(self) -> None:
        for slot in ("{query}", "{page_text}", "{max_passages}"):
            assert slot in PAGE_DIGEST_EXTRACTOR_PROMPT
