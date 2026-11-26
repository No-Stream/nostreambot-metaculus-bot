"""Tests for backtest question preparation."""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)

from metaculus_bot.backtest.question_prep import (
    _extract_ground_truth,
    _fetch_with_retries,
    _prepare_question_for_backtest,
    fetch_resolved_questions,
    fetch_resolved_questions_stratified,
)
from metaculus_bot.constants import (
    BACKTEST_DEFAULT_TOURNAMENT,
    BACKTEST_OVERFETCH_RATIO,
)
from metaculus_bot.prompts import binary_prompt

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_question_yes() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it rain tomorrow?",
        id_of_question=1,
        resolution_string="yes",
        community_prediction_at_access_time=0.7,
        state=QuestionState.RESOLVED,
        api_json={},
    )


@pytest.fixture
def binary_question_no() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it snow?",
        id_of_question=2,
        resolution_string="no",
        community_prediction_at_access_time=0.3,
        state=QuestionState.RESOLVED,
        api_json={},
    )


@pytest.fixture
def numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="How many widgets?",
        id_of_question=3,
        resolution_string="42.5",
        state=QuestionState.RESOLVED,
        lower_bound=0.0,
        upper_bound=100.0,
        open_upper_bound=False,
        open_lower_bound=False,
        api_json={},
    )


@pytest.fixture
def mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which option?",
        id_of_question=4,
        resolution_string="Option A",
        state=QuestionState.RESOLVED,
        options=["Option A", "Option B", "Option C"],
        api_json={},
    )


@pytest.fixture
def canceled_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will this be annulled?",
        id_of_question=5,
        resolution_string="annulled",
        community_prediction_at_access_time=0.5,
        state=QuestionState.RESOLVED,
        api_json={},
    )


# ---------------------------------------------------------------------------
# _extract_ground_truth tests
# ---------------------------------------------------------------------------


def test_extract_binary_true(binary_question_yes: BinaryQuestion):
    gt = _extract_ground_truth(binary_question_yes)
    assert gt is not None
    assert gt.resolution is True
    assert gt.question_type == "binary"
    assert gt.question_id == 1


def test_extract_binary_false(binary_question_no: BinaryQuestion):
    gt = _extract_ground_truth(binary_question_no)
    assert gt is not None
    assert gt.resolution is False
    assert gt.question_type == "binary"


def test_extract_numeric(numeric_question: NumericQuestion):
    gt = _extract_ground_truth(numeric_question)
    assert gt is not None
    assert gt.resolution == 42.5
    assert gt.question_type == "numeric"


def test_extract_mc(mc_question: MultipleChoiceQuestion):
    gt = _extract_ground_truth(mc_question)
    assert gt is not None
    assert gt.resolution == "Option A"
    assert gt.question_type == "multiple_choice"


def test_extract_canceled_returns_none(canceled_question: BinaryQuestion):
    gt = _extract_ground_truth(canceled_question)
    assert gt is None


# ---------------------------------------------------------------------------
# _prepare_question_for_backtest tests
# ---------------------------------------------------------------------------


def test_prepare_sets_state_open(binary_question_yes: BinaryQuestion):
    clean = _prepare_question_for_backtest(binary_question_yes)
    assert clean.state == QuestionState.OPEN


def test_prepare_clears_resolution(binary_question_yes: BinaryQuestion):
    clean = _prepare_question_for_backtest(binary_question_yes)
    assert clean.resolution_string is None
    assert clean.actual_resolution_time is None


def test_prepare_blanks_background_info_to_an_empty_string():
    """None would be printed as the word "None" under "Question background:" in every forecaster prompt."""
    question = BinaryQuestion(
        question_text="Will it rain tomorrow?",
        id_of_question=1,
        resolution_string="yes",
        state=QuestionState.RESOLVED,
        api_json={},
        background_info="Some background info",
        resolution_criteria="Resolves yes if it rains.",
        fine_print="",
        open_time=datetime(2026, 10, 1),
        scheduled_resolution_time=datetime(2026, 11, 1),
    )
    clean = _prepare_question_for_backtest(question)
    assert clean.background_info == ""
    prompt = binary_prompt(clean, "research")
    background_section = prompt.split("Question background:")[1].split("This question's outcome")[0]
    assert background_section.strip() == ""


def test_prepare_is_deep_copy(binary_question_yes: BinaryQuestion):
    clean = _prepare_question_for_backtest(binary_question_yes)
    clean.question_text = "MODIFIED"
    assert binary_question_yes.question_text == "Will it rain tomorrow?"


# ---------------------------------------------------------------------------
# Helpers + fixtures for fetch_resolved_questions / stratified tests
# ---------------------------------------------------------------------------


def _make_binary(qid: int, resolved_at: datetime) -> BinaryQuestion:
    """Build a resolved binary question with a specified actual_resolution_time."""
    q = BinaryQuestion(
        question_text=f"Binary {qid}?",
        id_of_question=qid,
        resolution_string="yes",
        community_prediction_at_access_time=0.6,
        state=QuestionState.RESOLVED,
        api_json={},
    )
    q.actual_resolution_time = resolved_at
    return q


def _make_numeric(qid: int, resolved_at: datetime) -> NumericQuestion:
    """Build a resolved numeric question with a specified actual_resolution_time."""
    q = NumericQuestion(
        question_text=f"Numeric {qid}?",
        id_of_question=qid,
        resolution_string="42",
        state=QuestionState.RESOLVED,
        lower_bound=0.0,
        upper_bound=100.0,
        open_upper_bound=False,
        open_lower_bound=False,
        api_json={},
    )
    q.actual_resolution_time = resolved_at
    return q


def _make_mc(qid: int, resolved_at: datetime) -> MultipleChoiceQuestion:
    """Build a resolved MC question with a specified actual_resolution_time."""
    q = MultipleChoiceQuestion(
        question_text=f"MC {qid}?",
        id_of_question=qid,
        resolution_string="Option A",
        state=QuestionState.RESOLVED,
        options=["Option A", "Option B", "Option C"],
        api_json={},
    )
    q.actual_resolution_time = resolved_at
    return q


def _make_canceled_binary(qid: int, resolved_at: datetime) -> BinaryQuestion:
    """A binary question whose resolution_string is 'annulled' (extracts as None)."""
    q = BinaryQuestion(
        question_text=f"Canceled {qid}?",
        id_of_question=qid,
        resolution_string="annulled",
        community_prediction_at_access_time=0.5,
        state=QuestionState.RESOLVED,
        api_json={},
    )
    q.actual_resolution_time = resolved_at
    return q


# Two consistent timestamps used in tests
LATE_2026 = datetime(2026, 3, 15)  # safely > "2026-01-01" lower default
EARLY_2026 = datetime(2026, 1, 5)
PRE_WINDOW = datetime(2025, 12, 31)  # before "2026-01-01"


# ---------------------------------------------------------------------------
# fetch_resolved_questions: resolved_before parameter tests
# ---------------------------------------------------------------------------


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_resolved_questions_resolved_before_filters_late_questions(mock_fetch):
    """Questions with actual_resolution_time >= resolved_before are excluded."""
    in_window = _make_binary(101, datetime(2026, 2, 1))
    after_upper = _make_binary(102, datetime(2026, 5, 15))
    on_upper = _make_binary(103, datetime(2026, 5, 1))  # >= upper => excluded

    mock_fetch.return_value = [in_window, after_upper, on_upper]

    result = await fetch_resolved_questions(
        total_questions=10,
        resolved_after="2026-01-01",
        resolved_before="2026-05-01",
    )

    qids = {q.id_of_question for q in result.questions}
    assert qids == {101}


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_resolved_questions_resolved_before_default_none_keeps_all(mock_fetch):
    """When resolved_before is None (default), no upper-bound filtering happens."""
    q1 = _make_binary(201, datetime(2026, 2, 1))
    q2 = _make_binary(202, datetime(2026, 6, 15))  # would be excluded if upper bound set
    q3 = _make_binary(203, datetime(2027, 1, 1))

    mock_fetch.return_value = [q1, q2, q3]

    result = await fetch_resolved_questions(
        total_questions=10,
        resolved_after="2026-01-01",
    )

    qids = {q.id_of_question for q in result.questions}
    assert qids == {201, 202, 203}


# ---------------------------------------------------------------------------
# fetch_resolved_questions_stratified tests
# ---------------------------------------------------------------------------


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_returns_per_type_targets(mock_fetch):
    """Given >=requested per type, the result has exactly the requested counts per type."""
    raw = []
    raw += [_make_binary(1000 + i, LATE_2026) for i in range(30)]
    raw += [_make_mc(2000 + i, LATE_2026) for i in range(30)]
    raw += [_make_numeric(3000 + i, LATE_2026) for i in range(30)]
    mock_fetch.return_value = raw

    result = await fetch_resolved_questions_stratified(
        num_binary=5,
        num_multiple_choice=5,
        num_numeric=5,
        resolved_after="2026-01-01",
    )

    types = [type(q).__name__ for q in result.questions]
    assert types.count("BinaryQuestion") == 5
    assert types.count("MultipleChoiceQuestion") == 5
    assert types.count("NumericQuestion") == 5
    assert len(result.questions) == 15
    assert len(result.ground_truths) == 15


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_undersaturated_takes_what_is_available_and_warns(mock_fetch, caplog):
    """If only 3 binary available but 10 requested: take all 3, emit a warning."""
    raw = []
    raw += [_make_binary(4000 + i, LATE_2026) for i in range(3)]
    raw += [_make_mc(5000 + i, LATE_2026) for i in range(15)]
    raw += [_make_numeric(6000 + i, LATE_2026) for i in range(15)]
    mock_fetch.return_value = raw

    with caplog.at_level(logging.WARNING):
        result = await fetch_resolved_questions_stratified(
            num_binary=10,
            num_multiple_choice=5,
            num_numeric=5,
            resolved_after="2026-01-01",
        )

    types = [type(q).__name__ for q in result.questions]
    assert types.count("BinaryQuestion") == 3
    assert types.count("MultipleChoiceQuestion") == 5
    assert types.count("NumericQuestion") == 5

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("binary" in m.lower() for m in warning_messages)


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_dedupes_across_tournaments(mock_fetch):
    """A qid present in two tournaments appears exactly once; first wins."""
    shared_q_t1 = _make_binary(7777, LATE_2026)
    shared_q_t1.question_text = "FROM_T1"
    shared_q_t2 = _make_binary(7777, LATE_2026)
    shared_q_t2.question_text = "FROM_T2"

    other_q = _make_binary(7778, LATE_2026)

    t1_questions = (
        [shared_q_t1]
        + [_make_mc(8000 + i, LATE_2026) for i in range(5)]
        + [_make_numeric(9000 + i, LATE_2026) for i in range(5)]
    )
    t2_questions = (
        [shared_q_t2, other_q]
        + [_make_mc(8100 + i, LATE_2026) for i in range(5)]
        + [_make_numeric(9100 + i, LATE_2026) for i in range(5)]
    )

    mock_fetch.side_effect = [t1_questions, t2_questions]

    result = await fetch_resolved_questions_stratified(
        num_binary=5,
        num_multiple_choice=2,
        num_numeric=2,
        resolved_after="2026-01-01",
        tournaments=["t1", "t2"],
    )

    binary_qs = [q for q in result.questions if isinstance(q, BinaryQuestion)]
    binary_ids = [q.id_of_question for q in binary_qs]
    # qid 7777 should appear at most once
    assert binary_ids.count(7777) == 1


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_filters_by_resolved_after_window(mock_fetch):
    """Questions with actual_resolution_time < resolved_after are excluded.

    Boundary semantic: half-open interval [resolved_after, resolved_before).
    A question resolving exactly at resolved_after is INCLUDED.
    """
    pre = _make_binary(10001, PRE_WINDOW)
    on = _make_binary(10002, datetime(2026, 1, 1))  # boundary => INCLUDED
    after = _make_binary(10003, datetime(2026, 1, 2))
    mock_fetch.return_value = [
        pre,
        on,
        after,
        *[_make_mc(11000 + i, LATE_2026) for i in range(5)],
        *[_make_numeric(12000 + i, LATE_2026) for i in range(5)],
    ]

    result = await fetch_resolved_questions_stratified(
        num_binary=5,
        num_multiple_choice=3,
        num_numeric=3,
        resolved_after="2026-01-01",
    )

    binary_ids = {q.id_of_question for q in result.questions if isinstance(q, BinaryQuestion)}
    assert binary_ids == {10002, 10003}


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_resolved_after_boundary_inclusive(mock_fetch):
    """A question resolving at exactly resolved_after_dt is INCLUDED.

    Plan spec: actual_resolution_time >= resolved_after (inclusive lower bound).
    """
    boundary = _make_binary(80001, datetime(2026, 1, 1))  # exactly at boundary
    just_before = _make_binary(80002, datetime(2025, 12, 31, 23, 59, 59))  # excluded
    mock_fetch.return_value = [
        boundary,
        just_before,
        *[_make_mc(81000 + i, LATE_2026) for i in range(3)],
        *[_make_numeric(82000 + i, LATE_2026) for i in range(3)],
    ]

    result = await fetch_resolved_questions_stratified(
        num_binary=2,
        num_multiple_choice=2,
        num_numeric=2,
        resolved_after="2026-01-01",
    )

    binary_ids = {q.id_of_question for q in result.questions if isinstance(q, BinaryQuestion)}
    assert 80001 in binary_ids, "Question resolving exactly at resolved_after must be INCLUDED"
    assert 80002 not in binary_ids, "Question resolving before resolved_after must be EXCLUDED"


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_resolved_before_boundary_exclusive(mock_fetch):
    """A question resolving at exactly resolved_before_dt is EXCLUDED.

    Plan spec: actual_resolution_time < resolved_before (exclusive upper bound),
    yielding the half-open interval [resolved_after, resolved_before).
    """
    boundary = _make_binary(85001, datetime(2026, 5, 1))  # exactly at upper bound
    just_before = _make_binary(85002, datetime(2026, 4, 30, 23, 59, 59))  # included
    mock_fetch.return_value = [
        boundary,
        just_before,
        *[_make_mc(86000 + i, datetime(2026, 2, 1)) for i in range(3)],
        *[_make_numeric(87000 + i, datetime(2026, 2, 1)) for i in range(3)],
    ]

    result = await fetch_resolved_questions_stratified(
        num_binary=2,
        num_multiple_choice=2,
        num_numeric=2,
        resolved_after="2026-01-01",
        resolved_before="2026-05-01",
    )

    binary_ids = {q.id_of_question for q in result.questions if isinstance(q, BinaryQuestion)}
    assert 85001 not in binary_ids, "Question resolving exactly at resolved_before must be EXCLUDED"
    assert 85002 in binary_ids, "Question resolving before resolved_before must be INCLUDED"


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_fetch_resolved_questions_resolved_after_boundary_inclusive(mock_fetch):
    """fetch_resolved_questions: boundary at resolved_after is INCLUDED.

    Mirror of test_resolved_after_boundary_inclusive for the non-stratified entry point.
    """
    boundary = _make_binary(90001, datetime(2026, 1, 1))
    just_before = _make_binary(90002, datetime(2025, 12, 31, 23, 59, 59))
    mock_fetch.return_value = [boundary, just_before]

    result = await fetch_resolved_questions(
        total_questions=10,
        resolved_after="2026-01-01",
    )

    qids = {q.id_of_question for q in result.questions}
    assert 90001 in qids, "Question resolving exactly at resolved_after must be INCLUDED"
    assert 90002 not in qids, "Question resolving before resolved_after must be EXCLUDED"


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_filters_by_resolved_before_window(mock_fetch):
    """Questions with actual_resolution_time >= resolved_before are excluded."""
    in_win = _make_binary(20001, datetime(2026, 2, 1))
    on_upper = _make_binary(20002, datetime(2026, 5, 1))  # >= upper => excluded
    after_upper = _make_binary(20003, datetime(2026, 6, 1))

    mock_fetch.return_value = [
        in_win,
        on_upper,
        after_upper,
        *[_make_mc(21000 + i, datetime(2026, 2, 1)) for i in range(5)],
        *[_make_numeric(22000 + i, datetime(2026, 2, 1)) for i in range(5)],
    ]

    result = await fetch_resolved_questions_stratified(
        num_binary=5,
        num_multiple_choice=3,
        num_numeric=3,
        resolved_after="2026-01-01",
        resolved_before="2026-05-01",
    )

    binary_ids = {q.id_of_question for q in result.questions if isinstance(q, BinaryQuestion)}
    assert binary_ids == {20001}


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_skips_canceled(mock_fetch):
    """Questions whose ground_truth extraction returns None are skipped."""
    good = _make_binary(30001, LATE_2026)
    canceled = _make_canceled_binary(30002, LATE_2026)

    mock_fetch.return_value = [
        good,
        canceled,
        *[_make_mc(31000 + i, LATE_2026) for i in range(5)],
        *[_make_numeric(32000 + i, LATE_2026) for i in range(5)],
    ]

    result = await fetch_resolved_questions_stratified(
        num_binary=5,
        num_multiple_choice=3,
        num_numeric=3,
        resolved_after="2026-01-01",
    )

    binary_ids = {q.id_of_question for q in result.questions if isinstance(q, BinaryQuestion)}
    assert binary_ids == {30001}
    # Skipped canceled should be reflected in metadata
    assert result.fetch_metadata["skipped_canceled"] >= 1


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_passes_min_forecasters_to_api_filter(mock_fetch):
    """The min_forecasters value is forwarded to _fetch_with_retries."""
    mock_fetch.return_value = [_make_binary(40000 + i, LATE_2026) for i in range(5)]

    await fetch_resolved_questions_stratified(
        num_binary=2,
        num_multiple_choice=0,
        num_numeric=0,
        resolved_after="2026-01-01",
        min_forecasters=77,
    )

    assert mock_fetch.call_count >= 1
    for call_args in mock_fetch.call_args_list:
        kwargs = call_args.kwargs
        assert kwargs.get("min_forecasters") == 77


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_iterates_all_tournaments(mock_fetch):
    """If two tournaments are passed, _fetch_with_retries is called for each, in order."""
    mock_fetch.side_effect = [
        [_make_binary(50000 + i, LATE_2026) for i in range(3)],
        [_make_binary(51000 + i, LATE_2026) for i in range(3)],
    ]

    await fetch_resolved_questions_stratified(
        num_binary=3,
        num_multiple_choice=0,
        num_numeric=0,
        resolved_after="2026-01-01",
        tournaments=["t1", "t2"],
    )

    assert mock_fetch.call_count == 2
    tournaments_passed = [call.kwargs.get("tournament") for call in mock_fetch.call_args_list]
    assert tournaments_passed == ["t1", "t2"]


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_metadata_populated(mock_fetch):
    """fetch_metadata exposes tournament list, raw counts, and per-type targets/actuals."""
    mock_fetch.side_effect = [
        [_make_binary(60000 + i, LATE_2026) for i in range(4)]
        + [_make_mc(61000 + i, LATE_2026) for i in range(4)]
        + [_make_numeric(62000 + i, LATE_2026) for i in range(4)],
        [_make_binary(70000 + i, LATE_2026) for i in range(2)],
    ]

    result = await fetch_resolved_questions_stratified(
        num_binary=3,
        num_multiple_choice=2,
        num_numeric=2,
        resolved_after="2026-01-01",
        tournaments=["t1", "t2"],
    )

    md = result.fetch_metadata
    assert md["tournaments"] == ["t1", "t2"]
    assert md["resolved_after"] == "2026-01-01"
    assert md["resolved_before"] is None
    assert md["per_tournament_raw_counts"]["t1"] == 12
    assert md["per_tournament_raw_counts"]["t2"] == 2
    assert md["per_type_targets"] == {"binary": 3, "multiple_choice": 2, "numeric": 2}
    assert md["per_type_actual"]["binary"] == 3
    assert md["per_type_actual"]["multiple_choice"] == 2
    assert md["per_type_actual"]["numeric"] == 2


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_default_tournaments_uses_default_constant(mock_fetch):
    """If tournaments is None, defaults to [BACKTEST_DEFAULT_TOURNAMENT]."""
    mock_fetch.return_value = [_make_binary(80000 + i, LATE_2026) for i in range(2)]

    await fetch_resolved_questions_stratified(
        num_binary=1,
        num_multiple_choice=0,
        num_numeric=0,
        resolved_after="2026-01-01",
    )

    assert mock_fetch.call_count == 1
    assert mock_fetch.call_args.kwargs.get("tournament") == BACKTEST_DEFAULT_TOURNAMENT


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_zero_targets_returns_empty_set(mock_fetch):
    """num_binary=num_multiple_choice=num_numeric=0 yields an empty set."""
    mock_fetch.return_value = [_make_binary(90000 + i, LATE_2026) for i in range(5)]

    result = await fetch_resolved_questions_stratified(
        num_binary=0,
        num_multiple_choice=0,
        num_numeric=0,
        resolved_after="2026-01-01",
    )

    assert result.questions == []
    assert result.ground_truths == {}


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_clean_questions_are_sanitized(mock_fetch):
    """Returned questions have state=OPEN, resolution_string/actual_resolution_time cleared."""
    q = _make_binary(99001, LATE_2026)
    q.background_info = "leak this"
    mock_fetch.return_value = [q]

    result = await fetch_resolved_questions_stratified(
        num_binary=1,
        num_multiple_choice=0,
        num_numeric=0,
        resolved_after="2026-01-01",
    )

    assert len(result.questions) == 1
    clean = result.questions[0]
    assert clean.state == QuestionState.OPEN
    assert clean.resolution_string is None
    assert clean.actual_resolution_time is None
    assert clean.background_info == ""
    # Original is untouched (deep copy)
    assert q.state == QuestionState.RESOLVED
    assert q.background_info == "leak this"


@patch("metaculus_bot.backtest.question_prep._fetch_with_retries", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_stratified_overfetches_per_tournament(mock_fetch):
    """Per-tournament call uses count = sum_of_targets * BACKTEST_OVERFETCH_RATIO."""
    mock_fetch.return_value = []

    await fetch_resolved_questions_stratified(
        num_binary=4,
        num_multiple_choice=3,
        num_numeric=2,
        resolved_after="2026-01-01",
        tournaments=["t1", "t2"],
    )

    expected_count = (4 + 3 + 2) * BACKTEST_OVERFETCH_RATIO
    for call in mock_fetch.call_args_list:
        assert call.kwargs.get("count") == expected_count


# ---------------------------------------------------------------------------
# _fetch_with_retries: insufficient-questions retry tests
#
# Real-world bug: when BACKTEST_OVERFETCH_RATIO * total_target exceeds the
# tournament's available resolved-question count, the Metaculus API raises
#   ValueError("Requested number of questions (252) does not match number of
#              questions found (222)")
# The retry loop must parse that count and retry once with the actual value.
# ---------------------------------------------------------------------------


def _patch_api(monkeypatch, side_effect):
    """Replace MetaculusApi.get_questions_matching_filter with an AsyncMock side-effect chain.

    Returns the AsyncMock so the test can inspect call_args_list.
    """
    mock = AsyncMock(side_effect=side_effect)
    monkeypatch.setattr(
        "metaculus_bot.backtest.question_prep.MetaculusApi.get_questions_matching_filter",
        mock,
    )
    return mock


class TestFetchWithRetriesInsufficientQuestions:
    """Coverage for the count-mismatch retry path in _fetch_with_retries."""

    @pytest.mark.asyncio
    async def test_insufficient_questions_retries_with_actual_count(self, monkeypatch, caplog):
        """First call (252) raises with parseable mismatch; retry with 222 succeeds."""
        questions_222 = [_make_binary(700_000 + i, LATE_2026) for i in range(222)]
        mismatch = ValueError("Requested number of questions (252) does not match number of questions found (222)")
        mock = _patch_api(monkeypatch, side_effect=[mismatch, questions_222])

        with caplog.at_level(logging.WARNING):
            result = await _fetch_with_retries(tournament="spring-aib-2026", count=252, min_forecasters=10)

        num_questions_per_call = [c.kwargs["num_questions"] for c in mock.call_args_list]
        assert num_questions_per_call == [252, 222]
        assert len(result) == 222
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("222" in m and "252" in m for m in warning_messages), (
            f"Expected a WARNING mentioning both counts; got: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_insufficient_questions_only_retries_once(self, monkeypatch):
        """If the smaller-count retry also says 'mismatch', do not infinite-loop."""
        mock = _patch_api(
            monkeypatch,
            side_effect=[
                ValueError("Requested number of questions (252) does not match number of questions found (222)"),
                ValueError("Requested number of questions (222) does not match number of questions found (210)"),
            ],
        )

        with pytest.raises(RuntimeError):
            await _fetch_with_retries(tournament="spring-aib-2026", count=252, min_forecasters=10)

        # Exactly two API calls: original (252) + insufficient-retry (222). The third
        # would-be retry would mean we entered the count-adjustment branch a second time.
        num_questions_per_call = [c.kwargs["num_questions"] for c in mock.call_args_list]
        assert num_questions_per_call == [252, 222]

    @pytest.mark.asyncio
    async def test_zero_actual_count_does_not_retry(self, monkeypatch):
        """API saying 'found (0)' should not trigger a count=0 retry."""
        mock = _patch_api(
            monkeypatch,
            side_effect=ValueError("Requested number of questions (252) does not match number of questions found (0)"),
        )

        with pytest.raises(RuntimeError):
            await _fetch_with_retries(tournament="spring-aib-2026", count=252, min_forecasters=10)

        # Only the original call - no retry with count=0.
        assert mock.await_count == 1

    @pytest.mark.asyncio
    async def test_actual_count_equal_to_requested_does_not_loop(self, monkeypatch):
        """Pathological: API says 'found (252)' when count=252. Don't loop forever."""
        mock = _patch_api(
            monkeypatch,
            side_effect=ValueError(
                "Requested number of questions (252) does not match number of questions found (252)"
            ),
        )

        with pytest.raises(RuntimeError):
            await _fetch_with_retries(tournament="spring-aib-2026", count=252, min_forecasters=10)

        # Only one call - actual_count is not strictly less than count, so no retry.
        assert mock.await_count == 1

    @pytest.mark.asyncio
    async def test_unparseable_value_error_falls_through_to_existing_logic(self, monkeypatch):
        """Non-mismatch ValueErrors hit the regular retry/raise path (not retryable here)."""
        mock = _patch_api(monkeypatch, side_effect=ValueError("totally different error format"))

        with pytest.raises(RuntimeError):
            await _fetch_with_retries(tournament="spring-aib-2026", count=252, min_forecasters=10)

        # Single call - the message has no mismatch token and isn't network-retryable.
        assert mock.await_count == 1
