"""Pin the backtest question universe against the forecasting-tools 0.2.54 -> 0.2.92 upgrade.

Seam (Workstream W5 of the retired design log (git history)): ``question_prep._fetch_with_retries`` builds an
``ApiFilter`` for the resolved-question backtest fetch WITHOUT setting
``allowed_types`` — it relies on the upstream default. On 0.2.54 that default is
``["binary", "numeric", "multiple_choice", "date", "discrete"]``, which excludes
the conditional question type, so the backtest universe never contains
conditionals. At ft 0.2.92 the ``ApiFilter`` default is documented to admit
conditional questions (a new type) with a reshaped ``api_json['question']``. If
our filter keeps riding the default, the backtest universe silently widens to
include conditionals on upgrade — and the extraction path would then have to
deal with a type it was never built for.

``TestBacktestFilterExcludesConditionals`` drives the real filter construction
(the fetch itself mocked at ``MetaculusApi.get_questions_matching_filter`` so no
network runs) and captures the ``ApiFilter`` question_prep actually hands to the
fetch, then pins that its allowed types exclude conditionals. Green on 0.2.54
(default excludes conditionals); goes red at 0.2.92 (default admits them) until
W5 pins ``allowed_types`` to the supported set explicitly.

``TestGroundTruthExtractionRejectsUnsupportedTypes`` pins the second line of
defense: even if an unsupported type leaks past the filter, the ground-truth
extraction path drops it rather than passing it through.
"""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from forecasting_tools import ApiFilter
from forecasting_tools.data_models.questions import DateQuestion

from metaculus_bot.backtest.question_prep import _extract_ground_truth, _fetch_with_retries

# The only three question types the backtest scoring path (_extract_ground_truth,
# and the stratified by-type dispatch) can handle. Everything else must be kept
# out of the universe — by the filter first, and by extraction as a backstop.
SUPPORTED_QUESTION_TYPES = ("binary", "multiple_choice", "numeric")


async def _capture_constructed_filter(monkeypatch: pytest.MonkeyPatch) -> ApiFilter:
    """Run _fetch_with_retries with the network boundary mocked, return the real ApiFilter it built.

    Patches ``MetaculusApi.get_questions_matching_filter`` (question_prep's fetch
    chokepoint) with an AsyncMock returning a non-empty page — non-empty dodges
    the "API returned 0 resolved questions" RuntimeError so the fetch completes
    on the first attempt. The filter question_prep constructed is captured as the
    mock's first positional arg, so this pins the ACTUAL filter, not a re-derived
    copy that could drift from the source.
    """
    mock = AsyncMock(return_value=[object()])
    monkeypatch.setattr(
        "metaculus_bot.backtest.question_prep.MetaculusApi.get_questions_matching_filter",
        mock,
    )

    await _fetch_with_retries(tournament="ft-pin-test", count=10, min_forecasters=5)

    assert mock.await_count == 1, "the fetch must call the (mocked) filter query exactly once"
    call = mock.call_args
    assert call is not None
    api_filter = call.args[0]
    assert isinstance(api_filter, ApiFilter), "first positional arg to the fetch must be the ApiFilter"
    return api_filter


class TestBacktestFilterExcludesConditionals:
    """W5: the backtest question filter must keep conditional questions out of the universe."""

    async def test_captures_question_preps_real_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuity: prove we captured question_prep's actual filter, not a bare ApiFilter().

        If any of these drift, the conditional-exclusion assertion below is
        testing the wrong object.
        """
        api_filter = await _capture_constructed_filter(monkeypatch)
        assert api_filter.allowed_statuses == ["resolved"]
        assert api_filter.allowed_tournaments == ["ft-pin-test"]
        assert api_filter.num_forecasters_gte == 5

    async def test_constructed_filter_excludes_conditional_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The filter's allowed question types must not admit conditionals on 0.2.54."""
        api_filter = await _capture_constructed_filter(monkeypatch)
        allowed = api_filter.allowed_types

        # An empty/None allowed_types disables type filtering entirely (both the
        # server-side ``forecast_type`` param and the local type filter are
        # skipped), which would ADMIT conditionals. Guard that first so the
        # membership check below is meaningful, not vacuously true on an empty list.
        assert allowed, "allowed_types must be non-empty or type filtering is off and conditionals leak in"

        # At ft 0.2.92 defaults admit conditional questions; W5 must pin allowed_types explicitly — this test goes red then until it does.
        assert "conditional" not in allowed, (
            f"backtest filter must exclude conditional questions; allowed_types={allowed}"
        )

    async def test_constructed_filter_admits_exactly_the_three_supported_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allowed set must be EXACTLY the three types the backtest can score.

        W5 pinned ``allowed_types`` to ``["binary", "multiple_choice", "numeric"]``,
        so this is now exact equality (order-independent) rather than the
        pre-pin subset check: the 0.2.54 default carried extra ``date``/``discrete``
        entries, but the explicit pin drops them, and ``discrete`` is folded under
        numeric (DiscreteQuestion subclasses NumericQuestion) so it needs no slot.
        """
        api_filter = await _capture_constructed_filter(monkeypatch)
        allowed = api_filter.allowed_types
        assert set(allowed) == set(SUPPORTED_QUESTION_TYPES), (
            f"allowed_types must be exactly {sorted(SUPPORTED_QUESTION_TYPES)}; got {allowed}"
        )


class TestGroundTruthExtractionRejectsUnsupportedTypes:
    """Backstop: even if the filter leaks one, extraction must drop non-supported types.

    ``_extract_ground_truth`` dispatches on the concrete question class and only
    handles Binary/Numeric/MultipleChoice; anything else hits an ``else`` branch
    that logs an ``unsupported question type`` warning and returns ``None``, so
    the question is skipped rather than passed downstream. ``DateQuestion`` is the
    clean unsupported example (``DiscreteQuestion`` subclasses ``NumericQuestion``
    and is deliberately treated as numeric).
    """

    def test_resolved_date_question_is_rejected_by_type_dispatch(self, caplog: pytest.LogCaptureFixture) -> None:
        date_question = DateQuestion(
            question_text="When will X happen?",
            id_of_question=999,
            resolution_string="2026-06-01T00:00:00Z",
            lower_bound=datetime(2020, 1, 1),
            upper_bound=datetime(2030, 1, 1),
            open_upper_bound=False,
            open_lower_bound=False,
            api_json={},
        )
        # Non-vacuity: the question IS resolved (typed_resolution parses to a
        # datetime), so a None ground truth can only come from the type dispatch
        # rejecting DateQuestion — not from the early null-resolution return.
        assert date_question.typed_resolution is not None

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.backtest.question_prep"):
            ground_truth = _extract_ground_truth(date_question)

        assert ground_truth is None, "an unsupported (date) question type must be dropped, not scored"
        assert any(
            "unsupported question type" in record.getMessage() and "DateQuestion" in record.getMessage()
            for record in caplog.records
        ), "the type-dispatch else branch must log an unsupported-type warning"
