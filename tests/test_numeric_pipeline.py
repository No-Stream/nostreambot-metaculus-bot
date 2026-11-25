import logging
from itertools import pairwise
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from scipy.stats import norm

from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.validation import detect_unit_mismatch
from tests.pipeline_test_helpers import assert_server_accepts_cdf


def _build_question(**overrides) -> NumericQuestion:
    base_kwargs: dict[str, Any] = {
        "id_of_question": 1,
        "id_of_post": 1,
        "page_url": "https://example.com/q/1",
        "question_text": "Test numeric question",
        "background_info": "",
        "resolution_criteria": "",
        "fine_print": "",
        "published_time": None,
        "close_time": None,
        "lower_bound": 0.0,
        "upper_bound": 100.0,
        "open_lower_bound": False,
        "open_upper_bound": False,
        "unit_of_measure": "units",
        "zero_point": 0.0,
    }
    base_kwargs.update(overrides)
    return NumericQuestion(**base_kwargs)


def test_sanitize_percentiles_orders_and_jitters(monkeypatch):
    question = _build_question(cdf_size=50)

    raw_values = [
        (0.8, 60.0),
        (0.025, 5.0),
        (0.95, 90.0),
        (0.5, 40.0),
        (0.2, 20.0),
        (0.4, 35.0),
        (0.6, 35.0),  # duplicate to trigger jitter
        (0.1, 10.0),
        (0.05, 7.5),
        (0.975, 95.0),
        (0.9, 80.0),
        (0.01, 2.5),
        (0.99, 97.5),
    ]
    percentiles = [Percentile(percentile=p, value=v) for p, v in raw_values]

    sanitized, zero_point = sanitize_percentiles(percentiles, question)

    # Ensure sorted by percentile
    sorted_percentiles = sorted([p for p, _ in raw_values])
    assert [p.percentile for p in sanitized] == sorted_percentiles

    # Values should be strictly increasing after jitter/clamp
    sanitized_values = [p.value for p in sanitized]
    assert all(b > a for a, b in pairwise(sanitized_values))

    # Discrete question (cdf_size != default) should force zero_point to None
    assert zero_point is None


class TestPointMassDeclaration:
    """A point-mass declaration must not be published as a distribution (H2, 2026-08-25).

    A model declaring the SAME value at all 13 percentiles used to be spread into
    a 12-unit-wide forecast by the cluster spreader (``COUNT_LIKE_DELTA_MULTIPLIER``
    is a full unit per position, and with no unclustered neighbour nothing
    compressed it). That fabricated span was ALSO what made the degenerate
    declaration pass ``detect_unit_mismatch``'s span-ratio test — the guard that
    exists to withhold exactly this input. Now sanitisation adds only the
    minimum separation the CDF format needs, so the guard sees the model's own
    (zero) span and the forecaster is withheld.
    """

    _POINT_MASS = 42.0
    _LABELS: ClassVar[list[float]] = [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99]

    def _declared(self) -> list[Percentile]:
        return [Percentile(percentile=p, value=self._POINT_MASS) for p in self._LABELS]

    def test_no_width_is_invented(self):
        question = _build_question(zero_point=None)
        sanitized, _zero_point = sanitize_percentiles(self._declared(), question)

        values = [float(p.value) for p in sanitized]
        # Strictly increasing (the CDF needs it) but only by the ordering epsilon:
        # max(MIN_BOUNDARY_DISTANCE * range, STRICT_ORDERING_EPSILON) per step.
        assert all(b > a for a, b in pairwise(values))
        span = max(values) - min(values)
        assert span < 1e-4, f"sanitisation invented {span} of width from a point mass"

    def test_unit_mismatch_guard_withholds_it(self):
        question = _build_question(zero_point=None)
        sanitized, _zero_point = sanitize_percentiles(self._declared(), question)

        mismatch, reason = detect_unit_mismatch(sanitized, question)
        assert mismatch is True, "a zero-width declaration must be withheld, not published"
        assert "span" in reason

    def test_marker_is_emitted(self, caplog):
        question = _build_question(zero_point=None)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric.pipeline"):
            sanitize_percentiles(self._declared(), question, model_name="some/model")

        markers = [r.getMessage() for r in caplog.records if "NUMERIC_DEGENERATE_DECLARATION:" in r.getMessage()]
        assert len(markers) == 1
        assert "question=1" in markers[0]
        assert "model=some/model" in markers[0]
        assert "n_unique=1" in markers[0]
        assert "spread_applied=false" in markers[0]

    def test_marker_names_unknown_model_when_caller_has_none(self, caplog):
        question = _build_question(zero_point=None)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric.pipeline"):
            sanitize_percentiles(self._declared(), question)
        markers = [r.getMessage() for r in caplog.records if "NUMERIC_DEGENERATE_DECLARATION:" in r.getMessage()]
        assert len(markers) == 1
        assert "model=unknown" in markers[0]

    def test_partial_plateau_still_spreads_and_publishes(self, caplog):
        """The spreader's real job is preserved: a plateau INSIDE a wider
        declaration is separated, no marker fires, and the guard passes it."""
        question = _build_question(lower_bound=0.0, upper_bound=20.0, zero_point=None)
        values = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        declared = [Percentile(percentile=p, value=v) for p, v in zip(self._LABELS, values, strict=False)]

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric.pipeline"):
            sanitized, _zero_point = sanitize_percentiles(declared, question)

        assert not [r for r in caplog.records if "NUMERIC_DEGENERATE_DECLARATION:" in r.getMessage()]
        sanitized_values = [float(p.value) for p in sanitized]
        assert all(b > a for a, b in pairwise(sanitized_values))
        assert detect_unit_mismatch(sanitized, question) == (False, "")


def test_build_numeric_distribution_fallback(monkeypatch):
    question = _build_question()
    sanitized = [
        Percentile(percentile=p, value=float(i * 10))
        for i, p in enumerate([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99], start=1)
    ]

    failing = MagicMock(side_effect=RuntimeError("boom"))
    fallback = MagicMock(return_value="fallback")
    validate = MagicMock()

    monkeypatch.setattr(
        "metaculus_bot.numeric.pipeline.generate_pchip_cdf_with_smoothing",
        failing,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric.pipeline.create_fallback_numeric_distribution",
        fallback,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric.pipeline.validate_cdf_construction",
        validate,
    )

    result = build_numeric_distribution(sanitized, question, zero_point=None, model_name="some/forecaster")

    assert result == "fallback"
    # model_name rides through to the fallback builder too: its get_cdf() runs its own
    # safe_cdf_bounds pass, so a CDF_MAXSTEP_CLIP from there must name the forecaster.
    fallback.assert_called_once_with(sanitized, question, None, model_name="some/forecaster")
    validate.assert_called_once_with("fallback", question)


def test_build_numeric_distribution_success(monkeypatch):
    question = _build_question()
    sanitized = [
        Percentile(percentile=p, value=float(i * 10))
        for i, p in enumerate([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99], start=1)
    ]

    mock_cdf = [i / 10 for i in range(len(sanitized))]
    generator = MagicMock(return_value=(mock_cdf, False, False))
    create_dist = MagicMock(return_value="pchip")
    validate = MagicMock()

    monkeypatch.setattr(
        "metaculus_bot.numeric.pipeline.generate_pchip_cdf_with_smoothing",
        generator,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric.pipeline.create_pchip_numeric_distribution",
        create_dist,
    )
    monkeypatch.setattr(
        "metaculus_bot.numeric.pipeline.validate_cdf_construction",
        validate,
    )

    result = build_numeric_distribution(sanitized, question, zero_point=None, model_name="some/forecaster")

    assert result == "pchip"
    generator.assert_called_once_with(
        sanitized,
        question,
        None,
        model_name="some/forecaster",
        p_below_lower_bound=None,
        p_above_upper_bound=None,
        tail_diagnostics=None,
    )
    create_dist.assert_called_once_with(mock_cdf, sanitized, question, None)
    validate.assert_called_once_with("pchip", question)


def test_build_numeric_distribution_discrete_resamples_to_cdf_size():
    """Discrete question with cdf_size=9 produces a 9-point CDF, not 201."""

    question = _build_question(
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        cdf_size=9,
    )

    raw_percentiles = [
        Percentile(percentile=0.01, value=-0.3),
        Percentile(percentile=0.025, value=0.0),
        Percentile(percentile=0.05, value=0.5),
        Percentile(percentile=0.1, value=1.0),
        Percentile(percentile=0.2, value=1.5),
        Percentile(percentile=0.4, value=3.0),
        Percentile(percentile=0.5, value=3.5),
        Percentile(percentile=0.6, value=4.0),
        Percentile(percentile=0.8, value=5.5),
        Percentile(percentile=0.9, value=6.5),
        Percentile(percentile=0.95, value=7.0),
        Percentile(percentile=0.975, value=7.2),
        Percentile(percentile=0.99, value=7.4),
    ]

    sanitized, zero_point = sanitize_percentiles(raw_percentiles, question)
    distribution = build_numeric_distribution(sanitized, question, zero_point)

    cdf = distribution.cdf
    assert len(cdf) == 9, f"Expected 9-point CDF for discrete question, got {len(cdf)}"

    probs = np.array([p.percentile for p in cdf], dtype=float)
    diffs = np.diff(probs)
    assert np.all(diffs > 0), "CDF must be monotonically increasing"

    assert abs(probs[0] - 0.0) < 1e-9, f"Closed lower bound CDF should start at 0.0, got {probs[0]}"
    assert abs(probs[-1] - 1.0) < 1e-9, f"Closed upper bound CDF should end at 1.0, got {probs[-1]}"


class TestFallbackDistributionOnClosedBounds:
    """The fallback's ``get_cdf()`` used to hand forecasting-tools' raw builder output back
    unchanged when BOTH bounds were closed, skipping the step enforcement the open-bound branch
    ran. A tight declaration then carried a 0.297 bin against the server's 0.2 cap and the
    lone-survivor path submitted it for an HTTP 400. Both bounds closed is the modal Mantic
    date shape and common on Metaculus percentage questions."""

    @staticmethod
    def _tight_declaration() -> list[Percentile]:
        return [Percentile(percentile=p, value=50.0 + 0.6 * norm.ppf(p)) for p in STANDARD_PERCENTILES]

    def test_a_tight_closed_closed_declaration_meets_the_server_rules(self, monkeypatch):
        question = _build_question(zero_point=None, open_lower_bound=False, open_upper_bound=False)
        sanitized, zero_point = sanitize_percentiles(self._tight_declaration(), question)

        def _pchip_fails(*_args, **_kwargs):
            raise ValueError("forced PCHIP failure")

        monkeypatch.setattr("metaculus_bot.numeric.pipeline.generate_pchip_cdf_with_smoothing", _pchip_fails)

        prediction = build_numeric_distribution(sanitized, question, zero_point)

        assert type(prediction).__name__ == "BoundSafeNumericDistribution"
        heights = np.asarray([p.percentile for p in prediction.get_cdf()], dtype=float)
        assert_server_accepts_cdf(heights, cdf_size=201, open_lower=False, open_upper=False)
        assert heights[0] == 0.0
        assert heights[-1] == 1.0

    def test_the_open_bound_path_is_unchanged(self, monkeypatch):
        question = _build_question(zero_point=None, open_lower_bound=True, open_upper_bound=True)
        sanitized, zero_point = sanitize_percentiles(self._tight_declaration(), question)

        def _pchip_fails(*_args, **_kwargs):
            raise ValueError("forced PCHIP failure")

        monkeypatch.setattr("metaculus_bot.numeric.pipeline.generate_pchip_cdf_with_smoothing", _pchip_fails)

        prediction = build_numeric_distribution(sanitized, question, zero_point)

        heights = np.asarray([p.percentile for p in prediction.get_cdf()], dtype=float)
        assert_server_accepts_cdf(heights, cdf_size=201, open_lower=True, open_upper=True)
