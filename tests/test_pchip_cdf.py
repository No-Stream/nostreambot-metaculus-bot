"""
Tests for PCHIP-based CDF construction.

Based on the panchul implementation with comprehensive validation.
"""

from itertools import pairwise
from typing import ClassVar, cast

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP
from metaculus_bot.numeric.pchip_cdf import (
    _rebuild_with_min_steps,
    enforce_min_steps,
    enforce_strict_increasing,
    generate_pchip_cdf,
    percentiles_to_pchip_format,
    safe_cdf_bounds,
)


class TestEnforceMinSteps:
    """Test the panchul-style forward+backward sweep that keeps adjacent
    CDF points at least ``min_step`` apart while respecting upper/lower caps.

    Used by both the PCHIP pipeline and the mixture-of-normals CDF builder."""

    def test_already_compliant_input_unchanged(self):
        y = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        out = enforce_min_steps(y, min_step=0.05)
        np.testing.assert_allclose(out, y)

    def test_below_min_step_lifts_via_forward_pass(self):
        y = np.array([0.0, 0.001, 0.002, 0.003])
        out = enforce_min_steps(y, min_step=0.1)
        for i in range(1, len(out)):
            assert out[i] - out[i - 1] >= 0.1 - 1e-12

    def test_upper_cap_pins_late_values_and_back_fills(self):
        # Mixture-CDF case: tight tails saturate early; backward pass back-fills.
        y = np.array([0.0, 0.5, 0.9, 0.99, 0.999, 1.0, 1.0, 1.0])
        out = enforce_min_steps(y, min_step=0.01, upper_cap=1.0)
        for i in range(1, len(out)):
            assert out[i] - out[i - 1] >= 0.01 - 1e-12
        assert out[-1] <= 1.0 + 1e-12

    def test_open_upper_cap_respected(self):
        # Open-upper question: cap at 0.999 instead of 1.0.
        y = np.array([0.0, 0.5, 0.99, 0.995, 0.999])
        out = enforce_min_steps(y, min_step=0.001, upper_cap=0.999)
        assert out[-1] <= 0.999 + 1e-12

    def test_lower_cap_pulls_negatives_up(self):
        # Open-lower question: backward pass could drop a value below 0.001.
        y = np.array([0.0005, 0.0006, 0.0007, 0.5, 1.0])
        out = enforce_min_steps(y, min_step=0.0005, upper_cap=1.0, lower_cap=0.001)
        assert out[0] >= 0.001 - 1e-12


class TestPercentilesToPchipFormat:
    """Test conversion from forecasting-tools format to PCHIP input format."""

    def test_basic_conversion(self):
        """Test basic percentile conversion."""
        percentiles = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.95, value=90.0),
        ]

        result = percentiles_to_pchip_format(percentiles)
        expected = {5.0: 10.0, 50.0: 50.0, 95.0: 90.0}

        assert result == expected

    def test_eight_percentile_conversion(self):
        """Test conversion of our standard 8-percentile set."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.40, value=40.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        result = percentiles_to_pchip_format(percentiles)
        expected = {
            5.0: 5.0,
            10.0: 10.0,
            20.0: 20.0,
            40.0: 40.0,
            60.0: 60.0,
            80.0: 80.0,
            90.0: 90.0,
            95.0: 95.0,
        }

        assert result == expected


class TestEnforceStrictIncreasing:
    """Test enforcement of strictly increasing values."""

    def test_already_increasing(self):
        """Test with already strictly increasing values."""
        data = {10.0: 1.0, 20.0: 2.0, 30.0: 3.0}
        result = enforce_strict_increasing(data)
        assert result == data

    def test_duplicate_values(self):
        """Test with duplicate values that need jittering."""
        data = {10.0: 5.0, 20.0: 5.0, 30.0: 5.0}  # All same value
        result = enforce_strict_increasing(data)

        # Values should be strictly increasing
        values = [result[k] for k in sorted(result.keys())]
        assert all(a < b for a, b in pairwise(values))

    def test_mixed_duplicates(self):
        """Test with some duplicates and some valid values."""
        data = {10.0: 1.0, 20.0: 3.0, 30.0: 3.0, 40.0: 5.0}
        result = enforce_strict_increasing(data)

        values = [result[k] for k in sorted(result.keys())]
        assert all(a < b for a, b in pairwise(values))
        assert result[10.0] == 1.0  # First value unchanged
        assert result[40.0] == 5.0  # Last value unchanged


class TestSafeCdfBounds:
    """Test CDF boundary constraint enforcement."""

    def test_open_bounds_clamping(self):
        """Test open bounds are clamped to [0.001, 0.999]."""
        cdf = np.array([0.0, 0.1, 0.6, 0.82, 0.95, 1.0])
        result = safe_cdf_bounds(cdf, open_lower=True, open_upper=True)

        assert result[0] >= 0.001
        assert result[-1] <= 0.999

    def test_closed_bounds_preserved(self):
        """Test closed bounds preserve [0, 1] values."""
        cdf = np.array([0.0, 0.2, 0.55, 0.8, 0.95, 1.0])
        result = safe_cdf_bounds(cdf, open_lower=False, open_upper=False)

        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_max_jump_enforcement(self):
        """Test that steps > NUM_MAX_STEP are redistributed."""
        cdf = np.array([0.0, 0.1, 0.6, 0.82, 0.95, 1.0])  # Includes steps above NUM_MAX_STEP
        result = safe_cdf_bounds(cdf, open_lower=False, open_upper=False)

        steps = np.diff(result)
        assert all(step <= NUM_MAX_STEP + 1e-6 for step in steps), f"Steps: {steps}"
        assert result[-1] - result[0] == pytest.approx(1.0, rel=1e-9)

    def test_iterative_max_jump_enforcement(self):
        """Ensure multiple violations are handled iteratively."""
        cdf = np.array([0.0, 0.35, 0.6, 0.82, 0.96, 1.0])  # Multiple steps above limit
        result = safe_cdf_bounds(cdf, open_lower=False, open_upper=False)

        steps = np.diff(result)
        assert all(step <= NUM_MAX_STEP + 1e-6 for step in steps), f"Steps: {steps}"
        assert result[-1] - result[0] == pytest.approx(1.0, rel=1e-9)

    def test_over_cap_excess_lands_next_to_the_spike(self):
        """The clipped excess is packed nearest-first, not spread across the whole grid.

        Regression for q45065: a spike the platform cap cannot hold used to be handed out
        in proportion to every bin's SLACK, and on a 201-point grid where every other bin
        is near-empty that is a near-uniform spread — 88% of the displaced mass landed 21+
        bins away. The cap is the platform's; where the excess goes is ours.
        """
        n_bins = 200
        background = 0.3 / (n_bins - 1)
        steps = np.full(n_bins, background)
        steps[100] = 0.7
        cdf = np.concatenate([[0.0], np.cumsum(steps)])

        result = safe_cdf_bounds(cdf, open_lower=False, open_upper=False)
        out = np.diff(result)

        assert float(out.max()) <= NUM_MAX_STEP + 1e-12
        assert result[-1] - result[0] == pytest.approx(1.0, rel=1e-9)
        # Rings 1 and 2 absorb everything: 3 bins at the cap plus a split remainder.
        assert float(out[99:102].sum()) == pytest.approx(3 * NUM_MAX_STEP)
        assert float(out[98:103].sum()) == pytest.approx(0.7 + 4 * background, abs=1e-9)
        # Symmetric declaration, symmetric remainder — no one-sided tail at the last ring.
        assert float(out[98]) == pytest.approx(float(out[102]))
        # Untouched far bins are the anti-smear assertion: the retired policy lifted every
        # one of them by roughly excess/n_bins.
        for far in (0, 50, 150, n_bins - 1):
            assert float(out[far]) == pytest.approx(background)

    def test_grid_too_short_to_hold_the_mass_raises(self):
        """Four bins at a 0.2 cap hold 0.8; a full unit of mass has nowhere to go.

        Unreachable on a real submission (the server's own cap scales so capacity is always
        >= 1.0), but the repair must fail loudly rather than silently drop the remainder.
        """
        cdf = np.array([0.0, 0.9, 0.95, 0.98, 1.0])
        with pytest.raises(RuntimeError, match="Failed to redistribute CDF probability mass"):
            safe_cdf_bounds(cdf, open_lower=False, open_upper=False, min_step=1e-4, max_step=0.2)

    def test_open_lower_flat_prefix_min_step_reenforced(self):
        """Regression: pinning cdf[0] to 0.001 + cummax flattens any sub-0.001
        prefix into 0-step bins, violating the 5e-5 server min-step. The
        ensemble ramp (7.5e-7/bin) can't rescue that, so safe_cdf_bounds must
        re-enforce the min-step itself after the pin+cummax."""
        ramp = np.linspace(0.0015, 0.999, 197)
        cdf = np.concatenate([[0.0002, 0.0004, 0.0006, 0.0008], ramp])
        result = safe_cdf_bounds(cdf, open_lower=True, open_upper=True)

        steps = np.diff(result)
        assert float(steps.min()) >= NUM_MIN_PROB_STEP - 1e-12, f"min-step violation: {float(steps.min())}"
        assert result[0] >= 0.001
        assert result[-1] <= 0.999 + 1e-12
        assert np.all(steps >= 0.0), "CDF must be monotone non-decreasing"


class TestGeneratePchipCdf:
    """Test the main PCHIP CDF generation function."""

    def test_basic_generation(self):
        """Test basic CDF generation with simple percentiles."""
        percentiles = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0  # Closed lower bound
        assert cdf[-1] == 1.0  # Closed upper bound
        assert all(a <= b for a, b in pairwise(cdf))  # Monotonic

    def test_open_bounds_generation(self):
        """Test CDF generation with open bounds."""
        percentiles = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] >= 0.001  # Open lower bound
        assert cdf[-1] <= 0.999  # Open upper bound
        assert all(a <= b for a, b in pairwise(cdf))  # Monotonic

    def test_minimum_step_enforcement(self):
        """Test that minimum step size is enforced."""
        percentiles = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}
        min_step = 5e-5

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=min_step,
        )

        steps = np.diff(cdf)
        assert all(step >= min_step - 1e-10 for step in steps), f"Min step violated: {np.min(steps)}"

    def test_eight_percentile_realistic(self):
        """Test with realistic 8-percentile data."""
        percentiles = {
            5.0: 100.0,
            10.0: 110.0,
            20.0: 120.0,
            40.0: 140.0,
            60.0: 160.0,
            80.0: 180.0,
            90.0: 190.0,
            95.0: 195.0,
        }

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=200.0,
            lower_bound=50.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0

        # Verify monotonicity
        assert all(a <= b for a, b in pairwise(cdf))

        # Verify minimum step
        steps = np.diff(cdf)
        assert all(step >= 5e-5 - 1e-10 for step in steps)

    def test_log_space_transform(self):
        """Test that log-space transform is applied when appropriate."""
        percentiles = {10.0: 1.0, 50.0: 10.0, 90.0: 100.0}  # All positive, good for log

        # Should work without error and produce smooth results
        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=1000.0,
            lower_bound=0.1,
            zero_point=None,  # No zero_point should enable log transform
        )

        assert len(cdf) == 201
        assert all(a <= b for a, b in pairwise(cdf))

    def test_zero_point_handling(self):
        """Test that zero_point prevents log transform."""
        percentiles = {10.0: 1.0, 50.0: 10.0, 90.0: 100.0}

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=1000.0,
            lower_bound=0.1,
            zero_point=0.0,  # Should prevent log transform
        )

        assert len(cdf) == 201
        assert all(a <= b for a, b in pairwise(cdf))

    def test_duplicate_values_handling(self):
        """Test handling of duplicate percentile values."""
        percentiles = {10.0: 5.0, 50.0: 5.0, 90.0: 5.0}  # All same value

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert all(a <= b for a, b in pairwise(cdf))

    def test_non_finite_value_raises_instead_of_dropping_the_anchor(self):
        """A non-finite VALUE fails the build instead of being silently skipped.

        The old ``continue`` built a CDF from the surviving points while
        ``declared_percentiles`` still advertised the full set — and NaN was
        reachable end-to-end (``json.loads`` accepts a bare ``NaN``, and the
        strictly-increasing check ``value <= prev`` is False for it). Every
        caller handles the ValueError: ``build_numeric_distribution`` falls
        back to forecasting-tools' builder, which re-validates.
        """
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError, match="non-finite value"):
                generate_pchip_cdf(
                    percentile_values={10.0: 1.0, 50.0: bad_value, 90.0: 9.0},
                    open_upper_bound=False,
                    open_lower_bound=False,
                    upper_bound=10.0,
                    lower_bound=0.0,
                )

    def test_non_numeric_value_raises(self):
        percentiles = cast(dict[int | float, float], {10.0: 1.0, 50.0: "five-ish", 90.0: 9.0})
        with pytest.raises(ValueError, match="non-numeric value"):
            generate_pchip_cdf(
                percentile_values=percentiles,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
            )

    def test_non_numeric_label_raises(self):
        percentiles = cast(dict[int | float, float], {10.0: 1.0, "p50": 5.0, 90.0: 9.0})
        with pytest.raises(ValueError, match="non-numeric percentile label"):
            generate_pchip_cdf(
                percentile_values=percentiles,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
            )

    def test_boundary_labels_stay_silently_filtered(self):
        """0.0 / 100.0 KEYS are dropped by design, never raised on — values != keys.

        A caller that hands a whole CDF over as prob*100 labels legitimately includes
        exactly 0.0 and 100.0; the filter drops them before the boundary points are
        re-added from the bound semantics, so the build with the boundary labels
        present must be byte-identical to the build without them.
        """
        base: dict[int | float, float] = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}
        with_boundaries: dict[int | float, float] = {0.0: 0.0, **base, 100.0: 10.0}

        cdf_base, _ = generate_pchip_cdf(
            percentile_values=base,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
        )
        cdf_with, _ = generate_pchip_cdf(
            percentile_values=with_boundaries,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
        )

        np.testing.assert_allclose(cdf_with, cdf_base)

    def test_error_handling_empty_percentiles(self):
        """Test error handling for empty percentiles."""
        with pytest.raises(ValueError, match="Empty percentile values dictionary"):
            generate_pchip_cdf(
                percentile_values={},
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
            )

    def test_error_handling_invalid_bounds(self):
        """Test error handling for invalid bounds."""
        percentiles = {50.0: 5.0}

        with pytest.raises(ValueError, match=r"Upper bound.*must be greater than lower bound"):
            generate_pchip_cdf(
                percentile_values=percentiles,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=0.0,
                lower_bound=10.0,  # Invalid: lower > upper
            )

    def test_error_handling_insufficient_percentiles(self):
        """Test error handling for too few percentiles."""
        percentiles = {50.0: 5.0}  # Only 1 percentile

        with pytest.raises(ValueError, match="Need at least 2 valid percentile points"):
            generate_pchip_cdf(
                percentile_values=percentiles,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
            )

    def test_realistic_numeric_distribution(self):
        """Test with realistic numeric distribution (right-skewed)."""
        # Simulating a right-skewed distribution like "days until event"
        percentiles = {
            5.0: 1.0,  # P5: 1 day
            10.0: 3.0,  # P10: 3 days
            20.0: 7.0,  # P20: 1 week
            40.0: 30.0,  # P40: 1 month
            60.0: 90.0,  # P60: 3 months
            80.0: 365.0,  # P80: 1 year
            90.0: 730.0,  # P90: 2 years
            95.0: 1095.0,  # P95: 3 years
        }

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=3650.0,  # 10 years
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        # Open bounds should be >= 0.001 and <= 0.999, but aggressive step enforcement
        # might result in slightly different values
        assert cdf[0] >= 0.001
        assert cdf[-1] <= 0.999

        # Should be smooth and monotonic despite the skewed nature
        steps = np.diff(cdf)
        assert all(step >= 5e-5 - 1e-10 for step in steps)
        assert all(step <= NUM_MAX_STEP + 1e-6 for step in steps)

    def test_concentrated_distribution_on_wide_grid(self):
        """Concentrated mass on a wide grid satisfies CDF constraints without aggressive enforcement.

        Reproduces the oldest-human-age scenario (Q14333): percentiles cluster
        around 111-172 on a [0, 200] grid.  Without the uniform pre-mix, PCHIP
        produces near-zero steps in the tails and requires aggressive enforcement
        on 11-25 % of bins.  The pre-mix should handle this proactively.
        """
        percentiles = {
            2.5: 111.0,
            5.0: 114.0,
            10.0: 117.0,
            20.0: 121.0,
            40.0: 126.0,
            50.0: 130.0,
            60.0: 134.0,
            80.0: 143.0,
            90.0: 152.0,
            95.0: 161.0,
            97.5: 172.0,
        }

        cdf, aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=True,
            open_lower_bound=False,
            upper_bound=200.0,
            lower_bound=0.0,
            zero_point=None,
            question_id=14333,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0
        assert cdf[-1] <= 0.999

        steps = np.diff(cdf)
        assert all(step >= 5e-5 - 1e-10 for step in steps), f"Min step violated: {np.min(steps)}"
        assert all(step <= NUM_MAX_STEP + 1e-6 for step in steps), f"Max step violated: {np.max(steps)}"

        assert not aggressive_enforcement, (
            "Uniform pre-mix should prevent aggressive enforcement for concentrated distributions"
        )

    def test_discrete_style_distribution(self):
        """Test with discrete-style distribution (integer values)."""
        percentiles = {
            5.0: 1.0,
            10.0: 2.0,
            20.0: 3.0,
            40.0: 5.0,
            60.0: 7.0,
            80.0: 10.0,
            90.0: 12.0,
            95.0: 15.0,
        }

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=20.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0
        assert all(a <= b for a, b in pairwise(cdf))


class TestAggressiveMinStepEnforcement:
    """Pin the LAST repair tier: the rebuild that fires when the min-step is still
    violated after ``safe_cdf_bounds``, and the tolerance that keeps it off float noise.

    AGENTS.md records that this tier never fires on real forecasts (0 of 1182
    archived numeric forecasts). A grid whose range is EXACTLY saturated by
    ``(num_points - 1) * min_step`` is the production shape of a forecast that puts
    essentially all of its mass beyond an open bound (the in-range CDF is then the
    bare min-step ramp), and the untoleranced trigger used to fire on that ramp's
    float error: a step 1e-18 short tripped it, and the rebuild's untoleranced range
    check then refused a range 1e-16 short and dropped the member (Mantic edge-case
    review, 2026-09, rank 3). The trigger and the range check now share the
    ``_MIN_STEP_TOLERANCE`` the post-check and final assertion always had, so the
    saturated grid builds as-is and only a genuine shortfall reaches the raise. After
    that change the tier is reachable through ``generate_pchip_cdf`` only as that
    ValueError, so its success branch is pinned by direct call below; the retired design log (git history) holds
    it as a dead-code cleanup candidate.
    """

    SIMPLE_PERCENTILES: ClassVar[dict[int | float, float]] = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}
    CONCENTRATED_PERCENTILES: ClassVar[dict[int | float, float]] = {
        2.5: 111.0,
        5.0: 114.0,
        10.0: 117.0,
        20.0: 121.0,
        40.0: 126.0,
        50.0: 130.0,
        60.0: 134.0,
        80.0: 143.0,
        90.0: 152.0,
        95.0: 161.0,
        97.5: 172.0,
    }

    def test_saturated_coarse_grid_builds_uniform_without_the_rebuild(self):
        cdf, aggressive_enforcement = generate_pchip_cdf(
            percentile_values=self.SIMPLE_PERCENTILES,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=0.1,
            max_step=1.0,
            num_points=11,
        )

        assert aggressive_enforcement is False
        np.testing.assert_allclose(cdf, np.linspace(0.0, 1.0, 11), atol=1e-10)
        assert np.diff(cdf).min() >= 0.1 - 1e-10

    def test_saturated_grid_on_concentrated_declaration_builds_uniform_without_the_rebuild(self):
        cdf, aggressive_enforcement = generate_pchip_cdf(
            percentile_values=self.CONCENTRATED_PERCENTILES,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=200.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=0.05,
            max_step=1.0,
            num_points=21,
        )

        assert aggressive_enforcement is False
        np.testing.assert_allclose(cdf, np.linspace(0.0, 1.0, 21), atol=1e-10)
        assert np.diff(cdf).min() >= 0.05 - 1e-10

    def test_saturated_grid_emits_no_rebuild_warn(self, caplog):
        """The float-epsilon shortfall of an exactly saturated ramp is not a repair event."""
        with caplog.at_level("INFO", logger="metaculus_bot.numeric.pchip_cdf"):
            generate_pchip_cdf(
                percentile_values=self.SIMPLE_PERCENTILES,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
                zero_point=None,
                min_step=0.1,
                max_step=1.0,
                num_points=11,
                question_id=4242,
                question_url="https://ex/q/4242",
            )

        messages = [record.getMessage() for record in caplog.records]
        assert not any("PCHIP minimum step enforcement required" in m for m in messages)
        assert not any("PCHIP aggressive enforcement completed" in m for m in messages)

    def test_the_rebuild_success_path_is_pinned_by_direct_call(self, caplog):
        """Exact values, so the tier can be restructured without silently changing what it
        emits: the shape-preserving allocation, the post-check and the completion line."""
        with caplog.at_level("INFO", logger="metaculus_bot.numeric.pchip_cdf"):
            out = _rebuild_with_min_steps(
                np.array([0.0, 0.05, 0.5, 1.0]),
                0.1,
                open_lower_bound=False,
                open_upper_bound=False,
                question_id=4242,
                question_url="https://ex/q/4242",
            )

        np.testing.assert_allclose(out, [0.0, 1 / 6, 0.5666666666666667, 1.0], rtol=0, atol=1e-15)
        assert float(np.diff(out).min()) == pytest.approx(1 / 6)
        assert out[0] == 0.0
        assert out[-1] == 1.0
        messages = [record.getMessage() for record in caplog.records]
        assert any("PCHIP aggressive enforcement completed for Q 4242" in m for m in messages)

    def test_range_too_small_for_min_steps_raises(self):
        with pytest.raises(ValueError, match="Cannot satisfy minimum step requirement"):
            generate_pchip_cdf(
                percentile_values=self.SIMPLE_PERCENTILES,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
                zero_point=None,
                min_step=0.3,
                max_step=1.0,
                num_points=5,
            )

    def test_two_point_grid_needs_no_rebuild(self):
        """``num_points=2`` is the only grid where the rebuild's shape-preserving
        allocation degenerates to plain linear spacing; pin that it stays legal."""
        cdf, aggressive_enforcement = generate_pchip_cdf(
            percentile_values=self.SIMPLE_PERCENTILES,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=0.6,
            max_step=1.0,
            num_points=2,
        )

        assert aggressive_enforcement is False
        assert cdf == [0.0, 1.0]


if __name__ == "__main__":
    pytest.main([__file__])
