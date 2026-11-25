"""Rendering checks for the loss-attribution report."""

import pytest

from metaculus_bot.performance_analysis.loss_attribution import corp_decomposition
from metaculus_bot.performance_analysis.loss_attribution_report import _calibration, _recalibration
from metaculus_bot.scoring_common import spot_peer_delta


def test_reliability_renders_counts_and_corp_spot_peer_bound() -> None:
    loss = {"n": 10, "mean_score": 0.3, "mcb": 0.04, "dsc": 0.1, "unc": 0.36}
    era = {
        "exclusions": {"binary: outcome diagnostics": {}, "multiple_choice: outcome diagnostics": {}},
        "calibration": [
            {
                "name": "binary p_yes",
                "brier": loss,
                "log": loss,
                "tied_top": 0,
                "reliability": [
                    {
                        "label": "[0, 1)",
                        "n": 10,
                        "n_questions": 10,
                        "mean_forecast": 0.4,
                        "observed": 0.5,
                        "jeffreys_interval": [0.2, 0.8],
                        "observed_minus_forecast": 0.1,
                        "observed_count": 5,
                        "expected_count": 4.0,
                        "observed_minus_expected": 1.0,
                        "cluster_interval": [-2.0, 3.0],
                    }
                ],
            }
        ],
    }
    text = "\n".join(_calibration(era))
    assert "observed count" in text
    assert "expected count" in text
    assert "[-2.00, +3.00]" in text
    assert "in-sample upper bound" in text
    assert "4.000" in text


def test_binary_corp_log_miscalibration_uses_spot_peer_scale() -> None:
    probabilities = [0.25, 0.25, 0.75, 0.75]
    outcomes = [False, True, True, True]
    result = corp_decomposition(probabilities, outcomes, "log")
    fitted = [0.5, 0.5, 1.0, 1.0]
    deltas = [
        spot_peer_delta(old_prob=p if outcome else 1 - p, new_prob=q if outcome else 1 - q, question_type="binary")
        for p, q, outcome in zip(probabilities, fitted, outcomes, strict=True)
    ]
    assert sum(deltas) / len(deltas) == pytest.approx(100 * result.mcb)


def test_recalibration_rendering_shows_train_parameters_and_pooled_results() -> None:
    result = {
        "era_order": ["pre_flip", "post_flip"],
        "exclusions": {"missing era": 2},
        "rows": [
            {
                "question_type": "binary",
                "method": "platt",
                "era": "post_flip",
                "n_train": 100,
                "n_test": 20,
                "status": "evaluated",
                "parameters": {"intercept": 0.1, "slope": 0.8},
                "mean_delta": 2.0,
                "interval": [-1.0, 4.0],
            }
        ],
        "pooled": [
            {
                "question_type": "binary",
                "method": "platt",
                "n_test": 20,
                "mean_delta": 2.0,
                "interval": [-1.0, 4.0],
                "skipped_eras": 1,
                "skipped_questions": 50,
            }
        ],
    }
    text = "\n".join(_recalibration(result))
    assert "post_flip" in text
    assert "intercept" in text
    assert "slope" in text
    assert "pooled" in text.lower()
    assert "missing era: 2" in text
    assert "positive" in text.lower()
    assert "decisive" in text
