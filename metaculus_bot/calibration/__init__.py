"""Post-hoc calibration of the bot's final published probabilities.

Implements logistic recalibration (Platt scaling) on the final binary and
multiple-choice outputs of the aggregation pipeline
(``metaculus_bot/aggregation_pipeline.py``). The hook is gated by
``PLATT_CALIBRATION_ENABLED`` and reads fitted parameters from ``params.py``.

Currently shipped with identity parameters by deliberate choice — see
``params.py`` for the rationale (in-sample benefit too small to justify
the cross-round parameter instability observed between fall-aib-2025 and
spring-aib-2026). The math, fit CLI, and integration hook are kept in
place so the analysis can be re-run cheaply when more data accumulates.

Numeric (CDF) recalibration is out of scope for this iteration and tracked
in backlog decision-1 "Calibration and post-hoc layers".

References:
- Upstream: Metaculus notebook "Improving Forecaster Performance via
  Automated Calibration Adjustment" (2026-05-01).
- Cached fit + stability check artifacts: ``scratch/platt_fit_2026-05_test/``
  and ``scratch/platt_fit_fall_aib_2025/``.
- Source ``performance_data.json`` snapshots:
  ``scratch/analysis_2026-05/performance_data.json`` (spring-aib-2026,
  closed), ``scratch/fall_aib_2025_performance.json`` (fall-aib-2025).
"""

from metaculus_bot.calibration.params import BINARY_PLATT_PARAMS, MC_PLATT_PARAMS
from metaculus_bot.calibration.platt import (
    PlattParams,
    apply_binary_platt,
    apply_mc_platt,
    fit_platt,
)

__all__ = [
    "BINARY_PLATT_PARAMS",
    "MC_PLATT_PARAMS",
    "PlattParams",
    "apply_binary_platt",
    "apply_mc_platt",
    "fit_platt",
]
