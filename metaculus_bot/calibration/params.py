"""Checked-in Platt scaling parameters.

Both fits are the identity (bias=0, slope=1) — a deliberate, evidence-based
choice, not a placeholder. We fit Platt on closed spring-aib-2026 data and
on fall-aib-2025 (cached at ``scratch/platt_fit_2026-05_test/`` and
``scratch/platt_fit_fall_aib_2025/``) and concluded:

1. **Marginal in-sample benefit.** Mean Brier improved by ~0.0015 binary /
   ~0.0001 MC on spring-aib-2026, and in-sample is optimistic.
2. **Cross-round instability.** Binary slope drifts from 0.83 (spring) to
   1.66 (fall) — opposite calibration shapes between rounds, well above any
   reasonable stability threshold. See ``stability_check.md``.

Given (1) + (2), shipping a fixed-coefficient calibration risks introducing
ensemble-drift bias that exceeds the modest expected benefit. Identity is
the considered choice. The infrastructure (math, fit CLI, integration hook)
remains so the analysis can be re-run cheaply when more data accumulates or
the ensemble stabilizes — see backlog decision-1 "Calibration and post-hoc layers" for the runbook.

The deviation caps in ``metaculus_bot.constants`` are also dormant until
non-identity params are checked in here — they apply on top of any
non-identity fit but are no-op against identity.

To revisit: re-run ``python -m metaculus_bot.calibration.fit_platt_cli``
against fresh ``performance_data.json`` snapshots from a future round and
re-do the cross-round stability check before considering any change to
these constants.
"""

from metaculus_bot.calibration.platt import PlattParams

# Final binary aggregation Platt fit. Identity by deliberate choice.
BINARY_PLATT_PARAMS: PlattParams = PlattParams.identity()

# Fit SEPARATELY from the binary one, never reused across the two: their calibration shapes differ.
MC_PLATT_PARAMS: PlattParams = PlattParams.identity()
