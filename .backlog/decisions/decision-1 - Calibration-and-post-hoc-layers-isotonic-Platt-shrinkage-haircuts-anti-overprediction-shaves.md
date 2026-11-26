---
id: decision-1
title: >-
  Calibration and post-hoc layers (isotonic, Platt, shrinkage, haircuts,
  anti-overprediction shaves)
date: '2026-11-24 18:23'
status: rejected
---
## Context

Migrated from FUTURE.md (deleted 2026-11-24), section "Rejected: do not re-propose without new evidence", paragraph **Calibration and post-hoc layers**. Verbatim text below, including dates and receipt paths.

## Decision

**Calibration and post-hoc layers**, all dead. Calibration as a lever at all (operator, 2026-07-16: the slope FLIPS by era, so any fit spans opposite-signed eras and power is about 0.25 even at N=400). With it: post-hoc isotonic (2026-05-10, a small-N artifact), Platt, directional shrink, YES-side shrink (2026-07-08, era-local, and a two-era fit degraded held-out fall; never touch the NO side), shrinkage toward 50% (2026-05-10, costs calibrated extremes), the signed deadzone haircut toward 0.5 (2026-07-08, 0 of 24 cells help in both well-powered eras, though the toward-0.5 direction constraint survives as a design rule), fixed-direction and one-sided anti-overprediction shaves (the binary slope flips sign across rounds).

## Addendum 2026-11-25: forward-chained out-of-sample check

The loss-attribution report now runs the out-of-sample era test as a standing section (metaculus_bot/performance_analysis/recalibration.py): fit on all earlier eras (pre_flip split at its median forecast time), score each later era in spot peer. On the 2026-11-25 round (strict cohort): binary Platt -3.1 [-6.5, -0.2] and isotonic -5.4 [-11.6, -0.6] per question over 364 held-out questions; MC power scaling -0.6 [-4.5, +3.1] over 62. The in-sample ceiling (binary log miscalibration) is 2.6 spot peer per question. This supports the rejection.

## Consequences

Status `rejected`. Do not re-propose without new evidence; the original text above states any revival condition.
