---
id: TASK-8
title: 'Metaculus 2% open-tail floor: re-check out of sample'
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
updated_date: '2026-11-24 18:43'
labels:
  - operator-decides
  - numeric
  - aggregation
dependencies: []
priority: medium
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Metaculus 2% open-tail floor, adopted 2026-11-24 (commit 4a3f713); re-check out of sample.** The operator chose to ship METACULUS_OUT_OF_RANGE_TAIL_FLOOR = 0.02 together with the stated open-bound probability fields (a6d09a6), instead of the tail investigation's recommendation to defer. At each residual round, using only forecasts after 2026-11-24 and bucketing by era and roster, count escapes on open sides whose members published under 2% beyond the bound, and compare the per-side escape rate with the 1.44% break-even. Revert or raise the floor only on that out-of-sample evidence; the investigation's bar was at least 150 such sides (half after the fall roster change) with a one-sided 90% lower bound above 1.44%. Also read NUMERIC_STATED_TAIL to see how often members now state tails above 2%, which makes the floor bind less. Receipt: scratch/trace_evals_2026-11-23/tail_mass_investigation.md, recommendation 1, and docs/constants.md "METACULUS_OUT_OF_RANGE_TAIL_FLOOR".

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, operator decides".
<!-- SECTION:DESCRIPTION:END -->
