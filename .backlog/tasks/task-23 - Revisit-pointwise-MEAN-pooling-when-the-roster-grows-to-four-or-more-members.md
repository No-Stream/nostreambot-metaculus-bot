---
id: TASK-23
title: Revisit pointwise MEAN pooling when the roster grows to four or more members
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - recorded-not-built
  - aggregation
  - numeric
dependencies: []
priority: medium
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Medium priority: revisit pointwise MEAN pooling when the roster grows to four or more members.** The 2026-11-24 replay over 280 Metaculus numeric and discrete questions found mean minus median at +2.0 spot peer per question [+0.0, +4.3] in the five- and six-member eras (n=176) and +0.2 [-1.6, +1.9] in the recent mostly-three-member eras (n=104), matching the 2026-09-15 bench. The mean pool is 6-10% wider at 5-95 but barely moves mass beyond open bounds; a median-body / mean-tail hybrid came out +0.5 [-0.0, +1.4]. The operator expects to add a fourth and perhaps a fifth member, at which point re-run `scratch/trace_evals_2026-11-23/work/mean_replay/` (artifacts repo) on the new roster's members.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, recorded and not built". Related decisions: decision-19, decision-18.
<!-- SECTION:DESCRIPTION:END -->
