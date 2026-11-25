---
id: TASK-46
title: Post-merge checks on the first batch under the fall 2026-11-24 config
status: To Do
assignee: []
created_date: '2026-11-24 22:48'
labels:
  - operations
  - roster
dependencies: []
priority: high
ordinal: 46000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After PR #87 merges and the first ~10 questions run: (1) make cost_report, confirm cost per question stays under ~$2 with Fable 5.2 medium as the fourth member; (2) NUMERIC_STATED_TAIL status mix: how often fields are missing, invalid, clamped or build_failed_ignored, and how often stated tails exceed the 2% floor; (3) ANTHROPIC_FORECAST_ROUTE on Mantic: share of direct vs fallback and reasons (a persistent reason=other means the direct request shape is wrong); (4) Fable refusal/empty-content rate and FORECASTERS_SURVIVED; (5) Fable medium wall time vs FORECASTER_SOFT_DEADLINE. Also record the merge timestamp in performance_analysis/eras.py.
<!-- SECTION:DESCRIPTION:END -->
