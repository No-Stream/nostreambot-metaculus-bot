---
id: TASK-50
title: 'Loss attribution: list question ids per ghost distance bin'
status: In Progress
assignee: []
created_date: '2026-11-25 09:00'
updated_date: '2026-11-25 16:58'
labels:
  - analysis
dependencies: []
priority: medium
ordinal: 50000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The loss-attribution ghost comparison (metaculus_bot/performance_analysis/loss_attribution.py, disagreement_report) prints per-bin n and mean spot-peer change only. On the 2026-11-25 round the ghost's MC lead (+6.3 [-7.4, +20.5], n=27) rests on 6 questions at total variation >= 0.15, and the binary deficit (-6.9) on 11 questions at |p-q| 0.15-0.3 (-44.7). Add the post ids / URLs of the high-disagreement questions to the JSON and markdown so they can be trace-reviewed, then review them.
<!-- SECTION:DESCRIPTION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added per-question ghost distance-bin trace details in JSON and all-era markdown, plus outcome-based observed-versus-expected winner counts with resolution-day intervals. Verified with the loss-attribution, ghost-scoring, and ghost-analysis tests and the requested free gates. Still open: the trace review of the listed high-disagreement questions.
<!-- SECTION:FINAL_SUMMARY:END -->
