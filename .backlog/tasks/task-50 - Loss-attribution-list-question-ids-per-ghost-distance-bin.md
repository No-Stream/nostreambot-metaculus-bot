---
id: TASK-50
title: 'Loss attribution: list question ids per ghost distance bin'
status: Done
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
Added per-question ghost distance-bin trace details in JSON and all-era markdown, plus outcome-based observed-versus-expected winner counts with resolution-day intervals. Verified with the loss-attribution, ghost-scoring, and ghost-analysis tests and the requested free gates. Trace review done 2026-11-25: three dossiers covering all 22 questions at distance >= 0.15, in scratch/residual_2026-11-25_floors/ghost_trace_review/. No exploitable pattern; recorded as backlog decision-38 "Publishing, blending or up-weighting the gap-fill ghost forecast". The review also found that the ghost was priced without the production clamp (fixed: post 44705 -340 to -271; binary mean -6.9 to -5.8) and that discrete 44916 is a pre-2026-08-31 CDF bug (commit d4ee57f), not a real disagreement.
<!-- SECTION:FINAL_SUMMARY:END -->
