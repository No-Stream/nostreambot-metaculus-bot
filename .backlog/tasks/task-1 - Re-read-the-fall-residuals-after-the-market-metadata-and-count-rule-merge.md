---
id: TASK-1
title: Re-read the fall residuals after the market-metadata and count-rule merge
status: To Do
assignee: []
created_date: '2026-11-05 06:42'
labels:
  - residual
  - analysis
dependencies: []
references:
  - scratch/residual_2026-11-04/SYNTHESIS.md
  - scratch/residual_2026-11-04/ROUND_PLAN.md
priority: medium
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to the 2026-11-04 residual round (scratch/residual_2026-11-04/SYNTHESIS.md, incl. the follow-up sections). Three changes shipped on branch resid-post-halloween and change what forecasters see: the labelled market metadata cell (2228513), the count-process prompt rule plus the gap-fill same-length-window nudge (b5b68c9). When the branch merges to main, add its merge-to-main committer timestamp as a labelled fine sub-era in metaculus_bot/performance_analysis/eras.py (pooled into triple_fall_config, operator ruling). At the next residual round (the preregistration's second checkpoint, 30 fall tournament STRICT records), re-run the reads below on all records, not just misses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Era label for the resid-post-halloween merge added to eras.py (+ pinned test) once merged to main
- [ ] #2 Type x era Bayesian pooling (python -m metaculus_bot.performance_analysis.type_pooling, config_era and triple_subera) re-run; report whether the three-model-era MC-minus-binary gap (-5.3 [-17.8, +3.8], P=0.83 on 2026-11-04) persists
- [ ] #3 Count questions: compare member method use (Poisson vs empirical windows vs negative binomial) and how often bundles now carry same-length window counts (2 of 26 before), with paired within-question scores
- [ ] #4 Markets: re-run the deep vs thin deferral cut and check whether the new metadata (trade age, spread, moves) changes how far forecasts sit from thin or stale markets
- [ ] #5 Prereg second checkpoint: era_gap watch verdict, width trigger, cost ceiling
<!-- AC:END -->
