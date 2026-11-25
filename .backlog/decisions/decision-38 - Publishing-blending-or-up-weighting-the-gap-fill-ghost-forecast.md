---
id: decision-38
title: 'Publishing, blending or up-weighting the gap-fill ghost forecast'
date: '2026-11-25 19:15'
status: rejected
---
## Context

The gap-fill driver writes a private "ghost" forecast (telemetry only). Over the 2026-11-25 round (strict cohort), pricing the ghost as if published, at the production clamp for its forecast time, gives spot peer per question: MC +6.3 [-7.4, +20.5] (n=27), binary -5.8 [-16.6, +2.3] (n=67), numeric -3.5 (n=28), discrete +5.2 (n=27). Backlog task-50 trace-reviewed all 22 questions where the ghost and the published forecast differ by a distance of at least 0.15 (three dossiers, receipts in scratch/residual_2026-11-25_floors/ghost_trace_review/; report from `python -m metaculus_bot.performance_analysis.loss_attribution`).

## Decision

Do not publish, blend in or up-weight the ghost. Findings:
- No question where the ghost had evidence the ensemble lacked, except 46195 (discrete, +193). The ensemble receives the ghost's research findings; the ghost never sees gap-fill v1's results because the two passes run concurrently (docs/agentic_gap_fill.md).
- Most of each large gap exists in the driver's pre-research dry run (e.g. +90 of +126 on MC 45040). Binary ghosts equal their dry run in 30 of 60 questions.
- The MC lead rests on 45040 (+126) and 45956 (+81); without them the ghost averages -1.5 on MC. The binary deficit is mostly 44705 (ghost 0.01 against a value stated in the question background). All 11 high-disagreement binary questions are in the retired summer setup; none of the 28 fall binary ghosts differs by 0.15.
- Ghost as a fourth binary member totals -30.8; a 50/50 binary blend totals -95.7.
- The wins have the "a dissenting member was right" shape that decision-11 rejected; the same shape also produced losses.

## Consequences

Status `rejected`. Revive only if, at about 30 resolved fall-config MC questions (backlog task-51), the ghost's MC lead survives without one or two questions carrying it AND passes the out-of-sample era test in docs/performance_analysis.md.
