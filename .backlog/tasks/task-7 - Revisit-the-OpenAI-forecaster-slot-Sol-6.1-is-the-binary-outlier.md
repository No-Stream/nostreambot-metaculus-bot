---
id: TASK-7
title: Revisit the OpenAI forecaster slot (Sol 6.1 is the binary outlier)
status: To Do
assignee: []
created_date: '2026-11-20 16:14'
labels: []
dependencies: []
priority: low
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
2026-11-20 analysis (scratchpad scripts sol_stats.py and loo_index.py, not kept): over 38 questions Sol 6.1 scores the same as Opus 5.5 (-0.7 spot peer, 95% CI -9.8 to +8.4) and trails Argon by -4.7 (CI -12.0 to +2.8). Sol 5.6 trailed Opus 4.8 by the same margin. But Sol 6.1 is the member farthest from the median on 13 of 19 binary questions (Sol 5.6: 24%; p=0.002 against one in three). A median of three hides this only while all three members survive: when one member drops, the median of two is the mean and Sol's outlier gets half the weight, and with MIN_FORECASTERS_TO_PUBLISH=1 a lone Sol publishes as is. Removing Sol from the median would have changed the published score by +1.1 (CI -1.1 to +3.5). Options: Sol 6.2 (released 2026-11-17, same price, mixed reception, so wait), Astra 6.2 (4x the price, would need a paid backtest), Sol 5.6 (2x the price, no better). Re-run when Sol and Argon reach about 80 shared resolved questions, and include how often a member drops so the brittleness can be measured.
<!-- SECTION:DESCRIPTION:END -->
