---
id: TASK-9
title: 'Gemini 4 Carbon in the Google slot: decided against, 2026-11-19; Argon stays'
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - operator-decides
  - roster
dependencies: []
priority: low
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Gemini 4 Carbon in the Google slot: decided against, 2026-11-19; Argon stays.** Reverted to Argon on 2026-11-14 after Carbon at effort `high` hit the 600s soft deadline on 3 of 3 questions (`docs/roster_history.md`). On 2026-11-19 the operator approved three timed calls on the prod binary prompt for Q46548, all through the personal OpenRouter key and reusing the 2026-11-03 research archive. Carbon `high` took 813s and wrote 44,052 output tokens (37,454 reasoning), costing $0.48. Carbon `medium` took 260s and wrote 13,105 tokens, costing $0.17; its first try died with an OpenRouter 504 after about 5 minutes. Opus 5.5 `xhigh` took 52s and wrote 5,390 tokens. Carbon generates at about 50 tokens/s, half Opus's speed, so the slowness is long thinking at low throughput rather than queueing. All three answered 0.40. A blinded gpt-6.2-astra judge preferred Opus narrowly over Carbon `medium`. In prod on 2026-11-03, Argon answered this question in about the same time as Opus. Revisit only if Carbon's throughput improves.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, operator decides".
<!-- SECTION:DESCRIPTION:END -->
