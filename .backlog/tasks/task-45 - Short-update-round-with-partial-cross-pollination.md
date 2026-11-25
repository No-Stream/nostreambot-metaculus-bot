---
id: TASK-45
title: Short update round with partial cross-pollination
status: To Do
assignee: []
created_date: '2026-11-24 22:48'
labels:
  - ensemble
  - cost
dependencies: []
priority: medium
ordinal: 45000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Idea from the 2026-11-24 fall triage: after the independent fan-out, give each member a short update prompt with its own prior answer plus a summary of the other members' forecasts and main reasons, and let it revise. Reuse the original prompt prefix so prompt caching makes the second pass cheap (cache writes still cost). Must not double per-question cost (~$2 ceiling). Prior evidence: the earlier full review round added latency and performed the same or worse; SwampBot runs a review round. Needs a priced design and a methodical backtest before shipping.
<!-- SECTION:DESCRIPTION:END -->
