---
id: TASK-45
title: Short update round with partial cross-pollination
status: To Do
assignee: []
created_date: '2026-11-24 22:48'
updated_date: '2026-11-26 01:22'
labels:
  - ensemble
  - cost
dependencies: []
priority: high
ordinal: 45000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Idea from the 2026-11-24 fall triage: after the independent fan-out, give each member a short update prompt with its own prior answer plus a summary of the other members' forecasts and main reasons, and let it revise. Reuse the original prompt prefix so prompt caching makes the second pass cheap (cache writes still cost). Must not double per-question cost (~$2 ceiling). Prior evidence: the earlier full review round added latency and performed the same or worse; SwampBot runs a review round. Needs a priced design and a methodical backtest before shipping.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-25: the operator confirmed that the 'earlier full review round' in the description is the stacker (a high-end model read every member's output and wrote one final prediction; backtest showed no gain; disabled in production, decision-9). Design for a cheap-model bench of the peer-update round: scratch_docs_and_planning/cross_pollination_bench_design_2026-11-25.md. Literature expectation is about 0.02 log loss at best (arXiv 2512.22625).

2026-11-25: design approved by the operator, including the statistics (hierarchical PyMC model of paired differences as primary, normal-likelihood refit and a Bayesian bootstrap by event as checks, effect prior Normal(0, 5 points)). Reviewed by Claude Fable 5.2 and GPT-6.2 Astra. Implementation plan: scratch_docs_and_planning/cross_pollination_bench_plan_2026-11-25.md (ten tasks; tasks 1 to 7 are enough for the paid smoke test). Build and runs happen in a later session; nothing is built or spent yet. Smoke test about USD 0.30 (48 calls), main run about USD 8 (58 questions, 1,392 calls), each needing its own operator approval.
<!-- SECTION:NOTES:END -->
