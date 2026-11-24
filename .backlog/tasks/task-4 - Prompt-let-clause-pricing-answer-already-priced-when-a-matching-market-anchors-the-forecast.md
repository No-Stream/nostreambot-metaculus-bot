---
id: TASK-4
title: >-
  Prompt: let clause pricing answer 'already priced' when a matching market
  anchors the forecast
status: To Do
assignee: []
created_date: '2026-11-19 16:43'
labels: []
dependencies: []
priority: medium
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Blind reviews of Sol/Opus/Argon on Q46503, Q46520, Q46504 (2026-11-19, Opus high and gpt-6.2-astra high reviewers) both flagged that the clause-pricing step double-discounts: members discount a clause (e.g. public access) on top of a market price whose criteria already require it. Proposed: price clauses only where the anchor's criteria differ from the question's; for numeric, replace the generic +/-10% check with a read-back of two or three decision-relevant bin probabilities; add a check that the final number follows from the member's own red team. Prompt change = config-era boundary; needs presence/absence pins in tests/prompts/. Reviews: session scratchpad member_review/review_opus.md and review_astra.md (copy into scratch/ before relying on them).
<!-- SECTION:DESCRIPTION:END -->
