---
id: TASK-7
title: 'Prompt: allow already-priced answers in clause pricing'
status: To Do
assignee: []
created_date: '2026-11-19 17:27'
labels: []
dependencies: []
priority: medium
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Blind reviews of Sol, Opus and Argon on Q46503, Q46520 and Q46504 (2026-11-19; Opus high and gpt-6.2-astra high reviewers) both found that the clause-pricing step double-discounts: members discount a clause such as public access on top of a market price whose criteria already require it. Proposed: price a clause only where the anchor's criteria differ from the question's; for numeric questions, replace the generic plus-or-minus 10 percent check with a read-back of two or three decision-relevant bin probabilities; add a check that the final number follows from the member's own red team. A prompt change is a config-era boundary and needs presence and absence pins in tests/prompts/. Reviews are in scratch/member_review_2026-11-19/.
<!-- SECTION:DESCRIPTION:END -->
