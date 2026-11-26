---
id: TASK-40
title: Separate outside-view and inside-view stages
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
updated_date: '2026-11-26 00:52'
labels:
  - study-only
  - prompts
dependencies: []
priority: low
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
separate outside and inside view stages

Migrated from FUTURE.md (deleted 2026-11-24), section "Study-only, not planned".
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-25, operator spitball: a concrete form is a closed-book forecast first (no sources), then the research and an update. The operator expects it to be counterproductive because the model anchors on its uninformed prior. It can run on the cross-pollination bench harness at the same cost (two calls sharing no cache prefix), and the closed-book call doubles as a check for outcome leakage in backtests. Not scheduled.
<!-- SECTION:NOTES:END -->
