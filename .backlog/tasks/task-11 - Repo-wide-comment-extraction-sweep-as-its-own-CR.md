---
id: TASK-11
title: Repo-wide comment-extraction sweep as its own CR
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - operator-decides
  - code-quality
dependencies: []
priority: medium
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Repo-wide comment-extraction sweep as its own CR**: 5,238 smell-hook findings over 436 files, four files done. Recipe that worked: one agent per file group, comments and docstrings only, an AST-equality check against HEAD with docstrings stripped, scanner at zero per file, prose to the topical doc, then `make all`.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, operator decides".
<!-- SECTION:DESCRIPTION:END -->
