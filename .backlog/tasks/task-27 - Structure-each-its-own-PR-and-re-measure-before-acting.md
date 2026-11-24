---
id: TASK-27
title: 'Structure, each its own PR, and re-measure before acting'
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - recorded-not-built
  - code-quality
dependencies: []
priority: medium
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Structure, each its own PR, and re-measure before acting.** Splitting `resolution_source.py` is blocked by 22 monkeypatch names, since a patch left on the old module stays green while proving nothing; `forecaster.py`'s remaining seam is the soft-deadline machinery and `prompts.py`'s is the shared continuous template both prompts import. Also open: a public `metaculus_get` plus the public-then-private comment paging three call sites duplicate, the fetch primitives `agentic/tools.py` reaches into `resolution_source.py` for, and `basedpyright` strict on the core pipeline.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, recorded and not built".
<!-- SECTION:DESCRIPTION:END -->
