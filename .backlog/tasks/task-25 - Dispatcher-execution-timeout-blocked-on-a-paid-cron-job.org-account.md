---
id: TASK-25
title: 'Dispatcher execution timeout, blocked on a paid cron-job.org account'
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - recorded-not-built
  - ops
dependencies: []
priority: medium
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Dispatcher execution timeout, blocked on a paid cron-job.org account.** The 2026-10-03 timeout mails matched dispatches that arrived and completed, so GitHub acknowledged slowly rather than work going missing. Setting `requestTimeout: 300` in `scripts/cronjob_dispatch_setup.py` would tolerate that, but the free tier caps execution at 30 s, so the field has no effect there. Until the account is upgraded, keep first-failure mail and reconcile a timeout with `make dispatch_watch`. The full change with tests and docs is commit `0ef7f9b` (closed PR #75).

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, recorded and not built".
<!-- SECTION:DESCRIPTION:END -->
