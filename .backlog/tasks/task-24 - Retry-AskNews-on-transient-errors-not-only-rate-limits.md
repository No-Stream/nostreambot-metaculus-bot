---
id: TASK-24
title: 'Retry AskNews on transient errors, not only rate limits'
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - recorded-not-built
  - research
dependencies: []
priority: low
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Low priority: retry AskNews on transient errors, not only rate limits.** `_is_asknews_retryable` (`research/providers.py`) retries only 429 and concurrency errors, so a 401 or 5xx during an AskNews auth blip drops the AskNews section for that question. On 2026-10-27 two local posts lost it to a few minutes of 401s from `auth.asknews.app` while credentials were valid. Since that date there is no Perplexity or Exa fallback, and native search, Gemini search and both gap-fill passes cover the loss, so this is a nicety. Any retry adds time on the research critical path, so keep it to one short retry inside the existing time budget.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, recorded and not built".
<!-- SECTION:DESCRIPTION:END -->
