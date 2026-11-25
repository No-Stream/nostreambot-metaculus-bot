---
id: TASK-16
title: Migrate Gemini grounded search to Google's Interactions API
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - operator-decides
  - research
dependencies: []
priority: low
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**LOW: migrate Gemini grounded search to Google's Interactions API** (`client.aio.interactions.create`, with `url_citation` annotations on text), and consider the same migration for gap-fill v2 `read_document` and the resolution-source `url_context` rung. The 2026-09-22 probe annotated 7/8 Interactions calls, all of which resolved, while one call with four searches dropped annotations; the same-day `generate_content` control annotated 5/6, so Interactions is not demonstrably better yet. Google staff named Interactions as the likely fix (forum thread 174074). It would remove roughly 3k output tokens of self-citation redirect URLs per call. Revisit if Google confirms a fix or deprecates `generate_content`; receipt: `scratch/gemini_grounding_2026-09-22/README.md`.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, operator decides".
<!-- SECTION:DESCRIPTION:END -->
