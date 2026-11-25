---
id: TASK-17
title: Gap-fill v2 driver on gpt-6-luna at xhigh/max instead of gpt-6.1-terra at low
status: To Do
assignee: []
created_date: '2026-11-24 18:24'
labels:
  - priced-lever
  - research
  - roster
  - cost
dependencies: []
priority: high
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**HIGH PRIORITY: gap-fill v2 driver on gpt-6-luna at xhigh/max instead of gpt-6.1-terra at low.** The driver sends a top-level `reasoning_effort`, which litellm 1.98's OpenRouter transform rewrites `max` -> `xhigh`, so the arm below ran at xhigh; testing true `max` needs the `reasoning={"effort": ...}` form. One-question replay on 2026-09-22 (Q44229, `scratch/model_migration_2026-09-22/`): both finished inside the 540 s loop (luna 146 s, sol 85 s, 31-32 tool calls), and a blind Opus judge called it a tie leaning luna, low confidence. Luna read 882k prompt tokens against sol's 711k at a twentieth of the input price, so the driver's spend drops by roughly an order of magnitude. Needs a multi-question replay before switching. Since 2026-10-27 the driver runs gpt-6.1-terra at half Sol's price, which narrows the saving to roughly tenfold over Terra rather than twentyfold over Sol. Same-day native-search probe: luna at true `max` gave the brief a blind judge preferred, but took 278 s against sol-low's 30 s; luna at `high` or `xhigh` there is the untested middle.

Migrated from FUTURE.md (deleted 2026-11-24), section "Open, priced levers not yet built".
<!-- SECTION:DESCRIPTION:END -->
