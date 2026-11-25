---
id: TASK-5
title: 'Trace review: verify the authenticated Mantic pull'
status: To Do
assignee: []
created_date: '2026-11-19 17:21'
updated_date: '2026-11-20 06:37'
labels: []
dependencies: []
priority: medium
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MANTIC_TOKEN is not in .env, so the Mantic pull path (collector --platform mantic, make trace_index with Mantic) is tested only on the public series-1 page with a hand-added my_forecasts block. Once the token exists: record one authenticated resolved-posts page as a fixture, confirm my_forecasts and score_data have the Metaculus shape, replace the hand-built block in tests/test_performance_analysis_collector.py TestManticPull, and run make trace_index with both platforms (plan section 4.9).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-19: MANTIC_TOKEN exists as a GitHub Actions secret, but gh cannot read secret values, so it has to be copied into .env by hand. Until then the index builds Metaculus only, and the build logs a warning that it skipped Mantic. Once the token is in .env, the next make sync_all pulls Mantic automatically; check that the pull returns the bot's forecasts.
<!-- SECTION:NOTES:END -->
