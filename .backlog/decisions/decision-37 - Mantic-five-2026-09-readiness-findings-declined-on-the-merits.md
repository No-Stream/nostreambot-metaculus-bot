---
id: decision-37
title: 'Mantic: five 2026-09 readiness findings declined on the merits'
date: '2026-11-24 18:23'
status: rejected
---
## Context

Migrated from FUTURE.md (deleted 2026-11-24), section "Rejected: do not re-propose without new evidence", paragraph **Mantic, accepted with no change (do not re-raise).**. Verbatim text below, including dates and receipt paths.

## Decision

Five 2026-09 readiness findings were declined on the merits: retrying the pre-spend tournament GET, a `MEMBER_FORECAST` line for a member whose CDF build raises (percentile and per-bin both), the supply probe's preflight as a frozen-dataclass `partial`, and letting the public snapshot override an empty `my_forecasts`. Each remedy was a broad except, a second marker call site, or a `partial` that breaks the patch-where-used contract.

## Consequences

Status `rejected`. Do not re-propose without new evidence; the original text above states any revival condition.
