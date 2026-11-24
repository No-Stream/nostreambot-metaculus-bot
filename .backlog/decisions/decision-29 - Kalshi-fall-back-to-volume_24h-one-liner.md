---
id: decision-29
title: Kalshi fall back to volume_24h one-liner
date: '2026-11-24 18:23'
status: rejected
---
## Context

Migrated from FUTURE.md (deleted 2026-11-24), section "Rejected: do not re-propose without new evidence", paragraph **Research architecture.**. Verbatim text below, including dates and receipt paths.

## Decision

The Kalshi "fall back to `volume_24h`" one-liner (rejected 2026-08-03 and pinned as rejected in `tests/test_prediction_market_liquidity_contract.py`: that field is 0 on long-horizon markets, so scoring off it labels a deep market thin).

## Consequences

Status `rejected`. Do not re-propose without new evidence; the original text above states any revival condition.
