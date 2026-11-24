---
id: doc-1
title: How to use this backlog
type: guide
created_date: '2026-11-24 18:25'
---

# How to use this backlog

This backlog replaced the repo-root `FUTURE.md` design log on 2026-11-24. The last committed copy of that file is `git show 3591f65:FUTURE.md`.

## Original FUTURE.md introduction (verbatim)

### Design log

Intent and dead ends, not current state, which lives in `docs/`. Cited `scratch/...` receipts are the operator's local
artifacts and gitignored on purpose, so a citation attributes a claim rather than pointing at a file a reader can open,
and planning docs named here may have moved out of the public repo. Neither is a broken link to fix.

## Where each FUTURE.md section went

- "Open, operator decides": tasks labelled `operator-decides`.
- "Open, priced levers not yet built": tasks labelled `priced-lever`.
- "Open, recorded and not built": tasks labelled `recorded-not-built` (the "Watch items" paragraph is also labelled `watch`).
- "Rejected: do not re-propose without new evidence": decisions (statuses `rejected`, `rejected-revivable`, `on-hold`, `parked`, `accepted-no-change`). The items it called "parked rather than killed" are also tasks labelled `parked`, priority low, linked to decision-17.
- "Study-only, not planned": tasks labelled `study-only`, priority low.

Area labels: research, numeric, aggregation, roster, prompts, ops, mantic, prediction-market, code-quality, cost, timing, backtest, analysis.

## Working rules

- Read `backlog task list --plain` and `backlog decision list --plain` before proposing work. `backlog task <id> --plain` and `backlog search <query> --plain` show details.
- Check the decisions before re-proposing anything: a rejected idea returns only with new evidence, and the decision text says what that evidence is.
- New levers and parked ideas are tasks (`backlog task create`), with the receipt path, date and numbers in the description. New rejections are decisions (`backlog decision create "<title>" -s rejected`) with the same receipts. Do not rename or delete a decision; add a new one that supersedes it.
- `.backlog/` is gitignored on purpose (internal notes, like `scratch*/`), so code and docs cite items by id and quoted title, for example `backlog task-17 "Gap-fill v2 driver on gpt-6-luna"`.
