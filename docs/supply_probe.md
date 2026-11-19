# The question-supply probe

`scripts/supply_probe.py` (`make supply_probe`, `make supply_probe_mantic`) is the read-only census of
a tournament's questions by post status, INCLUDING `closed`. This note carries the why behind its
design: the incidents it exists for, the API facts it is built around, what each block of the
report means, and the Mantic mode. The code carries one-line pointers here.

## Why it exists

Two consecutive residual rounds' supply projections missed, both for the same reason. A question
that has closed to forecasting but has not resolved yet sits at post status `closed`, and each
round's probe queried only `statuses=resolved` and `statuses=open`, so those questions were
invisible. On 2026-08-31 the summer tournament held 178 posts at `closed`; 26 of them were the
frozen-triple checkpoint cohort the projection was about, and 16 of those were already past their
own `scheduled_resolve_time` (worst 17.1 days). Both probes were scratch scripts, so the fix kept
getting re-lost, hence a tracked utility with tests (`tests/test_supply_probe.py`,
`tests/test_supply_probe_mantic.py`).

`closed` is therefore the whole point of the default `--statuses open closed resolved`; pass
`--statuses` to ask about any other status the API accepts.

## What it reports per slug

- **Status partition**: posts and questions at each requested status. Per-status rows report what
  the API returned for that status; the totals count each post and question once, because a post
  that resolves mid-probe can be paged under both `closed` and `resolved`.
- **Backlog**: UNRESOLVED questions already past their own `scheduled_resolve_time`, worst first,
  with the overdue margin. This is the number that tells a supply projection whether questions are
  late on the platform's side (nothing we can do) rather than missing from our pull. A question
  whose schedule is absent or unparseable is disclosed as such, never imputed on time. A naive
  `now` handed in by an analysis script is stamped UTC before the subtraction, so it answers
  instead of raising against the tz-aware scheduled times.
- **Forfeit sweep**: every question on a `closed` or `resolved` post that the bot never forecast at
  all, with its open-to-close window. Only those two statuses count: an OPEN question the bot has
  not forecast YET is not a forfeit.
- **Miss rate per UTC release hour**: the forfeit-eligible questions grouped by the hour of day of
  their `open_time`, with `miss_rate = no_forecast / (forecast + no_forecast)` (blank when nothing
  in the bucket was decided), a total row, and the distribution of the realized open-to-close
  window in minutes (min, median, max). Rendered for both platforms and carried in the JSON dump
  as `by_release_hour` and `window_minutes`.

## The forfeit sweep

This is a supply question, not a scoring one: a forfeited question never reaches the performance
dataset (the collector drops a question with no `my_forecasts.latest`), so a sweep that starts from
questions the bot intook cannot see one. The 2026-09-01 residual round found the triple era had
lost SIX questions to delivery where the prior sweep saw one: q44801 to a cron gap, q45085 to a
late submit against a 12:00 close, q45093 / q45374 / q45375 to cancelled runs, q45216 to a
retroactive close. That is why the sweep belongs in the weekly read rather than in a round's
scratch scripts.

Resolving "did we forecast this" on Metaculus needs `my_forecasts`, which the posts LIST payload
carries only under `with_cp=true` (the 2026-09-09 review measured it with the bot's token on the
summer tournament: present on 100 of 100 resolved questions with the flag, on none without, and
the four the list page called `no_forecast` matched their own detail payloads), so every list GET
carries the flag and the sweep issues a per-post detail GET only where a question still reads
`unknown`. A question's `forecast_state` is one of `forecast`, `no_forecast`, `unknown`;
`unknown` means no payload answered (a list page the sweep did not enrich, a detail page that
answered with a null block, or a detail GET that failed). Unknown questions are counted and
disclosed rather than filed as forfeits: under-reporting a forfeit is recoverable, calling a
forecast question forfeited is not. Forfeits list newest window first, because a weekly read is
about what we just lost; the window length rides each row instead of ordering it, since a short
window and a stale one are different diagnoses and only one of them is urgent.

Two report lines guard the reading. When every eligible question is `unknown`, the report names
the remedy (run without `--no-forfeits` on Metaculus; set `MANTIC_TOKEN` on Mantic). When no
question on a slug carries a bot forecast at all, the report says to check the identity first,
because a non-bot token (or, on Mantic's public path, a wrong `MANTIC_BOT_USER_ID`) is far
likelier than a total forfeit.

## The Mantic mode

`--platform mantic` runs the same census against Mantic's Crucible (competitions.mantic.com, a
Metaculus fork). Its read endpoints are public, so `MANTIC_TOKEN` is optional and only widens what
the sweep can classify:

- `with_cp=true` rides every list GET, token or not. It is what puts a RESOLVED question's public
  spot-time snapshot on the list page (without it every post's `score_data` is `{}` and nothing
  classifies, which is what the tokenless mode read while the flag was sent only under a token),
  and under a token it also puts `my_forecasts` there, so closed-but-unresolved questions classify
  too. No detail GET is ever issued on Mantic. A 100-post page with the flag is
  about 6 MB and 4 to 5 seconds, well inside the 45-second request timeout.
- Wherever `my_forecasts` does not answer (no token, or a payload that carries the block as
  null), a RESOLVED question is classified from that snapshot,
  `question.aggregations.recency_weighted.score_data.disagreement_forecasts.forecasts[]`, one entry
  per competitor keyed by `author_id`, read against `MANTIC_BOT_USER_ID` (81, `nostreambot-bot`).
  The snapshot is exactly what the platform scores, which is why every eventually-resolved
  question is measurable without a secret. A closed-but-unresolved question (null
  `disagreement_forecasts`), an open one (`score_data: {}`) and a resolved-but-unscored one (empty
  list) read `unknown`. Verified 2026-09-08 on the 556-post public corpus, read with the flag: the
  list on 520 of 520 resolved posts, on 4 of 32 closed ones. The two recorded pages under
  `tests/data/` are the same two Series 1 posts read with and without it.
- The snapshot's caveat, stated in the report header rather than modelled: it names one competitor
  fewer than the post's own `nr_forecasters` on nearly every resolved question (507 of the 524
  snapshot-bearing posts in that corpus; exact on 12, short by two on 4, one more on 1), so it can
  read `no_forecast` for an account that did forecast. Which account is missing varies across all
  eleven that appear in the snapshots, and the cause is not established (a forecast withdrawn or
  superseded before spot time is one candidate). No Mantic question the bot forecast has resolved
  yet, so the PRESENT branch has never been exercised against a live payload: read a public-path
  `no_forecast` as provisional until the first Series 2 question the bot forecast resolves and
  `make supply_probe_mantic` shows `MANTIC_BOT_USER_ID` in its snapshot.
- `forecaster_id` / `not_forecaster_id` are never sent: they answer 403 unauthenticated and are
  redundant with `my_forecasts` under a token.

The platform seams (the API base URL and the posts URL derived from it, token env and whether it is
required, default slugs, the forecast-state read, the list params, whether the sweep needs detail
GETs, the bot user id, and the three report prose strings) live in one `PlatformProbe` table in
`scripts/supply_probe_platforms.py`; paging, backlog, forfeit and rendering logic are shared. How
to read the per-hour table for the cron-cadence decision: `docs/operations.md` "Scheduling
reliability".

## API facts it is built around

- **Paging stops on the first short page.** The Metaculus tournament-filtered posts list gives no
  usable total, and Mantic advertises `next` past its last page with `count` null (the Series 1
  walk returned 100, 100, 100, 100, 100, 20, then 0), so page length is the only end-of-results
  signal either platform gives. `MAX_PAGES` (40, so 4,000 posts per status) bounds the walk, an
  order of magnitude above any slug we probe.
- **Bounded 429 retry, spaced requests.** The Metaculus endpoint rate-limits aggressively right
  after a full performance pull, so every request carries a linear-backoff retry (linear, not
  exponential, because the endpoint recovers in seconds and the probe pages several slugs) and
  pages are spaced one second apart. Per-post detail GETs are spaced at the scoring pull's own
  `MIN_REQUEST_INTERVAL_SECS` (2 s), imported rather than copied so the two read-only Metaculus
  walkers cannot drift into different politeness. The sweep logs progress every 25 posts: at
  2 s spacing that is an INFO line about every 50 seconds, enough to tell a slow sweep from a
  wedged one without burying the slug's own summary; every GET is a DEBUG line for a per-URL trace.
- **Dead slugs are error rows.** A slug that errors (the bare `metaculus-cup` slug and an unknown
  Mantic slug both answer 400) is reported as an error row and the survey continues, so one dead
  slug cannot hide the live ones. That also makes the probe the cheapest way to watch for the fall
  cup opening questions: the `metaculus-cup-fall-2026` row goes from zero posts to non-zero on the
  day it does. Only `requests.RequestException` is caught; anything else is a contract break and
  crashes.
- **The vetted host is the host the token goes to.** Each platform's `base_url` is one field of its
  `PlatformProbe`; `main` vets exactly that string with `verify_api_identity`, and the posts URL is
  derived from it, so the two cannot diverge. The Metaculus base is read off
  `MetaculusClient().base_url` rather than hardcoded, so it honors a `METACULUS_API_BASE_URL`
  override the same way the bot's own fetches do, including one set in a `.env` file: importing
  `metaculus_bot.constants` is what loads `.env` / `.env.local`, and the table is built after that
  import. The Mantic base is `MANTIC_API_BASE_URL`. Both pairings are pinned in the tests.
- **Question unwrapping is shared with the scoring pull** (`questions_on_post` in
  `performance_analysis.collector`): both read the same posts list, and a probe that counted
  questions differently from the pull it exists to project would be answering a subtly different
  question.
- **Default slugs** are the repo's own season slugs, deduplicated so re-pointing
  `METACULUS_CUP_ID` at the dated fall slug collapses two rows into one instead of probing it
  twice; minibench comes from forecasting-tools, the same spelling `cli.py` forecasts on. On
  Mantic the default is `MANTIC_TOURNAMENT_ID`.

## Cost

Read-only and free: it hits only the platform's posts list and post detail, with no LLM call, no
research provider and no publish, so it sits outside the repo's cost gate. `make supply_probe_mantic`
needs no secret at all.
