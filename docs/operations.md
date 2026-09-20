# Operations & configuration

How to set up, configure, and run the Metaculus forecasting bot. This is the
reference for a human operating the bot: local setup, API keys, the environment
flags, the GitHub Actions workflows, cost discipline, and the telemetry you can
grep after a run.

For a code-level map of the pipeline, read `docs/architecture.md`; `AGENTS.md` at
the repo root is the terse agent-facing starting point and indexes both. This doc
points at code by file and symbol name, so `rg <symbol>` takes you there.

## Setup

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/). The project
uses uv for everything. There is no poetry, conda, or pip in this repo.

```bash
uv sync --dev        # create/update .venv from uv.lock (or: make install)
cp .env.template .env
```

Then fill in `.env` with your keys (see below). Never commit secrets: `.env` is
gitignored and is the only place real keys should live locally. Run any command
inside the project environment with `uv run <cmd>`: uv resolves the in-project
`.venv` automatically, so you never activate it by hand.

Quick sanity checks (all free, no paid APIs):

```bash
make test            # full pytest suite (self-contained, needs no keys)
make lint            # ruff check
make typecheck       # basedpyright (must stay at 0 errors)
make check_credits   # print OpenRouter balances for both keys
```

### Git hooks

`make precommit_install` installs both hook types (`pre-commit` alone installs only
the first):

- **at commit**: the ruff hooks (check with `--fix --unsafe-fixes`, then format), plus
  `no-commit-to-main`, which refuses a commit whose HEAD is `main`. `main` is
  ruleset-protected on GitHub (PR required, `lint` + `test` required), so a direct push is
  rejected, but only at push time, once the commits already sit on local `main` and have
  to be replayed onto a branch. The guard moves that refusal to commit time, where the fix
  is one `git switch -c`. Its message names the recovery command and the `git commit
  --no-verify` bypass; `scripts/hooks/no_commit_to_main.sh` is a `language: script` hook,
  so it must stay executable in the index (mode `100755`, which `tests/test_no_commit_to_main_hook.py`
  asserts, along with the behavior on a feature branch and on a detached HEAD).
- **at push**: the full pytest suite, `uv run --frozen pytest --cov=metaculus_bot`, which
  is the command `.github/workflows/ci.yaml` runs. ~105s, too much friction per commit but
  the right price on the thing reviewers see.

Run the hooks by hand with `make precommit` (staged files) or `make precommit_all` (the
whole tree).

Two things can block or stale the install on a checkout that predates the uv migration:

- **A pre-existing `core.hooksPath` makes pre-commit refuse to install** ("Cowardly
  refusing to install hooks with `core.hooksPath` set"). Check where it points with
  `git config --show-origin --get core.hooksPath` against `git rev-parse --git-path hooks`.
  If they match, the setting is redundant (it names git's own default hooks directory),
  so `git config --unset-all core.hooksPath` unblocks the install and changes nothing
  about where git looks for hooks.
- **The generated `.git/hooks/pre-commit` can be stale.** Hooks installed before the uv
  migration hardcode an `INSTALL_PYTHON` under the old conda env, which still resolves on
  disk and so fails confusingly rather than obviously. `make precommit_install` regenerates
  the file against the current interpreter.

## Season-start checklist

Before a new tournament or Cup season opens its first question. These are
operator steps: the reads below are free metadata pulls with no inference spend,
but they are network calls and they feed a roster decision, so an implementing
session does not run them: it proposes, the operator runs and decides.

- **Resolve "latest per vendor" from a LIVE model-list read, never from memory.**
  The roster design (`FORECASTER_LLMS` in `metaculus_bot/llm_configs.py`) is the
  newest frontier reasoning model from each vendor, one slot each, and nothing in
  the repo can say what that currently resolves to. The 2026-08-31 gemini-slot
  review found that a roster decision needs this one read before anything else.
  OpenRouter's public models endpoint lists every slug with `created`, its
  listing time as a Unix timestamp (per the endpoint's OpenAPI schema):

  ```bash
  curl -s https://openrouter.ai/api/v1/models \
    | jq -r '.data[] | [.id, .created] | @tsv' | sort
  ```

  Filter per vendor prefix (`openai/`, `anthropic/`, `google/`, `x-ai/`) and
  read the newest `created` per vendor:

  ```bash
  curl -s https://openrouter.ai/api/v1/models \
    | jq -r '.data[] | [.id, (.created | todate)] | @tsv' \
    | grep -E '^(openai|anthropic|google|x-ai)/' | sort -t$'\t' -k2
  ```

  Then check, before touching the roster: the slug is a reasoning model, not a
  mini/flash/fast tier or a `:free` route; its provider is on the donated key's
  allowed list (`DONATED_KEY_PROVIDERS` in `fallback_openrouter.py`), or the slot
  knowingly bills the personal key like the pinned Google Pro slot
  (`DONATED_KEY_BLOCKED_GOOGLE_MODELS`); and its reasoning-effort enum accepts
  the tier the slot is configured for (the OpenAI ceiling is `xhigh`; `max` is
  Anthropic-only).
- **A roster change is a config-era boundary.** Residual analysis buckets by the
  merge-to-main timestamp, so make any swap once, in the same merge as everything
  else that shifts the forecast distribution, before the first question, never
  mid-window (`FUTURE.md`, "FREEZE the triple").
- **Refresh the tournament constants** (`TOURNAMENT_ID`, the date checks in
  `constants.py`) and flip the cup reminder off once configured
  (`FALL_CUP_CONFIGURED`), or every scheduled run reddens on the reminder.
  `TOURNAMENT_END_DATE` is the project's `forecasting_end_date`, not its
  `close_date`. On `fall-futureeval-2026` those are 2027-01-06 and 2027-03-05.
- **Read the project object before editing a slug, and note the route.** A project
  is served at `/api/projects/tournaments/<slug-or-id>/`; a bare
  `/api/projects/<id>/` 404s for every id, which reads exactly like "this project
  does not exist" rather than like a wrong route. Slug and numeric id both resolve
  on the working route. The list endpoint `/api/projects/tournaments/` omits
  anything whose `visibility` is `unlisted`, which is the state a new season sits
  in before its first question, so ABSENCE from that list is not evidence a
  project does not exist. Fetch the candidate directly, or walk the id space.
- **Take a question-supply census with `make supply_probe`** (Metaculus), or
  `make supply_probe_mantic` for the Crucible tournament (see "Scheduling reliability" under
  the Mantic workflow for its per-release-hour miss table). It counts posts and
  questions at each status per tournament slug, and unlike the two scratch probes
  it replaces it counts post status `closed` (closed to forecasting but not yet
  resolved), which is what made two consecutive residual rounds' supply
  projections miss. It also lists the backlog of unresolved questions already past
  their own `scheduled_resolve_time`, worst overdue first, which is how you tell
  "Metaculus is late resolving" from "our pull is missing questions". It also sweeps
  FORFEITS: every question on a `closed` or `resolved` post that the bot never
  forecast at all, newest window first, with each window's length in hours. That
  sweep exists because a forfeited question never enters the performance dataset, so
  nothing downstream of the scoring pull can see one. The 2026-09-01 residual round
  found six lost to delivery (a cron gap, a late submit, three cancelled runs, one
  retroactive close) where the prior sweep had found one. Resolving "did we forecast
  this" needs `my_forecasts`, which the posts list does not reliably carry, so the
  sweep issues one extra read-only detail GET per closed/resolved post that the list
  page did not already answer for; pass `ARGS="--no-forfeits"` to skip that, at the
  cost of every question's state reading `unknown`. A question whose state stays
  unreadable is reported as `unknown` rather than filed as a forfeit, and a slug
  where NOTHING carries a bot forecast prints a warning to check that
  `METACULUS_TOKEN` is the bot's own token before believing the number. Default slugs
  come from the repo's own constants (`TOURNAMENT_ID`, `METACULUS_CUP_ID`,
  `FALL_CUP_SLUG`, plus minibench off `MetaculusApi.CURRENT_MINIBENCH_ID`), so it
  needs no arguments; scope or redirect it with
  `ARGS="--slugs metaculus-cup-fall-2026 --output /tmp/supply.json"`. Read-only and
  free (the Metaculus posts list and post detail only, no LLM, research, or publish
  call), so it sits outside the cost gate. A dead slug renders as one error row and the rest
  report normally, which makes this the cheapest way to watch for the fall cup
  opening: the `metaculus-cup-fall-2026` row goes from zero posts to non-zero on
  the day it does.

### Fall 2026 season: what was done on 2026-09-03, 2026-09-06 and 2026-09-09

Metaculus granted $1,500 of API credits for the bot to compete in both the fall
Metaculus Cup and the fall bot tournament. Landed in the repo:

- `METACULUS_CUP_ID` now holds `metaculus-cup-fall-2026`. It used to hold the
  undated `metaculus-cup` slug and rely on Metaculus redirecting it; Metaculus
  rejects that slug now (the posts list answers HTTP 400 for
  `tournaments=metaculus-cup`, re-verified 2026-09-03 with `make supply_probe`), so
  a cup run under it would have found no questions and forfeited the season with
  nothing in the log saying why. Verified read-only against
  `/api/projects/tournaments/metaculus-cup-fall-2026/`: project id 33108, name
  "Metaculus Cup Fall 2026", `start_date` 2026-08-28T12:00:00Z,
  `forecasting_end_date` 2027-01-01T00:00:00Z, `close_date` 2027-01-04T00:00:00Z,
  `score_type` `peer_tournament`, `visibility` `unlisted`,
  `bot_leaderboard_status` `exclude_and_show`, `questions_count` 0. The cup is
  open but had published nothing yet, and bots forecast on it outside the human
  leaderboard. `exclude_and_show` is what every recent cup season carries (fall
  2025, spring 2026 and summer 2026 read identically), so it is the cup's normal
  setting rather than anything fall-specific. `visibility` reads `unlisted` where
  those older seasons read `normal`, which is the pre-first-question state.
  **One analysis consequence: the cup scores on `peer_tournament`, not
  `spot_peer_tournament` like the bot tournament.** Cup records therefore carry a
  coverage-scaled `peer_score` and no spot peer, so they cannot be pooled with
  tournament records on one score field. `performance_analysis/platform_scores.py`
  already handles that (`RankingScore.tier` keeps spot-scored and peer-only records
  in separate sort tiers), but any new cut written over fall data has to respect it.
- `FALL_CUP_CONFIGURED` is True, so the dated reminder that would have reddened
  every run from 2026-09-15 is discharged, and its CI time bomb in
  `tests/test_tournament_dates.py` is now a pin that the cup stays pointed at a
  dated slug. Re-arming it for the spring 2027 cup means re-dating
  `FALL_CUP_REMINDER_DATE` and setting the flag back to False.
- `run_bot_on_metaculus_cup.yaml` is at full parity with
  `run_bot_on_tournament.yaml` (same env block, step caps, Playwright install and
  artifact upload) and moved from `3 0 */2 * *` (00:03 every second day) to
  hourly at :13/:33/:53. Hourly costs nothing when nothing is new, because
  `skip_previously_forecasted_questions` is on, and it removes most of the
  open-to-forecast latency that forfeited six triple-era questions. The minutes
  are staggered off the tournament's :03/:23/:43 and minibench's :08/:38 because
  the three workflows are in separate concurrency groups, so a shared minute means
  simultaneous runs rather than a queue.
- Research records are labelled by run mode (`cli.persisted_tournament_id`), so
  cup runs archive under the cup slug instead of the bot tournament's.
- `TOURNAMENT_ID` now holds `fall-futureeval-2026` (project 33121). The project
  was read directly from `/api/projects/tournaments/33121/` on 2026-09-06:
  `start_date` 2026-09-28, `forecasting_end_date` 2027-01-06, `close_date`
  2027-03-05, `score_type` `spot_peer_tournament`, and
  `bot_leaderboard_status` `bots_only`. `TOURNAMENT_END_DATE` uses the
  forecasting end date, as it must; its two-week hard stop is 2027-01-20.

One adjacent thing the grant settles: credit alerting is back ON.
`make check_credits` on 2026-09-03 reads the donated key at **$1,449.19 remaining
of a $2,300 limit**, so a credit shortfall is real news again rather than the
expected state, and `CREDIT_ALERT_RESUME_DATE` was moved up from 2026-09-10 to
**2026-09-03** rather than left to expire. `OPENROUTER_CREDIT_FLOOR_USD` moved with
it, from $1.00 to **$100.00**: the operator cannot refill this key (Metaculus does),
so the warning has to arrive with runway left to ask, and $100 is roughly 56
questions at the measured $2.07 to $2.21 a question ($1.79 of it on the donated
key; the "$0.38-0.41" quoted at the time was an OpenRouter-only lower bound on one
key, see "Per-role spend"). A $1 floor would have fired only once the key was
already dry, which on an hourly cup cron is an hourly red check that arrives too
late to act on.

The one step no merge could do is done as well: `run_bot_on_metaculus_cup.yaml` was
`disabled_manually` on GitHub, a per-workflow state no file in this repo can change,
and the operator enabled it for the season; `gh workflow list --repo
No-Stream/metaculus-bot --all` read it as `active` on 2026-09-09. Nothing in the
repo warns when a workflow is disabled; the way to notice is a supply-probe row
showing a tournament's questions with no bot forecasts.

## API keys and the shared-vs-personal key model

The bot needs several credentials. `.env.template` lists them with inline
notes; copy it and fill in real values. The one piece of routing that trips
people up is the two OpenRouter keys.

- **`OAI_ANTH_OPENROUTER_KEY`: donated / shared.** Metaculus provides credits
  on this key for OpenAI, Anthropic, and Google models routed via OpenRouter.
  Its server-side allowed-providers list is locked to those three, so anything
  else (Grok via x-ai, Qwen, Perplexity) returns 404 on this key. This is the
  only shared credential in the bot; despite the name it covers all three
  providers, not just OpenAI and Anthropic. `DONATED_OPENROUTER_KEY_ENABLED`
  (default `true`) is its master switch: `--mode mantic` requires it to read false
  and fails shut otherwise, because the key was donated for Metaculus tournaments
  (see "Mantic" below).
- **`OPENROUTER_API_KEY`: personal.** Pays for what the donated key can't
  (Grok, Qwen, Perplexity-via-OpenRouter) and serves as the fallback when the
  donated key hits a credential, credit, or allowed-providers error. The
  fallback wrapper is `FallbackOpenRouterLlm` in
  `metaculus_bot/fallback_openrouter.py`.
- **`GOOGLE_API_KEY`: personal.** The operator's Google AI Studio key on a
  billing-enabled project. Powers the Gemini grounded-search provider and gap-
  fill v2's document reads. There is no donated Google AI Studio path. In CI
  this is stored as the `GEMINI_API_KEY` secret and surfaced to the workflow as
  `GOOGLE_API_KEY` so the `google-genai` SDK picks it up.

Gemini has two separate routes, which is the other easy thing to confuse:

- **OpenRouter Gemini** (forecaster / stacker / summarizer slots) routes
  donated-key-first with personal-key fallback, controlled by
  `GEMINI_USE_DONATED_OPENROUTER_KEY` (default `true`, since 2026-06-16). It is on
  by default because Metaculus raised the Google rate limits, so the donated key
  now serves most Gemini models. Verified by live call, `gemini-3.5-flash` and
  `gemini-3.1-flash-lite` both succeed on it. Setting the toggle to a false-y
  value (`false`/`0`/`no`) forces personal-key-only routing for ALL Gemini; the
  three prod workflow YAMLs and `test_bot.yaml` pin it to `'true'` explicitly.

  **Known exception:** the Gemini Pro forecaster slot is PINNED to the personal
  key by the `DONATED_KEY_BLOCKED_GOOGLE_MODELS` blocklist in
  `fallback_openrouter.py` (read the blocklist for which models it currently
  covers). `should_route_via_donated_key` returns `False` for anything on it even
  with the toggle ON, so there is no donated attempt, no 429, and no
  personal-key-fallback-counter bump (which would otherwise redden CI on every
  question), and a credit error on one of those models is always a personal-key
  issue. It is pinned rather than falling back because that model routes through a
  free-tier Google AI Studio BYOK key on the donated account with no Pro free tier
  (quota 0 → `is_byok:true` + `FreeTier limit: 0`). This is a temporary workaround
  tagged `TODO(gemini-3.1-pro-donated)` in code: remove the blocklist entry once
  Metaculus fixes the BYOK routing (enable Cloud billing on the BYOK key's GCP
  project, remove the Google AI Studio BYOK integration so native OpenRouter
  Google credits are used, or disable "Always use for this provider" on that BYOK
  key), then re-verify with one live call. See
  `metaculus_bot/fallback_openrouter.py:should_route_via_donated_key` and
  `FUTURE.md` "Gemini on the donated OpenRouter key".
- **Gemini grounded search** (`research/gemini_search.py`) always uses the
  personal `GOOGLE_API_KEY`. The donated toggle does not touch it, and neither
  does anything else on the OpenRouter side. What that key costs is its own
  subsection below.

Other keys, all personal, no shared variants: `METACULUS_TOKEN`, `MANTIC_TOKEN`
(the Crucible bot token, read only in `--mode mantic`), `ASKNEWS_CLIENT_ID`
+ `ASKNEWS_SECRET`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`, `FRED_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. The two direct provider keys only matter
if you bypass OpenRouter; most flows route through OpenRouter and don't need
them.

`SEC_EDGAR_CONTACT_EMAIL` is not a key but a contact address (also personal): the SEC EDGAR
client puts it in the fair-access User-Agent, and the client declines to dial without it. The
operator created it as a GitHub Actions repository secret on 2026-09-09, and the four
`run_bot_on_*.yaml` prod workflows now pass it into their `env:` blocks as
`SEC_EDGAR_CONTACT_EMAIL: ${{ secrets.SEC_EDGAR_CONTACT_EMAIL }}`. Its `.env.template` placeholder
stays commented; the real address lives only in the untracked local `.env`. Until the known-API
registry is wired into the fetch ladder nothing reads it at run time, so an unset value is
harmless (the EDGAR client simply stays inert).

Diagnosing auth errors: an OpenRouter 401/402 on an OpenAI or Anthropic call
means suspect the donated key first (it's always tried first for those
providers). A 401/402/credit error on an OpenRouter Gemini call also means the
donated key first, since OpenRouter Gemini routes donated-first by default with
personal fallback, unless `GEMINI_USE_DONATED_OPENROUTER_KEY` has been forced
OFF, in which case suspect `OPENROUTER_API_KEY`; and anything on
`DONATED_KEY_BLOCKED_GOOGLE_MODELS` is pinned to the personal key with no donated
attempt, so a credit error on one of those models is always a personal-key issue.
A 401/402 on Grok, Qwen, or Perplexity is always the personal key
(the donated key 404s on those), and so is every OpenRouter auth error on a
`--mode mantic` run, where the donated key is switched off entirely. A `google-genai` 401 or quota error is always
`GOOGLE_API_KEY`. A `403` splits three ways, with the reported status deciding the
branch and the spend-cap phrase outranking it:

- Body says `Key limit exceeded`: a drained spend cap. Falls back to the
  personal key and is credit-classified, whatever status came with it.
- Body says `no allowed providers`, `guardrail`, or `data policy`: scoped to
  the donated key's routing, so the personal key genuinely can serve the call.
  Falls back, but is NOT credit-classified.
- Anything else: a moderation or permission refusal. Does not fall back, since
  both keys would refuse the same prompt. Those two phrasings are the only ways
  out of this branch, so it holds even when the body happens to contain ordinary
  credit English like "insufficient funds": on a reported 403 the body is the
  least trustworthy input we have (see the `flagged_input` prompt replay below),
  and credit wording there classifies as neither credit nor a key issue.

The routing half of that decision (fall back to the personal key, or don't)
lives in `should_retry_with_general_key` (`fallback_openrouter.py`); the credit
classification is `_is_credit_failure`. Whether a spend-cap 403 is additionally
SUPPRESSED from CI alerting is a third, separate question, answered by the
`/auth/key` probe described below.

A `429 rate limit` is not a key defect but does fall back, since BYOK quotas are
per-key. See "What a dry donated key actually returns" below.

### The donated-key fallback wrapper and its classifiers (`fallback_openrouter.py`)

This subsection is where the code pointers in `metaculus_bot/fallback_openrouter.py` land. The
incident-level story of the drained donated key, the `/auth/key` drained-versus-revoked probe and
the three tiers of the credit arbiter is told once under "What a dry donated key actually returns
(and the drained-vs-revoked probe)" in the credit-telemetry section below, and the exit arithmetic
that consumes the fallback counters is under "The end-of-run breakdown and the exit ladder". What
follows is the per-element detail.

#### The donated-key provider set (`DONATED_KEY_PROVIDERS`)

These are the providers covered by the Metaculus-donated OpenRouter key
(`OAI_ANTH_OPENROUTER_KEY`), whose server-side allowed-providers preferences are locked to this
set. A model routed through any other provider 404s on the donated key, which is why the donated
key is preferred only for these. The environment variable name stays `OAI_ANTH_OPENROUTER_KEY` for
backward compatibility with the operator's GitHub secret, so adding `google` to the set did not
require changing the secret name.

#### The Google blocklist (`DONATED_KEY_BLOCKED_GOOGLE_MODELS`)

Google models that must not route through the donated key even when the donated-Gemini toggle is
on. The reason, the free-tier BYOK quota that forces it, and the conditions for removing an entry
are in the "Known exception" paragraph above; the `TODO(gemini-3.1-pro-donated)` tag lives beside
the constant in code. One mechanical detail belongs here: entries are matched by prefix
(`startswith`), so the bare GA slug and every suffixed variant (`-preview`,
`-preview-customtools`, OpenRouter `:free` and route suffixes) are all covered by one entry.

#### `should_route_via_donated_key`

The master switch runs first. While `DONATED_OPENROUTER_KEY_ENABLED`
(`constants.donated_openrouter_key_enabled`) is false-y, the predicate returns False for every
slug before any provider rule runs. A Mantic run sets it false, because Metaculus donated the key
for its own tournaments and a run for another platform must spend only the operator's personal
keys. Every reader of the donated environment variable gates on this predicate: the builder in the
same module, `api_key_utils.get_openrouter_api_key`, and the gap-fill v2 transport in
`research/agentic/llm.py`. One false-y value is therefore personal-only routing everywhere, and
the key-swap fallback wrapper is never built. It is an environment variable set before the process
starts rather than a runtime toggle, because the roster's module-level `GeneralLlm` objects in
`llm_configs` freeze their `api_key` at import. Unset means on, so Metaculus runs are unchanged.

Past the switch, the predicate matches OpenRouter model slugs of the form
`openrouter/<provider>/<model>` against `DONATED_KEY_PROVIDERS`. It returns False for
non-OpenRouter slugs such as `perplexity/sonar` and for unrecognized providers such as `x-ai` for
Grok. Google routing is additionally gated on `GEMINI_USE_DONATED_OPENROUTER_KEY`, which defaults
to on (`gemini_use_donated_openrouter_key`): after Metaculus raised the Google rate limits on
2026-06-16 the donated key serves most Gemini models, for example `gemini-3.5-flash` and
`gemini-3.1-flash-lite`. Setting that variable to a false-y value forces personal-key-only routing
for all Gemini.

The one exception past both gates is `DONATED_KEY_BLOCKED_GOOGLE_MODELS` (`gemini-3.1-pro`), which
is pinned to the personal key with no donated attempt, no 429 and no fallback-counter bump. Those
models run through a free-tier Google AI Studio BYOK key on the donated account that has no Pro
free tier (limit 0, so every call 429s), so a donated attempt would always fail over to personal
anyway. The pin is a temporary workaround; see the `TODO(gemini-3.1-pro-donated)` tag on that
constant, the "Known exception" paragraph above, and `FUTURE.md` "Gemini on the donated OpenRouter
key".

#### Prompt-echo truncation (`_PROMPT_ECHO_MARKERS`, `_without_prompt_echo`)

The markers say where a provider's error message stops being the provider's words and starts being
ours. Two carriers, both verified: an OpenRouter moderation 403 body replays up to about 100
characters of the prompt as `flagged_input`, and forecasting-tools' empty-completion guard raises
`RuntimeError("LLM answer is an empty string. The model was ... and the prompt was: <up to 2000
chars>")`.

Everything past a marker is text we sent, so it must not classify anything. Measured against the
963-bundle research archive, our own prompts contain "402" in 13.0% of bundles, "guardrail" in
1.9%, "unauthorized" in 4.7% and "deprecated" in 0.1%, so without the truncation a benign
zero-output blip on a question about a $402M revenue target billed the personal key, counted as an
expected empty wallet, and took a degraded run green.

The marker itself is kept rather than cut away: `flagged_input` is OpenRouter's own field name and
one of `_MODERATION_CUES`, so cutting at the marker rather than after it would disarm the veto on
the very bodies this exists to defend against. Only text cues read the truncated string. The
reported `status_code` is an int on the exception and cannot be echoed, so `_is_status` and
`llm_status_code` keep reading the full message.

#### Status detection over message digits (`_is_status`)

When the provider reported a status, that integer is the only numeric evidence consulted, and the
message is not. litellm formats the message as `APIError: {provider} - {raw body}`, and an
OpenRouter body carries a 64-hex key hash that has a small but non-negligible chance of containing
one of 401, 402, 403, 429, 502 or 503 (derived in
`test_key_hash_status_collision_is_small_but_nonnegligible`), plus, on a moderation refusal, up to
about 100 characters of our own prompt in `flagged_input`. Matching digits there reads coincidences
as statuses in both directions: a stray "429" sends a moderation 403 to the paid key for a call
that will refuse again.

With no status reported, the substring match is the fallback, but on the echo-stripped message
rather than the raw one. Without that, the digit fallback reopened at every status the exact hole
the prompt-echo truncation closed for the credit cues: a forecasting-tools empty-completion
`RuntimeError` replays up to 2000 characters of our prompt, so a question about "S.429 (the
Fentanyl Act)" or a bill numbered 401 read as a rate limit or a bad credential and billed the paid
key for a call that would return empty again. Measured against the 989-bundle research archive,
"429" appears in 10.2% of our own prompts and "401" in 13.8%, comparable to the 13.0% for "402"
that motivated the original truncation.

Truncating here costs the plain-`Exception` callers nothing, which is why the earlier carve-out for
them was unnecessary: `_without_prompt_echo` only cuts at an echo marker, and strings like "401
unauthorized" or "429 too many requests" carry none, so they pass through byte-identical (pinned in
`test_plain_status_strings_survive_echo_stripping`).

#### The credit arbiter (`_is_credit_failure`, `is_credit_caused_error`)

`_is_credit_failure` is the one arbiter of "the key is out of money". Routing
(`should_retry_with_general_key`) and alerting (`is_credit_caused_error`, and through it the
credit subset counter in `record_donated_key_fallback`) both reach the answer through it, so a cue
edit cannot make them disagree. The three tiers and their order are listed under "What a dry
donated key actually returns" below; three pieces of the reasoning behind that order live here.

The spend-cap phrase wins outright because it is OpenRouter's own wording for a drained per-key
budget, which it reports as 403 rather than the 402 its docs promise, and it will not turn up in a
question about an election. A reported status then decides alone because 402 is "Payment Required"
and has no second meaning, while OpenRouter words refusals as 403, so the int outranks any English
in the body. The failure asymmetry agrees: reading a real 402 as a refusal strands the ensemble on
a dry key, which is the production bug, whereas reading a hypothetical 402-shaped refusal as credit
costs one paid call that refuses again. Everything in the last tier is forgeable by a replayed
prompt, which is why it sits below the moderation veto and reads only `_without_prompt_echo`.

Nothing in the arbiter reads a live balance, since `status_code` is an int already on the
exception. The `/auth/key` probe belongs to `is_suppressible_credit_error` and the alerting
decision, never to routing.

`KEY_LIMIT_EXCEEDED_CUE` is the full phrase for the reason given below under "What a dry donated
key actually returns": the shorter "limit exceeded" is a substring of "rate limit exceeded:
free-models-per-day", so it would classify every 429 as an empty wallet and silently exempt real
rate-limit breakage from alerting for a whole suppression window.

`_GENERIC_CREDIT_PHRASES` collects the "out of money" wording that is ordinary English, so a
forecasting prompt can contain it innocently: "declared insolvent for insufficient funds", "the
ransom demand states payment required". Since a moderation body replays our prompt, these four are
trusted only when nothing says the body is a refusal. They are split out from
`KEY_LIMIT_EXCEEDED_CUE` by specificity rather than by category: that phrase is OpenRouter's own
spend-cap wording and outranks the veto, and these four cannot be given that power.

`_MODERATION_CUES` collects the signals that the body is a content-moderation refusal rather than a
billing problem. litellm builds the message as `APIError: {provider} - {raw body}`, and an
OpenRouter moderation 403 body carries `flagged_input`, up to about 100 characters of our own prompt
replayed back. A forecasting prompt full of dollar figures and bill numbers can easily contain the
token "402", which would otherwise read as an empty wallet, billing the personal key for a call
that will refuse again and exempting a real moderation block from alerting. Word cues only,
deliberately: a genuine 402 links to a key hash with a small but non-negligible chance of
containing the substring "403", and reading that as moderation would break the long-standing 402
fallback. The odds are derived in `test_key_hash_status_collision_is_small_but_nonnegligible`, which
pins two bands, one status alone and any of the six at once; the six-status band does not apply
here.

`is_credit_caused_error` is the public form of the arbiter, so the routing decision in
`should_retry_with_general_key` and the alerting decision in `is_suppressible_credit_error` answer
"was this about money?" the same way. They used to disagree: routing became status-aware while this
stayed text-only, so a terse reported-402 (`APIError(status_code=402, message="wallet empty")`)
fell back to the paid key without being credit-classified, reddening CI on exactly the expected
empty wallet the suppression window exists for.

#### The text cue sets (`_RATE_LIMIT_TEXT_CUES`, `_BAD_CREDENTIAL_TEXT_CUES`, `_ROUTE_SCOPED_TEXT_CUES`)

The rate-limit and bad-credential cues stay live regardless of the reported status, because they
are English wording rather than status digits and so carry no key-hash or prompt-echo risk. They
are what classifies a statusless exception, such as a plain `Exception("401 Unauthorized")` or a
non-litellm caller, and a provider that words the failure without a recognizable status.

`_ROUTE_SCOPED_TEXT_CUES` covers blocks that are scoped to the key's routing rather than to the
request, so the personal key genuinely can serve the same call. Two donated-key quirks produce
them: the server-side allowed-providers preferences ("no allowed providers"), and the Metaculus
data-collection guardrail that excludes OpenAI's native-search endpoint ("No endpoints available
matching your guardrail restrictions and data policy"), for which see `FUTURE.md` "Resolve
OAI_ANTH_OPENROUTER_KEY data-policy block". They are classified by text on purpose, and checked on
every status including 403: OpenRouter returns them as 404, the same status as a plain missing
model, which must not fall back, so the status alone cannot tell the two apart.

#### The fallback decision (`should_retry_with_general_key`)

It falls back to the personal key on:

- 429 Too Many Requests. The donated and personal keys have independent BYOK quotas per provider,
  so a 429 on the primary key does not imply the secondary is also throttled. It falls back
  immediately, with no wrapper-level retry, because the SDK already retried internally before
  raising.
- 401 Unauthorized, an invalid or disabled key.
- 402 Payment Required, insufficient credits.
- 404 with "no allowed providers". The donated key has server-side allowed-providers preferences,
  so a 404 there means the donated key cannot route this model while the general key, which has no
  preferences, can. It is treated as key-scoped so callers fall through to the secondary key.
- 403 carrying spend-cap wording ("Key limit exceeded"), which is how OpenRouter reports a drained
  per-key budget despite documenting credit exhaustion as 402. `_is_credit_failure` classifies it
  upstream of the 403 veto described below, and that ordering is load-bearing.
- The common text cues for these scenarios.

It keeps the donated key on a 403 without credit wording, which is a moderation or permission block
where both keys would refuse; on 502 and 503 upstream or provider outages, which are infrastructure
rather than key-scoped; and on a plain 404 missing model. That last case is why the 404 family is
classified by text rather than status: the same status covers both a route problem the paid key can
fix and a model that simply does not exist.

Numeric detection reads the status the provider reported (`llm_retry.llm_status_code` through
`_is_status`), never digits in the message, for the reason given under "Status detection over
message digits" above. Statusless exceptions still classify on text, unchanged. Direct google-genai
SDK 429s (`google.genai.errors.ClientError` with `code=429`) are out of scope for this wrapper
because they do not flow through OpenRouter; the Gemini search provider handles those separately.

The order inside the function is load-bearing in four places. The deprecation matcher runs first,
before any classification, and only records; the actual `sys.exit` happens later through
`check_deprecation_alerts_and_exit`. The model slug is not available at that point, so the
wrapper's `invoke` re-records with the slug, and this call is the safety net for any other call
site that routes through the predicate. Second, a reported 403 is decided ahead of every text cue,
because the body is the least trustworthy input on this path: a moderation 403 carries
`flagged_input`, and a forecasting question can say "insufficient funds", "payment required", "rate
limit" or "unauthorized" for entirely ordinary reasons, each of which would otherwise send a
content block to the paid key for a call it will refuse just the same. OpenRouter uses 403 for
refusals, so only two shapes deserve a key swap, the spend-cap phrase and a route-scoped block the
personal key genuinely can route. Third, the typed `litellm.exceptions.RateLimitError` branch is
followed by a belt-and-suspenders 429 text check, for the cases where litellm does not raise the
typed exception, such as class drift or non-standard wrapping; and the status read from
`llm_status_code` is authoritative for every numeric branch below it, being `None` only for
statusless exceptions, which then classify on message text exactly as they always have. The
prompt-echo strip applies to the word cues alone: the digit fallbacks inside `_is_status` still see
the whole message, because they only engage when no status was reported and shortening their input
there would change long-standing behaviour for plain `Exception("401 Unauthorized")` callers.
Fourth, route-scoped wording is the last positive signal, and everything it does not match keeps
the key. What reaches there is everything a key swap cannot help: moderation and permission
refusals, where both keys refuse the same prompt, 502 and 503 upstream outages, and a plain
missing-model 404. The explicit negative blocks this ordering replaced were unreachable in the
reported-status regime, because the 403 return and the positive branches had already claimed every
status they named.

#### The fallback counters (`_generic_key_fallback_count`, `_credit_key_fallback_count`, `_donated_404_fallback_count`)

`_generic_key_fallback_count` counts every successful donated-to-personal key fallback whatever the
cause (401, 402, 429, guardrail, 404). The operator pays for every personal-key fallback, so the
signal is loud and auditable whenever the donated key was supposed to cover a call and did not:
`cli.py` folds this count into the end-of-run alert, so a run that quietly leaked spend to the
personal key still turns CI red.

`_credit_key_fallback_count` is the credit-caused subset (402, payment required, insufficient
credit). `cli.py` subtracts it from the generic total while credit alerting is suppressed
(`constants.credit_alerts_active`), because an empty donated key is an expected condition during
that window rather than breakage; after the resume date the subtraction stops and behaviour is
exactly what it was before. Every other cause (401, 404, 429, guardrail) stays alertable.

`_donated_404_fallback_count` is the allowed-providers-404 subset, incremented every time the
donated key returns a "no allowed providers" 404 and the fallback to the general key succeeds. It
is not the alerting input: `cli.py` folds the all-causes total into `alertable`, and adding this
404 count as well would double-count events already inside that total. `cli.py` reads it through
`get_donated_404_fallback_count` only to break it out in the end-of-run log line ("... of which
donated_404=N"), so a stale allowed-providers list upstream is visible without losing the run.

Both counters are subsets of the generic total in the same way, which is what the exit ladder's
"generic adds, at most one subset subtracts" arithmetic depends on. Each counter has a `reset_*`
helper for tests, and each `+=` carries a `# noqa: PLW0603` because a module-global run counter is
the design.

#### The shared accounting seam (`record_donated_key_fallback`)

This function counts and logs exactly one donated-to-personal fallback that is about to happen, and
it is the shared accounting seam for every donated-first call path: the
`FallbackOpenRouterLlm.invoke` wrapper, and gap-fill v2's hand-rolled raw-litellm retry in
`research/agentic/llm.py`. Those two shared the retry predicate but not the accounting, so the
second one fell over silently, with no counter, no `PAID PERSONAL-KEY FALLBACK` warning and no line
in the end-of-run summary, despite firing on every question in all four prod workflows. That was
the `TODO(unify-fallback-routing)`.

Every successful donated-to-personal fallback means a paid personal-key call happened where the
free donated key was expected to cover it, so all of them are counted and logged loudly: silent
personal-key spend must not accumulate unnoticed. The counting invariant is that each event is
counted exactly once in the generic total, and at most one subset counter, either credit-caused or
the 404 "no allowed providers" quirk, also claims it. That is what lets `cli.py` compute
`alertable` as "generic adds, one subset subtracts" without drift. Call the function only when the
fallback will actually be attempted, because a rejected fallback bills nothing and must not count.

It is async because the probe has to leave the event loop while the counting must not. Only the
expected drained-donated-key subset is exempt from alerting, and a key that was revoked or
re-capped to zero produces identical "Key limit exceeded" text, so the verdict comes from asking
OpenRouter rather than from trusting the cue; otherwise the suppression window would have hidden
genuine breakage for six weeks.

The probe is threaded because on the spend-cap 403 path it reaches
`credit_telemetry.classify_donated_key_state`, which does blocking httpx. Called inline from these
coroutines it stalled every concurrently in-flight forecaster and research task, not just the call
that hit the 403, eating into per-question soft deadlines. Probing first also keeps the accounting
below it free of any await. It is bounded because `DONATED_KEY_PROBE_TIMEOUT_S` is a per-operation
httpx timeout rather than a cap on elapsed time, which is worked out under "What a dry donated key
actually returns" below along with the trickling-server measurement. The `wait_for` sits before
`_invoke_once_using_secondary`, so that latency delays the recovery call itself even though routing
was already decided textually, and a degraded-but-alive OpenRouter control plane is exactly what
co-occurs with a spend-cap 403. `wait_for` unblocks the coroutine without killing the worker
thread, which is fine here: the orphan only holds a socket and, under the probe's lock, writes the
cache.

The probe call is guarded because any failure in alerting bookkeeping must leave routing untouched.
The probe promises never to raise and now catches broadly enough to keep that promise, but an
escape there aborted the fallback and left the funded personal key untried, and the production
incident reached through the exception path rather than through a stale balance read. A timeout is
likewise inconclusive, so both degrade to "not suppressible" and stay alertable, exactly as
`unknown` does. That is why the `except Exception` around the probe is deliberately swallowed
rather than re-raised, against the usual fail-fast rule: re-raising there is the bug being fixed,
the decision it is bookkeeping for was already made textually, and the event is still loud, with
the exception logged with its traceback, and still alertable.

From the probe down there is no await, so the whole accounting block runs to completion on the
event loop. That is load-bearing rather than incidental: `+=` on a module global compiles to
LOAD_GLOBAL, INPLACE_ADD and STORE_GLOBAL and is interruptible between bytecodes, so threading this
function as a whole rather than just the probe would let N forecasters failing on one dry key, the
exact 2026-07-26 shape, race the increment, undercount the generic total, and take a degraded run
green.

#### The wrapper and the builder (`FallbackOpenRouterLlm`, `build_llm_with_openrouter_fallback`)

`FallbackOpenRouterLlm` prefers the donated key and falls back to the operator's general key on
credential, credit and allowed-providers errors, for models routed through providers covered by the
donated key. Its `role` argument tags every completion for the `CREDIT_ROLE_SPEND` ledger, with the
primary stamped `donated` and the secondary `personal`; see "Per-role spend (`CREDIT_ROLE_SPEND`)"
below for what that ledger does with the tag.

`invoke` widened `system_prompt` to match forecasting-tools 0.2.92's
`GeneralLlm.invoke(prompt, system_prompt=None)` signature, and threads it through both key paths so
a system prompt survives a donated-to-personal fallback unchanged. On failure it re-records the
deprecation match with the actual model slug; `should_retry_with_general_key` also calls the
matcher with `<unknown>`, and duplicates are fine because `cli.py` only checks that the list is
non-empty for the exit decision, but the log is clearer with the slug. Both awaits in that `except`
block trip Ruff's ASYNC120, because a checkpoint inside `except` can drop the active exception if
the task is cancelled mid-await. That is the correct behaviour here: on success the secondary's
output is returned, on cancellation the secondary is cancelled too, and the primary's exception is
intentionally discarded because the caller asked for a fallback rather than a re-raise.

`build_llm_with_openrouter_fallback` returns the wrapper only when both keys exist and are
distinct. Otherwise it returns a plain `GeneralLlm` on whichever key is available, since no runtime
fallback is possible. Its `role` argument names the spend line every completion of that LLM is
booked under; pass it at every production call site, because a missing role books as `untagged`.

The final plain-`GeneralLlm` return covers the OpenRouter models that bypass the donated wrapper
altogether: providers outside `DONATED_KEY_PROVIDERS` such as x-ai and qwen, Google when
`GEMINI_USE_DONATED_OPENROUTER_KEY` is explicitly off (the default is now on), the blocklisted
Google models (`DONATED_KEY_BLOCKED_GOOGLE_MODELS`, for example `gemini-3.1-pro`) which are pinned
to the personal key even when the toggle is on, and every slug while the
`DONATED_OPENROUTER_KEY_ENABLED` master switch is off, which is a Mantic run. No `api_key` is
passed on that path, so litellm picks up `OPENROUTER_API_KEY` from the environment, mirroring how
Grok via OpenRouter has always worked in production.

#### The model-deprecation tripwire (`_DEPRECATION_ALERTS`, `_DEPRECATION_PATTERNS`)

When OpenRouter retires a model the bot uses, CI should turn red so the operator notices, but the
run must not abort mid-flight, because the remaining ensemble can still publish through fallbacks.
The canonical case is the 2026-05-15 deprecation of `x-ai/grok-4.1-fast`, the native-search model,
which silently 404'd for about two days.

The mechanism: any LLM call site that observes an exception calls
`_record_deprecation_if_matched(model, str(exc))`, and matches are appended to `_DEPRECATION_ALERTS`
as `(model_slug, error_msg)` tuples. Matching is a case-insensitive substring test against
`_DEPRECATION_PATTERNS`, so every distinct error string adds an entry and duplicates are harmless,
since `cli.py` only checks that the list is non-empty. After the bot finishes submitting all
forecasts, `cli.py` calls `check_deprecation_alerts_and_exit()`, which logs a loud banner and exits
1 to fail the GitHub Actions job, or returns silently when nothing was recorded. Where that sits
relative to every other exit condition is under "The end-of-run breakdown and the exit ladder"
below. `has_deprecation_alerts` exists so the end-of-run summary cannot label a run "clean" when
the tripwire is about to turn it red; the alert list is module-private, so callers cannot inspect
it directly.

`_DEPRECATION_PATTERNS` is a deliberately conservative, high-precision set, because a false
positive turns CI red without justification. OpenRouter's deprecation 404s consistently include
both "deprecated" and "recommends switching to" in the message body, but either one alone matches,
to stay robust against minor copy changes.

### Google AI Studio billing and the grounded-search allowance

`GOOGLE_API_KEY` is the operator's personal Google AI Studio key on a
BILLING-ENABLED (paid-tier, prepaid-credit) project, and marginal cost at current
usage is near zero. In CI it is stored as `secrets.GEMINI_API_KEY` and surfaced to
the workflow env as `GOOGLE_API_KEY` so the `google-genai` SDK picks it up. There
is NO Metaculus-donated Google AI Studio key.

Billing mechanics, verified against the ai.google.dev pricing / billing /
google-search docs on 2026-07-17 (don't re-litigate without fetching them again):
Gemini 3.x grounding is paid-tier-ONLY (the free-tier column reads "Not
available") and includes **5,000 free grounded prompts/month shared across all
Gemini 3 models per project, then $14/1k individual search queries**. Multi-query
prompts bill per QUERY on overage, and deep-research prompts fire several.

**Count queries, never prompts.** Current usage is ≈ 70-110 grounded PROMPTS per
month, but the pool is counted per search QUERY and the measured profile is 12.7
queries per prompt (`usage_metadata` plus
`grounding_metadata.web_search_queries` over 113 archived calls, 2026-07-20 →
08-28), so the real draw is ≈ 850-1,400 queries/month, ≈ 17-28% of the allowance.

The spring-2026 billing arc, explained: gap-fill's 5x grounded-call multiplier
plus backtest volume (`backtest_large` = 600 grounded prompts/run) blew past
5,000/month → per-query overage → prepaid-credit top-up debits ("started getting
billed"); the 2026-06-25 resolver migration (`a51617e`) cut the multiplier and new
charges stopped, with a residual ~$1/month of token spend silently drawing down
the prepaid balance. A reconstruction of the whole summer season
(`scratch/fetch_ladder_2026-09-03/`, plan doc
`scratch_docs_and_planning/fetch_ladder_plan_2026-09-03.md`) puts June 2026 alone
at ≈ 6,600 queries, because the pre-06-25 gap-fill resolver added ≈ 4 grounded
calls per question. Any future feature that multiplies grounded-call counts, or
Gemini-grounded backtests at scale, re-eats the same monthly pool.

**Watch item: prepaid-balance exhaustion produces 429s, not surprise charges.** If
Gemini grounded search starts soft-failing across a run, check the AI Studio credit
balance FIRST (`docs/research.md` § Gemini grounded search says the same from the
provider side).

**A third native surface is live.** The resolution-source fetcher's last escalation
rung is a `url_context` read on this same key, gated by
`RESOLUTION_SOURCE_URL_CONTEXT_ENABLED`, which defaults off in code and is set to `true` in every
bot workflow yaml since 2026-09-04. It
shares the reader (`research/url_context_reader.py`) and the model with v2's `read_document`, so
it adds token spend on the operator's personal key rather than a new billing
relationship, and it draws no grounded queries because it retrieves a URL instead of searching.
What it costs is bounded twice over. First by how often the free rungs fail, since a read
happens only for a cited URL no free rung could read and the free `Google-Extended` robots
pre-check declines the hosts that would refuse Gemini anyway. Second by a per-question cap:
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` allows at most two paid reads per question across all
of its cited URLs, the analogue of the Wayback per-question cap. The cap slot is claimed LAST of
all the gates, so a read the wall budget or the robots pre-check declined spends no slot, and a
read the cap itself declines is recorded as a `url_context_cap` skip. That is a different quantity
from `RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS`, which is the SDK retry count for one read.

One change on 2026-09-03 widened the population that reaches this rung, and it is worth knowing
now that the flag is on. The extractor policy now withholds a page whose extraction clears the
400-character chrome floor on short lines alone, and that withhold is `no_resolving_content` with
reason `thin_page`, which is one of the paid rung's trigger statuses. On the calibration corpus
(`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`) that is 9 of the 59 labelled bodies, 6
chrome and 3 ambiguous, which the previous default-only extraction published, against a census of
68 cited successes; each of them reaches the paid rung only when the free rendered rung fails to
rescue it first. The sharper consequence is crowd-out rather than volume: the two slots go to
whichever of a question's concurrently fetched URLs reaches the gate first, so these new candidates
can take both of them ahead of the `blocked` pages whose Google-egress advantage is the rung's whole
reason to exist. If that ordering turns out to matter in the archive, the
honest fix is to prefer `blocked` and `error` over `thin_page` when the cap binds, not to drop the
population, because the withhold came from our own extractor's output rather than from the page and
Gemini reading the same URL is a different extractor.

Turning it off again, or widening its trigger population, is the operator's cost decision (see
`AGENTS.md` "Cost discipline"). Its spend is separable in the archive by the `GEMINI_USAGE` role
`resolution_source`, and its three own markers (`RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP`,
`RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED` and
`RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED`, under "Reading run logs") say how often the free
pre-check saved a read and how often a paid read served nothing.

Model prices, read from ai.google.dev on 2026-09-03: the two live native surfaces
(grounded search and gap-fill v2's `read_document`) run `gemini-3.8-flash` since
2026-09-03, verified live on the google-genai SDK. It draws the same grounding
pool (grounding costs $0 to switch) and its tokens are $0.75/$3.75 per M through
2026-12-31, then $1.50/$7.50, against $0.50/$3.00 for the
`gemini-3-flash-preview` it replaced on search and $1.50/$9.00 for the
`gemini-3.5-flash` it replaced on the reader. Either way that is a few dollars a
month, from prepaid credits. `url_context` carries no per-request fee; retrieved
documents bill as input tokens.

Don't confuse the OpenRouter Gemini path (donated route via
`OAI_ANTH_OPENROUTER_KEY`, minus whatever `DONATED_KEY_BLOCKED_GOOGLE_MODELS`
excepts) with this google-genai path: separate keys, separate billing. The
grounded-search side is `research/gemini_search.py`; v2's `read_document` /
`url_context` path is `research/agentic/tool_backends.py`. What a run actually drew
on this key is readable from the `GEMINI_USAGE` marker in its log (see "Reading run
logs").

## Environment flags

Flags are read at call time via `env_flag_enabled` in `constants.py`, which
treats `true`/`1`/`yes` as on and `false`/`0`/`no` as off (case-insensitive).
When a flag is unset it takes the code default shown below. The bot workflow
YAMLs set these explicitly, so the "prod value" column is what actually runs in
CI.

### Research providers

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `NATIVE_SEARCH_ENABLED` | off | `true` | OpenAI native web search via OpenRouter (model and reasoning effort from `NATIVE_SEARCH_DEFAULT_MODEL` / `NATIVE_SEARCH_REASONING_EFFORT_DEFAULT`), running in parallel with the primary provider |
| `GEMINI_SEARCH_ENABLED` | off | `true` | First-party Google grounded search via the `google-genai` SDK |
| `FINANCIAL_DATA_ENABLED` | off | `true` | yfinance + FRED data for questions an LLM classifier tags as financial |
| `PREDICTION_MARKETS_ENABLED` | off | `true` | Polymarket / Kalshi / Manifold / PredictIt snapshot (suppressed under `is_benchmarking=True`) |
| `RESOLUTION_SOURCE_ENABLED` | off | `true` | Tier-1 fetcher of URLs cited in the resolution criteria (plain HTTP + trafilatura, plus the free escalation rungs; no LLM call and no spend of its own) |
| `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` | off | `true` | The one PAID rung of that fetcher's escalation ladder: when every free rung has failed to read a cited page, Gemini's `url_context` reader is asked to read it, billed to the operator's personal `GOOGLE_API_KEY`. ON in every bot workflow since 2026-09-04, so the resolution-source provider is a paid surface; changing it anywhere is a cost-gate decision for the operator, not an agent |
| `RESOLUTION_SOURCE_IMPERSONATE_ENABLED` | on | unset (so on) | The free TLS-impersonating retry of a direct-fetch 403 (`research/impersonated_fetch.py`, the `route=impersonate` rung and gap-fill v2's `fetch` / `read_document` ladders alike). The only research flag whose code default is ON, read through `impersonated_fetch.impersonation_enabled()`: it costs no key, no model call and no spend, fires on a host's 403 only, is memoized per host for the run once a host refuses the impersonated client, and sits behind the same wall-budget floor as the other one-GET rungs, so the flag is a kill switch rather than an opt-in. No bot workflow sets it. Set it to `false` to fall straight through to the archive and the paid reader on a 403 |
| `TS_ANCHOR_ENABLED` | off | `true` | Time-series empirical P10/P50/P90 band from a question's own resolution series |
| `TS_ANCHOR_CHART_ENABLED` | off | `false` | Chart-image side-channel for the anchor (vision message to base models); held off pending a text-vs-image A/B |
| `RESEARCH_PROVIDER` | `auto` | unset | Forces one primary provider (`asknews`/`exa`/`perplexity`/`openrouter`) instead of the priority order |

The primary provider is chosen by priority: AskNews (when
`ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` are set, the prod case), then Exa, then
Perplexity, then Perplexity-via-OpenRouter. The flags above run on top of the
primary, each independently gated.

Those two are the only flags on the resolution-source escalation ladder (the paid rung's opt-in
and the impersonated retry's kill switch); the rest of it is tuned by constants, all in
`constants.py` and all read at call time. Each rung has a minimum-wall-budget floor below
which it is skipped: `RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S`,
`RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S` (the meta-refresh hop's floor, since the retry is
one GET against a host that just answered us), `RESOLUTION_SOURCE_PDF_MIN_BUDGET_S`,
`RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S`,
`RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S` (12 s, far above the others, because the rung launches a
browser and the launch slot is contended process-wide; it is the pre-gate floor, and the
transport's post-gate need of `RENDER_MIN_GOTO_MS` plus `RENDER_POST_GOTO_TAIL_MS` plus
`RENDER_EXIT_RESERVE_MS` is 15 s, so a render admitted with 12 to 15 s left declines at the gates
with a `wall_budget` skip; that band is deliberate and the operator's to change),
`RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S` and `RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S`. Every
floor is measured against the remaining provider wall less
`RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S`. Three more bound the two rungs that serve something
other than the live page: `RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS` (an archived capture older
than this is withheld as `stale_data` rather than served, matching the Datawrapper freshness
bound), `RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS` (snapshot fetches per question, since every
snapshot contends on one host gate) and `RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS` (the SDK retry
count for one paid read, deliberately fewer than gap-fill v2 allows its reader). One more bounds
the paid rung's spend per question: `RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` caps how many
paid reads one question may pay for across its cited URLs (the analogue of the Wayback cap, a
distinct quantity from the SDK retry count above, recorded as a `url_context_cap` skip when it
binds). One floor bounds CPU rather than a rung:
`RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S` (5 s) is what the extractor policy needs left on
the wall to run its second, `favor_precision` pass over a body already in hand, and below it the
page is withheld exactly as a failed precision pass would withhold it. `docs/research.md` has the
reasoning behind each. Read the values off `constants.py`; that
is the only authoritative copy.

### Gap-fill

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `GAP_FILL_ENABLED` | off | `true` | v1 gap-fill: analyzer finds up to `GAP_FILL_MAX_GAPS` factual gaps, parallel native searches resolve each |
| `GAP_FILL_V2_ENABLED` | off | `true` | v2 agentic research loop (`research/agentic/`); runs concurrently with v1 during the overlap window |

Both gap-fill passes run in prod as of 2026-07-21 (v2 was authored 2026-07-17 but reached
`main` in merge `b4e9df0`; era analysis keys on the latter). Each soft-fails to an empty
string on any error, and both are suppressed under `is_benchmarking=True`. v2's
driver model and reasoning effort come from `GAP_FILL_V2_DRIVER_MODEL` /
`GAP_FILL_V2_DRIVER_EFFORT`; its wall deadline is `GAP_FILL_V2_WALL_DEADLINE` and
its tool-call budget is `GAP_FILL_V2_MAX_TOOL_CALLS`. Every `GAP_FILL_V2_*`
setting is defined in `constants.py`, which is the only place their values are
worth reading; `docs/agentic_gap_fill.md` has the full env-var table.

### Stacking

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `BINARY_STACKING_ENABLED` | off | `false` | Stacker LLM on binary questions |
| `MC_STACKING_ENABLED` | off | `false` | Stacker LLM on multiple-choice questions |
| `NUMERIC_STACKING_ENABLED` | off | `false` | Stacker LLM on numeric questions |

The aggregation strategy is `CONDITIONAL_STACKING` (set in `cli.py`'s `main`), but
all three stacking flags are `false` in every workflow, so prod effectively runs
MEDIAN aggregation. The stacker chain stays live for backtests and ablation. The
disable is evidence-backed: an n=88 ablation found the stacker hurts numeric
CRPS and is no better than median on binary.

### Other flags

| Flag | Code default | Prod value | What it gates |
|---|---|---|---|
| `PROBABILISTIC_TOOLS_ENABLED` | off | `false` | The deterministic probability-math post-processor (`tool_runner.py`); wired but dormant |
| `PERSIST_RESEARCH_ENABLED` | off | `true` (every bot workflow, test ones included since 2026-08-03) | Writes per-question research to JSONL for offline backtest replay |
| `PLATT_CALIBRATION_ENABLED` | off | unset | Post-hoc logistic recalibration of the final published probability |
| `GEMINI_USE_DONATED_OPENROUTER_KEY` | on | `true` | Route OpenRouter Gemini calls through the donated key with personal fallback |
| `OPENROUTER_CREDIT_FLOOR_USD` | `100.0` (see `constants.py`) | unset (uses default) | Donated-key remaining-balance level for the end-of-run early warning to ask Metaculus for a top-up |
| `OPENROUTER_CREDIT_ALERT_RESUME_DATE` | `2026-09-03` | unset (uses default) | Date the credit alerts start reddening CI again; before it, credit shortfalls log but exit zero. Push it forward to re-arm a suppression window |

## GitHub Actions workflows

Six bot workflows live in `.github/workflows/`. They share the same setup
(checkout, `uv sync --no-dev --frozen`, install Playwright Chromium), the same env
block (the Mantic one differs only in its keys; see "Mantic" below), and a `timeout-minutes` job cap
that is a backstop for a wedged run, not a normal duration. Each tees stdout and
stderr to a `run_logs/` file and uploads it as an artifact with 90-day
retention.

That Chromium install now serves two rendered-fetch rungs rather than one: gap-fill v2's fetch
ladder and the Tier-1 resolution-source ladder, which share the transport in
`research/rendered_fetch.py`. The step is `continue-on-error` in every workflow, with a following
step that raises a GitHub warning annotation when it failed, because both callers degrade
gracefully without a browser. The cost of a missing browser is visible per question in the
resolution-source provider's `renderer_unavailable_skips` count, so a run whose install failed
is readable from the archive rather than only from the annotation. That count excludes a URL an
earlier question already rendered to nothing this run (its own `rendered_no_text_skips`) and the
four declines that are facts about the page rather than the runner, each under its own key: a
render the transport's DOM-read bound cut off (`render_timeout_skips`), a browser answered a
non-200 where the direct GET got 200 (`render_non_200_skips`), a rendered DOM over
`RENDERED_DOM_MAX_CHARS` (`render_dom_too_large_skips`) and a main frame that landed on a host
other than the one the transport pinned (`render_off_host_skips`, added 2026-09-04). Neither a memo
hit nor a hostile page can inflate the install-failed signal.

| Workflow | Trigger | Mode | What it does |
|---|---|---|---|
| `run_bot_on_tournament.yaml` | cron at :03/:23/:43 hourly, plus manual | `tournament` | Forecasts new questions in the current AI benchmark tournament (`TOURNAMENT_ID` in `constants.py`); publishes to Metaculus |
| `run_bot_on_minibench.yaml` | cron at :08/:38 hourly in the YAML, but the workflow is disabled on GitHub (see below) | `minibench` | Forecasts the current MiniBench question set; publishes |
| `run_bot_on_metaculus_cup.yaml` | cron at :13/:33/:53 hourly, plus manual | `metaculus_cup` | Forecasts open Metaculus Cup questions (`METACULUS_CUP_ID` in `constants.py`, the season's dated slug); publishes |
| `run_bot_on_mantic.yaml` | cron at :05/:15/:25 hourly, plus manual | `mantic` | Forecasts open questions in the Mantic Crucible tournament (`MANTIC_TOURNAMENT_ID` in `constants.py`) on the operator's personal keys only; publishes to competitions.mantic.com. See "Mantic" below |
| `test_bot.yaml` | manual only (`workflow_dispatch`) | `test_questions` | Runs a fixed handful of example questions end-to-end in prod mode; publishes comments |
| `test_bot_basic.yaml` | manual only (`workflow_dispatch`) | `test_questions` | One-question smoke test; publishes one comment. See below |

The four prod workflows are the only ones with a `schedule:` block; both test
workflows are `workflow_dispatch` and never fire on their own. Since 2026-09-09 an
external dispatcher (cron-job.org) also fires the enabled prod workflows twice an hour
through `workflow_dispatch`, because GitHub delivers only about a fifth of the crons; the
jobs, their minutes and the monitoring are under "Scheduling reliability" below.

**A `schedule:` block in the YAML is not the same as a workflow that runs.**
GitHub carries a per-workflow enabled/disabled state that no file in this repo can
set. `run_bot_on_minibench.yaml` is `disabled_manually` there by operator design
and has NEVER been enabled (confirmed 2026-09-03 and again 2026-09-09), so despite
the :08/:38 crons above the bot does not forecast MiniBench at all. Read the row
above as "would fire hourly if enabled". The practical consequence is that a
`make supply_probe` row showing minibench posts closed with zero bot forecasts is
the EXPECTED state, not a forfeit and not a `METACULUS_TOKEN` problem. It is the
only disabled bot workflow: `run_bot_on_metaculus_cup.yaml` sat in the same state
until the operator enabled it for the fall 2026 season and now reads `active`
(the season-start checklist above has the history), and `run_bot_on_mantic.yaml`
will be enabled the moment it merges to `main` (see "Mantic" below). To check the live state
rather than the YAML:

```bash
gh workflow list --repo No-Stream/metaculus-bot --all
```

`--repo` is required: `origin` is the fork, `upstream` is the Metaculus template,
and no default repo is configured, so a bare `gh workflow` command silently
targets upstream.

All six skip already-forecasted questions
(`skip_previously_forecasted_questions`) except in `test_questions` mode, where
`cli.py` deliberately turns that off so a re-run re-forecasts the same test
question. The four scheduled workflows split their cron across offset entries
because GitHub silently drops `*/N` schedules under runner load, and a
`concurrency` group prevents overlapping runs of the same workflow.
`test_bot_basic.yaml` has its own group, so a smoke run never contends with a
full `test_bot` run.

All six bot workflows (the four prod tournaments plus `test_bot` and
`test_bot_basic`) upload their artifact as `research-<run_id>` with both
`research_outputs/` and `run_logs/`, and all six set
`PERSIST_RESEARCH_ENABLED`. The two test workflows joined that shape on
2026-08-03: they previously uploaded `logs-<run_id>` with only `run_logs/` and
set no persist flag, which was framed as keeping test runs out of the research
archive but in practice just discarded their research. Three runs' worth of
assembled per-question research is gone that way. We still hold their raw
provider payloads and telemetry markers, but not the briefing the forecasters
read. Test runs now contribute to the archive on purpose; they forecast the
evergreen questions, so their records are the ones backtest replay wants most.

`ci.yaml` is the pull-request check (lint + tests); the `claude.yml` workflow is
repo automation unrelated to forecasting.

### The one-question smoke test (`test_bot_basic.yaml`)

This is the cheapest way to exercise the whole live pipeline end to end. It
forecasts exactly one question (Q14333, "Age of Oldest Human as of 2100", a
plain-continuous numeric), chosen because numeric carries the deepest
type-specific pipeline and is the likeliest thing to break. Every research flag
matches `test_bot.yaml`, so a run touches AskNews, native search, Gemini
grounded search, financial data, both gap-fill passes, prediction markets, the
resolution-source fetcher, and the time-series anchor. It runs in prod mode
(`is_benchmarking=False`), which means it **publishes a comment to Metaculus**.

Cost is about $2.60 per run at current config. That is real OpenRouter and
research-API money plus a published comment, so firing it is the operator's
call under the cost rule above. An agent may propose and price it; it does not
dispatch it.

The single question comes from `TEST_QUESTIONS_OVERRIDE` (the env var named by
`TEST_QUESTIONS_OVERRIDE_ENV` in `constants.py`), which `cli.py`'s
`test_questions` path reads as a whitespace- or comma-separated URL list. Unset,
the same mode would forecast the full `EXAMPLE_QUESTIONS` set, which is what
`test_bot.yaml` does.

Firing it, from the Actions UI or the CLI:

```bash
gh workflow run test_bot_basic.yaml --repo No-Stream/metaculus-bot --ref <branch>
```

The workflow has no inputs, so the only choice is which ref to run. Two things
about the plumbing are easy to get wrong.

First, pass `--repo`. This checkout has two remotes (`origin` is the operator's
fork `No-Stream/metaculus-bot`, `upstream` is the Metaculus template it was
forked from) and no `gh` default repo is configured, so a bare
`gh workflow run` or `gh workflow list` resolves against the *upstream* template
and reports a workflow list that does not include this one.

Second, and the yaml header calls this out: a `workflow_dispatch` workflow only
appears in the Actions "Run workflow" UI once its file exists on the **default**
branch. A brand-new dispatch-only workflow on a feature branch is invisible
until it merges to `main`. That is already satisfied here: the file is on
`origin/main` and `gh workflow list --repo No-Stream/metaculus-bot` shows "Test
Bot Basic (1 numeric Q smoke)" as active, so the `--ref` argument can point at
any branch you want to test.

Afterward, the log is in the `research-<run_id>` artifact (90-day retention),
tee'd from `run_logs/` during the run, alongside the run's
`research_outputs/` JSONL. Worth grepping in the downloaded log:

- `PAID PERSONAL-KEY FALLBACK` (`fallback_openrouter.py`): a call fell off the
  donated key onto the operator's personal one.
- `DONATED_KEY_STATE:` (`credit_telemetry.py`): the `/auth/key` probe's verdict
  on why a credit-shaped failure happened (`drained`, `zeroed`, `revoked`,
  `funded`, `unknown`).
- `CREDIT_BALANCE:` / `CREDIT_SPEND:` / `CREDIT_FLOOR_BREACH:`: the per-key
  balances at start and end, the run's spend delta, and the refill warning. Read
  `CREDIT_SPEND`'s `source=` field before trusting the number:
  - `source=remaining_delta` (the donated key) is reliable.
  - `source=usage_delta_unsettled` (the personal key, which reports no
    `limit_remaining`) is a **lower bound**, and frequently `0.00` on a run that
    spent real money: OpenRouter has usually not settled the spend by the time
    the end snapshot fires. A `CREDIT_SPEND_UNSETTLED` warning accompanies it.
    **Do not read `0.00` here as "this run was free."** Measured over 178
    archived personal-key runs: the markers captured 58% of true spend and 160 of
    178 read exactly `0.00`.
  - For the settled per-run figure, run
    `uv run python scripts/reconcile_credit_spend.py` (free, offline, reads the
    telemetry archive). It differences each run's start usage against the next
    run's, which is the only place the lagged spend is observable. The most
    recent run has no successor yet, so it shows as unsettled until another runs.
- `CREDIT_ROLE_SPEND:` (`credit_telemetry.py`): one line per (role, key), which
  pipeline stage spent what, off OpenRouter's own per-call accounting. On a
  one-question smoke run expect three `forecaster:<vendor>` rows plus the
  research roles; `usd=n/a` means no cost data, `role=untagged` means a builder
  call site is missing its `role=`. Described under "Credit telemetry" below.
- `FORECASTERS_SURVIVED:` (`forecaster.py`): the answer to "did every forecaster
  survive?", as `survived=n/N models=...`. Check it rather than inferring: the
  minimum to publish is low enough that a thinned ensemble still exits zero, and
  the failure-path "Only n/N forecasters succeeded" line stays silent on a
  degraded-but-published question. Anything below `n == N` means a model dropped,
  and `FORECASTER_DROPS` names which and why.

The general telemetry markers under "Reading run logs" below apply too; those
are just the money-shaped ones.

## Mantic (Crucible) tournament

Mantic runs a bot-only forecasting competition called Crucible at
<https://competitions.mantic.com> on a fork of the open-source Metaculus platform.
The API is the same shape as Metaculus (Swagger UI at
<https://competitions.mantic.com/api/>, spec at `/static/openapi.e7df88a15335.yml`),
authentication is the same `Authorization: Token <40-char token>` header, and the
bot's user there is `nostreambot-bot`. Mantic pays $3 per forecast; the bot's
per-question cost is about $3 (the post-650 smoke spent $3.17), all of it on the operator's
personal keys.
`--mode mantic` runs the whole per-question pipeline unchanged and swaps only the
platform client. Everything in this section was verified against the live API on
2026-09-08, and the post-651 per-bin smoke of 2026-09-09 re-verified the publish path (see
"Running it" below).

The current target is Preseason 2: slug `preseason-2` (`MANTIC_TOURNAMENT_ID`),
project id 4, forecasting closes 2026-09-20 12:00 UTC (`MANTIC_TOURNAMENT_END_DATE`),
`score_type` `spot_baseline_tournament`. It holds four questions: one binary, one
multiple choice with four options, one discrete with 450 bins and
`multi_resolution: true` (scored against eleven daily bitcoin prices and averaged),
and one date question with twelve daily bins. Series 2 follows it, under rules
Mantic published for the season: baseline scoring; the leaderboard is the sum of a
bot's top 95% of forecasts; a missed question scores the field's 25th percentile;
questions with fewer than two forecasts are excluded; a `quantitative` type merges
numeric and discrete with up to 2,000 bins; date questions default to daily bins;
and a multi-resolution question scores one distribution against several
resolutions. Series 1 windows were exactly one hour long, opened on the hour, with
up to three questions per hour. The Series 2 cadence is unannounced. Forecast every
question: a miss costs more than a poor forecast under that scoring. When Series 2
opens, re-point `MANTIC_TOURNAMENT_ID` and `MANTIC_TOURNAMENT_END_DATE` in
`constants.py`; an unknown slug answers HTTP 400 on the posts list and 404 on the
tournament route. Two startup checks make that hand-over hard to miss, the Series 2
discovery line and the red exit on a stale slug; both are described under "Startup checks
and robustness rules" below.

### How the mode works

`metaculus_bot/mantic.py` defines `ManticClient`, a `MetaculusClient` subclass
pointed at `MANTIC_API_BASE_URL` and authenticated with `MANTIC_TOKEN`
(`build_mantic_client()` raises a clear error when the variable is unset). `cli`
hands it to the framework through the `metaculus_client=` seam on
`TemplateForecaster`, so the tournament fetch, the forecast POST and the comment
POST all go through it. Subclassing keeps the repo's class-level fetch and publish
hardening patches, because they patch `MetaculusClient` methods the subclass does
not override. Comments stay private, which is the framework default on both
platforms. Each fetched question logs a `MANTIC_QUESTION` marker with the post id,
question id, raw type, `cdf_size`, `multi_resolution`, `date_granularity` and
`precision`, so the fields Mantic adds are recoverable from run logs. `page_url` is
rewritten to `https://competitions.mantic.com/questions/<post_id>/`, the URL shape
the site actually serves.

Research archive records from a Mantic run carry `tournament_id` equal to
`MANTIC_TOURNAMENT_ID` and an additive `platform` field (`mantic` or `metaculus`).
Filenames are not namespaced by platform, and `scripts/download_research.py`
(`build_archive`) groups records on the bare `qid`, so `platform` tells the two
platforms' records apart inside a group and is what any analysis keyed on bare
post ids across both platforms must filter on. It cannot stop a Mantic id and a
Metaculus id that meet from merging into one `by_qid` / `latest` / manifest entry.
The margin is smaller than "hundreds versus tens of thousands": the evergreen
`test_questions` set puts Metaculus ids 578, 14333 and 20683 in the archive (578
is already below the Mantic counter, harmless because the bot only archives the
open tournament posts it forecasts), so with the open Mantic posts in the 650s
the next Metaculus key above them is 14333, about 13,700 Mantic posts away. The
revisit is logged in `FUTURE.md`.

Three things differ from Metaculus in the API, each with its fix:

1. **The list filter `forecast_type` uses the value `quantitative`** for what the
   framework still calls `numeric` and `discrete`. The framework's default
   `ApiFilter` sends `forecast_type=numeric,discrete,...` and Mantic returns no
   quantitative questions for it at all (probed: `discrete` gives an empty list,
   `quantitative` gives the one bitcoin question, no filter gives all four). The
   Mantic fetch omits the type parameter and lets the bot's own type guard in
   `forecaster.forecast_questions` filter.
2. **A question's `type` may be the string `quantitative`.** The spec's enum
   includes it for Series 2 questions (the live preseason question still says
   `discrete`). forecasting-tools 0.2.92 raises on an unknown type and the caller
   swallows that as a warning, so such a question would be dropped silently. The
   client normalizes `quantitative` to `discrete` on the raw post JSON before
   parsing; the `scaling` semantics are identical.
3. **Grids exceed 200 bins.** The preseason bitcoin question has 450 bins (a
   451-point CDF) and Series 2 allows 2,000. The server's minimum CDF step is
   `round(0.01 / bins, 9)`; our grid constraints used to floor the step at the
   201-grid value `NUM_MIN_PROB_STEP` (5e-5), which is stricter than the server on
   fine grids and would have forced 2.25% of the mass into a uniform floor at 450
   bins (10% at 2,000). `numeric/config.grid_step_constraints` now uses the server
   formula, which is identical at every grid of 201 points or fewer because 0.01 / 200
   is 5e-5.

Three Metaculus-shaped guards were generalized rather than bypassed.

- The identity preflight (`api_preflight.py`) now has a general
  `verify_api_identity(base_url)`; `verify_metaculus_api_identity()` remains as the
  Metaculus wrapper. In mantic mode the preflight vets the Mantic API host and never
  touches metaculus.com, so a Mantic run does not depend on Metaculus DNS health.
  The failure class is `ApiIdentityError`. Mantic's fingerprint differs in one way: its read
  side answers unauthenticated with 200 JSON carrying a `results` key, which the
  existing acceptance branch already covers.
- The publish-hardening forced POST timeout applies to both `metaculus.com` and
  `competitions.mantic.com` (`QUESTION_PLATFORM_HOSTS` in `constants.py`). Without
  it a stalled Mantic POST would be abandoned by the caller while its worker thread
  ran on, the duplicate-publish shape layer 1 exists to prevent.
- The self-reference refusal shared by the resolution-source fetcher and gap-fill
  v2 (`resolution_url_scan.is_metaculus_self_ref`) refuses the question platform's
  own site, metaculus.com or `competitions.mantic.com`; the rest of mantic.com, such
  as blog.mantic.com, stays fetchable as an outside source. The function name and
  the `metaculus_self_ref` status token are unchanged, as data contracts.

### Startup checks and robustness rules

Five rules from the 2026-09-08 readiness review
(`scratch_docs_and_planning/mantic_research_2026-09-08/edge_case_report.md`) sit between the
identity preflight and the first paid call. Every one is free.

- **Forecast-permission preflight** (review item 20). After `build_mantic_client()` and before
  the forecaster is built, `mantic.preflight_mantic_tournaments` makes two authenticated GETs,
  neither retried (the next cron is the retry, as for the identity preflight): the tournament
  list `/api/projects/tournaments/` for the discovery line below, then the configured
  tournament's own route `/api/projects/tournaments/<slug>/`, and raises `ApiIdentityError`
  unless that route answers 200 with a `user_permission` that allows forecasting: `forecaster`,
  `curator`, `admin` or `creator`, the Metaculus backend's `ObjectPermission` vocabulary (the
  live token reads `forecaster`). The detail route rather than the list row because the list
  omits an `unlisted` project, the state a new season sits in before its first question opens
  (see "Read the project object before editing a slug" above), so the Series 2 hand-over cannot
  abort on a slug that exists but is not yet listed, while a slug no tournament has 404s there
  and stops the run naming `MANTIC_TOURNAMENT_ID`. A token that may only view reads the
  tournament fine and would otherwise research and forecast every question and fail at the
  publish POST, hourly, at about $3 a question. Because the GETs are authenticated, a revoked
  or mistyped token fails here too, as a 403 `Invalid token.` on either route (verified live),
  which the unauthenticated identity preflight cannot see. A DNS, TLS or connect failure and a
  200 that is not JSON (a captive portal) are the same `ApiIdentityError`, so every stop is one
  greppable exception.
- **Series 2 discovery** (item 5). The list GET logs
  `MANTIC_TOURNAMENTS: ongoing=<slugs> configured=<slug> new=<slugs>`, where `new` is the
  ongoing bots-only tournaments that are not the configured one, at WARNING when that set is
  non-empty and INFO otherwise (`none` for an empty list). A registered marker, so the first
  run that sees a Series 2 slug is findable in the archive.
- **Stale slug goes red** (item 5). `check_tournament_dates` warns from a slug's end date and
  raises at the shared two-week hard stop, and a zero-question run is green, so once
  Preseason 2 closes on 2026-09-20 every scheduled run would have stayed green and silent for
  a fortnight while a Series 2 slug went unforecast, about seventy questions at Series 1's
  rate. In mantic mode the check's verdict (`cli._check_tournament_dates`) is held and turned
  into a non-zero exit AFTER publishing, the fall-cup reminder's shape. The end date is the last
  OPEN day: Preseason 2 forecasts until 12:00 UTC on `MANTIC_TOURNAMENT_END_DATE`, so the red
  exits start on the UTC day after it, and a run in the final open hours publishes and exits 0.
  The Metaculus tournament keeps the warning advisory and `TOURNAMENT_HARD_STOP_WEEKS` is
  untouched.
- **Parse drops are counted** (item 12). A post the framework cannot parse (a new Mantic type
  string, a missing field) is caught by the framework's per-post loop, logged as one warning
  and otherwise forfeited silently on every run. `ManticClient` now counts the drop and logs
  `MANTIC_POST_DROPPED: post=<id> type=<wire type or n/a> error=<ExceptionClass>` before
  re-raising (nothing is swallowed), and cli adds the counter to the alertable arithmetic, so
  a dropped post reddens the run and the end-of-run breakdown carries
  `mantic_post_drops=<n>` whenever it is non-zero. The telemetry-only reads in the client
  (`id`, `type`) use `.get`, so the marker can never be what drops a question.
- **Pagination ceiling** (item 21). The tournament fetch asks the framework for
  `MANTIC_FETCH_QUESTION_CEILING` (500) questions with `error_if_question_target_missed=False`,
  so it walks offsets until an EMPTY page instead of reading one page of 100 and trusting
  Mantic's `next` link, which is advertised past the last page. With four open questions that
  is one extra GET of an empty page.

Cited-source URLs with brackets or backticks are extracted whole since the same review
(item 4): a truncated Federal Register API query answered 200 with the unfiltered count.
See `docs/research.md` "Resolution-source fetcher".

### Date questions

Every Mantic question type is forecast, date questions included. The question stays a
`DateQuestion` end to end (the framework builds a `DateReport`, telemetry says `qtype=date`)
and the numeric math runs on an adapter: `numeric/date_axis.py` views the question as a
`NumericQuestion` on the epoch-seconds axis, which is how forecasting-tools and the Metaculus
backend both treat a date question. The forecaster dispatches `DateQuestion` to
`run_date_forecast` (`forecaster_runners.py`); `date_prompt` (`prompts.py`) asks for ISO dates
and names the bin granularity; `DateStructured` (`structured_output_schema.py`) carries the
declared percentiles as datetimes; then the same PCHIP pipeline, CDF-space aggregation and
publish path run with `is_date` set so the comment renders dates. On a coarse grid (post 651's
twelve daily bins) the date question is instead elicited per bin, one probability per calendar
day, and its members are pooled by the mean; see the "Per-bin elicitation on enumerable grids"
bullet under "Mantic-optimized forecasting" below. Two conventions live in the
adapter and nowhere else: nominal bounds are read from the API's `scaling` block, never
derived (Mantic sets `nominal_max` to the last bin's left edge), and a date-only value means
noon UTC of that day, so its mass lands inside that day's bin under the platform's
right-closed bucketing. The design and its receipts are in
`scratch_docs_and_planning/mantic_phase2_plan_2026-09-08.md`; the pipeline map is in
`docs/architecture.md`. The analysis side stays date-free by decision until a date question
has resolved under this code: the backtest, the ablation harness and the residual dataset each
skip date questions at an explicit seam, and the ghost scorer counts them as unscoreable
(`docs/performance_analysis.md` "Date questions are excluded from the dataset").

### Mantic-optimized forecasting

The scoring reader and the edge-case review in
`scratch_docs_and_planning/mantic_research_2026-09-08/` priced what the Metaculus-shaped prompts
and numeric repairs cost under Mantic's baseline scoring, and the Phase 2 merge ships the fixes.
Every prompt clause is a named constant in `prompts.py` with its reason and has presence and
absence pins under `tests/prompts/`. The Mantic-gated clauses (the out-of-range base rate, the
multi-resolution rule, the scoring grid and the series-variant reading) appear in no stacking
prompt; the two platform-aware scoring texts are the exception, because the three stacking
prompts already carried the same wording: `_scoring_sentence` renders in all three and
`_CONTINUOUS_SCORING_RULE` in `stacking_numeric_prompt`.

- **Out-of-range base rate** (review item 1, the largest lever). Mantic scores a resolution
  outside the displayed range as its own outcome against a fixed 5% reference: a 1% tail scores
  -80.5, 5% scores 0, 50% scores +115, verified to zero error against the platform's own scores
  on 146 of 146 resolved out-of-range questions. The pipeline publishes exactly 1% beyond an open
  bound whenever every percentile sits inside the range, and Mantic escapes its ranges far more
  often than Metaculus. Series 1, annulled questions excluded: 24.8% of resolved discrete
  questions (35 of 141), 12.0% of numeric (16 of 133), 18.6% of quantitative questions combined
  (51 of 274) and 53.7% of date questions with an open upper bound (101 of 188), against 2 to 3%
  in the Metaculus archive. The numeric figure used to read 7%: Mantic stored seven numeric
  escapes as raw values outside the range rather than as `above_upper_bound` /
  `below_lower_bound`, which a filter on those strings misses (receipt:
  `scratch_docs_and_planning/mantic_adversarial_candidates_2026-09-08.md`, candidate 2).
  `_MANTIC_OUT_OF_RANGE_RATE_QUANTITY` and `_MANTIC_OUT_OF_RANGE_RATE_DATE` state that base rate
  in the bound messages on Mantic only (`question_platform.question_platform` reads the platform
  off `page_url`, so the Metaculus prompts are unchanged). **The mechanical 5% tail floor was
  approved and built on 2026-09-08** after being held through Phase 2
  (`docs/numeric_pipeline.md` "Step 11: the Mantic out-of-range tail floor"):
  `MANTIC_OUT_OF_RANGE_TAIL_FLOOR` (0.05, `constants.py`) is applied by
  `numeric/out_of_range_floor.py` from `TemplateForecaster._aggregate_predictions` to the Mantic
  aggregate CDF on open sides only, never on Metaculus, never on a closed bound, and never
  reducing a tail already at or above the floor. Moving each
  open side from 1% to 5% gains 80.5 points when the outcome escapes and, when it does not,
  costs 4.26 with both sides open (2.06 with one); the same floor applied to the eleven
  Series 1 competitors' own published distributions cost at most 1.9 points per question for
  any type. The additive `oor_low=` /
  `oor_high=` fields on the per-member `MEMBER_FORECAST` line keep measuring the models'
  unfloored behaviour, and the per-question `NUMERIC_AGGREGATE` marker carries the published
  tails beside additive `oor_low_raw=` / `oor_high_raw=` / `tail_floor=` fields, so the floor
  can be benchmarked on this bot's own forecasts once live telemetry accumulates (the ask is
  recorded in `FUTURE.md` "Mantic Crucible").
- **Platform-aware scoring text** (item 6). The numeric prompt no longer claims a uniform 0.01
  PDF floor or that sharpness above 35 stops paying, two Metaculus facts that told the model the
  out-of-range cliff was an order of magnitude shallower than it is; `_CONTINUOUS_SCORING_RULE`
  says mass beyond an open bound is scored as its own outcome against a reference of a few
  percent. `_METACULUS_SCORING_SENTENCE` names the spot peer score and `_MANTIC_SCORING_SENTENCE`
  names Crucible's spot baseline score, compared to a uniform distribution rather than to other
  forecasters, so nothing is gained by disagreeing with the obvious answer and nothing is lost by
  giving it.
- **Series-variant clause** (item 9). The displayed range is stated as weak evidence about which
  series variant resolves and as no evidence about the magnitude of the outcome. The old premise,
  that the bounds were set by someone who could see the real series, is false on Mantic, where
  question writers are paid for bot disagreement and more than half of the date questions with
  an open upper bound resolved above the ceiling (the Series 1 counts are in the out-of-range
  bullet above).
- **Multi-resolution questions** (post 650's shape, `multi_resolution: true`). Gated on the
  question's own API flag (`_multi_resolution_clause`, an identity test on `is True`) and
  type-aware: continuous and date questions forecast each resolution instance and report the
  mixture's percentiles, multiple choice the expected share of resolutions per option, binary the
  expected fraction of Yes. Priced on post 650: a single-day distribution loses 38.7 points. The
  count of resolutions is never interpolated; it lives only in the criteria prose while the
  question is open.
- **Coarse grids** (item 7). The prompt names the bin width and bin count, and the count-like
  cluster spread that used to spread a plateau a full unit per position is capped by the grid's
  bin width (`numeric/config.grid_bin_width`), so a concentrated forecast on a 3 to 21-bin grid
  publishes as declared rather than flattened, worth 24 to 43 points per affected question. The
  discrete-snap guard keys on the question type and the 201-point grid rather than on
  `cdf_size == 201` alone, because a 200-bin Mantic discrete question has that size too.
- **Near-total out-of-range forecasts build** (item 3). The min-step rebuild trigger and its
  raise in `numeric/pchip_cdf.py` carry the `_MIN_STEP_TOLERANCE` (1e-10) the file already used
  elsewhere, so a forecast with essentially all its mass beyond an open bound, the
  highest-scoring shape here (+148.8 baseline points at 0.98), builds instead of dropping the
  member on a float epsilon.
- **Bounds clamp on bin-defined grids** (item 11). The clamp buffer is at least one bin width, so
  a date one day outside a closed bound clamps instead of dropping the member; a scale error
  still raises.
- **Per-bin elicitation on enumerable grids** (Wave C, built 2026-09-09). A Mantic numeric or
  date question whose published bins are its outcome space and number 31 or fewer
  (`elicit_per_bin` in `numeric/config.py`: `PMF_ELICITATION_PLATFORMS` is Mantic-only,
  `PMF_ELICITATION_MAX_BINS` is 31) is asked for one probability per bin, keyed by a
  human-readable label (`numeric/pmf_grid.py`: the calendar day on a day or week grid, the bin
  centre on a count grid, `a to b` otherwise) plus `below_range` / `above_range` where a bound is
  open, instead of 13 percentiles. Motivation: post 651 names nine eligible trading days in a
  12-day window, and a percentile declaration cannot say "zero on the three weekend days";
  PCHIP spreads about a quarter of the mass onto them, 14.4 baseline points lost with zero
  information (11.6 under the Series 2 categorical form), and a count grid of 13 bins or fewer
  cannot carry 13 distinct percentiles at all (18 of the 46 coarse Series 1 discrete grids).
  The runner branch (`_run_pmf_forecast`, `forecaster_runners.py`) builds `pmf_prompt`, reads the
  `pmf` block through the extraction ladder and hands the `N + 2` declaration to
  `numeric/pmf_cdf.py`, which normalizes it, blends it to the server's per-cell floors (a bin the
  model set to 0 lands at exactly the platform minimum, a certain member keeps about 0.99 on its
  bin), assembles the CDF, runs `safe_cdf_bounds` and a fail-shut replica of the server's rules;
  the percentile sanitizer, the PCHIP repair tiers, the discrete vote and the unit-mismatch guard
  are not on this path (`docs/numeric_pipeline.md` "Per-bin elicitation" gives each reason).
  Per-bin members are aggregated by the linear opinion pool, the pointwise MEAN of their CDFs,
  because the pointwise median of three sharp members is the middle member's CDF outright and
  the platform floor on the bins the other two believed (about -230 baseline points when one of
  those resolves; the pool gives each believed bin a third); percentile members keep the MEDIAN.
  Telemetry: the `MEMBER_FORECAST` line carries `elicitation=pmf` with the `N + 2` PMF as `raw`
  and `published` (an absent field means percentiles), `NUMERIC_AGGREGATE` ends
  `method=mean|median|stacked|single` (`unrecorded` marks a bug), and `EXTRACTION_RUNG` reads `qtype=pmf` for this ladder.
  The Metaculus switch is adding `PLATFORM_METACULUS` to `PMF_ELICITATION_PLATFORMS`, its own
  config-era change (it would move about half of all Metaculus discrete questions), once a
  Mantic season shows the per-bin declaration is faithful. Neither the per-bin gate nor the 5%
  tail floor has an environment flag: `PMF_ELICITATION_PLATFORMS`, `PMF_ELICITATION_MAX_BINS`
  and `MANTIC_OUT_OF_RANGE_TAIL_FLOOR` are plain constants, so disabling either is a code edit
  and a commit, not a workflow env change. The paid smoke and its verification list are under
  "Running it" below.

### Personal keys only, and the switch fails shut

Metaculus donates `OAI_ANTH_OPENROUTER_KEY` for its own tournaments, so a run that
forecasts for another platform must not spend it. Every other key the bot reads is
already the operator's personal key (see the key model above), so one switch
suffices: `DONATED_OPENROUTER_KEY_ENABLED` (default `true`, read by
`constants.donated_openrouter_key_enabled()`). When it reads false,
`fallback_openrouter.should_route_via_donated_key` returns False before any provider
match, which covers all three readers of the donated env var and therefore every
OpenRouter call including the key-swap fallback; and `credit_telemetry` skips the
donated-key balance probe with one INFO line
(`CREDIT_BALANCE: key=donated phase=<p> skipped (donated routing disabled)`), so a
low donated balance can never redden a Mantic run.

It has to be an environment variable set before the process starts, not a CLI
flag: the roster's module-level `GeneralLlm` objects in `llm_configs.py` freeze
their API key at import, and `main.py` imports them before `cli.main` runs. So in
mantic mode `cli._assert_personal_keys_only()` raises `RuntimeError` before any
fetch or spend if the switch still reads true. The workflow is the other half of
the same rule: it never receives the donated secret, so no code path can reach it,
and `tests/test_workflow_reliability.py` pins that the file's raw text names
neither `OAI_ANTH_OPENROUTER_KEY` nor `METACULUS_TOKEN` while every Metaculus
workflow still wires the donated key.

The accepted consequence: with a single OpenRouter key there is no key-swap
fallback, so a personal-key 401, 402 or 429 is a hard failure of that question, by
design. No replacement mechanism was added (the proportion rule). On a Mantic run,
every OpenRouter auth error is the personal key.

### The workflow

`run_bot_on_mantic.yaml` is a copy of `run_bot_on_tournament.yaml` with four
differences: three cron entries at :05, :15 and :25; the run step passes
`--mode mantic`; the env block has `MANTIC_TOKEN` instead of `METACULUS_TOKEN` and
no `OAI_ANTH_OPENROUTER_KEY` at all; and it sets both
`DONATED_OPENROUTER_KEY_ENABLED: 'false'` and `GEMINI_USE_DONATED_OPENROUTER_KEY:
'false'`. Everything else, including the step caps, the Playwright install, the
personal `GOOGLE_API_KEY` and the paid `url_context` rung, is at parity, and the
artifact keeps the `research-<run_id>` name so `make sync_all` harvests Mantic runs
into the archive. The minutes sit off the tournament's :03/:23/:43, minibench's
:08/:38 and the cup's :13/:33/:53, because the workflows are in separate concurrency
groups and a shared minute means simultaneous runs, and off the top of the hour,
where GitHub's scheduling burst lives.

The three entries all sit in the first half of the hour, and that is deliberate. Mantic
questions open on the hour with 60-minute windows, and the per-question budget is the close
time minus now minus the 60-second publish reserve (`time_budget.py`). A pickup after about
:30 falls under the 1815-second fast-path threshold and gets only the degraded research path,
and a pickup within six minutes of close falls under the 300-second viability floor
(`TIME_BUDGET_MIN_VIABLE_S`) and gets nothing. A :45 entry would therefore buy only the
degraded fast path, and only if GitHub delivered it promptly. A :55 entry would buy nothing.
The workflow carries neither. `tests/test_workflow_reliability.py` pins that every
Mantic entry fires early enough for the full research path and that no two bot workflows share
a minute.

### Scheduling reliability

GitHub delivers only a minority of this repository's scheduled firings. Measured through the
GitHub API on the three-cron tournament workflow from 2026-08-27 through 2026-09-07: 7 to 23
of the 72 expected runs a day were delivered, about 22%, with gaps of up to 3.5 hours, and
the cup workflow reads the same, so the loss is repository-wide and outside our control. On
Mantic's one-hour windows a question is forfeited whenever no run lands in its first half
hour, while a delivered run that finds nothing new spends nothing because cli pins the
re-spend guard on. The delivered runs also cluster inside good hours, so more cron entries
buy less than independent drops would imply and cannot cover a multi-hour blackout.

The fix is an external dispatcher, because `workflow_dispatch` events are not subject to
schedule dropping, and it went live on 2026-09-09: three cron-job.org jobs call GitHub's
workflow-dispatch REST endpoint twice an hour, one job per bot workflow. Job 8417341
dispatches `run_bot_on_tournament.yaml` at :02 and :32 UTC, job 8417342 dispatches
`run_bot_on_metaculus_cup.yaml` at :12 and :42, and job 8417343 dispatches
`run_bot_on_mantic.yaml` at :01 and :16. The Mantic job was created disabled, because
GitHub answers 404 to a dispatch for a workflow file that `main` does not have, and
`make cronjob_dispatch_setup ARGS="--apply --enable-mantic"` turns it on once the Mantic
branch has merged. The first firing of each hour sits just ahead of the workflow's first
cron entry (:02 before :03, :12 before :13, :01 before :05), so when GitHub does deliver
that cron the dispatched run already holds the workflow's concurrency group and the cron
run queues behind it, finds nothing new and spends nothing. The six minutes are distinct
across the three jobs, so two full bot runs never share the runners or the research
quotas. On Mantic, :01 is the earliest pickup of a window that opens on the hour and :16 a
second chance inside the first half hour, after which a pickup gets only the fast path.
Two options that look like fixes are not: a self-hosted runner does not help, because the
drops happen in GitHub's scheduler before any runner is involved, and running the bot
directly on a box loses the artifact pipeline that `make sync_all` harvests
(`research_outputs/`, `run_logs/`, the 90-day retention).

Each job POSTs with a fine-grained GitHub personal access token scoped to this repository
with Actions read and write only, expiring one year after issue (September 2027). The
setup script reads it as `GH_DISPATCH_TOKEN` and the cron-job.org key as `CRONJOB_API_KEY`
from `.env`, and prints neither. GitHub answers a successful dispatch with 204, and
cron-job.org emails the operator when a job fails, so an expired or revoked token surfaces
as failure mail rather than as silence; that email is the dead-token monitor.
`make dispatch_watch` (free, one `gh run list`) is the delivery read: per bot workflow and
per UTC day, how many `schedule` and `workflow_dispatch` runs arrived and how they
concluded, against what the cron entries and the two-an-hour dispatcher say should have.
`make cronjob_dispatch_setup` is the idempotent re-creation path after a token rotation or
an accidental deletion: it matches the account's jobs by exact title, creates the missing
ones, patches the ones whose spec differs and leaves the rest alone; the bare target is a
dry run that prints the payloads with the token redacted and writes nothing, and
`ARGS="--apply"` is the paid step behind the ask-first gate. The GitHub crons stay in the
workflow files as the backstop.

The extra firings are safe because of two guards. The workflow-level `concurrency` group
queues an overlapping run instead of running it in parallel, and the
skip-previously-forecasted guard, which `cli.py` pins on in every tournament-shaped mode,
makes a run that finds no new question spend nothing. Since 2026-09-09 that guard fails
shut. The framework derives `already_forecasted` inside a blanket except that answers
False, so a list payload with no `my_forecasts` field would have read every question as
never forecast and re-published the whole tournament on every firing. Such a question is
now dropped before any spend, with one
`SKIP_GUARD_UNREADABLE: question=<id> post_id=<id> platform=<metaculus|mantic> reason=my_forecasts_missing`
WARNING per post and a count line. When it fires, the list read lost `with_cp=true`, which
is what puts the field on the list page, or the platform token (`METACULUS_TOKEN`, or
`MANTIC_TOKEN` on Mantic, whose public list lacks the field entirely), or the API changed
shape. Check the run's environment and the token, and read one post by hand with
`?with_cp=true` to see whether the field is back. Nothing was double-forecast, and the
dropped questions are picked up by the next firing once the field reads again. A present
field with an empty history is a never-forecast question and stays eligible. The marker
is registered in `scripts/telemetry/markers.py`, so the drops outlive the log expiry.

GitHub runs a new scheduled workflow as soon as its file is on the default branch, so
merging the branch to `main` starts the hourly crons with no further UI step. The
per-workflow enabled state only ever bites a workflow someone has disabled in the Actions
UI, and today that is minibench alone (see "GitHub Actions workflows" above).

The instrument that checks the cadence is `make supply_probe_mantic` (free,
read-only; `ARGS="--slugs <series-2-slug> --output scratch/mantic_supply_$(date -u +%Y%m%d).json"`
after the first Series 2 week). It pages the tournament's open, closed and resolved posts,
sweeps the closed and resolved ones for forfeits, classifying each as forecast, no_forecast or
unknown, and prints the miss rate per UTC release hour (the hour of `open_time`, which on a
60-minute window is the hour a run had to land in) plus the realized open-to-close window
distribution, so the 60-minute assumption is checked by the same run. `with_cp=true` rides every list GET: it is what puts the public spot-time snapshot on the list
page, so without a token the probe still classifies every RESOLVED question from that snapshot
(`score_data.disagreement_forecasts.forecasts[]`, author id against `MANTIC_BOT_USER_ID`), which is
exactly what is scored, and a closed-but-unresolved question reads `unknown`. `MANTIC_TOKEN` adds
`my_forecasts` to the same page, so closed-but-unresolved questions classify too. One caveat,
stated in the report header rather than modelled: the snapshot names one competitor fewer than
`nr_forecasters` on nearly every resolved question and the cause is not established, so a
`no_forecast` read from it is provisional until the first question the bot forecast resolves and
its snapshot names the bot (`docs/supply_probe.md` "The Mantic mode" has the numbers).

### Running it, and what is left for the operator

The local QA run is paid and publishes. It spends about $3 per question (the post-650 smoke
spent $3.17, the post-651 per-bin smoke $2.28) on the personal OpenRouter, AskNews, Exa and Google keys, posts a forecast for every open
question the bot has not yet forecast to Mantic, and goes through the ask-first
gate like every other live mode:

```bash
DONATED_OPENROUTER_KEY_ENABLED=false uv run python main.py --mode mantic
# or: make run_mantic
```

The smoke run is the same command narrowed to one chosen question. The Phase 1 smoke was
this command on post 650 (the multi-resolution discrete question); it ran and passed on
2026-09-08 at 18:00 PT:

```bash
DONATED_OPENROUTER_KEY_ENABLED=false uv run python main.py --mode mantic --only-posts 650
# or: make run_mantic_one POST=650
```

`--only-posts` takes comma-separated post ids (the number in the question URL) and
forecasts only those of the tournament's open questions. It works in every
tournament-shaped mode (`tournament`, `minibench`, `metaculus_cup`, `mantic`) and is
refused with `test_questions`. It fetches the tournament's open questions exactly as an
unfiltered run does, on the same client, keeps the listed posts, and logs one line such as
`ONLY_POSTS: requested=650 matched=650 dropped=3`, a registered marker, so the archive
records which question a smoke run spent on. `matched` is which of the requested ids were
among the open questions; when none are, the run logs a warning and forecasts nothing
rather than the whole tournament. The re-spend guard still applies, so a listed post the
bot has already forecast is skipped like any other.

The per-bin smoke (Wave C) was the same command on the preseason's date question, post 651
(twelve daily bins, both bounds closed), which the gate elicits per bin. It ran and passed on
2026-09-09 at 11:24 PT for $2.28 of personal spend (the sum of the run's `CREDIT_ROLE_SPEND`
lines; run log `~/logs/mantic-smoke-651.log` on the operator's laptop) and was verified with
the authenticated read below: a 13-value CDF with `cdf[0] == 0.0` and `cdf[12] == 1.0`, the
three weekend bins at the platform minimum because all three members declared 0 there, the
modal bin on 2026-09-16 (FOMC day) at 27.0%, three `MEMBER_FORECAST ... elicitation=pmf` lines,
one `NUMERIC_AGGREGATE ... method=mean` line, one `CLOSE_MARGIN` line and a clean exit. The
forecast-permission preflight it runs first passed against the live token: the tournament route
read `user_permission` `forecaster`, `is_ongoing` true and `bot_leaderboard_status` `bots_only`.
Like every paid run it fires once per approval; the command, for the record and for any later
smoke on another question:

```bash
DONATED_OPENROUTER_KEY_ENABLED=false uv run python main.py --mode mantic --only-posts 651
# or: make run_mantic_one POST=651
```

Verify any smoke afterwards. The authoritative confirmation is the authenticated read
`curl -s -H "Authorization: Token $MANTIC_TOKEN" "https://competitions.mantic.com/api/posts/651/?with_cp=true"`;
the run log is the second witness, with one trap: forecasting-tools logs
`Posted prediction on question 651` and `Posted comment on post 651` BEFORE it checks the HTTP
status, so both lines also appear in the log of a run whose POST the server rejected, and
neither counts as evidence. Check:

- `my_forecasts.latest.forecast_values` on the read is a 13-value CDF the server accepted, with
  `cdf[0] == 0.0` and `cdf[12] == 1.0` (both bounds closed);
- every bin carries at least the platform minimum, `round(0.01 / 12, 9)` plus the 1e-9 margin.
  The weekend bins (labelled 2026-09-12, 2026-09-13 and 2026-09-19) sit exactly at that minimum
  only when all three members declared 0 on them; the members are pooled by the pointwise
  mean, so one member putting 0.02 on a Saturday lifts the pooled bin above the minimum, which
  is the pool working rather than a defect;
- the run log has three `MEMBER_FORECAST ... qtype=date ... elicitation=pmf` lines whose `raw` and
  `published` are 14-entry vectors, and one `NUMERIC_AGGREGATE ... method=mean` line;
- the run log has one `CLOSE_MARGIN: question=651` line (the question id, which equals the post
  id for 651), no match for `grep -E 'PUBLISH_HARDENING|PUBLISH_SKIPPED_CLOSED|HTTPError|Traceback'`,
  and ends with `Run completed clean with 0 alertable degradation event(s)`; the command itself
  exits 0;
- the comment renders ISO dates and no epoch second.

All three commands need `MANTIC_TOKEN` in `.env` (the operator also keeps it at
`~/.keys/MANTIC_TOKEN`). Without `DONATED_OPENROUTER_KEY_ENABLED=false` the run stops
at `_assert_personal_keys_only` before any spend.

Operator steps, in order:

1. Done 2026-09-08: the token is stored as the `MANTIC_TOKEN` repository secret
   (`gh secret list --repo No-Stream/metaculus-bot` shows it, set 2026-09-08 20:32 UTC).
2. Done 2026-09-09: the per-bin smoke on post 651 ran once, passed the five checks above and
   was verified on the API ($2.28; the paragraph above has the readings).
3. Merge `mantic-competition` to `main`. The schedule is live from that moment; there is
   nothing to enable in the Actions UI. Then enable the Mantic dispatcher job with
   `make cronjob_dispatch_setup ARGS="--apply --enable-mantic"` (paid, ask-first; see
   "Scheduling reliability" above).
4. When Series 2 opens, update `MANTIC_TOURNAMENT_ID` and `MANTIC_TOURNAMENT_END_DATE`;
   the "Series 2 discovery" and "Stale slug goes red" checks above are what flag the
   hand-over.

## Cost discipline

Every credit spend goes through the operator. Anything that hits a live LLM or
research API spends real money, and the run modes also publish forecasts and
comments to the platform they forecast (Metaculus, or competitions.mantic.com in
`--mode mantic`), a visible external action that is hard to retract. Nothing
in that class launches without the operator saying yes first. `AGENTS.md` at the
repo root carries the terse agent-facing version of the same rule.

The gate is on the **spend**, not on the mechanism. It covers anything that
causes a paid call no matter who or what finally makes it: a local `make`
target, a GitHub Actions dispatch of a bot workflow, an edit that adds cron
entries to a `schedule:` block, a flag change that raises per-run cost, or a
one-off script that wraps any of those. There is no clean-gates exemption and no
threshold below which a run is small enough to skip asking. A two- or
three-dollar smoke run still goes through the operator. When a paid run is the
only way to verify a change, the right move is to name the exact command, price
it, and stop there.

What the gate forbids is an agent *deciding* to spend. An explicit instruction is
the approval: told to fire a run already discussed, an agent should run it and not
re-ask. That approval is per-run. One go-ahead is not standing authorization for
the next run, or for re-running the same one after further changes.

Paid runs are a final pre-merge check rather than part of the verification loop.
The one-question smoke test below exists to be fired once, deliberately, when a
change is otherwise finished and about to merge. Its small per-run cost is the
trap: an agent that treats it as a normal check-my-work step fires it several
times in a session and spends real money for no added signal, since the run tells
it nothing the free gates did not. The loop is `make test`, `make lint`, and
`make typecheck`, with unit and integration coverage as the proof of correctness.
The paid run is the operator's last step.

### Paid and externally visible

- `uv run python main.py` / `make run` in any live mode (`tournament`,
  `minibench`, `metaculus_cup`, `test_questions`): spends credits and publishes
  to Metaculus. `cli.py` builds the bot with `publish_reports_to_metaculus=True`
  in every mode.
- `--mode mantic` / `make run_mantic`: spends the operator's personal keys
  (about $3 per question, $3.17 on the post-650 smoke; the donated key is refused) and publishes to
  competitions.mantic.com. `--only-posts <ids>` / `make run_mantic_one POST=<id>`
  is the same run narrowed to the listed post ids, so one question's worth of
  spend. See "Mantic" above.
- `make backtest_smoke_test` / `_small` / `_medium` / `_large`: spends on every
  forecaster and research call, plus one `LEAKAGE_DETECTOR_MODEL` call per
  question for the leakage screen. No publish (the benchmark config sets
  `publish_reports_to_metaculus=False` and `is_benchmarking=True`), but real
  money. The per-target question counts are the `--num-questions` values in the
  Makefile.
- `make backtest_with_cache`: the `--research-dir` flag replays archived
  research instead of fetching it, so the research and leakage-screen calls go
  away. The live ensemble still forecasts every question, so forecaster spend is
  real. A question with no archived record falls back to live research and the
  run logs a warning saying so.
- `make ablation_qa_research` / `ablation_smoke` / `ablation_small` /
  `ablation_medium`: real research plus forecaster spend.
- `make benchmark_run_*`: deprecated, since `community_benchmark.py` baseline
  scoring broke when Metaculus dropped `aggregations` from the list API, but the
  `run` and `custom` modes still fan the real ensemble over real questions.
  Prefer `make backtest_*`.
- `make test_live`: the only test target that leaves the network. It pins a
  `:free` OpenRouter model slug so the dollar figure is near zero, but the calls
  are real and need a live key, so it still goes through the operator.
- GitHub Actions runs of any bot workflow. A dispatched run spends exactly what
  the same mode spends locally and publishes to the platform that workflow forecasts,
  Metaculus for every other bot workflow and competitions.mantic.com for
  `run_bot_on_mantic.yaml`. See the workflow table above for triggers, and the
  smoke-test subsection there for the one-question variant.
- `make cronjob_dispatch_setup ARGS="--apply"`: creates or changes the live cron-job.org
  jobs that dispatch the bot workflows, so every firing it adds is a paid, publishing bot
  run; `--enable-mantic` turns the Mantic job on and waits for `run_bot_on_mantic.yaml` to
  be on `main`. See "Scheduling reliability" above.
- `make probe_resolver QUESTION=<id>`: replays the gaps the archive recorded for one
  question through the production gap-fill v1 resolver path at every model and
  search-context cell of a grid (default: the current resolver model and `gpt-5.6-luna`,
  each at high, medium and low) and writes the answers beside OpenRouter's per-call cost
  to `scratch/probes/`. Up to about $0.20 a call on the operator's personal OpenRouter key
  (the donated key is forced off); the script prints its ceiling first and refuses without
  `ARGS="--i-accept-spend"`. Narrow with `ARGS="--grid current:high luna:low"` or
  `ARGS="--gaps 1,2"`.
- `make strip_bench`: the paired section-strip bench forecasts every resolved gap-fill
  pair four ways (the bundle as published, minus v1, minus v2, minus both) with one cheap
  model on the personal OpenRouter key and scores the arms against the resolutions. No
  research runs and nothing publishes; the estimate at the default three replicates is
  about $1.25, under `--max-spend-usd` (default 10). A bare call prints the plan and
  refuses, `ARGS="--dry-run"` is the free view, `ARGS="--i-accept-spend"` runs it, and
  `ARGS="--rescore <run dir>"` rebuilds results offline. The default model is Meta's Muse
  Spark 1.3 Contributor tier, which may train on prompts (accepted by the operator for this
  open-source repo); the run stays blocked until the operator's OpenRouter account allows
  training providers and confirms the 18+ attestation, which the first reachability call on
  2026-09-09 hit as a 403 naming `age_18plus`.
- Any script that invokes a research provider or the ensemble against real
  questions, including one an agent writes on the spot.

### Free and safe

- Gates and formatting: `make test`, `make test_fast`, `make test_e2e`,
  `make lint`, `make format`, `make typecheck`, `make typecheck_ty`, `make cov`,
  `make audit`, `make deps`, `make lint_imports`, `make precommit*`. One blind spot in
  `make audit`: osv-scanner reads `uv.lock`, and the `curl_cffi` wheel behind the impersonated retry vendors its own
  libcurl and BoringSSL binaries that no lockfile entry names, so a libcurl CVE is not
  reported by that gate and is picked up only by bumping `curl_cffi`.
- Read-only Metaculus, Mantic and GitHub-artifact pulls: `make sync_all` and its parts
  (`sync_research`, `sync_telemetry`, `sync_raw_research`, the `download_*` and
  `backfill_*` targets), the `performance_analysis` package and its width
  monitor, `make score_ghosts`, `make close_margin_watch`, `make supply_probe` and
  `make supply_probe_mantic` (public Mantic reads; `MANTIC_TOKEN` is optional there,
  see "Scheduling reliability" above).
- `make ablation_score`: `--stages score` hydrates every artifact off disk
  (`_hydrate_working_set_from_cache`) and makes no provider call.
- `make dispatch_watch`: one `gh run list`, tabulated per bot workflow and UTC day into
  scheduled versus dispatched runs. The bare `make cronjob_dispatch_setup` is a dry run:
  one read-only GET of the cron-job.org account when both secrets are set, no request at
  all otherwise, and never a write.
- `make cost_report`: cost per question off the telemetry archive (run
  `make sync_telemetry` first): per run (questions, charged dollars, dollars a question),
  per role (dollars a question, prompt and output tokens a question, prompt-cache share,
  largest single prompt) and the week-over-week median. The question denominator is
  `CREDIT_RUN_SUMMARY` where a run has one, else its `FORECASTERS_SURVIVED` lines.
  `ARGS="--days 7"` narrows the default 30-day window.
- `make benchmark_display`: views saved benchmark results, no forecasting.
- `make check_credits`: reads the `/auth/key` balance for both OpenRouter keys.

The test suite is safe by construction, not by convention. The `e2e` marker
means a full-pipeline test with mocked LLMs, and `tests/conftest.py` installs an
autouse `_block_network_egress` fixture that raises on any AF_INET connect to a
non-loopback host. `addopts` deselects only the `live` marker, which is the one
suite that opts out of the egress guard because real calls are its whole point.
So a plain `make test` cannot reach a paid API even if a new test tries to.

`make score_ghosts ARGS="--tournament <slug>"` is worth calling out because
"live pull" reads like spend: it is a Metaculus-only fetch through
`build_performance_dataset`, with no LLM or research provider in the path.

## Credit telemetry and the refill floor

Every run logs OpenRouter balances for both keys at start and end, and computes
per-run spend. The code is `metaculus_bot/credit_telemetry.py`, whose
`CreditTelemetry` is wired into `cli.py`'s `main`; balances come from the
`/auth/key` endpoint via `check_openrouter_credits.py`.

Marker lines land in the `run_logs/` artifact (every bot workflow tees stdout +
stderr), so per-run spend is durably grep-able:

- `CREDIT_BALANCE: key=<donated|personal> phase=<start|end> remaining=... usage=...`
- `CREDIT_SPEND: key=... run_delta_usd=... remaining=... source=...` at end of
  run. `source` is `remaining_delta` (reliable), `usage_delta_unsettled` (a lower
  bound: see the smoke-test grep list above), or `unavailable`.
- `CREDIT_SPEND_UNSETTLED: key=... run_delta_usd=... is a LOWER BOUND ...` beside
  every `usage_delta_unsettled` figure, so a `0.00` is never mistaken for
  no-spend. `scripts/reconcile_credit_spend.py` recovers the settled number.
- `CREDIT_ROLE_SPEND: role=... key=... usd=... calls=... costed_calls=...
  byok_usd=... prompt_tokens=... completion_tokens=... cached_tokens=...
  reasoning_tokens=... charged_usd=... byok_calls=...`: one line per (role, key)
  at end of run, saying WHERE the OpenRouter dollars and tokens went. See
  "Per-role spend" below.
- `CREDIT_FLOOR_BREACH: key=donated remaining=... floor=...` when the donated
  key's remaining balance drops below `OPENROUTER_CREDIT_FLOOR_USD`
  (`constants.py`, $100). That level is an early warning, not an empty tank. Read
  it as "ask Metaculus for a top-up", not "the key is dry".

### Which field the spend delta reads, and the personal key's settlement lag

Verified against live `/auth/key` pulls on 2026-07-17: `usage` counts only spend
billed as native OpenRouter credits. Spend routed through a BYOK provider
integration (the donated Metaculus key routes nearly everything that way) lands
in `byok_usage` instead, so `usage` can sit frozen while real money burns; it sat
at $4.16 across a $3.34 donated-key run that day. `limit_remaining` is
`limit - usage - byok_usage` when the key sets `include_byok_in_limit`, which
makes it the only field that reliably tracks total spend on a limit-bearing key.
So per-run spend comes from the `limit_remaining` delta when the key reports one,
and from the `usage` delta otherwise. The personal key is the "otherwise": it
reports a null `limit_remaining`, and its spend does land in `usage`.

**The personal key's per-run delta is a lower bound, and the cause is settlement
lag.** The BYOK paragraph above is the wrong explanation for it and misled two
separate investigations. On the personal key `usage` genuinely does climb
($154.58 to $160.24 over 2026-07-20 to 2026-07-27), so nothing is hiding in
`byok_usage`. What happens is that OpenRouter has not booked the run's spend by
the time the end snapshot fires, seconds after the last call. Measured over
`backtests/telemetry_archive/credit_balance.jsonl`, 178 paired personal-key runs:
the within-run deltas summed to $3.31 against $5.66 of true lifetime-usage growth
(58% captured), and 160 of the 178 runs reported exactly $0.00. The missing $2.35
is fully accounted for by the gap between each run's `phase=end` usage and the
next run's `phase=start` usage, since $3.31 + $2.35 = $5.66 to the cent. The
money is late, not lost.

The tightest version of that evidence restricts to runs that demonstrably spent.
Of the 25 paired runs carrying at least one `extraction_rung` record (a forecast
provably happened, and `gemini-3.1-pro-preview`, the slot pinned to the personal
key, produced one in all 25), 7 reported exactly $0.00: a 28% false-zero rate on
runs that cannot have been free. `scripts/reconcile_credit_spend.py` recovers a
real figure for all 7, $0.10 to $0.32 each, which is the direct demonstration
that the zeros are lag rather than absence.

There is deliberately no wait-and-re-read in the telemetry. The earliest
confirmed settlement in the archive is 153 s after the end snapshot and the
median is about 25 minutes, so any delay short enough to sit in `cli.main`'s
`finally`, where telemetry must never stall a run, is below anything the data can
show would work: an unverifiable guess that also slows every run. Instead the
marker states its own provenance (`source=usage_delta_unsettled`) and the sibling
`CREDIT_SPEND_UNSETTLED` warning says the figure is a floor, so a `0.00` can
never be misread as "this run was free". The settled per-run number is recovered
after the fact by `scripts/reconcile_credit_spend.py`, which differences each
run's start usage against its successor's, the only place the lag is observable.

A BYOK route on the personal key is a separate blind spot that adds to the lag,
not the same one. Those calls (the OpenAI slugs, per "Per-role spend" below)
never reach `usage` at all, so no balance field of either key ever sees them and
they are visible only on the `CREDIT_ROLE_SPEND` ledger.

Two smaller caveats on any of these numbers: an out-of-band top-up mid-run skews
the remaining-based delta (rare, and per-run spend is indicative anyway), and
OpenRouter caches the balance values briefly, so exact figures are not something
to build on.

Balance fetching itself can never fail a run. Any error is logged as a WARNING
and read as "unknown", and unknown never trips the floor. The catch in
`_fetch_snapshot` is deliberately total rather than a curated tuple: `cli.main`
calls `log_end_and_check_floor` from a `finally`, so an escape there replaces
whatever the run was already raising and takes the whole end-of-run diagnostic
surface with it (report summary, alertable arithmetic, deprecation tripwire, all
downstream). A narrow tuple already missed three real shapes: `FileNotFoundError`
from a stale `SSL_CERT_FILE`, `httpx.InvalidURL`, which is not an
`httpx.HTTPError` subclass, and the `RuntimeError` this repo's own autouse
network guard raises. The snapshot's field reads sit inside the same `try` for a
related reason: `fetch_auth_key` returns `payload.get("data", payload)`, so a 200
whose body carries a non-mapping `data` (`{"data": null}`, `{"data": [...]}`)
yields a non-dict and `data.get(...)` raises `AttributeError`. Keeping those
calls under the `try` degrades that malformed-but-200 case to a WARNING and
`None` like any other fetch failure.

### Per-role spend (`CREDIT_ROLE_SPEND`)

The per-key deltas above say what a run cost; the role lines say which part of
the pipeline spent it: `forecaster:openai` / `forecaster:anthropic` /
`forecaster:google` (the vendor slot, so the series survives a model swap),
`stacker`, `stacker_fallback`, `parser`, `summarizer`, `crux_analyzer`,
`native_search`, `targeted_search`, `gap_fill_analyzer`, `gap_fill_resolver`,
`gap_fill_v2_driver`, `market_query_author`, `market_ranker`,
`financial_classifier`, `page_digest_extractor`, `perplexity_research`. This list is the only
enumeration: each role is a string literal at its builder call site, stamped onto every completion
through `credit_telemetry.llm_call_metadata`.

How the number is produced, because it decides how to read it:

- Every LLM built through `build_llm_with_openrouter_fallback(..., role=...)`
  (and the raw-`acompletion` gap-fill v2 driver) stamps a litellm `metadata=`
  tag with its role and the key it bills (`donated` for the wrapper's primary,
  `personal` for its fallback or a personal-key-pinned model, `direct` for a
  non-OpenRouter slug). `metadata` is a litellm-only kwarg: it lands in
  `litellm_params["metadata"]` for callbacks and is never forwarded to
  OpenRouter, and `GeneralLlm` passes unknown kwargs through to `acompletion`
  unchanged, so a tag stamped at construction reaches every completion that LLM
  makes. A litellm success callback (`RoleSpendTracker`) reads the tag back
  together with **OpenRouter's own per-call usage accounting** off the
  response, which is the provider's figure, not litellm's price table. The
  callback is the one seam that still sees the raw `ModelResponse`:
  forecasting-tools' `GeneralLlm.invoke` returns only the text, and the
  `TextTokenCostResponse` it builds keeps litellm's `response_cost` hidden param
  (about $0 for every BYOK call) and drops the usage object. Only
  `async_log_success_event` is implemented, because litellm skips the sync hook
  for `acompletion` unless a sync-only callback is registered, and implementing
  both would double count. The callback runs on the event loop inside litellm's
  logging worker and the accumulation has no `await`, so the ledger needs no
  lock (the same bytecode-atomic argument
  `fallback_openrouter.record_donated_key_fallback` makes for its counters).
- The usage fields, and what each one means for the money:
  `usage.cost` is what OpenRouter charged the key's credits; on a BYOK route that
  is only OpenRouter's fee (5% of list price, waived under the plan's monthly
  allowance, so 0 in practice), off BYOK it is the whole charge.
  `usage.cost_details.upstream_inference_cost` is the provider's own charge on a
  BYOK route, billed to the BYOK account's owner (Metaculus for the donated key;
  the operator's own provider account for a personal-key BYOK route), and it is
  what `/auth/key` books as `byok_usage` and subtracts from `limit_remaining`.
  OpenRouter's docs say it is 0 or null off BYOK, but since at least 2026-09-03
  it is reported on non-BYOK calls too, equal to `cost`. `usage.is_byok` says
  which route the call took. `byok_usd` is the upstream sum on its own.
- **`usd` double counts non-BYOK rows and is kept as it was.** `usd` is
  `cost + upstream_inference_cost` summed over the row, the definition it shipped
  with, and a marker field's meaning never changes in place. Off BYOK that adds
  the echoed upstream figure to the real charge. The 2026-09-09 cost pass proved
  it on the three production runs whose only personal-key row was the Google
  forecaster slot: the key's settled usage moved by that row's `byok_usd` to the
  cent (0.57 against `usd=1.1433`, 0.14 against 0.2787, 0.67 against 1.3268), so
  the Google slot costs $0.13 a question, the cheapest of the three, not the
  $0.27 the raw ledger showed. Since 2026-09-09 every row also carries
  `charged_usd` (`cost`, plus the upstream cost only on calls with `is_byok`
  true: the money actually charged across both payers) and `byok_calls` (how
  many of `calls` routed BYOK). Sum `charged_usd`; `usd` on a pre-2026-09-09
  personal-key row that reads twice its `byok_usd` is the double count.
- **Which balance sees which part.** The donated key's `limit_remaining` drop
  covers `byok_usage`, so it matches `charged_usd`. The personal key's `usage`
  moves by the `cost` part only (`usd - byok_usd`), so a personal-key BYOK route
  never appears in any OpenRouter balance and is visible only on this ledger.
  The 2026-09-03 dry-donated-key run (33775800806) is the worked example: the
  personal key's usage grew by exactly the Anthropic and Google slots' `cost`
  ($1.18 + $0.65 = $1.83, settled and unchanged the next day), while the
  OpenAI-model rows (the resolver, the OpenAI slot, the v2 driver, native
  search, the summarizer: $8.53 of `byok_usd` with `cost` 0) never touched the
  credits. The ledger therefore implies the personal OpenRouter account has a
  BYOK key configured for OpenAI, billed to the operator's own OpenAI account;
  `byok_calls` on the next personal-key fallback run confirms it directly.
- `usd=n/a` means none of that row's calls carried cost data. It is never a
  fabricated zero; `costed_calls` says how many of `calls` the sum covers.
- The four token fields (since 2026-09-09) sum over every call of the row:
  `prompt_tokens` / `completion_tokens` are the base counts, `cached_tokens` is
  the prompt tokens the provider served from its prompt cache
  (`prompt_tokens_details.cached_tokens`), and `reasoning_tokens` the hidden
  reasoning output (`completion_tokens_details.reasoning_tokens`). Cache hit rate
  per role is `cached_tokens / prompt_tokens`; the gap-fill v2 driver should sit
  near 0.8 once the ghost call reuses the cache (`docs/agentic_gap_fill.md` "The
  ghost forecast"), and a forecaster slot near 0.
- `role=untagged` means a completion nobody stamped: one of forecasting-tools'
  own helpers (`SmartSearcher`), an ablation or benchmark harness, or a builder
  call site that forgot its `role=`. It is visible on purpose rather than folded
  into another row. `key=unknown` is the same thing for the key.
- The two `metadata=` field names (`role`, `key_alias`) and the four key-alias
  values (`donated`, `personal`, `direct`, `unknown`) are separate vocabularies:
  the first pair names FIELDS, the second names KEYS. `donated` and `personal`
  are the `KEY_SPECS` aliases verbatim, so `CREDIT_ROLE_SPEND key=` joins onto
  `CREDIT_SPEND key=` and `CREDIT_BALANCE key=` without translation. `direct` is
  a non-OpenRouter slug, a `perplexity/` or `exa/` model billed to its own
  provider key: outside this ledger's remit, but still counted rather than
  dropped.
- Dollar figures on these lines render at four decimals, not the balance lines'
  two, because a per-role figure is a fraction of a cent per call: the parser
  costs about $0.0005 a question.
- `max_prompt_tokens` on each row is the largest single prompt that role sent
  this run, the packet-size read a summed `prompt_tokens` hides, and the same
  callback logs `PROMPT_SIZE_ALERT` (WARNING) for any one call whose prompt
  exceeds `PROMPT_TOKENS_ALERT_THRESHOLD` (150k; the forecaster prompt is about
  17k and the v2 loop peaks near 41k). The alert names the question only when the
  call site stamped one, which today is the v2 driver alone; it reads, never gates.
  Field detail: `docs/telemetry_markers.md` "PROMPT_SIZE_ALERT".
- Not on OpenRouter, so never in this ledger: Gemini grounded search and gap-fill
  v2's `read_document` (google-genai on the personal Google AI Studio key), the
  AskNews subscription, Exa. The ledger is therefore an OpenRouter-only figure;
  the all-in $2.07 to $2.21 a question below adds those from their own consoles.
- The lines are logged from the same `finally` as `CREDIT_SPEND`, after the
  forecast loop has drained litellm's callback queue (`cli.py`
  `_forecast_with_callback_drain`), so a crashed run still reports what it booked.
  That drain is bounded at `LITELLM_CALLBACK_DRAIN_TIMEOUT_S` (10s) and swallows
  its own timeout, because telemetry must never be able to fail a run that already
  published. When the bound trips, the run logs one
  `LITELLM_CALLBACK_DRAIN_TIMEOUT` WARNING and the rows below it may be missing
  the last few completions. Treat that warning as "this run's ledger is a lower
  bound"; without it, the ledger covers every completion of the run. That warning
  is harvested in its own right, as `litellm_callback_drain_timeout.jsonl` (one
  row per affected run, carrying the bound it used), so the caveat is answerable
  offline instead of only from a live log. It carries its own marker prefix rather
  than `CREDIT_ROLE_SPEND` on purpose: the ledger's harvester spec expects
  `role=` / `key=` / `usd=` / `calls=` fields, so prose under that prefix would
  pollute every grep of a run log without ever parsing as a row. Since 2026-09-04
  the prefix has its own spec (`scripts/telemetry/markers.py`,
  `litellm_callback_drain_timeout`), which reads the `within <n>s` clause of the
  message; the rest of that sentence is free to reword, that clause is a data
  contract.
- The 10s bound is reachable two ways, not one. A wedged worker is the obvious
  one (a worker loop that dies on any non-`CancelledError` leaves `queue.join()`
  outstanding forever). The other is a single callback slower than 10s, which
  litellm itself still considers healthy: it allows each queued coroutine 20s
  (`LOGGING_WORKER_MAX_TIME_PER_COROUTINE`), twice this window. It is left at
  10.0 deliberately. Both callbacks the bot registers are in-memory arithmetic,
  so raising the bound would be an unverified retune whose only effect is a longer
  pointless wait on a dead worker.

Harvested as `credit_role_spend.jsonl` in the telemetry archive.
`uv run python scripts/reconcile_credit_spend.py --roles` (free, offline) prints
each run's role-ledger total (`charged_usd`, falling back to `usd` on the rows
archived before 2026-09-09) beside its settled per-key spend (the two measure
the same money from opposite ends, so their ratio is the ledger's own coverage
check), plus a per-role table over the selected runs. On the personal key the
ratio compares all charged money against a credits-only balance, so a run that
fell back to the personal key for OpenAI models reads above 100% by exactly the
BYOK part.

Every run also ends with one `CREDIT_RUN_SUMMARY` line beside the rows: the
ledger folded to dollars per question, split by key, with the run's prompt-token
total, cache share and largest single prompt (field detail in
`docs/telemetry_markers.md`). **`make cost_report` is the instrument for the
per-question figure**: over the last 30 days of the archive (`--days` to change
it) it prints each run's questions, charged dollars and dollars per question, each
role's dollars and tokens per question with its cache share and largest prompt,
and the median dollars per question this week against the prior week. It takes a
run's question count from `CREDIT_RUN_SUMMARY` and, on the runs archived before
that line, from its `FORECASTERS_SURVIVED` lines. Free and offline; run
`make sync_telemetry` first.

**The per-question spend figure to quote is $2.07 to $2.21 all in** as booked,
about $2.00 once the ledger's non-BYOK double count is removed, measured on
2026-09-09 over the 14 question-intakes that carry a role ledger
(`scratch/cost_pass_2026-09-09/COST_PASS.md`). The three forecaster slots sit
within 12% of each other at $0.24 to $0.27 a question; gap-fill v1's resolver is
the largest single line at about $0.80. The `$0.38 to $0.41` quoted here until
2026-09-09 was an OpenRouter-only LOWER bound read off one key's settled balance,
and it excluded the BYOK-routed OpenAI spend, Google AI Studio prepaid (Gemini
grounded search and gap-fill v2 document reads), the AskNews subscription, and
Exa; it was wrong by five-fold, and neither it nor the older "~$3.05 → ~$1.65
after the 6→3 roster drop" estimate may appear in a roster re-add decision.
`CREDIT_ROLE_SPEND` plus `make cost_report` is how a re-add gets priced per role
rather than estimated.

A floor breach does not abort the run. Forecasting and publishing complete
normally, and outside a suppression window `cli.py` then exits non-zero so the
GitHub Actions check turns red as a reminder to ask Metaculus to top the donated
key up. The floor is an EARLY-WARNING level ($100, roughly 56 questions of
runway at $1.79 a question on the donated key) rather than an empty tank, because only Metaculus can refill this key and a
reminder that arrives when the balance hits $1 arrives too late to act on. The
floor is only checked against the donated key (the personal key reports no
`limit_remaining`). Per-run spend prefers the `limit_remaining` drop because the
donated key routes nearly all spend through BYOK provider integrations, which
leaves the plain `usage` field frozen while real money burns.

### The credit-alert suppression window (closed since 2026-09-03)

Credit alerting is ON. It was suppressed from 2026-07-26, when the donated key
drained and the operator started funding the season out of pocket, so an empty
donated key was the expected state rather than a defect. Metaculus granted $1,500
of credits on 2026-09-03, and `CREDIT_ALERT_RESUME_DATE` in `constants.py` was
moved up from 2026-09-10 to `2026-09-03` that day: a credit shortfall reddens CI
again. The machinery below is unchanged and re-armable: push
`CREDIT_ALERT_RESUME_DATE` forward in `constants.py`, or set
`OPENROUTER_CREDIT_ALERT_RESUME_DATE` in the workflow env, and the window reopens
with no other edit. Inside a window two paths are gated, because either one alone
would keep the check red:

1. The floor breach. `cli.py` skips the `sys.exit(1)` and logs an INFO line
   saying the breach was observed but alerting is suppressed until the resume
   date.
2. The credit-caused donated-to-personal key fallbacks. Each fallback counts
   toward `alertable` outside the window. `record_donated_key_fallback` tracks the
   suppressible subset in `_credit_key_fallback_count`, a subset of the
   all-causes `_generic_key_fallback_count`, and `cli.py` subtracts the subset
   back out while alerting is suppressed. Every event is counted exactly once:
   generic adds it, at most one subset subtracts it. That is why the whole
   accounting block in `record_donated_key_fallback` has to contain no `await`
   after the threaded probe: `+=` on a module global is interruptible between
   bytecodes, so an await there would let N forecasters failing on one dry key
   race the increment, undercount the generic total, and take a degraded run
   green.

Non-credit fallback causes alert in full whatever the window says, since each
means real breakage rather than an empty wallet: 401 invalid or disabled key, 404
"no allowed providers", 429 rate limit, and the guardrail / data-policy block.
Bot-side degradation is untouched by a suppression too: every counter in the
`Degradation counters:` summary always alerts in full (they are enumerated under
"Reading run logs" below).

### What a dry donated key actually returns (and the drained-vs-revoked probe)

A breached per-key spend cap does **not** come back as the 402 OpenRouter's
error docs describe. It comes back as HTTP **403** with the message
`Key limit exceeded (total limit)`, and litellm has no 403 branch for
OpenRouter, so it always surfaces as a bare `litellm.APIError` whose body
carries a `"code":403` field. On 2026-07-26 that cost a tournament run two of
three forecasters, native search, the AskNews summarizer, the financial-data
classifier, prediction-market keyword extraction, and both gap-fill passes: the
wrapper's negative rule vetoed any message containing "403" (written for content
moderation, where both keys really would refuse), so the operator's funded
personal key was never tried. The classifier now matches the phrase
`key limit exceeded`, which flips both the fallback decision and the credit
classification through the single shared helper (`_is_credit_failure`).

The cue has to be the full phrase. `limit exceeded` alone is a substring of
`rate limit exceeded: free-models-per-day`, so the short form would classify
every 429 as an empty wallet and silently exempt real rate-limit breakage from
alerting for the whole suppression window.

Text alone cannot tell a genuinely **drained** key from one Metaculus
**revoked** or **re-capped to zero** (all three produce that same 403), and the
operator wants opposite CI colors for them. So on the first spend-cap failure of
a run, `credit_telemetry.classify_donated_key_state` reads the free, read-only
`/auth/key` endpoint once (verdict cached for the process) and classifies. It
goes through the same `check_openrouter_credits.fetch_auth_key` as the start and
end balance telemetry, so "how much is left on the donated key" has one endpoint
and one parser rather than two. With no donated key configured it returns before
any network call at all, which also keeps the probe silent in tests that do not
stub it. The classification:

| `/auth/key` says | State | Alerting |
| --- | --- | --- |
| 200, cap > 0, nothing remaining | `drained` | suppressible: the expected empty wallet |
| 200, cap == 0 | `zeroed` | **red**: Metaculus cut us off, never an "empty wallet" |
| 401 / 404 | `revoked` | **red**: key is gone, not empty |
| 200, money remaining | `funded` | **red**: the failure was not about credit |
| probe failed, or no donated key configured | `unknown` | **red**: fail safe |

"Nothing remaining" is `limit_remaining <= 0` rather than `== 0`, because
OpenRouter clamps that field at 0 even when the true arithmetic is negative (live:
`limit=850`, `usage=4.39`, `byok_usage=846.42`, reported as 0.00). An uncapped key,
which reports no `limit` or no `limit_remaining` at all, classifies as `unknown`:
it has no cap to exceed, so a spend-cap failure on one is unexplained rather than
expected.

Only `drained` is ever subtracted from `alertable`, and only inside a suppression
window (none is open since 2026-09-03). A probe that errors or times out classifies
as `unknown` and stays red, so a broken probe can never silently turn a red run
green.

The verdict is cached once per process behind a `threading.Lock`, and both halves
of that matter. Without caching, a run that lost every donated-key call would fire
one HTTP request per failure, and caching failures counts as much as caching
verdicts, since a dead endpoint would otherwise cost one timeout per failed call.
The lock is `threading` rather than `asyncio` because every production caller
arrives on an `asyncio.to_thread` worker (`fallback_openrouter.record_donated_key_fallback`),
so the contention is between real OS threads. The one-verdict half is the more
important one: without the lock each caller keeps its own probe result, so an
intermittently failing `/auth/key` splits a single drained-key incident into some
suppressed and some alertable events, and `cli.py` then exits red on the very
condition the suppression window exists for. The `DONATED_KEY_STATE` line is
logged inside the lock too, so it appears exactly once per run; N copies of one
verdict would read as N separate probes to whoever greps the run log. A run that
never needed to probe leaves the cache at `None`, which the CLI renders
differently from any verdict.

The probe is what the *ambiguous* spend-cap 403 needs, so it is the only path that
pays for one. A documented 402 or plain insufficient-credit response says the
wallet is empty and nothing else, so `is_suppressible_credit_error` suppresses that
family before reaching the probe at all. That is deliberate, since it predates the
discriminator and an unreachable `/auth/key` must not change long-standing
behavior. Read the table above as the verdict on a spend-cap 403 specifically, not
on every credit failure (`test_documented_402_needs_no_probe` in
`tests/test_fallback_openrouter.py` pins the carve-out).

`DONATED_KEY_PROBE_TIMEOUT_S` is 5.0 s, shorter than the shared `fetch_auth_key`
default (`AUTH_KEY_REQUEST_TIMEOUT_S`) by design: this probe can fire mid-run, so
it must not be able to stall a forecast, while the shared default is fine for the
CLI and the start/end telemetry, which both run outside the forecasting window.
It bounds the probe, but read what shape of promise
that is: httpx applies a bare float **per network operation** (connect, read,
write and pool each get the full budget independently), so it is not a cap on
elapsed time. A server trickling bytes slower than the read timeout resets the
clock on every chunk, and a probe can run many multiples of the nominal budget
(measured against a local trickling server, a one-second timeout took ten
seconds to return twenty bytes). The hard total cap therefore lives at the one
latency-sensitive call site rather than in the timeout: on the fallback path
`record_donated_key_fallback` runs the probe on `asyncio.to_thread` under an
`asyncio.wait_for`, so the awaiting coroutine gives up on schedule however long
the socket takes. `wait_for` doesn't kill the worker thread, so a trickling probe
outlives that cap, orphaned, holding a socket and (under the probe's lock)
writing the cache, while the fallback proceeds without it. Callers outside that
path (the CLI, the start/end telemetry) run outside the forecasting window and
take the per-operation budget only. The state is logged as
`DONATED_KEY_STATE: state=<state>` (INFO for `drained`, WARNING for everything
else) and is echoed in the end-of-run summary as `donated_key=<state>` whenever a
probe actually ran.

Fallback **routing** reads the status the provider reported
(`llm_retry.llm_status_code`, an int already on the exception) and never a live
balance. A reported 403 falls back only on the spend-cap phrase or route-scoped
wording; a reported 402 always falls back; an exception carrying no status falls
back on text alone. The
`/auth/key` probe is consulted for alerting only (`is_suppressible_credit_error`),
so a stale or cached read reporting `funded` can never strand the ensemble on a
dry key. That is the exact failure this change exists to fix.

Two related hardenings ride along, both about how little the body can be trusted.

First, "was this about money?" has exactly one arbiter, `_is_credit_failure`
(in `fallback_openrouter.py`, whose docstring is the canonical version of this),
which both the routing decision and the alerting counter reach through. It reads
three tiers in a fixed order:

1. The spend-cap phrase `key limit exceeded` outranks everything, including the
   moderation veto below, and fires on any status or none. The production body
   renders as `403 Forbidden: Key limit exceeded`, and `forbidden` is both a
   moderation cue and generic HTTP boilerplate, so gating the phrase behind the
   veto would keep the dry key from falling back all over again.
2. Otherwise, a reported status decides alone: credit means exactly 402. So a
   reported 402 outranks moderation wording (`APIError(status_code=402,
   message="Blocked by moderation policy")` both falls back and is
   credit-classified), and credit English on any other reported status does not
   classify.
3. With no status reported, moderation wording (`moderation`, `forbidden`,
   `flagged_input`, `flagged for`) vetoes; failing that, a bare `402` or one of
   `payment required` / `insufficient credit` / `out of credits` /
   `insufficient funds` classifies.

That last ordering is why `insufficient credit` alone classifies as credit while
`blocked by moderation: insufficient credit` does not.

Second, OpenRouter moderation 403 bodies include `flagged_input`, up to ~100
characters of our own prompt replayed back, and a forecasting prompt full of
dollar figures and bill numbers can easily contain the token `402`. A bare `402`
substring match therefore read an ordinary moderation refusal as an empty wallet,
billing the personal key for a call that would refuse again, and exempting a
real moderation block from alerting. Everything after a prompt-echo marker is now
stripped before any word cue reads the body, and the bare digits are only trusted
when nothing in what remains looks like a moderation refusal. Word cues only,
deliberately: a genuine 402 links to a key hash with a small but non-negligible
chance of containing the substring `403` somewhere in it, and reading that as
moderation would break the long-standing 402 fallback. The odds are derived (and
pinned as bands) by `test_key_hash_status_collision_is_small_but_nonnegligible`
in `tests/test_llm_retry.py`, which is the only place that arithmetic lives.

Nothing is silenced. Every `CREDIT_*` marker line, `CREDIT_FLOOR_BREACH`
included, and every `PAID PERSONAL-KEY FALLBACK` warning fires exactly as
before; only the process exit status and the `alertable` arithmetic change. The
end-of-run summary renders the breakdown, including how many credit events were
suppressed and until when. The window is read from the system clock at call
time, so alerting resumes on the resume date with no redeploy, and behavior from
that date on is what it was before the suppression. `credit_alerts_active()` in
`constants.py` takes an optional `today` so tests pin both sides of the
boundary.

### Checking balances

The donated Metaculus OpenRouter key (`OAI_ANTH_OPENROUTER_KEY`) is shared and
rate-limited, so its burn rate is worth checking periodically rather than only
when a run complains. `make check_credits` prints `limit` / `limit_remaining` /
`usage` for both `OAI_ANTH_OPENROUTER_KEY` (donated) and `OPENROUTER_API_KEY`
(personal); pass `ARGS="--key donated"` to check just one.

```bash
make check_credits                    # both keys
make check_credits ARGS="--key donated"
```

Raw curl backup, which avoids putting the key on disk by pulling it from `.env`:

```bash
curl -s -H "Authorization: Bearer $OAI_ANTH_OPENROUTER_KEY" \
  https://openrouter.ai/api/v1/auth/key | jq
```

Never paste a full key into chat and never commit one; `.env` is gitignored.

## Backtesting

The primary benchmark scores bot predictions against actual question
resolutions. It spends API credits (it runs the real ensemble and research), so
it is gated by the cost rule above.

```bash
make backtest_smoke_test   # 4 questions
make backtest_small        # 12
make backtest_medium       # 32
make backtest_large        # 100
```

The prediction-market snapshot and the resolution-source fetcher are hard-off
under `is_benchmarking=True` to avoid leaking post-resolution data, so their
forecasting value cannot be measured by these targets. They were validated via
manual `test_bot.yaml` prod-mode runs and opt-in live integration tests instead.

To backtest against cached, non-leaky research from the archive:

```bash
make backtest_with_cache   # uses backtests/research_archive/latest
```

The old `community_benchmark.py` path is deprecated: Metaculus removed the
`aggregations` field from the list API, so baseline scoring is broken.
`make benchmark_display` still views old results.

The same API removal is why `question.community_prediction_at_access_time` is
always `None` on a newly-fetched question. Benchmark files written before the
removal still carry real values in that field, so
`ensemble_analysis/ensemble_simulator.py` keeps reading it for its binary
baseline score: on fresh data it simply finds nothing and skips the question.

## Comment privacy

Every rationale comment the bot posts is private: the framework's `post_question_comment` sends
`is_private: true` and the repo never overrides it. That is by Metaculus's request, not an
accident. The FutureEval Bot Tournament Resources Page (Metaculus notebook 38928, read 2026-09-10)
says bots "should only leave private comments on questions. These will automatically be made public
at regular intervals for FutureEval tournaments. Though comments will stay private for any questions
on the main site, unless specific permission is given", and its rules section asks bots to "use
private notes as their comment type" and says Metaculus converts them to public comments after
questions close weekly. So a summer tournament comment reads public in the API while a fresh fall or
Metaculus Cup comment reads private, and a comment missing from the bot's default author listing is
expected until that weekly flip, not a publish failure. Do not pass `is_private=False`. Analysis
tooling reads both listings (`fetch_bot_comments` in `performance_analysis/collector.py` and
`scripts/backfill_research_from_comments.py`), so a private comment is still visible to the residual
round. Receipt: the six 2026-09-07 fall comments were private and invisible to the collector until
367b67c; the operator confirmed keeping them private on 2026-09-10.

## Performance analysis and the width monitor (read-only, free)

This section is the runbook: the commands, and what each one prints. The
methodology and the conventions that make a number trustworthy (era bucketing and
the merge-to-main rule, the exclusion cohorts, the PIT convention, the starved
outer tail, the supply probe, per-model recovery, the spot-peer rule,
`spot_peer_delta`, and the clip-threshold sweep) live in
`docs/performance_analysis.md`.

**Routine residual refreshes use the committed CLI and `RoundSpec` library API described
there and in the residual playbook.** Do not write new scratch scripts, copy prior-round
drivers, or recreate standard dimensions without an agreed functionality change. Keep dated
`scratch/residual_<date>/` directories for round data and reports. Focused follow-up analyses
may use scratch scripts; if routine support is missing, report the gap and agree on a maintained
addition.

`metaculus_bot/performance_analysis/` evaluates the live bot's calibration
against actual resolutions. The pull hits only the Metaculus API (resolved
questions plus the bot's own comments, user id 275109, auth via
`METACULUS_TOKEN`). It makes no LLM or research calls and does not publish, so
it is not subject to the cost gate.

```bash
uv run python -m metaculus_bot.performance_analysis --tournament <slug> --output <path>
```

The `--tournament` default is `DEFAULT_TOURNAMENT` (`performance_analysis/cli.py`)
and lags the live season, so pass the current slug explicitly. It collects one Metaculus slug
per call; on a routine refresh, also pass `--prior <previous same-slug dataset>` to detect
platform re-resolutions. Pass `--cached <path>` to re-analyze a saved dataset without re-fetching.

The width monitor (`performance_analysis/width_monitor.py`) tracks how wide the
published numeric distributions are and how well that width is calibrated, split
by config era. Era-bucketing is mandatory for any calibration claim: the bot's
roster and pipeline change often enough that pooled calibration numbers are
misleading. The monitor reports central-80% and central-50% coverage with
Jeffreys-prior CIs, tail coverage (cov@10/50/90), PIT std, median relative
band width, and `band_miss (lo/hi)` per era. That last one is the out-of-band
rate split by tail: it distinguishes a band that is too tight (both tails
elevated) from one of roughly the right width that is mis-centered (misses piled
in one tail), which `cov80` cannot express and which call for opposite
corrections.

A resolution the platform reports as out of range (`above_upper_bound` /
`below_lower_bound`) carries no value, so its PIT is a SET rather than a number:
`[cdf[-1], 1]` above the ceiling, `[0, cdf[0]]` below the floor. Those readings
count toward every coverage column when the interval intersects the band, and are
excluded from PIT std and mean PIT, where no midpoint is imputed; the
`set-valued (pt n)` column states how many were excluded and what the
point-metric denominator therefore is. The convention lives in
`analysis.out_of_range_pit_reading` / `analysis.PitReading` and both PIT paths
read it. It matters because our own CDF decides the interval: q44842 deliberately
published 13% of its mass above the displayed ceiling, resolved
`above_upper_bound` and won spot peer +24.4, which the old PIT-1.0 convention
scored as a high-side band miss. A starved tail (`cdf[-1]` at the 0.999
open-bound floor) still misses the band, because that interval lies wholly above
0.90.

The same command prints a second, per-QUESTION section: the **starved outer tail**
scan, which lives in `performance_analysis/outer_tail.py` (the width monitor owns
only the CLI wiring). `docs/performance_analysis.md` defines the defect, why it is a
cliff at a fixed location that widening does not fix, and what the archived fire rate
means; this section covers running it and reading a row. q45218 published its winning
rig-count forecast with 27 such bins starting one rig above its declared p99, a flat
-219.5 zone sixteen rigs from the resolution, and the same shape is what made q44182
(-219.0) the worst record on the board. A side is flagged when its band's mean per-bin
mass is under `STARVED_OUTER_TAIL_FLOOR_MULTIPLE` (2.0) times the platform's per-bin
minimum step (`0.01/N`); each flagged row reports the declared anchor, how many member
curves set it and how many were dropped, the displayed bound, the band's mass and bin
count, the mass sitting beyond the bound, and the log score a resolution in the band's
thinnest bin would earn. The member census is there because the anchor is a median over the members
whose declared curve is usable, so dropping one (an anonymous positional
`Forecaster N` bucket, an unparseable curve, or one carrying fewer than two
distinct percentile labels) moves the boundary the verdict is measured against;
the section header states how many sides dropped a member, and every scanned side
carries `members_used` / `members_dropped` in the JSON dump.
`--output-starved-json <path>` writes every scanned side with its verdict,
flagged or not. This is a DETECTOR: any width response stays gated on the
standing `k_tail` hold, and there is no publish-time twin of it.

Its era boundaries are **merge-to-main timestamps** (`WIDENING_FLIP`,
`TS_ANCHOR_ENABLE`), not authoring dates: prod runs from `main`, so a change is
live only once its merge commit lands there, and keying on the authoring date
files every run in the author-to-merge gap under the wrong config. Empty eras are
omitted, so while no post-july15-bundle numeric has resolved the `ts_anchor` row
is absent from the table rather than present-and-empty.

```bash
uv run python -m metaculus_bot.performance_analysis.width_monitor --tournament <slug>
# or against a cached dataset:
uv run python -m metaculus_bot.performance_analysis.width_monitor --cached <path>
# drop a standing exclusion cohort from every row; the excluded count is rendered
# in the table, so the exclusion is never silent. Three shorthands, known_bug
# (since-fixed pipeline defects), degraded_run (dry-key 1-of-3 publishes) and
# partial_degraded (2-of-3), compose with each other and with explicit ids; the
# id sets live in performance_analysis/cohorts.py (EXCLUSION_COHORTS):
uv run python -m metaculus_bot.performance_analysis.width_monitor --cached <path> --exclude-qids known_bug,degraded_run
```

Before either analysis, run `make sync_all` (also read-only and free) so the
local archives are fresh: the per-provider research archive
(`backtests/research_archive/latest/`), the run-log telemetry archive
(`backtests/telemetry_archive/`), and the raw research-provider payload archive
(`backtests/research_archive/raw/`). Use `sync_all` rather than one of the
narrower `sync_*` targets: it is a single download pass over the union of
artifact families, so it is cheaper than running them in sequence, and GHA
artifacts expire at 90 days, which makes anything a partial pull skipped
permanently unrecoverable. The twice-weekly launchd job in
`scripts/research_sync/` is wired to `sync_all` for the same reason.

### Auditing a round pull, and probing the season slugs

Use these maintained commands to preflight a pull and audit its output:

```bash
make probe_slugs                                     # before the pull: is the season config current?
make verify_pull ARGS="--records scratch/residual_<date>/perf_<slug>.json \
  --prior-records scratch/residual_<prior>/perf_<slug>.json"
```

`verify_pull` requires the raw posts from the same pull in a sibling file named
`perf_<slug>_checkpoint.json`. The performance-analysis CLI does not write this checkpoint;
`collector.fetch_resolved_questions` returns raw posts, but a separate call is a separate
snapshot. Comparing separate snapshots cannot establish coverage of the exact scored pull.
If maintained CLI support for checkpoint output is needed, agree on that addition rather than
adding a scratch capture script.

These are pull checks, not a full-round driver. For routine round assembly, use the committed
`RoundSpec` library API described in `docs/performance_analysis.md` and the residual playbook.
There is no integrated multi-source round CLI; keep `scratch/residual_<date>/` for round inputs
and outputs rather than copied driver or dimension scripts.

`scripts/probe_slugs.py` reads one project object per candidate slug,
`/api/projects/tournaments/<slug>/`, and reports whether it exists, its visibility,
question count and forecasting end date. Candidates are the slugs the repo's own
constants point at (`supply_probe_platforms.DEFAULT_SLUGS`) plus the season-successor
spellings for this season and the next, generated from a template list rather than
hand-maintained, so the tracked file needs no per-round edit. The project object rather
than the tournaments LIST because the list omits anything whose `visibility` is
`unlisted`, which is precisely the state a new season sits in before its first question.
It flags two things: a configured slug that is absent or past its forecasting window,
where every scheduled run finds no question and forfeits the season silently, and a live
season under a spelling the constants do not name. Post and question counts stay
`make supply_probe`'s job; this probe pages nothing.

`scripts/verify_pull.py` is fully offline. It reads the pull's checkpoint (the raw post
payloads saved beside the records file as `<records stem>_checkpoint.json`, so one
`--records` path locates both) and runs five checks: which resolved posts produced no record and why, which
resolved members of a covered group post produced none, the diff against the prior round
(`--output` writes that cohort as JSON), whether every platform score on an overlapping
record reproduces exactly, and whether the pull still parses per-model forecasts and
comment text out of the bot's comments. Check 4 is the one with a receipt: Metaculus
re-resolves in place without moving any timestamp, so a moved score means a table in the
prior round's write-up went stale silently.

A recordless post classifies as already-known when the PRIOR pull fetched it and emitted
no record for it, which is derived from that round's own checkpoint rather than listed.
That is deliberate: these are POST ids and a round-relative expected state, not an
incident-defined scoring cohort, so they do not belong in
`performance_analysis/cohorts.py` (question ids, one per incident). A recordless post the
prior pull never fetched reads INVESTIGATE until a human classifies it and the next round
inherits the verdict for free, which is how post 44950 surfaced on 2026-09-09 after
entering the resolved list mid-season.

### The persisted artifact store, and re-parsing for free

`sync_all` downloads each artifact into `backtests/gha_artifact_store/<artifact-name>/`
and leaves it there: the extracted contents as `gh run download` unzipped them,
plus a `_meta.json` holding `artifact_id` / `name` / `created_at` / `run_id`. All
three archives are parsed FROM that store, never from a self-destructing temp
dir, which is the point: 90 days is a hard ceiling for this repo
(`{"days":90,"maximum_allowed_days":90}`), so GHA is a staging area and local
disk is the source of truth the moment an artifact is grabbed. An artifact
already in the store is never re-downloaded: uploads are immutable, so only
absent or half-extracted dirs are fetched.

```bash
make resync_from_store    # rebuild all three archives from local disk, zero network
```

Reach for that after fixing an ingest or parse bug: the bytes are already on
disk, so a corrected harvest costs nothing and still works on artifacts GitHub
has since deleted. Each sync script also accepts `--from-store` / `--store-dir`.
In `download_research.py` the two offline flags differ in an important way:
`--rebuild-only` re-merges the records already in `by_qid/`, while `--from-store`
re-reads the persisted JSONL and so can RECOVER records a past ingest bug
dropped. The offline path cannot ask GitHub which workflow a run belonged to, so
it recovers that from the telemetry archive's own `runs.jsonl`; a run entering the
store for the first time during an offline re-parse reads `workflow: unknown`
until the next online sync.

Storage is not a concern at this scale: 859 artifacts occupy 38 MB (median 4.4
KiB, mean 44 KiB, largest under 1 MB), and at ~13 artifacts/day that is roughly
17 MB/month, so about 210 MB after a year. Nothing needs compression, and
nothing is pruned on purpose: permanence is the whole point.

`uv run python -m scripts.research_sync.verify_completeness` checks store
coverage as its own FAIL condition (a live artifact missing from the store is
research one clock-tick from unrecoverable), separately from archive coverage.
Read the two signals differently: most artifacts legitimately hold no research at
all: 632 of the 859 carry only `run_logs/`, which is why the archive holds
artifact records from 227 runs rather than 859.

## Reading run logs

Each run tees to `run_logs/run_<run_id>_<timestamp>.log`, uploaded as a workflow
artifact (`research-<run_id>` for every bot workflow; the two test
workflows used `logs-<run_id>` before 2026-08-03, and those older artifacts are
still harvested: `RUN_LOG_ARTIFACT_PREFIXES` covers both names). Grep these for
the telemetry markers:

- `EXTRACTION_RUNG: question=... model=... qtype=... rung=... block_present=...`:
  one line per forecast value extraction. Watch for `rung=llm` (LLM salvage
  fired) and `block_present=false` (a forecaster stopped emitting a well-formed
  structured block). Emitted by `_log_extraction` in `value_extraction.py`.
- `MEMBER_FORECAST: question=... model=... role=member|stacker qtype=... raw=... published=...`:
  one line per forecast VALUE that leaves a runner, for every ensemble member and
  for the stacker. `raw` is what the extraction ladder read off the rationale before
  any clamp, renormalise or sanitise; `published` is what the runner handed on.
  Both are whitespace-free JSON literals, so `json.loads` them whatever the type:
  binary a probability each (`raw=0.005 published=0.02`), multiple choice the
  option-probability vector in `question.options` order (`raw=[0.9,0.005,0.095]`),
  numeric the declared `[percentile, value]` pairs with the percentile as the
  block's decimal (`raw=[[0.025,9.2],...]`) and `published` the post-sanitise list.
  The numeric line precedes the unit-mismatch guard, so a withheld member still
  leaves its raw declaration (the drop is in `FORECASTER_DROPS`). Emitted by
  `forecaster_runners.py` (members), `stacking.py` (stacker binary / MC) and
  `aggregation_pipeline.py` (stacker numeric, where its percentiles are sanitised);
  formatter in `member_forecast.py`. Added 2026-09-02 because no marker carried a
  member's value on every question and the published comment, the only other
  writer, is middle-trimmed and carries the block only since 2026-05.
- `OPEN_BOUND_PILING: question=... model=... bound=... bin_mass=... ...`: a
  forecaster put enough mass on the terminal displayed bin of an open-bound
  numeric question, without declaring any percentile beyond the edge, to trip
  `OPEN_BOUND_PILING_THRESHOLD` (`numeric/config.py`). Emitted by
  `numeric/diagnostics.py`.
- `EXTREME_CALL: question=... model=... p=... side=low|high lone=... survivors=...`:
  one line per surviving ensemble member of a BINARY question whose probability
  sat at or past an edge of the extreme band (`EXTREME_CALL_LOW` /
  `EXTREME_CALL_HIGH` in `constants.py`, currently 0.05 / 0.95, inclusive).
  Emitted by `extreme_call.py` right after `FORECASTERS_SURVIVED`, which supplies
  the denominator: a member inside the band leaves no line, so a rate needs the
  survivor list too. `lone=true` means no other survivor was extreme on the same
  side, which is the measurement: the 2026-08-31 round found lone extremes right 4
  of 9 against 21 of 23 for accompanied ones (those counts used a looser
  either-side rule, and `extreme_call.py` explains the difference before you pool old
  and new numbers). `survivors=1` marks a record where "lone" is vacuous because
  that member was the whole ensemble; drop those from a lone rate. This band
  membership check gates and clamps nothing; the single-survivor publish clamp
  below is a separate rule, keyed on the survivor count, that reuses the same two
  constants.
- `THIN_PUBLISH_FLOOR: question=... raw=... clamped=... survivors=1`: a WARN
  that a BINARY question published on exactly ONE surviving forecaster had its
  published probability clamped into `[THIN_PUBLISH_BINARY_FLOOR,
  THIN_PUBLISH_BINARY_CEIL]`, which `constants.py` defines by aliasing
  `EXTREME_CALL_LOW` / `EXTREME_CALL_HIGH` (currently 0.05 / 0.95), so the band is
  one definition rather than a second pair of literals. That range is narrower
  than the per-model `[BINARY_PROB_MIN, BINARY_PROB_MAX]` = [0.02, 0.98] clamp
  every member already passed: median-of-1 supplies no variance reduction, so the
  published value's admissible range is narrowed in exactly that state to price
  the missing aggregation. `raw` is what the survivor declared and is what the
  comment's per-model summary bullet still shows; `clamped` is what went to
  Metaculus. Emitted by `aggregation_pipeline.py` at the base-combine step, only
  when the value actually moved, so the line count is the floor's incidence: a
  lone survivor already inside the band leaves no line, and a multi-member median
  is never floored, however extreme (the receipt behind the rule priced that
  global variant at -52.02 spot peer). Expect it only alongside a
  `FORECASTERS_SURVIVED: ... survived=1/N` line for the same question.
- `GAP_FILL_V2: model=... steps=... tool_calls=... searches=... fetches=...
  rendered=... reads=... dup_tool_calls=... deadline_hit=... concluded_early=...
  wall_s=... findings=... pending_leads=... lint_rejections=...
  provenance_rejections=... quote_mismatch_warnings=... plan_gaps=...
  plan_skipped=... conclude_gate_rejections=... error=...`: one summary line per
  gap-fill v2 loop, emitted by `_log_completion` in `research/agentic/loop.py`.
  `error=` is what separates a step-zero crash from an idle run; both otherwise
  emit `steps=0 tool_calls=0 findings=0`. Companion `GHOST_PRE` /
  `GHOST_PRE_JSON` and `GHOST_FORECAST` / `GHOST_FORECAST_JSON` lines log the
  loop's pre- and post-research private forecasts for telemetry only; neither is
  ever published. `docs/agentic_gap_fill.md` reads the fields in full.
- `AGENTIC_FETCH_THROTTLED: url=... method=... chars=... phrase=...`: a WARN, one per
  gap-fill v2 fetch whose HTTP 200 body was the host's rate-limit interstitial rather than
  the page (a body at or under `FETCH_THROTTLE_PAGE_MAX_CHARS` carrying one of
  `FETCH_THROTTLE_PHRASES`, both in `research/agentic/fetch_outcomes.py`). Such a fetch
  returns `status=throttled`, earns no verification tier, and is never cached, so the
  driver's retry is a real request. `phrase` names the rule that fired and `chars` the body
  it fired on: together they say whether a line is a true throttle or the rule over-reaching,
  which is what the phrase list and the cap get retuned on. Emitted by
  `_throttled_fetch_outcome` in `research/agentic/tools.py`; harvested as
  `agentic_fetch_throttled`. Receipt: q45191, where two throttled ogimet.com fetches reached
  the driver as successful ones and the driver's own retry was served the cached refusal.
- `AGENTIC_FETCH_LOCAL_DOC: url=... method=pdf_local|digest_local chars=... pages=...
  passages=...`: an INFO, one per document the gap-fill v2 ladder read without paying a
  Gemini `url_context` call for it. `passages=0` on a `digest_local` is the reading that
  matters (the document does not discuss what was asked, which in the block itself reads
  exactly like a successful read), and `pages` is `n/a` for a page with no page structure.
  The line fires only where text was actually served, so its absence measures nothing: a
  refused digest leaves no line and the paid read that followed shows up only in the spend.
  `docs/agentic_gap_fill.md` defines the two methods and the `chars` convention. Emitted by
  `log_local_document_read` in `research/agentic/local_document.py`; harvested as
  `agentic_fetch_local_doc`. This is how
  the local-first rung is measured at all: before it every PDF the driver met went to a paid
  reader, 191 calls over the 2026 summer season, and the only trace of one was the spend.
- `RESOLUTION_SOURCE_FETCH: question=... url=... status=... http=... embeds=... [reason=...]
  [route=...]`: one line per URL the resolution-source provider fetched, emitted by
  `_log_fetch_outcome_markers` in `research/resolution_source.py`. `status` is `ok`
  for a success and the verbatim `FetchStatus` otherwise (`blocked`, `js_wall`,
  `no_resolving_content`, `stale_data`, `ungrounded`, ...). Since the escalation ladder it may
  be a RUNG's verdict rather than the direct fetch's: the Wayback rung's `stale_data` where the
  direct fetch said `blocked` / `error` / `not_found`, the paid reader's `ungrounded` where it
  said `blocked` / `js_wall` / `error` / `no_resolving_content`. An era-bucketed `blocked` rate
  off this field alone shows a drop at that merge that is bookkeeping, not hosts refusing us
  less; the direct outcome is `from_status` on the sibling `RESOLUTION_SOURCE_ESCALATION` line,
  and `route` partitions the two populations. `http` is `n/a` when no response ever
  arrived; `embeds` names the routeless data-embed providers (Infogram / Flourish /
  Tableau) found in the page's raw HTML, which is what makes an unreadable-embed
  page queryable even when its prose made the fetch a legitimate `ok`. `reason` is
  appended only where the status alone is ambiguous: `no_resolving_content` is
  `embed_shell` when the page named such a provider, `thin_page` when the extraction was
  simply under the chrome floor (the population the floor gained on 2026-09-02 when it
  stopped being gated on a named provider), and `no_matching_passage` when a cited
  document read in full discusses nothing the question asks about;
  `unreadable_document` splits into `no_text_layer` / `encrypted` / `malformed`, and
  `unsupported_type` carries `budget_skipped` / `parse_contention` when it was a document
  we held and declined to parse. Its absence means no reason applies, on a fresh line as much as on an
  archived one. `route` names which rung of the escalation ladder produced the recorded
  outcome: `direct` for the plain fetch, and `meta_refresh`, `impersonate`, `pdf_local`,
  `derived_api`, `rendered`, `wayback` or `url_context` for an escalated one (`impersonate`
  is the TLS-impersonating retry of a 403, live since 2026-09-04 behind the default-on
  `RESOLUTION_SOURCE_IMPERSONATE_ENABLED`; an impersonated PDF reads `pdf_local`, since the
  local read is what produced the text). Without it
  a rescued page reads exactly like one the direct route managed on its own, so "what
  did the ladder actually buy" would not be a query. Three more optional keyed fields carry
  failure diagnostics on a non-success fetch, so the archive can separate an egress-reputation
  refusal from a host fault (the archived Akamai 403s reproduce only from the GitHub runner
  IP): `failure_class` is a small token vocabulary (`http_403`, `http_4xx`, `http_5xx` off the
  response, or `tls`, `dns`, `timeout`, `connection`, `decode`, `malformed_response` off the
  transport exception), `exc` is that exception's class name, and `server` is the `Server`
  response header lower-cased with internal spaces collapsed to `_` (the strongest tell of which
  CDN refused us). `malformed_response` is the one our own client raised rather than the host: a
  `ClientResponseError` on the fetch path, which aiohttp uses for a response it will not accept at
  all (a `Content-Encoding` it cannot decode, a header over the byte cap, a bad status line), and
  which used to fall into the catch-all `connection` bucket alongside genuine connect failures.
  All optional fields are keyed and sit at the end of the line in a fixed order
  (`reason`, `route`, `failure_class`, `exc`, `server`), so a line carrying a later field but
  not an earlier one parses correctly and every archived line still parses byte-identically.
  Tier-2 Datawrapper dataset hops ride the same line and are identifiable by their url
  (`static.dwcdn.net/data/<chart_id>.csv`). This replaced the older free-text
  `resolution_source fetched <netloc> (<status>)` lines rather than joining them, so
  each fetch appears exactly once; the remaining free-text lines are REASON lines (a
  decode score, an unread content-type, an SSRF rejection) carrying what the marker
  cannot.
- `RESOLUTION_SOURCE_ESCALATION: question=... url=... from_status=... rung=... outcome=...
  wall_s=...`: one line per escalated rung attempt, emitted by
  `research/resolution_source.py` when the direct fetch could not read a page and a
  heavier route was tried. `from_status` is the verbatim `FetchStatus` that triggered
  the escalation, and its domain is per rung rather than shared, because each rung's trigger set
  is: `js_wall` or `no_resolving_content` for `meta_refresh`, `derived_api` and `rendered`, the
  200s that carried nothing readable; `unsupported_type` for `pdf_local`, a body we held and had
  not parsed; `blocked`, `error` or `not_found` for `wayback`, whose whole point is a page our
  address never reached; and `blocked`, `error`, `js_wall` or `no_resolving_content` for
  `url_context`, the only rung that draws from both families (`not_found` is not in its set,
  because a 404 or 410 has no page for a third-party fetcher to read, and a
  `no_resolving_content` whose reason is `no_matching_passage` is excluded on the reason). A pair
  outside that table (a `blocked` render, say) is a defect rather than a rare case. `rung` is the route tried. `outcome`
  and `wall_s` are that RUNG's own, stamped as
  the dispatcher closes it: `outcome` is the status that stood once the rung was over (its
  rescue, its own verdict such as `stale_data` or `ungrounded`, or the direct status it left
  standing when it declined) and `wall_s` is what that rung alone cost. Two exclusions in `wall_s`
  are worth knowing before it is read as latency: the `url_context` line excludes the free robots
  pre-check that runs in front of the paid read, and the `pdf_local` line excludes time the
  document spent queued for a parse slot. A URL with several
  lines therefore reads as a sequence, and on a page where a dead feed GET was followed by a
  rescuing render the first line carries the direct status and the second carries `success`,
  with neither billed for the other's latency. One combination reads oddly until you know the
  order it comes from: after the Wayback rung withheld a capture as `stale_data`, a paid attempt
  that then declines records `outcome=<the direct fetch's status>` rather than the withhold, while
  the `RESOLUTION_SOURCE_FETCH` line for that URL keeps `route=wayback` and the `stale_data` verdict
  that replaced the direct result. The two lines disagree by design, so take the page's own outcome
  from the FETCH line or from `from_status`, never from the last escalation line. Live since the
  paid flag went on in every bot workflow. The `RESOLUTION_SOURCE_FETCH` line above records
  only the FINAL outcome per URL, so on its own it cannot say how many rungs were spent or
  which one rescued the page; this marker is where a rung that fires often and rescues
  nothing becomes distinguishable from one that never fires, and where the latency case
  for keeping a rung on a question under a close-derived time budget gets made. A rung that
  never RAN (no wall budget, no browser, the per-question snapshot cap, the robots pre-check)
  emits no line here by design and is counted in the provider's `details["counts"]` instead;
  `docs/research.md` lists every count key.
  Harvested as `resolution_source_escalation`.
- `RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url=... host=...`, an INFO, one per paid
  `url_context` read the resolution-source ladder skipped because the host's robots.txt disallows
  `Google-Extended` (`research/resolution_source.py`; the same pre-check and per-host cache as
  `AGENTIC_URLCONTEXT_ROBOTS_SKIP` below, through `research/robots_policy.py`). A fire is a paid
  call NOT billed, so it is not a failure; the rate against the handful of hosts publishing the
  directive is what says whether the group parser is over-matching. No question id: the rung runs
  per cited URL inside its provider, so a join goes through the run id. Registered on 2026-09-04
  together with the flag flip that put `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` on in every bot
  workflow, so no archived run from before that merge carries one. Harvested as
  `resolution_source_urlcontext_robots_skip`.
- `RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=... statuses=...`, a WARN, one per
  paid `url_context` read on the ladder that came back with zero successful retrievals and was
  discarded as `ungrounded` rather than rendered under the primary-grading-evidence caption: the
  same floor `GEMINI_UNGROUNDED_SUPPRESSED` and `AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` apply, so
  the three suppression rates read as one family. The read WAS billed, so each record is money
  spent on nothing served. `statuses` is the comma-joined list of `url_retrieval_status` values
  the SDK reported, or `none` (harvested as null) when it attached no entry at all, which splits a
  retrieval that failed for a nameable reason from one that never happened. Same registration
  date and no question id, as above. Harvested as
  `resolution_source_urlcontext_ungrounded_suppressed`.
- `RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url=... host=...`, a WARN, one per paid
  `url_context` read that retrieved the page but answered with the prompt's `NOT_ADDRESSED`
  sentinel, the model's designed reply when the page does not discuss the ask, so the read was
  withheld as `no_resolving_content` / `not_addressed` instead of rendered as prose standing in
  for an absent section. Distinct from the ungrounded line: the page WAS retrieved, so Gemini
  reaches the host, and the money bought a true negative. `host` because the rollout question is
  which hosts Gemini reaches but finds nothing on. Same registration date and no question id.
  Harvested as `resolution_source_urlcontext_not_addressed`.
- `url_context not_addressed reply for <host>: <first 300 chars>` and its `ungrounded`
  twin are an INFO line each, emitted immediately after the two markers above and
  deliberately NOT registered, so nothing archives them and the markers' own line shapes
  stay the contract. They carry the head of the reply the withhold discarded,
  whitespace-collapsed and capped at `RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS`. Read
  them when a withhold needs explaining: a `not_addressed` verdict meaning the page truly
  does not discuss the ask and one meaning the model summarized the bot-challenge page the
  host served it instead are otherwise indistinguishable, and the reply is what tells them
  apart. The `ungrounded` line appears only when that read said something, since the same
  branch also fires on an empty reply.
- `RENDERED_FETCH_OFF_HOST: scope=<resolution_source|gap_fill_v2> pinned_host=<host>
  landed_host=<host> same_publisher=<true|false>`, a WARNING, one per headless-Chromium render whose main frame ended up on a
  host other than the one the transport pinned at launch, emitted by the shared transport
  `research/rendered_fetch.py` and so fired for either caller, which is what `scope` names. The
  DOM was refused unread on the transport's pre-read check, or discarded unpublished when the
  navigation committed during the read itself, so nothing from that render is published either way
  and the direct fetch's result stands. Hostnames only, never the landing URL, which can carry a
  session token. `same_publisher` is `true` when the landing host shares the pinned host's registrable domain (a benign client-side hop such as `example.com` to `www.example.com`, refused by strict hostname equality and priced by this value) and `false` otherwise, including every landing with no hostname, so a `false` record is the security signal and a `true` record prices the strictness.
  This is the ONLY per-event record of an off-host landing, because a refused render is a skip and
  a skip emits no `RESOLUTION_SOURCE_ESCALATION` line; the per-question rate is
  `render_off_host_skips` in the resolution-source provider's `details["counts"]`. Registered on
  2026-09-04 with the check itself, so no archived run from before that merge carries one, and a
  local probe of 22 real render targets on that date produced zero of them. Harvested as
  `rendered_fetch_off_host`.
- `AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=... [statuses=...]`: a WARN, one per
  gap-fill v2 `read_document` call whose `url_context` retrieval brought back nothing,
  so the answer would have been unsourced recall and the `fetched` verification tier is
  withheld (`research/agentic/tools.py`). Worth watching because a `fetched` document
  discrepancy is the only kind that enters the findings artifact's SUPERSEDE block, the
  one that tells every forecaster to override the briefing. `statuses` is the
  comma-joined list of url_context retrieval statuses the SDK reported for that call, or
  `none` when it reported none at all, which splits a retrieval that was attempted and
  failed for a nameable reason from one that never happened. Both `none` and an absent
  field harvest as null, so an archived pre-field line reads the same way. Harvested as
  `agentic_document_ungrounded_suppressed`.
- `AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=... host=...`: an INFO, one per paid `url_context`
  read skipped because the host's robots.txt disallows `Google-Extended`, the product token
  Gemini's retrieval obeys, so the read would have been spend with a known-zero return
  (`research/agentic/tools.py`; the group parser is `research/robots_policy.py`, moved out of
  `research/agentic/` on 2026-09-03 when the resolution-source ladder became its second caller
  and now sharing one per-host cache between them).
  Non-alertable: a fire is a paid call NOT billed, and the free fetch rungs are unaffected by
  the check. Harvested as `agentic_urlcontext_robots_skip`. `docs/agentic_gap_fill.md` covers
  the group parser and what a high rate would mean.
- `FINANCIAL_NOISE_FLAG: surface=financial_data|ts_anchor symbol=... vr_lag=... vr=...
  floor=... short_vol=... long_vol=... robust_vol=...`: the series behind a rendered
  volatility is noise-dominated: its variance ratio sits below
  `FINANCIAL_VARIANCE_RATIO_FLOOR`, meaning most of each day's move is reversed the
  next, which inflates any volatility computed from one-day returns. The flagged
  block leads with `robust_vol`, measured on overlapping `vr_lag`-step returns, and
  labels the short-window figure noise-suspect. Two surfaces log it, sharing the
  screen and the line itself (`research/noise_flag.py`): `financial_data.py`'s
  `_volatility_lines` and `ts_render.py`'s `_realized_vol_lines`. Only the
  financial-data surface computes a long-horizon window, so a `ts_anchor` record
  reads `long_vol` as null rather than zero. `surface` is what tells that apart
  from a yfinance series too short to hold one. Per-identifier, so
  one question can fire several and the line carries no question id. `symbol` (the
  ticker or FRED series id, same field position as `FINANCIAL_STALE_LATEST`) is what
  tells two flagged identifiers in one run apart and joins a noise-flag record to the
  stale-latest record for the same series. Informational and NOT alertable: it
  describes the vendor's data, not a bot defect.
- `GEMINI_USAGE: role=grounded_search|read_document|resolution_source model=... prompt_tokens=...
  tool_use_prompt_tokens=... candidates_tokens=... thoughts_tokens=... total_tokens=...
  search_queries=... [question=...]`: one line per response from the paths that call
  Google natively rather than through OpenRouter, so their spend on the operator's personal
  AI Studio key is readable from a run log. Emitted by `log_gemini_usage`
  (`research/gemini_usage.py`), called from `gemini_search.py` (`grounded_search`, before
  any formatting branch, so an ungrounded-and-suppressed response still records what it
  cost), `research/agentic/tool_backends.py` (`read_document`, which carries no question
  id, hence the trailing field's absence there), and the resolution-source ladder's paid
  url_context rung (`resolution_source`, which likewise carries no question id and appears
  only from runs with `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` on: every bot workflow since
  2026-09-04, so an archived run from before that merge has no rows in this role). No surface
  here bills through OpenRouter,
  so none shows up in `CREDIT_ROLE_SPEND`, and before this marker the whole Google AI
  Studio side of a run's spend was invisible to the archive. That side is metered against a
  monthly grounded-prompt allowance per project and billed per QUERY on overage, which makes
  `search_queries` the billable unit and any feature that multiplies grounded calls a re-run
  of the spring-2026 billing arc. `model` is the response's own `model_version` where it
  reported one and the configured id otherwise. Any count Google
  did not report reads `n/a` rather than 0, since `thoughts_tokens=0` is a real reading;
  `search_queries` is the exception and reads a genuine 0 when the search tool issued none
  (an absent `web_search_queries` list IS a count of none), `n/a` only when the grounding
  metadata could not be walked, so separate the two surfaces on `role`, never on this
  field. **The ledger covers COMPLETED responses only.** `log_gemini_usage` runs after the
  SDK returns, so a call that timed out or raised billed unknown tokens and emitted no row:
  14 of 154 archived `read_document` calls (9.1%) hit that handler. A spend total from these
  rows is a LOWER bound, biased toward undercounting the largest calls; the denominator is
  `provider_results['gemini_search'].status` per question plus `research_provider_failures`,
  never this marker's row count. `thoughts_tokens` is the field worth watching: 71% of
  grounded-search output tokens were thinking before the
  explicit levels (`GEMINI_SEARCH_THINKING_LEVEL`, `GAP_FILL_V2_READER_THINKING_LEVEL`) were
  set. Nothing about this is alertable; it is spend accounting, not degradation. Harvested as
  `gemini_usage`.
- `CREDIT_BALANCE` / `CREDIT_SPEND` / `CREDIT_ROLE_SPEND` / `CREDIT_FLOOR_BREACH`:
  credit telemetry, described above. `CREDIT_FLOOR_BREACH` fires whatever the
  credit-alert window says, so a breach on a GREEN run means a suppression window
  is open (none is, since 2026-09-03); the adjacent INFO line names the resume
  date.
  `CREDIT_ROLE_SPEND` is the per-(role, key) decomposition of the run's
  OpenRouter spend; a run with no completions logs a single no-completions line
  under the same token instead of rows.
- `TIME_BUDGET: question=... budget_s=... close_time=... close_limited=...
  fast_path=...`: one line per question, emitted by `time_budget.py` before any
  research runs. Emitted even on roomy questions on purpose: `CLOSE_MARGIN` fires
  only after a SUCCESSFUL submission, so it is censored on exactly the thin-window
  questions this budget exists for. `close_limited=true` means the question's own
  close time, not the static `PER_QUESTION_WALL_CLOCK_DEADLINE`, set the budget.
  `fast_path=true` means it fell below `TIME_BUDGET_FAST_PATH_THRESHOLD`, so the
  optional research stages were dropped to protect the prediction POST. Companion
  `TIME_BUDGET_FAST_PATH` and `GAP_FILL_SKIPPED_FOR_BUDGET` WARNs say so too, and
  `RESEARCH_PHASE_DEADLINE` names any provider cancelled at the phase deadline.
  A question with no publishable budget at all (close already passed, or so near
  that the prediction POST cannot fit) is skipped before any spend and bumps
  `questions_failed_to_publish`.
- `QUESTION_CAP_FORFEIT: platform=<metaculus|mantic> cap=<n> total=<n> dropped=<n>
  posts=<ids>`: one WARNING per run, from `forecast_questions` (`forecaster.py`), when
  more questions are open than `max_questions_per_run` allows. Questions are sorted
  tightest close first before the cap, so the posts named are the latest-closing ones
  left behind; on Mantic, which opens an hour's batch at once, each is a real forfeit,
  and the marker is registered so the loss outlives the 90-day log expiry.
- `SKIP_GUARD_UNREADABLE: question=<id> post_id=<id> platform=<metaculus|mantic>
  reason=my_forecasts_missing`: one WARNING per question, from `forecast_questions`
  (`forecaster.py`), when the skip-previously-forecasted guard could not read the
  `my_forecasts` field it derives "already forecast" from, so the question was dropped
  before any spend rather than treated as new; a count line follows. It means the list
  read lost `with_cp=true` or the platform token, or the API changed shape. Nothing was
  double-forecast, and the next firing picks the question up once the field reads again.
  What to check is under "Scheduling reliability" above.
- `Degradation counters: forecasters_dropped=..., questions_failed_to_publish=...,
  stacker_primary_failed=..., stacker_fallback_used=...,
  stacker_fallback_failed=..., research_provider_failures=...,
  summarizer_failures=..., gap_fill_v1_errors=..., gap_fill_v2_errors=...,
  prediction_market_degraded=..., prediction_market_source_losses=...,
  provider_degradation=..., publish_attempt_failures=...,
  publish_skipped_closed=..., time_budget_fast_path=...`: the
  end-of-run summary from `forecaster.py`'s `forecast_questions`, and the line
  that decides CI color: these are exactly the counters `alertable_count` sums, so
  any one of them non-zero exits the run non-zero.
  `time_budget_fast_path` is the earliest-firing member of the publish-side family:
  the other three fire once a publish has already failed or been withheld, while
  this one fires while the question is still savable and says latency is closing in
  on a close deadline.
  `research_provider_failures` counts any provider exception, not only timeouts:
  it was named `research_provider_timeouts` until 2026-07-26, when
  `prediction_market_platform_failures` also became
  `prediction_market_source_losses`. `scripts/telemetry/markers.py` matches both
  spellings, so archived pre-rename logs still harvest.
  `prediction_market_degraded` kept its name when the counter behind it moved off
  the retired Kalshi `/series` index onto the full events-catalogue pull, so the
  field name is stable across that change while what it guards got strictly more
  load-bearing: the catalogue feeds both the settlement-source join and the fuzzy
  channel. Note that a lost catalogue pull bumps BOTH this counter and
  `prediction_market_source_losses`, so one outage adds 2 to `alertable_count`;
  that is deliberate over-counting (the two carry different marker fields) and not
  two separate failures. `prediction_market_source_losses` is alertable by operator
  decision: any prediction-market source losing a fetch reddens CI.

**One analysis hazard from ranked market retrieval, worth knowing before you diff
`providers_used` across eras.** The ranker may legitimately return zero rows, in
which case the provider renders nothing and the `## Prediction Market Snapshot`
header never appears. An ARTIFACT record still lists the provider under
`providers_attempted` (it ran, it just had nothing to say), but a COMMENT- or
LOG-backfilled record reconstructs `providers_used` by scanning for that header,
so the provider simply vanishes from it. So a drop in prediction-market presence
across backfilled records can mean "the ranker declined" rather than "the provider
broke", and the two are only distinguishable from an artifact record or from the
`MARKET_RANKING:` line's `outcome=` field. No code change: the header-scan
reconstruction is lossy by construction and always was.

`outcome=` alone does not say WHY a question fell back, so read the sibling
`MARKET_RANKING_DEGRADED:` line beside it: `reason=shape_regression` means a
well-formed but non-empty ranking array yielded no usable row (a renamed index key,
or every index outside the pool), i.e. OUR prompt/parser contract broke, and before
2026-08-25 that case was reported as `ok(0)` and rendered the deliberate-empty
sentence ("prediction markets were retrieved and reviewed… none was judged to bear on
it") to forecasters. `reason=unreadable` means the completion was not a ranking array
at all. Both are harvested as `market_ranking_degraded`, so the split survives the
90-day GHA log expiry; a `MARKET_RANKING` line with `outcome=failopen` and no
degraded sibling in the archive predates this marker.

A third market line, `MARKET_TIER_CAPPED: question=... rows=... capped=venue@rank`,
fires only when the deterministic staleness pass refuses a row the top relation tier:
the ranker graded a market that stopped trading more than
`MARKET_STALENESS_TIER_CAP_DAYS` (60, in `market_retrieval/ranking.py`) before the
question opened as `same_quantity_same_date`. The row keeps its rank, price and
liquidity cells; what it gains is a note in the `why` cell stating the demotion and
its arithmetic (`demoted from same-date: closed 162d before the question opened`).
Silence is the normal case, and the cap fires on nothing in the 102 archived
snapshots, so a first line in a run log IS the finding. The demotion also rides the
archived snapshot as `MarketMatch.tier_cap_note`, so its incidence is answerable
offline; this line is the prod-log half and the one that survives a run whose
snapshot the research archive never captured.

A run can also exit non-zero for degradation alerts (the counters above,
personal-key fallbacks, or the model-deprecation tripwire) even when every
question that met the minimum-forecaster threshold was published. The non-zero
exit is the CI red-check signal to investigate; it does not mean publishing
failed. Credit-caused shortfalls alert again as of 2026-09-03, and are exempt only
inside a suppression window (see that section above); every other cause always alerts. The
exact conditions are in the next subsection, "The end-of-run breakdown and the exit ladder".

### The end-of-run breakdown and the exit ladder (`_report_degradation_and_exit`)

`cli._report_degradation_and_exit` emits the one-line degradation breakdown, harvested as
`run_alertable_summary`, and then decides the process exit status. It runs after
`forecast_on_tournament` or `forecast_questions` has finished publishing, and every non-zero
exit path in the run lives in this one function.

**The arithmetic** is
`alertable = bot_alertable + generic_fallback - suppressed_credit_fallback + mantic_post_drops`.

- `bot_alertable` is `TemplateForecaster.alertable_count`, the sum of the degradation
  counters enumerated under "Reading run logs" above.
- `generic_fallback` counts donated-to-personal key fallbacks. They are counted in
  `fallback_openrouter.py` at the wrapper level, process-global, because the wrapper has no
  link back to the bot. Each fallback was successful, in that the run completed on the paid
  personal key, and it is still alertable because a call that should have hit the free
  donated key billed the operator instead. The count covers ALL fallback causes (401, 402,
  429, the guardrail or data-policy block, and the 404 "no allowed providers" case).
  `donated_404` and `credit` are two disjoint subsets of that same total, broken out for
  diagnostics only: just the all-causes total is added to `alertable`, because adding a
  subset as well would double-count events already inside it.
- `suppressed_credit_fallback` is the credit subset, subtracted back out only while credit
  alerting is suppressed (see "The credit-alert suppression window" above), because inside
  such a window an empty donated key is an accepted state. Every other cause keeps its full
  weight, since 401, 404, 429 and the guardrail block each mean real breakage, and every
  event is still counted exactly once: the generic total adds it, and at most one subset
  subtracts it. The subset counts only the SUPPRESSIBLE credit case, a donated key that
  genuinely drained. A key that was revoked or re-capped to zero returns the same
  `Key limit exceeded` text but is classified separately by
  `fallback_openrouter.is_suppressible_credit_error`, which probes `/auth/key`, so it stays
  inside the generic total and keeps the run red.
- `mantic_post_drops` counts posts the Mantic client could not parse (see "Parse drops are
  counted" above). The framework's per-post loop swallows the error as a warning, so this
  process-global counter is the only thing that turns a forfeited post into a red run.

**Two fields render conditionally**, so that a term appears on the line exactly when it
applies. `donated_key` is rendered only when a spend-cap failure actually made the wrapper
probe the donated key, because a rendered `unknown` would read as a failed probe rather than
as "no run of this shape ever needed one". `mantic_post_drops` is rendered only when it is
non-zero, which is when it explains a non-zero `alertable`. Both are optional groups in the
registry's `run_alertable_summary` regex (`scripts/telemetry/markers.py`).

**The line is emitted on every path**: degraded, suppressed-green, crashed, and fully clean.
The green paths need it as much as the red one. When every donated-key call fell back and
the credit subset cancels the whole generic total, `alertable` reads 0, which is the exact
shape of the 2026-07-26 drained-key run, so gating the line on the exit status would leave
that run's degradation and probe verdict entirely unrecorded.

A fully clean run says so explicitly, under a distinguishable "clean" phrase that harvests
as the same marker. It used to emit NOTHING, so that the line's presence would stay a signal
rather than boilerplate, and the operator overturned that on 2026-08-25. The reason: silence
is not distinguishable from a run that died before reaching this block, and once the donated
key is refilled (past `CREDIT_ALERT_RESUME_DATE`) the clean shape becomes the COMMON one, so
the archive's per-run census would lose exactly the runs that went well. During the
drained-key window the question was moot, because every run fell back at least once and 0 of
the 73 archived records are the clean shape. A raising `log_report_summary` is never clean no
matter what the counters read, since that run lost a question, and its counters can
legitimately be all-zero (q45085's shape), which is why the phrase rather than the fields is
what marks a run clean. The `run_clean` predicate therefore has to stay the exact complement
of every non-zero exit path in the function, including the two that run AFTER the line is
emitted, the credit-floor breach and the deprecation tripwire. Without those two terms a run
about to exit red could first stamp the archive's record with the clean token.

**Emit then raise.** `TemplateForecaster.log_report_summary` raises by design when any
report is an exception (`compact_log_report_summary` re-raises, so a failed question reddens
CI under `return_exceptions=True`). It used to sit ABOVE the alertable block, so the one run
that most needed a summary record left none: q45085's publish failure on 2026-08-03
propagated out of it, `alertable` was never computed, and that run is the single forecasting
run since 2026-07-26 with no `run_alertable_summary` line in the archive. Now the error is
held, deliberately without narrowing it to a class, the breakdown is emitted on that path
too, and the original exception is re-raised, so it keeps its traceback and CI stays exactly
as red as before. Re-raising rather than calling `sys.exit` is what preserves that traceback
in the log, and it takes precedence over the alertable exit because the exception is the
richer red signal.

**The ladder, in order.**

1. A held report-summary error is re-raised, after the breakdown line.
2. `alertable > 0` logs a WARNING and exits 1.
3. `generic_fallback > 0` with `alertable` at zero logs an INFO and stays green. That state
   is reachable only under suppression with every fallback credit-caused, since the
   subtraction cannot otherwise reach zero from a positive total, so the line states that
   rather than leaving a reader to derive it from the arithmetic.
4. A clean run logs the all-clear census line described above.
5. Anything else logs an INFO saying a post-summary check below decides the exit status. The
   counters are quiet there while a red condition further down may still fire, so no green
   claim may be made at that point.
6. A donated-key balance below the early-warning floor exits 1, or, inside a suppression
   window, logs an INFO saying the breach was observed and alerting is suppressed until the
   resume date. The run completed and published normally either way, and this exit is purely
   the ask-Metaculus-for-a-top-up signal; the INFO exists so that a reader who sees the
   `CREDIT_FLOOR_BREACH` warning beside a green run does not have to guess why.
7. The fall-cup configuration reminder exits 1. Its ERROR is logged at startup by
   `check_fall_cup_reminder` (`constants.py`) so the operator sees it before the run's noise,
   and it is checked in every run mode on purpose, so the cup and minibench crons and manual
   runs all keep reddening until `FALL_CUP_CONFIGURED` is flipped. Same shape as the
   credit-floor path: the run published normally, and the exit is purely the
   reminder-to-configure signal.
8. A stale Mantic slug exits 1 with the re-point-the-constants ERROR, because a zero-question
   run is otherwise green while a Series 2 slug goes unforecast (see "Stale slug goes red"
   above).
9. The post-submission deprecation tripwire runs LAST, so that submission has fully completed
   and every other alertable condition exits first with its own log line. When OpenRouter
   retires a model the bot uses, the canonical case being the 2026-05-15 deprecation of
   `x-ai/grok-4.1-fast`, which silently 404'd for about two days,
   `check_deprecation_alerts_and_exit` (`fallback_openrouter.py`) prints a loud banner and
   exits 1 so the Actions check turns red. It returns silently when no deprecation was
   observed.

The startup half of `cli.py`, the logging levels, the two hardening patches, the identity
preflight and the research-archive labels, is in `docs/architecture.md` "CLI startup
wiring".
