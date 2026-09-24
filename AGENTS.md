# Metaculus Forecasting Bot: Agent Guidelines

Repo-specific context only; general coding style lives in the operator's private global config. A starting point, not
the manual: the cost gate, the overrides, the layout, the pipeline, and the standing rules whose violation is silent and
expensive. Depth is in `docs/`, indexed by `docs/README.md`.

## Cost gate: the operator approves every credit spend

Anything hitting live LLM or research APIs spends the operator's own money, and live bot modes also publish public
comments to Metaculus or Mantic. **Never launch one autonomously. Ask first, every time**, even after clean gates and a
clean `/forge`. The gate is on the SPEND, not the mechanism, so a local command, an Actions dispatch, an edit adding a
cron firing and a script wrapping any of those are one rule, with no clean-gates exemption and no "small enough"
threshold. An explicit operator instruction IS the approval, per-run: it never carries forward to the next run or to a
re-run after further changes. Paid runs are a final pre-merge check, never part of a verification loop, because the free
gates are the loop and unit plus integration coverage is how an agent earns confidence. When verification needs a paid
run, surface the exact command and rough cost, then stop.

| Ask before running | Cost |
|---|---|
| `make run` / `python main.py` in any live mode (`test_questions`, `tournament`, `metaculus_cup`, `minibench`, `mantic`) | credits AND publishes |
| `make run_mantic`, `make run_mantic_one POST=<id>` (`--mode mantic`, `--only-posts`) | ~$3/question, publishes to Mantic Crucible, PERSONAL keys only: needs `DONATED_OPENROUTER_KEY_ENABLED=false`, fails shut without it |
| `make backtest_smoke_test` / `_small` / `_medium` / `_large` | every forecaster and research call plus the leakage screen; no publish |
| `make backtest_with_cache` | research replayed from `--research-dir`, forecaster spend still real |
| `make ablation_qa_research` / `_smoke` / `_small` / `_medium` | real research and forecaster spend (`ablation_score` is the free exception) |
| `make benchmark_run_*` | deprecated, still fans the real ensemble over real questions |
| `make test_live` | the only suite leaving the network; a `:free` slug, so cents, but real calls needing a key |
| `make probe_resolver QUESTION=<id>` | resolver grid, personal key; refuses without `ARGS="--i-accept-spend"` |
| `make strip_bench` | ~$1.25; `ARGS="--dry-run"` is the free view |
| `uv run python scripts/probes/gemini_verify.py --i-accept-spend` | three live google-genai calls, personal AI Studio key; refuses without the flag |
| Any bot workflow run or dispatch (`run_bot_on_*.yaml`, `test_bot*.yaml`); any edit to a `schedule:` block or to a research or model flag that adds runs or raises per-run cost | spends and publishes exactly as a local run |
| `make cronjob_dispatch_setup ARGS="--apply"` (and `--enable-mantic`) | every firing it adds is a paid, publishing bot run |
| `fetch_diagnostic.yaml` dispatch | cannot spend or publish by construction, but burns Actions minutes and probes federal hosts from the runner IP |
| Any one-off script reaching a research provider or the ensemble | same rule |

Inside a paid run the resolution-source ladder is itself paid: `page_digest_extractor` and each gap-fill `read_document`
make their own OpenRouter calls, and the Gemini `url_context` rung spends the personal `GOOGLE_API_KEY` up to
`RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS` per question. Two enablement facts: `run_bot_on_minibench.yaml` is
`disabled_manually` by operator design and has never been enabled, so minibench questions with zero bot forecasts are
expected and not worth raising, and the Metaculus Cup workflow is enabled for fall 2026. `gh` needs
`--repo No-Stream/metaculus-bot`. Detail: `docs/operations.md`.

**Free, run freely.** `make test`, `test_fast`, `test_e2e`, `lint`, `format`, `typecheck`, `typecheck_ty`, `cov`,
`audit`, `deps`, `lint_imports`, `precommit*`. The suite is self-contained: `e2e` means full pipeline with MOCKED LLMs,
and the autouse `_block_network_egress` fixture in `tests/conftest.py` raises on any non-loopback connect. Read-only
pulls and views: `make sync_all` and its parts, the `performance_analysis` package, `score_ghosts`,
`close_margin_watch`, `ablation_score`, `supply_probe`, `supply_probe_mantic`, `benchmark_display`, `dispatch_watch`, a
bare `cronjob_dispatch_setup`, `cost_report`, `check_credits`, and `scripts/probes/fetch_diagnostic.py`, which forces
the paid rung off before probing.

## Repo overrides

**Python 3.12+**; package manager **uv** (`uv.lock`; `uv sync --dev` installs editable so no `PYTHONPATH=.`; add deps
with `uv add` / `uv add --dev` and commit `pyproject.toml` + `uv.lock`; **never `pip` or `poetry`**, both blocked;
`exclude-newer = "1 week"`). Build backend `uv_build`, flat layout (`module-root = ""`). Formatter **Ruff** at
120 chars. **basedpyright** in standard mode must stay at 0 errors, `ty` secondary and advisory. **pytest +
pytest-asyncio**, self-contained, no keys in CI. Copy `.env.template` to `.env`; it is gitignored, CI keys are secrets.

## What this repo is, and where things live

A fork of the Metaculus starter template on the `forecasting-tools` framework. Per question it gathers research from
several providers in parallel, runs a small ensemble of frontier LLMs for independent forecasts, combines them, and
publishes the result as a comment. `--mode mantic` runs the same pipeline against Mantic's Crucible competition, a
Metaculus fork with the same API shape, through a swapped platform client. Aggregation defaults to
`CONDITIONAL_STACKING`, but **stacking is disabled in production, so prod publishes the MEDIAN of the raw forecasts.**

`main.py` is a shim re-exporting `TemplateForecaster` and invoking the CLI, `backtest.py` scores predictions against real
resolutions, and `community_benchmark.py` is DEPRECATED apart from `make benchmark_display`. Inside `metaculus_bot/`: the
per-question pipeline is `forecaster.py` and `cli.py`, research `research/`, numeric math `numeric/`, aggregation and
publish hardening `aggregation_pipeline.py` / `publish_hardening.py` / `publish_gate.py`, residual analysis
`performance_analysis/`, the Mantic seam `mantic.py` / `question_platform.py`. Sync, analysis and probe tooling is in
`scripts/`, where only `probes/fetch_diagnostic.py` is free to run. `docs/architecture.md` is the module map, and
`scratch_docs_and_planning/` holds dated plans, the live ones being the residual playbook, the fall 2026
preregistration and the fetch-ladder unification plan.

New outbound fetches go through the existing transports, never a hand-rolled client: `research/http_fetch.py`,
`research/fetch_ladder/`, `impersonated_fetch.py`, `rendered_fetch.py`, `url_context_reader.py`, `robots_policy.py`.

## Pipeline outline

Per question, in `forecaster.py:_research_and_make_predictions`. Stage by stage: `docs/architecture.md`.

0. **Time budget** from close time, before any spend: it can skip the question, take a fast path dropping the slow
   optional providers and both gap-fill passes, or cancel stragglers. Logs `TIME_BUDGET`.
1. **Research**: one primary provider by priority, the env-gated add-ons in parallel, then two gap-fill passes.
2. **Forecaster fan-out** under `FORECASTER_SOFT_DEADLINE`, values read out of a fenced STRUCTURED FORECAST block by the
   extraction ladder. A Mantic question on an enumerable grid (`elicit_per_bin`) is elicited PER BIN instead and
   bypasses the sanitizing, PCHIP repair and unit-mismatch guard.
3. **Min-forecasters guard** (`MIN_FORECASTERS_TO_PUBLISH`): at a floor of 1 a lone survivor publishes and routing
   short-circuits before spread, which needs two predictions. Logs `FORECASTERS_SURVIVED`.
4. **Aggregation** pointwise in CDF space; MEDIAN in prod since the spread gates are off, except that per-bin Mantic
   members pool by MEAN and Mantic's open tails are lifted toward 5%. `NUMERIC_AGGREGATE ... method=` records the rule.
5. **Publish behind a close-time gate** (`publish_gate.py`): a question past its window is skipped entirely, prediction
   and comment together, and the skip is alertable.

## Standing rules

Each has cost real work at least once. The group's pointer carries the reasoning and the receipts.

**Config and models** (`docs/roster_history.md`, `docs/constants.md`, `docs/operations.md`)

- Model ids live only in `llm_configs.py` and `constants.py`; `tests/test_model_name_locations.py` pins the files.
- Never state the roster from memory; "latest per vendor" resolves only from a live model-list read.
- A roster or pipeline-behaviour change is a config-era boundary: one merge, before the season's first question.
- Never restate `STANDARD_PERCENTILES`, its count or its label; derive from `numeric/config.py`.
- Constants go in `constants.py`, or `numeric/config.py` when grid-scoped, never inline in a function.
- Only `OAI_ANTH_OPENROUTER_KEY` is shared; every other key, `GOOGLE_API_KEY` included, is personal.

**Telemetry** (registry `scripts/telemetry/markers.py`, guide `docs/telemetry_markers.md`)

- Markers and status strings are data contracts keyed on exact spelling: ADD one, never rename or repurpose in place.
- A new signal without a marker spec is invisible after 90 days, when the Actions logs expire.
- A provider with nothing to say returns `""` plus a loss token, because any non-empty return flips the orchestrator to
  `ok` and defeats every downstream empty guard at once; a count in `details["counts"]` keeps "found none" apart from
  "never ran".

**Proportion and safety** (operator ruling, 2026-09-04)

- The medicine must be better than the disease. Fix what has cost or will cost forecasts: missed deadlines, wrong
  values, silent drops, spend. Refuse a remedy that adds a flag, branch, mechanism or parallel path for a case this
  deployment does not produce, even when the finding is correct. "Real but not worth it" is a normal verdict.
- A guard fails SHUT; wrapping the unit-mismatch guard in try/except once published the order-of-magnitude error the
  guard exists to block. Let it raise.
- The fetch transports already carry the timeout inside the wall, the byte cap, the redirect limit and per-host
  politeness. Their DNS pins, IP assertions, landing-host checks and WebSocket block were built for a threat this
  deployment cannot realise: keep them, never extend them, never cite them as a reason to add branches.
- Timing, deadline and fallback code gets strictly-safer changes only, since missed deadlines cost real forecasts this
  quarter. Anything not obviously safer stays put and goes in `FUTURE.md`.
- `prediction_market` and `resolution_source` return `""` under `is_benchmarking`; never soften a leakage guard.

**Prompts** (`docs/prompts.md`)

- A prompt rule survives only if the pipeline needs it, it scaffolds the model's reasoning, or it corrects a measured
  failure with no shorter form. State it once as a named constant.
- Every surviving rule has a presence pin and every removed one an absence pin in `tests/prompts/`, and no base-prompt
  rule may appear in the three stacking prompts.

**Analysis** (`docs/performance_analysis.md`; per-round procedure in the residual playbook)

- Run `make sync_all` first, which "residual analysis" always implies: a single-source pull silently drops what it did
  not fetch, and Actions artifacts expire at 90 days.
- Routine residual refreshes use the committed collectors, analysis tools and `RoundSpec` library API documented in
  the playbook and `docs/performance_analysis.md`. Do not write new scratch scripts, copy prior-round drivers, or
  recreate standard dimensions for a routine refresh without an agreed functionality change. `scratch/residual_<date>/`
  holds round inputs and outputs. There is no integrated multi-source round CLI today; report missing standard
  functionality and agree on a maintained addition before building it. Focused follow-up analyses may use scratch
  scripts.
- Era boundaries are merge-to-main committer timestamps, never authoring dates, which have produced a phantom era and a
  wrong presence rate (`performance_analysis/eras.py`).
- Era-bucket every calibration, aggregation or bias claim; three conclusions have flipped under it. A fitted
  calibration layer ships only after a decisive out-of-sample era test.
- Rank on SPOT PEER through `spot_peer_score()` / `ranking_score()` / `spot_peer_delta`, never `peer_score`.
- Exclusion cohorts live only in `performance_analysis/cohorts.py`, and they are QUESTION ids: use `id_mapping`.
- Never pool the `artifact`, `comment_backfill` and `log_backfill` record classes; read `source`, not `run_id`.
- A comment with fewer than N per-model bullets is ambiguous; never read a missing bullet as a model that declined.

**Code structure** (`docs/architecture.md` "Import conventions")

- A function-scoped import needs one of exactly three justifications named in its `# noqa: PLC0415`: a genuinely
  optional dependency, late binding for a patch surface, or a real circular import. No `noqa` or
  `HARNESS-SCAN-EXEMPT` marker may be added without one, and deleting one by fixing the import is always welcome.
- Patch a name where it is USED: every `Fred` / `fetch_series` target is `research.fred_rendering`, not
  `financial_data`, and hoisting a late-bound import binds the unpatched object and defeats the test.
- Fix all occurrences: if you found a bug pattern, grep for its siblings.

## Development, commits, API notes

`uv sync --dev` (or `make install`), then `uv run <cmd>`; the Makefile is authoritative for what exists.
`make precommit_install` adds the Ruff hooks on commit plus a `pytest-full-suite` pre-push hook, per-push because the
suite takes about 105 s. `make audit` is osv-scanner over `uv.lock`, blind to the libcurl and BoringSSL binaries
vendored inside the `curl_cffi` wheel, so a libcurl CVE surfaces only through a `curl_cffi` bump. For autonomous
stretches use the environment's goal tool (or `/loop`) to keep multi-phase work moving until the objective is complete,
and wait on running subagents rather than idling when no independent work is left.

**CI green is the gate, not a local green run.** A test depending on the developer's environment (an absolute path, the
checkout location, `$HOME`, gitignored local data) passes locally by construction, so CI is the first place it can fail,
and one such test has shipped. Assert a repo-relative suffix, or skip when the artifact is machine-specific. Then
`gh run list --repo No-Stream/metaculus-bot --branch <branch>`.

Commits take a concise imperative subject ("fix test cmd", "migrate to uv") and a short body when context helps. `main`
is ruleset-protected and a pre-commit hook refuses commits on it, so work on a branch. PRs: clear description, linked
issues, config and docs updates, logs or screenshots for behaviour changes, all checks passing, and no changes outside
workflow files unless CI behaviour is meant to change.

Metaculus API: Swagger UI at <https://www.metaculus.com/api/>; the backend, where the validation lives, is open at
<https://github.com/Metaculus/metaculus> (`questions/serializers/common.py`). Server-side `continuous_cdf` constraints
are in `docs/numeric_pipeline.md`. `/api/comments/?author=X` returns only the caller's own comments, so analysing other
bots' comments at scale needs a support exemption.
