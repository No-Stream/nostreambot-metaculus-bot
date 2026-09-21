.PHONY: install lock test test_verbose all lint lint_imports deps format typecheck typecheck_ty cov audit run benchmark precommit precommit_all precommit_install analyze_correlations analyze_correlations_latest backtest_smoke_test backtest_small backtest_medium backtest_large ablation_qa_research ablation_smoke ablation_small ablation_medium ablation_score test_e2e test_live test_fast check_credits sync_research sync_telemetry sync_raw_research sync_all resync_from_store backfill_research download_research download_run_logs download_raw_research backfill_comments score_ghosts close_margin_watch supply_probe supply_probe_mantic probe_slugs verify_pull cost_report dispatch_watch cronjob_dispatch_setup backtest_with_cache run_mantic run_mantic_one strip_bench probe_resolver replay_ladder artifacts_link artifacts_check artifacts_migrate

# Stream logs live from recipes; avoid per-target buffering
MAKEFLAGS += --output-sync=none

# OS detection for cross-platform unbuffered output with PTY
# - Linux: stdbuf + script -c "cmd" /dev/null
# - macOS: script -q /dev/null cmd (no stdbuf needed, different script syntax)
# `uv run` executes inside the in-project .venv that `uv sync` manages.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS: script allocates PTY; PYTHONUNBUFFERED handles Python buffering
    define RUN_UNBUFFERED
        PYTHONUNBUFFERED=1 script -q /dev/null uv run python -u $(1)
    endef
else
    # Linux: stdbuf for system-level line buffering + script for PTY
    define RUN_UNBUFFERED
        PYTHONUNBUFFERED=1 stdbuf -oL -eL script -q -e -c "uv run python -u $(1)" /dev/null
    endef
endif

install:
	uv sync --dev

lock:
	uv lock

lint:
	uv run ruff check .

# Dependency hygiene (deptry): an import of a package nothing declares, or a dev-only
# package imported by the shipped code. The second class is the one that bites — CI
# installs with `uv sync --dev` and the bot workflows install with `uv sync --no-dev`,
# so a misplaced dev dependency is green here and a ModuleNotFoundError in prod.
# Config + every per-rule ignore's justification live in [tool.deptry] in pyproject.toml.
deps:
	uv run deptry .

# Import-direction contracts (import-linter). Six forbidden-direction contracts derived
# from the real grimp graph; see [tool.importlinter] in pyproject.toml for the rationale
# on each. Sub-second, so it belongs in the normal loop rather than a pre-merge ritual.
lint_imports:
	uv run lint-imports

format:
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	uv run basedpyright

typecheck_ty:
	uv run ty check

cov:
	$(call RUN_UNBUFFERED,-m pytest --cov=metaculus_bot --cov-report=term-missing)

# Scan uv.lock for known vulnerabilities. osv-scanner is a Go binary (not a
# PyPI package), so it can't be run via uvx — install it with
# `brew install osv-scanner` (see https://google.github.io/osv-scanner/installation/).
# CI runs the equivalent google/osv-scanner-action.
audit:
	osv-scanner scan --lockfile=uv.lock --config=osv-scanner.toml

# Pre-commit helpers (use local cache to avoid readonly home cache)
# Both hook types: the ruff hooks fire pre-commit, the full-suite hook fires
# pre-push, and `pre-commit install` alone would install only the former.
precommit_install:
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit install
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit install --hook-type pre-push

precommit:
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit run

precommit_all:
	PRE_COMMIT_HOME=.pre-commit-cache uv run pre-commit run -a

test:
	$(call RUN_UNBUFFERED,-m pytest)

.PHONY: test_fetch_formats
test_fetch_formats:
	$(call RUN_UNBUFFERED,-m pytest tests/test_source_documents.py tests/test_source_presentation.py tests/test_fetch_ladder_local_sources.py tests/test_local_source_tools.py tests/test_image_tools.py tests/test_image_assets.py tests/test_image_leads.py tests/test_agentic_images.py tests/test_image_persistence.py tests/test_fetch_formats_e2e.py tests/test_source_probe.py)

# Verbose test run: shows which tests are running/failing and where, with
# short tracebacks. Useful when debugging a regression.
test_verbose:
	$(call RUN_UNBUFFERED,-m pytest -v --tb=short)

# One-stop pre-merge check: format, lint, then run tests with verbose output.
# Lint output is informative on violations; tests use -v + short tracebacks so
# you can see which test failed and why without wading through bare pass/fail.
# Recipes run sequentially (default make behavior); if any step fails the
# subsequent steps don't run, so failures surface immediately.
# deps + lint_imports sit here rather than only in CI because both are sub-second and
# both are things CI will reject: without them the first sign of a misplaced dependency
# or a wrong-direction import is a red required check after a push.
all: format lint deps lint_imports typecheck test_verbose

run:
	$(call RUN_UNBUFFERED,main.py)

# PAID and PUBLISHES (ask-first gate, see AGENTS.md): forecasts the Mantic Crucible tournament
# on the operator's PERSONAL keys only. The env var below is what lets cli start a Mantic run;
# without it the run fails shut before any fetch or spend. ~$2.60 per question.
run_mantic:
	DONATED_OPENROUTER_KEY_ENABLED=false $(call RUN_UNBUFFERED,main.py --mode mantic)

# PAID and PUBLISHES (ask-first gate, see AGENTS.md): the one-question Mantic smoke run. The same
# run as run_mantic, narrowed to the open post id(s) in POST (comma-separated): make run_mantic_one POST=650
run_mantic_one:
	$(if $(POST),,$(error run_mantic_one needs POST=<post id>[,<post id>...], e.g. make run_mantic_one POST=650))
	DONATED_OPENROUTER_KEY_ENABLED=false $(call RUN_UNBUFFERED,main.py --mode mantic --only-posts $(POST))

# DEPRECATED: Community benchmark baseline scoring is broken because Metaculus removed
# the aggregations field from their list API. Use backtest_* targets instead.
# These targets still work for fetching/running questions, but expected_baseline_score is unreliable.
benchmark_run_smoke_test_binary:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_smoke_test'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode run --num-questions 1)

benchmark_run_smoke_test:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_smoke_test'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 4 --mixed)

benchmark_run_binary_only:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_small'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode run --num-questions 30)

benchmark_run_small:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_small'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 12 --mixed)

benchmark_run_medium:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_medium'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 32 --mixed)

benchmark_run_large:
	@echo "WARNING: community benchmark is deprecated — baseline scoring is broken. Prefer 'make backtest_large'."
	$(call RUN_UNBUFFERED,community_benchmark.py --mode custom --num-questions 100 --mixed)

benchmark_display:
	$(call RUN_UNBUFFERED,community_benchmark.py --mode display)

analyze_correlations:
	$(call RUN_UNBUFFERED,analyze_correlations.py $(if $(FILE),$(FILE),benchmarks/))

analyze_correlations_latest:
	$(call RUN_UNBUFFERED,analyze_correlations.py $$(ls -t benchmarks/benchmarks_*.jsonl | head -1))

analyze_correlations_latest_excluding:
	$(call RUN_UNBUFFERED,analyze_correlations.py $$(ls -t benchmarks/benchmarks_*.jsonl | head -1) --exclude-models grok-4 gemini-2.5-pro)

backtest_smoke_test:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 4)

backtest_small:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 12)

backtest_medium:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 32)

backtest_large:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 100)

# Probabilistic-tools ablation benchmark.
# CLI: metaculus_bot/ablation/cli.py (entry point: python -m metaculus_bot.ablation.cli).
# Tournaments default in the CLI (spring-aib-2026 + other 2026 slugs); not pinned here.
ablation_qa_research:
	# Runs only fetch + research + leakage screen + QA dump, then halts.
	# Uses 3/3/3 question mix. No forecasting, no stacking.
	# Config-in-code: --no-gap-fill (gap-fill amplifies leakage on resolved Qs);
	# --gemini-model gemini-2.5-flash (free tier, no Tier 1 billing required).
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 3 --num-multiple-choice 3 --num-numeric 3 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash --qa-research)

ablation_smoke:
	# 9 questions: 3 binary, 3 MC, 3 numeric. Full pipeline through scoring.
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 3 --num-multiple-choice 3 --num-numeric 3 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash)

ablation_small:
	# 15 questions: 5/5/5.
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 5 --num-multiple-choice 5 --num-numeric 5 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash)

ablation_medium:
	# 60 questions: 20/20/20. PENDING USER SIGN-OFF — do not run without explicit go-ahead.
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --num-binary 20 --num-multiple-choice 20 --num-numeric 20 --resolved-after 2026-01-01 --no-gap-fill --gemini-model gemini-2.5-flash)

ablation_score:
	# Re-runs scoring against existing caches (no API spend).
	$(call RUN_UNBUFFERED,-m metaculus_bot.ablation.cli --stages score)

test_e2e:
	$(call RUN_UNBUFFERED,-m pytest -m e2e -v --tb=short)

test_live:
	$(call RUN_UNBUFFERED,-m pytest -m live -v --tb=short --timeout=300)

test_fast:
	$(call RUN_UNBUFFERED,-m pytest -m "not live and not e2e" --tb=short)

# --- Research persistence (backtest replay) ---

# Sync research archive: download GHA artifacts (source of truth) + backfill
# from Metaculus comments for anything missing.
#
# WHY RUN THIS REGULARLY: GHA uploads each run's research_outputs/ artifact with
# retention-days: 90 (run_bot_on_{tournament,metaculus_cup,minibench}.yaml). After
# 90 days the artifact is deleted FOREVER, so backtests/research_archive/ is the
# only durable copy. download_research.py enumerates EVERY research-* artifact via
# the complete, paginated artifacts REST endpoint (no run-list cap). Schedule it
# WEEKLY (well inside 90 days); see scripts/research_sync/ for the launchd job.
#
# ORDERING IS LOAD-BEARING: the comment backfill runs FIRST so it writes
# comments_backfill.jsonl into the backfill dir, then download_research.py does ONE
# authoritative build — download artifacts, load ALL backfill (incl. the fresh
# comments), dedup by (qid, run_id), build. One build that sees both sources is still
# the right shape: a separate rebuild pass used to CLOBBER the just-downloaded artifact
# records, since they live only in the build's in-memory list and never in the backfill
# dir. Since 2026-08-03 a rebuild also re-ingests by_qid/, so that pass is survivable
# rather than destructive — but there is no reason to run two builds.
sync_research:
	@echo "=== Backfilling from Metaculus comments (historical; non-fatal — see sync_all) ==="
	-uv run python scripts/backfill_research_from_comments.py
	@echo ""
	@echo "=== Downloading GHA artifacts + building archive (artifacts + backfill) ==="
	uv run python scripts/download_research.py $(ARGS)
	@echo ""
	@echo "Archive ready at backtests/research_archive/latest/"

# Harvest run-log telemetry markers (EXTRACTION_RUNG, GAP_FILL_V2, GHOST_PRE[_JSON],
# GHOST_FORECAST[_JSON], OPEN_BOUND_PILING, CREDIT_*) from GHA artifacts into the durable local archive
# (backtests/telemetry_archive/). Every bot run bundles run_logs/ inside research-*; the
# downloader also pulls the logs-* family the test workflows used before 2026-08-03.
# Read-only + free (GitHub API only) and idempotent (replace-by-run), so it's safe on
# the weekly schedule. Pass ARGS="--since-days N" to scope the pull.
sync_telemetry:
	@echo "=== Harvesting run-log telemetry from GHA artifacts ==="
	uv run python scripts/download_run_logs.py $(ARGS)
	@echo ""
	@echo "Telemetry archive ready at backtests/telemetry_archive/"

# Archive the raw research-provider payload logs (raw_research_<run_id>.jsonl) that
# metaculus_bot.research.raw_log appends to run_logs/. Pulls both artifact families
# (every bot run bundles run_logs/ inside research-*; the test workflows used a separate
# logs-* before 2026-08-03),
# harvests the raw JSONL, and writes one file per run to backtests/research_archive/raw/
# (replace-by-run, idempotent). Read-only + free; safe on the weekly schedule.
sync_raw_research:
	@echo "=== Archiving raw research-provider payload logs from GHA artifacts ==="
	uv run python scripts/download_raw_research.py $(ARGS)
	@echo ""
	@echo "Raw-research archive ready at backtests/research_archive/raw/"

# Pull EVERYTHING sync-shaped in one command: the research archive, the telemetry
# archive, AND the raw research-provider payload archive. Residual analyses should call
# this (never a single sync) so a future source is never silently missed. Read-only + free.
#
# SINGLE-PASS: unlike running the three sync_* targets in sequence (which each
# re-enumerate every artifact and re-download the overlapping research-*/logs-* families
# — ~300 downloads for ~100 artifacts), scripts/sync_all.py enumerates ONCE over the
# union family and downloads each artifact ONCE into the PERSISTED STORE
# (backtests/gha_artifact_store/), then runs all three harvests over those persisted run
# dirs. An artifact already in the store is never re-downloaded. The Metaculus-comment
# backfill runs FIRST
# (it hits Metaculus, not GHA) so its comments_backfill.jsonl is on disk when the
# driver's research build loads it. NOTE: ARGS is forwarded only to sync_all.py, which
# accepts --repo / --since-days (and the per-archive --*-dir overrides).
#
# THE BACKFILL IS NON-FATAL (leading `-`), and that asymmetry is the whole point.
# The two halves have opposite deadlines: comments live on Metaculus forever, while GHA
# artifacts are deleted at 90 days, so the GHA pull is the only half that can lose data
# permanently. Under `set -euo pipefail` (scripts/research_sync/run_sync.sh) a bare
# backfill failure aborted the recipe before sync_all.py downloaded a single artifact —
# which is exactly what happened on all five scheduled runs from 2026-06-28 to 2026-07-26:
# launchd fires at the first wake after Sun 03:00, the laptop's network is not up yet, the
# backfill's un-retried requests.get raises ConnectionError, and the expiring half never
# ran. The `-` lets the recoverable half fail without taking the unrecoverable half with
# it. Keeping it FIRST preserves the one-authoritative-build invariant above (sync_all.py's
# research build loads comments_backfill.jsonl); on a failed backfill the build simply
# reads the previous run's file.
sync_all:
	@echo "=== Backfilling from Metaculus comments (historical; non-fatal — see comment above) ==="
	-uv run python scripts/backfill_research_from_comments.py
	@echo ""
	@echo "=== Single-pass GHA sync: research + telemetry + raw-research (one download pass) ==="
	uv run python scripts/sync_all.py $(ARGS)
	@echo ""
	@echo "=== sync_all complete: research + telemetry + raw-research archives refreshed ==="

# OFFLINE re-parse: rebuild all three archives from the persisted artifact store
# (backtests/gha_artifact_store/) with ZERO network calls. This is the payoff of
# persisting downloads — after fixing an ingest/parse bug, the artifacts' bytes are
# already on local disk, so the corrected harvest re-runs for free and works on artifacts
# GHA has since deleted (90-day retention). Skips the Metaculus backfill for the same
# reason: comments_backfill.jsonl from the last sync is already on disk and the research
# build loads it. Free, and safe to run repeatedly (every archive build is replace-by-run).
resync_from_store:
	@echo "=== Offline re-harvest of all three archives from backtests/gha_artifact_store/ ==="
	uv run python scripts/sync_all.py --from-store $(ARGS)
	@echo ""
	@echo "=== resync_from_store complete (no network was used) ==="

# Score gap-fill v2 GHOST_FORECAST markers vs published forecasts on resolved questions
# (paired log-score deltas — the retire-v1 gate). Read-only + free. Expects ~0
# scoreable today (v2 shipped 2026-07-17); pass ARGS="--tournament <slug>" for a live
# read-only resolutions pull, or ARGS="--perf-json <path>" for a pre-built dataset.
score_ghosts:
	uv run python scripts/score_ghosts.py $(ARGS)

# Weekly close-margin watch over the CLOSE_MARGIN telemetry archive: p50/p10/min of
# window-remaining-at-submit per ISO week + questions under the 30% red line. Read-only
# + free (reads backtests/telemetry_archive/close_margin.jsonl; run sync_telemetry first).
# ARGS="--red-line 0.5" for a tighter line, ARGS="--output <path>" to dump the summary JSON.
close_margin_watch:
	uv run python scripts/close_margin_watch.py $(ARGS)

# Question-supply census per tournament slug, counting post status `closed` (closed to
# forecasting, NOT yet resolved) alongside open/resolved, plus the backlog of unresolved
# questions past their own scheduled_resolve_time. Read-only + free (Metaculus posts list
# only; no LLM, no research, no publish). Omitting `closed` is what made two consecutive
# residual rounds' supply projections miss. ARGS="--slugs <a> <b>" to scope,
# ARGS="--statuses open closed" to narrow, ARGS="--output <path>" to dump JSON.
supply_probe:
	uv run python scripts/supply_probe.py $(ARGS)

# The same probe against Mantic's Crucible (competitions.mantic.com), read-only + free: its
# posts list is public, so MANTIC_TOKEN is optional and only lets closed-but-unresolved
# questions classify; resolved ones classify from the public spot-time snapshot. Adds the
# miss-rate-per-UTC-release-hour table that answers the cron-cadence question.
supply_probe_mantic:
	uv run python scripts/supply_probe.py --platform mantic $(ARGS)

# Which season slugs exist, which the repo's constants point at, and which are stale: one project
# GET per candidate (the configured slugs plus the generated season-successor spellings). Read-only
# + free (Metaculus project metadata only; no LLM, no research, no publish). The project object is
# the authoritative existence check because the tournaments LIST omits an `unlisted` project, which
# is what a new season is before its first question. Post counts are supply_probe's job.
# ARGS="--seasons-ahead 2", ARGS="--slugs <a> <b>" for extras, ARGS="--output <path>".
probe_slugs:
	uv run python scripts/probe_slugs.py $(ARGS)

# Coverage audit of a completed residual-round pull, FULLY OFFLINE (no network at all): which
# resolved posts produced no record and why, resolved group members with no record, the diff versus
# the prior round, whether every overlapping platform score reproduces, and whether the pull still
# parses per-model forecasts out of the bot's comments. Recordless posts a prior round already
# classified are derived from that round's own checkpoint, so nothing is hardcoded per round.
# Needs ARGS="--records <path>"; add --prior-records for checks 3-5 and --output for the new cohort.
verify_pull:
	uv run python scripts/verify_pull.py $(ARGS)

# Cost per question off the telemetry archive (read-only + free; run sync_telemetry first): per
# run (questions, charged $, $/question), per role ($/question, prompt and output tokens per
# question, prompt-cache share, largest single prompt) and the week-over-week median $/question.
# The question denominator is CREDIT_RUN_SUMMARY where a run has one, else its
# FORECASTERS_SURVIVED lines. ARGS="--days 7" to narrow the window (default 30).
cost_report:
	uv run python scripts/cost_report.py $(ARGS)

# One-question probe of gap-fill v1's per-gap resolver: replays the gaps the archive recorded
# for question QUESTION=<question id> through the production resolver path
# (gap_fill_search_prompt + build_native_search_llm) at every model x search_context_size cell
# of a grid (default: the current resolver model and gpt-5.6-luna, each at high/medium/low),
# and writes the answers side by side with OpenRouter's per-call cost and tokens to
# scratch/probes/. PAID (ask-first gate, see AGENTS.md): gaps x cells resolver calls at up to
# ~$0.20 each on the operator's PERSONAL OpenRouter key (the donated key is forced off); the
# script prints its ceiling first and refuses without ARGS="--i-accept-spend". Narrow with
# ARGS="--grid current:high luna:low" or ARGS="--gaps 1,2".
probe_resolver:
	$(if $(QUESTION),,$(error probe_resolver needs QUESTION=<question id>, e.g. make probe_resolver QUESTION=44267))
	uv run python scripts/probes/gap_fill_resolver_probe.py --question $(QUESTION) $(ARGS)

# Trigger-delivery watch: per bot workflow and per UTC day, how many `schedule` and
# `workflow_dispatch` runs GitHub Actions ran and how they concluded, against the cron entries
# in .github/workflows and the external dispatcher's cadence (2/h by default), flagging a day
# whose dispatched runs fall short or whose cron delivery is below half. Read-only + free (one
# `gh run list`; no dispatch, no LLM, no publish). ARGS="--days 14", ARGS="--expected-dispatch-per-hour 0".
dispatch_watch:
	uv run python scripts/dispatch_watch.py $(ARGS)

# Replay the archived per-URL fetch outcomes through the unified fetch ladder's rung selection: for
# each archived direct outcome, which rungs each LadderPolicy preset would consult and in what
# order, plus any archived rescue (impersonate, rendered, derived_api, wayback, url_context) the new
# dispatcher no longer attempts. Every rung is stubbed to decline, so nothing is dialed: read-only
# and free, no network, no LLM, no publish. A LOCAL pre-merge check, never CI. Point --archive-dir
# at a checkout that HAS the gitignored archive. ARGS="--format json", ARGS="--policy NAME",
# ARGS="--limit 200".
replay_ladder:
	uv run python scripts/fetch_ladder_replay.py $(ARGS)

# Idempotent setup of the cron-job.org jobs that dispatch the bot workflows twice an hour each
# (GitHub delivers about 22% of scheduled cron firings, see docs/operations.md "Scheduling
# reliability"; workflow_dispatch events are not dropped, and a run that finds no new question
# spends nothing). The default is a DRY RUN: it prints the three job payloads with the GitHub
# token redacted and, when CRONJOB_API_KEY and GH_DISPATCH_TOKEN are both set, the
# create/update/unchanged plan from a read-only list of the account; it never writes.
# ARGS="--apply" is PAID (ask-first gate, see AGENTS.md): it creates or changes a live schedule,
# and every firing it adds is a paid, publishing bot run. ARGS="--apply --enable-mantic" once
# run_bot_on_mantic.yaml is on main.
cronjob_dispatch_setup:
	uv run python scripts/cronjob_dispatch_setup.py $(ARGS)

# Download run-log artifacts + harvest telemetry only (no research sync). Same script
# as sync_telemetry; kept as a named target for parity with download_research.
download_run_logs:
	uv run python scripts/download_run_logs.py $(ARGS)

# Download artifacts + archive raw research-provider logs only (no other sync). Same
# script as sync_raw_research; kept as a named target for parity with download_research.
download_raw_research:
	uv run python scripts/download_raw_research.py $(ARGS)

# Backfill research from existing GitHub Actions logs (Nov 2025 onward).
# Pass ARGS="--limit 100 --status completed" to customize.
backfill_research:
	uv run python scripts/backfill_research_from_logs.py $(ARGS)

# Download research artifacts into the local archive + rebuild. Enumerates EVERY
# research-* artifact (all run-workflows) via the complete paginated artifacts REST
# endpoint, merges with backfill, dedups, and builds. Pass ARGS="--since-days N" to
# scope, or ARGS="--rebuild-only" to rebuild from local data with no artifact fetch
# (by_qid/ + backfill/ — offline and free, and it keeps the artifact records).
download_research:
	uv run python scripts/download_research.py $(ARGS)

# Backfill from Metaculus bot comments (historical, covers full tournament).
backfill_comments:
	uv run python scripts/backfill_research_from_comments.py $(ARGS)

# Run backtest using cached (non-leaky) research from the archive.
backtest_with_cache:
	$(call RUN_UNBUFFERED,backtest.py --num-questions 20 --research-dir backtests/research_archive/latest $(ARGS))

# Check OpenRouter key balances. Pass ARGS="--key donated" or ARGS="--key personal"
# to limit which key is queried (default: both).
check_credits:
	@uv run python -m metaculus_bot.check_openrouter_credits $(ARGS)

# PAID, PERSONAL KEY ONLY (ask-first gate, see AGENTS.md): the paired section-strip bench forecasts
# every resolved gap-fill pair four ways (full bundle, minus v1, minus v2, minus both) with one cheap
# model on OPENROUTER_API_KEY and scores the arms against the resolutions. Nothing publishes and no
# research runs; the estimate at the default 3 replicates is about $1.25. Every mode except
# --rescore needs --pairs and --perf-json, which name the round directory being benched (they used
# to default to a fixed round and silently went stale). So the plan-and-refuse call is
# ARGS="--pairs <pairs.jsonl> --perf-json <perf_all_tagged.json>", the same plus ARGS="--dry-run"
# is the free view, plus ARGS="--i-accept-spend" runs it under --max-spend-usd (default 10), and
# ARGS="--rescore <run dir>" rebuilds results offline from a finished run.
strip_bench:
	uv run python -m scripts.probes.section_strip_bench $(ARGS)

# Symlink the gitignored artifact trees (backtests/, scratch/, benchmarks/, research_outputs/,
# run_logs/) to the private sibling repo that carries them between machines. Free, offline, and
# no LLM or research call. `artifacts_check` reports without changing anything; `artifacts_migrate`
# is the one-time move on the machine that currently holds the data. See scripts/artifacts_link.sh
# for why the artifacts live in a separate PRIVATE repo and why they are stored raw.
artifacts_link:
	@scripts/artifacts_link.sh

artifacts_check:
	@scripts/artifacts_link.sh --check

artifacts_migrate:
	@scripts/artifacts_link.sh --migrate
