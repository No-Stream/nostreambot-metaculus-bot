"""Paths, thresholds and names that only the trace-review tool uses."""

from __future__ import annotations

from pathlib import Path

from metaculus_bot.constants import MANTIC_SITE_URL
from metaculus_bot.performance_analysis.research_tags import DEFAULT_RESEARCH_ARCHIVE_LATEST

# Where the builder writes and the app reads; the env var is how tests point the app at a fixture.
DEFAULT_BUNDLE_DIR = Path("backtests/trace_index")
BUNDLE_DIR_ENV = "TRACE_REVIEW_BUNDLE_DIR"
PULLS_DIR_NAME = "pulls"
BUILDING_DIR_NAME = ".building"

FORECASTS_FILE = "forecasts.parquet"
MEMBERS_FILE = "members.parquet"
TEXTS_FILE = "texts.parquet"
TELEMETRY_FILE = "telemetry.parquet"
MANIFEST_FILE = "manifest.json"
BUNDLE_TABLE_FILES: tuple[str, ...] = (FORECASTS_FILE, MEMBERS_FILE, TEXTS_FILE, TELEMETRY_FILE)

# Inputs: the residual rounds, the telemetry archive and both halves of the research archive.
DEFAULT_SCRATCH_DIR = Path("scratch")
ROUND_DIR_PREFIX = "residual_"
ROUND_DATASET_NAME = "perf_all_tagged.json"
DEFAULT_TELEMETRY_DIR = Path("backtests/telemetry_archive")
DEFAULT_RESEARCH_LATEST_DIR = DEFAULT_RESEARCH_ARCHIVE_LATEST
DEFAULT_RESEARCH_RAW_DIR = Path("backtests/research_archive/raw")

# A cached pull older than this is pulled again; which comments it expanded depends on the round.
PULL_MAX_AGE_HOURS = 6
FRESH_PROVENANCE_PREFIX = "trace_review_fresh_"

PARQUET_COMPRESSION = "zstd"
# Small row groups so the detail page's one-row read decompresses ~16 comments, not the whole file.
TEXTS_ROW_GROUP_SIZE = 16

# The "confident and wrong" preset: at most this much probability on what happened...
CONFIDENT_WRONG_MAX_PROB = 0.15
# ...or, on a numeric question, a PIT this close to either end (or this little published mass on
# the side an out-of-range resolution landed).
CONFIDENT_WRONG_TAIL = 0.05

RECORD_ORIGIN_ROUND = "round"
RECORD_ORIGIN_FRESH = "fresh_pull"

METACULUS_QUESTION_URL = "https://www.metaculus.com/questions/{post_id}/"
MANTIC_QUESTION_URL = MANTIC_SITE_URL + "/questions/{post_id}/"

# A run is a Mantic run exactly when its run_id logged this marker (the preflight logs it every run).
MANTIC_RUN_MARKER = "mantic_tournaments"
RUNS_INDEX_FILE = "runs.jsonl"
# Run-level lines copied into each forecast's telemetry rows.
RUN_LEVEL_MARKERS: tuple[str, ...] = (
    "credit_run_summary",
    "forecaster_drops",
    "degradation_counters",
    "run_status_json",
)
# Logged only when something happens, so absence means "no" only after the marker existed.
EVENT_MARKERS: tuple[str, ...] = ("extreme_call", "ghost_forecast_json")
