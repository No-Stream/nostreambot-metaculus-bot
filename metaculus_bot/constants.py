"""
Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules. Each value's receipt, meaning the measurement,
incident or operator decision behind it, lives in docs/constants.md under a
heading named for the constant.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta

from metaculus_bot.config import load_environment
from metaculus_bot.time_utils import _as_utc

# =============================================================================
# TOURNAMENT IDs - UPDATE THESE EACH QUARTER/SEASON
# =============================================================================
# The end date is the project's forecasting_end_date. Receipt: docs/constants.md "TOURNAMENT_ID".
TOURNAMENT_ID: str = "fall-futureeval-2026"  # Fall 2026 FutureEval Bot Tournament (project ID: 33121)
TOURNAMENT_END_DATE: str = "2027-01-06"  # forecasting_end_date on project 33121 (API-verified 2026-09-06)
TOURNAMENT_HARD_STOP_WEEKS: int = 2  # ~2 weeks of wiggle room past close before erroring

# The undated `metaculus-cup` slug now answers HTTP 400. Receipt: docs/constants.md "METACULUS_CUP_ID".
METACULUS_CUP_ID: str = "metaculus-cup-fall-2026"

# Mantic's Crucible competition, a Metaculus fork. Receipt: docs/constants.md "MANTIC_HOST".
MANTIC_HOST: str = "competitions.mantic.com"
MANTIC_SITE_URL: str = f"https://{MANTIC_HOST}"
MANTIC_API_BASE_URL: str = f"{MANTIC_SITE_URL}/api"
MANTIC_TOURNAMENT_ID: str = "preseason-2"
MANTIC_TOURNAMENT_END_DATE: str = "2026-09-20"  # forecasting_end_date on project 4 (API-verified 2026-09-08)
# Mantic advertises `next` past the last page. Receipt: docs/constants.md "MANTIC_FETCH_QUESTION_CEILING".
MANTIC_FETCH_QUESTION_CEILING: int = 500
# The bot's own account (``nostreambot-bot``, ``/api/users/81/``): an unauthenticated read has no ``my_forecasts``.
MANTIC_BOT_USER_ID: int = 81
# Scored against a fixed 0.05 reference. Receipt: docs/constants.md "MANTIC_OUT_OF_RANGE_TAIL_FLOOR".
MANTIC_OUT_OF_RANGE_TAIL_FLOOR: float = 0.05

# The hosts we publish to are research self-references. Receipt: docs/constants.md "METACULUS_HOST".
METACULUS_HOST: str = "metaculus.com"
QUESTION_PLATFORM_HOSTS: tuple[str, ...] = (METACULUS_HOST, MANTIC_HOST)

# Data-contract tokens: add, never re-spell. Receipt: docs/constants.md "PLATFORM_METACULUS".
PLATFORM_METACULUS: str = "metaculus"
PLATFORM_MANTIC: str = "mantic"


def gemini_use_donated_openrouter_key() -> bool:
    """Whether OpenRouter Gemini calls should route through the Metaculus-donated key.

    Default True: the donated key (``OAI_ANTH_OPENROUTER_KEY``) serves most Gemini models
    since Metaculus raised the Google rate limits on 2026-06-16. A false-y value
    (``"false"`` / ``"0"`` / ``"no"``) forces personal-key-only routing. Read at call time
    (not import) so workflow env changes take effect without re-importing.

    The ``gemini-3.1-pro-preview`` forecaster slot is PINNED to the personal key regardless,
    and this toggle does not touch the google-genai grounded-search provider. Both
    exceptions, with their receipts: docs/constants.md "gemini_use_donated_openrouter_key".
    """
    return env_flag_enabled(GEMINI_USE_DONATED_OPENROUTER_KEY_ENV, default=True)


def donated_openrouter_key_enabled() -> bool:
    """Whether ANY OpenRouter call may route through the Metaculus-donated key.

    Default True, so Metaculus runs are unchanged. A Mantic run sets
    ``DONATED_OPENROUTER_KEY_ENABLED=false`` so it spends only the operator's personal keys;
    ``should_route_via_donated_key`` (fallback_openrouter) consults this before any provider
    match, so one false-y value covers every OpenRouter key choice in the process. Read at
    call time (not import) so a workflow env change needs no re-import.

    Why an env var set before process start rather than a CLI flag, and how Mantic mode fails
    shut on it: docs/constants.md "donated_openrouter_key_enabled".
    """
    return env_flag_enabled(DONATED_OPENROUTER_KEY_ENABLED_ENV, default=True)


class TournamentExpiredError(Exception):
    """Raised when the tournament has ended and the ID needs to be updated."""


def check_tournament_dates(
    logger: logging.Logger | None = None,
    *,
    tournament_id: str | None = None,
    end_date_str: str | None = None,
) -> bool:
    """Check if tournament dates are stale and warn/error accordingly; True when past the end date.

    - Warns, and returns True, once the current UTC date is past the tournament's end date. The
      end date is the LAST open day, not the first dead one: Preseason 2 forecasts until 12:00 UTC
      on ``MANTIC_TOURNAMENT_END_DATE``, and a run in those hours publishes normally.
    - Raises TournamentExpiredError if past end date + TOURNAMENT_HARD_STOP_WEEKS

    Defaults to the Metaculus bot tournament (``TOURNAMENT_ID`` / ``TOURNAMENT_END_DATE``);
    the Mantic mode passes ``MANTIC_TOURNAMENT_ID`` / ``MANTIC_TOURNAMENT_END_DATE``. The
    defaults resolve at CALL time (None sentinels), so a module-level patch of the constants
    is honored. Call this at bot startup to catch stale tournament IDs. The verdict is what lets
    a caller make staleness alertable rather than advisory: cli reddens a Mantic run on it, while
    the Metaculus modes keep the warning only (their questions stay open for weeks, so a fortnight
    of warnings costs nothing; on Mantic the same fortnight forfeits every Series 2 question).
    """
    log = logger or logging.getLogger(__name__)
    tournament_id = TOURNAMENT_ID if tournament_id is None else tournament_id
    end_date_str = TOURNAMENT_END_DATE if end_date_str is None else end_date_str

    # Both operands go through _as_utc. Receipt: docs/constants.md "check_tournament_dates".
    try:
        end_date = _as_utc(datetime.strptime(end_date_str, "%Y-%m-%d"))  # noqa: DTZ007  # stamped UTC by _as_utc
    except ValueError:
        log.warning(f"Invalid tournament end date format for '{tournament_id}': {end_date_str}")
        return False

    today = _as_utc(datetime.now(UTC))
    stale_from = end_date + timedelta(days=1)
    hard_stop_date = end_date + timedelta(weeks=TOURNAMENT_HARD_STOP_WEEKS)

    if today > hard_stop_date:
        raise TournamentExpiredError(
            f"Tournament '{tournament_id}' ended on {end_date_str} and hard stop "
            f"date ({hard_stop_date.date()}) has passed. Please update its tournament id and "
            f"end date (and TOURNAMENT_HARD_STOP_WEEKS if needed) in constants.py for the new season."
        )
    if today >= stale_from:
        days_past = (today - end_date).days
        days_until_error = (hard_stop_date - today).days
        log.warning(
            f"⚠️  Tournament '{tournament_id}' likely ended on {end_date_str} "
            f"({days_past} days ago). Update constants.py for the new season! "
            f"Bot will error out in {days_until_error} days."
        )
        return True
    return False


# --- Cup-season configuration reminder (dated, DISCHARGED for fall 2026, re-armable) ---

# Discharged 2026-09-03; re-arm instructions: docs/constants.md "FALL_CUP_SLUG".
FALL_CUP_SLUG: str = METACULUS_CUP_ID  # one definition, so the probe's slug list and the reminder can't drift
FALL_CUP_REMINDER_DATE: str = "2026-09-15"
FALL_CUP_CONFIGURED: bool = True  # set False to re-arm the reminder for the next cup season


def fall_cup_reminder_due(today: date | None = None) -> bool:
    """Whether the cup-season configuration reminder should redden runs.

    False before ``FALL_CUP_REMINDER_DATE``, and always False while
    ``FALL_CUP_CONFIGURED`` is True — which it is, so this returns False on every date
    until somebody re-arms it. ``today`` defaults to the system clock read at CALL time,
    same contract as ``credit_alerts_active`` below: tests inject a fixed date, and a
    long-lived process crosses the date without a redeploy.
    """
    if FALL_CUP_CONFIGURED:
        return False
    # Local calendar day is deliberate here. Receipt: docs/constants.md "fall_cup_reminder_due".
    return (today or date.today()) >= date.fromisoformat(FALL_CUP_REMINDER_DATE)  # noqa: DTZ011  # see comment above


def check_fall_cup_reminder(logger: logging.Logger | None = None, today: date | None = None) -> bool:
    """Log the loud FALL_CUP_REMINDER line when due; return whether it fired.

    Dormant while ``FALL_CUP_CONFIGURED`` is True (the shipped state since 2026-09-03).
    When re-armed, the caller (cli.main) holds the returned bool and exits non-zero at end
    of run, the same shape as the credit-floor path: forecasting and publishing complete
    normally, and the red exit is purely the reminder signal.
    """
    if not fall_cup_reminder_due(today):
        return False
    log = logger or logging.getLogger(__name__)
    log.error(
        f"FALL_CUP_REMINDER: the Metaculus Cup season constants look unconfigured. "
        f"METACULUS_CUP_ID still points at {FALL_CUP_SLUG} — re-point it at the new season's "
        f"DATED slug (there is no auto-resolving 'metaculus-cup' spelling left; Metaculus "
        f"answers HTTP 400 for that one) and enable the 'Forecast on Metaculus Cup' workflow "
        f"on GitHub. Flip FALL_CUP_CONFIGURED=True in constants.py to retire this reminder. "
        f"This run will exit non-zero as the reminder signal."
    )
    return True


load_environment()  # early, so ASKNEWS_* values are read correctly at import time in local runs

DEFAULT_MAX_CONCURRENT_RESEARCH: int = 6  # conservative for AskNews; adjust after observing rate limits

BENCHMARK_BATCH_SIZE: int = 4  # modest, to balance concurrency against provider rate limits

# One budget per section, trimmed before assembly. Receipt: docs/constants.md "FORECASTS_SECTION_CHAR_LIMIT".
FORECASTS_SECTION_CHAR_LIMIT: int = 89_999
RESEARCH_SECTION_CHAR_LIMIT: int = 44_999
SUMMARY_SECTION_CHAR_LIMIT: int = 13_999
COMMENT_CHAR_LIMIT: int = 149_999

RESEARCH_PROVIDER_ENV: str = "RESEARCH_PROVIDER"  # auto|asknews|exa|perplexity|openrouter, case-insensitive

# A URL list here replaces cli.py's EXAMPLE_QUESTIONS. Receipt: docs/constants.md "TEST_QUESTIONS_OVERRIDE_ENV".
TEST_QUESTIONS_OVERRIDE_ENV: str = "TEST_QUESTIONS_OVERRIDE"

# Named so the literals are not duplicated. Receipt: docs/constants.md "OPENROUTER_API_KEY_ENV".
OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"
OAI_ANTH_OPENROUTER_KEY_ENV: str = "OAI_ANTH_OPENROUTER_KEY"
ASKNEWS_CLIENT_ID_ENV: str = "ASKNEWS_CLIENT_ID"
ASKNEWS_SECRET_ENV: str = "ASKNEWS_SECRET"  # noqa: S105  # env var NAME, not a credential
EXA_API_KEY_ENV: str = "EXA_API_KEY"
PERPLEXITY_API_KEY_ENV: str = "PERPLEXITY_API_KEY"
METACULUS_TOKEN_ENV: str = "METACULUS_TOKEN"  # noqa: S105  # env var NAME, not a credential
MANTIC_TOKEN_ENV: str = "MANTIC_TOKEN"  # noqa: S105  # env var NAME, not a credential; personal, never donated
# Master switch; a Mantic run sets it false and fails shut. Receipt: docs/constants.md "donated_openrouter_key_enabled".
DONATED_OPENROUTER_KEY_ENABLED_ENV: str = "DONATED_OPENROUTER_KEY_ENABLED"


def env_flag_enabled(env_name: str, *, default: bool = False) -> bool:
    """Return True iff env var is set to "true"/"1"/"yes" (case-insensitive).

    When the env var is unset (or empty string), returns ``default``.
    Explicit "false"/"0"/"no" always returns False, regardless of default.
    """
    raw = os.getenv(env_name, "").lower()
    if raw == "":
        return default
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _date_env(name: str, default: date) -> date:
    """Parse an ISO ``YYYY-MM-DD`` env var into a date, falling back on garbage."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return default


# Global in-process and conservative for pro plans. Receipt: docs/constants.md "ASKNEWS_MAX_CONCURRENCY".
ASKNEWS_MAX_CONCURRENCY: int = max(1, _int_env("ASKNEWS_MAX_CONCURRENCY", 1))
ASKNEWS_MAX_RPS: float = max(0.1, _float_env("ASKNEWS_MAX_RPS", 0.8))

ASKNEWS_MAX_TRIES: int = max(1, _int_env("ASKNEWS_MAX_TRIES", 3))
ASKNEWS_BACKOFF_SECS: float = max(0.0, _float_env("ASKNEWS_BACKOFF_SECS", 2.0))
# Above the whole two-phase retry envelope. Receipt: docs/constants.md "ASKNEWS_WALL_TIMEOUT".
ASKNEWS_WALL_TIMEOUT: int = 300

# --- OpenRouter credit telemetry ---

# Early warning with lead time; only Metaculus can refill. Receipt: docs/constants.md "OPENROUTER_CREDIT_FLOOR_USD".
OPENROUTER_CREDIT_FLOOR_USD: float = _float_env("OPENROUTER_CREDIT_FLOOR_USD", 100.0)

# Suppresses the ALERTS, never the CREDIT_* logs. Receipt: docs/constants.md "CREDIT_ALERT_RESUME_DATE".
CREDIT_ALERT_RESUME_DATE: date = _date_env("OPENROUTER_CREDIT_ALERT_RESUME_DATE", date(2026, 9, 3))

# Well above the 41k gap-fill v2 peak, so a fire is a blow-up. Receipt: docs/constants.md "PROMPT_TOKENS_ALERT_THRESHOLD".
PROMPT_TOKENS_ALERT_THRESHOLD: int = 150_000


def credit_alerts_active(today: date | None = None) -> bool:
    """Whether credit shortfalls should still exit non-zero.

    False during the suppression window, True from ``CREDIT_ALERT_RESUME_DATE``
    onward. ``today`` defaults to the system clock read at CALL time (not at
    import), so a long-lived process crosses the resume date without a redeploy
    and tests can inject a fixed date instead of depending on the wall clock.
    """

    # Local calendar day is deliberate here. Receipt: docs/constants.md "fall_cup_reminder_due".
    return (today or date.today()) >= CREDIT_ALERT_RESUME_DATE  # noqa: DTZ011  # see comment above


# Ships EMPTY; dated and per-venue. Receipt: docs/constants.md "PROVIDER_DEGRADATION_SUPPRESSED_UNTIL".
PROVIDER_DEGRADATION_SUPPRESSED_UNTIL: dict[str, date] = {}


def provider_degradation_alerts_active(venue: str, today: date | None = None) -> bool:
    """Whether ``venue``'s provider-degradation findings should still exit non-zero.

    ``today`` defaults to the system clock read at CALL time (not at import), so a
    resume needs no redeploy and tests can inject a fixed date. A venue with no
    entry is always alertable.
    """
    resume = PROVIDER_DEGRADATION_SUPPRESSED_UNTIL.get(venue)
    # Local calendar day is deliberate here. Receipt: docs/constants.md "fall_cup_reminder_due".
    return resume is None or (today or date.today()) >= resume  # noqa: DTZ011  # see comment above


# --- Forecasting clamps and numeric smoothing ---

# The clip half of Preseen-Atlas's tail protection. Receipt: docs/constants.md "BINARY_PROB_MIN".
BINARY_PROB_MIN: float = 0.02
BINARY_PROB_MAX: float = 0.98

# Telemetry only, inclusive at both edges. Receipt: docs/constants.md "EXTREME_CALL_LOW".
EXTREME_CALL_LOW: float = 0.05
EXTREME_CALL_HIGH: float = 0.95

# Aliased so telemetry and clamp cannot drift. Receipt: docs/constants.md "THIN_PUBLISH_BINARY_FLOOR".
THIN_PUBLISH_BINARY_FLOOR: float = EXTREME_CALL_LOW
THIN_PUBLISH_BINARY_CEIL: float = EXTREME_CALL_HIGH

# A no-op on our output under ft 0.2.92's PredictedOptionList validator. Receipt: docs/constants.md "MC_PROB_MIN".
MC_PROB_MIN: float = 0.01
MC_PROB_MAX: float = 0.99

# Present only where that bound is OPEN; they share ``bin_probs`` with the labels so one object sums to 1.
PMF_BELOW_RANGE_KEY: str = "below_range"
PMF_ABOVE_RANGE_KEY: str = "above_range"

# --- Post-hoc Platt calibration of the final published probability ---

# Hard absolute caps after the logistic transform. Receipt: docs/constants.md "PLATT_CALIBRATION_ENABLED_ENV".
PLATT_CALIBRATION_ENABLED_ENV: str = "PLATT_CALIBRATION_ENABLED"
PLATT_BINARY_MAX_ABS_DEVIATION: float = 0.10
PLATT_MC_MAX_ABS_DEVIATION: float = 0.05

# Numeric CDF smoothing and spacing; the pipeline that consumes them is docs/numeric_pipeline.md.
NUM_VALUE_EPSILON_MULT: float = 1e-9
NUM_SPREAD_DELTA_MULT: float = 1e-6
NUM_MIN_PROB_STEP: float = 5e-5
NUM_MAX_STEP: float = 0.2
NUM_RAMP_K_FACTOR: float = 3.0

DISCRETE_SNAP_MAX_INTEGERS: int = 200  # snapping for integer-outcome "continuous" questions
DISCRETE_SNAP_UNIFORM_MIX: float = 0.0

# --- Conditional Stacking Thresholds ---

# Probability range, not log-odds. Receipt: docs/constants.md "CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD".
CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD: float = 0.15
# Multiple choice: max per-option probability spread (max - min across models for worst option).
CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD: float = 0.20
# Numeric: max percentile spread normalized by question range (at 10th/50th/90th percentiles).
CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD: float = 0.15

# --- Native Search Provider ---
NATIVE_SEARCH_ENABLED_ENV: str = "NATIVE_SEARCH_ENABLED"
NATIVE_SEARCH_MODEL_ENV: str = "NATIVE_SEARCH_MODEL"
# Critical-path research; sol->terra 2026-07-17. Receipt: docs/constants.md "NATIVE_SEARCH_DEFAULT_MODEL".
NATIVE_SEARCH_DEFAULT_MODEL: str = "openai/gpt-5.6-terra"
NATIVE_SEARCH_MAX_TOKENS: int = 16_000  # no temperature / top_p: temperature=None, so litellm omits it
# The litellm per-request timeout, 240->360 on 2026-05-17. Receipt: docs/constants.md "NATIVE_SEARCH_TIMEOUT".
NATIVE_SEARCH_TIMEOUT: int = 360
# A hard wall the HTTP layer cannot defeat (2026-05-20). Receipt: docs/constants.md "NATIVE_SEARCH_WALL_TIMEOUT".
NATIVE_SEARCH_WALL_TIMEOUT: int = 420
# Low since 2026-05-20, ~4.5x faster. Receipt: docs/constants.md "NATIVE_SEARCH_REASONING_EFFORT_DEFAULT".
NATIVE_SEARCH_REASONING_EFFORT_ENV: str = "NATIVE_SEARCH_REASONING_EFFORT"
NATIVE_SEARCH_REASONING_EFFORT_DEFAULT: str = "low"
NATIVE_SEARCH_VERBOSITY_ENV: str = "NATIVE_SEARCH_VERBOSITY"
NATIVE_SEARCH_VERBOSITY_DEFAULT: str = "low"
NATIVE_SEARCH_MAX_RESULTS: int = 20  # web options passed to the OpenRouter plugins
NATIVE_SEARCH_CONTEXT_SIZE: str = "high"  # "low", "medium", "high"

# --- Perplexity (fallback research provider; dormant while AskNews wins the ladder) ---

# One constant; the two call sites drifted apart. Receipt: docs/constants.md "PERPLEXITY_RESEARCH_MODEL".
PERPLEXITY_RESEARCH_MODEL: str = "perplexity/sonar-reasoning-pro"
PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER: str = f"openrouter/{PERPLEXITY_RESEARCH_MODEL}"

# Both call sites had NO wall bound at all. Receipt: docs/constants.md "PERPLEXITY_WALL_TIMEOUT".
PERPLEXITY_WALL_TIMEOUT: float = 300.0

# --- Resolution-Source Fetcher (Tier 1) ---

# The char caps here bound RAW fetched content only; LLM-emitted research is never truncated.
RESOLUTION_SOURCE_ENABLED_ENV: str = "RESOLUTION_SOURCE_ENABLED"
RESOLUTION_SOURCE_HTTP_TIMEOUT: float = 20.0  # per-request (probe: 0-2s typical; slack for slow gov sites)
RESOLUTION_SOURCE_WALL_TIMEOUT: float = 45.0  # hard cap on the whole provider
RESOLUTION_SOURCE_MAX_URLS: int = 5  # 58 URLs / 40 Qs ≈ 1.45 avg; bounds pathological multi-URL Qs
RESOLUTION_SOURCE_MAX_RESPONSE_BYTES: int = 5 * 1024 * 1024  # CISA KEV JSON ~1.5 MB; 5 MiB headroom
LOCAL_SOURCE_MAX_EXPANDED_BYTES: int = 20 * 1024 * 1024
LOCAL_SOURCE_MAX_ENTRIES: int = 128
LOCAL_SOURCE_MAX_SHEETS: int = 32
LOCAL_SOURCE_MAX_CELLS: int = 250_000
LOCAL_SOURCE_MAX_CHARS: int = 2_000_000
LOCAL_SOURCE_CACHE_MAX_BYTES: int = 64 * 1024 * 1024
LOCAL_SOURCE_READ_CHUNK_BYTES: int = 64 * 1024
LOCAL_SOURCE_CSV_SNIFF_CHARS: int = 8192
# The elbow of the full-extraction distribution. Receipt: docs/constants.md "RESOLUTION_SOURCE_PER_URL_MAX_CHARS".
RESOLUTION_SOURCE_PER_URL_MAX_CHARS: int = 6000
RESOLUTION_SOURCE_TOTAL_MAX_CHARS: int = 18000
# Above the longest provenance lead. Receipt: docs/constants.md "RESOLUTION_SOURCE_MIN_SECTION_CHARS".
RESOLUTION_SOURCE_MIN_SECTION_CHARS: int = 300
RESOLUTION_SOURCE_JS_WALL_MIN_CHARS: int = 100  # 200-OK with < this extracted text == JS wall (FINDINGS)
RESOLUTION_SOURCE_GLOBAL_CONCURRENCY: int = 5  # TCPConnector limit; per-host serialized separately
# The observed elbow: under this, chrome. Receipt: docs/constants.md "RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS".
RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS: int = 400
# Chrome tops out at 0.329; thinnest content 0.431. Receipt: docs/constants.md "RESOLUTION_SOURCE_CONTENT_SHARE_MIN".
RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS: int = 60
RESOLUTION_SOURCE_CONTENT_SHARE_MIN: float = 0.38
# CPU spent after the body is in hand. Receipt: docs/constants.md "RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S".
RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S: float = 5.0
# --- Inline chart configs (Highcharts), read straight out of the page we already hold ---

# Zero LLM calls and no second request. Receipt: docs/constants.md "Inline chart configs".
RESOLUTION_SOURCE_CHART_MAX_CHARTS: int = 3
RESOLUTION_SOURCE_CHART_MAX_SERIES: int = 4  # IOM's widest chart is 3 series (Undetermined / Female / Male)
# Kept from the END; the resolving value is newest. Receipt: docs/constants.md "RESOLUTION_SOURCE_CHART_MAX_POINTS".
RESOLUTION_SOURCE_CHART_MAX_POINTS: int = 16
# Budgeted out of the per-URL page cap. Receipt: docs/constants.md "RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS".
RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS: int = 2000
# So a pathological page costs fixed work. Receipt: docs/constants.md "RESOLUTION_SOURCE_CHART_MAX_CANDIDATES".
RESOLUTION_SOURCE_CHART_MAX_CANDIDATES: int = 20
RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS: int = 200_000
# --- Datawrapper second hop (Tier 2) ---

# Trackers hide their daily series in iframes trafilatura drops. Receipt: docs/constants.md "Datawrapper second hop".
RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS: int = 3
# One CDN host serializes the datasets. Receipt: docs/constants.md "RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S".
RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S: float = 3.0
RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S: float = 2.0  # so the inner bound fires before the outer wait_for
# Well under the page cap. Receipt: docs/constants.md "RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS".
RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS: int = 3000
# Months-old snapshots came back as HTTP 200. Receipt: docs/constants.md "RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS".
RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS: float = 30.0
# About OUR clock, shared by both guards. Receipt: docs/constants.md "RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE".
RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE: timedelta = timedelta(hours=6)

# --- Local document text (PDFs read with pypdf, `research/document_text.py`) ---

# A PDF we already hold is read locally. Receipt: docs/constants.md "Local document text".
DOCUMENT_TEXT_MAX_PAGES: int = 400
# A BETWEEN-PAGES checkpoint, not an elapsed bound. Receipt: docs/constants.md "DOCUMENT_TEXT_MAX_SECONDS".
DOCUMENT_TEXT_MAX_SECONDS: float = 20.0
DOCUMENT_TEXT_PDF_MAX_BYTES: int = 40 * 1024 * 1024  # ~6x the measured 6.7 MB file; refused before pypdf allocates
# Together ~3.6k chars, one cited page's order. Receipt: docs/constants.md "DOCUMENT_DIGEST_TOP_K".
DOCUMENT_DIGEST_TOP_K: int = 6
DOCUMENT_DIGEST_WINDOW_CHARS: int = 600
# Above this the digest serves it, not a paid read. Receipt: docs/constants.md "URL_CONTEXT_SIZE_GATE_TOKENS".
URL_CONTEXT_SIZE_GATE_TOKENS: int = 100_000

# --- Page digest (`research/page_digest.py`, the `page_digest_extractor` support role) ---

# "Luna is dirt cheap and medium will still be fast enough" (operator). Receipt: docs/constants.md "PAGE_DIGEST_EXTRACTOR_MODEL".
PAGE_DIGEST_EXTRACTOR_MODEL: str = "openrouter/openai/gpt-5.6-luna"
PAGE_DIGEST_EXTRACTOR_EFFORT: str = "medium"
# Luna-medium probed 13-76 s on 20-84k-token SEARCH prompts. Receipt: docs/constants.md "PAGE_DIGEST_EXTRACTOR_TIMEOUT_S".
PAGE_DIGEST_EXTRACTOR_TIMEOUT_S: float = 20.0
# Left to the caller's outer wait_for so the digest returns first. Receipt: docs/constants.md "PAGE_DIGEST_WALL_MARGIN_S".
PAGE_DIGEST_WALL_MARGIN_S: float = 2.0
# Under this a paid call cannot finish, so none is made. Receipt: docs/constants.md "PAGE_DIGEST_MIN_CALL_BUDGET_S".
PAGE_DIGEST_MIN_CALL_BUDGET_S: float = 5.0
# About 4k tokens, a third of the smallest probed prompt. Receipt: docs/constants.md "PAGE_DIGEST_PREFILTER_MAX_CHARS".
PAGE_DIGEST_PREFILTER_MAX_CHARS: int = 16_000

# --- Resolution-source escalation rungs (free ones: meta-refresh hop, local PDF read) ---

# Every rung self-bounds inside the 45 s wall. Receipt: docs/constants.md "RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S".
RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S: float = 2.0
RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S: float = 3.0  # one page GET; the "0-2 s typical" probe basis
# Doubles as extract_pdf_text's minimum max_seconds. Receipt: docs/constants.md "RESOLUTION_SOURCE_PDF_MIN_BUDGET_S".
RESOLUTION_SOURCE_PDF_MIN_BUDGET_S: float = 3.0

# --- Resolution-source escalation rungs that need a browser ---

# Pre-gate floor, under the transport's 15 s need. Receipt: docs/constants.md "RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S".
RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S: float = 12.0
# A CHARACTER count on page.content(), sized to the byte cap. Receipt: docs/constants.md "RENDERED_DOM_MAX_CHARS".
RENDERED_DOM_MAX_CHARS: int = RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
# One GET against an already-probed host. Receipt: docs/constants.md "RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S".
RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S: float = 3.0
# --- The impersonated retry (research/impersonated_fetch.py, shared with gap-fill v2) ---

# A 403 is a TLS fingerprint verdict, recoverable client-side. Receipt: docs/constants.md "The impersonated retry".
RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S: float = 3.0
# ON by default: free and triple-bounded. Receipt: docs/constants.md "RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV".
RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV: str = "RESOLUTION_SOURCE_IMPERSONATE_ENABLED"
# A spent budget still gets a token attempt. Receipt: docs/constants.md "RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S".
RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S: float = 0.5
# Pinned, not the floating "chrome" alias. Receipt: docs/constants.md "IMPERSONATE_BROWSER_TARGET".
IMPERSONATE_BROWSER_TARGET: str = "chrome146"
# --- Wayback Machine snapshots ---

# The one free route whose egress is not ours. Receipt: docs/constants.md "Wayback Machine snapshots".
RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS: float = 30.0
# The archive is slower and adds a redirect hop. Receipt: docs/constants.md "RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S".
RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S: float = 8.0
# One netloc, so Semaphore(1) serializes them. Receipt: docs/constants.md "RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS".
RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS: int = 2
# --- The one PAID rung: Gemini url_context, last and behind its own flag ---

# Default OFF in code, on per workflow yaml since 2026-09-04. Receipt: docs/constants.md "The one PAID rung".
RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV: str = "RESOLUTION_SOURCE_URL_CONTEXT_ENABLED"
# A model round-trip that also fetches. Receipt: docs/constants.md "RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S".
RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S: float = 15.0
# One SDK attempt, against gap-fill v2's two. Receipt: docs/constants.md "RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS".
RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS: int = 1
# Billed READS per question, not attempts. Receipt: docs/constants.md "RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS".
RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS: int = 2
# Tells not_addressed from a challenge page. Receipt: docs/constants.md "RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS".
RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS: int = 300

# --- Gemini Search Provider (Google AI Studio direct SDK) ---

# First-party Google Search grounding, a new index beside OpenRouter's Exa-backed one.
GEMINI_SEARCH_ENABLED_ENV: str = "GEMINI_SEARCH_ENABLED"
GEMINI_SEARCH_MODEL_ENV: str = "GEMINI_SEARCH_MODEL"
# Personal AI Studio key with no donated equivalent. Receipt: docs/constants.md "GOOGLE_API_KEY_ENV".
GOOGLE_API_KEY_ENV: str = "GOOGLE_API_KEY"
# OpenRouter Gemini routing only. Receipt: docs/constants.md "GEMINI_USE_DONATED_OPENROUTER_KEY_ENV".
GEMINI_USE_DONATED_OPENROUTER_KEY_ENV: str = "GEMINI_USE_DONATED_OPENROUTER_KEY"
# Verified live on the native SDK 2026-09-03. Receipt: docs/constants.md "GEMINI_SEARCH_DEFAULT_MODEL".
GEMINI_SEARCH_DEFAULT_MODEL: str = "gemini-3.8-flash"
# 6 min: a 10-round AFC chain takes 150-200 s. Receipt: docs/constants.md "GEMINI_SEARCH_TIMEOUT".
GEMINI_SEARCH_TIMEOUT: int = 360
# Explicit (operator, 2026-09-03); the default is HIGH. Receipt: docs/constants.md "GEMINI_SEARCH_THINKING_LEVEL".
GEMINI_SEARCH_THINKING_LEVEL: str = "medium"
# Per-attempt cap just under the outer wall. Receipt: docs/constants.md "GEMINI_SEARCH_HTTP_TIMEOUT_MS".
GEMINI_SEARCH_HTTP_TIMEOUT_MS: int = 350_000
GEMINI_SEARCH_HTTP_ATTEMPTS: int = 2
# A tier lower (operator, 2026-09-03). Receipt: docs/constants.md "GAP_FILL_V2_READER_THINKING_LEVEL".
GAP_FILL_V2_READER_THINKING_LEVEL: str = "low"
GAP_FILL_V2_READER_HTTP_ATTEMPTS: int = 2

# --- Second-pass gap-fill ---

# Analyzer then parallel resolvers, failing soft to first-pass research alone.
GAP_FILL_ENABLED_ENV: str = "GAP_FILL_ENABLED"
# Non-grounded decomposition under a tight wall. Receipt: docs/constants.md "GAP_FILL_ANALYZER_MODEL".
GAP_FILL_ANALYZER_MODEL: str = "openrouter/openai/gpt-5.6-terra"
# 5 -> 4 on 2026-07-20; do NOT go below 4. Receipt: docs/constants.md "GAP_FILL_MAX_GAPS".
GAP_FILL_MAX_GAPS: int = 4
GAP_FILL_ANALYZER_TIMEOUT: int = 120  # tight, so a hung analyzer cannot hold a research slot
# Headroom over the request timeout. Receipt: docs/constants.md "GAP_FILL_ANALYZER_WALL_TIMEOUT".
GAP_FILL_ANALYZER_WALL_TIMEOUT: int = 135
GAP_FILL_MIN_RESEARCH_CHARS: int = 200  # under this every provider likely soft-failed
# Moved off grounded Gemini 2026-06-25; sol->terra 2026-07-20. Receipt: docs/constants.md "GAP_FILL_RESOLVER_MODEL".
GAP_FILL_RESOLVER_MODEL: str = "openai/gpt-5.6-terra"
GAP_FILL_RESOLVER_REASONING_EFFORT: str = "low"

# --- Agentic gap-fill v2 (bounded research loop) ---

# A bounded loop, concurrent with v1 and soft-failing to "". Receipt: docs/agentic_gap_fill.md.
GAP_FILL_V2_ENABLED_ENV: str = "GAP_FILL_V2_ENABLED"
GAP_FILL_IMAGE_MAX_SOURCE_PIXELS: int = 25_000_000
GAP_FILL_IMAGE_MAX_EDGE: int = 2048
GAP_FILL_IMAGE_MAX_PIXELS: int = 2_000_000
GAP_FILL_IMAGE_MAX_BYTES: int = 2 * 1024 * 1024
GAP_FILL_IMAGE_MAX_VIEWS: int = 4
GAP_FILL_IMAGE_MAX_LEADS: int = 3
GAP_FILL_IMAGE_LEADS_MAX_CHARS: int = 1000
GAP_FILL_IMAGE_METADATA_MAX_CHARS: int = 160
GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS: int = 512
# terra-low won the blind 5-arm replay eval 2026-07-17. Receipt: docs/constants.md "GAP_FILL_V2_DRIVER_MODEL".
GAP_FILL_V2_DRIVER_MODEL: str = os.getenv("GAP_FILL_V2_DRIVER_MODEL") or "openai/gpt-5.6-terra"
GAP_FILL_V2_DRIVER_EFFORT: str = os.getenv("GAP_FILL_V2_DRIVER_EFFORT") or "low"
# A wrong id, or a robots-gated host, kills the rung silently. Receipt: docs/constants.md "GAP_FILL_V2_READER_MODEL".
GAP_FILL_V2_READER_MODEL: str = os.getenv("GAP_FILL_V2_READER_MODEL") or "gemini-3.8-flash"
# Raised with the W2 ambition floor 2026-07-21. Receipt: docs/constants.md "GAP_FILL_V2_MAX_TOOL_CALLS".
GAP_FILL_V2_MAX_TOOL_CALLS: int = _int_env("GAP_FILL_V2_MAX_TOOL_CALLS", 30)
# Inside v1's worst-case envelope. Receipt: docs/constants.md "GAP_FILL_V2_WALL_DEADLINE".
GAP_FILL_V2_WALL_DEADLINE: float = _float_env("GAP_FILL_V2_WALL_DEADLINE", 540.0)
# Under this, only the conclude tool is accepted, forcing a wrap-up inside the wall deadline.
GAP_FILL_V2_CONCLUDE_THRESHOLD: float = _float_env("GAP_FILL_V2_CONCLUDE_THRESHOLD", 90.0)
# The JS-wall heuristic that escalates plain HTTP to headless Chromium; tools.py consumes it.
GAP_FILL_V2_MIN_CONTENT_CHARS: int = _int_env("GAP_FILL_V2_MIN_CONTENT_CHARS", 500)
# Independent of v1's cap; v2 gaps share one tool budget. Receipt: docs/constants.md "GAP_FILL_V2_MAX_GAPS".
GAP_FILL_V2_MAX_GAPS: int = _int_env("GAP_FILL_V2_MAX_GAPS", 4)

# --- Financial Data Provider ---
FINANCIAL_DATA_ENABLED_ENV: str = "FINANCIAL_DATA_ENABLED"
FRED_API_KEY_ENV: str = "FRED_API_KEY"
# Capability-saturated, so the cheapest capable tier. Receipt: docs/constants.md "FINANCIAL_CLASSIFIER_MODEL".
FINANCIAL_CLASSIFIER_MODEL: str = "openrouter/openai/gpt-5.6-luna"
FINANCIAL_CLASSIFIER_TIMEOUT: int = 30
# Never spent as a bare period="Nd". Receipt: docs/constants.md "FINANCIAL_YFINANCE_LOOKBACK_DAYS".
FINANCIAL_YFINANCE_LOOKBACK_DAYS: int = 390
FINANCIAL_YFINANCE_RECENT_DAYS: int = 30
# Screens a vendor-noise-dominated series (q44797). Receipt: docs/constants.md "FINANCIAL_VARIANCE_RATIO_LAG".
FINANCIAL_VARIANCE_RATIO_LAG: int = 5
FINANCIAL_VARIANCE_RATIO_FLOOR: float = 0.6
FINANCIAL_VARIANCE_RATIO_MIN_RETURNS: int = 120
# Revising series resolve on the FIRST print. Receipt: docs/constants.md "FINANCIAL_FRED_VINTAGE_PRINTS".
FINANCIAL_FRED_VINTAGE_PRINTS: int = 4
# Unbounded before, and the executor is shared. Receipt: docs/constants.md "MAX_FINANCIAL_IDENTIFIERS".
MAX_FINANCIAL_IDENTIFIERS: int = 12

# --- SEC EDGAR client (research/sec_edgar.py; standalone, not yet a ladder rung) ---

# Unset means the client refuses to dial: no anonymous User-Agent. Receipt: docs/constants.md "SEC_EDGAR_CONTACT_EMAIL_ENV".
SEC_EDGAR_CONTACT_EMAIL_ENV: str = "SEC_EDGAR_CONTACT_EMAIL"
# SEC's fair-access form is "<company or person> <contact email>". Receipt: docs/constants.md "SEC_EDGAR_USER_AGENT_TEMPLATE".
SEC_EDGAR_USER_AGENT_TEMPLATE: str = "metaculus-bot {contact_email}"
# Under SEC's published 10/s ceiling, with margin for sleep granularity. Receipt: docs/constants.md "SEC_EDGAR_MAX_REQUESTS_PER_SECOND".
SEC_EDGAR_MAX_REQUESTS_PER_SECOND: float = 8.0
# A large filer's companyfacts and an inline-XBRL 10-K both pass 5 MiB. Receipt: docs/constants.md "SEC_EDGAR_MAX_RESPONSE_BYTES".
SEC_EDGAR_MAX_RESPONSE_BYTES: int = 16 * 1024 * 1024

# --- Soft deadlines to keep batch wall-clock inside the tournament cron window ---

# Caps one stuck forecaster at a loud drop. Receipt: docs/constants.md "FORECASTER_SOFT_DEADLINE".
FORECASTER_SOFT_DEADLINE: int = 600

# 1 since 2026-07-20; drops stay CI-visible. Receipt: docs/constants.md "MIN_FORECASTERS_TO_PUBLISH".
MIN_FORECASTERS_TO_PUBLISH: int = 1

# Named because two deadlines are sized against it. Receipt: docs/constants.md "METACULUS_CLOSE_WINDOW_SECONDS".
METACULUS_CLOSE_WINDOW_SECONDS: int = 3600

# The remainder is exactly the stacking budget. Receipt: docs/constants.md "PER_QUESTION_WALL_CLOCK_DEADLINE".
PER_QUESTION_WALL_CLOCK_DEADLINE: int = 3510

# Clears both publish POSTs plus headroom. Receipt: docs/constants.md "WALL_CLOCK_STACKING_MIN_BUDGET".
WALL_CLOCK_STACKING_MIN_BUDGET: int = 90

# --- Close-aware per-question time budget (metaculus_bot/time_budget.py) ---

# Held back so the PREDICTION POST can still land. Receipt: docs/constants.md "PUBLISH_RESERVE_SECONDS".
PUBLISH_RESERVE_SECONDS: int = 60

# Exactly the full pipeline's configured worst case. Receipt: docs/constants.md "TIME_BUDGET_FAST_PATH_THRESHOLD".
TIME_BUDGET_FAST_PATH_THRESHOLD: int = 1815

# Below ~5 minutes nothing lands, so intake forfeits. Receipt: docs/constants.md "TIME_BUDGET_MIN_VIABLE_S".
TIME_BUDGET_MIN_VIABLE_S: int = 300

# ONE fixed window; a rolling share compounds to ~75%. Receipt: docs/constants.md "RESEARCH_PHASE_BUDGET_SHARE".
RESEARCH_PHASE_BUDGET_SHARE: float = 0.5

# Stock forecasting-tools POSTs with no timeout. Receipt: docs/constants.md "PUBLISH_POST_TIMEOUT".
PUBLISH_POST_TIMEOUT: int = 20
PUBLISH_POST_RETRIES: int = 1

# Sized for a CDN/WAF overload that clears in 10-60 s. Receipt: docs/constants.md "FETCH_GET_TIMEOUT".
FETCH_GET_TIMEOUT: int = 60
FETCH_GET_RETRIES: int = 2
FETCH_GET_BACKOFF_BASE: float = 10.0
FETCH_GET_BACKOFF_JITTER: float = 3.0

# Just above the stacker LLM's own litellm timeout. Receipt: docs/constants.md "STACKER_SOFT_DEADLINE".
STACKER_SOFT_DEADLINE: int = 500
STACKER_FALLBACK_SOFT_DEADLINE: int = 300

# The analyzer's own bound is looser than the crux's use. Receipt: docs/constants.md "CRUX_SOFT_DEADLINE".
CRUX_SOFT_DEADLINE: int = 180

# Matches the summarizer's litellm per-request timeout. Receipt: docs/constants.md "SUMMARIZER_WALL_TIMEOUT".
SUMMARIZER_WALL_TIMEOUT: int = 300

# --- Benchmark driver tuning ---
HEARTBEAT_INTERVAL: int = 60
FETCH_RETRY_BACKOFFS: list[int] = [5, 15]
TYPE_MIX: tuple[float, float, float] = (0.5, 0.25, 0.25)  # (binary, numeric, multiple_choice)
FETCH_PACING_SECONDS: int = 2

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
BACKTEST_DEFAULT_RESOLVED_AFTER: str = "2025-12-01"
BACKTEST_DEFAULT_TOURNAMENT: str = "fall-aib-2025"
BACKTEST_DEFAULT_MIN_FORECASTERS: int = 40
BACKTEST_OVERFETCH_RATIO: int = 3
# Saturated backtest-only screen, cheapest capable tier. Receipt: docs/constants.md "LEAKAGE_DETECTOR_MODEL".
LEAKAGE_DETECTOR_MODEL: str = "openrouter/openai/gpt-5.6-luna"

# --- Per-type stacking gates ---

# All three default DISABLED; numeric is evidence-backed. Receipt: docs/constants.md "BINARY_STACKING_ENABLED_ENV".
BINARY_STACKING_ENABLED_ENV: str = "BINARY_STACKING_ENABLED"
MC_STACKING_ENABLED_ENV: str = "MC_STACKING_ENABLED"
NUMERIC_STACKING_ENABLED_ENV: str = "NUMERIC_STACKING_ENABLED"

# --- Prediction-market provider (Workstream G) ---

# Hard-disabled under is_benchmarking. Receipt: docs/constants.md "PREDICTION_MARKETS_ENABLED_ENV".
PREDICTION_MARKETS_ENABLED_ENV: str = "PREDICTION_MARKETS_ENABLED"

# 150 is the ranked pipeline's 131.5 s worst case plus margin. Receipt: docs/constants.md "PREDICTION_MARKET_TIMEOUT".
PREDICTION_MARKET_TIMEOUT: float = float(os.environ.get("PREDICTION_MARKET_TIMEOUT", "150.0"))

# --- Ranked market retrieval (the two LLM stages and the catalogue pull) ---

# The ranker gets NO retry (36/36 parsed first try). Receipt: docs/constants.md "MARKET_QUERY_AUTHOR_WALL_TIMEOUT".
MARKET_QUERY_AUTHOR_WALL_TIMEOUT: float = 20.0
MARKET_QUERY_AUTHOR_BACKOFFS: tuple[float, ...] = (1.0,)
MARKET_RANKER_WALL_TIMEOUT: float = 60.0
MARKET_RANKER_BACKOFFS: tuple[float, ...] = ()

# KALSHI_PAGE_SLEEP_S is measured on the value itself. Receipt: docs/constants.md "KALSHI_CATALOGUE_WALL_TIMEOUT".
KALSHI_CATALOGUE_WALL_TIMEOUT: float = 40.0
KALSHI_PAGE_SLEEP_S: float = 0.25
KALSHI_PREFETCH_EVENT_LIMIT: int = 20_000
KALSHI_PREFETCH_MAX_PAGES: int = 120

# --- Time-Series Anchor Provider (Phase B) ---

# An empirical band, no model selection, backtest-safe. Receipt: docs/constants.md "TS_ANCHOR_ENABLED_ENV".
TS_ANCHOR_ENABLED_ENV: str = "TS_ANCHOR_ENABLED"
# OFF until the text-vs-image A/B; matplotlib is dev-only. Receipt: docs/constants.md "TS_ANCHOR_CHART_ENABLED_ENV".
TS_ANCHOR_CHART_ENABLED_ENV: str = "TS_ANCHOR_CHART_ENABLED"
# Fetches run in to_thread under wait_for; a hung endpoint soft-fails to "".
TS_ANCHOR_TIMEOUT: float = float(os.environ.get("TS_ANCHOR_TIMEOUT", "20.0"))
TS_ANCHOR_HTTP_TIMEOUT: float = 15.0
# The spread window excludes negative WTI (2020-04-20). Receipt: docs/constants.md "TS_ANCHOR_LOOKBACK_YEARS".
TS_ANCHOR_LOOKBACK_YEARS: int = 15
TS_ANCHOR_SPREAD_LOOKBACK_YEARS: int = 5
TS_ANCHOR_SECTION_MAX_CHARS: int = 6000  # self-budgeted, so multi-leg spreads stay bounded
TS_ANCHOR_NATIVE_TABLE_ROWS: int = 10  # last-N native rows; weekly ~3 months, monthly ~2 years
TS_ANCHOR_WEEKLY_TABLE_ROWS: int = 13
TS_ANCHOR_MONTHLY_TABLE_ROWS: int = 24
# Without it the backstop disarms on open bounds. Receipt: docs/constants.md "TS_ANCHOR_OPEN_BOUND_SPAN_TOLERANCE".
TS_ANCHOR_OPEN_BOUND_SPAN_TOLERANCE: float = 0.25

# --- Research persistence (write path for backtest replay) ---
PERSIST_RESEARCH_ENABLED_ENV: str = "PERSIST_RESEARCH_ENABLED"

# --- Raw research-provider payload logging (durable GHA-artifact tape) ---

# Each provider's RAW return as JSONL. Receipt: docs/constants.md "RAW_RESEARCH_LOG_ENABLED_ENV".
RAW_RESEARCH_LOG_ENABLED_ENV: str = "RAW_RESEARCH_LOG_ENABLED"
# run_logs/ is already teed and uploaded wholesale. Receipt: docs/constants.md "RAW_RESEARCH_LOG_DIR_ENV".
RAW_RESEARCH_LOG_DIR_ENV: str = "RAW_RESEARCH_LOG_DIR"
RAW_RESEARCH_LOG_DIR_DEFAULT: str = "run_logs"
# Beyond this a payload becomes a truncation marker. Receipt: docs/constants.md "RAW_RESEARCH_MAX_PAYLOAD_CHARS".
RAW_RESEARCH_MAX_PAYLOAD_CHARS: int = 200_000
