"""Config-era boundaries and the per-record era tags: the one home for both.

Every consumer imports the instants and the tag functions from here, so no two modules can file
the same record into different eras. Each boundary is the merge-to-main COMMITTER timestamp of
the merge that landed the change, never an authoring date, because prod runs from ``main`` and a
config change is live only once its merge lands there. Tags key on ``bot_comment_created_at``,
the submission time, because a question was forecast under whichever config was live when the
bot published its comment. What each merge changed, why the three September merges share one
coarse bucket, and which standing instrument selects on which tag field:
``docs/performance_analysis.md`` "Era boundaries are merge-to-main timestamps, never authoring
dates".
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from metaculus_bot.time_utils import parse_iso_utc

# 0e85e1b: numeric k_tail 1.25 -> 1.0, SPAN_FLOOR_GAMMA 1.0 -> 0.0, binary clamp 0.01/0.99 -> 0.02/0.98.
WIDENING_FLIP_MERGED_AT = datetime(2026, 5, 18, 17, 21, 19, tzinfo=UTC)
# b4e9df0, PR #55, the july15 bundle: the 3-member roster, gap-fill v2, the TS anchor, MIN_FORECASTERS 3->1.
B4E9DF0_MERGED_AT = datetime(2026, 7, 21, 17, 7, 37, tzinfo=UTC)
# 325b1b0, PR #57 (forecasting-tools 0.2.54 -> 0.2.92): the MC option clamp 0.005/0.995 -> 0.01/0.99.
FT_0292_MERGED_AT = datetime(2026, 7, 24, 19, 16, 26, tzinfo=UTC)
# 73e4782, PR #58: the zero-output retry carve-out.
JULY25_MERGED_AT = datetime(2026, 7, 26, 4, 38, 40, tzinfo=UTC)
# c3c91cb, PR #59: a drained donated key falls back to the personal key.
DRY_KEY_FIX_MERGED_AT = datetime(2026, 7, 28, 3, 7, 53, tzinfo=UTC)
# bfd5df2, PR #61: ranked prediction-market retrieval.
RANKED_MARKET_MERGED_AT = datetime(2026, 8, 6, 1, 28, 49, tzinfo=UTC)
# 951f8e4, PR #64: the close-derived time budget.
TIME_BUDGET_MERGED_AT = datetime(2026, 8, 26, 17, 23, 30, tzinfo=UTC)
# eded193, PR #65 (the lint campaign): recorded per record, never opened as a sub-era.
LINTERS_MERGED_AT = datetime(2026, 8, 28, 3, 54, 3, tzinfo=UTC)
# 8d5a082, PR #66, the next-season bundle: prompt de-bloat and the shared fetch ladder.
FALL_CONFIG_MERGED_AT = datetime(2026, 9, 5, 1, 59, 24, tzinfo=UTC)
# a9cbe03, PR #67: the TLS-impersonation fetch rung, pooled into the fall_config bucket.
IMPERSONATE_RUNG_MERGED_AT = datetime(2026, 9, 5, 15, 31, 40, tzinfo=UTC)
# 660fd35, PR #68: the fall tournament target, pooled into the fall_config bucket.
FALL_TARGET_MERGED_AT = datetime(2026, 9, 7, 5, 52, 20, tzinfo=UTC)
# Fall roster and research-role merges: labelled as fine sub-eras, pooled into triple_fall_config for reads.
# 021e059: gpt-5.6-sol -> gpt-6-sol (xhigh), claude-opus-4.8 -> claude-opus-5.5 (first real xhigh).
FALL_GPT6_OPUS55_MERGED_AT = datetime(2026, 9, 24, 20, 50, 53, tzinfo=UTC)
# 6489289, PR #73: gpt-6-sol -> gpt-6.1-sol in every Sol role.
FALL_GPT61_SOL_MERGED_AT = datetime(2026, 9, 29, 21, 36, 57, tzinfo=UTC)
# 28e82b6, PR #77: gemini-3.1-pro-preview -> gemini-4-argon, plus delivered-evidence research publication.
FALL_GEMINI4_ARGON_MERGED_AT = datetime(2026, 10, 14, 15, 14, 9, tzinfo=UTC)
# 2c45ed7, PR #79: gap resolver, gap-fill driver and AskNews summarizer -> gpt-6.1-terra; AskNews-only primary.
FALL_TERRA_RESEARCH_MERGED_AT = datetime(2026, 10, 28, 20, 3, 31, tzinfo=UTC)

# 9f1175c (grid-scaled max-step for discrete CDF resampling) rode b4e9df0: one instant, two gates.
GRID_SCALED_MAX_STEP_MERGED_AT = B4E9DF0_MERGED_AT

# Every merge since the widening flip, tag boundary or not; written into a round's counts_by_era.json.
BOUNDARIES_UTC: dict[str, str] = {
    "flip": WIDENING_FLIP_MERGED_AT.isoformat(),
    "triple_start_b4e9df0": B4E9DF0_MERGED_AT.isoformat(),
    "ft_unfreeze_325b1b0": FT_0292_MERGED_AT.isoformat(),
    "july25_73e4782": JULY25_MERGED_AT.isoformat(),
    "dry_key_fix_c3c91cb": DRY_KEY_FIX_MERGED_AT.isoformat(),
    "ranked_markets_bfd5df2": RANKED_MARKET_MERGED_AT.isoformat(),
    "time_budget_951f8e4": TIME_BUDGET_MERGED_AT.isoformat(),
    "fall_config_8d5a082": FALL_CONFIG_MERGED_AT.isoformat(),
    "impersonate_rung_a9cbe03_inside_fall_config": IMPERSONATE_RUNG_MERGED_AT.isoformat(),
    "fall_target_660fd35_inside_fall_config": FALL_TARGET_MERGED_AT.isoformat(),
    "fall_gpt6_opus55_021e059": FALL_GPT6_OPUS55_MERGED_AT.isoformat(),
    "fall_gpt61_sol_6489289": FALL_GPT61_SOL_MERGED_AT.isoformat(),
    "fall_gemini4_argon_28e82b6": FALL_GEMINI4_ARGON_MERGED_AT.isoformat(),
    "fall_terra_research_2c45ed7": FALL_TERRA_RESEARCH_MERGED_AT.isoformat(),
    "linters_eded193_not_a_boundary": LINTERS_MERGED_AT.isoformat(),
}

SuberaTable = Sequence[tuple[str, datetime]]

# The coarse partition of the triple era; every fall sub-era pools into triple_fall_config.
TRIPLE_SUBERA_TABLE: SuberaTable = (
    ("triple_pre_market", B4E9DF0_MERGED_AT),
    ("triple_ranked_market", RANKED_MARKET_MERGED_AT),
    ("triple_time_budget", TIME_BUDGET_MERGED_AT),
    ("triple_fall_config", FALL_CONFIG_MERGED_AT),
)
# The fine partition, one tag per configuration a resolved question can be compared under.
FINE_SUBERA_TABLE: SuberaTable = (
    ("pre_ft_unfreeze", B4E9DF0_MERGED_AT),
    ("ft_0292", FT_0292_MERGED_AT),
    ("july25", JULY25_MERGED_AT),
    ("post_dry_key_fix", DRY_KEY_FIX_MERGED_AT),
    ("ranked_markets", RANKED_MARKET_MERGED_AT),
    ("time_budget", TIME_BUDGET_MERGED_AT),
    ("fall_config", FALL_CONFIG_MERGED_AT),
    ("fall_gpt6_opus55", FALL_GPT6_OPUS55_MERGED_AT),
    ("fall_gpt61_sol", FALL_GPT61_SOL_MERGED_AT),
    ("fall_gemini4_argon", FALL_GEMINI4_ARGON_MERGED_AT),
    ("fall_terra_research", FALL_TERRA_RESEARCH_MERGED_AT),
    # A new sub-era is one appended row: (<name>, <merge-to-main committer timestamp>).
)

PRE_FLIP = "pre_flip"
POST_FLIP = "post_flip"
TRIPLE_ERA = "triple_era"
NO_TS = "no_ts"
ERAS: tuple[str, ...] = (PRE_FLIP, POST_FLIP, TRIPLE_ERA, NO_TS)
TRIPLE_SUBERAS: tuple[str, ...] = tuple(name for name, _ in TRIPLE_SUBERA_TABLE)
FINE_SUBERAS: tuple[str, ...] = tuple(name for name, _ in FINE_SUBERA_TABLE)
# Sub-eras whose questions were forecast at or after the time-budget merge.
POST_TIME_BUDGET_FINE: tuple[str, ...] = tuple(
    name for name, opens in FINE_SUBERA_TABLE if opens >= TIME_BUDGET_MERGED_AT
)


def era_of(record: dict) -> str:
    """The coarse config era a record was forecast under, written to the ``config_era`` field."""
    dt = parse_iso_utc(record.get("bot_comment_created_at"))
    if dt is None:
        return NO_TS
    if dt >= B4E9DF0_MERGED_AT:
        return TRIPLE_ERA
    if dt >= WIDENING_FLIP_MERGED_AT:
        return POST_FLIP
    return PRE_FLIP


def _subera_of(record: dict, table: SuberaTable) -> str | None:
    """The latest row of ``table`` open at the record's submission time; None outside the triple era."""
    dt = parse_iso_utc(record.get("bot_comment_created_at"))
    if dt is None or dt < B4E9DF0_MERGED_AT:
        return None
    return max((opens, name) for name, opens in table if opens <= dt)[1]


def triple_subera_of(record: dict) -> str | None:
    """The coarse split inside the triple era."""
    return _subera_of(record, TRIPLE_SUBERA_TABLE)


def triple_subera_fine_of(record: dict) -> str | None:
    """The fine split: the field the standing era read selects its arms on."""
    return _subera_of(record, FINE_SUBERA_TABLE)


def ft_unfreeze_side_of(record: dict) -> str | None:
    """Which side of 325b1b0 (forecasting-tools 0.2.92, the MC clamp) a record sits on."""
    dt = parse_iso_utc(record.get("bot_comment_created_at"))
    if dt is None or dt < B4E9DF0_MERGED_AT:
        return None
    return "ft_0292" if dt >= FT_0292_MERGED_AT else "pre_ft_0292"


def post_linters_merge_of(record: dict) -> bool | None:
    """Provenance-only tag: submitted after the eded193 lint campaign, which is NOT a sub-era."""
    dt = parse_iso_utc(record.get("bot_comment_created_at"))
    if dt is None or dt < B4E9DF0_MERGED_AT:
        return None
    return dt >= LINTERS_MERGED_AT
