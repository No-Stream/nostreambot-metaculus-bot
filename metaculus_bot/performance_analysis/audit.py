"""Audit: compare per-model reasoning on questions the bot got wrong.

The standard performance-analysis pipeline scores the ensemble's aggregate and
writes narrative post-mortems against the full bot comment. This module adds
a cut that ranks the *individual ensemble members* on each wrong question so
the per-question reasoning can be diffed — e.g. "gpt-5.2 got closest at 55%,
the ensemble dragged it down to 22%."

Cohort selection ranks on SPOT peer score (``platform_scores``), the metric the
tournament leaderboard actually uses. Coverage-scaled ``peer_score`` is reported beside
it as a labelled secondary and is only ever a ranking key for records that carry no spot
peer, in its own sort tier.

Every ranking here goes through ``ranking_cohort.per_model_ranking_cohort``,
which applies the two exclusions the aggregate cuts in ``analysis.py`` get from
``per_model_cohort`` (stacker-fired records, anonymous positional keys) plus one
this module needs on top: a declared percentile curve too sparse to be scored
against a full one. What a ranking dropped is rendered under the table by
:func:`ranking_caveats`, so a short ranking is a stated exclusion rather than a
silently smaller table.

External human-forecaster comments were the original intent but the Metaculus
``/api/comments/`` endpoint is restricted to the bot's own author id and
staff. As a workaround, this module also looks for manually-curated comments
at ``<audit_dir>/human_comments/<post_id>.md`` and inlines them verbatim.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from metaculus_bot.performance_analysis.collector import load_dataset
from metaculus_bot.performance_analysis.member_replay import (
    MemberScoringInputs,
    member_cdf,
    member_reasoning,
    member_scoring_inputs,
    score_member_cdf,
)
from metaculus_bot.performance_analysis.parsing import (
    MIN_SCOREABLE_ANCHORS,
    _parse_probability,
    declared_anchors,
    parse_per_model_forecasts,
)
from metaculus_bot.performance_analysis.platform_scores import (
    FALLBACK_TIER,
    NO_SCORE_TIER,
    RANKING_FIELDS,
    SPOT_PEER_FIELD,
    log_ranking_score_sources,
    platform_score_fragments,
    ranking_score,
    score_field,
)
from metaculus_bot.performance_analysis.ranking_cohort import (
    PerModelRankingCohort,
    log_ranking_cohort,
    per_model_ranking_cohort,
)
from metaculus_bot.performance_analysis.scoring import brier_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def load_combined_dataset(q1_path: str, q2_path: str) -> list[dict]:
    """Load Q1 and Q2 datasets, dedupe by post_id (Q2 wins) and tag cohort."""
    q1 = load_dataset(q1_path)
    q2 = load_dataset(q2_path)
    q2_pids = {r["post_id"] for r in q2}

    merged: dict[int, dict] = {r["post_id"]: dict(r) for r in q1}
    for r in q2:
        merged[r["post_id"]] = dict(r)

    for pid, rec in merged.items():
        rec["_cohort"] = "Q2" if pid in q2_pids else "Q1"
    return list(merged.values())


def _has_rankable_score(r: dict, fallback_field: str) -> bool:
    return ranking_score(r) is not None or r.get(fallback_field) is not None


def _filter_by_type(records: list[dict], q_types: tuple[str, ...], fallback_field: str) -> list[dict]:
    return [r for r in records if r.get("type") in q_types and _has_rankable_score(r, fallback_field)]


# Brier is a LOSS (higher = worse); both log scores are GAINS (higher = better).
# A field absent here raises rather than silently defaulting to a direction.
_FALLBACK_HIGHER_IS_BETTER: dict[str, bool] = {
    "brier_score": False,
    "numeric_log_score": True,
    "mc_log_score": True,
}


def _ranking_goodness(r: dict, fallback_field: str) -> tuple[int, float]:
    """``(tier, goodness)`` where higher goodness is always BETTER within its tier."""
    ranked = ranking_score(r)
    if ranked is not None:
        return (ranked.tier, ranked.value)
    value = r.get(fallback_field)
    if value is None:
        return (NO_SCORE_TIER, 0.0)
    return (FALLBACK_TIER, value if _FALLBACK_HIGHER_IS_BETTER[fallback_field] else -value)


def _rank_key(r: dict, fallback_field: str, *, best: bool) -> tuple[int, float]:
    """Sort key: ascending sort yields worst-first, or best-first with ``best=True``."""
    tier, goodness = _ranking_goodness(r, fallback_field)
    return (tier, -goodness if best else goodness)


def _banding_field(records: list[dict]) -> str | None:
    """Which platform score the whole cohort bands on: spot whenever any record carries it."""
    for field in RANKING_FIELDS:
        if any(score_field(r, field) is not None for r in records):
            return field
    return None


def _middle_band(records: list[dict]) -> list[dict]:
    """Return the records in the 20-80 percentile band of the platform peer score.

    Banded on ONE field across the whole cohort — spot peer whenever any record carries
    it, coverage-scaled peer only when none does. Mixing the two would band a set whose
    order is partly submission timing rather than skill (see ``platform_scores``).

    Records without the banding field are dropped — middle-mode is peer-anchored only.
    That is intentional: the cohort is meant to surface "ordinary questions" relative to
    the crowd, which requires peer comparability.
    """
    field = _banding_field(records)
    if field is None:
        return []
    if field != SPOT_PEER_FIELD:
        logger.warning(
            f"MIDDLE_BAND_PEER_FALLBACK: banding on {field}; the band's own 20/80 edges are set "
            "by coverage-scaled values, not just the records' sort tier "
            "(see PLATFORM_RANKING_SOURCE above)"
        )
    ranked = sorted(
        (r for r in records if score_field(r, field) is not None),
        key=lambda r: score_field(r, field) or 0.0,
    )
    n = len(ranked)
    dropped = len(records) - n
    if dropped:
        logger.warning(
            f"MIDDLE_BAND_DROPPED_UNSCORED: {dropped} record(s) carry no {field}; middle mode is "
            "peer-anchored, so a record with no banding score has no 'middle' and is excluded "
            "from the cohort rather than tiered behind it"
        )
    lower_idx = int(0.2 * n)
    upper_idx = max(lower_idx, int(0.8 * n))
    return ranked[lower_idx:upper_idx]


def select_cohort(
    records: list[dict],
    mode: Literal["worst", "best", "middle"],
    *,
    n_binary: int = 10,
    n_numeric: int = 5,
    n_mc: int = 2,
    extra_post_ids: list[int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Pick records per question type, ranked by SPOT peer score under one of three modes.

    Peer score compares our log score to the crowd's mean log score (negative = worse
    than peers). ``metaculus_scores.spot_peer_score`` is the canonical ranking signal:
    it's comparable across question types, accounts for question difficulty, AND it is
    the metric the tournament leaderboard ranks on. Coverage-scaled ``peer_score`` is a
    strictly worse ranking key and is used only for records that carry no spot peer, in
    a separate sort tier so the two are never interleaved — see ``platform_scores``.

    Modes:
    - ``"worst"``: spot peer ascending (most negative = worst). Fallback: highest
      Brier (binary) / lowest log score (numeric, MC).
    - ``"best"``: spot peer descending (most positive = best). Fallback: lowest
      Brier / highest log score.
    - ``"middle"``: rank by spot peer, drop the bottom 20% and top 20% as
      extremes, then ``random.Random(seed).sample(...)`` to draw N per type.
      Middle 60% (not pure random) so the cohort surfaces "ordinary"
      questions; pure random risks repeatedly drawing near-extremes that
      defeat the purpose. Records with no platform peer score at all are dropped
      from middle mode (no peer comparability → no "middle").

    Records missing both a platform peer score and the type-specific fallback are
    dropped (worst/best modes). Middle mode additionally drops records without a
    platform peer score regardless of fallback availability.

    ``extra_post_ids`` unions a manually-curated list of post_ids onto the
    selection (deduplicated, appended in input order). Useful for spot_peer
    outliers or known-interesting questions outside the auto-selected top-N.

    Returns concatenated list in (binary, numeric, mc, extras) order.
    """
    log_ranking_score_sources(records, cut=f"select_cohort:{mode}")
    if mode in ("worst", "best"):
        best = mode == "best"
        binary_pool = _filter_by_type(records, ("binary",), "brier_score")
        numeric_pool = _filter_by_type(records, ("numeric", "discrete"), "numeric_log_score")
        mc_pool = _filter_by_type(records, ("multiple_choice",), "mc_log_score")
        sel_binary = sorted(binary_pool, key=lambda r: _rank_key(r, "brier_score", best=best))[:n_binary]
        sel_numeric = sorted(numeric_pool, key=lambda r: _rank_key(r, "numeric_log_score", best=best))[:n_numeric]
        sel_mc = sorted(mc_pool, key=lambda r: _rank_key(r, "mc_log_score", best=best))[:n_mc]
    elif mode == "middle":
        rng = random.Random(seed)  # noqa: S311  # deterministic audit sampling, not cryptographic
        middle = _middle_band(records)
        binary_pool = [r for r in middle if r.get("type") == "binary"]
        numeric_pool = [r for r in middle if r.get("type") in ("numeric", "discrete")]
        mc_pool = [r for r in middle if r.get("type") == "multiple_choice"]
        sel_binary = rng.sample(
            binary_pool, min(n_binary, len(binary_pool))
        )  # HARNESS-SCAN-EXEMPT-subsampling  # seeded audit-cohort pick is this function's purpose
        sel_numeric = rng.sample(
            numeric_pool, min(n_numeric, len(numeric_pool))
        )  # HARNESS-SCAN-EXEMPT-subsampling  # seeded audit-cohort pick is this function's purpose
        sel_mc = rng.sample(
            mc_pool, min(n_mc, len(mc_pool))
        )  # HARNESS-SCAN-EXEMPT-subsampling  # seeded audit-cohort pick is this function's purpose
    else:
        raise ValueError(  # type: ignore[reportUnreachable]
            f"Unknown mode {mode!r}; expected 'worst', 'best', or 'middle'"
        )

    selected = sel_binary + sel_numeric + sel_mc
    if extra_post_ids:
        seen = {r["post_id"] for r in selected}
        by_pid = {r["post_id"]: r for r in records}
        for pid in extra_post_ids:
            if pid in seen:
                continue
            rec = by_pid.get(pid)
            if rec is None:
                logger.warning(f"extra_post_ids includes {pid} but no matching record — skipped")
                continue
            selected.append(rec)
            seen.add(pid)
    return selected


def select_worst_misses(
    records: list[dict],
    *,
    n_binary: int = 10,
    n_numeric: int = 5,
    n_mc: int = 2,
    extra_post_ids: list[int] | None = None,
) -> list[dict]:
    """Backward-compat alias for ``select_cohort(mode='worst', ...)``.

    Preserved so the April audit driver script (``scratch/audit_2026-04/run_audit.py``)
    keeps working unchanged. Identical behavior to the worst-mode path of
    ``select_cohort``.
    """
    return select_cohort(
        records,
        mode="worst",
        n_binary=n_binary,
        n_numeric=n_numeric,
        n_mc=n_mc,
        extra_post_ids=extra_post_ids,
    )


# ---------------------------------------------------------------------------
# Per-model accuracy ranking
# ---------------------------------------------------------------------------


def rank_our_models_by_accuracy(record: dict) -> list[dict]:
    """Rank the ensemble members on a single record by per-model score.

    Binary: each ``{model: percentage_string}`` entry from
    ``per_model_forecasts`` is parsed to a probability and scored with Brier
    (lower is better). Sort is **ascending** by Brier (best first).

    Numeric / discrete: each ``{model: [(percentile, value), ...]}`` entry
    from ``per_model_numeric_percentiles`` is converted to a 201-point CDF
    via PCHIP, then scored with Metaculus-style ``numeric_log_score`` (higher
    is better). Sort is **descending** by score (best first). The return
    semantics differ between binary and numeric — both list "best first" but
    via different score directions; downstream consumers should treat the
    list as ordered, not the score values directly.

    For numeric / discrete, **we score the raw model output — pre tail-
    widening, pre validation**. This isolates *model judgment* from
    *post-processing* effects. The bot's actual published CDF goes through
    additional ``numeric_pipeline.py`` post-processing (tail widening, etc.)
    which is intentionally excluded from this audit-side scoring.

    Returns a list of dicts. Binary entries: ``{model, prob, score, raw}``.
    Numeric/discrete entries: ``{model, percentiles, score, raw, n_anchors}``
    where ``raw`` is a short summary like ``"P10≈X P50≈Y P90≈Z"`` and
    ``n_anchors`` is the number of distinct declared percentile labels the score
    was built from.

    Only entries in the record's per-model cohort are ranked — see
    :func:`per_model_ranking_cohort`, which drops stacker aggregates, anonymous
    positional keys, and density-mismatched percentile curves.

    Empty list when:
    - Type isn't binary/numeric/discrete.
    - Resolution isn't usable (None binary; non-recognized numeric resolution).
    - Numeric scaling bounds are missing.
    - No parseable per-model entries.
    - The record is out of the per-model cohort (stacker-fired, or every entry
      was an anonymous key / a too-sparse curve).
    """
    q_type = record.get("type")
    if q_type == "binary":
        return _rank_binary(record)
    if q_type in ("numeric", "discrete"):
        return _rank_numeric(record)
    return []


def _rank_binary(record: dict) -> list[dict]:
    resolution = record.get("resolution_parsed")
    if not isinstance(resolution, bool):
        return []
    cohort = per_model_ranking_cohort(record)
    log_ranking_cohort(record, cohort, cut="audit_rank_binary")

    ranked: list[dict] = []
    for model, raw in cohort.entries.items():
        prob = _parse_probability(raw)
        if prob is None:
            logger.debug(f"Skipping unparseable per-model value: {model}={raw!r}")
            continue
        ranked.append(
            {
                "model": model,
                "prob": prob,
                "score": brier_score(prob, resolution),
                "raw": raw,
            }
        )
    ranked.sort(key=lambda d: d["score"])
    return ranked


def _score_member_curve(
    model: str,
    percentile_pairs: list,
    inputs: MemberScoringInputs,
    *,
    post_id: object,
    reasoning: str = "",
) -> dict | None:
    """Rebuild one member's CDF on the record's own grid and log-score it.

    None means the member could not be scored (PCHIP or scoring failure), which the
    caller counts toward the degraded-ranking warning.
    """
    percentile_dict, conflicts = declared_anchors(percentile_pairs)
    if conflicts:
        logger.warning(
            f"Conflicting percentile restatement post={post_id} model={model}: "
            f"{conflicts} label(s) declared twice with different values; the last value is scored"
        )
    try:
        cdf = member_cdf(percentile_pairs, inputs, reasoning=reasoning)
    except (ValueError, RuntimeError) as exc:
        logger.warning(f"Per-model PCHIP failure post={post_id} model={model}: {exc}")
        return None

    try:
        score = score_member_cdf(cdf, inputs)
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning(f"Per-model scoring failure post={post_id} model={model}: {exc}")
        return None

    return {
        "model": model,
        "percentiles": percentile_pairs,
        "score": score,
        "raw": _summarize_percentiles(percentile_dict),
        "n_anchors": len(percentile_dict),
    }


def _rank_numeric(record: dict) -> list[dict]:
    inputs = member_scoring_inputs(record)
    if inputs is None:
        return []

    cohort = per_model_ranking_cohort(record)
    log_ranking_cohort(record, cohort, cut="audit_rank_numeric")
    per_model = cohort.entries
    post_id = record.get("post_id")

    ranked: list[dict] = []
    skipped_count = 0
    for model, percentile_pairs in per_model.items():
        entry = _score_member_curve(
            model, percentile_pairs, inputs, post_id=post_id, reasoning=member_reasoning(record, model)
        )
        if entry is None:
            skipped_count += 1
        else:
            ranked.append(entry)

    # If the majority of models failed, the audit output for this question
    # would be misleading (empty or near-empty ranking) — surface loudly
    # rather than silently produce a degraded report.
    if per_model and skipped_count >= len(per_model) / 2:
        logger.error(
            f"Per-model numeric scoring degraded post={post_id}: "
            f"{skipped_count}/{len(per_model)} models failed; ranking unreliable"
        )

    ranked.sort(key=lambda d: -d["score"])
    return ranked


def _summarize_percentiles(percentile_dict: dict[float, float]) -> str:
    """Render a tiny "P10≈X P50≈Y P90≈Z" preview for the audit Markdown."""
    parts: list[str] = []
    for target in (10, 50, 90):
        nearest_key = min(percentile_dict.keys(), key=lambda k: abs(k - target))
        parts.append(f"P{target}≈{percentile_dict[nearest_key]:g}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------


def _format_score_header(record: dict) -> str:
    q_type = record["type"]
    # Spot peer leads and is bolded; peer follows with its scaling named. The tournament
    # ranks on spot peer, and the two differ enough on the same record (q44872: -38.8 vs
    # -15.0) that an unlabelled peer figure reads as a much smaller miss than it was.
    parts: list[str] = platform_score_fragments(record)
    if q_type == "binary":
        b = record.get("brier_score")
        log = record.get("log_score")
        if b is not None:
            parts.append(f"Brier {b:.3f}")
        if log is not None:
            parts.append(f"log {log:.2f}")
    elif q_type in ("numeric", "discrete"):
        nlog = record.get("numeric_log_score")
        if nlog is not None:
            parts.append(f"numeric log {nlog:.2f}")
    elif q_type == "multiple_choice":
        mlog = record.get("mc_log_score")
        if mlog is not None:
            parts.append(f"MC log {mlog:.2f}")
    return " | ".join(parts) if parts else "no score"


def _format_our_prediction(record: dict) -> str:
    q_type = record["type"]
    if q_type == "binary":
        p = record.get("our_prob_yes")
        return f"P(YES)={p * 100:.1f}%" if p is not None else "unknown"
    if q_type == "multiple_choice":
        options = record.get("options") or []
        fvs = record.get("our_forecast_values") or []
        if options and len(fvs) == len(options):
            pairs = sorted(zip(options, fvs, strict=False), key=lambda t: -t[1])
            top3 = pairs[:3]  # HARNESS-SCAN-EXEMPT-subsampling  # display truncation
            return ", ".join(f"{o}={p * 100:.1f}%" for o, p in top3) + (", ..." if len(pairs) > 3 else "")
        return str(fvs)[:80]
    if q_type in ("numeric", "discrete"):
        fvs = record.get("our_forecast_values") or []
        if not fvs:
            return "unknown"
        # Show rough 10/50/90 from CDF if it's a 201-point CDF.
        if len(fvs) >= 201:
            return f"P10≈{fvs[20]:.3f}  P50≈{fvs[100]:.3f}  P90≈{fvs[180]:.3f}  (as CDF prob)"
        return str(fvs)[:80]
    return "unknown"


def _format_resolution(record: dict) -> str:
    raw = record.get("resolution_raw", "?")
    parsed = record.get("resolution_parsed")
    if isinstance(parsed, bool):
        return "YES" if parsed else "NO"
    if parsed is None:
        return str(raw)
    return f"{parsed} (raw: {raw})"


def ranking_caveats(cohort: PerModelRankingCohort) -> list[str]:
    """Markdown lines stating what a ranking excluded, one per exclusion kind.

    An excluded row is invisible in the table, so a dossier that does not say
    what was dropped reads as a complete ranking of a thinner ensemble.
    """
    notes: list[str] = []
    if cohort.stacker_fired:
        notes.append(
            "_Not ranked: the stacker produced this question's published value, so the summary "
            "bullet holds the stacker's aggregate rather than any single member's forecast. "
            "Read member dissent from the `## Base Model Reasoning` sub-blocks instead._"
        )
    if cohort.anonymous_keys:
        notes.append(
            f"_Excluded {len(cohort.anonymous_keys)} unattributed bullet(s) "
            f"({', '.join(cohort.anonymous_keys)}): a positional key is not a model — on a "
            "stacking-era comment that slot holds the stacker's own aggregate._"
        )
    if cohort.sparse_anchors:
        detail = ", ".join(f"{model} ({n} anchors)" for model, n in sorted(cohort.sparse_anchors.items()))
        notes.append(
            f"_Excluded from scoring — insufficient anchors (fewer than {MIN_SCOREABLE_ANCHORS} "
            f"declared percentiles to rebuild the distribution): {detail}. Their siblings' scores "
            "stand; a sparse curve PCHIP'd onto the full grid scores tens of log points off._"
        )
    if cohort.sparse_era:
        anchor_counts = sorted({len(declared_anchors(pairs)[0]) for pairs in cohort.entries.values()})
        counts = "/".join(str(n) for n in anchor_counts)
        notes.append(
            f"_Sparse-era question: every member declared only {counts} percentiles, so the "
            "ranking compares equals and is kept — but the absolute log scores are not comparable "
            "with a question whose members declared the full set._"
        )
    return notes


def _format_model_ranking(ranked: list[dict], record: dict) -> str:
    """Render the per-model ranking table, followed by the exclusions behind it."""
    caveats = ranking_caveats(per_model_ranking_cohort(record))
    if not ranked:
        return "\n".join(caveats or ["_No per-model forecasts parsed from the comment._"]) + "\n"

    if str(record.get("type")) in ("numeric", "discrete"):
        lines = [
            "| rank | model | declared | anchors | log score (higher = better) |",
            "|---|---|---|---|---|",
        ]
        for i, r in enumerate(ranked, 1):
            lines.append(f"| {i} | {r['model']} | {r['raw']} | {r['n_anchors']} | {r['score']:+.2f} |")
    else:
        lines = [
            "| rank | model | forecast | Brier (lower = better) | delta vs ensemble |",
            "|---|---|---|---|---|",
        ]
        our = record.get("our_prob_yes")
        for i, r in enumerate(ranked, 1):
            delta = ""
            if our is not None:
                delta_pp = (r["prob"] - our) * 100.0
                delta = f"{delta_pp:+.1f}pp"
            lines.append(f"| {i} | {r['model']} | {r['raw']} | {r['score']:.3f} | {delta} |")
    if caveats:
        lines.extend(["", *caveats])
    return "\n".join(lines) + "\n"


EXTERNAL_COMMENTS_DIRNAME = "external_forecaster_comments"


def _load_external_comments(post_id: int, audit_dir: Path) -> str | None:
    """Return the body of ``<audit_dir>/external_forecaster_comments/<post_id>.md`` if present.

    "External" means any forecaster other than our own bot — other competing
    bots and a smaller number of humans. Metaculus's ``/api/comments/``
    endpoint is restricted to the bot's own author id, so these comments are
    curated manually (pasted from the web UI) rather than fetched.
    """
    path = audit_dir / EXTERNAL_COMMENTS_DIRNAME / f"{post_id}.md"
    if not path.exists():
        return None
    text = path.read_text().strip()
    # Skip if file is still the placeholder stub (only a header, no body).
    if text.count("\n") < 2 or text.startswith("<!-- PLACEHOLDER"):
        return None
    return text


def _crowd_size_text(meta: dict) -> str:
    """Render ``metadata.nr_forecasters`` as three states, not two.

    Absent or None means the collector never read a crowd size (the post carried no
    field). A literal 0 is the pre-2026-08-25 sentinel: every record pulled before the
    writer fix carries 0 because the collector read the field off the question dict,
    which never carries it. A genuine zero crowd cannot occur on a question this bot
    forecast on — the bot's own forecast counts — so a bare "0" would state an empty
    crowd nobody measured. The sentinel renders as unknown, with the reason, so it stays
    distinguishable from both n/a and a real count.
    """
    crowd = meta.get("nr_forecasters")
    if crowd == 0:
        return "unknown (0 = pre-2026-08-25 record; the field was never read)"
    if crowd is None:
        return "n/a"
    return str(crowd)


def _miss_header_lines(record: dict) -> list[str]:
    """The question's identity, resolution, our forecast and its score."""
    post_id = record["post_id"]
    meta = record.get("metadata") or {}
    was_stacked = record.get("was_stacked")

    lines = [
        f"# {record.get('title', '(no title)')}\n",
        f"- **Metaculus**: https://www.metaculus.com/questions/{post_id}/",
        f"- **post_id**: {post_id}  |  **cohort**: {record.get('_cohort', '?')}  |  **type**: {record['type']}",
        f"- **Resolved**: {_format_resolution(record)}",
        f"- **Our prediction**: {_format_our_prediction(record)}",
        f"- **Score**: {_format_score_header(record)}",
        f"- **nr_forecasters**: {_crowd_size_text(meta)}",
        f"- **was_stacked**: {'unknown' if was_stacked is None else was_stacked}",
    ]
    if meta.get("category"):
        lines.append(f"- **category**: {meta['category']}")
    lines.append("")
    return lines


def _reasoning_order(ranked_models: list[dict], per_model_reasoning: dict[str, str]) -> list[str]:
    """Models best-first by rank, then any reasoning-only member appended.

    A reasoning-only member is MC (no per-member ranker), an unparseable value, or a
    member the ranking cohort excluded (anonymous key, too-sparse curve). Their prose
    is still worth reading — only their SCORE was untrustworthy.
    """
    if not ranked_models:
        return list(per_model_reasoning.keys())
    ordered = [r["model"] for r in ranked_models if r["model"] in per_model_reasoning]
    ordered.extend(m for m in per_model_reasoning if m not in ordered)
    return ordered


def _per_model_reasoning_lines(
    record: dict,
    ranked_models: list[dict],
    per_model_reasoning: dict[str, str],
) -> list[str]:
    """Each member's rationale, headed by its forecast and score when it was ranked.

    Opens with a truncation warning when a summary bullet lost its rationale. That read
    deliberately stays OUTSIDE the ranking cohort: it asks "did every bullet keep its
    rationale", which needs every bullet — filtering it would hide a trimmed rationale
    behind an exclusion.
    """
    lines: list[str] = []
    summary_models = set(parse_per_model_forecasts(record.get("comment_text") or "").keys())
    missing = summary_models - set(per_model_reasoning.keys())
    if missing:
        lines.append(f"> ⚠️ Reasoning missing for {sorted(missing)} — comment likely trimmed to COMMENT_CHAR_LIMIT.")
        lines.append("")

    lines.append("## Per-model reasoning")
    lines.append("")

    score_label = "log score" if record["type"] in ("numeric", "discrete") else "Brier"
    scored_by_model = {r["model"]: r for r in ranked_models}
    for model in _reasoning_order(ranked_models, per_model_reasoning):
        scored = scored_by_model.get(model)
        forecast_str = (
            f" — forecast **{scored['raw']}** ({score_label} {scored['score']:.3f})" if scored is not None else ""
        )
        lines.append(f"### {model}{forecast_str}\n")
        lines.append(per_model_reasoning.get(model, ""))
        lines.append("")
    return lines


def emit_miss_markdown(
    record: dict,
    ranked_models: list[dict],
    per_model_reasoning: dict[str, str],
    *,
    audit_dir: Path,
    out_path: Path,
) -> None:
    """Write a per-question audit markdown file."""
    lines = _miss_header_lines(record)

    # Binary (Brier) and numeric/discrete (log score off each member's rebuilt
    # CDF) are ranked; MC has no per-member ranker yet.
    lines.append("## Per-model accuracy ranking")
    lines.append("")
    lines.append(_format_model_ranking(ranked_models, record))

    lines.extend(_per_model_reasoning_lines(record, ranked_models, per_model_reasoning))

    # Optional external-forecaster-comment inclusion (other bots + humans,
    # curated manually since the comments API is restricted).
    external = _load_external_comments(record["post_id"], audit_dir)
    if external is not None:
        lines.append("## External forecaster comments (other bots + humans, manually curated)")
        lines.append("")
        lines.append(external)
        lines.append("")

    lines.append("## Diff notes")
    lines.append("")
    lines.append("<!-- Add cross-model and bot-vs-external observations during synthesis. -->")
    lines.append("")

    out_path.write_text("\n".join(lines))


def emit_external_comment_stub(record: dict, out_path: Path) -> None:
    """Write a placeholder external_forecaster_comments/<post_id>.md if none exists."""
    if out_path.exists():
        return
    post_id = record["post_id"]
    title = record.get("title", "(no title)")
    url = f"https://www.metaculus.com/questions/{post_id}/"
    resolution = _format_resolution(record)
    out_path.write_text(
        f"<!-- PLACEHOLDER: paste top comments from other bots / humans below this line, then delete this marker. -->\n\n"
        f"# {title}\n\n"
        f"- URL: {url}\n"
        f"- Resolved: {resolution}\n\n"
        f"## Comments\n\n"
        f"_Paste 2-3 top comments here (usually other bots, occasionally humans)._\n"
    )


# ---------------------------------------------------------------------------
# Synthesis + combined report
# ---------------------------------------------------------------------------


_DEFAULT_SYNTHESIS_FRAMING: dict[str, str] = {
    "title": "Audit synthesis — bot misses vs per-model dissent",
    "intro_paragraph": (
        "For each worst-scored question, which ensemble members were closest to the truth, "
        "and how far did the ensemble aggregate sit from the best member?"
    ),
    "cohort_name": "binary worst misses",
    "tally_best_label": "times best",
    "tally_worst_label": "times worst",
    "delta_section_label": "Ensemble-vs-best delta (binary, sorted by worst regret)",
    "spread_section_label": "High-spread misses (dissenting-model signal)",
}


@dataclass(frozen=True, slots=True)
class _BinaryRankingTallies:
    """Cross-question counts and deltas read off the binary per-model rankings."""

    binary_count: int
    best_model_tally: dict[str, int]
    avoided_model_tally: dict[str, int]
    # (post_id, best model, ensemble Brier, best-model Brier, ensemble - best)
    ensemble_vs_best_deltas: list[tuple[int, str, float, float, float]]
    missing_ensemble_brier: list[int]


def _binary_ranking_tallies(records_with_rankings: list[dict]) -> _BinaryRankingTallies:
    """Who finished best/worst per question, and how much the aggregate gave up.

    A record with no ensemble Brier is NAMED as omitted rather than given a delta. The
    old fallback rendered the missing score as 0.000 — a PERFECT ensemble — with delta
    +0.000, which reads as "aggregating cost nothing" on a record that was never scored.
    """
    best_model_tally: dict[str, int] = {}
    avoided_model_tally: dict[str, int] = {}
    binary_count = 0
    ensemble_vs_best_deltas: list[tuple[int, str, float, float, float]] = []
    missing_ensemble_brier: list[int] = []

    for entry in records_with_rankings:
        rec = entry["record"]
        ranked = entry["ranked"]
        if rec.get("type") != "binary" or not ranked:
            continue
        binary_count += 1
        best = ranked[0]
        worst = ranked[-1]
        best_model_tally[best["model"]] = best_model_tally.get(best["model"], 0) + 1
        avoided_model_tally[worst["model"]] = avoided_model_tally.get(worst["model"], 0) + 1
        if rec.get("our_prob_yes") is None:
            continue
        ensemble_brier = rec.get("brier_score")
        if ensemble_brier is None:
            missing_ensemble_brier.append(rec["post_id"])
            continue
        ensemble_vs_best_deltas.append(
            (rec["post_id"], best["model"], ensemble_brier, best["score"], ensemble_brier - best["score"])
        )

    return _BinaryRankingTallies(
        binary_count=binary_count,
        best_model_tally=best_model_tally,
        avoided_model_tally=avoided_model_tally,
        ensemble_vs_best_deltas=ensemble_vs_best_deltas,
        missing_ensemble_brier=missing_ensemble_brier,
    )


def _closest_to_truth_lines(tallies: _BinaryRankingTallies, framing: dict[str, str]) -> list[str]:
    """Per-model best/worst finish counts, most-often-best first."""
    if not tallies.best_model_tally:
        return []

    lines = [
        f"## Closest-to-truth tally ({framing['cohort_name']})\n",
        f"| model | {framing['tally_best_label']} | {framing['tally_worst_label']} |",
        "|---|---|---|",
    ]
    all_models = sorted(set(tallies.best_model_tally) | set(tallies.avoided_model_tally))
    lines.extend(
        f"| {m} | {tallies.best_model_tally.get(m, 0)} | {tallies.avoided_model_tally.get(m, 0)} |"
        for m in sorted(all_models, key=lambda k: -tallies.best_model_tally.get(k, 0))
    )
    lines.append("")
    return lines


def _ensemble_vs_best_lines(tallies: _BinaryRankingTallies, framing: dict[str, str]) -> list[str]:
    """What aggregating cost against the best member, worst regret first."""
    lines: list[str] = []
    if tallies.ensemble_vs_best_deltas:
        lines.append(f"## {framing['delta_section_label']}\n")
        lines.append("| post | best model | ensemble Brier | best-model Brier | Δ (lost by aggregating) |")
        lines.append("|---|---|---|---|---|")
        lines.extend(
            f"| {pid} | {best_m} | {ens:.3f} | {best:.3f} | {delta:+.3f} |"
            for pid, best_m, ens, best, delta in sorted(tallies.ensemble_vs_best_deltas, key=lambda t: -t[4])
        )
        lines.append("")
    if tallies.missing_ensemble_brier:
        lines.append(
            f"_Omitted from the delta table — no ensemble Brier on the record, so there is no delta: "
            f"{sorted(tallies.missing_ensemble_brier)}._\n"
        )
    return lines


def _high_spread_lines(records_with_rankings: list[dict], framing: dict[str, str]) -> list[str]:
    """The ten widest per-model disagreements — the stacking candidates.

    Binary only: "spread" here is max - min over the per-model probabilities (a number
    in [0, 1]). Numeric/discrete entries produced by ``_rank_numeric`` carry
    ``percentiles`` rather than ``prob``, and a percentile-based spread isn't directly
    comparable, so they are skipped rather than given a number the reader would misread.
    """
    high_spread_miss = []
    for entry in records_with_rankings:
        rec = entry["record"]
        ranked = entry["ranked"]
        if rec.get("type") != "binary" or len(ranked) < 2:
            continue
        probs = [r["prob"] for r in ranked]
        high_spread_miss.append(
            (
                rec["post_id"],
                rec.get("title", "")[:60],
                max(probs) - min(probs),
                rec.get("brier_score"),
                rec.get("was_stacked"),
            )
        )

    if not high_spread_miss:
        return []

    lines = [
        f"## {framing['spread_section_label']}\n",
        "| post | title | best-vs-worst spread | Brier | was_stacked |",
        "|---|---|---|---|---|",
    ]
    for pid, t, spread, brier, stacked in sorted(high_spread_miss, key=lambda x: -x[2])[:10]:
        stacked_text = "unknown" if stacked is None else str(stacked)
        brier_text = f"{brier:.3f}" if brier is not None else "n/a"
        lines.append(f"| {pid} | {t} | {spread * 100:.1f}pp | {brier_text} | {stacked_text} |")
    lines.append("")
    return lines


def emit_synthesis(
    records_with_rankings: list[dict],
    out_path: Path,
    framing: dict[str, str] | None = None,
) -> None:
    """Write a cross-cutting synthesis of model-level patterns across all misses.

    ``framing`` lets callers override the misses-cohort defaults so the same
    function can emit synthesis files for hits or middle cohorts (where labels
    like "times best / times worst" or "High-spread misses" no longer fit).
    Missing keys fall back to defaults; ``framing=None`` is identical to the
    pre-parameterization behavior.

    The tallies count whatever ``entry["ranked"]`` holds, so they inherit the
    per-model cohort applied by :func:`rank_our_models_by_accuracy`: a
    stacker-fired record contributes no ranking at all, and an anonymous
    positional key can no longer win "times best".
    """
    f = {**_DEFAULT_SYNTHESIS_FRAMING, **(framing or {})}
    tallies = _binary_ranking_tallies(records_with_rankings)

    lines: list[str] = [f"# {f['title']}\n", f"{f['intro_paragraph']}\n"]
    lines.append(f"## Scope\n\n- {tallies.binary_count} binary records ranked.\n")
    lines.extend(_closest_to_truth_lines(tallies, f))
    lines.extend(_ensemble_vs_best_lines(tallies, f))
    lines.extend(_high_spread_lines(records_with_rankings, f))

    lines.append("## Manual synthesis\n")
    lines.append("<!-- Populate by hand after reading per-question miss_*.md files. -->\n")

    out_path.write_text("\n".join(lines))


def emit_combined_report(
    records_with_rankings: list[dict],
    miss_paths: list[Path],
    synthesis_path: Path,
    out_path: Path,
) -> None:
    """Write one combined markdown report that inlines each per-question file plus synthesis."""
    lines: list[str] = []
    lines.append("# Audit: bot misses (combined report)\n")
    lines.append("## Table of contents\n")
    for entry, _path in zip(records_with_rankings, miss_paths, strict=False):
        rec = entry["record"]
        anchor = f"miss-{rec['post_id']}"
        lines.append(f"- [{rec['post_id']} — {rec.get('title', '')[:70]}](#{anchor})")
    lines.append("- [Synthesis](#synthesis)")
    lines.append("")

    for entry, path in zip(records_with_rankings, miss_paths, strict=False):
        rec = entry["record"]
        lines.append(f"<a id='miss-{rec['post_id']}'></a>\n")
        lines.append(path.read_text())
        lines.append("\n---\n")

    lines.append("<a id='synthesis'></a>\n")
    lines.append(synthesis_path.read_text())

    out_path.write_text("\n".join(lines))
