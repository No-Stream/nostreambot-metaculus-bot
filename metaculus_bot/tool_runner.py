"""
Post-hoc probabilistic tool dispatch over forecaster-declared structured blocks.

Runs deterministic probability math on the structured declarations each
base forecaster emits (priors, base rates, hazards, percentiles, scenarios,
etc.) and returns markdown-ready strings for injection into the stacker's
view.

Active surface: wired into ``metaculus_bot/forecaster.py:_make_prediction``
(per-forecaster ``## Computed quantities``) and into the stacker prompt via
``build_cross_model_aggregation``. Per-question-type gating uses
``_feature_enabled(qtype)`` against ``PROBABILISTIC_TOOLS_TYPES`` so
numeric/binary/MC can be enabled independently.

Feature flag: ``PROBABILISTIC_TOOLS_ENABLED`` (env var, false-y by default).
The public entry points ``run_tools_for_forecaster`` and
``build_cross_model_aggregation`` check the flag internally and return
an empty string when it is not set — callers do not need to branch.

Responsibilities split:
- ``run_tools_for_forecaster`` handles a single forecaster's rationale and
  returns a per-forecaster ``## Computed quantities`` markdown block.
- ``build_cross_model_aggregation`` runs once per question over all
  forecasters' final predictions + structured blocks and returns a single
  ``## Cross-model aggregation`` markdown block.

Fail-visible, not fail-silent: every skipped tool (malformed JSON, missing
field, unexpected question type) logs at DEBUG or WARNING and is omitted
from the output block. Aggregation continues over whoever did emit valid
data.

Scope note: ``DiscreteCountStructured`` (schema) is intentionally not
dispatched here — discrete-count tools are phase-3 work.

Some of this dispatch is unreachable on a FRESH forecast, because the prompts stopped
eliciting the block fields it reads. The binary anchor / clause telemetry lines are gone
outright. ``_run_mc_tools``'s declared-Other line and its Dirichlet-with-Other branch both
key on ``other_mass`` / ``concentration``, which the MC prompt no longer asks for, so they
fire only on an archived block or on a model emitting the field from habit; turning
``PROBABILISTIC_TOOLS_ENABLED`` on today would produce binary lines and no MC ones. The
schema fields stay optional so archived blocks still parse. Why those fields were retired,
and what reviving any of it would cost, is in backlog decision-30 "Probabilistic tooling for base forecasters" and AGENTS.md rather than here.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Literal

import numpy as np
from forecasting_tools import (
    MetaculusQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.constants import env_flag_enabled
from metaculus_bot.probabilistic_tools import (
    DEFAULT_INFORMATIVE_PRIOR_STRENGTH,
    BetaBinomialResult,
    ConsistencyResult,
    SurvivalResult,
    TailMassResult,
    base_rate_blend,
    beta_binomial_update,
    cdf_at_threshold,
    dirichlet_with_other,
    implied_likelihood_ratio,
    linear_pool,
    linear_pool_options,
    log_pool,
    out_of_bounds_mass,
    percentile_family_consistency,
    prob_event_before,
    satopaa_extremize,
    stated_base_rate_consistency,
)
from metaculus_bot.question_types import QuestionType, question_type_of
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    StatedHazard,
    StructuredBlock,
    parse_structured_block,
)

logger = logging.getLogger(__name__)

FEATURE_FLAG_ENV = "PROBABILISTIC_TOOLS_ENABLED"

# Per-question-type allow-list. CSV of {"binary", "multiple_choice", "numeric"}.
# Defaults to every supported type when unset (back-compat: setting only the
# global flag enables everything as before). Production workflows set this
# explicitly to scope which types receive tool augmentation — empirically,
# binary + MC benefit, numeric is inconclusive.
TYPES_ENV = "PROBABILISTIC_TOOLS_TYPES"
DEFAULT_TYPES_CSV = "binary,multiple_choice,numeric"
_VALID_TYPES = frozenset({"binary", "multiple_choice", "numeric"})

# Z-gap between P10 and P90 in a standard normal — mirrors
# ``metaculus_bot.probabilistic_tools.distributions._P10_P90_Z_GAP``. Used to
# derive an approximate sigma from declared P10/P90 percentile pairs in the
# spread plausibility check.
_P10_P90_Z_GAP: float = 2.5631

# Per-forecaster sigma ratio (vs ensemble median sigma) below which we emit a
# ⚠ Spread anomaly line. Calibrated to catch the qid 43171 GLM-4.5-air case
# (sigma=13K vs ensemble median sigma ~965K, ratio ~1.3%) while being generous enough
# that legitimate-but-tighter forecasters don't get flagged. Defense-in-depth
# atop the family-consistency check.
_SPREAD_ANOMALY_RATIO_THRESHOLD: float = 0.10


def _feature_enabled(question_type: QuestionType | None = None) -> bool:
    """Return True iff global PROBABILISTIC_TOOLS_ENABLED is set AND
    question_type (when given) appears in the PROBABILISTIC_TOOLS_TYPES allow-list.

    The allow-list defaults to every supported type; production workflows set it explicitly
    to scope which types receive tool augmentation. When question_type is None
    (rare callers that don't know the type yet), only the global flag is checked.
    """
    if not env_flag_enabled(FEATURE_FLAG_ENV):
        return False
    if question_type is None:
        return True
    raw = os.environ.get(TYPES_ENV, DEFAULT_TYPES_CSV)
    allowed = {t.strip() for t in raw.split(",") if t.strip()}
    if not allowed.issubset(_VALID_TYPES):
        invalid = allowed - _VALID_TYPES
        logger.warning("PROBABILISTIC_TOOLS_TYPES has invalid entries %s; ignoring them", invalid)
        allowed = allowed & _VALID_TYPES
    return question_type in allowed


# ---------------------------------------------------------------------------
# Per-forecaster tool execution
# ---------------------------------------------------------------------------


def _format_beta_binom(result: BetaBinomialResult, ref_class: str) -> str:
    return (
        f"- **Beta-binomial (ref class: {ref_class})**: "
        f"posterior mean {result.posterior_mean:.3f}, "
        f"80% CI [{result.ci_80_low:.3f}, {result.ci_80_high:.3f}] "
        f"(α={result.posterior_alpha:.1f}, β={result.posterior_beta:.1f})"  # noqa: RUF001  # Greek math notation in rendered output
    )


def _format_survival(result: SurvivalResult, hazard: StatedHazard) -> str:
    return (
        f"- **Survival / hazard**: rate {hazard.rate_per_unit:.3g}/{hazard.unit}, "
        f"elapsed={hazard.elapsed_fraction:.2f}, remaining={hazard.remaining_fraction:.2f} → "
        f"P(event in full window) = {result.unconditional_prob:.3f}, "
        f"P(event in remaining | none yet) = {result.conditional_prob_given_no_event_yet:.3f}"
    )


def _format_prior_posterior_check(result: ConsistencyResult, prior_prob: float, posterior_prob: float) -> str:
    lr = result.details.get("implied_lr")
    flag_mark = " ⚠ FLAGGED" if result.flag else ""
    reason = f" — {result.flag_reason}" if result.flag_reason else ""
    lr_str = f"{lr:.2f}" if lr is not None else "n/a"
    return (
        f"- **Prior → posterior{flag_mark}**: "
        f"prior {prior_prob:.3f} → posterior {posterior_prob:.3f}, implied LR = {lr_str}{reason}"
    )


def _format_tail_mass(result: TailMassResult, family: str) -> str:
    return (
        f"- **Out-of-bounds mass ({family} fit)**: "
        f"P(< lower) = {result.prob_below_min:.3f}, "
        f"P(> upper) = {result.prob_above_max:.3f}, "
        f"interior mass = {result.interior_mass:.3f}"
    )


def _format_family_consistency(result: ConsistencyResult) -> str:
    best = result.details.get("best_fit_family", "?")
    claimed = result.details.get("claimed_family", "?")
    flag_mark = " ⚠ FLAGGED" if result.flag else ""
    reason = f" — {result.flag_reason}" if result.flag_reason else ""
    return f"- **Percentile-family consistency{flag_mark}**: claimed {claimed!r}, best-fit {best!r}{reason}"


def _lr_chained_posterior(prior_prob: float, lrs: list[float]) -> float | None:
    """Chain evidence-LRs onto prior odds. Returns None if prior is at a
    boundary (LR update undefined)."""
    if not (0.0 < prior_prob < 1.0):
        return None
    prior_odds = prior_prob / (1.0 - prior_prob)
    post_odds = prior_odds
    for lr in lrs:
        if lr <= 0 or not math.isfinite(lr):
            return None
        post_odds *= lr
        if not math.isfinite(post_odds):
            return 1.0 - 1e-9  # saturated — evidence overwhelmingly supports hypothesis
    return post_odds / (1.0 + post_odds)


def _run_binary_tools(block: BinaryStructured) -> list[str]:
    lines: list[str] = []

    # Beta-binomial on k/n. If the forecaster also declared a prior, use it
    # as an informative prior centered on prior.prob; otherwise use a
    # Jeffreys-ish weakly informative prior.
    if block.base_rate is not None:
        if block.prior is not None and 0.0 < block.prior.prob < 1.0:
            prior_mean = block.prior.prob
            prior_strength = DEFAULT_INFORMATIVE_PRIOR_STRENGTH
        else:
            prior_mean = 0.5
            prior_strength = 1.0
        bb_result = beta_binomial_update(
            k=block.base_rate.k,
            n=block.base_rate.n,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
        )
        lines.append(_format_beta_binom(bb_result, block.base_rate.ref_class))
    else:
        bb_result = None

    lines.extend(_scenario_decomposition_lines(block))
    lines.extend(_hazard_lines(block))
    lines.extend(_prior_posterior_consistency_lines(block))

    # Bayesian combine of stated prior with Beta-binomial posterior — surfaced
    # only when both prior and base_rate are declared so the stacker can see
    # how the stated posterior compares.
    if block.prior is not None and block.base_rate is not None and bb_result is not None:
        lines.append(
            f"- **Prior + k/n Bayesian combine**: stated prior {block.prior.prob:.3f} + "
            f"k/n {block.base_rate.k}/{block.base_rate.n} (strength {DEFAULT_INFORMATIVE_PRIOR_STRENGTH:.1f}) → "
            f"posterior {bb_result.posterior_mean:.3f} "
            f"[80% CI {bb_result.ci_80_low:.3f}-{bb_result.ci_80_high:.3f}]; "
            f"declared posterior {block.posterior_prob:.3f}. "
            f"Δ = {block.posterior_prob - bb_result.posterior_mean:+.3f}"
        )

    lines.extend(_lr_chained_posterior_lines(block))

    return lines


def _scenario_decomposition_lines(block: BinaryStructured) -> list[str]:
    """Count-only scenario line.

    The schema already enforces that the branch probs sum to ~1.0, and
    ``conditional_outcome`` is free text, so there is no arithmetic to verify against the
    posterior — hence a count rather than a check.
    """
    if not block.scenarios:
        return []
    n_scenarios = len(block.scenarios)
    scenario_names = ", ".join(s.name for s in block.scenarios[:3])
    if n_scenarios > 3:
        scenario_names += f", +{n_scenarios - 3} more"
    return [f"- **Declared scenario decomposition**: {n_scenarios} branches ({scenario_names})"]


def _hazard_lines(block: BinaryStructured) -> list[str]:
    """Survival / hazard line. Units cancel: ``window_duration_units`` shares ``rate_per_unit``'s."""
    if block.hazard is None:
        return []
    survival = prob_event_before(
        hazard_rate=block.hazard.rate_per_unit,
        elapsed_fraction=block.hazard.elapsed_fraction,
        remaining_fraction=block.hazard.remaining_fraction,
        window_length=block.hazard.window_duration_units,
    )
    return [_format_survival(survival, block.hazard)]


def _prior_posterior_consistency_lines(block: BinaryStructured) -> list[str]:
    """Prior -> posterior coherence check, or the base-rate -> posterior implied LR.

    A declared prior takes precedence; the k/n implied-LR line is the fallback for a block
    that declared a base rate but no prior.
    """
    if block.prior is not None:
        max_strength = _max_evidence_strength(block.evidence)
        try:
            cons_result = stated_base_rate_consistency(
                stated_base_rate_prob=block.prior.prob,
                stated_posterior_prob=block.posterior_prob,
                evidence_strength_max=max_strength,
            )
            return [_format_prior_posterior_check(cons_result, block.prior.prob, block.posterior_prob)]
        except ValueError as exc:
            logger.debug("stated_base_rate_consistency skipped: %s", exc)
            return []

    if block.base_rate is None:
        return []
    br_mean = block.base_rate.k / max(block.base_rate.n, 1)
    if not (0.0 < br_mean < 1.0 and 0.0 < block.posterior_prob < 1.0):
        return []
    try:
        lr = implied_likelihood_ratio(br_mean, block.posterior_prob)
        return [
            f"- **Base-rate → posterior**: k/n = {block.base_rate.k}/{block.base_rate.n} = "
            f"{br_mean:.3f} → posterior {block.posterior_prob:.3f}, implied LR = {lr:.2f}"
        ]
    except ValueError as exc:
        logger.debug("implied_likelihood_ratio skipped: %s", exc)
        return []


def _lr_chained_posterior_lines(block: BinaryStructured) -> list[str]:
    """Posterior implied by chaining the declared per-item likelihood ratios off the prior."""
    if block.prior is None:
        return []
    declared_lrs = [ev.likelihood_ratio for ev in block.evidence if ev.likelihood_ratio is not None]
    if not declared_lrs:
        return []
    chained = _lr_chained_posterior(block.prior.prob, declared_lrs)
    if chained is None:
        return []
    return [
        f"- **Evidence-LR-chained posterior**: prior {block.prior.prob:.3f} × "
        f"{len(declared_lrs)} declared LR(s) ({', '.join(f'{lr:.2f}' for lr in declared_lrs)}) → "
        f"{chained:.3f}; declared posterior {block.posterior_prob:.3f}. "
        f"Δ = {block.posterior_prob - chained:+.3f}"
    ]


def _run_numeric_tools(block: NumericStructured, question: NumericQuestion) -> list[str]:
    lines: list[str] = []

    # Fit once, reuse for both consistency check and tail-mass computation.
    try:
        if not block.declared_percentiles:
            raise ValueError("declared_percentiles is missing or empty")
        family_result = percentile_family_consistency(
            declared_percentiles=block.declared_percentiles,
            claimed_family=None,
            student_t_df=None,
        )
        lines.append(_format_family_consistency(family_result))
    except ValueError as exc:
        logger.debug("percentile_family_consistency skipped: %s", exc)
        family_result = None

    if family_result is not None:
        hint = family_result.details.get("best_fit_family")
        fit = family_result.details["fits_by_family"].get(hint) if hint else None
        if fit is not None:
            lower = question.lower_bound if not question.open_lower_bound else None
            upper = question.upper_bound if not question.open_upper_bound else None
            try:
                tail = out_of_bounds_mass(fit, lower_bound=lower, upper_bound=upper)
                lines.append(_format_tail_mass(tail, family=hint or type(fit).__name__))
            except ValueError as exc:
                logger.debug("out_of_bounds_mass skipped: %s", exc)

    return lines


def _max_evidence_strength(evidence: list) -> Literal["strong", "moderate", "weak", "none"]:
    if not evidence:
        return "none"
    strengths = {e.strength for e in evidence}
    if "strong" in strengths:
        return "strong"
    if "moderate" in strengths:
        return "moderate"
    return "weak"


def _run_mc_tools(block: MultipleChoiceStructured) -> list[str]:
    lines: list[str] = []
    if block.other_mass is not None:
        lines.append(
            f"- **Declared Other / residual mass**: {block.other_mass:.3f} "
            f"(over {len(block.option_probs)} named options)"
        )

    # If the forecaster declared an Other mass OR a concentration, surface
    # Dirichlet-with-Other CIs for the top-3 options by mean. Deliberately
    # skipped when neither is declared — forcing a concentration would add
    # noise without forecaster intent.
    #
    # Schema contract vs tool contract: the pydantic schema allows
    # ``option_probs`` to sum to ~1.0 *and* carry an ``other_mass`` alongside
    # (i.e., the option_probs are conditional on "not Other"). The
    # ``dirichlet_with_other`` tool expects option_probs + other_mass to sum
    # to ~1.0 together. We renormalize into the tool's contract before
    # calling it, treating option_probs as the (1 - other_mass) mass
    # redistributed proportionally across named options.
    if block.other_mass is not None or block.concentration is not None:
        # Treat a declared other_mass of exactly 0 as equivalent to no Other:
        # the tool requires alpha_k > 0 for every component, and forcing
        # Other=0 would raise. Semantically, "declared residual mass is 0"
        # means "all mass is on the named options".
        effective_other_mass = block.other_mass if (block.other_mass or 0.0) > 0.0 else None
        try:
            if effective_other_mass is not None:
                non_other = max(0.0, 1.0 - effective_other_mass)
                option_sum = sum(block.option_probs.values()) or 1.0
                scaled = {k: v * non_other / option_sum for k, v in block.option_probs.items()}
            else:
                scaled = dict(block.option_probs)
            cis = dirichlet_with_other(
                option_probs=scaled,
                other_mass=effective_other_mass,
                concentration=block.concentration or 10.0,
            )
            top = sorted(cis.items(), key=lambda kv: -kv[1].mean)[:3]
            parts = [f"{name} {ci.mean:.3f} [80% CI {ci.ci_80_low:.3f}-{ci.ci_80_high:.3f}]" for name, ci in top]
            lines.append(f"- **Dirichlet-with-Other (top 3 by mean)**: {'; '.join(parts)}")
        except ValueError as exc:
            logger.debug("dirichlet_with_other skipped: %s", exc)

    return lines


def run_tools_for_forecaster(
    question: MetaculusQuestion,
    rationale: str,
    forecaster_id: str,
) -> str:
    """
    Extract + dispatch tools for a single forecaster's rationale.

    Returns a markdown section (without leading header) or empty string
    when the feature flag is off, no structured block was found, or no
    tool produced output.
    """
    qtype = question_type_of(question)
    if qtype is None:
        logger.debug(
            "Unsupported question type %s for tool runner; skipping (forecaster=%s)",
            type(question).__name__,
            forecaster_id,
        )
        return ""

    if not _feature_enabled(qtype):
        return ""

    block = parse_structured_block(rationale, qtype)
    if block is None:
        return ""

    if isinstance(block, BinaryStructured):
        lines = _run_binary_tools(block)
    elif isinstance(block, NumericStructured):
        # qtype=="numeric" guarantees question is a NumericQuestion (see
        # question_type_of); cast is needed because pyright can't prove the
        # cross-field invariant across the discriminated union.
        assert isinstance(question, NumericQuestion)
        lines = _run_numeric_tools(block, question)
    elif isinstance(block, MultipleChoiceStructured):
        lines = _run_mc_tools(block)
    else:
        return ""

    if not lines:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-model aggregation
# ---------------------------------------------------------------------------


def _parse_all_blocks(
    rationales: list[str],
    qtype: Literal["binary", "numeric", "multiple_choice"],
) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []
    for r in rationales:
        b = parse_structured_block(r, qtype)
        if b is not None:
            blocks.append(b)
    return blocks


def _aggregate_binary_lines(prediction_probs: list[float], blocks: list[BinaryStructured]) -> list[str]:
    lines: list[str] = []
    if len(prediction_probs) < 2:
        return lines

    try:
        lp = linear_pool(prediction_probs)
        logp = log_pool(prediction_probs)
        ext = satopaa_extremize(prediction_probs, alpha=2.5)
        lines.append(
            f"- **Pools over {len(prediction_probs)} forecasters**: "
            f"linear {lp:.3f}, log {logp:.3f}, Satopää α=2.5 {ext:.3f}"  # noqa: RUF001  # Greek math notation in rendered output
        )
    except ValueError as exc:
        logger.debug("binary pools skipped: %s", exc)

    base_rate_probs: list[float] = []
    for b in blocks:
        if b.base_rate is not None and b.base_rate.n > 0:
            base_rate_probs.append(b.base_rate.k / b.base_rate.n)
    if len(base_rate_probs) >= 2:
        try:
            blended = base_rate_blend(base_rate_probs, method="linear")
            lines.append(
                f"- **Blended base rate across {len(base_rate_probs)} forecasters**: "
                f"{blended:.3f} (range {min(base_rate_probs):.3f}–{max(base_rate_probs):.3f})"  # noqa: RUF001  # en dash range typography in rendered output
            )
        except ValueError as exc:
            logger.debug("base_rate_blend skipped: %s", exc)

    flagged = [b for b in blocks if b.prior is not None]
    if flagged:
        priors = [b.prior.prob for b in flagged if b.prior is not None]
        posteriors = [b.posterior_prob for b in flagged]
        if priors and posteriors:
            lines.append(
                f"- **Prior/posterior snapshot**: {len(priors)} forecasters declared priors, "
                f"priors range {min(priors):.3f}–{max(priors):.3f}, "  # noqa: RUF001  # en dash range typography in rendered output
                f"posteriors range {min(posteriors):.3f}–{max(posteriors):.3f}"  # noqa: RUF001  # en dash range typography in rendered output
            )

    return lines


def _derive_sigma_from_percentiles(pcts: list[Percentile]) -> float | None:
    """Extract sigma ≈ (P90 - P10) / z_gap. Returns None if either percentile is missing.

    Preferred over fitting a parametric distribution because (a) it's cheap,
    (b) it's the same approximation used by ``_initial_normal_guess`` in
    probabilistic_tools/distributions.py, and (c) we just need an order-of-
    magnitude proxy to compare against the ensemble median.
    """
    p10_val: float | None = None
    p90_val: float | None = None
    for p in pcts:
        if abs(p.percentile - 0.10) < 1e-6:
            p10_val = p.value
        elif abs(p.percentile - 0.90) < 1e-6:
            p90_val = p.value
    if p10_val is None or p90_val is None:
        return None
    sigma = (p90_val - p10_val) / _P10_P90_Z_GAP
    if sigma <= 0 or not math.isfinite(sigma):
        return None
    return sigma


def _spread_plausibility_lines(prediction_percentiles: list[list[Percentile]]) -> list[str]:
    """Per-forecaster sigma vs ensemble median sigma. ⚠ flag when ratio < threshold.

    Defense-in-depth atop ``percentile_family_consistency`` — catches
    confidently-narrow forecasters whose claimed-vs-best-fit family looks
    fine but whose sigma is implausibly small relative to peers (qid 43171).

    Skip semantics:
    * Forecasters missing P10 or P90 are recorded at DEBUG and excluded
      from the sigma pool (no crash). They still consume a forecaster_idx slot.
    * If fewer than 2 forecasters have valid sigma, the section is suppressed
      entirely (median undefined for n=1, and there's nothing to compare).
    """
    lines: list[str] = []
    sigmas_with_idx: list[tuple[int, float]] = []
    for idx, pcts in enumerate(prediction_percentiles, start=1):
        sigma = _derive_sigma_from_percentiles(pcts)
        if sigma is None:
            logger.debug(
                "spread_plausibility: forecaster %d missing P10 or P90; skipping sigma derivation",
                idx,
            )
            continue
        sigmas_with_idx.append((idx, sigma))

    if len(sigmas_with_idx) < 2:
        return lines

    sigmas = [s for _, s in sigmas_with_idx]
    median_sigma = float(np.median(sigmas))
    if median_sigma <= 0 or not math.isfinite(median_sigma):
        return lines

    anomalies: list[str] = []
    for idx, sigma in sigmas_with_idx:
        ratio = sigma / median_sigma
        if ratio < _SPREAD_ANOMALY_RATIO_THRESHOLD:
            anomalies.append(
                f"- ⚠ Spread anomaly (forecaster {idx}): σ={sigma:.3g} is "  # noqa: RUF001  # Greek math notation in rendered output
                f"{ratio * 100:.1f}% of ensemble median σ={median_sigma:.3g}"  # noqa: RUF001  # Greek math notation in rendered output
            )

    if anomalies:
        lines.extend(anomalies)
    else:
        threshold_factor = round(1.0 / _SPREAD_ANOMALY_RATIO_THRESHOLD)
        lines.append(
            f"- **Spread plausibility**: all {len(sigmas)} forecasters within "
            f"{threshold_factor}× spread of ensemble median "
            f"(σ range {min(sigmas):.3g}–{max(sigmas):.3g})"  # noqa: RUF001  # Greek + en dash typography in rendered output
        )
    return lines


def _aggregate_numeric_lines(
    prediction_percentiles: list[list[Percentile]],
    blocks: list[NumericStructured],
) -> list[str]:
    lines: list[str] = []
    if len(prediction_percentiles) < 2:
        return lines

    medians: list[float] = []
    for pcts in prediction_percentiles:
        for p in pcts:
            if abs(p.percentile - 0.5) < 1e-6:
                medians.append(p.value)
                break
    if len(medians) >= 2:
        lines.append(f"- **Forecaster medians**: min {min(medians):.3g}, max {max(medians):.3g}, n={len(medians)}")

    lines.extend(_spread_plausibility_lines(prediction_percentiles))

    return lines


def _aggregate_mc_lines(prediction_options: list[PredictedOptionList]) -> list[str]:
    lines: list[str] = []
    if len(prediction_options) < 2:
        return lines

    option_dicts: list[dict[str, float]] = []
    for pred in prediction_options:
        option_dicts.append({o.option_name: o.probability for o in pred.predicted_options})

    keys = set(option_dicts[0].keys())
    if not all(set(d.keys()) == keys for d in option_dicts):
        lines.append("- **MC aggregation skipped**: option sets differ across forecasters")
        return lines

    try:
        pooled = linear_pool_options(option_dicts)
    except ValueError as exc:
        logger.debug("linear_pool_options skipped: %s", exc)
        return lines

    top = sorted(pooled.items(), key=lambda kv: -kv[1])[:3]
    top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
    lines.append(f"- **Linear pool across {len(option_dicts)} forecasters** (top 3): {top_str}")
    return lines


def aggregate_binary_values(rationales: list[str], prediction_probs: list[float]) -> str:
    """Public entry for binary aggregation (typed).

    Returns empty string when the feature flag is off or there is nothing to report.
    """
    if not _feature_enabled("binary"):
        return ""
    blocks_all = _parse_all_blocks(rationales, "binary")
    binary_blocks = [b for b in blocks_all if isinstance(b, BinaryStructured)]
    lines = _aggregate_binary_lines(prediction_probs, binary_blocks)
    return "\n".join(lines) if lines else ""


def aggregate_numeric_values(
    rationales: list[str],
    prediction_percentiles: list[list[Percentile]] | list[NumericDistribution],
) -> str:
    """Public entry for numeric aggregation (typed).

    Accepts either ``list[list[Percentile]]`` (legacy) or
    ``list[NumericDistribution]`` (current). ``main.py`` passes the latter
    because ``ReasonedPrediction.prediction_value`` for numeric questions is a
    ``NumericDistribution``; iterating that Pydantic model yields
    ``(field_name, value)`` tuples and silently broke the median-extraction
    loop. Normalize at the boundary so both shapes work.
    """
    if not _feature_enabled("numeric"):
        return ""
    normalized: list[list[Percentile]] = []
    for entry in prediction_percentiles:
        if isinstance(entry, NumericDistribution):
            # declared_percentiles preserves the standard anchor points the
            # forecaster asserted; that's what the median-extraction loop
            # expects. The full ``PCHIP_CDF_POINTS`` CDF grid would also work but
            # is less information-dense (median is one of the declared anchors).
            normalized.append(list(entry.declared_percentiles))
        else:
            normalized.append(list(entry))
    blocks_all = _parse_all_blocks(rationales, "numeric")
    numeric_blocks = [b for b in blocks_all if isinstance(b, NumericStructured)]
    lines = _aggregate_numeric_lines(normalized, numeric_blocks)
    return "\n".join(lines) if lines else ""


def aggregate_mc_values(_rationales: list[str], prediction_options: list[PredictedOptionList]) -> str:
    """Public entry for multiple-choice aggregation (typed).

    ``_rationales`` is accepted for API symmetry with the binary/numeric
    facades but unused: MC aggregation only needs option probability lists.
    """
    if not _feature_enabled("multiple_choice"):
        return ""
    lines = _aggregate_mc_lines(prediction_options)
    return "\n".join(lines) if lines else ""


def build_cross_model_aggregation(
    question: MetaculusQuestion,
    rationales: list[str],
    prediction_values: list,
) -> str:
    """
    Type-dispatching facade around the typed ``aggregate_*`` entry points.

    Returns empty string when the feature flag is off, the question type is
    unsupported, or there is nothing useful to report. Callers can instead
    use the typed entry points (``aggregate_binary_values`` etc.) directly.
    """
    qtype = question_type_of(question)
    if qtype is None:
        return ""

    if not _feature_enabled(qtype):
        return ""

    if qtype == "binary":
        return aggregate_binary_values(rationales, prediction_values)
    if qtype == "numeric":
        return aggregate_numeric_values(rationales, prediction_values)
    if qtype == "multiple_choice":
        return aggregate_mc_values(rationales, prediction_values)
    return ""


# ---------------------------------------------------------------------------
# Convenience: threshold-based CDF extraction (numeric only)
# ---------------------------------------------------------------------------


def cdf_at_threshold_for_forecaster(
    rationale: str,
    question: NumericQuestion,
    threshold: float,
) -> float | None:
    """
    Fit the forecaster's declared percentiles and return P(X <= threshold).

    Useful for threshold-binary questions where we want to check a numeric
    forecaster's implied probability against a specific cutoff. Returns
    None when the feature flag is off, no structured block, no fit
    succeeds, or threshold is unreachable from the fit.

    Logs a debug line when ``threshold`` falls outside the question's closed
    bounds — the result is still computed (it's legitimately a tail-mass
    query) but the line helps trace unexpected inputs.
    """
    if not _feature_enabled("numeric"):
        return None
    if not question.open_lower_bound and threshold < question.lower_bound:
        logger.debug(
            "cdf_at_threshold: threshold %.6g below closed lower bound %.6g",
            threshold,
            question.lower_bound,
        )
    if not question.open_upper_bound and threshold > question.upper_bound:
        logger.debug(
            "cdf_at_threshold: threshold %.6g above closed upper bound %.6g",
            threshold,
            question.upper_bound,
        )
    block = parse_structured_block(rationale, "numeric")
    if not isinstance(block, NumericStructured):
        return None
    try:
        if not block.declared_percentiles:
            raise ValueError("declared_percentiles is missing or empty")
        family_result = percentile_family_consistency(
            block.declared_percentiles,
            claimed_family=None,
            student_t_df=None,
        )
    except ValueError as exc:
        logger.debug("percentile_family_consistency skipped: %s", exc)
        return None
    hint = family_result.details.get("best_fit_family")
    if not hint:
        return None
    fit = family_result.details["fits_by_family"].get(hint)
    if fit is None:
        return None
    try:
        return cdf_at_threshold(fit, threshold)
    except ValueError as exc:
        logger.debug("cdf_at_threshold failed: %s", exc)
        return None


__all__ = [
    "FEATURE_FLAG_ENV",
    "aggregate_binary_values",
    "aggregate_mc_values",
    "aggregate_numeric_values",
    "build_cross_model_aggregation",
    "cdf_at_threshold_for_forecaster",
    "run_tools_for_forecaster",
]
