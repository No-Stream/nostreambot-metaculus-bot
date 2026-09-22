"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = [
    "DISAGREEMENT_ANALYZER_LLM",
    "FORECASTER_LLMS",
    "FORECASTER_MODEL_NAMES",
    "MARKET_QUERY_AUTHOR_LLM_CONFIG",
    "MARKET_RANKER_LLM_CONFIG",
    "PARSER_LLM",
    "RESEARCHER_LLM",
    "STACKER_FALLBACK_LLM",
    "STACKER_LLM",
    "SUMMARIZER_LLM",
]
# Reasoning models ignore (or degrade under) explicit sampling params, so we
# defer to provider defaults. temperature=None is explicit but redundant on
# ft 0.2.92, whose GeneralLlm ctor already defaults temperature to None (0.2.54
# injected 0 when the arg was omitted); top_p flows via **kwargs and is never set.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 64_000,  # Prevent truncation; all current forecasters/stackers support 64k output
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}
# Low-effort utility slots (parser, summarizer, analyzer). Same sampling-param
# rationale as REASONING_MODEL_CONFIG: temperature=None defers to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p
# left unset.
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
ACCEPTABLE_QUANTS = [
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]

# Per-instance allowed_tries=1 override (Round-2): forecaster .invoke is wrapped
# in the broad retry gated on TRANSIENT_RETRY_MAX_ELAPSED_S (forecaster_runners.py)
# so we can impose the universal "never retry a slow failure" deadline-safety rule
# that forecasting-tools' un-gated tenacity cannot. Spread per-instance (NOT by mutating
# REASONING_MODEL_CONFIG) so PARSER_LLM / STACKER configs are untouched.
_FORECASTER_CONFIG = {**REASONING_MODEL_CONFIG, "allowed_tries": 1}


def forecaster_role(model: str) -> str:
    """``forecaster:<vendor>`` for an ``openrouter/<vendor>/<model>`` roster slug.

    The CREDIT_ROLE_SPEND spend line every roster slot books under. The roster is
    latest-per-vendor, one slot each, so the VENDOR is the stable identity of a slot
    across model rotations — a per-model role would start a new time series at every swap
    and defeat the era-over-era cost comparison this exists for.
    """
    parts = model.split("/")
    if len(parts) < 3 or parts[0] != "openrouter":
        raise ValueError(f"forecaster_role expects an openrouter/<vendor>/<model> slug, got {model!r}")
    return f"forecaster:{parts[1]}"


def _forecaster_slot(model: str, **kwargs: Any) -> GeneralLlm:
    """One roster member, booked in the CREDIT_ROLE_SPEND ledger under ``forecaster:<vendor>``.

    The role is derived from the slug rather than written beside it so a roster swap cannot
    leave a slot mislabeled.
    """
    return build_llm_with_openrouter_fallback(model=model, role=forecaster_role(model), **_FORECASTER_CONFIG, **kwargs)


# SEASON-START RITUAL (operator, not an implementing session): resolve "latest per vendor"
# with a LIVE OpenRouter model-list read, never from memory — nothing in this repo can say
# what the newest OpenAI/Anthropic/Google model currently is, and the 2026-08-31 gemini-slot
# review found that a roster decision needs that one read before anything else:
#   curl -s https://openrouter.ai/api/v1/models | jq -r '.data[] | [.id, .created] | @tsv' | sort
# filtered per vendor prefix (openai/, anthropic/, google/, x-ai/); what to check on the
# result is in docs/operations.md "Season-start checklist". Any change here is a config-era
# boundary for residual analysis, so make it once, before the first question.
FORECASTER_LLMS: list[GeneralLlm] = [
    # 2026-07-20: forecaster roster dropped from 6 to a 3-member latest-per-vendor
    # triple (1 OpenAI / 1 Anthropic / 1 Google). This is the SECOND roster change
    # on 2026-07-20 and supersedes the morning fable-5 → opus-4.7 swap (7a76df6) as
    # the config-era boundary for residual analysis. Removed: gpt-5.5,
    # claude-opus-4.7, grok-4.5. Two adversarially-verified analyses
    # (scratch/ensemble_3member_audit_2026-07-20/ +
    # scratch/ensemble_power_model_2026-07-20/) found the triple non-inferior on
    # binary/MC and only a fragile numeric lean toward the full roster (+3.24,
    # 95% CI [-2.5, +9.1], P(loss>1pt/Q)=0.80, driven by 2 questions) — accepted as
    # a ship-and-watch bet; see FUTURE.md "Frozen-triple numeric watch". Dropping
    # grok (x-ai, 404s on the donated key) also ends routine personal-key forecaster
    # spend: only the gemini-3.1-pro-preview personal-key PIN bills
    # OPENROUTER_API_KEY now; the other two slots route via the donated key.
    # (Dates anchor config eras for residual analysis.)
    #
    # OpenAI flagship (5.6 series). 2026-07-20: effort xhigh -> high. The
    # reasoning-effort audit (scratch/reasoning_effort_audit_2026-07-20/) found
    # default->high clearly worth it but high->xhigh UNMEASURED, so we stop paying
    # the unmeasured premium here (the three slots measure within 12% of each other,
    # $0.24 to $0.27 a question, 2026-09-09). opus-4.8 keeps xhigh below as the remaining premium bet
    # (FUTURE.md "Price the high->xhigh reasoning-effort premium"). (2026-07-15
    # had bumped this high -> xhigh.) Live-verified: OpenRouter's effort enum is
    # max|xhigh|high|medium|low|minimal|none and this model accepts high (bogus
    # values 400). NOTE: "max" is Anthropic-only — OpenAI's ceiling is xhigh and
    # OpenAI rejects max upstream even though OpenRouter's enum validation admits it.
    # 2026-09-22: sol -> gpt-6-sol (GPT-6 release) and high -> xhigh (operator), matching the
    # Anthropic slot; a single prod-prompt timing probe checked it against FORECASTER_SOFT_DEADLINE.
    _forecaster_slot(
        "openrouter/openai/gpt-6-sol",
        reasoning={"effort": "xhigh"},
    ),
    # Anthropic slot. 2026-07-15: enabled:True (provider-default adaptive thinking)
    # -> explicit effort=xhigh. Anthropic also exposes "max" one tier above xhigh —
    # held back deliberately for latency: unbounded adaptive thinking caused silent
    # FORECASTER_SOFT_DEADLINE stalls on the retired opus-4.6 slot, e.g. Q14333 on
    # 2026-05-07.
    # 2026-09-22: opus-4.8 -> opus-5.5 (Anthropic release). Effort and verbosity unchanged.
    _forecaster_slot(
        "openrouter/anthropic/claude-opus-5.5",
        reasoning={"effort": "xhigh"},
        extra_body={"verbosity": "high"},
    ),
    # Google slot. No explicit reasoning-effort kwarg — gemini-3.1-pro-preview has
    # no xhigh tier and uses provider defaults. PINNED to the personal
    # OPENROUTER_API_KEY via the DONATED_KEY_BLOCKED_GOOGLE_MODELS blocklist in
    # fallback_openrouter (the donated key routes it through a free-tier Google
    # AI Studio BYOK integration with quota 0, so it would 429 there); see the
    # TODO(gemini-3.1-pro-donated) tag pending the Metaculus-side BYOK fix.
    _forecaster_slot("openrouter/google/gemini-3.1-pro-preview"),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster (e.g. 'claude-opus-5.5') — strips the 'openrouter/<provider>/' prefix.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

# Summarizer: compresses raw AskNews article markdown into an analyst briefing
# (AskNews-only; all other providers already emit LLM prose). sol → terra
# 2026-07-18 operator decision: AskNews is an auxiliary/augmenting source
# (content audit: 16% unique content vs native-search 54% / gap-fill 59%), so
# the absolute-frontier tier isn't warranted. The role audit
# (scratch/research_role_audit_2026-07-17/) had sol 1st but verdict "MARGINAL
# EDGE" with terra 2nd (one attribution blur, no fabrications), and 4/5 briefing
# failures in the AskNews quality audit (scratch/asknews_quality_audit_2026-07-18/)
# were prompt-era (mini summarizer + missing no-forecast rule), not model-tier.
# Terra: -43% cost, ~50s vs ~118s wall. Effort stays low (latency).
# allowed_tries=1 (Round-2): the summarizer invoke is wrapped in the broad,
# elapsed-gated retry (orchestrator._summarize_asknews) to impose the universal
# "never retry a slow failure" deadline rule. Per-instance override so PARSER_LLM (which
# also uses UTILITY_MODEL_CONFIG) keeps its allowed_tries=3.
# 2026-09-22: terra -> gpt-6-sol. Terra has no GPT-6 successor, so every Terra role
# moves to Sol 6 at the same (low) effort it ran at.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-6-sol",
    role="summarizer",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
# Parser: deterministic extraction of percentiles/JSON from rationales — a
# capability-saturated task, so it rides the cheapest tier that saturates it and
# keeps allowed_tries=3 for robustness. mini → luna 2026-08-03: the per-token
# comparison that used to favor mini inverted. Luna was $0.20/$1.20 vs mini's
# $0.75/$4.50 per 1M, so the newer model was also the ~3.75x cheaper one. (The
# models API showed $0.10/$0.60 behind a "50% off" badge on 2026-08-03; a live
# call on 2026-08-04 billed at double that, so the promo does not apply on this
# route — see the ranker cost comment below. The swap still won, by less.)
# 2026-09-22: gpt-5.6-luna -> gpt-6-luna (GPT-6 release), now $0.10/$0.50 per 1M.
# Effort unchanged at low.
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-6-luna",
    role="parser",
    reasoning={"effort": "low"},
    **UTILITY_MODEL_CONFIG,
)
# Researcher slot in the forecasting-tools LLM config dict. Effectively dead
# code in our pipeline — we use research providers (AskNews/Gemini/native_search)
# rather than the framework's researcher path — but the slot must be populated
# to avoid silent framework defaults. Aliasing to SUMMARIZER_LLM rather than
# constructing a duplicate config: same model, same effort, same job tier, no
# reason to maintain two parallel definitions.
RESEARCHER_LLM = SUMMARIZER_LLM

# Stacker meta-model for conditional stacking (invoked only on high-disagreement questions).
#
# allowed_tries=1: a single attempt at REASONING_MODEL_CONFIG's timeout, no
# retries. The outer STACKER_SOFT_DEADLINE catches wholly stuck calls; on failure
# we fall back to STACKER_FALLBACK_LLM rather than burning two more full-timeout
# attempts against the same Anthropic API that just stalled. Retrying against the same
# provider after a stall rarely succeeds (we're almost certainly re-rolling a
# dice with the same distribution), and the budget is better spent on a
# different-provider fallback.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    # 2026-07-20: fable-5 → opus-4.8 (fable-5 pulled from BOTH roles after
    # content=None failures in the 2026-07-19 test_bot run — see the forecaster-slot
    # comment above + FUTURE.md). Stacking is prod-disabled, so this is
    # backtest/ablation-only exposure today. 2026-09-22: opus-4.8 -> opus-5.5
    # (Anthropic release), effort and verbosity unchanged. Anthropic uses
    # effort-based adaptive thinking, not a max_tokens budget. Live-verified
    # OpenRouter effort enum: none/minimal/low/medium/high/xhigh/max.
    # effort=xhigh matches the forecaster slot; "max" (one tier above xhigh) is
    # deliberately held back for latency — the stacker runs under STACKER_SOFT_DEADLINE.
    "openrouter/anthropic/claude-opus-5.5",
    role="stacker",
    reasoning={"effort": "xhigh"},
    extra_body={"verbosity": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors.
# Reasoning slot → strongest OpenAI tier (gpt-6-sol, gpt-5.6-sol -> gpt-6-sol on
# the 2026-09-22 GPT-6 migration) at high effort; deliberately cross-provider
# from the Anthropic primary so an Anthropic stall doesn't take both attempts
# down. Tighter timeout and single try since we're already running late on the
# critical path by the time this fires. Stays at high (not xhigh) for that same
# reason — the 2026-07-15 xhigh bump covers the primary stacker and forecaster
# slots, not this tighter-budget path.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-6-sol",
    role="stacker_fallback",
    reasoning={"effort": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# --- The prediction-market provider's two LLM stages ---
#
# Both are RAW DICTS rather than built GeneralLlm singletons, unlike PARSER_LLM and friends:
# the provider is gated OFF by default, so paying construction cost at import would be waste,
# and the tests patch `build_llm_with_openrouter_fallback` at the provider's one invocation
# helper. Both route `openrouter/openai/...` through that wrapper, which tries the donated
# Metaculus key first and falls back to the personal key on credential / credit / route errors
# — so prod spend lands on OAI_ANTH_OPENROUTER_KEY.
#
# `allowed_tries=1` is required, not decorative: the repo's elapsed-gated `llm_retry` wrapper
# (prediction_market._invoke_market_llm) is the SOLE retry layer, and leaving this unpinned
# inherits forecasting-tools' default of 2 with an UN-GATED `random.uniform(5, 10)` tenacity
# sleep — a large slice of PREDICTION_MARKET_TIMEOUT spent sleeping blind, which is exactly
# what llm_retry exists to eliminate. `temperature=None` defers reasoning models to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p left unset. Each
# litellm `timeout` sits ABOVE its elapsed-gated wall cap in constants.py, so the wall is the
# binding bound.
#
# Luna is the cheapest tier that saturates both tasks. The measured rate on this route was
# $0.20/M in and $1.20/M out — TWICE the $0.10/$0.60 the bake-off read off the models API on
# 2026-08-03, where a "50% off" badge was displayed that has since lapsed or never applied here. A
# live ranking call reconciled the true rates to 7 significant figures against OpenRouter's own
# `upstream_inference_cost` (26,250 in / 685 out / a 25% cache-WRITE surcharge on the input,
# `scratch/market_port_2026-08-04/QA_DRY_RUN.md`), so this is measured rather than quoted.
# 2026-09-22: gpt-5.6-luna -> gpt-6-luna (GPT-6 release), now $0.10/$0.50 per 1M; the cost figures
# below predate that swap and are receipts, not current pricing.
#
# MEASURED cost per question: ranker $0.0074 (26k in at the median post-enrichment,
# full-PredictIt shape + ~685 out, cache write included); author ~1.4k in + ~300 out ≈ $0.0005.
# The two keyword calls they replace measured ~170 tok in / ~50 out ≈ $0.0001, so net new is
# ≈ +$0.008 per question — under a cent per run at the prod shape of 1-2 questions, and ~$0.24
# of ranker spend across a 30-question tournament run. The earlier ~$0.003-0.004 arithmetic in
# the port plan understated by 2.4x purely because of the promo price; the token shapes were
# right. This traffic is ~97% input, so the input rate is the whole cost.

# Prediction-market RANKER: one call per question over the whole ~380-440-candidate pool,
# emitting up to 8 ranked rows with a relation tier and a one-phrase label. Measured completion
# averages 589 tokens including reasoning, max 1,042 (scratch/bakeoff_run_2026-08-03/results/
# RANKED_ARM_RESULTS.md). No max_tokens since 2026-09-22 (operator): a TRUNCATED ranking is a
# fail-open that loses the whole ranking, and MARKET_RANKER_WALL_TIMEOUT already bounds a runaway.
MARKET_RANKER_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-6-luna",
    "role": "market_ranker",
    "temperature": None,
    "reasoning_effort": "low",
    "timeout": 90,
    "allowed_tries": 1,
}

# Prediction-market QUERY AUTHOR: one call per question emitting the domain vocabulary the
# question's own tokens cannot reach (up to 8 synonyms + 3 framings). Its output is ADDITIVE to
# a deterministic query set, so its failure costs recall nothing. Measured completion max 588
# tokens including reasoning. No max_tokens since 2026-09-22: MARKET_QUERY_AUTHOR_WALL_TIMEOUT bounds it.
MARKET_QUERY_AUTHOR_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-6-luna",
    "role": "market_query_author",
    "temperature": None,
    "reasoning_effort": "low",
    "timeout": 45,
    "allowed_tries": 1,
}


# Tier-B auxiliary: read-and-synthesize work that needs taste but not deep
# reasoning. Identifies the crux of forecaster disagreement; output text seeds
# the targeted-search query downstream. Runs under CRUX_SOFT_DEADLINE;
# effort deliberately low since 2026-05-20 for latency — the tier was upgraded
# instead (smarter-model-at-lower-effort beats more effort on a smaller model).
# 2026-07-17: sol→terra per the role audit; terra 2nd (sol 3rd) at -49% cost;
# the role fires rarely (stacking disabled in prod).
# allowed_tries=1 (Round-2): the crux-analyzer invoke is wrapped in the broad,
# elapsed-gated retry (targeted.extract_disagreement_crux) to impose the universal
# "never retry a slow failure" deadline rule on the conditional-stacking critical path.
# Per-instance override so PARSER_LLM keeps its allowed_tries=3.
# 2026-09-22: terra -> gpt-6-sol. Terra has no GPT-6 successor, so every Terra role
# moves to Sol 6 at the same (low) effort it ran at.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-6-sol",
    role="crux_analyzer",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
