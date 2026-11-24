"""Deterministic-first extraction ladder for forecast values.

Forecaster (and stacker) LLMs emit their forecast exactly once: a fenced ```json STRUCTURED
FORECAST block as the LAST thing in the rationale. This module extracts the value with a
four-rung ladder: **block** (``parse_structured_payload``), **repair** (``json_repair`` of a
malformed fenced block, or a balanced-braces scan of the rationale tail when no fence survived),
**llm** (``parse_structured`` over the full rationale, as salvage), then ``ValueExtractionError``,
so the caller drops the forecaster exactly as parser failures propagated before the ladder.

Every rung's output must be a value the rationale could have STATED. The LLM rung decodes under
a schema and cannot express "absent", so the post-rung validators are FIDELITY checks (finite,
ordered, in bounds, on the question's option set), and the repair rung may neither invent nor
drop a numeric value (``_repair_infidelity_reason``). The two deterministic rungs run
CANDIDATE-major: for each candidate in selection order both the strict parse and the repair are
tried before a lower-ranked candidate, so a malformed final block beats a superseded valid draft.

Every successful extraction emits one ``EXTRACTION_RUNG`` INFO line; ``rung=llm`` and
``block_present=False`` are the drift signals. The per-type output contracts (raw pre-clamp
binary decimal, ``McForecast``, percentiles for ``sanitize_percentiles``, epoch-second percentiles
for dates, the ``N + 2`` ``PmfForecast`` vector for per-bin grids) and the full rationale are in
docs/value_extraction.md.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from forecasting_tools import BinaryPrediction, GeneralLlm, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile
from json_repair import repair_json
from pydantic import ValidationError

from metaculus_bot.constants import PMF_ABOVE_RANGE_KEY, PMF_BELOW_RANGE_KEY
from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.mc_processing import (
    accumulate_declared_option_probs,
    build_mc_prediction,
    clamp_and_renormalize_probs,
    fold_option_label,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.date_axis import to_epoch
from metaculus_bot.numeric.pmf_grid import PmfGrid, fold_bin_label
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.structured_output_schema import (
    _MAX_STRUCTURED_BLOCK_BYTES,
    _MC_OPTION_PROB_SUM_TOLERANCE,
    BinaryStructured,
    BlockType,
    DateStructured,
    MultipleChoiceStructured,
    NumericStructured,
    PmfStructured,
    StructuredBlock,
    extract_json_block_candidates,
    iter_balanced_braces,
    parse_structured_payload,
    pmf_prob_sum_tolerance,
)
from metaculus_bot.structured_parse import BinProbability, IsoDatePercentile, parse_structured

logger = logging.getLogger(__name__)

Rung = Literal["block", "repair", "llm"]

# Why: the block is prompted LAST, so a lost fence still leaves the payload in the rationale tail.
_TAIL_SCAN_CHARS = 4000
# Why: JSON round-trips leave 0.1 as 0.10000000001, so percentile keys match STANDARD_PERCENTILES with slack.
_PERCENTILE_KEY_TOLERANCE = 1e-6

# --- Repair-rung fidelity ---------------------------------------------------

# Why: refuse truncated forms json_repair completes by inventing digits; see docs/value_extraction.md "The ladder".
_COMPLETE_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?$")
# Why: single-quoted output is the commonest malformation; see docs/value_extraction.md "The ladder: design notes".
_JSON_STRING_RE = re.compile(
    r'"(?:[^"\\]|\\.)*(?:"|\Z)' r"|'(?:[^'\\]|\\.)*(?:'|\Z)",
    re.DOTALL,
)
# Why: the loose body set swallows "1-2" into one token, so a malformed run surfaces as incomplete, not as two numbers.
_NUMBER_RUN_RE = re.compile(r"[-+.0-9][0-9.eE+-]*")


def _numeric_tokens_outside_strings(text: str) -> list[str]:
    """Numeric-literal runs sitting OUTSIDE string literals, in order.

    String contents are skipped because structured blocks carry prose fields
    (``ref_class``, evidence descriptions) where "3 of 4 cases." or "2019-2023"
    would otherwise read as a truncated number. Substituting an empty quoted pair
    keeps the tokens adjacent to a blanked string from merging into one run.
    """
    return _NUMBER_RUN_RE.findall(_JSON_STRING_RE.sub('""', text))


def _repair_infidelity_reason(candidate: str, repaired: str) -> str | None:
    """Why this ``json_repair`` output cannot be trusted, or None when it can.

    ``json_repair`` fixes SYNTAX, but it also completes a truncated VALUE and collapses a
    repeated key to its last value, and either result is indistinguishable from a declaration
    once it parses. Two rules keep the repair rung a repairer rather than an author: a raw
    candidate carrying an incomplete numeric literal is refused outright, and the repaired
    payload's numeric values must be exactly the raw candidate's, as a multiset. Detail:
    docs/value_extraction.md "The ladder".
    """
    candidate_tokens = _numeric_tokens_outside_strings(candidate)
    incomplete = [token for token in candidate_tokens if not _COMPLETE_NUMBER_RE.match(token)]
    if incomplete:
        return f"raw candidate carries truncated numeric literal(s) {incomplete}; repair would invent digits"

    repaired_tokens = _numeric_tokens_outside_strings(repaired)
    malformed = [token for token in repaired_tokens if not _COMPLETE_NUMBER_RE.match(token)]
    if malformed:
        return f"repaired payload carries malformed numeric literal(s) {malformed}"

    candidate_values = Counter(float(token) for token in candidate_tokens)
    repaired_values = Counter(float(token) for token in repaired_tokens)
    invented = sorted((repaired_values - candidate_values).elements())
    if invented:
        return f"repair introduced numeric value(s) {invented} absent from the raw candidate"
    dropped = sorted((candidate_values - repaired_values).elements())
    if dropped:
        return f"repair dropped numeric value(s) {dropped} present in the raw candidate"
    return None


@dataclass(frozen=True)
class TailProbability:
    """One optional tail declaration, keeping an invalid stated number for telemetry."""

    value: float | None = None
    status: Literal["valid", "missing", "invalid"] = "missing"


@dataclass(frozen=True)
class TailProbabilities:
    """Per-side declarations from the same accepted block as the percentiles."""

    below: TailProbability = field(default_factory=TailProbability)
    above: TailProbability = field(default_factory=TailProbability)


def _tail_probability(value: float | None, *, declared: bool) -> TailProbability:
    if not declared:
        return TailProbability()
    valid = value is not None and math.isfinite(value) and 0.0 <= value < 1.0
    return TailProbability(value=value, status="valid" if valid else "invalid")


def _tail_probabilities_from_block(
    block: StructuredBlock, *, open_lower_bound: bool, open_upper_bound: bool
) -> TailProbabilities:
    if not isinstance(block, (NumericStructured, DateStructured)):
        raise TypeError(f"expected a percentile block, got {type(block).__name__}")
    below = _tail_probability(block.prob_below_lower_bound, declared="prob_below_lower_bound" in block.model_fields_set)
    above = _tail_probability(block.prob_above_upper_bound, declared="prob_above_upper_bound" in block.model_fields_set)
    if open_lower_bound and open_upper_bound and below.status == "valid" and above.status == "valid":
        assert below.value is not None
        assert above.value is not None
        if below.value + above.value >= 1.0:
            below = TailProbability(value=below.value, status="invalid")
            above = TailProbability(value=above.value, status="invalid")
    return TailProbabilities(below=below, above=above)


@dataclass
class ExtractionOutcome[T]:
    """A validated forecast value plus which ladder rung produced it."""

    value: T
    rung: Rung
    block_present: bool
    tail_probabilities: TailProbabilities | None = None


@dataclass
class _DeterministicHit[T]:
    """A value recovered from ONE candidate body, plus the rung that produced it."""

    value: T
    rung: Rung
    tail_probabilities: TailProbabilities | None = None


def _log_extraction(
    qtype: BlockType,
    rung: Rung,
    *,
    block_present: bool,
    question_id: int | None,
    model_name: str,
) -> None:
    logger.info(
        "EXTRACTION_RUNG: question=%s model=%s qtype=%s rung=%s block_present=%s",
        question_id,
        model_name,
        qtype,
        rung,
        block_present,
    )


def _try_candidate[T](
    candidate: str,
    *,
    qtype: BlockType,
    convert_block: Callable[[StructuredBlock], T],
    validate: Callable[[T], T],
    try_strict: bool,
    label: str,
    failures: list[str],
    question_id: int | None,
    extract_tail_probabilities: Callable[[StructuredBlock], TailProbabilities] | None,
) -> _DeterministicHit[T] | None:
    """Strict-parse then ``json_repair`` ONE candidate body; None when neither yields a value.

    Both deterministic mechanisms hit the same candidate before the caller moves
    on, so a malformed block that is plausibly the model's final answer gets
    repaired instead of losing to a lower-ranked valid one.

    ``try_strict=False`` for unfenced tail blobs: there is no block to have
    parsed, so they are reported as ``rung="repair"`` even when the blob happens
    to be well-formed JSON.

    Candidates are probed with ``log_failures=False``: any single one may be a
    junk recap that a lower-ranked candidate recovers from, so the reasons are
    accumulated into ``failures`` (surfaced by the rung-3 log or the ladder
    error) rather than each emitting its own WARNING.
    """
    if try_strict:
        strict = parse_structured_payload(candidate, qtype, log_failures=False)
        if strict is not None:
            try:
                return _DeterministicHit(
                    value=validate(convert_block(strict)),
                    rung="block",
                    tail_probabilities=extract_tail_probabilities(strict) if extract_tail_probabilities else None,
                )
            except (ValueError, TypeError) as exc:
                # Why: json_repair cannot alter valid JSON; see docs/value_extraction.md "The ladder: design notes".
                failures.append(f"block: {label}: {exc}")
                return None

    if len(candidate) > _MAX_STRUCTURED_BLOCK_BYTES:
        failures.append(f"repair: {label}: exceeds size cap; refusing to repair")
        return None
    repaired = repair_json(candidate)
    if not (isinstance(repaired, str) and repaired.strip()):
        failures.append(f"repair: {label}: json_repair produced no usable output")
        return None
    infidelity = _repair_infidelity_reason(candidate, repaired)
    if infidelity is not None:
        failures.append(f"repair: {label}: {infidelity}")
        return None
    payload_model = parse_structured_payload(repaired, qtype, log_failures=False)
    if payload_model is None:
        failures.append(f"repair: {label}: repaired JSON failed schema validation")
        return None
    try:
        value = validate(convert_block(payload_model))
    except (ValueError, TypeError) as exc:
        failures.append(f"repair: {label}: {exc}")
        return None
    if repaired != candidate:
        logger.info(
            "json_repair modified candidate for qtype=%s question=%s (len %d -> %d)",
            qtype,
            question_id,
            len(candidate),
            len(repaired),
        )
    return _DeterministicHit(
        value=value,
        rung="repair",
        tail_probabilities=extract_tail_probabilities(payload_model) if extract_tail_probabilities else None,
    )


async def _run_ladder[T](
    *,
    text: str,
    qtype: BlockType,
    convert_block: Callable[[StructuredBlock], T],
    validate: Callable[[T], T],
    llm_extract: Callable[[], Awaitable[T]],
    question_id: int | None,
    model_name: str,
    extract_tail_probabilities: Callable[[StructuredBlock], TailProbabilities] | None = None,
) -> ExtractionOutcome[T]:
    """Shared rung driver. ``convert_block``/``validate`` raise ValueError to fail a rung."""
    failures: list[str] = []
    fenced = extract_json_block_candidates(text)
    block_present = bool(fenced)

    # --- Rungs 1+2: deterministic walk over candidates, best-first ---------

    # Why: candidate-major, or a superseded draft wins; see docs/value_extraction.md "The ladder: design notes".
    walk: list[tuple[str, bool]]
    if block_present:
        walk = [(candidate, True) for candidate in fenced]
    else:
        logger.info("No fenced JSON block in rationale for qtype=%s question=%s", qtype, question_id)
        failures.append("block: no fenced JSON block")
        # Why: no fence survived, so rescue bare JSON from the tail, LAST first; repair-only (see _try_candidate).
        tail_blobs = list(iter_balanced_braces(text[-_TAIL_SCAN_CHARS:]))
        if not tail_blobs:
            failures.append("repair: no candidate JSON in rationale tail")
        walk = [(blob, False) for blob in reversed(tail_blobs)]

    for rank, (candidate, try_strict) in enumerate(walk):
        hit = _try_candidate(
            candidate,
            qtype=qtype,
            convert_block=convert_block,
            validate=validate,
            try_strict=try_strict,
            label=f"candidate {rank + 1}/{len(walk)}",
            failures=failures,
            question_id=question_id,
            extract_tail_probabilities=extract_tail_probabilities,
        )
        if hit is None:
            continue
        if rank > 0:
            # Why: a non-last hit may not be the final block; see docs/value_extraction.md "The ladder: design notes".
            logger.info(
                "BLOCK_FALLBACK: question=%s model=%s qtype=%s skipped=%d rung=%s reasons=%s",
                question_id,
                model_name,
                qtype,
                rank,
                hit.rung,
                " | ".join(failures),
            )
        _log_extraction(qtype, hit.rung, block_present=block_present, question_id=question_id, model_name=model_name)
        return ExtractionOutcome(
            value=hit.value, rung=hit.rung, block_present=block_present, tail_probabilities=hit.tail_probabilities
        )

    # --- Rung 3: LLM parser salvage ---------------------------------------
    try:
        value = validate(await llm_extract())
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # terminal rung: fold ANY parser failure into the typed ladder error so callers see one exception type
        failures.append(f"llm: {type(exc).__name__}: {exc}")
    else:
        logger.warning(
            "EXTRACTION_RUNG=llm salvage: question=%s model=%s qtype=%s block_present=%s "
            "(deterministic rungs failed: %s)",
            question_id,
            model_name,
            qtype,
            block_present,
            " | ".join(failures),
        )
        _log_extraction(qtype, "llm", block_present=block_present, question_id=question_id, model_name=model_name)
        return ExtractionOutcome(
            value=value,
            rung="llm",
            block_present=block_present,
            tail_probabilities=TailProbabilities() if extract_tail_probabilities else None,
        )

    # --- Rung 4: typed failure --------------------------------------------
    raise ValueExtractionError(
        f"All extraction rungs failed for qtype={qtype} question={question_id} model={model_name}: "
        + " | ".join(failures)
    )


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------


def _binary_from_block(block: StructuredBlock) -> float:
    if not isinstance(block, BinaryStructured):
        raise ValueError(f"expected binary block, got {type(block).__name__}")
    return float(block.posterior_prob)


def _validate_binary(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"binary probability {value} is not finite")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"binary probability {value} outside [0, 1]")
    return value


async def extract_binary(
    text: str,
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
) -> ExtractionOutcome[float]:
    """Extract the RAW decimal probability (pre-clamp; caller applies the binary clamp)."""

    async def _llm() -> float:
        prediction: BinaryPrediction = await parse_structured(
            text, BinaryPrediction, parser_llm, prompt_notes=prompt_notes
        )
        return float(prediction.prediction_in_decimal)

    return await _run_ladder(
        text=text,
        qtype="binary",
        convert_block=_binary_from_block,
        validate=_validate_binary,
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


def _numeric_from_block(block: StructuredBlock) -> list[Percentile]:
    if not isinstance(block, NumericStructured) or not block.declared_percentiles:
        raise ValueError("block lacks declared_percentiles")
    # Why: absorbs the old numeric_format_router F5 fallback; see docs/value_extraction.md "The ladder".
    return [
        Percentile(percentile=float(pct), value=float(val)) for pct, val in sorted(block.declared_percentiles.items())
    ]


def _validate_numeric(percentiles: list[Percentile]) -> list[Percentile]:
    """Require every ``STANDARD_PERCENTILES`` entry; return exactly that set, never padded.

    Beyond presence, the values must be FINITE and ordered the way their labels
    claim. Both checks exist because the sanitizer downstream cannot tell a bad
    salvage from a concentrated forecast: ``sort_by_percentile_level`` orders by
    label, so a value-disordered set is never reordered — it is force-monotonized,
    which on one out-of-place value pins most of the set at a bound and publishes
    a distribution nobody declared. A strict DECREASE with rising percentile is
    incoherent by construction, so it fails the rung instead. Ties are allowed:
    a repeated value is a legitimate concentrated (often count-like) declaration,
    and the cluster spreader exists to separate exactly those.
    """
    matched: dict[float, Percentile] = {}
    for standard in STANDARD_PERCENTILES:
        for p in percentiles:
            if abs(float(p.percentile) - standard) <= _PERCENTILE_KEY_TOLERANCE:
                matched[standard] = p
                break
    missing = [s for s in STANDARD_PERCENTILES if s not in matched]
    if missing:
        raise ValueError(
            f"missing standard percentiles {missing}; got {sorted(float(p.percentile) for p in percentiles)}"
        )
    ordered = [matched[s] for s in STANDARD_PERCENTILES]
    non_finite = [(float(p.percentile), float(p.value)) for p in ordered if not math.isfinite(float(p.value))]
    if non_finite:
        raise ValueError(f"non-finite percentile value(s) {non_finite}")
    for previous, current in pairwise(ordered):
        if float(current.value) < float(previous.value):
            raise ValueError(
                f"value {float(current.value)} at percentile {float(current.percentile)} is below "
                f"{float(previous.value)} at percentile {float(previous.percentile)}; "
                "value-disordered percentiles cannot be trusted as a salvage"
            )
    return ordered


async def extract_numeric(
    text: str,
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
    open_lower_bound: bool = True,
    open_upper_bound: bool = True,
) -> ExtractionOutcome[list[Percentile]]:
    """Extract the ``STANDARD_PERCENTILES`` set (caller feeds it to ``sanitize_percentiles``).

    ``prompt_notes`` should be ``build_parse_notes(question)`` so the rung-3
    parser keeps today's bound-aware extraction instructions.
    """

    async def _llm() -> list[Percentile]:
        return await parse_structured(text, list[Percentile], parser_llm, prompt_notes=prompt_notes)

    return await _run_ladder(
        text=text,
        qtype="numeric",
        convert_block=_numeric_from_block,
        validate=_validate_numeric,
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
        extract_tail_probabilities=lambda block: _tail_probabilities_from_block(
            block, open_lower_bound=open_lower_bound, open_upper_bound=open_upper_bound
        ),
    )


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------


def _date_from_block(block: StructuredBlock) -> list[Percentile]:
    """The date block's ISO dates as epoch-second ``Percentile``s, ready for the numeric pipeline."""
    if not isinstance(block, DateStructured):
        raise ValueError(f"expected date block, got {type(block).__name__}")
    return [
        Percentile(percentile=float(pct), value=to_epoch(moment))
        for pct, moment in sorted(block.declared_percentiles.items())
    ]


def _epoch_percentiles(dates: Sequence[IsoDatePercentile]) -> list[Percentile]:
    return [Percentile(percentile=float(item.percentile), value=to_epoch(item.value)) for item in dates]


# Why: on the epoch axis a date forecast IS a numeric forecast; see docs/value_extraction.md "The ladder".
_validate_date = _validate_numeric


async def extract_date(
    text: str,
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
    open_lower_bound: bool = True,
    open_upper_bound: bool = True,
) -> ExtractionOutcome[list[Percentile]]:
    """Extract a date question's ``STANDARD_PERCENTILES`` set as EPOCH SECONDS (UTC floats).

    Mirrors ``extract_numeric``: the caller hands ``outcome.value`` to the same guarded numeric
    distribution build, on the question's ``numeric.date_axis.as_epoch_question`` view. Every
    rung converts through one parser (``numeric.date_axis.parse_forecast_date``): the block rung via
    ``DateStructured``, the LLM salvage rung via ``IsoDatePercentile``, so a date-only value lands
    at noon UTC inside its day bin on both paths and a naive timestamp is never read as host
    time. ``prompt_notes`` should be the date sibling of ``build_parse_notes`` so the rung-3
    parser keeps the ISO format and open-bound instructions.
    """

    async def _llm() -> list[Percentile]:
        dates: list[IsoDatePercentile] = await parse_structured(
            text, list[IsoDatePercentile], parser_llm, prompt_notes=prompt_notes
        )
        return _epoch_percentiles(dates)

    return await _run_ladder(
        text=text,
        qtype="date",
        convert_block=_date_from_block,
        validate=_validate_date,
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
        extract_tail_probabilities=lambda block: _tail_probabilities_from_block(
            block, open_lower_bound=open_lower_bound, open_upper_bound=open_upper_bound
        ),
    )


# ---------------------------------------------------------------------------
# Multiple choice
# ---------------------------------------------------------------------------


@dataclass
class McForecast:
    """A multiple-choice extraction: the constructible option list beside the probabilities as declared.

    ``option_list`` is the pre-``clamp_and_renormalize_mc`` ``PredictedOptionList`` the
    runner publishes from. ``declared_probs`` is the SAME options' probabilities as the
    block or the parser declared them, in ``question.options`` order, BEFORE the
    pre-construction ``clamp_and_renormalize_probs`` (which exists so ft 0.2.92's
    ``PredictedOptionList`` validator is a no-op). The list can never carry them, since it
    is clamped on construction, and they are what the MEMBER_FORECAST marker's ``raw``
    field records. Every rung of ``extract_mc`` produces both halves from one ordered
    source, so they align index for index by construction.
    """

    option_list: PredictedOptionList
    declared_probs: list[float]


def _parsed_mc_forecast(raw_options: Sequence[OptionProbability], options: Sequence[str]) -> McForecast:
    """The LLM rung's product: ``build_mc_prediction``'s clamped list beside the SAME
    accumulated pairs unclamped, so the declared vector aligns with it by construction."""
    declared = [prob for _, prob in accumulate_declared_option_probs(raw_options, options)]
    return McForecast(build_mc_prediction(raw_options, options), declared)


def _make_mc_from_block(options: list[str]) -> Callable[[StructuredBlock], McForecast]:
    def _mc_from_block(block: StructuredBlock) -> McForecast:
        if not isinstance(block, MultipleChoiceStructured):
            raise ValueError(f"expected multiple_choice block, got {type(block).__name__}")
        # Why: build_mc_prediction is bypassed: its _normalize_name would strip "Option " (see fold_option_label).
        canonical_by_norm = {fold_option_label(opt): opt for opt in options}
        matched: dict[str, float] = {}
        for key, prob in block.option_probs.items():
            canonical = canonical_by_norm.get(fold_option_label(key))
            if canonical is None:
                raise ValueError(f"block option {key!r} does not match any question option {options}")
            matched[canonical] = matched.get(canonical, 0.0) + float(prob)
        total = sum(matched.values())
        if total <= 0:
            raise ValueError(f"block option probabilities sum to {total}")
        ordered = [(name, matched[name]) for name in options if name in matched]
        declared = [prob for _, prob in ordered]
        # Why: clamp before constructing, so ft's PredictedOptionList validator is a no-op; see McForecast.
        clamped = clamp_and_renormalize_probs(declared)
        option_list = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=prob)
                for (name, _), prob in zip(ordered, clamped, strict=True)
            ]
        )
        return McForecast(option_list, declared)

    return _mc_from_block


def _make_validate_mc(options: list[str]) -> Callable[[McForecast], McForecast]:
    def _validate_mc(forecast: McForecast) -> McForecast:
        pol = forecast.option_list
        names = [o.option_name for o in pol.predicted_options]
        if set(names) != set(options):
            raise ValueError(f"option set mismatch: got {names}, expected {options}")
        for option in pol.predicted_options:
            if not (0.0 <= option.probability <= 1.0):
                raise ValueError(f"option {option.option_name!r} probability {option.probability} outside [0, 1]")
        total = sum(o.probability for o in pol.predicted_options)
        if abs(total - 1.0) > _MC_OPTION_PROB_SUM_TOLERANCE:
            raise ValueError(f"option probabilities sum to {total}, outside 1.0 ± {_MC_OPTION_PROB_SUM_TOLERANCE}")
        return forecast

    return _validate_mc


async def extract_mc(
    text: str,
    options: list[str],
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
) -> ExtractionOutcome[McForecast]:
    """Extract the option list mapped onto ``options`` (pre-``clamp_and_renormalize_mc``) beside
    the probabilities as declared; see ``McForecast``."""
    options = list(options)

    async def _llm() -> McForecast:
        """The pre-ladder two-stage tolerant parse: strict ``PredictedOptionList``, then the loose pair list."""
        # Why: both sub-paths route through build_mc_prediction, since the parser may return "option a".
        try:
            strict = await parse_structured(text, PredictedOptionList, parser_llm, prompt_notes=prompt_notes)
            as_raw = [
                OptionProbability(option_name=o.option_name, probability=o.probability)
                for o in strict.predicted_options
            ]
            # Why: ft clamps on construction, so this sub-path's declared vector is post-clamp, unlike the fallback's.
            return _parsed_mc_forecast(as_raw, options)
        except (ValidationError, ValueError) as exc:
            logger.warning("Primary MC parse failed in llm rung, using tolerant fallback: %s", exc)
            raw: list[OptionProbability] = await parse_structured(
                text, list[OptionProbability], parser_llm, prompt_notes=prompt_notes
            )
            return _parsed_mc_forecast(raw, options)

    return await _run_ladder(
        text=text,
        qtype="multiple_choice",
        convert_block=_make_mc_from_block(options),
        validate=_make_validate_mc(options),
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Per-bin PMF (the ``pmf`` block a coarse-grid question is elicited with)
# ---------------------------------------------------------------------------


@dataclass
class PmfForecast:
    """A per-bin extraction in the platform's own PMF shape.

    ``declared`` is ``[below, p_0, ..., p_{N-1}, above]`` (length ``N + 2``), as the block or the
    parser declared it and BEFORE the floor blend in ``numeric.pmf_cdf``; a closed bound's tail
    is 0.0. This is the shape Mantic itself exposes for every resolved forecast
    (``disagreement_forecasts.forecasts[].pmf``), so the ``MEMBER_FORECAST`` ``raw`` field and the
    platform's record of the same forecast read alike.
    """

    declared: list[float]


# Why: a reserved key absent from ``grid.keys`` names a CLOSED bound, since ``PmfGrid.keys`` carries it only when open.
_RESERVED_KEY_BOUNDS = {PMF_BELOW_RANGE_KEY: "lower", PMF_ABOVE_RANGE_KEY: "upper"}


def _pmf_from_pairs(pairs: Iterable[tuple[str, float]], grid: PmfGrid) -> PmfForecast:
    """Map label/probability pairs onto ``grid.keys`` the way ``_make_mc_from_block`` maps a ballot.

    Every value is range-checked BEFORE folding, so two out-of-range aliases of one key cannot
    cancel into a clean bin. Both sides fold through ``fold_bin_label`` (``"7.0"``, ``" 7 "`` and
    ``"55,000"`` land on ``"7"`` and ``"55000"``; a timestamp label's own fold differs from it, so
    the grid's keys are folded too), a duplicate fold sums onto one bin, an unmatched key fails, a
    reserved key on a closed bound fails, and EVERY grid key must be present: a block cut before its
    last bins repairs into a valid partial declaration, and the every-key rule is what stops that
    publishing. The block rung and the LLM rung share this one conversion.
    """
    canonical_by_fold = {fold_bin_label(key): key for key in grid.keys}
    matched: dict[str, float] = {}
    for key, prob in pairs:
        if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
            raise ValueError(f"block key {key!r} probability {prob} outside [0, 1]")
        folded = fold_bin_label(key)
        canonical = canonical_by_fold.get(folded)
        if canonical is None:
            closed_bound = _RESERVED_KEY_BOUNDS.get(folded)
            if closed_bound is not None:
                raise ValueError(f"block declares {key!r} but the question's {closed_bound} bound is closed")
            raise ValueError(f"block key {key!r} matches no bin of this grid")
        matched[canonical] = matched.get(canonical, 0.0) + float(prob)
    missing = [key for key in grid.keys if key not in matched]
    if missing:
        raise ValueError(f"block is missing bin(s) {missing}; a partial declaration cannot be published")
    total = sum(matched.values())
    if total <= 0:
        raise ValueError(f"block bin probabilities sum to {total}")
    below = matched.get(PMF_BELOW_RANGE_KEY, 0.0)
    above = matched.get(PMF_ABOVE_RANGE_KEY, 0.0)
    return PmfForecast([below, *(matched[label] for label in grid.labels), above])


def _make_pmf_from_block(grid: PmfGrid) -> Callable[[StructuredBlock], PmfForecast]:
    def _pmf_from_block(block: StructuredBlock) -> PmfForecast:
        if not isinstance(block, PmfStructured):
            raise ValueError(f"expected pmf block, got {type(block).__name__}")
        return _pmf_from_pairs(block.bin_probs.items(), grid)

    return _pmf_from_block


def _make_validate_pmf(grid: PmfGrid) -> Callable[[PmfForecast], PmfForecast]:
    def _validate_pmf(forecast: PmfForecast) -> PmfForecast:
        expected_length = len(grid.labels) + 2
        if len(forecast.declared) != expected_length:
            raise ValueError(f"pmf vector has {len(forecast.declared)} entries, expected {expected_length}")
        for index, prob in enumerate(forecast.declared):
            if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
                raise ValueError(f"pmf entry {index} probability {prob} outside [0, 1]")
        total = sum(forecast.declared)
        tolerance = pmf_prob_sum_tolerance(len(grid.keys))
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"pmf probabilities sum to {total}, outside 1.0 ± {tolerance} for {len(grid.keys)} keys")
        return forecast

    return _validate_pmf


async def extract_pmf(
    text: str,
    grid: PmfGrid,
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
    question_id: int | None = None,
    model_name: str = "",
) -> ExtractionOutcome[PmfForecast]:
    """Extract a per-bin declaration on ``grid`` as the platform's ``N + 2`` PMF vector; see ``PmfForecast``.

    Mirrors ``extract_mc``: the block rung reads ``PmfStructured.bin_probs``, the LLM salvage rung
    reads ``list[BinProbability]`` (``structured_parse``), and both run ``_pmf_from_pairs``.
    ``prompt_notes`` should be the per-bin sibling of ``build_parse_notes`` listing the grid's
    exact keys, since the salvage parser has nothing else to spell them from. The ladder logs
    ``EXTRACTION_RUNG ... qtype=pmf``: the marker's ``qtype`` is the BLOCK type the ladder parsed,
    so the question's own type is joined from the ``MEMBER_FORECAST`` line beside it.
    """

    async def _llm() -> PmfForecast:
        bins: list[BinProbability] = await parse_structured(
            text, list[BinProbability], parser_llm, prompt_notes=prompt_notes
        )
        return _pmf_from_pairs(((item.label, item.probability) for item in bins), grid)

    return await _run_ladder(
        text=text,
        qtype="pmf",
        convert_block=_make_pmf_from_block(grid),
        validate=_make_validate_pmf(grid),
        llm_extract=_llm,
        question_id=question_id,
        model_name=model_name,
    )


__all__ = [
    "ExtractionOutcome",
    "McForecast",
    "PmfForecast",
    "Rung",
    "extract_binary",
    "extract_date",
    "extract_mc",
    "extract_numeric",
    "extract_pmf",
]
