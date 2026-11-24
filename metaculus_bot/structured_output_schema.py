"""
Pydantic schemas for structured forecaster output blocks.

Base-forecaster LLMs (binary / multiple-choice / numeric / date) are asked to append
a fenced ```json block to their free-text rationale that declares structured
fields (prior, base rate, hazard, percentiles, scenarios, etc.). A post-hoc
tool runner extracts these blocks and feeds them to probabilistic tools
(Beta-binomial, log-pooling, distribution fitting).

This module defines the schemas and extraction helpers. Active surface,
gated by ``PROBABILISTIC_TOOLS_ENABLED`` env flag and per-question-type
via ``PROBABILISTIC_TOOLS_TYPES``. See ``metaculus_bot/tool_runner.py``
for dispatch and ``metaculus_bot/forecaster.py:_make_prediction`` for the
activation site.

Note: ``DiscreteCountStructured`` is defined here but not dispatched by the
current tool runner — discrete-count question dispatch is phase-3 work. The
class remains so that forecaster prompts can be updated first and the
runtime wiring can follow later.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Annotated, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from metaculus_bot.numeric.date_axis import parse_forecast_date
from metaculus_bot.question_types import QuestionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_HAZARD_FRACTION_TOLERANCE = 0.01
_SCENARIO_PROB_SUM_TOLERANCE = 0.02
_MC_OPTION_PROB_SUM_TOLERANCE = 0.02
# Why: a forecaster who rounds every probability to two decimals drifts up to 0.005 a key, and a pmf block has up to 33.
_PMF_PROB_SUM_TOLERANCE_PER_KEY = 0.005
_PMF_PROB_SUM_TOLERANCE_FLOOR = _MC_OPTION_PROB_SUM_TOLERANCE
_REQUIRED_NUMERIC_PERCENTILES: frozenset[float] = frozenset({0.1, 0.5, 0.9})


def pmf_prob_sum_tolerance(key_count: int) -> float:
    """How far a per-bin block's probabilities may sum from 1.0: the ballot's floor or the per-key drift, whichever is larger."""
    return max(_PMF_PROB_SUM_TOLERANCE_FLOOR, _PMF_PROB_SUM_TOLERANCE_PER_KEY * key_count)


# Why: an unclosed fence can swallow a transcript; see docs/value_extraction.md "Block schemas: design notes".
_MAX_STRUCTURED_BLOCK_BYTES: int = 200_000


def _reject_boolean_probability(value: object) -> object:
    """pydantic's lax float reads ``true`` as 1.0 and ``false`` as 0.0; neither is a declared probability."""
    if isinstance(value, bool):
        raise ValueError(f"a probability must be a number, got {value!r}")
    return value


# Why: the value-bearing probability fields; see docs/value_extraction.md "Block schemas: design notes".
_Probability = Annotated[float, BeforeValidator(_reject_boolean_probability)]


def _read_declared_tail_probability(value: object) -> float | None:
    """Keep numeric declarations for audit; unusable optional tails cannot discard percentiles."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        probability = float(value)
    except (ValueError, OverflowError):
        return None
    return probability if math.isfinite(probability) else None


_DeclaredTailProbability = Annotated[float | None, BeforeValidator(_read_declared_tail_probability)]


# ---------------------------------------------------------------------------
# Shared submodels
# ---------------------------------------------------------------------------


class StatedPrior(BaseModel):
    """A forecaster's declared outside-view prior (before updating on evidence)."""

    model_config = ConfigDict(extra="forbid")

    prob: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)


class StatedBaseRate(BaseModel):
    """Explicit k successes out of n trials in a declared reference class."""

    model_config = ConfigDict(extra="forbid")

    k: int
    n: int
    ref_class: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_k_n(self) -> StatedBaseRate:
        if self.n < 1:
            raise ValueError(f"StatedBaseRate.n must be >= 1, got {self.n}")
        if self.k < 0:
            raise ValueError(f"StatedBaseRate.k must be >= 0, got {self.k}")
        if self.k > self.n:
            raise ValueError(f"StatedBaseRate requires k <= n, got k={self.k}, n={self.n}")
        return self


class StatedHazard(BaseModel):
    """Constant-hazard model: rate per unit time plus the window length in the same units.

    Contract: ``rate_per_unit`` is expressed per ``unit`` (e.g., 0.25/day),
    and ``window_duration_units`` is the full forecast-window length in the
    SAME unit (e.g., 30 for "rate/day over a 30-day window"). Units cancel
    when the tool runner computes the survival integral, so no conversion
    to years (or any other canonical unit) is performed.

    ``elapsed_fraction`` + ``remaining_fraction`` describe how much of
    ``window_duration_units`` has already passed at forecast time.
    """

    model_config = ConfigDict(extra="forbid")

    rate_per_unit: float = Field(ge=0.0)
    unit: Literal["day", "week", "month", "year"]
    window_duration_units: float = Field(gt=0.0)
    elapsed_fraction: float = Field(ge=0.0, le=1.0)
    remaining_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_fractions_sum(self) -> StatedHazard:
        total = self.elapsed_fraction + self.remaining_fraction
        if abs(total - 1.0) > _HAZARD_FRACTION_TOLERANCE:
            raise ValueError(
                f"StatedHazard.elapsed_fraction + remaining_fraction must be ~1.0 "
                f"(tol {_HAZARD_FRACTION_TOLERANCE}), got {total}"
            )
        return self


class EvidenceItem(BaseModel):
    """A single piece of evidence with direction and strength.

    ``summary`` and ``direction`` are prompt-scaffolding: they structure the
    forecaster's reasoning but are not consumed by the numeric tool runner
    (which uses only ``strength`` and ``likelihood_ratio``). Keep them as
    required fields so prompts continue to demand explicit decomposition.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    direction: Literal["up", "down", "neutral"]
    strength: Literal["strong", "moderate", "weak"]
    likelihood_ratio: float | None = None

    @field_validator("likelihood_ratio")
    @classmethod
    def _check_lr(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"EvidenceItem.likelihood_ratio must be > 0 if set, got {v}")
        return v


class ScenarioBranch(BaseModel):
    """One branch of a declared scenario decomposition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    prob: float = Field(ge=0.0, le=1.0)
    conditional_outcome: str | None = None


class BaseRateAnchor(BaseModel):
    """The forecaster's stated outside-view base-rate range (archived blocks only).

    Prompted 2026-07-08 to 2026-09-02 and read by an anchor-overshoot telemetry line
    that was deleted with it; it never clamped or mutated a forecast (the 2026-07
    residual experiments buried the anchor-guard clamp for sign-flipping across eras).
    Retained so the 49 published comments carrying the field still strict-parse in
    ``performance_analysis``.
    """

    model_config = ConfigDict(extra="forbid")

    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> BaseRateAnchor:
        if self.low > self.high:
            raise ValueError(f"BaseRateAnchor requires low <= high, got low={self.low}, high={self.high}")
        return self


class CriteriaClause(BaseModel):
    """One priced resolution clause from the conjunctive-criteria table (archived blocks only).

    Same history as ``BaseRateAnchor``: prompted 2026-07-08 to 2026-09-02, read only by a
    clause-product divergence line that is gone, retained for the 12 published comments
    that carry it. The clause-pricing REASONING stays in the binary prompt's step 5b —
    what went is the JSON echo of the table.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    prob: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _readable_optional_float(value: object, *, low: float | None, high: float | None) -> float | None:
    """A finite float inside [low, high], or None when the declaration is unusable.

    The lenient half of the 2026-09-02 retirement of ``other_mass`` / ``concentration``.
    Both were inputs to a Dirichlet tool behind ``PROBABILISTIC_TOOLS_ENABLED`` (off in
    prod) and neither is prompted any more, so a value that fails its range is worth
    nothing and must cost nothing: it reads as absent instead of raising. The strict
    version cost a real forecast — on q45189 gemini wrote ``"concentration": 0.0`` beside
    a valid three-option ballot, the ``> 0`` check rejected the whole block, ``json_repair``
    cannot alter valid JSON, and MC has no telemetry strip-and-retry, so the ballot was
    re-read by the LLM salvage rung (``rung=llm`` in the extraction archive).

    None is the honest reading rather than a clamp: a value outside its own range means the
    model was not declaring the quantity we asked for, and inventing an in-range substitute
    for a dormant field would put a number nobody stated into the archive.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        # Why: pydantic would not convert an OverflowError; see docs/value_extraction.md "Block schemas: design notes".
        return None
    if not math.isfinite(number):
        return None
    if low is not None and number < low:
        return None
    if high is not None and number > high:
        return None
    return number


# Why: declared once; a restated copy fails asymmetrically; see docs/value_extraction.md "Block schemas: design notes".
NumericOutcomeType = Literal["discrete_integer", "continuous"]
_NUMERIC_OUTCOME_TYPES: tuple[str, ...] = get_args(NumericOutcomeType)


def _validate_scenario_sum(scenarios: list[ScenarioBranch]) -> list[ScenarioBranch]:
    if not scenarios:
        return scenarios
    total = sum(s.prob for s in scenarios)
    if abs(total - 1.0) > _SCENARIO_PROB_SUM_TOLERANCE:
        raise ValueError(
            f"Non-empty scenarios must have probs summing to ~1.0 (tol {_SCENARIO_PROB_SUM_TOLERANCE}), got {total}"
        )
    return scenarios


def _validate_probability_dict(v: dict[str, float], *, label: str, tolerance: float) -> dict[str, float]:
    """The probability-vector contract the ballot and the per-bin block share.

    Non-empty, no blank key, every value in [0, 1] (which also refuses NaN), and a sum within
    ``tolerance`` of 1.0. ``label`` names the field in the error text, which the archive sees.
    """
    if not v:
        raise ValueError(f"{label} must be non-empty")
    for key, prob in v.items():
        if not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings, got {key!r}")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"{label} values must be in [0, 1], got {prob}")
    total = sum(v.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{label} must sum to ~1.0 (tol {tolerance}), got {total}")
    return v


# ---------------------------------------------------------------------------
# Per-question-type models
# ---------------------------------------------------------------------------


class BinaryStructured(BaseModel):
    """Structured declaration for a binary question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["binary"]
    prior: StatedPrior | None = None
    base_rate: StatedBaseRate | None = None
    hazard: StatedHazard | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    scenarios: list[ScenarioBranch] = Field(default_factory=list)
    posterior_prob: _Probability = Field(ge=0.0, le=1.0)
    # Why: archived blocks only; see docs/value_extraction.md "Block schemas: design notes".
    base_rate_anchor: BaseRateAnchor | None = None
    criteria_clauses: list[CriteriaClause] = Field(default_factory=list)

    @field_validator("scenarios")
    @classmethod
    def _check_scenarios_sum(cls, v: list[ScenarioBranch]) -> list[ScenarioBranch]:
        return _validate_scenario_sum(v)


def _check_declared_percentiles[V: (float, datetime)](declared_percentiles: dict[float, V]) -> dict[float, V]:
    """The percentile-map contract both continuous block types share.

    Keys must include ``_REQUIRED_NUMERIC_PERCENTILES`` and sit in [0, 1]; values must be
    non-decreasing with the percentile. Non-decreasing rather than strictly increasing because
    ties are valid concentrated declarations (the cluster spreader separates them downstream).
    The prompt still requests strict increases; this is the safety net that rejects a DECREASE
    before the sanitizer orders by percentile level and would force-monotonize it into a
    distribution nobody declared. Generic over the value type so a date block's datetimes get
    the identical ordering check as a numeric block's floats.
    """
    missing_percentiles = _REQUIRED_NUMERIC_PERCENTILES - declared_percentiles.keys()
    if missing_percentiles:
        raise ValueError(
            f"declared_percentiles must include at least "
            f"{sorted(_REQUIRED_NUMERIC_PERCENTILES)}, missing {sorted(missing_percentiles)}"
        )
    for percentile_level in declared_percentiles:
        if not (0.0 <= percentile_level <= 1.0):
            raise ValueError(f"Percentile keys must be in [0, 1], got {percentile_level}")
    previous_value: V | None = None
    for percentile_level in sorted(declared_percentiles):
        current_value = declared_percentiles[percentile_level]
        if previous_value is not None and current_value < previous_value:
            raise ValueError(
                f"declared_percentiles values must be non-decreasing with percentile; "
                f"got {current_value} at pct {percentile_level} after {previous_value}"
            )
        previous_value = current_value
    return declared_percentiles


class NumericStructured(BaseModel):
    """Structured declaration for a numeric question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["numeric"]
    prior: StatedPrior | None = None
    declared_percentiles: dict[float, float] | None = None
    prob_below_lower_bound: _DeclaredTailProbability = None
    prob_above_upper_bound: _DeclaredTailProbability = None
    outcome_type: NumericOutcomeType | None = None
    scenarios: list[ScenarioBranch] = Field(default_factory=list)

    @field_validator("outcome_type", mode="before")
    @classmethod
    def _tolerate_unknown_outcome_type(cls, v: object, info: ValidationInfo) -> str | None:
        """An unrecognised spelling reads as absent instead of failing the whole block.

        ``outcome_type`` only gates discrete snapping, so a near-miss spelling costs the one parser
        call the field was meant to save rather than the whole forecast. Logged at WARNING with the
        raw value (a spelling the roster starts using is a prompt signal), or at DEBUG when the caller
        passed ``log_failures=False`` as validation context. Detail: docs/value_extraction.md
        "Block schemas: design notes".
        """
        if v is None:
            return None
        if v in _NUMERIC_OUTCOME_TYPES:
            return str(v)
        context = info.context or {}
        log = logger.warning if context.get("log_failures", True) else logger.debug
        log(
            "Unrecognised outcome_type %r in numeric structured block; reading it as absent "
            "(the discrete vote falls back to the parser call)",
            v,
        )
        return None

    @field_validator("declared_percentiles")
    @classmethod
    def _check_percentiles(cls, declared_percentiles: dict[float, float] | None) -> dict[float, float] | None:
        if not declared_percentiles:
            return declared_percentiles
        return _check_declared_percentiles(declared_percentiles)

    @model_validator(mode="after")
    def _require_percentiles(self) -> NumericStructured:
        if not self.declared_percentiles:
            raise ValueError(
                f"NumericStructured requires declared_percentiles with at least {sorted(_REQUIRED_NUMERIC_PERCENTILES)}"
            )
        return self

    @field_validator("scenarios")
    @classmethod
    def _check_scenarios_sum(cls, v: list[ScenarioBranch]) -> list[ScenarioBranch]:
        return _validate_scenario_sum(v)


class DateStructured(BaseModel):
    """Structured declaration for a date question: the 13 percentiles as ISO-8601 dates.

    The values are parsed by ``numeric.date_axis.parse_forecast_date`` and nothing else (a calendar
    date means noon UTC of that day, a timestamp is taken as written with a naive time read as UTC,
    every other spelling fails, and a non-string is refused so pydantic never reads ``2027`` as a
    1970 unix timestamp). That one parser is also the TRUNCATION GUARD for dates, since the repair
    rung's numeric-literal check cannot see a date cut inside a string. No ``outcome_type``: integer
    snapping on an epoch axis is meaningless. Detail: docs/value_extraction.md "Block schemas:
    design notes".
    """

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["date"]
    declared_percentiles: dict[float, datetime]
    prob_below_lower_bound: _DeclaredTailProbability = None
    prob_above_upper_bound: _DeclaredTailProbability = None

    @field_validator("declared_percentiles", mode="before")
    @classmethod
    def _parse_iso_dates(cls, declared: object) -> object:
        if not isinstance(declared, dict):
            return declared
        parsed: dict[object, datetime] = {}
        for percentile_level, value in declared.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"declared_percentiles[{percentile_level!r}] must be an ISO-8601 date string, got {value!r}"
                )
            parsed[percentile_level] = parse_forecast_date(value)
        return parsed

    @field_validator("declared_percentiles")
    @classmethod
    def _check_percentiles(cls, declared_percentiles: dict[float, datetime]) -> dict[float, datetime]:
        if not declared_percentiles:
            raise ValueError("DateStructured requires a non-empty declared_percentiles")
        return _check_declared_percentiles(declared_percentiles)


class MultipleChoiceStructured(BaseModel):
    """Structured declaration for a multiple-choice question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["multiple_choice"]
    prior: StatedPrior | None = None
    option_probs: dict[str, _Probability]
    # Why: archived blocks only, read leniently so a bad value cannot cost the ballot; see _readable_optional_float.
    other_mass: float | None = None
    concentration: float | None = None

    @field_validator("other_mass", mode="before")
    @classmethod
    def _tolerate_other_mass(cls, v: object) -> float | None:
        return _readable_optional_float(v, low=0.0, high=1.0)

    @field_validator("concentration", mode="before")
    @classmethod
    def _tolerate_concentration(cls, v: object) -> float | None:
        """A positive Dirichlet concentration, or None when the declaration is unusable."""
        # Why: no upper bound, since the widely-copied example value was 20.0.
        read = _readable_optional_float(v, low=None, high=None)
        return read if read is not None and read > 0.0 else None

    @field_validator("option_probs")
    @classmethod
    def _check_option_probs(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_probability_dict(
            v, label="MultipleChoiceStructured.option_probs", tolerance=_MC_OPTION_PROB_SUM_TOLERANCE
        )


class PmfStructured(BaseModel):
    """Per-bin declaration on an enumerable grid: one probability per bin label, plus the reserved
    ``below_range`` / ``above_range`` keys where the question's bound is open.

    Not a question type: a coarse-grid numeric or date question (``numeric.config.elicit_per_bin``)
    is elicited this way instead of as percentiles, so the block declares ``question_type: pmf``
    while the question keeps its own type everywhere else. The keys are the labels of
    ``numeric.pmf_grid.PmfGrid.keys``; matching them onto the grid is the extraction ladder's job
    (``value_extraction.extract_pmf``), so this schema checks only that the object is a probability
    vector: non-empty, string-keyed, every value in [0, 1], summing to about 1.0.
    """

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["pmf"]
    bin_probs: dict[str, _Probability]

    @field_validator("bin_probs")
    @classmethod
    def _check_bin_probs(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_probability_dict(v, label="PmfStructured.bin_probs", tolerance=pmf_prob_sum_tolerance(len(v)))


class DiscreteCountStructured(BaseModel):
    """Structured declaration for a discrete-count question."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["discrete_count"]
    prior: StatedPrior | None = None
    mean_estimate: float = Field(ge=0.0)
    dispersion: Literal["poisson", "negbinom", "beta_binom_ceiling"]
    ceiling: int | None = None
    overdispersion_factor: float | None = None
    declared_percentiles: dict[float, float] | None = None

    @field_validator("ceiling")
    @classmethod
    def _check_ceiling(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"DiscreteCountStructured.ceiling must be >= 1 if set, got {v}")
        return v

    @field_validator("overdispersion_factor")
    @classmethod
    def _check_overdispersion(cls, v: float | None) -> float | None:
        if v is not None and v <= 1:
            raise ValueError(f"DiscreteCountStructured.overdispersion_factor must be > 1 if set, got {v}")
        return v

    @model_validator(mode="after")
    def _check_dispersion_requirements(self) -> DiscreteCountStructured:
        if self.dispersion == "beta_binom_ceiling" and self.ceiling is None:
            raise ValueError("DiscreteCountStructured with dispersion='beta_binom_ceiling' requires ceiling to be set")
        return self


StructuredBlock = Annotated[
    BinaryStructured | NumericStructured | MultipleChoiceStructured | DateStructured | PmfStructured,
    Field(discriminator="question_type"),
]

# Why: pmf is an elicitation, not a question type; see docs/value_extraction.md "The ladder".
BlockType = QuestionType | Literal["pmf"]


# Why: ``DiscreteCountStructured`` is intentionally unmapped, phase-3 work; see the module docstring.
_QUESTION_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "binary": BinaryStructured,
    "numeric": NumericStructured,
    "multiple_choice": MultipleChoiceStructured,
    "date": DateStructured,
    "pmf": PmfStructured,
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Any fence with an optional language tag; the tag and body preferences are ranked in extract_json_block_candidates.
_FENCE_PATTERN = re.compile(
    r"```[ \t]*(?P<tag>[A-Za-z]*)[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```",
    re.DOTALL,
)


def extract_json_block_candidates(rationale_text: str) -> list[str]:
    """Fenced JSON-block bodies in SELECTION order (best candidate first).

    Tagged ```json fences outrank untagged fences whose body begins with ``{``, and within a tier
    the LAST block by document position ranks first, because the prompt asks for the STRUCTURED
    FORECAST block last. Empty bodies are skipped. Callers walk the list and keep the first body
    that validates (or, on the publish path, repairs). Detail: docs/value_extraction.md "The
    ladder: design notes".
    """
    if not rationale_text:
        return []

    tagged: list[str] = []
    untagged: list[str] = []
    for match in _FENCE_PATTERN.finditer(rationale_text):
        tag = match.group("tag").strip().lower()
        body = match.group("body").strip()
        if not body:
            continue
        if tag == "json":
            tagged.append(body)
        elif tag == "" and body.lstrip().startswith("{"):
            untagged.append(body)
    return [*reversed(tagged), *reversed(untagged)]


def extract_json_block(rationale_text: str) -> str | None:
    """The best-positioned fenced JSON block body, or None: ``extract_json_block_candidates``' first pick.

    Schema-blind, so it is for callers that need only a block's raw text (peeking at a self-declared
    ``question_type`` before the schema is known); a caller that knows the type should use
    ``parse_structured_block``, which keeps the first candidate that validates.
    """
    candidates = extract_json_block_candidates(rationale_text)
    return candidates[0] if candidates else None


def iter_balanced_braces(s: str) -> Iterator[str]:
    """Yield each top-level balanced ``{...}`` block in ``s``, in document order.

    String-literal-aware: braces inside JSON string literals are not counted,
    and backslash escapes are respected so ``"\\""`` does not terminate a
    string. This makes the scan safe on inputs like ``{"foo": "has a } brace"}``
    which a naive brace-counter would truncate. After closing one block the scan
    resumes AFTER it, so a rationale tail with several bare objects surfaces them
    all — the caller (the value-extraction repair rung) repairs+validates each
    and keeps the first that passes, rather than giving up on a junk leading
    blob (the same iterate-to-valid selection the fenced path uses).
    """
    idx = 0
    length = len(s)
    while idx < length:
        start_idx = s.find("{", idx)
        if start_idx == -1:
            return
        end_idx = _scan_to_matching_brace(s, start_idx)
        if end_idx is None:
            # Why: nothing after an unclosed "{" can close, so no later top-level block completes.
            return
        yield s[start_idx : end_idx + 1]
        idx = end_idx + 1


def _scan_to_matching_brace(s: str, start_idx: int) -> int | None:
    """Index of the ``}`` that closes the ``{`` at ``start_idx``, or None if unbalanced.

    String-literal-aware: braces inside JSON string literals are not counted, and
    backslash escapes are respected so ``"\\""`` does not terminate a string.
    """
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start_idx, len(s)):  # HARNESS-SCAN-EXEMPT-python-numeric-hotloop: character scan, not numeric data
        c = s[i]
        if escape_next:
            escape_next = False
        elif in_string:
            if c == "\\":
                escape_next = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_first_balanced_braces(s: str) -> str | None:
    """Return the first balanced ``{...}`` block in ``s``, or None if none exists.

    Thin wrapper over ``iter_balanced_braces`` (which yields every top-level
    balanced block); this returns only the first. Kept for callers that want a
    single blob — e.g. gap-list JSON salvage in ``research/targeted.py``.
    """
    return next(iter_balanced_braces(s), None)


def parse_structured_payload(
    raw_json: str,
    question_type: BlockType,
    *,
    log_failures: bool = True,
) -> StructuredBlock | None:
    """Validate a raw JSON payload string against the structured-block schemas, or None on any failure.

    Runs the size cap, the duplicate-key-refusing decode, the dict-shape check, the
    ``question_type`` inject-when-absent / mismatch guard, and ``model_validate`` (with the binary
    telemetry strip-and-retry); the calling ladder decides how to log and whether to fall through.
    ``log_failures`` gates the WARNING lines on the failure paths and is also handed down as
    validation context so a lenient validator can respect it; a caller probing several candidates
    passes False. ``"discrete_count"`` is intentionally unsupported at runtime (module docstring).
    Detail: docs/value_extraction.md "Block schemas: design notes".
    """
    payload = _decode_structured_payload(raw_json, question_type, log_failures=log_failures)
    if payload is None:
        return None

    model_cls = _QUESTION_TYPE_TO_MODEL[question_type]
    context: Mapping[str, object] = {"log_failures": log_failures}
    try:
        return model_cls.model_validate(payload, context=context)  # type: ignore[return-value]
    except ValidationError as exc:
        retry = _retry_without_binary_telemetry(model_cls, payload, question_type, exc, context=context)
        if retry is not None:
            return retry
        if log_failures:
            logger.warning(
                "Structured block failed validation for question_type=%s: %s",
                question_type,
                exc,
            )
        return None


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``json.loads`` hook: an object repeating a key is refused instead of keeping the last value.

    The default decoder is last-write-wins, so ``{"0": 0.9, "0": 0.4, "1": 0.6}`` would read as a
    valid vector nobody declared. Raising here fails the decode, the block rung falls through, and
    only the parser LLM, reading the prose, may resolve which value was meant.
    """
    duplicates = sorted(key for key, count in Counter(key for key, _ in pairs).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate key {', '.join(repr(key) for key in duplicates)}")
    return dict(pairs)


def _decode_structured_payload(
    raw_json: str,
    question_type: BlockType,
    *,
    log_failures: bool,
) -> dict | None:
    """Size-cap, decode, and shape-check a raw structured block into a dict.

    Returns None on any failure (over the byte cap, malformed JSON or a repeated key, a non-object
    payload, or a ``question_type`` that contradicts the caller's, ``null`` included). On success
    the expected ``question_type`` is injected when ABSENT, so the Pydantic discriminator resolves.
    """
    if len(raw_json) > _MAX_STRUCTURED_BLOCK_BYTES:
        if log_failures:
            logger.warning(
                "Structured block exceeds size cap (%d bytes > %d); refusing to parse (question_type=%s)",
                len(raw_json),
                _MAX_STRUCTURED_BLOCK_BYTES,
                question_type,
            )
        return None

    try:
        payload = json.loads(raw_json, object_pairs_hook=_object_without_duplicate_keys)
    # Why: json.JSONDecodeError is a ValueError, and the duplicate-key hook raises a bare one.
    except ValueError as exc:
        if log_failures:
            snippet = raw_json[:200].replace("\n", " ")  # HARNESS-SCAN-EXEMPT-subsampling: a log snippet, not a sample
            logger.warning(
                "Malformed JSON in structured block (question_type=%s): %s. Snippet: %s", question_type, exc, snippet
            )
        return None

    if not isinstance(payload, dict):
        if log_failures:
            logger.warning(
                "Structured block must decode to a JSON object, got %s (question_type=%s)",
                type(payload).__name__,
                question_type,
            )
        return None

    if "question_type" not in payload:
        return {**payload, "question_type": question_type}
    if payload["question_type"] != question_type:
        if log_failures:
            logger.warning(
                "question_type mismatch: arg=%s, payload=%s. Refusing to parse.",
                question_type,
                payload["question_type"],
            )
        return None
    return payload


def _retry_without_binary_telemetry(
    model_cls: type[BaseModel],
    payload: dict,
    question_type: str,
    exc: ValidationError,
    *,
    context: Mapping[str, object] | None = None,
) -> StructuredBlock | None:
    """Re-validate a failed BINARY block with only the two telemetry fields dropped.

    ``base_rate_anchor`` and ``criteria_clauses`` are telemetry nothing reads to mutate a forecast,
    so a malformed one (``criteria_clauses: null``, a reversed anchor) must not take a good
    ``posterior_prob`` down with it. Only those two fields are stripped, so an error in a core field
    still surfaces as None; unprompted since 2026-09-02, it survives for archived blocks and habit.
    Detail: docs/value_extraction.md "Block schemas: design notes".
    """
    telemetry_fields = {"base_rate_anchor", "criteria_clauses"}
    if question_type != "binary" or not telemetry_fields & payload.keys():
        return None

    stripped_keys = sorted(telemetry_fields & payload.keys())
    stripped_payload = {k: v for k, v in payload.items() if k not in telemetry_fields}
    try:
        retry = model_cls.model_validate(stripped_payload, context=context)
    except ValidationError:
        return None
    logger.warning(
        "Dropping malformed telemetry fields %s and keeping core binary block (original error: %s)",
        stripped_keys,
        exc,
    )
    return retry  # type: ignore[return-value]


def parse_structured_block(
    rationale_text: str,
    question_type: BlockType,
) -> StructuredBlock | None:
    """Extract and validate a structured JSON block from a rationale, or None.

    Validity-aware selection: candidates are walked best-first (``extract_json_block_candidates``)
    and the FIRST that validates for ``question_type`` wins, so a trailing schema-recap block
    cannot shadow a valid forecast earlier in the rationale. None when no fence exists (INFO) or no
    candidate validates (the last candidate's WARNING names the reason); a candidate recovered by
    a later valid one logs one INFO instead. Selection is STRICT-only, which is why the publish
    path (``value_extraction._run_ladder``) does not use it. Detail: docs/value_extraction.md
    "Block schemas: design notes".
    """
    candidates = extract_json_block_candidates(rationale_text)
    if not candidates:
        logger.info("No JSON block found in rationale for question_type=%s", question_type)
        return None

    last_index = len(candidates) - 1
    for index, candidate in enumerate(candidates):
        # Why: only the last candidate's failure is the honest end state; a valid block may still follow the others.
        parsed = parse_structured_payload(candidate, question_type, log_failures=index == last_index)
        if parsed is not None:
            if index > 0:
                logger.info(
                    "Structured-block selection skipped %d trailing block(s) that did not validate "
                    "for question_type=%s before a usable one; the model may be emitting blocks after "
                    "the forecast block (prompt block-last contract eroding).",
                    index,
                    question_type,
                )
            return parsed
    return None
