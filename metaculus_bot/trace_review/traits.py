"""Regex traits of one member's rationale: counts where a phrase is prompt scaffolding, booleans only
where a stricter pattern holds up.

The forecasting prompts put "reference class", a status-quo checklist line and a resolution-criteria
line into nearly every rationale, so a phrase-presence boolean for those is constant; they are
counts here. Prevalence moves with the prompt, so a trait comparison only means something within
one config era. Measurements and near misses: docs/trace_review.md "Text traits", pinned by
tests/data/trace_review_rationale_snippets.jsonl.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_PERCENT = r"(?<![\w.])\d{1,3}(?:\.\d+)?\s?%"
_PERCENT_RE = re.compile(_PERCENT)
_HEDGE_RE = re.compile(
    r"\b(?:uncertain(?:ty)?|unclear|(?:hard|difficult) to (?:say|know|predict)|cannot be (?:sure|certain|ruled out)"
    r"|can't rule out|may or may not|perhaps|might)\b",
    re.IGNORECASE,
)
_BASE_RATE_NUMBER_RE = re.compile(r"base rate[^\n]{0,120}?" + _PERCENT, re.IGNORECASE)
_STATUS_QUO_RE = re.compile(r"status[ -]quo", re.IGNORECASE)
_RESOLUTION_CRITERIA_RE = re.compile(r"resolution (?:criteria|source)", re.IGNORECASE)
_VENUE = r"(?:polymarket|kalshi|manifold|predictit|betfair|bookmakers?|betting (?:odds|markets?))"
_PRICE = r"\d+(?:\.\d+)?\s?(?:%|¢|cents?\b)"
_MARKET_PRICE_RE = re.compile(rf"{_VENUE}[^\n]{{0,100}}?{_PRICE}|{_PRICE}[^\n]{{0,100}}?{_VENUE}", re.IGNORECASE)
_MARKET_ABSENT_RE = re.compile(
    r"\bno\b[^.\n]{0,60}?prediction[ -]markets?"
    r"|prediction[ -]markets?[^.\n]{0,80}?\b(?:none|not found|unavailable)\b",
    re.IGNORECASE,
)
_CROWD_FORECAST_RE = re.compile(
    r"community (?:prediction|forecast|median)|metaculus (?:community|forecasters|median|crowd)"
    r"|crowd (?:forecast|median)|superforecaster",
    re.IGNORECASE,
)
# Not "bait-and-switch" or a bare "ambiguous": the first is a prompt checklist line, and with
# both terms the pattern fires on 46.9% of rationales.
_RESOLUTION_AMBIGUITY_RE = re.compile(
    r"\b(?:fine print|technicalit(?:y|ies)|(?:strict|literal) reading|as written|as worded|edge case)", re.IGNORECASE
)

CHARS_PER_THOUSAND = 1000.0


def rationale_chars(text: str) -> int:
    return len(text)


def percent_count(text: str) -> int:
    return len(_PERCENT_RE.findall(text))


def hedge_count(text: str) -> int:
    return len(_HEDGE_RE.findall(text))


def hedge_per_1k(text: str) -> float:
    """Hedges per thousand characters, so a long rationale does not read as a hedgy one."""
    return hedge_count(text) * CHARS_PER_THOUSAND / len(text) if text else 0.0


def base_rate_number_count(text: str) -> int:
    """ "base rate" followed on the same line, within 120 characters, by a percentage."""
    return len(_BASE_RATE_NUMBER_RE.findall(text))


def status_quo_count(text: str) -> int:
    return len(_STATUS_QUO_RE.findall(text))


def resolution_criteria_count(text: str) -> int:
    return len(_RESOLUTION_CRITERIA_RE.findall(text))


def market_price_cited(text: str) -> bool:
    """A betting venue within 100 characters of a price (%, ¢ or cents) on one line."""
    return _MARKET_PRICE_RE.search(text) is not None


def market_absent_stated(text: str) -> bool:
    """The rationale says, within one sentence, that no prediction market covers the question."""
    return _MARKET_ABSENT_RE.search(text) is not None


def crowd_forecast_cited(text: str) -> bool:
    return _CROWD_FORECAST_RE.search(text) is not None


def resolution_ambiguity_flagged(text: str) -> bool:
    return _RESOLUTION_AMBIGUITY_RE.search(text) is not None


TRAIT_KIND_COUNT = "count"
TRAIT_KIND_RATE = "rate"
TRAIT_KIND_FLAG = "flag"


@dataclass(frozen=True, slots=True)
class Trait:
    """One trait: its member column, its reader-facing label, its kind and the function computing it."""

    name: str
    label: str
    kind: str
    compute: Callable[[str], int | float | bool]


TRAITS: tuple[Trait, ...] = (
    Trait("rationale_chars", "Rationale length (characters)", TRAIT_KIND_COUNT, rationale_chars),
    Trait("percent_count", "Percentages quoted", TRAIT_KIND_COUNT, percent_count),
    Trait("hedge_count", "Hedge words", TRAIT_KIND_COUNT, hedge_count),
    Trait("hedge_per_1k", "Hedge words per 1,000 characters", TRAIT_KIND_RATE, hedge_per_1k),
    Trait("base_rate_number_count", "Base rates given with a number", TRAIT_KIND_COUNT, base_rate_number_count),
    Trait("status_quo_count", "Status-quo mentions", TRAIT_KIND_COUNT, status_quo_count),
    Trait("resolution_criteria_count", "Resolution-criteria mentions", TRAIT_KIND_COUNT, resolution_criteria_count),
    Trait("market_price_cited", "Cites a market price", TRAIT_KIND_FLAG, market_price_cited),
    Trait("market_absent_stated", "Says no market exists", TRAIT_KIND_FLAG, market_absent_stated),
    Trait("crowd_forecast_cited", "Cites a crowd forecast", TRAIT_KIND_FLAG, crowd_forecast_cited),
    Trait(
        "resolution_ambiguity_flagged", "Flags a resolution technicality", TRAIT_KIND_FLAG, resolution_ambiguity_flagged
    ),
)
TRAITS_BY_NAME: dict[str, Trait] = {trait.name: trait for trait in TRAITS}


def trait_values(text: str) -> dict[str, int | float | bool]:
    """Every trait of one rationale, keyed by member column."""
    return {trait.name: trait.compute(text) for trait in TRAITS}
