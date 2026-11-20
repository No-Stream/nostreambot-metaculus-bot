"""Every trait on every labelled rationale snippet equals its label.

The snippets (tests/data/trace_review_rationale_snippets.jsonl) are short excerpts of real member
rationales from the 2026-11-04 round, the bot's own sentences only, each labelled by reading it. They
include the near misses that broke the spec's phrase-presence traits: "No prediction market data
found", a bare "Expert and market priors" heading, "labor market", the "Bait-and-switch check"
checklist line, "base rate" with no number, an interest rate quoted as a percentage, and "was
written" inside a sentence about something else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metaculus_bot.trace_review.traits import TRAITS, TRAITS_BY_NAME, hedge_count, hedge_per_1k, trait_values

SNIPPETS_PATH = Path(__file__).parent.parent / "data" / "trace_review_rationale_snippets.jsonl"
SNIPPETS: list[dict] = [json.loads(line) for line in SNIPPETS_PATH.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("snippet", SNIPPETS, ids=[f"snippet{index}" for index in range(len(SNIPPETS))])
def test_each_trait_matches_its_label(snippet: dict) -> None:
    actual = {name: TRAITS_BY_NAME[name].compute(snippet["text"]) for name in snippet["expected"]}

    assert actual == snippet["expected"]


def test_the_labels_cover_every_regex_trait_and_every_trait_fires_somewhere() -> None:
    """A trait no snippet exercises would pass the parametrized test vacuously."""
    derived = {"rationale_chars", "hedge_per_1k"}
    labelled = set(SNIPPETS[0]["expected"])
    assert labelled == {trait.name for trait in TRAITS} - derived
    for name in labelled:
        assert any(snippet["expected"][name] for snippet in SNIPPETS), f"no snippet has {name} set"
        assert not all(snippet["expected"][name] for snippet in SNIPPETS), f"every snippet has {name} set"


def test_the_derived_traits_are_length_and_hedge_rate() -> None:
    text = next(snippet["text"] for snippet in SNIPPETS if snippet["expected"]["hedge_count"])
    values = trait_values(text)

    assert values["rationale_chars"] == len(text)
    assert values["hedge_per_1k"] == pytest.approx(hedge_count(text) * 1000 / len(text))
    assert hedge_per_1k("") == 0.0
