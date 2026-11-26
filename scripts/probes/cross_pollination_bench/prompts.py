"""Update-round prompts and neutral peer forecast summaries."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Variant = Literal["peers_full", "peers_forecasts", "second_look"]

UPDATE_ROUND_MARKER = "── UPDATE ROUND ──"
PEER_LABELS = ("Forecaster A", "Forecaster B")
CORRECTION_KINDS = ("fact", "criteria", "base_rate", "reweighting", "none")
CORRECTION_UNPARSED = "unparsed"

UPDATE_INSTRUCTION = """── UPDATE ROUND ──
The step-by-step output format in the instructions above no longer applies to this reply. The STRUCTURED FORECAST schema above still applies.

Reconsider your first answer against the research and the resolution criteria above. If answers from other forecasters appear above, they read exactly the same research you did, so their agreement with you or with each other adds no evidence. Weigh only a specific fact, a reading of the resolution criteria, or a calculation that you can check against the research above.

Write at most 300 words before the JSON block:
1. Any material correction to your first answer and its basis in the research or the resolution criteria. If there is none, say so.
2. One line in exactly this form:
UPDATE: prior=<first headline value> revised=<revised headline value> correction=<fact|criteria|base_rate|reweighting|none>
The headline value is the probability for a binary question, the probability of your most likely option for a multiple-choice question, and your median for a numeric question. fact = a fact you missed or misread; criteria = a different reading of the resolution criteria; base_rate = a different base rate; reweighting = the same evidence weighed differently; none = no correction.

Give your best forecast even if it is unchanged, and revise by as much as the correction warrants. End with exactly one fenced ```json block in the STRUCTURED FORECAST schema above. It must be the last thing you write."""


@dataclass(frozen=True)
class PeerAnswer:
    reply_text: str
    forecast_text: str


def peer_order(question_id: int, member: str, peers: Sequence[str], *, seed: int) -> list[str]:
    """Return a deterministic peer order shared by variants and draws for this member/question."""
    ordered_peers = list(peers)
    random.Random(f"{seed}:{question_id}:{member}").shuffle(ordered_peers)  # noqa: S311  # deterministic prompt order, not cryptography
    return ordered_peers


def render_forecast_values(
    qtype: str,
    forecast: float | Sequence[float],
    *,
    options: Sequence[str] | None,
    stated_percentiles: Sequence[Sequence[float]] | None,
) -> str:
    """Render only a peer's declared forecast values, without its reasoning."""
    if qtype == "binary":
        if not isinstance(forecast, float):
            raise ValueError("Binary peer value must be a probability")
        return f"Probability of Yes: {float(forecast):g}"
    if qtype == "multiple_choice":
        if options is None:
            raise ValueError("Multiple-choice peer values require option labels")
        if isinstance(forecast, (int, float)):
            raise ValueError("Multiple-choice peer values must be a probability sequence")
        probabilities = [float(probability) for probability in forecast]
        if len(probabilities) != len(options):
            raise ValueError(f"Expected {len(options)} multiple-choice probabilities, got {len(probabilities)}")
        return "\n".join(
            f"- {option}: {probability:g}" for option, probability in zip(options, probabilities, strict=True)
        )
    if qtype in ("numeric", "discrete"):
        if stated_percentiles is None:
            raise ValueError(f"{qtype} peer values require stated percentiles")
        return "\n".join(
            f"Percentile {float(percentile) * 100:g}: {float(value):g}" for percentile, value in stated_percentiles
        )
    raise ValueError(f"Unsupported question type for peer forecast rendering: {qtype}")


def update_block(variant: Variant, own_reply: str, peers: Sequence[PeerAnswer]) -> str:
    """Build the shared update prompt, adding peer content only for peer variants."""
    if variant not in ("peers_full", "peers_forecasts", "second_look"):
        raise ValueError(f"Unknown update variant: {variant}")
    if variant != "second_look" and not 1 <= len(peers) <= len(PEER_LABELS):
        raise ValueError(f"Expected one or two peer answers, got {len(peers)}")

    sections = ["── YOUR FIRST ANSWER ──", own_reply]
    if variant != "second_look":
        peer_lines = ["── OTHER FORECASTERS ──"]
        for label, peer in zip(PEER_LABELS, peers, strict=False):
            peer_lines.extend((label, peer.reply_text if variant == "peers_full" else peer.forecast_text))
        sections.append("\n".join(peer_lines))
    sections.append(UPDATE_INSTRUCTION)
    return "\n\n".join(sections)


# Models number the line as the instruction's list item ("2. UPDATE:", "**2.** UPDATE:") or bold the label.
UPDATE_LINE_START = re.compile(r"^[\s*#>\-]*(?:\d+[.)]\**\s*)?\**UPDATE:")


def parse_correction_kind(reply: str) -> str:
    """Read the correction kind from the final UPDATE line, or report it as unparsed."""
    update_lines = [line for line in reply.splitlines() if UPDATE_LINE_START.match(line)]
    if not update_lines:
        return CORRECTION_UNPARSED
    match = re.search(r"(?:^|\s)correction=\**(\w+)\**(?:\s|$)", update_lines[-1])
    if match is None or match.group(1) not in CORRECTION_KINDS:
        return CORRECTION_UNPARSED
    return match.group(1)
