"""Per-provider research-observability primitives.

A ``ProviderResult`` captures, for one research provider's call, what the
orchestrator's ``_run_one`` chokepoint already sees: outcome status, output
size, wall-clock latency, and any exception detail. The orchestrator persists
these to the durable research artifact (schema v2) and renders them into a
compact, pipe-delimited block that is logged at INFO, archived, and appended to
the published Metaculus comment — but deliberately kept OUT of the
forecaster-facing research text (the diagnostics seam in
``ResearchOrchestrator.run_research`` / ``TemplateForecaster``), so
failures/empties/timing survive beyond the ephemeral GHA run-log without
distracting the forecasters.
"""

from dataclasses import dataclass, field
from typing import Literal

# Status of one provider call. ``ok``: non-empty output. ``empty``: returned but
# blank/whitespace. ``errored``: raised an unexpected exception. ``inactive``:
# AskNews off-season subscription error (expected, not alertable). ``fallback``:
# AskNews failed but a prose fallback provider supplied the result. ``deadline``:
# still running when the research phase's time budget expired, so the orchestrator
# cancelled it (see ``QuestionTimeBudget.research_phase_deadline_s``) — a budget
# decision rather than a provider defect, which is why it is its own status and
# not folded into ``errored``.
ProviderStatus = Literal["ok", "empty", "errored", "inactive", "fallback", "deadline"]

# Markdown header for the diagnostics section. Named so the leakage screen can
# recognize a diagnostics-only blob without hardcoding the literal in two places.
PROVIDER_DIAGNOSTICS_HEADER = "## Provider Diagnostics"


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Structured outcome of a single research-provider call."""

    name: str
    status: ProviderStatus
    chars: int
    latency_ms: int
    error_type: str | None = None
    error_message: str | None = None
    # Provider-INTERNAL detail, surfaced so PARTIAL degradation is visible: a
    # multi-source provider (prediction_market's 4 venues, resolution_source's N
    # URLs, financial_data's tickers/FRED series) that returns usable output while
    # silently losing an upstream source otherwise reads as a healthy ``ok``.
    # Populated by the providers via :func:`record_provider_detail` and drained
    # into here by the orchestrator's ``_run_one``. Convention: ``details["sources"]``
    # is an ordered ``{source_name: token}`` map; a token starting with ``"ok"``
    # contributed, ``"none"`` was queried-but-empty (benign), anything else is a
    # loss (``"dropped(size_cap)"`` / ``"blocked"`` / ``"js_wall"`` / ``"error(...)"`` /
    # ``"empty"``). ``details["counts"]`` is the second convention: an ordered
    # ``{name: number}`` map of provider-internal quantities that are neither a source
    # outcome nor a failure (see :func:`_counts_suffix`).
    # ``asdict(r)`` serializes this straight into the research archive.
    details: dict = field(default_factory=dict)
    # Historical: on ``status == "fallback"``, the vendor that answered when AskNews failed
    # ("openrouter" / "perplexity" / "exa"). That fallback was removed on 2026-10-27, so new
    # records always carry None; the field stays so the archive schema keeps its shape.
    fallback_provider: str | None = None


# ---------------------------------------------------------------------------
# Per-source token classification (the partial-degradation convention)
# ---------------------------------------------------------------------------

# A ``details["sources"]`` token that starts with one of these is NOT a loss:
# ``ok`` / ``ok(N)`` contributed output; ``none`` was queried successfully but
# had no match (a normal, benign outcome — e.g. no relevant prediction market
# exists). Every other token (``dropped(...)`` / ``blocked`` / ``js_wall`` /
# ``error(...)`` / ``empty`` / ``timeout`` / ...) means the source was attempted
# and FAILED — that is the degradation the diagnostics line must surface.
_SOURCE_NON_LOSS_PREFIXES: tuple[str, ...] = ("ok", "none")

# Compact-line hygiene: bound a single reason token and the number of losses
# rendered so a pathological provider payload can't blow up the one-liner.
_LOST_TOKEN_MAX_CHARS: int = 40
_MAX_LOST_SOURCES_RENDERED: int = 8


def is_lost_source(token: str) -> bool:
    """A source token signals a lost/failed upstream (not contributed, not benign-empty).

    Public because the multi-source PROVIDERS classify their own tokens with it before the
    orchestrator ever sees them (`prediction_market` routes seven sources through it), and a
    provider-side copy of the prefix tuple is exactly the drift this module exists to prevent.
    """
    return not token.startswith(_SOURCE_NON_LOSS_PREFIXES)


# Historical private name, still imported by name in the provider test modules. An alias rather
# than a second body, so there is one definition of "lost".
_is_lost_source = is_lost_source


def _partial_loss_suffix(details: dict) -> str:
    """Render the ``| sources=<ok>/<total> | lost=<a:tok,b:tok>`` segment, or "".

    Empty unless ``details["sources"]`` is a non-empty map with at least one LOST
    source — so a fully-healthy multi-source provider (and any provider with no
    ``sources`` detail) renders byte-identically to the base line, keeping the
    archive/comment format stable.
    """
    sources = details.get("sources")
    if not isinstance(sources, dict) or not sources:
        return ""
    lost = {name: token for name, token in sources.items() if isinstance(token, str) and _is_lost_source(token)}
    if not lost:
        return ""
    contributed = sum(1 for token in sources.values() if isinstance(token, str) and token.startswith("ok"))
    total = len(sources)
    rendered = [
        f"{name}:{token[:_LOST_TOKEN_MAX_CHARS]}" for name, token in list(lost.items())[:_MAX_LOST_SOURCES_RENDERED]
    ]
    if len(lost) > _MAX_LOST_SOURCES_RENDERED:
        rendered.append(f"+{len(lost) - _MAX_LOST_SOURCES_RENDERED} more")
    return f" | sources={contributed}/{total} | lost={','.join(rendered)}"


# ---------------------------------------------------------------------------
# Per-(qid, provider) detail registry — the provider -> orchestrator seam
# ---------------------------------------------------------------------------
#
# A multi-source provider knows its per-source outcome internally, but the
# ``ResearchCallable`` contract only lets it return a ``str``, so that structured
# detail can't ride the return value. This module-level registry is the seam:
# the provider records once via :func:`record_provider_detail`; the orchestrator's
# ``_run_one`` pops it once via :func:`pop_provider_detail` and attaches it to the
# ``ProviderResult``. It mirrors the ``research.raw_log.record_raw_research`` sink
# (a module function providers call, keyed by qid+provider). Keying on
# (qid, provider) makes it safe under the orchestrator's parallel ``_run_one``
# fan-out: each coroutine writes its own key inside the provider call and pops the
# same key immediately after, so writes happen-before their matching pop and
# distinct providers/questions never collide.
_PROVIDER_DETAIL_REGISTRY: dict[tuple[int, str], dict] = {}


def record_provider_detail(qid: int | None, provider: str, detail: dict) -> None:
    """Record a provider's per-source ``detail`` for ``_run_one`` to drain.

    No-op when ``qid`` is ``None`` (matches the raw-log / comment-diagnostics
    handling — a question with no id can't be keyed or joined downstream).
    """
    if qid is None:
        return
    _PROVIDER_DETAIL_REGISTRY[(qid, provider)] = detail


def pop_provider_detail(qid: int | None, provider: str) -> dict:
    """Return-and-clear the recorded detail for ``(qid, provider)``, or ``{}``.

    Popping (not peeking) keeps the registry from growing across a batch and
    stops a stale entry from leaking into a later same-key call.
    """
    if qid is None:
        return {}
    return _PROVIDER_DETAIL_REGISTRY.pop((qid, provider), {})


# Statuses that count as "this provider contributed usable research". ``ok`` =
# its own non-empty output; ``fallback`` = AskNews failed but a prose fallback
# supplied the result. The orchestrator derives ``providers_succeeded`` from this
# set, so it lives here as the single source of truth.
SUCCEEDED_STATUSES: tuple[ProviderStatus, ...] = ("ok", "fallback")


def _counts_suffix(details: dict) -> str:
    """Render the ``| <name>=<value>`` segment for ``details["counts"]``, or "".

    Second detail convention beside ``sources``: ``details["counts"]`` is an ordered
    ``{name: number}`` map of provider-INTERNAL quantities that are neither a source
    outcome nor a failure — ``gemini_search``'s ``unsupported_attributions`` is the first.
    A zero renders nothing, so a healthy provider's line stays byte-identical to what it
    was before the map existed, while the archive keeps the zero (``asdict`` serializes
    the whole ``details``) — which is what makes "the check ran and found none"
    distinguishable from "the check never ran".
    """
    counts = details.get("counts")
    if not isinstance(counts, dict):
        return ""
    rendered = [f"{name}={value}" for name, value in counts.items() if value]
    return f" | {' | '.join(rendered)}" if rendered else ""


def _format_one(result: ProviderResult) -> str:
    line = f"- {result.name}: {result.status} | {result.chars} chars | {result.latency_ms} ms"
    if result.status == "errored" and result.error_type is not None:
        line += f" | {result.error_type}"
    line += _counts_suffix(result.details)
    line += _partial_loss_suffix(result.details)
    return line


def format_provider_diagnostics_block(results: list[ProviderResult]) -> str:
    """Render the compact provider-diagnostics block, or "" when there are none.

    One ``- <name>: <status> | <chars> chars | <ms> ms[ | <error_type>]`` line per
    provider, pipe-delimited so it greps/parses trivially. The leading ``---``
    mirrors the gap-fill addendum so the block renders as its own comment section.
    """
    if not results:
        return ""
    lines = [_format_one(result) for result in results]
    return f"---\n\n{PROVIDER_DIAGNOSTICS_HEADER}\n\n" + "\n".join(lines)
