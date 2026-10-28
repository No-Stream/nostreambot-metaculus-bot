"""Markdown assembly for the research bundle: provider headers and heading levels.

Split out of ``orchestrator.py``: how a provider's text is LABELLED and stitched into
one document is presentation, and it is independent of how the providers were chosen,
run, or budgeted. Everything here is pure — no I/O, no orchestrator state — which is
why it is all plain module functions and directly unit-testable.

``PROVIDER_SECTION_HEADERS`` is the one provider-to-header map in the repo, and it is
read in BOTH directions: ``provider_header`` reads it forwards to label a section, and
``detect_providers`` walks it backwards to decode which providers a historical bundle
carried.
"""

import re

from metaculus_bot.prompts import MARKET_SNAPSHOT_SECTION_HEADER, TS_ANCHOR_SECTION_HEADER
from metaculus_bot.research.provider_diagnostics import ProviderResult

_LEADING_HEADING_RE = re.compile(r"^(#{1,2})(?=\s|$)", re.MULTILINE)

# A RETIRED provider's row must STAY here: the map is read backwards by the log and
# comment backfills, which decode bundles written under earlier configs, so deleting a
# row makes every historical record silently under-report that provider as absent. Same
# reasoning as the dead ``logs-<run_id>`` prefix kept in RUN_LOG_ARTIFACT_PREFIXES.
PROVIDER_SECTION_HEADERS: dict[str, str] = {
    "asknews": "## News Articles (AskNews)",
    "native_search": "## Web Research (Native Search)",
    "gemini_search": "## Web Research (Google Search via Gemini)",
    "financial_data": "## Financial & Economic Data",
    "timeseries_anchor": TS_ANCHOR_SECTION_HEADER,
    "prediction_market": MARKET_SNAPSHOT_SECTION_HEADER,
    "resolution_source": "## Resolution Source Snapshot",
    "exa": "## Web Research (Exa)",
    "perplexity": "## Web Research (Perplexity)",
    "openrouter": "## Web Research (OpenRouter)",
    "custom": "## Research (Custom)",
}


def detect_providers(research_text: str) -> list[str]:
    """Detect which research providers contributed based on their rendered headers.

    Walking the live pipeline's provider-to-header map backwards keeps historical
    backfills aligned with every provider the pipeline can render, including retired
    providers whose headers remain in the map for archive decoding.
    """
    providers = []
    for provider_name, header in PROVIDER_SECTION_HEADERS.items():
        if header in research_text:
            providers.append(provider_name)
    return providers


def detect_gap_fill(research_text: str) -> bool:
    """Detect if gap-fill was used based on presence of the gap-fill header."""
    return "## Targeted Gap-Fill" in research_text


def _demote_inner_headings(text: str) -> str:
    """Shift any in-body h1/h2 heading down by two levels (h1→h3, h2→h4).

    Provider headers are h2 (``provider_header``). If an LLM-written body emits
    its own h1/h2 (e.g. ``# Historical Context``), it sits at/above the provider
    header and breaks the framework's ``report_sections_to_markdown``
    renormalization, which degrades to the ugly ``[Hashtag]`` fallback. Demoting
    keeps every provider header the minimum-level section.
    """
    return _LEADING_HEADING_RE.sub(lambda m: "##" + m.group(1), text)


def provider_header(name: str) -> str:
    """The h2 header a provider's section is rendered under."""
    return PROVIDER_SECTION_HEADERS.get(name, f"## Research ({name})")


def assemble_provider_sections(results: list[tuple[str, ProviderResult]]) -> tuple[str, list[ProviderResult]]:
    """Join each provider's text under its own h2 header; also return the results in order.

    ``results`` arrives in SELECTION order (``await_providers_within_deadline``
    rebuilds it from the provider list rather than from ``asyncio.wait``'s
    unordered sets), and that order is both the section order in the bundle and
    the row order in the diagnostics block. A provider that produced nothing is
    skipped in the text but still returned in ``provider_results``, so the
    diagnostics block and the research archive still name it.
    """
    combined_parts = []
    provider_results: list[ProviderResult] = []
    for raw, provider_result in results:
        provider_results.append(provider_result)
        if raw and raw.strip():
            header = provider_header(provider_result.name)
            combined_parts.append(f"{header}\n{_demote_inner_headings(raw)}")

    combined = "\n\n---\n\n".join(combined_parts) if combined_parts else ""
    return combined, provider_results
