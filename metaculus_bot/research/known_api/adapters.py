"""The two thin adapters from a :class:`KnownApiResult` to the two callers of rung 0.

One returns the gap-fill loop's ``ToolOutcome`` (method ``known_api``); one returns the
resolution-source fetcher's ``FetchResult`` (a success carries the rendered text, route
``known_api``). Detail: docs/research.md "Known-API registry".
"""

from __future__ import annotations

from typing import cast

from metaculus_bot.research.agentic.types import EvidencePassage, ToolOutcome
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute, FetchStatus

KNOWN_API_METHOD = "known_api"
KNOWN_API_ROUTE = "known_api"

# A known-API status maps onto the fetcher's existing vocabulary; only ``ok`` carries text.
_STATUS_TO_FETCH_STATUS: dict[str, str] = {
    "ok": "success",
    "empty": "no_resolving_content",
    "not_found": "not_found",
    "error": "error",
}


def to_tool_outcome(result: KnownApiResult) -> ToolOutcome:
    """The gap-fill loop's outcome for a known-API result; ``method=known_api``, status verbatim."""
    return ToolOutcome(
        content_markdown=result.content_markdown,
        links=list(result.links),
        method=KNOWN_API_METHOD,
        status=result.status,
        evidence_passages=[
            EvidencePassage(
                source_url=passage.source_url,
                text=passage.text,
                retrieval_method=passage.retrieval_method,
                content_origin=passage.content_origin,
                clipped=passage.clipped,
            )
            for passage in result.evidence_passages
        ]
        if result.status == "ok"
        else [],
    )


def to_fetch_result(result: KnownApiResult, *, url: str, route: FetchRoute = KNOWN_API_ROUTE) -> FetchResult:
    """The resolution-source fetcher's result for a known-API answer; a success carries the text.

    ``route`` stays injectable for adapter-level tests and future callers, while the production
    default is the registered ``known_api`` route.
    """
    status = cast(FetchStatus, _STATUS_TO_FETCH_STATUS[result.status])
    return FetchResult(
        url=url or result.source_url,
        status=status,
        text=result.content_markdown if result.status == "ok" else "",
        http_status=None,
        content_type=None,
        route=route,
        links=list(result.links),
        evidence_passages=list(result.evidence_passages) if result.status == "ok" else [],
    )
