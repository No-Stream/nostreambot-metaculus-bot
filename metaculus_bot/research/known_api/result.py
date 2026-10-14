"""The neutral result every known-API backend returns, and the two adapters read.

One shape for a FRED series, a Yahoo history, a market snapshot or an EDGAR filing, so the
adapter to the gap-fill loop's ``ToolOutcome`` and the adapter to the resolution-source
fetcher's ``FetchResult`` each read one thing. The ``status`` token stays inside the existing
``ok`` / ``empty`` / ``not_found`` / ``error`` family the fetch vocabulary already uses, so
neither adapter invents a new status. Detail: docs/research.md "Known-API registry".

A backend NEVER raises to its caller: an unknown id is ``not_found`` with the provider's
message, an empty window is ``empty``, a transport or quota failure is ``error`` naming the
exception class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from metaculus_bot.research.source_evidence import SourceExcerpt

KnownApiStatus = Literal["ok", "empty", "not_found", "error"]
KNOWN_API_STATUSES: frozenset[str] = frozenset({"ok", "empty", "not_found", "error"})


@dataclass(frozen=True, slots=True)
class KnownApiResult:
    """One known-API answer: the rendered markdown, the canonical URL, and enough for both adapters.

    ``content_markdown`` is the block a forecaster or the driver reads; ``source_url`` is the
    human page for the series/market so provenance can cite it; ``links`` carries that same URL
    for the loop's link harvest. ``status`` is ``ok`` only when ``content_markdown`` carries
    real content, mirroring the fetcher's success-implies-content rule.
    """

    status: KnownApiStatus
    content_markdown: str
    source_url: str
    links: list[str] = field(default_factory=list)
    evidence_passages: list[SourceExcerpt] = field(default_factory=list)

    @property
    def is_content(self) -> bool:
        """True when this result carries content to render (``ok``); the failure tokens do not."""
        return self.status == "ok"
