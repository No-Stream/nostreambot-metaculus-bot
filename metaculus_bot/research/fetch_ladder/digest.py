"""The seat a page digest plugs into, and the free BM25 selection that fills it by default.

``LadderPolicy.digest`` takes any :class:`DigestFn`: the shape is ``page_digest.digest_page``'s
exactly, so the model-run digest drops in with no adapter, and :func:`bm25_digest` is what the
seat's ``None`` means. Both hand back the ranked passages rather than a rendered block, because
how many characters a reader sees is the CALLER's knob (the fetcher caps per URL, the loop
windows), so the ladder renders the block at presentation with that cap.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

from metaculus_bot.constants import DOCUMENT_DIGEST_TOP_K
from metaculus_bot.research.document_text import select_passages
from metaculus_bot.research.source_evidence import SourceExcerpt

# `agentic.local_document.DIGEST_LOCAL_METHOD`, spelled here so the ladder needs no caller import.
DIGEST_METHOD_BM25 = "digest_local"


class DigestPassages(Protocol):
    """What any digest hands back: the passages to show, and the three counters the fetch marker carries.

    ``passages`` opens with the page's opening window when the digest keeps one, then the accepted
    passages in rank order. An empty selection means no passage was selected from the stated
    inspection scope; it does not establish absence from the entire document. ``passages_returned``
    and ``passages_grounded`` are a model-run digest's raw answer count and how many were literal substrings of the page,
    both zero when no model ran; ``fallback_used`` says the ranked passages came from BM25 after a
    model call failed or grounded nothing; ``method`` names which produced them.
    """

    @property
    def passages(self) -> list[str]: ...
    @property
    def passages_returned(self) -> int: ...
    @property
    def passages_grounded(self) -> int: ...
    @property
    def fallback_used(self) -> bool: ...
    @property
    def method(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    @property
    def scope(self) -> str: ...
    @property
    def evidence_passages(self) -> tuple[SourceExcerpt, ...]: ...


class DigestFn(Protocol):
    """One digest call: the page text, what to read it for, and the wall it must fit inside."""

    async def __call__(self, text: str, query: str, *, budget_seconds: float) -> DigestPassages: ...


@dataclass(frozen=True, slots=True)
class LadderDigest:
    """The deterministic digest's result, one field for one on :class:`DigestPassages`."""

    passages: list[str]
    passages_returned: int
    passages_grounded: int
    fallback_used: bool
    method: str
    outcome: Literal[
        "answered",
        "valid_empty",
        "ungrounded",
        "malformed",
        "api_error",
        "timeout",
        "empty_completion",
        "insufficient_budget",
        "empty_page",
    ] = "answered"
    scope: Literal["full_page", "prefiltered_windows"] = "full_page"
    evidence_passages: tuple[SourceExcerpt, ...] = ()

    def __post_init__(self) -> None:
        if not self.passages and self.outcome == "answered":
            object.__setattr__(self, "outcome", "valid_empty")


async def bm25_digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
    """The free default: the ``DOCUMENT_DIGEST_TOP_K`` windows of ``text`` BM25 ranks highest for ``query``.

    Deterministic, I/O-free and CPU-bound on the text alone, so ``budget_seconds`` is unread; a
    caller that needs the selection off the event loop wraps this in the thread hop it already
    runs the PDF digest in. No model ran, so both counters are the selection's own size.
    """
    del budget_seconds
    ranked = await asyncio.to_thread(select_passages, text, query, top_k=DOCUMENT_DIGEST_TOP_K)
    excerpts = tuple(
        SourceExcerpt(source_url="", text=passage.text, retrieval_method="bm25_candidate") for passage in ranked
    )
    return LadderDigest(
        passages=[passage.text for passage in ranked],
        passages_returned=len(ranked),
        passages_grounded=len(ranked),
        fallback_used=False,
        method=DIGEST_METHOD_BM25,
        outcome="answered" if ranked else "valid_empty",
        evidence_passages=excerpts,
    )
