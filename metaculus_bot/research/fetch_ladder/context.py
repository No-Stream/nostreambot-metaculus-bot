"""Per-URL and per-question bookkeeping for one run of the fetch ladder.

:class:`LadderContext` is the one-per-fetched-URL record: the question text a document's
passages are ranked against, the wall-clock origin every rung bounds itself against, and the
rung attempts that become the result's ``route`` and its escalation telemetry.
:class:`QuestionRungBudget` is the per-question half, shared across a question's cited URLs so
the capped rungs cannot be paid for once per URL.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
    RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS,
)
from metaculus_bot.research.fetch_ladder.policy import RESOLUTION_SOURCE_POLICY, LadderPolicy
from metaculus_bot.research.http_fetch import semaphore_for_host
from metaculus_bot.research.resolution_fetch_result import (
    FetchResult,
    FetchRoute,
    FetchStatus,
    RungAttempt,
    RungSkipReason,
)

if TYPE_CHECKING:
    from metaculus_bot.research.fetch_ladder.run_cache import ReadArtifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReadCapture:
    """A result and its complete read plus acquisition route, tied by object identity."""

    result: FetchResult
    artifact: ReadArtifact
    route: FetchRoute


@dataclass(slots=True)
class LocalReadReceipt:
    """Mutable per-call receipt proving local source bytes reached the parser path."""

    encountered: bool = False


@dataclass
class QuestionRungBudget:
    """The rung allowances one QUESTION shares across its cited URLs.

    Per-question rather than per-URL because of what is being bounded; its default is a fresh
    budget, so a monkeypatched fetch driven with one URL and no shared state behaves exactly as it
    did. Receipts: ``docs/architecture.md`` "The per-URL context and the per-question budget".
    """

    wayback_attempts_left: int = RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS
    # The Wayback cap's analogue, so one question cannot pay per dead source inside one wall.
    url_context_attempts_left: int = RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS
    # One browser escalation per host at a time WITHIN the question, see `browser_escalation_gate`.
    browser_escalation_gates: dict[str, asyncio.Semaphore] = field(default_factory=dict)

    def take_wayback_attempt(self) -> bool:
        """Claim one snapshot attempt for this question, or False when they are spent."""
        if self.wayback_attempts_left <= 0:
            return False
        self.wayback_attempts_left -= 1
        return True

    def take_url_context_attempt(self) -> bool:
        """Claim one paid url_context read for this question, or False when they are spent."""
        if self.url_context_attempts_left <= 0:
            return False
        self.url_context_attempts_left -= 1
        return True

    def browser_escalation_gate(self, url: str) -> asyncio.Semaphore:
        """The ``Semaphore(1)`` that serializes this question's derived-feed-then-browser
        escalations on ``url``'s host.

        Held across the PAIR so a same-host sibling re-asks ``endpoint_for`` once the first
        escalation is over and takes the feed off an ordinary GET instead of launching its own
        browser, and per question rather than loop-wide so no unbounded process-global acquire
        sits in front of the provider's wall. Both receipts: ``docs/architecture.md`` "The per-URL
        context and the per-question budget".
        """
        return semaphore_for_host(url, self.browser_escalation_gates)


# Keyed by route so `claim_rung_budget`'s one template reads as the six lines it replaced.
_RUNG_WALL_SKIP_PHRASE: dict[FetchRoute, str] = {
    "meta_refresh": "the meta-refresh hop",
    "impersonate": "the impersonated retry",
    "pdf_local": "the local PDF read",
    "derived_api": "the derived-feed GET",
    "rendered": "the rendered rung",
    "wayback": "the wayback rung",
    "url_context": "the url_context rung",
}


@dataclass
class LadderContext:
    """Per-URL inputs and rung bookkeeping for one :func:`_fetch_one` call.

    ONE per fetched URL, so ``rungs`` belongs to that URL and can be stamped onto its result,
    while ``query`` (what a PDF's passages are ranked against), ``started`` (the monotonic wall
    origin every rung bounds itself with), ``now`` (its wall-clock counterpart, which dates a
    Wayback capture), ``shared``, ``policy``, ``session``, ``host_sems`` and ``fast_path`` are the
    same for every URL in one provider call. Every field has a default, so the monkeypatched fetch
    surface can still be driven with three positional arguments and a default context behaves as a
    direct fetch did before the ladder existed. What each field decides, and why the clock is
    taken here rather than inside a rung: ``docs/architecture.md`` "The per-URL context and the
    per-question budget".
    """

    query: str = ""
    started: float = field(default_factory=time.monotonic)
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    shared: QuestionRungBudget = field(default_factory=QuestionRungBudget)
    rungs: list[RungAttempt] = field(default_factory=list)
    fast_path: bool = False
    policy: LadderPolicy = RESOLUTION_SOURCE_POLICY
    session: Any = None
    host_sems: dict[str, asyncio.Semaphore] | None = None
    read_captures: list[ReadCapture] = field(default_factory=list, repr=False)
    local_read_receipt: LocalReadReceipt = field(default_factory=LocalReadReceipt, repr=False)

    def capture_read(self, result: FetchResult, artifact: ReadArtifact) -> None:
        fired: list[FetchRoute] = [attempt.rung for attempt in self.rungs if not attempt.skipped_reason]
        route = fired[-1] if fired and result.route == "direct" else result.route
        self.read_captures.append(ReadCapture(result=result, artifact=artifact, route=route))

    def read_capture_for(self, result: FetchResult) -> ReadCapture | None:
        return next((capture for capture in reversed(self.read_captures) if capture.result is result), None)

    def artifact_for(self, result: FetchResult) -> ReadArtifact | None:
        capture = self.read_capture_for(result)
        return None if capture is None else capture.artifact

    def rung_budget_s(self) -> float:
        """Wall-clock seconds a rung may spend before the outer ``wait_for`` fires.

        Same arithmetic as the Datawrapper hop's, and for the same reason: that timeout
        discards every page that already fetched, so a rung that overruns costs the
        whole question's resolution evidence rather than just its own attempt.
        """
        return self.policy.total_wall_s - (time.monotonic() - self.started) - self.policy.rung_wall_margin_s

    def claim_rung_budget(
        self, rung: FetchRoute, from_status: FetchStatus, url: str, floor_s: float, *, note: str = ""
    ) -> float | None:
        """The remaining wall budget for ``rung``, or None once it has recorded the skip.

        The one home for the wall-budget preamble every rung copied: read the remaining wall,
        and below the rung's own floor log once, record a ``wall_budget`` skip on this context and
        return None; otherwise hand back the budget the rung sizes its work off. Structural rather
        than stylistic — an incomplete copy still returned None and simply never appeared in the
        ``rung_budget_skips`` count, so the archive under-reported how often the wall is the
        binding constraint. ``note`` is the one place the message varies (the paid rung's second
        check adds " after the robots pre-check", the PDF read's re-check " after queueing").
        """
        budget_s = self.rung_budget_s()
        if budget_s < floor_s:
            logger.warning(
                "resolution_source: skipping %s for %s — %.1fs of wall budget left%s",
                _RUNG_WALL_SKIP_PHRASE[rung],
                urlparse(url).netloc,
                budget_s,
                note,
            )
            self.skip_rung(rung, from_status, url, "wall_budget")
            return None
        return budget_s

    def start_rung(self, rung: FetchRoute, from_status: FetchStatus, url: str) -> RungAttempt:
        attempt = RungAttempt(rung=rung, from_status=from_status, url=url, started_at=time.monotonic())
        self.rungs.append(attempt)
        return attempt

    def skip_rung(self, rung: FetchRoute, from_status: FetchStatus, url: str, reason: RungSkipReason) -> None:
        self.rungs.append(
            RungAttempt(
                rung=rung,
                from_status=from_status,
                url=url,
                started_at=time.monotonic(),
                wall_s=0.0,
                skipped_reason=reason,
            )
        )

    def close_rungs(self, first_new: int, outcome: FetchStatus) -> None:
        """Close every attempt opened since ``first_new`` with ONE clock reading and one outcome.

        The dispatcher calls this the moment a rung's result is known — ``first_new`` is
        ``len(ctx.rungs)`` read before the rung ran — so each attempt's ``wall_s`` measures
        that rung alone and its ``outcome`` is the status that stood once it was over: the
        rescue it returned, or the direct status it left standing when it declined. A rung
        that already stamped either field for itself keeps its stamp (``finish`` respects the
        PDF read's own ``wall_s``; the browser rung's own ``outcome`` is the rendered DOM's
        verdict), which is what keeps a harvested feed's rescue from being credited to the
        render that only found the endpoint.
        """
        now = time.monotonic()
        for attempt in self.rungs[first_new:]:
            attempt.finish(now)
            if attempt.outcome is None and not attempt.skipped_reason:
                attempt.outcome = outcome


def _aux_ctx(ctx: LadderContext) -> LadderContext:
    """A child context for a request a rung makes on the cited URL's BEHALF.

    The Wayback snapshot, the remembered derived feed and the robots.txt pre-check all go
    through :func:`_fetch_direct`, whose own rungs (the meta-refresh hop, the local PDF read)
    stamp attempts onto whatever context they are handed. On the PAGE's context those stamps
    hijack the record: an archived PDF capture came back ``route="pdf_local"`` — ``route`` is
    the last rung that fired — was counted as a Wayback attempt AND a document read, and lost
    the archived-copy caveat that ``resolution_presentation._route_caveats`` keys on the route. The child shares the
    clock, the query, the wall-clock origin and the per-question budget, and owns a rung list
    nobody reads, which is :class:`LadderContext`'s one-per-fetched-URL invariant applied to a
    URL the question never cited. Its attempts are deliberately NOT merged back: that would put
    ``pdf_local`` last again and the route would still be wrong.
    """
    return replace(ctx, rungs=[], read_captures=[])


def _skip_for_fast_path(ctx: LadderContext, rung: FetchRoute, direct: FetchResult, url: str) -> None:
    """Record that an EXPENSIVE rung declined because the question is on the time-budget fast path.

    Its own token rather than ``wall_budget``: the wall here is the provider's fixed 45 s, which
    a fast-path question may have plenty of, and a residual round reading the counts has to be
    able to tell "the question's close left no room for a browser" from "this rung ran out of
    the provider's own clock". The saving is narrower than the fast path's name implies — a
    close-limited budget under the intake floor is skipped outright, so the band this gate
    protects is the high-pre-research-elapsed question — but a 12-35 s Chromium launch inside a
    thin window is still a launch the prediction POST would rather have.
    """
    logger.info(
        "resolution_source: declining the %s rung for %s — the question is on the time-budget fast path",
        rung,
        urlparse(url).netloc,
    )
    ctx.skip_rung(rung, direct.status, url, "fast_path")


def _stamped_with_route(result: FetchResult, ctx: LadderContext) -> FetchResult:
    """Attach the ladder bookkeeping to a finished result.

    ``route`` is the LAST rung that fired, which is the one that produced this outcome
    (a meta-refresh hop onto a PDF reads ``pdf_local``: the hop got us the bytes, the
    local read is what the text came from). Skipped rungs never claim the route. The one
    exception is a result that already names its rung: a rung's VERDICT (the Wayback
    ``stale_data`` withhold) can be returned as the ladder's fallback after a later rung
    fired and failed, and the last rung to fire is then not the one that produced it.

    Every rung the dispatcher ran has already been closed with its own wall and outcome
    (:func:`_run_rung`); the close here is the last resort for an attempt a future rung
    opens without bracketing, and stamps it with the final status rather than leaving the
    marker to print ``None``.
    """
    ctx.close_rungs(0, result.status)
    result.rung_attempts = list(ctx.rungs)
    fired: list[FetchRoute] = [attempt.rung for attempt in ctx.rungs if not attempt.skipped_reason]
    if fired and result.route == "direct":
        result.route = fired[-1]
    return result


# Derived, never spelled twice: the two tables drifted in both directions (see the doc).
_BUDGET_GATED_RUNGS: tuple[FetchRoute, ...] = tuple(_RUNG_WALL_SKIP_PHRASE)
