"""The headless-Chromium page render, shared by both SSRF-guarded fetch paths.

One browser rung, two callers. The gap-fill v2 ``fetch`` tool has escalated a JavaScript
wall to Chromium since 2026-07; the Tier-1 resolution-source fetcher gained the same rung on
2026-09-03, after a replay of 47 archived fetch failures measured Chromium rescuing 6 of the
8 pages that still failed from a residential address. Keeping the transport here is what
makes that one rung rather than two: the DNS pin, the per-request route guard, the
landing-host check, the WebSocket block, the process-global launch cap and the wait-condition
arithmetic are all SSRF- or memory-load-bearing, and a second copy of any of them would drift.

What this module owns is the TRANSPORT — launch, navigate, read the DOM, optionally record
the JSON the page fetched for itself — and it hands back a :class:`RenderedPage` rather than
either caller's result type. Classification stays with the caller, because the two disagree
about what a rendered page MEANS: v2 wants extracted text plus outbound links for a driver
model, while Tier-1 re-enters its own chart-config / ARIA-rewrite / chrome-floor path and can
still withhold the page. That split is also why the "rendered to nothing" memo is written by
the caller (:func:`note_rendered_no_text`) and only READ here, and why both render memos are
keyed by the caller's ``memo_scope`` as well as the URL: only the caller knows whether its
own extraction found anything, and the two callers' answers to that question are not
interchangeable (see :func:`rendered_to_nothing`). The timed-out memo is the one this module
writes itself, because only the transport knows whether a browser actually ran before the
clock cut it off (see :class:`RenderTimeout`).

The SSRF guard is read off :mod:`metaculus_bot.research.fetch_ladder.guard` at call time, which
is what keeps the guard to one patch surface — see :func:`resolve_pinned_host`.

The DNS-pin helpers here, :func:`pinnable_url_host` and :func:`resolve_pinned_host`, have a
THIRD consumer: the TLS-impersonation transport (:mod:`metaculus_bot.research.impersonated_fetch`,
2026-09-04), which pins libcurl to the vetted address with ``CURLOPT_RESOLVE`` exactly as this
module pins Chromium with ``--host-resolver-rules``. Both helpers are the one home of that
vetting for both pin syntaxes, which is why their return and rejection contracts are stated in
their docstrings for libcurl as well as Chromium; the browser render itself is still this
module's alone.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from metaculus_bot.constants import RENDERED_DOM_MAX_CHARS
from metaculus_bot.research.fetch_ladder import guard
from metaculus_bot.research.http_fetch import BROWSER_HEADERS
from metaculus_bot.research.public_suffix import registrable_domain

logger = logging.getLogger(__name__)

# Hard ceiling on one render, and the fixed settle after the DOM is ready that stands in for
# waiting on network idle. Measured 2026-09-03: 4 of the replay's 10 render rescues came from
# pages where ``page.goto`` raised TimeoutError with the DOM fully rendered (both ballotpedia
# questions, both fts.unocha.org summaries) — network idle never arrives on a page carrying a
# long-poll widget or an analytics beacon, so waiting for it discarded content Chromium already
# had. The navigation's worst case is the cap: the goto budget is the cap MINUS the settle, and
# the one call after the settle, the DOM read, carries its own bound below.
RENDER_TIMEOUT_MS: int = 35_000
RENDER_SETTLE_MS: int = 2_000
# Bound on ``page.content()``. On a settled DOM it is a sub-second round trip to the browser; it
# runs long only when the page keeps navigating after the settle. Measured 2026-09-03 on
# ogimet.com: the goto timed out at 33 s as designed, the settle ran, and the unbounded read then
# blocked for a further 40 s, so the render ran 76 s against a Tier-1 provider wall of 45 s and
# every page that question had already fetched was discarded. A DOM that has not answered in this
# long is one that will not settle. The pinned 1.61 driver does NOT retry the read: it evaluates
# ``outerHTML`` once and, when the page navigates mid-evaluate, raises its plain ``Error`` ("the
# page is navigating and changing the content"), which :func:`_navigate_and_read_dom` reads as the
# same cut-off. Fixed rather than derived from the remaining goto budget, because the read that
# matters most is the salvage AFTER a goto timeout, when that remainder is zero by construction.
RENDER_DOM_READ_TIMEOUT_MS: int = 5_000
# What a render spends AFTER its goto returns or times out, at worst: the settle, then the DOM
# read at its bound. A caller-derived deadline has to leave this much room past the goto, or the
# salvage-after-goto-timeout population above is cut off by the caller's own bound the instant
# the settle ends — which is what happened at every Tier-1 budget under 40 s before the
# navigation budget was recomputed here (see :func:`_goto_budget_ms`). Two costs are NOT in this
# tail and are the CALLER's to reserve, in the deadline it hands the transport
# (:data:`RENDER_EXIT_RESERVE_MS`): the launch, which runs AFTER the recompute (0.3 s warm,
# seconds cold), and the shared teardown bound. So when the goto runs its budget out, the DOM
# read's fixed bound lands past the transport's deadline by the launch time, inside that reserve,
# and the harvest drain is clamped to the deadline itself in :func:`_navigate_and_read_dom` so a
# body still in flight cannot spend the reserve on the harvest's behalf.
RENDER_POST_GOTO_TAIL_MS: int = RENDER_SETTLE_MS + RENDER_DOM_READ_TIMEOUT_MS
# Floor under the navigation budget once the gates are held. Below it the render DECLINES rather
# than making a token attempt: a launch is 100-300 MB and one of the process-wide slots, and a
# page that could not answer in five seconds after a queue this long is not one we would have
# read in the seconds that were left. Live only through :func:`_goto_budget_ms`, which is the one
# place the budget is computed after both gates are held.
RENDER_MIN_GOTO_MS: int = 5_000
# ONE bound on the whole teardown (unroute, context close, browser close together), shared by
# the three steps through :class:`_TeardownBudget`. A healthy browser answers each in well under
# a second; a wedged one never answers `BrowserContext.close`, which awaits its closed-future with
# no timeout of its own, and a render whose DOM was already read would then sit in teardown until
# the caller's bound cut it — discarding the page it had. Shared rather than per step because
# three separate 2 s bounds let a wedged browser run 6 s past the caller's cut (`asyncio.wait_for`
# cancels the render and then AWAITS its unwinding finallys), which is what tripped the Tier-1
# provider's 45 s wall. Past the bound a step is abandoned and the driver stop at the end of the
# ``async_playwright()`` block kills the browser process anyway, so nothing is left running.
RENDER_TEARDOWN_TIMEOUT_MS: int = 2_000
# What a caller with a hard deadline must reserve for the render's EXIT, on top of the goto and
# the post-goto tail: the shared teardown bound plus one second for the costs that run between
# the budget recompute and the goto (the launch at 0.3 s warm, `new_context`, `new_page`) and for
# the driver stop after the closes. The Tier-1 rung hands the transport its wall budget LESS this
# reserve while keeping its own `wait_for` at the full budget, so a render that runs every bound
# out still hands its DOM back before the outer cut instead of being cancelled in teardown. It
# can only shorten the goto or decline earlier at `RENDER_MIN_GOTO_MS`, never lengthen a wait.
RENDER_EXIT_RESERVE_MS: int = RENDER_TEARDOWN_TIMEOUT_MS + 1_000

# Process-global cap on concurrent headless Chromium launches. Module-level, so the bound
# spans every question running under the orchestrator's Semaphore(6) AND both fetch paths:
# each Chromium is ~100-300MB, one v2 step's parallel_tool_calls can request many fetches, a
# Tier-1 question can escalate up to RESOLUTION_SOURCE_MAX_URLS pages, and an unbounded launch
# fan-out would OOM the runner (an escape try/except cannot catch that). Cap 2 covers real
# bursts of 1-3 rendered pages while bounding worst-case memory.
RENDER_LAUNCH_CAP: int = 2
_RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(RENDER_LAUNCH_CAP)

# The two per-run render memos, one bounded map. ``no_text``: Chromium ran and the caller's
# extraction found nothing, so a second launch on the same URL in the same run is skipped: 100-300 MB
# and a whole slot out of the cap above, to learn what this run already knows. ``timed_out``: the
# render ran and read no DOM, because the read was cut off or refused (the page kept navigating) or
# the navigation failed onto Chromium's own document, and the transport re-raises
# :class:`RenderTimeout` rather than launching again. Both deliberately record ONLY those outcomes:
# a `blocked` or `error` GET is never memoized, because 429 is retryable and a throttle
# interstitial has to stay re-requestable (the q45191 receipt). Keyed by (scope, url): the two
# callers mean different things by "rendered to nothing" (gap-fill v2 writes it on bare trafilatura
# emptiness; Tier-1 only after the ARIA rewrite, the inline-chart read and the XHR-harvest fallback
# have ALL failed), so one path's negative must never answer the other's question.
MemoScope = Literal["resolution_source", "gap_fill_v2"]
_MemoKind = Literal["no_text", "timed_out"]
_RENDER_MEMO_MAX_ENTRIES = 50
_RENDER_MEMO: OrderedDict[tuple[MemoScope, str], _MemoKind] = OrderedDict()
_PLAYWRIGHT_WARNED = False

# --- JSON the page fetched for itself (the derived-API rung's discovery half) -------------
#
# Measured 2026-09-03 across six JavaScript dashboards: grepping the served HTML for an API
# URL found ONE candidate (a Maps key), 3 of 4 hand-guessed endpoints were wrong, and XHR
# capture during a render found a working unauthenticated JSON endpoint for all six. So the
# endpoints are discoverable only by rendering, which is why this rides the browser rung
# instead of being a static registry.
#
# Every bound here is a memory bound: the bodies are buffered in the render task while the
# page is live, on a runner also holding a 100-300 MB browser.
HARVEST_MAX_RESPONSES: int = 5
# Below this a JSON body is a ping, a feature flag or an empty envelope, not a dataset.
HARVEST_MIN_BODY_BYTES: int = 200
HARVEST_MAX_BODY_BYTES: int = 256 * 1024
# Hosts other than the page's own publisher that may serve a harvestable dataset. Deliberately
# almost empty: an allow-list entry is a decision that a third-party host's JSON may be read as
# the cited page's content, and only Datawrapper's CDN has an already-trusted data route in this
# repo (the Tier-2 hop fetches from it by construction). Anything else waits for a measurement.
HARVEST_ALLOWED_CDN_HOSTS: frozenset[str] = frozenset({"static.dwcdn.net"})
_JSON_CONTENT_TYPE_TOKENS = ("application/json", "text/json", "+json")


class RenderTimeout(TimeoutError):
    """A render that ran and was CUT OFF, or a URL a render in this run already ran to no DOM.

    "Cut off" has two shapes, and both are the page keeping on navigating after the settle: the
    DOM read outlives ``RENDER_DOM_READ_TIMEOUT_MS``, or the browser refuses it first, because the
    pinned 1.61 driver evaluates ``outerHTML`` once and raises its plain ``Error`` ("Unable to
    retrieve content because the page is navigating and changing the content.") when the document
    changes under the evaluate. The second is caught at that one call and re-raised as this class,
    since at the transport's Playwright boundary it read as "the renderer is unavailable", burned
    the once-per-run warn latch (so a real install failure later in the run logged nothing) and
    wrote no memo (so the next question relaunched Chromium for the same page).

    A subclass of the builtin so both callers' ``except TimeoutError`` catch it unchanged, and a
    class of its own so the transport can tell its deliberate cut-off from a raw OS-level
    ``TimeoutError`` (also a builtin subclass, via ``OSError``) surfacing from somewhere under the
    render: the first is re-raised for the caller to record as ``render_timeout``, the second is
    not a fact about the page and lands in the logged boundary like any other unexpected error.

    The timed-out memo is written inside the transport at the RAISE SITE, of this exception and
    of the failed-navigation half of :class:`RenderOffHost` (Chromium's own document, from which a
    retry reads nothing either), and nowhere else: a caller that bounds the whole call with its
    own ``wait_for`` cannot tell a render that ran out the clock from one that never left the
    queue behind the two gates, and memoising the second would switch the URL off for every later
    question in the run over a wait that was never about the page. At the raise site rather than
    in a handler because the exception unwinds through the bounded teardown before any handler
    sees it, and a caller's cancellation landing during that teardown replaces it, so a handler
    would never run. A memo hit re-raises this class before any gate is taken, so a URL memoised by
    either raise site is recorded by the Tier-1 caller as ``render_timeout`` from the second
    question on.
    """


class RenderBudgetExpired(TimeoutError):
    """The caller's deadline left less than ``RENDER_MIN_GOTO_MS`` once both gates were held.

    Raised instead of returning ``None`` so the caller can record it as what it is — its wall
    budget ran out in the queue — rather than as the renderer being unavailable, the count the
    operator reads as the Chromium install having failed. A ``TimeoutError`` because that is what
    it is (a deadline passed while waiting) and because a caller that bounds the whole call with
    its own ``wait_for`` already handles that class; deliberately NOT a :class:`RenderTimeout`,
    because no browser ran and there is nothing to memoise. Only reachable with a deadline, so
    the gap-fill v2 path never sees it.
    """


class RenderDomOverCeiling(Exception):
    """Chromium rendered the page and its DOM is over ``RENDERED_DOM_MAX_CHARS``.

    Raised instead of returning ``None`` so the caller can record it as what it is, a fact about
    the page (it rendered, and it is too big to read safely), rather than as the renderer being
    unavailable, the count the operator reads as the Chromium install having failed. Not a
    ``TimeoutError``: no clock ran out. Not memoised: the DOM had content, so "rendered to
    nothing" would be false, and a size is not something a retry changes within a run either
    way. Both callers decline on it the way they decline on ``None``.
    """


class RenderOffHost(Exception):
    """Chromium's main frame landed on a host other than the one the DNS pin covers.

    The landing is checked before AND after ``page.content()`` is read, so a DOM from the other
    host is either refused unread or, when the navigation committed during the read itself,
    discarded unpublished; either way none of it leaves :func:`_navigate_and_read_dom`. The pin
    (``--host-resolver-rules=MAP <host> <ip>``) forces the resolution of ONE hostname, the one the
    render was asked for; a server-side redirect hop is dialed by the browser with no route
    handler of ours involved (Playwright constructs a Route only for a request with no
    ``redirectedFrom``), so a main frame that ended up anywhere else was reached through
    Chromium's own resolver, outside every check this transport makes. Refusing the DOM is what
    closes that channel: an attacker who answers our aiohttp GET with a wall and the browser with
    a 302 to a private address gets nothing published for it.

    "Off the pinned host" is read fail-shut (:func:`_landed_off_host`), so it also covers
    Chromium's own error document: a navigation that fails outright can leave the frame on
    ``chrome-error://chromewebdata/`` rather than ``about:blank`` (observed live on 2026-09-04
    after ``net::ERR_UNSAFE_PORT`` and after ``net::ERR_CONNECTION_REFUSED`` on a redirect to a
    loopback target), and that is refused like any other stranger. The callers' ``render_off_host``
    count is therefore an UPPER bound on hostile landings: it also counts failed navigations, and
    the marker's ``landed_host`` tells the two apart (``chromewebdata`` is the error document).

    Its own class for the same reason as :class:`RenderDomOverCeiling`: it is a fact about the
    page rather than about the browser install, and the callers count it under its own token. Not
    a ``TimeoutError``: no clock ran out. Memoised by halves (:func:`_refuse_landing`). A genuine
    http(s) landing on another host is not: the page was not "rendered to nothing", the host it
    sent the browser to is not something a retry within the run changes, and the next question
    citing it should be refused on its own record. Chromium's own document (any non-http(s)
    landing, the error document above included) is: the page rendered nothing and a retry changes
    nothing, which is the population the empty-DOM read used to memoise through the callers'
    no-text memo, so the transport writes its timed-out memo at the raise site and a later render
    of the URL in the run declines without a launch. ``final_url`` is carried for the caller's own
    classification; the transport's log names only its hostname, because a landing URL can carry
    a token.
    """

    def __init__(self, *, requested_url: str, final_url: str, pinned_host: str) -> None:
        self.requested_url = requested_url
        self.final_url = final_url
        self.pinned_host = pinned_host
        super().__init__(
            f"the render of {pinned_host} landed on {urlparse(final_url).hostname or final_url!r}, off the pinned host"
        )


@dataclass(frozen=True, slots=True)
class HarvestedJson:
    """One JSON response the rendered page fetched for itself."""

    url: str
    body: bytes


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """What a successful render hands back: the DOM, plus any JSON the page fetched.

    ``html`` is ``page.content()`` — the DOM as it stood after the settle, which is the
    whole point of the rung. ``content_type`` and ``http_status`` are the main-frame
    response's, empty and ``None`` when the goto timed out and the DOM was salvaged (no
    response object survives that path). The status is carried because a browser can be
    answered differently from the direct GET on the same URL — a 403 or 429 interstitial whose
    markup clears a chrome floor is not the page — and only the caller decides what to do
    about that.

    ``url`` is the URL the render was ASKED for, and stays so: both callers key their memos on
    it. ``final_url`` is ``page.url`` as it stood after the DOM read, the URL of the document the
    DOM came from, and it is on the same host as ``url`` whenever a page is returned: the landing
    is checked before and after the read, and a stranger's DOM is refused unread or discarded
    unpublished (:class:`RenderOffHost`), with Chromium's own error document counted as a
    stranger. It is ``about:blank`` when the navigation never committed, which is the salvage of a
    genuine navigation error. ``document_url`` is what the callers classify and resolve links
    against: ``final_url`` when a navigation committed, else ``url``.
    """

    url: str
    content_type: str
    html: str
    json_responses: tuple[HarvestedJson, ...] = ()
    http_status: int | None = None
    final_url: str = ""

    @property
    def document_url(self) -> str:
        """The URL of the document ``html`` is, for classification and link resolution.

        The landing when one committed, because a same-host client-side redirect or meta refresh
        moves the document's base (6 of 22 render targets in the 2026-09-04 probe ended on a
        different path or query), and a relative link or a published section URL resolved against
        the requested URL would name the wrong document. A no-document landing (``about:blank``,
        or an empty ``final_url`` from a caller-built page) names nothing, so the requested URL
        stands. Never a memo key: the memos are keyed on ``url``.
        """
        return self.final_url if urlparse(self.final_url).hostname else self.url


def reset_render_state() -> None:
    """Drop the run-scoped render state: the memos, the warn latch, the launch gate.

    For tests. The launch gate is REBOUND rather than drained because an ``asyncio.Semaphore``
    binds to the loop that first blocks on it and raises from any other, so a contended
    acquire in one test's event loop would otherwise leak that binding into the next.
    """
    global _PLAYWRIGHT_WARNED, _RENDERED_FETCH_GLOBAL_SEMAPHORE  # noqa: PLW0603  # run-scoped render state: the warn latch and the loop-bound launch gate
    _RENDER_MEMO.clear()
    _PLAYWRIGHT_WARNED = False
    _RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(RENDER_LAUNCH_CAP)


def rendered_to_nothing(url: str, *, memo_scope: MemoScope) -> bool:
    """True when a browser already read ``url`` to nothing in this run, by ``memo_scope``'s own test."""
    return _RENDER_MEMO.get((memo_scope, url)) == "no_text"


def render_timed_out(url: str, *, memo_scope: MemoScope) -> bool:
    """True when a render of ``url`` under ``memo_scope`` was already cut off in this run."""
    return _RENDER_MEMO.get((memo_scope, url)) == "timed_out"


def note_rendered_no_text(url: str, *, memo_scope: MemoScope) -> None:
    """Record that Chromium rendered ``url`` and the caller found no content in the result."""
    _note(url, memo_scope, "no_text")


def _note_render_timeout(url: str, *, memo_scope: MemoScope) -> None:
    """Record that a render of ``url`` RAN and read no DOM, so this run does not pay for it again.

    Cut off or refused because the page kept navigating, or a navigation that failed onto
    Chromium's own document (:func:`_refuse_landing`).

    Transport-private: only :func:`render_page` knows a browser actually ran (see
    :class:`RenderTimeout`), so no caller writes this one.
    """
    _note(url, memo_scope, "timed_out")


def _note(url: str, memo_scope: MemoScope, kind: _MemoKind) -> None:
    key = (memo_scope, url)
    _RENDER_MEMO[key] = kind
    _RENDER_MEMO.move_to_end(key)
    while len(_RENDER_MEMO) > _RENDER_MEMO_MAX_ENTRIES:
        _RENDER_MEMO.popitem(last=False)


def _warn_playwright_unavailable_once(exc: BaseException) -> None:
    global _PLAYWRIGHT_WARNED  # noqa: PLW0603  # one-shot process-wide warn latch so the rendered rung logs once per run
    if _PLAYWRIGHT_WARNED:
        return
    _PLAYWRIGHT_WARNED = True
    logger.warning("rendered fetch rung unavailable: %s: %s", type(exc).__name__, exc)


def _ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _host_resolver_rule(host: str, ip: str) -> str:
    """Build the Chromium ``--host-resolver-rules`` MAP value pinning ``host`` to ``ip``.

    IPv6 literals must be bracketed in the MAP target per Chromium's rule parser
    (``MAP host [dead::beef]``); IPv4 literals are bare. A malformed ``ip`` is
    passed through unbracketed — callers only ever feed this a value already
    vetted by :func:`resolve_pinned_host`, so that branch is defensive only.
    """
    parsed_ip = _ip_literal(ip)
    target = f"[{ip}]" if parsed_ip is not None and parsed_ip.version == 6 else ip
    return f"--host-resolver-rules=MAP {host} {target}"


def pinnable_url_host(url: str) -> str | None:
    """Hostname of a URL eligible for DNS pinning, or None when the URL itself disqualifies it.

    Serves both pin syntaxes: Chromium's ``--host-resolver-rules=MAP`` and libcurl's
    ``CURLOPT_RESOLVE``. A pin is only as good as its host match, and both matchers compare a
    canonicalised host that ``urlparse`` does not produce. Chromium canonicalises (punycode for a
    unicode name, the trailing dot dropped) before it consults ``--host-resolver-rules``, and a
    ``CURLOPT_RESOLVE`` entry matches libcurl's own host string exactly; a pattern or entry built
    from a unicode or trailing-dot host is accepted and matches NOTHING, so the client resolves
    through its own resolver and the rebinding window the pin closes re-opens. Failing closed on a
    non-ASCII or trailing-dot host is the only safe answer for either matcher, and reproducing
    Chromium's canonicalisation is not: the stdlib ``idna`` codec is IDNA2003 and diverges from
    Chromium's UTS#46 form on real names (``straße`` maps to ``strasse`` there and to
    ``xn--strae-oqa`` in Chromium), so it would emit a pattern that looks right and is inert. So
    this rejects unicode, trailing-dot, userinfo and non-HTTP(S) hosts for both callers rather
    than relaxing the rule on one client's reasoning; relaxing it would reopen the pin on the other.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    # Userinfo defeats hostname-based trust (`https://trusted@10.0.0.1/`).
    if parsed.username is not None or parsed.password is not None:
        return None
    host = parsed.hostname
    if not host or not host.isascii() or host.endswith("."):
        return None
    return host


async def resolve_pinned_host(url: str) -> tuple[str, str] | None:
    """Vet ``url``'s host and resolve it to ONE public IP for DNS pinning, browser or libcurl.

    Returns ``(host, vetted_ip)`` or ``None`` when the URL is non-public, unresolvable, or ANY
    resolved address is disallowed. ``vetted_ip`` is a BARE address string, never bracketed: each
    caller brackets an IPv6 literal itself for its own syntax (Chromium's ``--host-resolver-rules
    =MAP host [ip]`` in :func:`_host_resolver_rule`, libcurl's ``host:port:[ip]`` in the
    impersonation transport's ``_resolve_entry``), so returning the bracketed form here would make
    one caller's pin match nothing. That bare-address return is part of the contract. Mirrors
    :func:`guard.is_public_http_url`'s classification (scheme, userinfo, and the shared
    :func:`guard.resolve_vetted_public_ip` predicate) so neither client can dial an
    address the airtight aiohttp ``FilteringResolver`` path would also refuse.

    This is what closes the DNS-rebinding TOCTOU on both pinned transports: the per-request route
    guard runs its own ``getaddrinfo`` independently of the client's socket connect, so a rebinding
    host (TTL 0) can pass the preflight and connect to a private IP. Pinning the host to a single
    pre-vetted IP removes that second resolution entirely — the connect can only reach the vetted
    address, whether that pin is Chromium's ``--host-resolver-rules`` or libcurl's
    ``CURLOPT_RESOLVE``.

    Fails CLOSED: on any rejection the browser caller skips Chromium for that host (its fetch
    ladder degrades to whatever the plain rung already had), and the impersonation caller declines
    the retry with :class:`impersonated_fetch.ImpersonateUnpinnable`.

    Every predicate is read off the ``guard`` module at call time, which is what keeps the guard
    to ONE patch surface: both fetch paths' suites monkeypatch ``guard.is_public_http_url`` /
    ``resolve_vetted_public_ip`` / ``_ip_is_disallowed`` there, and a second binding is how a
    patch at the wrong module stays green while proving nothing.
    """
    host = pinnable_url_host(url)
    if not host:
        return None

    # IP-literal host: no DNS to rebind. Vet directly and pin to itself.
    literal = _ip_literal(host)
    if literal is not None:
        if guard._ip_is_disallowed(literal):
            return None
        return host, str(literal)

    vetted_ip = await guard.resolve_vetted_public_ip(host)
    if vetted_ip is None:
        return None
    return host, vetted_ip


async def _vet_route(route: Any, request: Any, playwright_error: type[BaseException]) -> None:
    """Let one request Chromium is about to make through, or abort it.

    ``playwright_error`` is passed in because the class comes from the caller's function-scoped
    optional import. A request can still be in flight when the page/context tears down (typically
    after a goto timeout): ``continue_``/``abort`` then races the close and raises
    ``TargetClosedError`` in this detached event-listener task — the unhandled-error storm seen
    2026-07-25. It is swallowed because a closed target has no live socket, so an abort that
    "fails" because the target is already gone still lets nothing through and the SSRF guarantee
    is unaffected; ``unroute_all`` in the caller's ``finally`` is the primary drain and this is the
    residual-race backstop. Only Playwright's own Error is caught, so a genuine bug still
    propagates.

    The guard is read off the ``guard`` module at call time for the reason
    :func:`resolve_pinned_host` states.
    """
    try:
        if await guard.is_public_http_url(request.url):
            await route.continue_()
        else:
            await route.abort("blockedbyclient")
    except playwright_error as exc:  # the teardown race documented above, not a broad catch
        logger.debug("rendered fetch route guard race during teardown: %s", exc)


def _block_web_socket(web_socket_route: Any) -> None:
    """Refuse one WebSocket the page tried to open, by doing nothing with it.

    A routed socket dials the server only if its handler calls ``connect_to_server()``
    (Playwright 1.61: ``_after_handle`` sends ``ensureOpened`` instead, so the page sees an open
    socket that goes nowhere). Not calling it is the whole block. Registered with
    ``context.route_web_socket`` because ``context.route`` never sees a handshake: HTTP
    interception is the CDP ``Fetch`` domain, WebSockets surface only on the report-only
    ``Network.webSocket*`` events, and an IP-literal target such as ``ws://127.0.0.1`` never
    consults the resolver the DNS pin rewrites.

    The mechanism, and its limits, read out of the pinned Playwright 1.61 driver source on
    2026-09-04. Registering the pattern makes the driver inject an init script into the context
    at that moment, before any socket exists, and that script runs in every frame of every page
    the context opens and replaces ``globalThis.WebSocket`` with a routed stand-in; the page
    then sees an object that reports itself open and delivers nothing. So this is an IN-PAGE
    mitigation rather than a network boundary: a dedicated Worker has its own global scope,
    which the injected script does not reach, so a socket opened from one very likely keeps the
    native constructor and is not routed here. The injection also leaves two enumerable globals
    in every frame, ``__pwWebSocketBinding`` and ``__pwWebSocketDispatch``, which a page can
    read; that is a fingerprinting surface on the one rung whose job is getting past hostile
    edges, and it is unmeasured. Chromium's own Local Network Access check stands behind the
    Worker residual for local and loopback targets (see the route-guard comment in
    :func:`_render_in_context`), and a filtering forward proxy is the remedy that would make
    this a boundary (item 8 of the retired design log (git history)).

    The log line is INFO, deliberately: ``cli.py`` configures the root logger at INFO, so a DEBUG
    line is dropped in every real run, and this line is the only record that a page's socket-fed
    content was withheld from the render. There is no count key or marker for it yet; the
    2026-09-04 probe measured 0 of 22 render targets opening a socket, so the counter waits for
    a real INFO line to show up in a run log.

    Provably raise-free, deliberately. Playwright dispatches this on a task it creates itself, the
    detached-listener shape behind the 2026-07-25 traceback storm, and it stays registered through
    the context close (``unroute_all`` clears HTTP routes only and there is no
    ``unroute_web_socket``). So the host is taken with ``str.partition``, which cannot raise on any
    string, rather than ``urlparse``, which raises on an unbalanced IPv6 bracket; the userinfo is
    dropped so a credentialled socket URL does not reach the log.
    """
    host_port = web_socket_route.url.partition("//")[2].partition("/")[0].rpartition("@")[2]
    logger.info("rendered fetch blocked a page WebSocket to %s", host_port)


def _landed_off_host(final_url: str, pinned_host: str) -> bool:
    """Whether the main frame's landing URL is anywhere but ``pinned_host``.

    A guard that fails SHUT. The only landings it lets through without a matching hostname are
    the no-document ones: an empty ``page.url``, and an ``about:`` URL with NO hostname
    (``about:blank`` when a navigation never committed, ``about:srcdoc``), which are nobody's host
    and fall through to the empty DOM read they always produced. The scheme alone is not the
    allowance: ``about://evil.example/`` parses with a hostname, and let through on its scheme it
    would reach ``RenderedPage.document_url`` as the base the callers classify against. So the
    invariant every allowed landing satisfies is "no hostname, or the pinned one", which is exactly
    what ``document_url`` relies on when it falls back to the requested URL. Everything else is
    compared by ``urlparse`` hostname, lower-cased with the brackets stripped, the same form the pin
    was built from, and a URL with no hostname at all is off-host. That covers an http(s) URL with
    an empty authority, every non-http(s) scheme Chromium can leave the frame on (``data:``,
    ``file:``, ``blob:``), and Chromium's own error document: live QA on 2026-09-04 observed
    ``page.url == "chrome-error://chromewebdata/"`` after a goto that failed with
    ``net::ERR_UNSAFE_PORT`` and after ``net::ERR_CONNECTION_REFUSED`` on a redirect to a loopback
    target, and the tri-state predecessor of this helper, which allowed every non-http(s) scheme,
    let that landing through to a DOM read that returned an empty page carrying that URL. Under a
    boolean whose allow case is the named sentinel it is refused with no arm of its own.
    """
    parsed = urlparse(final_url)
    if not final_url or (parsed.scheme.lower() == "about" and parsed.hostname is None):
        return False
    return parsed.hostname != pinned_host


def _refuse_landing(requested_url: str, final_url: str, pinned_host: str, *, memo_scope: MemoScope) -> RenderOffHost:
    """Build the off-host refusal, memoising first when the landing is Chromium's own document.

    Two populations share :class:`RenderOffHost`, and only one is worth remembering. A landing
    whose scheme is not http(s) is a document the browser made rather than a redirect to a
    stranger's host: ``chrome-error://chromewebdata/`` after a failed navigation, or a ``data:``,
    ``blob:`` or ``file:`` document the page navigated itself to. The page rendered nothing a caller
    could read and a retry within the run changes nothing, which is exactly the population the
    empty-DOM read used to memoise through the callers' no-text memo, so without a memo a run
    relaunched Chromium (100-300 MB and one of the two slots) for the same dead URL on every later
    question that cited it. The timed-out memo is the one the transport owns, and it is written
    here, at the raise site, for the reason :class:`RenderTimeout` gives. An http(s) landing on
    another host is memoised by nobody: it is a fact about where the page sent the browser, not
    about the page having nothing, and the next question citing it runs the render and is refused
    on its own record.
    """
    if urlparse(final_url).scheme.lower() not in ("http", "https"):
        _note_render_timeout(requested_url, memo_scope=memo_scope)
    return RenderOffHost(requested_url=requested_url, final_url=final_url, pinned_host=pinned_host)


def _goto_budget_ms(goto_timeout_ms: int, deadline_monotonic_s: float | None) -> int | None:
    """The navigation budget once both gates are held, or None when it is under the floor.

    Computed AFTER the acquires, for the same reason ``_fetch_one_hop`` computes its hop timeout
    after the host semaphore: the caller measured its budget before two unbounded waits — the
    loop-wide per-host gate shared by every concurrent question, and the process-global launch
    cap shared with the other fetch path — and a render admitted after queueing behind both
    would otherwise navigate on a figure that expired in the queue. The caller's figure stays a
    CEILING (it carries the transport's own 35 s cap); the deadline, when given, tightens it to
    what is actually left less the post-goto tail, so a goto that runs its budget out can still
    be settled and read inside the caller's bound (:data:`RENDER_POST_GOTO_TAIL_MS`). Without a
    deadline the caller's figure stands: gap-fill v2 bounds the whole call with its own ceilings.
    """
    budget_ms = goto_timeout_ms
    if deadline_monotonic_s is not None:
        remaining_ms = int((deadline_monotonic_s - time.monotonic()) * 1000) - RENDER_POST_GOTO_TAIL_MS
        budget_ms = min(budget_ms, remaining_ms)
    return budget_ms if budget_ms >= RENDER_MIN_GOTO_MS else None


async def _in_own_task(call: Any, *, timeout_ms: int) -> Any:
    """Await one Playwright call under a bound, in a task of its own.

    The task is the point, not a convenience. Playwright registers the cancellation of a protocol
    call on the TASK that sent it: when that task finishes cancelled, the pending reply future is
    cancelled with it. ``asyncio.wait_for`` on a bare coroutine runs it in the current task and
    converts the cancellation into ``TimeoutError``, so the current task never finishes
    cancelled, the reply future stays pending, and the late reply (or the ``TargetClosedError``
    teardown sets on it) has nobody to retrieve it — one "Future exception was never retrieved"
    ERROR per cut-off. Wrapping the call in its own task makes the cancellation land where
    Playwright looks for it.
    """
    return await asyncio.wait_for(asyncio.ensure_future(call), timeout=timeout_ms / 1000)


class _TeardownBudget:
    """One ``RENDER_TEARDOWN_TIMEOUT_MS`` shared by every teardown step of one render.

    Started by the FIRST step that asks, not when the render begins: computed eagerly before the
    launch it would already be spent by the time teardown runs, and every step would be abandoned
    to the driver stop. Each later step gets what the earlier ones left, floored at zero, so the
    steps together hold a render for at most one bound past its last read whatever the browser
    does. A step handed zero is scheduled and cancelled before it runs, which the driver stop then
    covers, rather than skipped: the one code path keeps the one log line.
    """

    def __init__(self) -> None:
        self._deadline_s: float | None = None

    def remaining_ms(self) -> int:
        now = time.monotonic()
        if self._deadline_s is None:
            self._deadline_s = now + RENDER_TEARDOWN_TIMEOUT_MS / 1000
        return max(0, int((self._deadline_s - now) * 1000))


async def _teardown_step(name: str, call: Any, playwright_error: type[BaseException], budget: _TeardownBudget) -> None:
    """Run one teardown call so that it can neither wedge the render nor replace its exception.

    Bounded by what is left of the render's shared teardown ``budget``; past it the call is
    abandoned and the driver stop kills the browser. Playwright's own errors are swallowed at
    DEBUG because teardown races them by construction: ``BrowserContext.close`` does not swallow
    ``TargetClosedError`` the way ``Browser.close`` does, and the next protocol call re-raises any
    error a detached listener stored on the connection, so an unguarded close in a ``finally``
    would replace the :class:`RenderTimeout` (or the caller's cancellation) unwinding through it
    and turn a cut-off render into "the renderer is unavailable". Anything else is a bug and
    propagates.
    """
    remaining_ms = budget.remaining_ms()
    try:
        await _in_own_task(call, timeout_ms=remaining_ms)
    except TimeoutError:
        logger.warning(
            "rendered fetch teardown: %s did not finish in the %dms left of the %dms teardown budget; "
            "leaving it to the driver stop",
            name,
            remaining_ms,
            RENDER_TEARDOWN_TIMEOUT_MS,
        )
    except playwright_error as exc:
        logger.debug("rendered fetch teardown: %s raced the close: %s", name, exc)


async def _navigate_and_read_dom(
    page: Any,
    url: str,
    playwright_error: type[BaseException],
    *,
    memo_scope: MemoScope,
    goto_timeout_ms: int,
    deadline_monotonic_s: float | None,
    harvest: _JsonHarvest | None,
    pinned_host: str,
) -> RenderedPage:
    """Navigate to ``url``, let it settle, and return the rendered page.

    Two departures from a plain ``networkidle`` navigation, both measured 2026-09-03. The wait
    condition is ``domcontentloaded`` plus a fixed settle, because network idle never arrives on
    a page carrying a long-poll widget or an analytics beacon. And a goto failure is SALVAGED
    rather than treated as a dead rung: Playwright's ``TimeoutError`` subclasses ``Error``, and a
    timed-out goto routinely leaves a fully rendered DOM behind — 4 of the replay's 10 render
    rescues came from exactly that. A genuine navigation error lands here too and salvages an
    empty ``about:blank``, which reaches the ladder as the same "rendered read nothing".

    The main frame has to be on ``pinned_host``, the one hostname the launch's
    ``--host-resolver-rules`` covers, and that is checked twice. A server-side redirect hop is
    followed by Chromium with no route handler of ours involved, so ``page.url`` is the only
    record of where the frame actually went, and it is read after the settle on BOTH paths: on the
    salvage path ``response`` is None and there is no other source, and that path is where a
    timed-out navigation may already have landed on a redirect target. A landing on any other
    host, an IP literal included, raises :class:`RenderOffHost` before ``page.content()`` is
    called, which is the cheap early exit (memoised by halves, see :func:`_refuse_landing`: a
    failed navigation left on Chromium's own document is, a redirect to a stranger's host is not).
    Then the same check runs again on ``page.url`` as it
    stands after the read, because a navigation can commit in the one driver round trip between
    the first check and the read (see the comment at that line), and a mismatch there raises the
    same exception with the DOM that was read discarded unpublished. The page's ``final_url`` is
    that second read, the document the DOM came from. Hostnames are compared as ``urlparse``
    normalises them (lower-cased, brackets stripped), which is the same form the pin was built
    from, so the compare is like with like, and the predicate fails shut
    (:func:`_landed_off_host`): ``about:blank`` is nobody's host and falls through to the empty
    read it always produced, and every other scheme is a stranger. One deliberate departure from
    the design note in item 8 of the retired design log (git history), which also asked for the goto response's URL to be
    compared when a response exists: only ``page.url`` is checked, because after the settle it is
    the authoritative record of the document whose DOM is read, and the response URL adds no
    DOM-safety value (an off-host response whose document then navigated the frame back onto the
    pinned host inside the settle leaves ``content_type`` and ``http_status`` from that response
    beside a DOM from the pinned host, a provenance mismatch on two metadata fields rather than a
    hole). Refusing it would also cost recall on a benign bounce, on a surface reserved for
    strictly-safer changes.

    The DOM read is bounded by ``RENDER_DOM_READ_TIMEOUT_MS`` and raises :class:`RenderTimeout`
    when it fires — deliberately NOT swallowed into the salvage, because a page that keeps
    navigating has no DOM to salvage and the caller needs to tell this from a browser that is
    missing or broken (see :func:`render_page`). The browser's own refusal of the read, Playwright's
    plain ``Error`` for a page that navigated during the evaluate, is the same fact about the page
    and is caught at this one call and raised as the same class; every other Playwright error on
    the render path keeps its own boundary. The timed-out memo for ``memo_scope`` is written
    here, immediately before the raise, so it lands even when the caller's own cut arrives while
    the exception is still unwinding through teardown. The harvest's in-flight body reads are
    drained INSIDE what is left of that same bound, never after it, and no later than the
    caller's ``deadline_monotonic_s`` when a deadline was given: the harvest is opportunistic and
    may not lengthen a render whose real product is the DOM. The second clamp is there because
    the read bound is fixed and the launch runs after the budget recompute, so in the salvage
    shape the read bound lands past the transport's deadline by the launch time, and a body still
    in flight there used to hold the drain into the caller's exit reserve and past its outer cut,
    which then discarded a DOM this function had already read. The deadline already excludes
    that reserve (:data:`RENDER_EXIT_RESERVE_MS`), so nothing is subtracted from it again here.
    The DOM read itself keeps its fixed bound (see :data:`RENDER_DOM_READ_TIMEOUT_MS` for why).

    The DOM is measured in characters against ``RENDERED_DOM_MAX_CHARS`` before anything copies
    it, because the copies are the hazard: the Tier-1 caller encodes it, decodes it back,
    rewrites it and hands trafilatura a tree several times its size, all while the browser is
    still resident. A DOM over the ceiling raises :class:`RenderDomOverCeiling`, its own decline,
    so the caller can count it apart from a browser that is missing.
    """
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout_ms)
    except playwright_error as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # the salvage above: Playwright's own Error class, passed in from the optional import
        logger.debug("rendered fetch goto failed, salvaging DOM: %s: %s", type(exc).__name__, exc)
        response = None
    await page.wait_for_timeout(RENDER_SETTLE_MS)
    final_url: str = page.url
    if _landed_off_host(final_url, pinned_host):
        raise _refuse_landing(url, final_url, pinned_host, memo_scope=memo_scope)
    # ``goto`` returns None for an about:blank or same-document navigation, and the salvage
    # above leaves no response either; both read as "no main-frame response".
    content_type = (response.headers.get("content-type") or "").lower() if response is not None else ""
    http_status: int | None = response.status if response is not None else None
    read_deadline_s = time.monotonic() + RENDER_DOM_READ_TIMEOUT_MS / 1000
    try:
        html = await _in_own_task(page.content(), timeout_ms=RENDER_DOM_READ_TIMEOUT_MS)
    except TimeoutError as exc:
        _note_render_timeout(url, memo_scope=memo_scope)
        raise RenderTimeout(
            f"the DOM read of {urlparse(url).netloc} outlived {RENDER_DOM_READ_TIMEOUT_MS}ms: the page kept navigating"
        ) from exc
    except playwright_error as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # Playwright's own Error class, passed in from the optional import; this one call only
        # The driver's other answer to the same page (see :class:`RenderTimeout`): in 1.61,
        # ``Frame.content()`` evaluates ``outerHTML`` once and raises its plain ``Error`` when the
        # page navigates mid-evaluate. Left to the boundary in ``render_page`` it was classified
        # as the renderer being unavailable, which latched the once-per-run warning and memoised
        # nothing. Every Playwright error out of this one call is read as the page rather than the
        # install; the browser dying during this exact call would ride along, a shape the
        # 2026-09-04 probe never produced, and the alternative is matching on the message text.
        _note_render_timeout(url, memo_scope=memo_scope)
        raise RenderTimeout(
            f"the browser refused the DOM read of {urlparse(url).netloc}: the page kept navigating ({exc})"
        ) from exc
    # The same check again, on the document the read actually came from. `page.url` is a cached
    # attribute the driver updates on its `navigated` event and `page.content()` is a round trip
    # evaluated in whatever document is current when the driver handles it, so a navigation that
    # committed between the check above and the read hands back the other host's DOM with that
    # check already passed. The driver's pipe is ordered, so that commit's `navigated` event
    # arrived before the content reply and this second read sees it. No await and no protocol
    # call: reading `page.url` costs a local attribute access, so the render is no longer for it.
    final_url = page.url
    if _landed_off_host(final_url, pinned_host):
        raise _refuse_landing(url, final_url, pinned_host, memo_scope=memo_scope)
    if len(html) > RENDERED_DOM_MAX_CHARS:
        logger.warning(
            "rendered fetch declined the DOM of %s: %d chars is over the %d-char ceiling",
            urlparse(url).netloc,
            len(html),
            RENDERED_DOM_MAX_CHARS,
        )
        raise RenderDomOverCeiling(
            f"the rendered DOM of {urlparse(url).netloc} is {len(html)} chars, over the {RENDERED_DOM_MAX_CHARS}-char ceiling"
        )
    json_responses: tuple[HarvestedJson, ...] = ()
    if harvest is not None:
        drain_until_s = read_deadline_s if deadline_monotonic_s is None else min(read_deadline_s, deadline_monotonic_s)
        await harvest.drain(until_monotonic_s=drain_until_s)
        json_responses = tuple(harvest.bodies)
    return RenderedPage(
        url=url,
        content_type=content_type,
        html=html,
        json_responses=json_responses,
        http_status=http_status,
        final_url=final_url,
    )


def _same_publisher(host: str | None, other: str) -> bool:
    """True when ``host`` and ``other`` belong to one publisher, by registrable domain.

    The public suffix plus one label, from the public-suffix list vendored by
    ``research/public_suffix.py`` (``registrable_domain``). A page on ``www.x.gov`` whose data
    endpoint is ``api.x.gov`` or ``data.x.gov`` is the ordinary dashboard shape, and a page on
    ``forest-fire.emergency.copernicus.eu`` fed by ``api2.effis.emergency.copernicus.eu`` is the
    sibling-subdomain shape live QA found (2026-09-03); a bare ``endswith`` relation matched
    neither. The PSL is also what makes ``a.co.uk`` and ``b.co.uk`` strangers and a stranger's site
    on a shared suffix (``x.github.io`` against ``github.io``) no match. IP literals have no
    registrable domain and the PSL algorithm would collapse two of them to their last two octets,
    so they are compared exactly and never otherwise, and a missing hostname is nobody's publisher.

    Two readers, one judgment: the XHR harvest deciding whose JSON a page fetched
    (:func:`_harvestable_json_host`), and the ``RENDERED_FETCH_OFF_HOST`` marker's
    ``same_publisher`` field, which tells a benign hop inside the publisher's own domain from a
    landing on a stranger's.
    """
    if not host or not other:
        return False
    if host == other:
        return True
    if _ip_literal(host) is not None or _ip_literal(other) is not None:
        return False
    publisher = registrable_domain(other)
    return publisher is not None and registrable_domain(host) == publisher


def _harvestable_json_host(response_host: str, page_host: str) -> bool:
    """True when ``response_host`` may serve harvestable JSON for a page on ``page_host``.

    Same publisher (:func:`_same_publisher`), which is exactly where a wrong answer reads someone
    else's JSON as the cited page's content, or one of the explicitly allow-listed CDNs, the one
    exception to same-publisher.
    """
    if not response_host or not page_host:
        return False
    if response_host in HARVEST_ALLOWED_CDN_HOSTS:
        return True
    return _same_publisher(response_host, page_host)


def is_json_content_type(content_type: str) -> bool:
    """Whether a (lower-cased) Content-Type names a JSON body: ``application/json``, ``text/json``
    or any ``+json`` structured suffix (``application/geo+json``, ``application/vnd.api+json``).

    The ONE JSON vocabulary for the fetch paths. The harvest applies it at discovery, the
    derived-feed rung at reuse, and the resolution-source 200-response router when it decides
    what a directly cited URL is; a narrower spelling at any one of them strands a feed the
    others accept, and a remembered endpoint is first-find-wins for the run.
    """
    return any(token in content_type for token in _JSON_CONTENT_TYPE_TOKENS)


def _declared_length_over_cap(content_length: str | None) -> bool:
    """Whether the response DECLARES a body over the harvest cap.

    The pre-screen: ``Response.body()`` materialises the whole body (base64 over the driver pipe,
    then decoded, so ~2.3x the body at peak) before any post-read size test can run, and a
    dashboard's 60 MB GeoJSON would buffer beside a 100-300 MB browser on the strength of its
    content type alone. A declared length over the cap is enough to refuse the read: for a
    compressed response it is the compressed size, and the decoded body is only bigger. Absent,
    unparseable or lying headers fall through to the post-read test, which stays as the backstop.

    That backstop is the harvest's one residual. HTTP/2 and chunked responses routinely declare
    no length, and Playwright's ``body()`` offers no streaming route, so a single undeclared
    oversized body is buffered whole before the post-read size test drops it. The reads are
    serialised (:class:`_JsonHarvest`), so that is at most ONE such body at a time rather than
    every layer a dashboard requests at once; refusing undeclared bodies outright would drop the
    payload the rung exists to find.
    """
    if content_length is None:
        return False
    try:
        return int(content_length) > HARVEST_MAX_BODY_BYTES
    except ValueError:
        return False


def _harvestable_json_response(response: Any, *, page_host: str) -> bool:
    """Whether a response event is worth reading: same publisher, JSON, not declared over the cap.

    Synchronous on purpose, so the listener can apply it before any task exists: ``response.url``
    and ``response.headers`` (lower-cased names) are sync properties of Playwright's documented
    ``Response`` contract, and the fake in the tests honours the same one. Every rejection is
    silent: this is opportunistic discovery attached to a render whose real product is the DOM.
    """
    if not _harvestable_json_host(urlparse(response.url).hostname or "", page_host):
        return False
    headers = response.headers
    if not is_json_content_type((headers.get("content-type") or "").lower()):
        return False
    return not _declared_length_over_cap(headers.get("content-length"))


class _JsonHarvest:
    """The JSON a page fetches for itself, collected from its response events during one render.

    Owns the tasks the events spawn. Playwright's ``Page`` is a pyee emitter, and pyee wraps a
    coroutine listener in ``ensure_future`` and forgets it, so every ``response`` firing used to
    be a detached task nobody joined: the snapshot was taken while reads were still pending and
    teardown then made their ``body()`` raise into a swallowed path, which dropped exactly the
    derived-API rung's payload — and the miss stuck, because the caller memoised the empty render.
    The listener is therefore SYNC and creates the tasks itself, so pyee neither tracks them nor
    re-emits their exceptions on the page's ``error`` event, and :meth:`drain` joins them before
    the snapshot, inside the DOM read's own bound.

    Every bound here is a memory bound, and two things make it hold. The listener SCREENS before
    it spawns, so a page's hundreds of subresources never become tasks and only a response that
    will actually be read does. And the reads are SERIALISED behind ``_read_gate``: every
    response event fires before any body comes back, so without it every response that cleared
    the screens entered ``body()`` together and the count cap bounded nothing, while Playwright
    materialised each body base64 over the driver pipe and then decoded it (about 2.3x the body
    at peak) beside a 100-300 MB browser. One body at a time makes the count check under the
    gate exact and peak harvest memory one body.
    """

    def __init__(self, *, page_host: str, playwright_error: type[BaseException]) -> None:
        self._page_host = page_host
        self._playwright_error = playwright_error
        self.bodies: list[HarvestedJson] = []
        self._pending: set[asyncio.Task[None]] = set()
        self._read_gate = asyncio.Semaphore(1)

    def on_response(self, response: Any) -> None:
        if len(self.bodies) >= HARVEST_MAX_RESPONSES:
            return
        if not _harvestable_json_response(response, page_host=self._page_host):
            return
        task = asyncio.create_task(self._read(response))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _read(self, response: Any) -> None:
        """Read one screened body under the gate and keep it if its size clears the floor and cap.

        The body is read inside the response event's lifetime so it is still available (after the
        page navigates or closes, Playwright discards it). A body we could not read must never be
        able to fail the render, so Playwright's own error is a DEBUG line and nothing more. The
        count is checked again under the gate, where it is exact: no other read is in flight, and
        the append below runs before this task yields again.
        """
        async with self._read_gate:
            if len(self.bodies) >= HARVEST_MAX_RESPONSES:
                return
            try:
                body = await response.body()
            except self._playwright_error as exc:
                logger.debug("rendered fetch could not read a harvested response body: %s", exc)
                return
        if not (HARVEST_MIN_BODY_BYTES <= len(body) <= HARVEST_MAX_BODY_BYTES):
            return
        self.bodies.append(HarvestedJson(url=response.url, body=body))

    async def drain(self, *, until_monotonic_s: float) -> None:
        """Wait for the in-flight body reads, but no later than ``until_monotonic_s``; cancel the rest."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=max(0.0, until_monotonic_s - time.monotonic()))
        self.cancel_pending()

    def cancel_pending(self) -> None:
        for task in self._pending:
            task.cancel()


async def render_page(
    url: str,
    *,
    memo_scope: MemoScope,
    host_gate: asyncio.Semaphore,
    goto_timeout_ms: int = RENDER_TIMEOUT_MS - RENDER_SETTLE_MS,
    deadline_monotonic_s: float | None = None,
    harvest_json: bool = False,
) -> RenderedPage | None:
    """Render ``url`` in headless Chromium and return its DOM, or None when the rung declines.

    ``None`` is the one graceful-failure signal both callers already handle, and it covers
    every way this rung can DECLINE with nothing rendered: a URL a browser already read to
    nothing in this run under the caller's own ``memo_scope``, Playwright missing or broken, a
    host that cannot be pinned to a public IP, and any error out of the browser. Three declines
    are NOT ``None``, because each is a fact the caller counts apart from a missing browser: a
    navigation budget under the floor once the gates are held raises :class:`RenderBudgetExpired`
    (the caller's wall budget ran out in the queue), a rendered DOM over
    ``RENDERED_DOM_MAX_CHARS`` raises :class:`RenderDomOverCeiling` (the page rendered and is too
    big to read safely), and a main frame that landed on a host other than the pinned one raises
    :class:`RenderOffHost` (the page sent the browser somewhere the pin does not cover, and its
    DOM was refused unread or discarded unpublished). A caller that wants to tell the ``None`` causes apart reads
    :func:`rendered_to_nothing` and :func:`resolve_pinned_host` itself.

    A render that ran and was CUT OFF is different, and raises :class:`RenderTimeout` (a
    builtin ``TimeoutError``) instead: the DOM read outlived ``RENDER_DOM_READ_TIMEOUT_MS``, or the
    browser refused it with its own "navigating" error, because the page kept navigating. It is
    not folded into ``None`` because the two mean
    different things to a caller — a timeout says nothing about whether Chromium works, so it
    must neither trip the once-per-process unavailable warning nor be recorded as the browser
    being absent — and it is its own class so the Tier-1 rung, which also bounds this whole call
    with a ``wait_for`` of its own, can tell a render that ran out the clock (this exception) from
    one its outer bound cut while it was still queued behind the two gates (a bare
    ``TimeoutError``, which says nothing about the page). The transport memoises the timed-out
    URL for the run at the raise site — and only there, for that same reason — and a memoised
    timeout is RE-RAISED here rather than declined, so a second question citing the same hostile
    page records the same reason again instead of reading as a missing browser. The same memo is
    written for a navigation that failed onto Chromium's own document, the failed-navigation half
    of :class:`RenderOffHost`, so a dead URL is not relaunched for either; from the second question
    on it is recorded as ``render_timeout``.

    What a deadline bounds, and what it does not. ``deadline_monotonic_s`` is the instant by
    which the transport must have its DOM (and any harvested bodies) in hand: the goto is sized
    off it less the post-goto tail, the harvest drain is clamped to it, and the salvage DOM read
    can land at most the launch time past it. Teardown then runs inside the shared
    ``RENDER_TEARDOWN_TIMEOUT_MS`` bound, and the driver stop at the end of the
    ``async_playwright()`` block runs after that, so a caller with a hard wall has to hand in a
    deadline that reserves both (:data:`RENDER_EXIT_RESERVE_MS`). The driver stop is the one
    step left UNBOUNDED, deliberately: it is what actually kills the Chromium process, and
    abandoning it on a wedged browser would leak 100-300 MB past ``RENDER_LAUNCH_CAP``, an OOM
    nothing can catch. A healthy stop is milliseconds and fits inside the reserve's spare second.

    The failure boundary around the browser is split three ways, and the split is the point.
    Playwright's own ``Error`` keeps the once-per-process latch: a launch that cannot find the
    executable is "the renderer is unavailable" and one line per run is the right volume.
    Anything else is a bug in this module or a dependency behaving unexpectedly, and it is
    logged with a full traceback EVERY time — one line per run and then silence was how a
    programming error in the render path hid behind the latch (2026-09-03). It is still
    swallowed rather than raised, because a raise here propagates out of the Tier-1 provider's
    ``gather`` and cancels the question's OTHER pages, which is the one thing a rung failure
    must never do.

    ``host_gate`` is the caller's own per-host politeness semaphore for this URL — passed in
    rather than derived, because the two fetch paths keep separate maps (Tier-1's is
    loop-wide across concurrent questions; v2's is its own module global). It is acquired
    OUTSIDE the launch cap, in that order, and HELD across the launch-cap queue, the launch,
    the navigation, the settle and the teardown, because Chromium dials that host itself. Both
    acquires are unbounded by design (bounding them is the operator's reserved call, backlog task-14 "The 45 s resolution-source wall discards pages that already succeeded"), which is why the navigation budget is recomputed only once both are held: pass
    ``deadline_monotonic_s`` and the goto gets what is left of it less the post-goto tail, or
    the render declines under ``RENDER_MIN_GOTO_MS`` before a browser is launched
    (:func:`_goto_budget_ms`).

    ``harvest_json`` turns on recording the JSON the page fetches for itself. Off by default:
    the bodies are buffered alongside a 100-300 MB browser, and only the caller that has a
    use for a derived endpoint should pay that.
    """
    if rendered_to_nothing(url, memo_scope=memo_scope):
        logger.debug("rendered fetch skipped (already rendered to nothing): %s", urlparse(url).netloc)
        return None
    if render_timed_out(url, memo_scope=memo_scope):
        logger.info("rendered fetch skipped (a render already ran to no DOM this run): %s", urlparse(url).netloc)
        raise RenderTimeout(
            f"a render of {urlparse(url).netloc} already ran to no DOM this run: cut off, or a failed navigation"
        )
    try:
        from playwright.async_api import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
            Error as PlaywrightError,
        )
        from playwright.async_api import async_playwright  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # optional-dep import boundary: playwright missing/broken degrades the rendered rung, never the run
        _warn_playwright_unavailable_once(exc)
        return None

    # Pin Chromium's DNS to a single pre-vetted public IP BEFORE launch. If the host can't be
    # resolved to a public address, fail closed: skip Chromium and let the caller's ladder fall
    # back (same graceful-failure signal as a playwright-unavailable / render-error return).
    pinned = await resolve_pinned_host(url)
    if pinned is None:
        logger.warning("rendered fetch skipped (host not pinnable to a public IP): %s", urlparse(url).netloc)
        return None
    host, vetted_ip = pinned

    try:
        async with host_gate, _RENDERED_FETCH_GLOBAL_SEMAPHORE, async_playwright() as playwright:
            return await _render_in_browser(
                playwright,
                url,
                PlaywrightError,
                memo_scope=memo_scope,
                host=host,
                vetted_ip=vetted_ip,
                goto_timeout_ms=goto_timeout_ms,
                deadline_monotonic_s=deadline_monotonic_s,
                harvest_json=harvest_json,
            )
    # The three declines the caller records under their own tokens. Nothing is memoised for the
    # first two (nothing ran for the first; the second rendered content, just too much of it). The
    # third memoises by halves at its raise site (`_refuse_landing`): a page that sent the browser
    # to another host is not, a failed navigation left on Chromium's own document is.
    except (RenderBudgetExpired, RenderDomOverCeiling):
        raise
    except RenderOffHost as exc:
        # Marker-shaped (``rendered_fetch_off_host`` in scripts/telemetry/markers.py) because this
        # is the only per-EVENT record that a page sent the browser off the pin: the caller's
        # ``render_off_host`` skip count says how many, never which host sent us where, and a
        # free-text WARNING is gone once the GitHub Actions logs expire at 90 days. ``scope`` names
        # which caller's render it was, since the two have different URL populations.
        # Hostnames only: the landing URL can carry a session token or a credential. A landing URL
        # with no hostname at all renders as ``None``, which harvests as no data. ``same_publisher``
        # is the registrable-domain judgment the harvest makes about a page's own JSON, applied to
        # the landing: strict hostname equality also refuses a benign hop inside the publisher's own
        # domain (``example.com`` to ``www.example.com``), and without the field every such record
        # read as a security event. ``true`` prices the strictness; ``false`` is the signal.
        landed_host = urlparse(exc.final_url).hostname
        logger.warning(
            "RENDERED_FETCH_OFF_HOST: scope=%s pinned_host=%s landed_host=%s same_publisher=%s",
            memo_scope,
            exc.pinned_host,
            landed_host,
            "true" if _same_publisher(landed_host, exc.pinned_host) else "false",
        )
        raise
    # Ordered before Playwright's Error deliberately, though the order is not load-bearing:
    # Playwright's TimeoutError derives from its own Error, not the builtin, so this clause never
    # sees one, and a raw OS-level TimeoutError is not a RenderTimeout and falls through to the
    # logged boundary below. The memo was already written at the raise site; only the log is here.
    except RenderTimeout as exc:
        # The exception's own message says which shape it was: the bound firing, or the browser
        # refusing the read because the page was navigating.
        logger.warning("rendered fetch timed out reading the DOM of %s: %s", urlparse(url).netloc, exc)
        raise
    except PlaywrightError as exc:
        _warn_playwright_unavailable_once(exc)
        return None
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # soft-fail boundary for the rendered rung: a raise here cancels the question's sibling pages (see the docstring); logged in full every time so it can never be silent
        logger.exception(
            "rendered fetch failed unexpectedly for %s; leaving the plain result standing", urlparse(url).netloc
        )
        return None


async def _render_in_browser(
    playwright: Any,
    url: str,
    playwright_error: type[BaseException],
    *,
    memo_scope: MemoScope,
    host: str,
    vetted_ip: str,
    goto_timeout_ms: int,
    deadline_monotonic_s: float | None,
    harvest_json: bool,
) -> RenderedPage:
    """Recompute the budget, launch, render inside a guarded context, tear both down bounded.

    Split out of :func:`render_page` so the gates it runs under, and the soft-fail boundary
    around them, read as one statement each there. The three teardown steps (the unroute inside
    the context, then the two closes here) share ONE :class:`_TeardownBudget`, so whatever the
    browser does the exit costs at most ``RENDER_TEARDOWN_TIMEOUT_MS`` before the driver stop.
    """
    teardown = _TeardownBudget()
    goto_budget_ms = _goto_budget_ms(goto_timeout_ms, deadline_monotonic_s)
    if goto_budget_ms is None:
        logger.warning(
            "rendered fetch declined after the gates for %s: under %dms of navigation budget left",
            urlparse(url).netloc,
            RENDER_MIN_GOTO_MS,
        )
        raise RenderBudgetExpired(
            f"under {RENDER_MIN_GOTO_MS}ms of navigation budget left for {urlparse(url).netloc} once the gates were held"
        )
    # --host-resolver-rules pins the browser's own resolution to the IP vetted above, so
    # Chromium's socket connect cannot independently re-resolve `host` to a private address
    # (the DNS-rebinding TOCTOU the per-request preflight alone cannot close). A fresh browser
    # is launched per call, so per-launch host-resolver-rules is clean.
    browser = await playwright.chromium.launch(headless=True, args=[_host_resolver_rule(host, vetted_ip)])
    # Two nested try/finally blocks, one per resource, so a failure at ANY step after the launch
    # — context creation, the route registration, the page, the navigation — still closes what
    # exists rather than leaving the driver stop as the only thing that kills Chromium.
    try:
        context = await browser.new_context(
            user_agent=BROWSER_HEADERS["User-Agent"],
            extra_http_headers={key: value for key, value in BROWSER_HEADERS.items() if key != "User-Agent"},
            # Service workers are the one request channel `browser_context.route` does not
            # intercept (Playwright 1.61 documents it and recommends exactly this whenever
            # interception is in use), so a worker could dial past the route guard below. Nothing
            # is lost: a fresh context has no installed worker, so no first-load content is ever
            # served from one.
            service_workers="block",
        )
        try:
            return await _render_in_context(
                context,
                url,
                playwright_error,
                memo_scope=memo_scope,
                pinned_host=host,
                goto_budget_ms=goto_budget_ms,
                deadline_monotonic_s=deadline_monotonic_s,
                harvest_json=harvest_json,
                teardown=teardown,
            )
        finally:
            await _teardown_step("context.close", context.close(), playwright_error, teardown)
    finally:
        await _teardown_step("browser.close", browser.close(), playwright_error, teardown)


async def _render_in_context(
    context: Any,
    url: str,
    playwright_error: type[BaseException],
    *,
    memo_scope: MemoScope,
    pinned_host: str,
    goto_budget_ms: int,
    deadline_monotonic_s: float | None,
    harvest_json: bool,
    teardown: _TeardownBudget,
) -> RenderedPage:
    """Guard the context, open the page, navigate and read; leave the closes to the caller."""

    # Defense-in-depth on top of the main-frame pin above, and not a complete boundary. What the route guard
    # below DOES see, and re-checks against is_public_http_url: the main-frame goto, every client-side
    # redirect (a meta refresh or a `location` assignment is a fresh request with no redirect predecessor),
    # and every subresource, XHR and fetch Playwright can attribute to a frame. Three request channels never
    # reach it, each confirmed in the pinned Playwright 1.61 driver source on 2026-09-04, and each is closed
    # or bounded somewhere else. A SERVER-SIDE redirect hop is auto-continued by the driver, which constructs
    # a Route only in the else of `if (redirectedFrom || ...)`; Playwright's own `page.route` docs say "the
    # handler will only be called for the first url if the response is a redirect". So a 3xx answer that
    # sends the main frame elsewhere is dialed with no check of ours; what closes it is the landing-host
    # check in `_navigate_and_read_dom`, which runs before AND after the DOM read, so a `page.url` off the
    # pinned host has its DOM refused unread, or discarded unpublished when the navigation committed during
    # the read itself. A request Playwright cannot attribute to a frame is auto-continued the same way, and
    # stays the one channel with no check of ours. A WebSocket handshake is invisible to `context.route` by
    # construction, because HTTP interception is the CDP `Fetch.enable` domain while WebSockets surface only
    # on the report-only `Network.webSocket*` events; it is closed for page scripts by the `route_web_socket`
    # handler registered below, `_block_web_socket`, which never connects the socket to a server. Its
    # docstring carries the mechanism and its limits, the short form being that it is an in-page replacement
    # of `globalThis.WebSocket` rather than a network boundary, so a dedicated Worker very likely keeps the
    # native constructor. Service-worker
    # traffic is a fourth such channel and is closed by `service_workers="block"` on the context above. One
    # shape that looks like a hole and is not: a CORS preflight OPTIONS is auto-fulfilled by the driver with
    # a synthetic 204 and never reaches the network.
    #
    # Threat model: these fetches run on GitHub-hosted Azure runners, where a request to a link-local /
    # RFC1918 host (Azure IMDS at 169.254.169.254, localhost services, the internal runner network) would
    # exfiltrate internal content into the research prompt AND the public Metaculus comment. The main-frame
    # host is pinned, so its own rebinding TOCTOU is closed, and a main frame that ends up on any other host
    # is refused. A subresource host is guarded only by this per-request preflight, whose getaddrinfo
    # resolves independently of Chromium's connect, so a rebinding host (TTL 0) can still win that race.
    # Chromium 149 (the build Playwright 1.61 pins) narrows that: Local Network Access gates a public page's
    # subresource requests, and since Chrome 147 its WebSocket handshakes, to local and loopback addresses
    # behind a permission this headless context never grants, which is the one boundary-level mitigation
    # standing behind the in-page WebSocket block's Worker residual. That mitigation is INFERRED from
    # Chromium's feature lists and vendor docs rather than observed, because the test suite blocks every real
    # browser launch, and it is a browser default we neither pin nor assert. A filtering forward proxy is the
    # only remedy that covers every channel at connect time,
    # and it stays deferred as its own change: item 8 of the retired design log (git history) under "Resolution-source fetcher: Tier-2 LLM
    # fetch + oversized-source summarization" carries the ranked candidates. Harvested JSON passes the guard
    # below like any other response, so a harvestable response is one Chromium was allowed to dial.
    async def _guard_route(route: Any, request: Any) -> None:
        # A thin closure so the registration keeps Playwright's expected handler shape while
        # the vetting itself stays module-level and directly testable.
        await _vet_route(route, request, playwright_error)

    await context.route("**/*", _guard_route)
    # Before `new_page`, because only sockets created after the registration are routed. Not wrapped:
    # a Playwright error here takes the same pre-page path as a failed `route` or `new_page`.
    await context.route_web_socket("**/*", _block_web_socket)
    page = await context.new_page()
    harvest: _JsonHarvest | None = None
    if harvest_json:
        # The same-publisher filter on harvested JSON judges against the host this render is
        # allowed to be on, which is the host the landing check holds the browser to: one fact,
        # not two derivations of it that a reader has to prove agree.
        harvest = _JsonHarvest(page_host=pinned_host, playwright_error=playwright_error)
        page.on("response", harvest.on_response)
    try:
        return await _navigate_and_read_dom(
            page,
            url,
            playwright_error,
            memo_scope=memo_scope,
            goto_timeout_ms=goto_budget_ms,
            deadline_monotonic_s=deadline_monotonic_s,
            harvest=harvest,
            pinned_host=pinned_host,
        )
    finally:
        # A body read still pending here (the DOM read timed out, the DOM was over the ceiling, or
        # the main frame landed off the pinned host) would otherwise race the close below and raise
        # into its own swallowed path. The off-host raise is the one where a read is likeliest to
        # be in flight: on the pre-read refusal no `page.content()` round trip ran between the
        # settle and the raise to give the body tasks time to finish.
        if harvest is not None:
            harvest.cancel_pending()
        # Drain in-flight route handlers BEFORE teardown. Without this, a request still in
        # flight when we close (common after a goto timeout) fires _guard_route against the
        # closing context and raises TargetClosedError in a detached event listener — the
        # unhandled traceback storm seen 2026-07-25 that buries real fetch failures in the
        # logs. unroute_all(ignoreErrors) removes the handlers and silently swallows any still
        # mid-flight (Playwright's own remedy for this exact message). SSRF is unaffected: the
        # guard already ran for every request dialed while the page was live, and a request
        # racing teardown has no live target to exfiltrate through. Bounded and guarded like the
        # two closes that follow it, on the same shared budget, so neither a race nor a wedged
        # browser can skip them or hold the render past one teardown bound. It clears the HTTP
        # guard only: `unroute_all` touches `_routes` and never `_web_socket_routes`, and Playwright
        # has no `unroute_web_socket`, so the WebSocket handler stays registered through the
        # context close by design. That is safe because it is raise-free (see `_block_web_socket`)
        # and issues no protocol call of its own, so there is nothing for the close to race.
        await _teardown_step("unroute_all", context.unroute_all(behavior="ignoreErrors"), playwright_error, teardown)
