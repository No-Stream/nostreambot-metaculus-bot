"""The per-URL fetch-outcome vocabulary for the resolution-source provider.

One responsibility: what a single fetch attempt is allowed to SAY. That is the
``FetchStatus`` set, the one HTTP-status table both the Tier-1 page fetch and the
Tier-2 Datawrapper CDN hop read, the ``FetchResult`` record with its
success-implies-content invariant, the content-vacuity rule that hands a 200
carrying nothing a failure status instead, the one token an outcome is REPORTED
as (``ok`` or the verbatim status, shared by the diagnostics map and the run-log
marker), and the two pure reductions over a finished result list (the
provider-diagnostics source map and the compact failure line the section
renders).

Split out of ``research.resolution_source`` so the status contract — the thing
the section renderer, the diagnostics block and the run-log telemetry all key on
— can be read and tested without the network layer around it. Every status
string here is a telemetry token: changing one is a breaking change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from metaculus_bot.constants import RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS
from metaculus_bot.research.http_fetch import MAX_UNDECODABLE_CHAR_RATIO, DatawrapperChartRef
from metaculus_bot.research.image_leads import ImageLead

# `stale_data` has two producers, and only one of them earns the benign diagnostics token.
# The Tier-2 Datawrapper hop reached a dataset whose Last-Modified is outside the freshness
# window — older than the bound, missing, unparseable, or implausibly far in the FUTURE —
# and withheld it rather than serve it as live; that result carries `chart_id`, is a hop
# artifact rather than a cited source, and maps to `none` in the diagnostics source map.
# The Tier-1 Wayback rung read an archived capture of a CITED page and withheld it because
# the capture is past `RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS` or carries no datable stamp
# (`_wayback_snapshot_result`); that result has no `chart_id`, IS a lost cited source, and
# keeps `stale_data` as its loss token. The Wayback rung is not flag-gated, so from its
# merge a cited page's `status` can be this verdict where it used to be the direct
# `blocked` / `error` / `not_found`, which survives only as `from_status` on the sibling
# escalation line — see the accounting note on `FetchResult.status` below.
#
# `empty_body` is the 200-with-nothing-in-it case: a body that is empty or
# whitespace-only carries no information, so calling it `success` published an
# empty "primary grading evidence" section, suppressed the all-failed notice for
# every sibling URL, and reported `ok` to provider diagnostics. It is a FAILURE
# status for the same reason the HTML branch treats an empty extraction as
# `js_wall`: content is what makes a fetch a success.
#
# `no_resolving_content` is the 200 whose extracted text carries no content worth
# grading against — page chrome, and nothing else, or a document that discusses
# nothing the question asks about. Four ways a fetch earns it, told apart by
# `FetchResult.status_reason`:
#
#   `embed_shell` — the page's numbers exist but live inside a third-party data
#   embed (Infogram / Flourish / Tableau) our extractor cannot read, and what
#   trafilatura returned is the chrome around it. Shipped for qids 44554/44556,
#   whose tracker page rendered 2.9k chars of forecast background as "primary
#   grading evidence" with zero polling numbers in it.
#
#   `thin_page` — the extraction is under the same shell floor with no named embed
#   provider anywhere in the raw HTML. The 2026-09-01 round found five content-free
#   `success` renders and the embed-gated verdict reached none of them: q45088's
#   127-char SPA tab list (`Nationwide / Midwest / Northeast / …`) and q45215's
#   385 chars of Kazakh region names both published under the "primary grading
#   evidence" caption with nothing resolving in them. The floor is what the
#   verdict rests on either way, so gating it on a named provider was withholding
#   one shape of chrome and publishing the other.
#
#   `no_matching_passage` — a cited document we read END TO END whose BM25 passage
#   selection matched no query term. It is the one member of this status that is a
#   DOCUMENT rather than a page, which is why it is excluded from the paid rung's
#   population (`_url_context_rung_applies`): the bytes were never the problem, so a
#   model re-reading the same PDF buys nothing. See the reason table below for what
#   publishing it as `success` cost.
#
#   `not_addressed` — the PAID reader retrieved the page and its answer opened with the
#   prompt's `NOT_ADDRESSED` sentinel, the designed reply for a document that does not
#   discuss the ask. The read was paid for and the page WAS retrieved (so it is not
#   `ungrounded`), but the answer is a non-answer, and rendered under the url_context lead
#   it was prose standing in for an absent section: the paid rung's twin of
#   `no_matching_passage`.
#
# Distinct from `empty_body` (nothing was there at all) and from `js_wall` (the
# page needs JS to assemble ANY content, i.e. under the much lower JS-wall floor):
# here the page answered with prose-shaped chrome, which is why — like `blocked` /
# `js_wall` — it is a Tier-2 ESCALATION SEAM rather than a refusal.
#
# `unreadable_document` is a document we DID read the bytes of and could not turn
# into text: a scan with no text layer, an encrypted file, or a malformed one
# (`FetchResult.status_reason` says which). Deliberately NOT `unsupported_type`,
# which keeps meaning "a content type we do not read at all" — the two answer
# different questions, and a paid document read is only ever worth spending on
# this one.
# `ungrounded` is the PAID reader answering without having retrieved anything: Gemini's
# url_context tool reported zero successful retrievals, so whatever text came back is recall
# rather than a read of the page, and it is DISCARDED rather than rendered. Its own token because
# it is the one failure that cost money, and because it says something no other status does —
# the host answered a third-party fetcher's request with nothing while refusing ours. Mirrors the
# grounded-chunk floor `gemini_search` applies and the identical guard on gap-fill v2's
# `read_document`; the receipt for why is Q38195 (2026-07-19), where 30 search queries and 0
# grounding chunks produced a confident fabricated table with fake `[primary]` tags.
FetchStatus = Literal[
    "success",
    "throttled",
    "blocked",
    "not_found",
    "js_wall",
    "error",
    "unsupported_type",
    "ssrf_blocked",
    "stale_data",
    "empty_body",
    "no_resolving_content",
    "unreadable_document",
    "ungrounded",
]

# Which rule produced a status that has more than one rule behind it. A telemetry token
# like every `FetchStatus`: it rides the `RESOLUTION_SOURCE_FETCH` marker as `reason=`,
# which is what separates the embed-gated population (queryable since 2026-08) from the
# generalised thin-page one, so a later "how often does the floor withhold a page
# nothing else would have caught?" cut is a query rather than a re-derivation.
#
# `embed_shell` / `thin_page` / `no_matching_passage` / `not_addressed` belong to
# `no_resolving_content`; `no_text_layer` / `encrypted` / `malformed` to `unreadable_document`,
# where the split is what says whether a paid document read could ever help (only
# `no_text_layer` — the other two are bytes no reader gets text out of, and
# `no_matching_passage` is a document whose text we already hold).
#
# The document rung adds three. `no_matching_passage` is a document we DID read whose BM25
# selection matched no query term: the digest renders its header, its outline and the "no
# passage matched" sentence, and published as `success` that was byte-identical in the run log
# and the archive to a document that handed the forecasters the resolving paragraph — prose
# standing in for an absent section, on the one surface whose stated contract is that `success`
# means CONTENT. It is a `no_resolving_content` for that reason. `budget_skipped` and `parse_contention`
# belong to the `unsupported_type` a held-but-unparsed document earns, and say which rule
# declined: the question ran out of wall, or every parse slot was taken. Without them a
# skipped document is indistinguishable from a body that was never a document at all.
#
# The paid rung adds one. `not_addressed` is a page Gemini DID retrieve whose answer opened with
# the prompt's `NOT_ADDRESSED` sentinel, the designed reply for a document that does not discuss
# the ask; the same prose-for-an-absent-section shape as `no_matching_passage`, one rung over,
# and the token that keeps "we paid and the page has nothing on this" distinguishable from
# "we paid and Gemini retrieved nothing" (`ungrounded`).
#
# `metaculus_self_ref` belongs to `blocked`: a redirect hop WE refused because it landed on the
# question platform's own site (`resolution_source._vetted_hop_target`). The status stays
# `blocked` because that spelling is the contract; the reason is what keeps the paid rung off the
# URL (`_url_context_rung_applies`), the way `ssrf_blocked` is kept out of its trigger set, and
# what separates "the host refused us" from "we refused the host" on the fetch marker.
#
# `oversize_document` and `image_needs_reader` are the two bodies a caller declined to READ, and
# `undecodable_body` the `unsupported_type` of one that arrived as mojibake (docs/architecture.md).
FetchStatusReason = Literal[
    "embed_shell",
    "thin_page",
    "no_text_layer",
    "encrypted",
    "malformed",
    "no_matching_passage",
    "budget_skipped",
    "parse_contention",
    "not_addressed",
    "metaculus_self_ref",
    "oversize_document",
    "image_needs_reader",
    "undecodable_body",
]

# Why a RUNG ATTEMPT never ran, carried on `RungAttempt.skipped_reason` (empty when the rung
# fired). A closed vocabulary rather than bare strings so a typo is a type error instead of a
# permanently-zero count: `_rung_counts` indexes a `Counter` keyed on this Literal, so a
# misspelt reason cannot silently vanish from `details["counts"]`. Distinct from
# `FetchStatusReason`, which qualifies a result's STATUS — a skip produced no result, so its
# reason has nowhere else to live. `parse_contention` is the one member shared with
# `FetchStatusReason`: a held document declined for want of a parse slot both records the skip
# here and stamps the withheld result's `status_reason` there.
#
# `wall_budget` — the rung's own floor was below the remaining provider wall (`claim_rung_budget`),
#   or, for the impersonated retry, the wall ran out while the transport waited on a pre-dial
#   await (`impersonated_fetch.ImpersonateBudgetExhausted`); nothing was dialed either way.
# `wayback_cap` — the question spent its per-question snapshot attempts on earlier cited URLs.
# `url_context_cap` — the question spent its per-question PAID-read attempts on earlier cited URLs
#   (the paid rung's analogue of `wayback_cap`, bounding spend when the flag is on).
# `fast_path` — an expensive rung (the browser, the paid reader) declined because the question's
#   close-derived time budget put it on the fast path, a fact about the window not the 45 s wall.
# `no_api_key` — the paid rung is flag-on but `GOOGLE_API_KEY` is unset, a misconfiguration that
#   is otherwise byte-identical in the archive to the flag being off.
# `robots_disallowed` — the paid rung's `Google-Extended` pre-check found the host refuses that
#   token, so the read would be spend with a known-zero return: spend AVOIDED, not a page lost.
# `parse_contention` — a held document left unread because both parse slots were taken.
# `rendered_no_text` — the browser rung skipped because an earlier question in this run already
#   rendered the same URL to nothing (the memo doing its job, not a runner without Chromium).
# `renderer_unavailable` — the browser declined before rendering anything: Playwright missing or
#   broken, a host that will not pin to a public IP, or a browser error. Nothing was rendered, so
#   nothing about the page changed, which is why it is a skip reason and not a `status_reason`.
# `render_timeout` — the browser rung CUT OFF by the transport's own bound: Chromium launched and
#   navigated, and the DOM read outlived `RENDER_DOM_READ_TIMEOUT_MS` or the browser refused it
#   because the page kept navigating (or the URL was memoised earlier in the run: cut off that way,
#   or a navigation that failed onto Chromium's own error document, which the transport memoises
#   the same way so a dead URL is not relaunched for). A fact about the
#   page. The rung's own outer bound firing is NOT this token but `wall_budget`: it fires while the
#   render is still queued behind the launch gates, which says nothing about the page. Its own
#   token rather than `renderer_unavailable` because it says nothing about whether Chromium works
#   — the receipt is ogimet.com (2026-09-03), where a 76 s render was recorded as the renderer
#   being unavailable and latched the once-per-run warning, so a real outage later would have
#   logged nothing.
# `render_non_200` — the browser rendered the page and the main frame was answered with something
#   other than a 200 where the direct GET got one: the edge telling the browser apart, whose
#   interstitial markup (a 403 or 429 challenge) is not the page. A skip rather than a fired rung
#   because nothing about the page was read; its own token because "Chromium is refused where our
#   GET was not" is the rate a residual round asks for, and folded into the fired count it was
#   byte-identical to a render that ran and produced chrome again. The URL is deliberately NOT
#   memoised, because a 429 is retryable.
# `render_dom_too_large` — the browser rendered the page and its DOM is over `RENDERED_DOM_MAX_CHARS`
#   (`rendered_fetch.RenderDomOverCeiling`), so it was declined unread. A fact about the page,
#   kept out of `renderer_unavailable` whose comment points triage at the Playwright install.
# `render_off_host`: the browser's main frame landed on a host other than the one its DNS pin
#   covers (`rendered_fetch.RenderOffHost`), a server-side redirect hop the route guard never sees,
#   so the transport refused the DOM unread or discarded it unpublished. Read fail-shut, so it also
#   counts Chromium's own error document after a failed navigation, which makes it an upper bound
#   on hostile landings. A fact about the page, kept out of `renderer_unavailable` for the same
#   reason as the two above. A genuine http(s) landing on another host is not memoised, because
#   nothing about the page was published and the next question should be refused on its own
#   record; the error-document landing is memoised by the transport (its timed-out memo), so a
#   later question in the run records `render_timeout` for it without a launch.
# `impersonate_disabled`: the impersonated retry declined on its kill switch
#   (`RESOLUTION_SOURCE_IMPERSONATE_ENABLED`, on by default in code). A configuration rather than a
#   tuning signal, and counted because without it a run with the switch off is byte-identical in
#   the archive to one where no cited page ever earned the retry.
# `impersonate_unpinnable`: the impersonated retry declined because a hop's host would not resolve
#   to a vetted public address to pin the connection to (`impersonated_fetch.ImpersonateUnpinnable`).
#   The pin can fail on the FIRST hop, where nothing was dialed and nothing about the page changed,
#   or on a later redirect hop, where the earlier hops were dialed and wall was spent; the skip says
#   the pin failed on some hop. On the first hop the direct fetch resolved the same host through the
#   filtering resolver moments earlier, so a nonzero count there means DNS disagreed with the direct
#   fetch's own resolution (a flake, or a rebinding host that flipped) rather than a host refusing
#   us; a later-hop failure is a redirect target the direct fetch never resolved, so that reading
#   does not apply to it.
# `impersonate_host_refused`: the impersonated retry skipped because an earlier impersonated fetch
#   this run already answered with a block status from the same host, or dialed this exact URL into
#   one, by this fetcher or by gap-fill v2 (the memo is process-global and shared, keyed by the
#   answering host plus the dialed URL, so the earlier fetch may have been a v2 `fetch` or
#   `read_document` of a URL no question cited). The memo doing its job rather than a
#   failure, the same distinction `rendered_no_text` draws for the browser; folded into the fired
#   count it would read as a host refusing us twice.
# `rung_not_enabled` — the caller's policy does not carry this rung (docs/architecture.md, the knob table).
RungSkipReason = Literal[
    "rung_not_enabled",
    "wall_budget",
    "wayback_cap",
    "url_context_cap",
    "fast_path",
    "no_api_key",
    "robots_disallowed",
    "parse_contention",
    "rendered_no_text",
    "renderer_unavailable",
    "render_timeout",
    "render_non_200",
    "render_dom_too_large",
    "render_off_host",
    "impersonate_disabled",
    "impersonate_unpinnable",
    "impersonate_host_refused",
]

# Which rung of the escalation ladder produced a result. The vocabulary is pinned to the
# `route=` group of the `resolution_source_fetch` marker spec
# (`scripts/telemetry/markers.py`) — adding a token is fine, renaming one is a breaking
# telemetry change. `direct` is the plain fetch and is the only value that does NOT ride
# the marker, so every line the archive already holds stays byte-identical.
FetchRoute = Literal[
    "direct",
    "known_api",
    "meta_refresh",
    "impersonate",
    "pdf_local",
    "derived_api",
    "rendered",
    "wayback",
    "url_context",
]

LocalKind = Literal["archive", "workbook", "word", "image"]

# One forecaster-facing sentence per non-direct route, rendered under the "primary grading
# evidence" caveat for every route present in a question's snapshot. Keyed by `FetchRoute` and
# ITERATED, so the mapping is both the vocabulary check and the render order: cheapest and most
# transparent first, model-mediated last.
#
# `direct` is deliberately ABSENT rather than mapped to "". A route that adds nothing must be
# unrepresentable here, because that is what makes an all-direct question's section
# byte-identical to what it rendered before the ladder existed — the overwhelming majority of
# questions, and the thing a diff against the archive has to keep clean.
#
# Each sentence says the same two things in its own terms: where the bytes came from, and what
# the reader must not conclude from having them. The section they sit in is captioned primary
# grading evidence, so a rung that quietly substituted one artifact for another would overstate
# what was retrieved by exactly the amount that decides a forecast.
ROUTE_CAVEATS: dict[FetchRoute, str] = {
    "known_api": (
        "One or more sections below came from a deterministic public API for a cited URL rather "
        "than from the cited page's HTML; the API response is the source shown in the section."
    ),
    "meta_refresh": (
        "One or more sections below came from the page a cited URL redirected to through a "
        "`<meta refresh>` tag rather than an HTTP redirect; the content is the target page's, "
        "fetched normally."
    ),
    "pdf_local": (
        "One or more sections below are the query-relevant passages of a cited PDF, extracted "
        "locally — not the whole document. A figure that does not appear was not selected by that "
        "extraction, which is different from being absent from the document."
    ),
    "impersonate": (
        "One or more sections below were fetched on a retry that presented a different client "
        "fingerprint after the host refused ours; the content is the page's own."
    ),
    "rendered": (
        "One or more sections below were read from a page rendered in a headless browser, because "
        "a plain fetch of it returned no text; the content is the page's own after its scripts ran."
    ),
    "derived_api": (
        "One or more sections below are the JSON data feed a page loads its figures from rather "
        "than page text, because the page itself carried no readable content. The lead on each "
        "names the endpoint and says whether it was found on that page or on another page of the "
        "same host."
    ),
    "wayback": (
        "One or more sections below are ARCHIVED copies from the Wayback Machine rather than the "
        "live page, because the live page could not be fetched. Each states its capture date and "
        "its age in days: a quantity that has moved since that capture will not be in it."
    ),
    "url_context": (
        "One or more sections below are a model's reading of a page this bot could not fetch, not "
        "a copy of that page. Treat figures in them as REPORTED rather than retrieved, and weight "
        "them below sections quoted from bytes the host served."
    ),
}


# HTTP status -> FetchStatus for non-OK terminal responses — the ONE table both the
# Tier-1 page fetch (_resolution_status_outcome) and the Tier-2 Datawrapper CDN hop
# (_datawrapper_hop_status) read, so a future addition (451, 503, ...) cannot land on
# one surface only. 3xx is deliberately absent: Tier-1 vets redirects upstream via
# _resolution_redirect_outcome, and the CDN hop intentionally maps a 3xx to `error`.
_NON_OK_FETCH_STATUS: dict[int, FetchStatus] = {
    403: "blocked",
    406: "blocked",
    429: "blocked",
    404: "not_found",
    410: "not_found",
}

# The declared Content-Types every fetch path reads as a PDF: the ONE vocabulary the Tier-1 page
# fetch (the larger body cap and the local document read), the impersonated transport (its
# document re-dial) and gap-fill v2's plain rung (its `pdf_local` routing) all test against, kept
# here because all three already import this module and a third copy had drifted (v2 read
# `application/pdf` alone). Matched as substrings of the lower-cased header, so a charset suffix
# does not defeat it.
PDF_CONTENT_TYPES: tuple[str, ...] = ("application/pdf", "application/x-pdf")

# The `Server` header is free text and delimiter-hostile; bound it so one hostile value cannot
# blow the marker line's width, and lower-case it so `akamaighost` and `AkamaiGHost` bucket
# together.
_SERVER_HEADER_MAX_CHARS = 40


def http_failure_class(status: int | None) -> str | None:
    """The failure-class token for an HTTP status, or None for a 2xx/3xx or missing status.

    ``http_403`` is called out on its own because it is the egress-reputation refusal the
    escalation ladder exists for; ``http_4xx`` / ``http_5xx`` bucket the rest by side (client
    vs server). Riding the ``RESOLUTION_SOURCE_FETCH`` marker, so a query can ask "how often is a
    cited host giving us 403 specifically" without re-scraping expiring run logs.
    """
    if status is None or status < 400:
        return None
    if status == 403:
        return "http_403"
    return "http_4xx" if status < 500 else "http_5xx"


def server_header_token(server: str | None) -> str | None:
    """The ``Server`` header lower-cased and truncated for the marker, or None when absent.

    Internal whitespace is collapsed to ``_`` so the value stays ONE ``\\S+`` token on the
    space-delimited marker line (``Apache/2.4 (Ubuntu)`` would otherwise split mid-value); the
    marker cares about the CDN name (``cloudflare``, ``akamaighost``), which the collapse keeps.
    """
    if not server:
        return None
    return "_".join(server.strip().lower().split())[:_SERVER_HEADER_MAX_CHARS] or None


@dataclass
class RungAttempt:
    """One escalation-rung attempt on one URL: what triggered it, and what it cost.

    Carried on the result rather than logged where it happens, because the
    ``RESOLUTION_SOURCE_ESCALATION`` marker names the question and the question id only
    exists at the provider's per-question aggregation point (same reason the fetch
    marker is emitted there).

    ``from_status`` is the status the DIRECT route would have returned — the trigger
    population — so the marker answers "how often does this rung fire" without a join.
    ``url`` is the URL the rung was invoked ON, which for a meta-refresh hop is the stub
    rather than the page it led to: the stub is the URL the QUESTION cited and the one
    every earlier fetch record is filed under, so it is what a "which cited sources need
    this rung" cut has to key on.
    ``wall_s`` and ``outcome`` are THIS rung's own: what the attempt cost, and the status
    that stood once it was over — its rescue, its verdict (the Wayback withhold, the paid
    reader's ``ungrounded``), or the direct status it left standing when it declined. Both
    are None until the dispatcher closes the rung (``LadderContext.close_rungs``), because a
    rung is only over once its result is known a layer above where the attempt is created:
    the meta-refresh hop ends when the followed request comes back, the browser rung when
    its harvest fallback has been tried. A rung that measures something finer stamps itself
    and the closer leaves the stamp alone: the local PDF read stamps ``wall_s`` inside the
    parse gate so queueing for a slot is not billed to the parse, and the browser rung
    stamps ``outcome`` with the rendered DOM's own verdict before the harvested feed gets
    its turn, so a feed that rescues the page does not read as the render having done so.

    ``skipped_reason`` marks an attempt that produced no result for the page. Most never
    ran: no wall budget left, a per-question cap spent, the fast path, a memo hit, no
    browser, no key, a robots refusal. Four DID run a browser and were thrown away: ``render_timeout``
    is a render cut off before its DOM could be read, ``render_non_200`` one whose main frame
    the edge answered with something other than a 200, ``render_dom_too_large`` one whose DOM
    exceeded the ceiling, ``render_off_host`` one whose main frame landed off the DNS-pinned
    host, refused before its DOM was read or discarded unpublished; so a skip means "nothing
    came of this rung", not "no work was done". Skips are NOT escalation lines — the marker means "a rung
    fired and finished" — so they ride the provider's ``details["counts"]`` instead, where a
    zero renders nothing but survives into the archive; that is also why the most expensive
    render the ladder produces, a cut-off one, shows up in ``render_timeout_skips`` and never
    on the latency marker.
    """

    rung: FetchRoute
    from_status: FetchStatus
    url: str
    started_at: float
    wall_s: float | None = None
    outcome: FetchStatus | None = None
    skipped_reason: RungSkipReason | Literal[""] = ""

    def finish(self, now: float) -> None:
        """Stamp the elapsed wall-clock, unless the rung already measured its own."""
        if self.wall_s is None:
            self.wall_s = max(0.0, now - self.started_at)


@dataclass
class FetchResult:
    url: str
    # ACCOUNTING NOTE for anyone bucketing statuses by era: since the escalation ladder, a
    # `status` may be a RUNG's verdict rather than the direct fetch's outcome — the Wayback
    # rung's `stale_data` in place of the `blocked` / `error` / `not_found` that triggered it,
    # the paid reader's `ungrounded` or `no_resolving_content` / `not_addressed` in place of its
    # trigger. An era-bucketed `blocked` rate
    # taken off this field alone will therefore show a drop at the ladder's merge that is a
    # bookkeeping change, not hosts refusing us less. Take the direct outcome from
    # `from_status` on the `RESOLUTION_SOURCE_ESCALATION` line, or partition `status` by
    # `route` (`direct` rows are unchanged).
    status: FetchStatus
    text: str  # extracted + truncated; "" unless status == "success"
    http_status: int | None
    content_type: str | None
    # Charts seen in a fetched page's raw HTML (set on Tier-1 HTML results,
    # including js_wall pages — a JS-walled page still exposes its embeds).
    datawrapper_charts: list[DatawrapperChartRef] = field(default_factory=list)
    # Data-embed providers the page references that we have NO route to
    # (`unreadable_data_embed_providers`). Set on Tier-1 HTML results. Drives both
    # the `no_resolving_content` verdict and, on a page that DID carry prose, the
    # disclosure appended to its rendered text.
    unreadable_embeds: list[str] = field(default_factory=list)
    # Which rule produced the status, where the status alone is ambiguous. Set on
    # `no_resolving_content` (`embed_shell` / `thin_page` / `no_matching_passage` /
    # `not_addressed`), `unreadable_document` (`no_text_layer` / `encrypted` / `malformed`) and
    # the `unsupported_type` of a held-but-unparsed document (`budget_skipped` /
    # `parse_contention`); None everywhere else.
    status_reason: FetchStatusReason | None = None
    # Which rung of the ladder produced this result, and the per-rung attempts behind
    # it. `direct` plus an empty list is the plain fetch, which is the overwhelming
    # majority and renders no extra telemetry at all.
    route: FetchRoute = "direct"
    rung_attempts: list[RungAttempt] = field(default_factory=list)
    # The HTML extractor policy's decisions (`classify._extract_page_text`); False off
    # the HTML path. `chrome_metric_withheld`: the line-shape metric withheld an HTML extraction
    # of this URL somewhere on its ladder — an extraction that cleared the chrome floor on
    # navigation alone. That is a fact about the URL's ladder, not necessarily about this
    # result's text: on a chart-rescued page the chart block still publishes alone and this
    # stays True; on a page with no chart block the `thin_page` reason covers this and the
    # under-floor case alike; and on a page a later rung rescued (the rendered DOM, a derived
    # feed, the paid reader) the flag is CARRIED from the direct fetch onto the rescue
    # (`_fetch_one`), whose own extraction the metric never saw, so the withholds the ladder
    # then paid off are counted rather than lost. `precision_rescued`: the published text is the
    # `favor_precision` re-extraction, taken after the default one failed that metric (this
    # result's own; an archived copy carries the snapshot's). Both ride `details["counts"]`
    # (`chrome_metric_withholds`, `chrome_metric_withholds_rescued`,
    # `precision_fallback_rescues`), so no status or reason token moved.
    chrome_metric_withheld: bool = False
    precision_rescued: bool = False
    # Page results are empty unless `policy.collect_links`; deterministic known-API results may
    # carry their backend canonical sources even when the page policy leaves link collection off.
    links: list[str] = field(default_factory=list)
    # The caller's thin-content signal (`policy.thin_content_escalation_chars`); False for the fetcher.
    escalate_rendered: bool = False
    # Provenance for Tier-2 dataset results (None on ordinary page fetches).
    chart_id: str | None = None
    chart_title: str | None = None
    parent_url: str | None = None
    data_last_modified: str | None = None  # ISO-8601; None when the header was missing/unparseable
    # Failure diagnostics for the `RESOLUTION_SOURCE_FETCH` marker, so the archive can separate an
    # egress-reputation refusal from a host-side fault — the measurement the escalation ladder's
    # own case rests on (the archived Akamai 403s reproduce only from the GitHub runner). All None
    # on a success and on every path that never touched a response. `failure_class` is a small
    # token vocabulary: `http_403` / `http_4xx` / `http_5xx` off the response, or `tls` / `dns` /
    # `timeout` / `connection` / `decode` / `malformed_response` off the transport exception
    # (`resolution_source._network_failure_class`; the last is a response aiohttp's parser refused,
    # an undecodable Content-Encoding or an oversized header, which is neither a connection fault
    # nor a body we held and could not decode). `exc` is that exception's
    # class name. `server` is the `Server` response header, lower-cased and truncated — the
    # strongest tell of which CDN refused us. A rung verdict that REPLACES a failed direct fetch
    # on the marker line (the Wayback `stale_data` withhold, the paid reader's `ungrounded`)
    # keeps the direct fetch's diagnostics and `http_status`: they are facts about the cited host
    # that triggered the rung, and only a rung that served bytes reports its own status.
    failure_class: str | None = None
    exc: str | None = None
    server: str | None = None
    # True only on a freshly presented process-run cache hit. The cached payload itself is a
    # separate policy-neutral artifact and never rides this archive-facing result.
    cache_hit: bool = False
    # The shared 200-interstitial detector's evidence, retained so the loop can emit its
    # existing marker without carrying the refused body as fetch text.
    throttle_phrase: str | None = None
    throttle_chars: int | None = None
    # Page-digest counts; absent on ordinary reads so old marker lines stay byte-identical.
    passages_returned: int | None = None
    passages_grounded: int | None = None
    fallback_used: bool | None = None
    # Small routing metadata only. Parsed source contents and raster bytes stay in the bounded
    # process-run cache, never in the archive-facing result.
    local_kind: LocalKind | None = None
    navigation_only: bool = False
    local_read_refused: bool = False
    image_leads: tuple[ImageLead, ...] = ()

    def __post_init__(self) -> None:
        """Enforce the ``text`` invariant the field comment states.

        A `success` carrying blank text is the defect this guard exists for: the
        JSON/text/CSV branch used to ship an empty 200 body as `success`, which
        rendered an empty section under the "primary grading evidence" caveat,
        suppressed the all-failed notice, and told provider diagnostics `ok`.
        Every construction site now decides vacuity BEFORE it picks a status
        (:func:`vacuous_body_status`), so a blank success can only come from a
        future edit — and it should crash the provider (the orchestrator turns a
        provider exception into one `errored` provider result) rather than quietly
        publish a hole in the grading evidence.
        """
        if self.status == "success" and not self.text.strip():
            raise ValueError(f"FetchResult(status='success') with blank text for {self.url}")


# Delimiters a Datawrapper "Get the data" export uses. Checked on the HEADER
# line only: a dataset's first row names its columns, so a header with no
# delimiter at all is not the CSV this route is supposed to serve.
_CSV_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")


def looks_like_csv_rows(text: str) -> bool:
    """True when ``text`` is shaped like a dataset: a delimited header + ≥1 row.

    The precondition for asserting a dataset is LIVE. The Tier-2 lead stamps
    ``Dataset published <ts>`` off the ``Last-Modified`` header alone, so without
    this an empty or soft-404 CDN body renders under an authoritative freshness
    claim — the same shape as a venue's manufactured $0.50 price.

    Markup is rejected outright (a body whose first non-blank character is ``<``
    is an error page, not a dataset) because an HTML error page can easily carry
    a comma somewhere and pass the delimiter test.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    header = lines[0].lstrip()
    if header.startswith("<"):
        return False
    return any(delimiter in header for delimiter in _CSV_DELIMITERS)


def vacuous_body_status(text: str, undecodable_ratio: float, *, require_csv_rows: bool) -> FetchStatus | None:
    """The failure status a 200 body earns for carrying no usable content, else None.

    The one place "does this body carry information?" is decided, so every
    ``FetchResult(status="success")`` on a raw-body branch is predicated on
    content rather than on the status line. Three ways a 200 carries nothing:

    - it could not be DECODED (``undecodable_ratio`` above the bound) — mojibake
      like ``0�.�4�2�`` type-checks as text and rendered as grading evidence;
    - it is empty or whitespace-only;
    - (Tier-2 datasets only) it is not row-shaped, so nothing may claim it is the
      chart's live series.
    """
    if undecodable_ratio > MAX_UNDECODABLE_CHAR_RATIO:
        return "unsupported_type"
    if not text.strip():
        return "empty_body"
    if require_csv_rows and not looks_like_csv_rows(text):
        return "unsupported_type"
    return None


def fetch_outcome_token(result: FetchResult) -> str:
    """The telemetry token for one fetch: ``"ok"`` for a success, else the verbatim status.

    Shared by the provider-diagnostics source map below and the per-URL
    ``RESOLUTION_SOURCE_FETCH`` run-log marker, so the two can never disagree about
    what a fetch outcome is called. ``"ok"`` (not ``"success"``) because the
    diagnostics formatter recognizes that prefix as "this source contributed".
    """
    return "ok" if result.status == "success" else result.status


def _fetch_result_sources(results: list[FetchResult]) -> dict[str, str]:
    """Per-URL outcome map for provider diagnostics: ``{domain: "ok" | <FetchStatus>}``.

    A fetched URL normalizes to ``"ok"`` (the shared "contributed" token the
    diagnostics formatter recognizes); every other ``FetchStatus``
    (``blocked`` / ``js_wall`` / ``not_found`` / ``error`` / ``unsupported_type`` /
    ``ssrf_blocked`` / ``empty_body`` / ``no_resolving_content`` /
    ``unreadable_document`` / ``ungrounded``, and ``stale_data`` on a CITED page) is kept
    verbatim as the loss token so the reason survives into the ``lost=`` segment. The last
    two are rung verdicts standing in for the direct outcome: ``ungrounded`` is the paid
    reader retrieving nothing, and a chart_id-less ``stale_data`` is the Wayback rung
    withholding an over-age or undatable capture of a page our own address could not
    reach — a lost cited source either way, so neither takes the amnesty below. Duplicate
    domains are disambiguated with a ``#N`` suffix so no per-URL outcome is silently
    overwritten.

    Tier-2 dataset results are keyed ``datawrapper:<chart_id>`` — they are hop
    artifacts, not cited sources, and every dataset URL shares one CDN netloc, so
    domain keys would collapse them into ``static.dwcdn.net#N`` noise. Their
    ``stale_data`` verdict maps to the benign ``"none"``: that is the freshness
    guard REFUSING to serve months-old data as live, i.e. the feature working as
    designed, and reporting it in ``lost=`` would dress a by-design withhold as a
    lost cited source. The ``chart_id`` is what tells the two ``stale_data`` producers
    apart, which is why the amnesty is keyed on it and not on the status. A genuinely
    failed hop (``error``/``blocked``/``not_found``/``empty_body``/``unsupported_type``)
    keeps its verbatim loss token — that is real signal about the CDN, and it is the reason
    the content check runs BEFORE the freshness guard: an empty CDN body must not borrow
    ``stale_data``'s by-design amnesty.
    """
    sources: dict[str, str] = {}
    for r in results:
        if r.chart_id is not None:
            key = f"datawrapper:{r.chart_id}"
        else:
            try:
                key = urlparse(r.url).netloc or r.url
            except ValueError:
                key = r.url
        if key in sources:
            n = 2
            while f"{key}#{n}" in sources:
                n += 1
            key = f"{key}#{n}"
        if r.chart_id is not None and r.status == "stale_data":
            sources[key] = "none"
        else:
            sources[key] = fetch_outcome_token(r)
    return sources


def _forecaster_facing_status(r: FetchResult) -> str:
    """The outcome token a forecaster reads for one failed CITED page.

    The verbatim ``status`` for every outcome but two. Since the ladder, a cited page's status
    can be a RUNG's verdict about an artifact the forecaster never sees: the Wayback rung's
    ``stale_data`` for an over-age or undatable capture it withheld, and the paid reader's
    ``ungrounded`` for a read that retrieved nothing. Rendered bare, ``www.bls.gov: stale_data``
    asserts a false thing about the LIVE page, one a forecaster on a "will X publish by date"
    question can act on, when the direct outcome (``blocked``) is the fact about that page. So
    those two render the DIRECT status, taken off the ``from_status`` the verdict's own rung
    attempt recorded (the ladder asks every rung about the direct outcome, so the attempt for
    that rung always carries it), glossed with what the rung found. The status tokens themselves
    are telemetry and do not move: this is the forecaster-facing line only, never the marker or
    the diagnostics map.

    Only cited pages reach here (the caller partitions datasets into their own withheld note),
    which is what keeps the Datawrapper hop's ``stale_data`` out of this branch.
    """
    if r.status not in ("stale_data", "ungrounded"):
        return r.status
    verdict_rung: FetchRoute = "wayback" if r.status == "stale_data" else "url_context"
    from_status = next(a.from_status for a in r.rung_attempts if a.rung == verdict_rung and not a.skipped_reason)
    if r.status == "stale_data":
        return (
            f"{from_status} (live page could not be fetched; the newest archived copy is older than "
            f"{RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS:.0f} days or undatable)"
        )
    return f"{from_status} (a model-mediated read retrieved nothing)"


def _render_fetch_failures(failures: list[FetchResult]) -> str:
    """Render failed fetches as a compact ``"domain: status, domain: status"`` list.

    Every pre-ladder status renders byte-identically; the two rung verdicts a cited page can
    carry are glossed by :func:`_forecaster_facing_status`.
    """
    parts: list[str] = []
    for r in failures:
        try:
            domain = urlparse(r.url).netloc or r.url
        except ValueError:
            domain = r.url
        parts.append(f"{domain}: {_forecaster_facing_status(r)}")
    return ", ".join(parts)
