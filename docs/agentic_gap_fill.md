# Agentic gap-fill (gap-fill v2)

Gap-fill v2 is the bounded agentic research loop that runs after the main
research providers have built a question's briefing. It gives one driver LLM a
small set of research tools and a strict time/tool budget, lets it decide what
the briefing is missing or getting wrong, and returns a citation-only findings
artifact that gets appended to the bundle every forecaster reads.

It is the newest and largest research subsystem on this branch. It runs
alongside the older v1 gap-fill pass (`research/targeted.py`), not instead of
it. Both are on in production as of 2026-07-21 (v2 was authored 2026-07-17 and reached
`main` in merge `b4e9df0` four days later), and both feed the same bundle.

The code lives under `metaculus_bot/research/agentic/`, with a thin seam at
`metaculus_bot/research/agentic_gap_fill.py` that wires it into the research
orchestrator.

## Why it exists

The always-on providers (AskNews, native search, Gemini, and so on) build a
broad briefing, but they do not read it back and ask "what's missing, and is any
of this actually wrong?" A step-zero audit found that most of the bot's worst
misses were not gaps in coverage. They were the panel leaning on a briefing
claim that was stale, misread, or hallucinated. So v2's highest-value job is
verification, not just filling holes: it re-reads the briefing through the lens
of the panel's own forecasting template, checks the two or three claims the
forecast leans on hardest against primary sources, and flags any that don't hold
up.

## How it works, end to end

1. The orchestrator finishes the first research pass and has a briefing bundle.
2. It kicks off v1 and v2 gap-fill concurrently (one `asyncio.gather`). v2 sees
   the bundle without v1's addendum, and v2's section is appended after v1's.
3. The v2 seam builds three prompts, the four tools, and a `LoopConfig`, then
   runs the loop.
4. The driver LLM does a private dry run of the forecast to decide what to
   research, then works through search/fetch/read tools, banking findings as it
   goes.
5. When it concludes (or the budget runs out), the loop renders the banked
   findings into a markdown section and returns it.
6. The orchestrator appends that section to the bundle under
   `## Agentic Research Findings`.

The whole thing is optional enrichment. If any of it fails, the forecast
proceeds on the first-pass research alone. That soft-fail contract is described
in its own section below because it is load-bearing.

## The driver: dry run, then research

The driver is briefed as a research analyst supporting a forecasting panel, not
as a forecaster. Its system prompt (`driver_prompt.py` `build_system_prompt`)
walks it through three steps:

**Step 1, private dry run.** The driver reads the question, its resolution
criteria and fine print, and the current briefing, then privately walks through
how it would forecast the question using the panel's own template. As it does,
it notes several kinds of target:

- **FILL** targets: facts the reasoning needed that the briefing doesn't
  contain, or contains only in a secondhand or stale form.
- **VERIFY** targets: the two or three claims the reasoning leaned on hardest.
  These get checked against primary sources, because a wrong load-bearing claim
  poisons every panelist.
- **RESOLUTION** targets: if the resolution criteria name a specific source,
  metric, or clause, the driver quotes the operative language and current value
  from the authoritative source itself, not from news coverage of it.
- **TIMING** is folded into every target: the question only resolves on events
  inside its window, so the driver pins the exact date of each candidate trigger
  and flags any event that pre-dates the question's open date.
- **BASE-RATE** targets: if the dry run leaned on a reference class, the driver
  decides whether to look up the real denominator and count. It researches
  conditional or niche or uncertain rates and skips common-knowledge ones.

The user brief (`build_user_brief`) is what gets these steps grounded in the
real question. It carries the question text, resolution criteria, fine print,
forecasting window, the panel's real per-question-type template skeleton, and
the full briefing bundle. The template skeleton is the actual `binary_prompt` /
`multiple_choice_prompt` / `numeric_prompt` output with only the research slot
replaced by a placeholder. Everything else in it (units, bounds, open/closed
bound notes, options) is the question's real values, because the dry run and the
later ghost forecast are only meaningful against the real template. The
placeholder itself carries the prediction-market section header, because the
panel's market-reading clause renders only when the research carries that header
and prod emits it on every question (present on 59 of the 60 newest archived
artifact records; the provider omits it only on an empty pool, a soft-fail, a
flag-off run or benchmarking); without it the skeleton would show the driver a
template the panel never actually sees. It deliberately does not carry the
time-series anchor header, whose incidence runs the other way (5 of 322 artifact
records, 0 of the newest 30): hardcoding that one would manufacture a divergence
on about 98% of numeric prompts to close one on about 2%, and the numeric template
already names the anchor section in its resolution-metric bullets.

**Step 2, research.** The driver pursues its targets with the tools, follows
leads (a fetched page that references a more authoritative PDF is usually worth
chasing over a fresh search), batches independent calls in parallel, and records
findings as it confirms them rather than holding everything for the end.

**Step 3, conclude.** It calls `conclude` when every target is resolved or
confidently unreachable, or when the budget line tells it to. If the briefing
already covers everything, it concludes right after spot-checking the single
most load-bearing claim. Most questions need little; a few need a lot.

The user brief is frozen before the loop starts and the loop only ever appends
to the message list, so the brief acts as a stable prompt-cache prefix.

## Search and document tools

Built by `tools.py` `build_gap_fill_tools`. Each returns a `ToolOutcome`
(content markdown, links, a `method` tag, a status, a truncation flag). Each
also carries its own per-call timeout, set as `timeout_s` on its `ToolSpec` in
`build_gap_fill_tools` and scaled to how heavy the tool is; the loop enforces it
and turns a breach into an error `ToolOutcome` rather than a crash.

The serialized tool reply stays within `LoopConfig.max_result_chars` (8,000 by
default). That budget includes the tool metadata, source labels, image-lead
metadata, budget line, and any continuation or truncation marker; those headers
do not get added on top of the text allowance. Dispatch reserves up to 512
characters for the remaining-budget line
(`GAP_FILL_V2_TOOL_BUDGET_LINE_RESERVE_CHARS`) before truncating the tool body,
then clips the line to the room available.

- **`search_news`**: recent and historical news via AskNews, using the same
  rate gate and concurrency semaphore as the primary AskNews provider. Returns a
  digest of matching articles with dates and URLs.
- **`search_web`**: semantic web search via Exa (direct SDK, not OpenRouter).
  Meant for official documents, datasets, reports, and primary sources the
  driver believes exist. Returns results with URLs and excerpts, which the
  driver is told to follow up with `fetch` since excerpts rarely verify a claim
  on their own. This is the lightest search or document tool, and the one on the
  tightest timeout.
- **`fetch`**: fetch a URL and return its main content as markdown plus
  outbound links. This is an auto-escalating ladder (detailed below), so the
  driver is told not to avoid a URL because of its format. Supports windowed
  reads with `start_char=N`; continuations are served from cache. A PDF is read
  here too, in full text, and paginates the same way. ZIP, workbook, and Word
  sources accept exact-name `member` and `sheet` selectors; their inventory is
  navigation only until content is selected. An HTML result can also include
  bounded image-lead metadata. Call `view_image` to deliver selected pixels on
  the next driver turn. Its `ToolSpec` timeout leaves headroom above its own
  fetch budget for document escalation on the last rung.
- **`read_document`**: ask a specific question of a specific document, and get
  back the passages of it that bear on the ask. Acquisition-first: it runs the
  free rungs (this run's cache, then plain HTTP, the impersonated retry of a 403,
  and headless Chromium) and answers from the page's own text with a deterministic
  BM25 passage digest
  (`method=digest_local`). For a remote page or PDF, Gemini reads the URL through the
  `url_context` tool on the native `google-genai` SDK (`method=document`) only when the
  local ladder has no usable content or reaches the one refused shape described below. Local
  ZIP, spreadsheet, and Word reads use the local parser and the same BM25 digest,
  optionally narrowed by exact `member` or `sheet` selectors; parser refusals do
  not enter the paid fallback. So it is for targeted extraction from long or complex
  documents and as a fallback when `fetch` was blocked, and the paid half of it
  is now reserved for hosts our own client cannot read: measured 2026-09-03,
  two of 47 archived fetch failures. Requires a precise `ask`: it is what selects
  the passages. Its deadlines nest: the `ToolSpec` timeout sits above a total
  budget the two rungs share (`_READ_DOCUMENT_TOTAL_BUDGET_S`), which caps local
  acquisition at `_LOCAL_DOCUMENT_BUDGET_S` and hands the reader whatever is
  left, itself bounded by Gemini's own read timeout
  (`_READ_DOCUMENT_TIMEOUT_S`), which sits above the HTTP timeout handed to the
  SDK, so the innermost one fires first and the driver gets a clean error
  outcome instead of a tool-level kill. A question-platform URL (metaculus.com or
  `competitions.mantic.com`) is refused before any rung runs, with the same
  `blocked` outcome `fetch` gives it: the paid reader dials from Google's address,
  so it is the one rung the plain fetch's self-reference refusal could not
  otherwise reach, and on Mantic the page it would read carries the other bots'
  forecasts.
- **`view_image`**: fetch a selected image URL and normalize supported pixels to
  a bounded PNG. The loop sends those pixels to the same driver on its next
  ordinary request, which incurs image input tokens but makes no separate
  image-reader call. Crop and provenance rules are detailed below.

### The three known-API tools (built and wired)

Three more tools are built in `research/known_api/tools.py` (`build_known_api_tools`) and are
appended to the driver's list by `build_gap_fill_tools`. They answer a
FRED/Yahoo/Kalshi read deterministically, no LLM and no paid key, and return a date window on
demand that the URL forms cannot express. The registry also translates those URL shapes at the
ladder's rung 0, so the driver can still just `fetch` the URL and get the API answer. Detail:
`docs/research.md` "Known-API registry".

- **`fred_series`**: one FRED (Federal Reserve Economic Data) series over a date window, or a
  free-text search over FRED's catalogue. `series_id` with optional `start`/`end` ISO dates and
  `first_release`, or `search`.
- **`yahoo_history`**: one Yahoo Finance symbol's price history over a date window. `ticker` with
  optional `start`/`end` and `column` in Close/High/Low/Open.
- **`market_snapshot`**: a prediction-market snapshot for one `venue` (kalshi, polymarket,
  manifold, predictit) and `market` given as a venue id/ticker or free text.

Each handler adapts its backend's result to a `ToolOutcome` with method `known_api`; the market
handler binds the per-question session and per-question Kalshi detail-GET budget. Optional
catalogue and PredictIt-dump arguments remain available for callers that already have those
resources; the gap-fill binding does not create a new catalogue fetch. The same session and
detail budget are bound to the fetch-ladder rung-0 callback for that question. The bounds (window
cap, per-call timeouts, five market rows, four Kalshi GETs) live with the backends, documented in
`docs/research.md` "Known-API registry".

### The fetch ladder

`fetch` runs the SHARED fetch ladder, `metaculus_bot/research/fetch_ladder/`, through one entry
point: `ladder.fetch_url(url, policy=..., ctx=...)`. The rung order, the transports, the SSRF guard,
the classification of a body and the per-rung wall bounds are all documented once, in
`docs/architecture.md` "The shared fetch ladder"; this section is only what is THIS caller's.

**What the loop contributes to the ladder.** Three presets in `fetch_ladder/policy.py`:
`GAP_FILL_FETCH_POLICY` for the `fetch` tool (a 90 s wall, matching the ToolSpec ceiling; the
impersonated retry, the browser and the archive), `GAP_FILL_DOCUMENT_POLICY` for `read_document`'s
free acquisition ladder (25 s, and no archive rung, because that ladder sits immediately in front of
the paid reader and an archived copy is not what a document read was asked for) and
`GAP_FILL_DIRECT_POLICY` for the robots.txt pre-check (one direct fetch, no rung). All three carry
`GAP_FILL_VERDICT`, which is what makes the driver's reading of a body different from the fetcher's:
the bytes decide the branch rather than the Content-Type header, any non-empty extraction is content
rather than having to clear a 400-character chrome floor, a success under
`GAP_FILL_V2_MIN_CONTENT_CHARS` with no chart block earns the browser, and a document is served as
its whole text with its parse held for the run rather than as a query-ranked digest.

**Rungs this caller does not run.** The derived-API REUSE rung (a feed an earlier render on the host
recorded) and the paid `url_context` rung are absent from every gap-fill preset, so the dispatcher
records a `rung_not_enabled` skip and moves on. The HARVEST half of the derived feed — the JSON a
fruitless render already captured, served with `derived_api_lead` for provenance and
`method=derived_api` — is inside the browser rung and does run.

**The per-question context.** `agentic_gap_fill.run_gap_fill_v2` builds ONE `LadderContext` per
question (`tools.question_ladder_context`) and captures it into the two handlers that fetch. It
carries the per-question rung budget, which is what caps a question at two archive snapshots and two
paid reads however many URLs the driver picks — a cap this loop did not have before it moved onto the
shared ladder. Everything per call (the ask, the wall origin, the rung list) is derived off it in
`tools._per_call_ctx`, because one tool call is one wall.

**What stays in `tools.py`.** The `start_char` window presentation over the ladder's complete
process-run artifact; the question-platform refusal, which runs before the ladder because it is this caller's own
policy and must refuse before anything is dialed (the resolution-source fetcher drops those URLs when
it selects them); the throttle outcome and marker presentation after the shared ladder has classified
the body; the auto-escalation to `read_document` for an unreadable document; and the two shared
fetch markers, emitted after each tool call with `question=None`.

### The shared fetch ladder, and the adapter over it

The ladder speaks the resolution-source vocabulary — thirteen `FetchStatus` values, eight `FetchRoute`
values, a `status_reason` where a status has more than one rule behind it — and the driver reads seven
statuses plus a `method` the verification-tier map keys on. `agentic/ladder_adapter.py` is the ONE
place the two meet, so a status the ladder gains cannot silently become an `ok` the loop stamps
`fetched`. The table shows those mappings and the local-source selector outcomes that `tools.py`
applies after the ladder returns:

| `FetchStatus` (and reason) | loop `status` | loop `method` | what the driver is told |
|---|---|---|---|
| `success` (ordinary text or document) | `ok` | `cache`, or from the `route`: `direct`/`meta_refresh` → `plain`, `pdf_local`, `impersonate`, `derived_api`, `rendered`, `wayback` | the text the ladder read |
| `success` + selected local source | `ok` | `local` | the selected parsed text |
| `success` + `navigation_only` | `ok` | `local_navigation` | a member/sheet inventory or internal image-acquisition notice |
| `throttled` | `throttled` | `throttled` | blank text; the original rung is retained in the throttle marker |
| `js_wall`, `empty_body`, `no_resolving_content` | `empty` | `plain` | "Plain fetch returned no extractable text." |
| `unsupported_type` + `undecodable_body` | `empty` | `plain` | "Plain fetch could not decode the body as text." |
| `unsupported_type` + `local_read_refused` | `error` | `plain` | terminal local-parser refusal; no paid fallback |
| other `unsupported_type` | `error` | `plain` | "Unsupported content type: X" |
| `unreadable_document` (scanned, encrypted, or malformed PDF) | `ok` | `document_needed` | the use-`read_document` placeholder |
| `blocked` with a host status | `blocked` | `plain` | "Fetch blocked with HTTP N." |
| `blocked` with none (a refusal we made) | `blocked` | `plain` | the platform-block message, hosts named |
| `ssrf_blocked` | `blocked` | `plain` | non-public URL, or non-public redirect target |
| `not_found` | `error` | `plain` | "Fetch failed with HTTP N." |
| `error` + `oversize_document` | `error` | `oversize_document` | the too-large-to-read message |
| `error` (a 5xx, a malformed redirect, an over-cap body, a transport failure, a spent redirect budget) | `error` | `plain` | one clause per shape |

For the current gap-fill image path, the shared ladder retains the raster and returns `success` with
`navigation_only`; the adapter maps that intermediate result to `local_navigation`, which earns
neither finding provenance nor a verification tier. The public `fetch` and `read_document` handlers
intercept `local_kind=image` before returning a result and deliver pixels through the same viewer
path as `view_image`. The internal acquisition notice is not returned to the driver, and the
`image_needs_reader` adapter case is not used for supported gap-fill image responses.

Five facts of the tool contract the adapter exists to preserve, each pinned by a test: status alone
does not grant a verification tier; `method` must also map through `provenance._METHOD_TO_TIER`, and
`local_navigation` is excluded from finding provenance; `http_status` is set only by a host's
response, never by a refusal the ladder made itself; `escalate_rendered` is the caller's thin-content
signal with a chart block pinning it off; and links resolve against the DOCUMENT url, after a
client-side redirect.

One capability the shared browser rung does not carry: the loop's old rendered rung escalated a
render whose Content-Type was a document to `read_document`, and the shared rung classifies every
rendered DOM as HTML instead. A page whose client-side redirect lands on a PDF now reads as a
JavaScript wall rather than escalating. `FUTURE.md` carries the entry.

### Local archives, spreadsheets, and Word documents

The shared classifier recognizes ZIP, TSV, valid UTF-8 JSON served as
`application/octet-stream`, Excel workbooks, and `.docx` files. Generic ZIPs
are opened locally without extracting files. Supported readable members are
plain text (`.txt`, `.md`, `.json`, `.xml`, `.log`), CSV/TSV, `.xlsx`/`.xlsm`,
`.xls`, and `.docx`; nested archives and legacy `.doc` are not read. OOXML
packages are identified before the same ZIP bytes can be mistaken for an
ordinary archive.

The complete bounded parse lives in the shared fetch cache. It is not shaped
around the first caller's query, member, sheet, or output window: each later
presentation applies its own selectors, digest, and pagination. In `fetch`, an
archive with no `member` returns an exact-name member inventory. Selecting a
workbook member without a `sheet` returns its sheet inventory. These inventories
are navigation only, never evidence for a finding. Select a member or sheet to
read the labeled text; long text uses the existing `start_char` continuation
contract. `read_document` with no selectors runs its query digest across all
readable sections; `member` and `sheet` narrow that search first.

Spreadsheet output uses values saved in the workbook. `.xlsx` and `.xlsm`
formula cells show the cached result when one exists; formulas are not evaluated
locally, and a missing cached result is called out. `.xls` uses saved values and
includes the same notice. For `.xls`, active percent format tokens display the
stored value as a percentage; quoted or escaped literal percent signs preserve
the stored number and include a format notice. Word extraction preserves body
paragraph/table order, including nested tables, then reads each distinct
non-empty default, first-page, and even-page header and footer. Their labels
identify the variant and section, such as `first-page header section 1`.
Directories appear in ZIP inventories as unreadable entries because they have
no file content. Images, OCR, embedded objects, tracked changes, text boxes,
footnotes, and unsupported Word parts are omitted with a disclosure in the
returned text.

The page response remains capped at 5 MiB. Expanded source content is capped at
20 MiB, ZIP files at 128 entries, workbooks at 32 sheets and 250,000 cells, and
all extracted text at 2 million characters. The process-run cache keeps at most
64 MiB across parsed local-source text and retained image bodies. Exceeding a
limit refuses the parse instead of returning partial content. A local parser
refusal—including an unsupported or malformed container, encryption, or an
exceeded cap—is terminal and never falls through to Gemini's paid `url_context`
reader. Unsupported archive members are identified in the inventory and left
unparsed. These limits live in `constants.py` and are spelled out in
`docs/constants.md`.

Run `make test_fetch_formats` for offline coverage of local parsing, image
normalization, and image delivery. Its fake-driver checks verify that image
bytes are serialized into the next driver request; they do not measure live
model visual accuracy. For a network-backed check of one source, run
`uv run python -m scripts.probes.fetch_diagnostic --source-url <URL> --query <query>`.
That maintained diagnostic uses the production acquisition ladder and BM25
passage selection, with the paid `url_context` rung disabled.

The classifier parses supported formats locally without adding a ladder rung or
making a paid call. Selected content is returned with `method=local` and receives
the fetched verification tier. An inventory is returned with
`method=local_navigation`, which does not earn a verification tier until the
driver selects and reads content.

### Image leads and same-driver image reading

HTML fetches may expose up to three likely image URLs from figure captions or
descriptive `alt` text. The adjacent metadata is untrusted and explicitly says
that pixels have not been read. The driver calls `view_image(url, crop=None)` to
inspect one. When the URL itself is an image, `fetch(url)` and
`read_document(url, ask)` immediately deliver its pixels through the same view
path as `view_image`; these calls do not return navigation instructions and do
not fall through to the paid document reader. All three tools share one budget
of four distinct normalized PNG hashes per question, including crops; duplicate
pixels reuse their image ID. `view_image` reuses acquired bytes or obtains them
through the shared ladder, then normalizes them to a metadata-free PNG. Static
PNG, JPEG, WebP, BMP, and GIF are supported; SVG and all animated formats,
including APNG, are rejected. Source images are limited to 25 megapixels. A
delivered image is at most 2,048 pixels on its long edge, 2 megapixels total,
and 2 MiB. Crop coordinates are `[left, top, right, bottom]` in the displayed
image orientation after EXIF rotation.

The loop waits for all tool replies in a batch, appends one byte-free message
with the newly returned image IDs, then includes the pixels in the next normal
request to the same driver model. This uses input tokens on that driver request
and adds no separate image-reader LLM call. Transcripts store references, not
base64. The plain ghost and v1 ghost keep those references in their shared
context and resolve them through the same request wrapper. Visual findings
carry `evidence_kind=image`, the delivered `image_id`,
the exact registered source or final image URL, and a `visual_observation`.
Numeric readings must be identified as transcribed or estimated. The loop
accepts an image ID only after delivery on a preceding driver turn and checks
that the cited URL belongs to that image. That proves which pixels were shown,
not that the visual interpretation is correct.

Normalized PNG bytes are persisted separately as
`research_outputs/media/<sha256>.png`. Each byte-free entry in the artifact's
`images` list has one `asset_path` per normalized PNG hash; `image_id` and
`png_sha256` are that same hash. Entries carry `byte_count`,
`representative_metadata`, and exact-deduplicated `observations`.
Each observation retains source hashes and URLs, parent-page URLs, original and
normalized dimensions, and crop coordinates. `source_urls` and
`parent_page_urls` are stored separately as unions. Both artifact harvest paths
validate each manifest hash before copying the sidecar into the canonical
research archive. Existing text-only records continue to work without a media
manifest.

### The shared throttle check

A host that is throttling us answers HTTP 200 with a short interstitial in place of the page it was
asked for, so every status check on the ladder passes and the driver reads the refusal as the page's
content. Receipt: question 45191 (2026-08-10), where three parallel fetches of ogimet.com daily
summaries tripped that host's spacing rule and two came back as a 304-character body reading
"gsynext: Limit for old data queries exceeded. Permitted a query per 20 seconds per IP" under
`status="ok"` — which was then cached and replayed on the driver's own retry, so the exact-date
reference class it published came to 4 years instead of 6 and the forecast under-committed to the
winner it had already named.

The neutral `fetch_ladder.throttle.matched_throttle_phrase` predicate runs before either caller's
verdict presents an HTML or raw-text body. The size half is why it
is safe: the phrases alone would demote a real page that merely discusses rate limits, while a size
floor alone would demote every legitimately short source (a one-line official statement), which the
ladder deliberately keeps as `ok`. Bare "slow down" is left out on purpose, being ordinary English
where the rest are throttle idiom, and missing a throttle only preserves today's behaviour whereas a
false positive discards a page we really did read. `FETCH_THROTTLE_PAGE_MAX_CHARS` is 1,200 against
the receipt's 304-character body (303 stripped, which is what the cap sees). PDF bodies bypass this
detector. A direct throttle is terminal, so it does not launch the browser, consult an offsite rung,
or write a cache entry. A rendered throttle keeps `route=rendered` and the rendered attempt's
`outcome=throttled`. The FetchResult carries blank text plus only the matching phrase and stripped
character count needed for the existing marker. An interstitial is deliberately NOT cached, which
is the half question 45191 turned on: the body was cached under
`method="rendered"` and served straight back on the driver's retry, so that retry could not have
succeeded however many slots it spent.

### Why the question platforms' own hosts are refused

`fetch_outcomes._fetch_plain_url_block` refuses metaculus.com and `competitions.mantic.com` from our
runner IP, on the caller-supplied URL and again on every redirect hop. A Metaculus question page is a
JavaScript SPA whose near-empty plain fetch would escalate to headless Chromium, whose route guard
then permits the SPA's own XHR fan-out to the Metaculus API, all from our IP and on the same host the
critical API calls use. Refusing before the ladder runs kills both our-IP rungs. `read_document` runs
the same check first, before its free ladder and before the paid Gemini read, which dials from
Google's address and is the one rung the our-IP refusal could not otherwise reach: it would read a
platform page the driver handed it, other bots' forecasts included on a Mantic question page.
`FETCH_DESCRIPTION` also tells the driver not to fetch those hosts, and the resolution-source
pre-filter never cites one, so the driver only ever meets a platform URL it picked out of a search
result. The resolution-source ladder's own paid rung is closed to a self-reference the same way
(`rungs._url_context_rung_applies`).

### The free digest, and the one shape it refuses

`read_document` answers from text the ladder already holds with a deterministic BM25
passage digest. It runs in a worker thread rather than on the event loop:
`select_passages` tokenises every window of the whole document and holds a counter per
window, which measured a 1,365 ms contiguous stall for six concurrent 400-page digests,
inside a research phase whose wall discards work that already succeeded.

The digest is refused (and the paid reader runs instead) for exactly one shape: held text
under `GAP_FILL_V2_MIN_CONTENT_CHARS`, with no PDF parse behind it, whose digest selected
NO passage. That is a JavaScript shell whose browser rescue already failed, and digesting
its navigation chrome stamped an unread page `fetched` (the one tier that supersedes the
briefing) while the tool description tells the driver a zero-passage digest means the
document does not discuss the ask. All three conditions are load-bearing: a thin-but-real
short page that matches the ask is still served free, a held parse is a real local read of
something a browser cannot help with, and a matching passage is the evidence that the text
is the page rather than its frame.

The paid rung's deadline arithmetic is fixed by design, and it can overrun. The wait is
`min(_READ_DOCUMENT_TIMEOUT_S, _READ_DOCUMENT_TOTAL_BUDGET_S − acquisition elapsed)`: 60 s
when acquisition failed fast, 40 s at the `_LOCAL_DOCUMENT_BUDGET_S` cap. The reader's own
in-thread ceiling is a FIXED 55 s (two attempts plus backoff, `tool_backends.py`), so past
about 10 s of acquisition the wait is the shorter of the two, and `wait_for` cannot cancel a
`to_thread` worker: a worker can outlive the wait by up to 15 s and finish a billed call
whose answer is discarded. What it cannot do is start a NEW billed request after the wait
fires: the last attempt begins by 28.5 s in, inside the 40 s floor. Sizing the attempts off
the variable wait instead would cut one attempt to 19 s on the handover path and fail reads
that succeed today, so the timeout values are unchanged and the overrun is documented rather
than traded away.

### The robots pre-check on the paid read

Before the paid `url_context` read (and only there; the free rungs are unaffected),
`read_document` fetches `<scheme>://<host>/robots.txt` once per host through the shared direct
fetch in the ladder (`fetch_ladder.direct_fetch._fetch_direct`, under
`robots_policy.ROBOTS_FETCH_TIMEOUT_S`, the bound the Tier-1 resolution-source reader shares), with the verdict cached process-wide and
filled single-flight, so concurrent callers on one host share one read. Only the
`Google-Extended` group is honoured,
because that is the product token Gemini's retrieval obeys: a host disallowing it refuses
the fetch server-side, so the read is spend with a known-zero return, which is what makes one
free request worth it. `urllib.robotparser` cannot express that:
`can_fetch("Google-Extended", url)` falls back to the `User-agent: *` group when no
Google-Extended group exists, which would skip the paid read on every host that merely
disallows generic crawlers. The group parser is therefore our own, in
`metaculus_bot/research/robots_policy.py` (shared with the Tier-1 url_context rung), and every ambiguity there resolves toward
PAYING rather than skipping (an unreadable robots.txt, an unmodelled rule shape, an absent
group all come back "not disallowed").

A disallow returns `status="robots_disallowed"` and earns no verification tier, because
nothing was read: only a `method` with an entry in `provenance._METHOD_TO_TIER` can be
stamped, which is also why a `throttled` fetch can never claim `fetched` and supersede the
briefing. It logs one `AGENTIC_URLCONTEXT_ROBOTS_SKIP` line (fields under Telemetry).

## The output artifact

`artifact.py` `render_findings` turns the banked findings into the markdown
section that gets appended to the bundle. It returns an empty string when there
are no findings and no pending leads.

Structure:

- `## Agentic Research Findings` header.
- `### ⚠ Corrections to the briefing` first, if any finding is flagged
  `discrepancy=true`. This block carries language telling the panel these
  findings contradict and supersede the corresponding briefing content. Putting
  it first is deliberate: a flagged briefing error is the single most valuable
  thing v2 can produce.
- The remaining findings grouped by topic, sorted.
- A text finding renders as Claim, Source, a blockquoted Quote, Date, and
  Retrieved how. An image finding renders its image ID and driver-reported
  visual observation in place of the quote, along with the same source, date,
  and retrieval fields.
- A `Pending leads:` list at the end for things the driver couldn't verify (dead
  links, paywalls, no coverage) but wanted to flag rather than guess at.

### Detachment lint

Findings are supposed to state facts, never a view on how the question resolves.
`artifact.py` `detachment_lint` enforces this with a banned-register regex over
each finding's claim and topic fields. Banned phrases include likelihood and
verdict language such as "likely", "unlikely", "probably", "suggests",
"indicates that", "we believe", "we expect", "points to", "this implies",
"bullish", "bearish", "in our view", and "odds are". A finding that trips the
lint is rejected rather than banked, and the rejection is fed back to the driver
in the tool result so it can rephrase. The `lint_rejections` counter tracks how
often this happens.

Alongside the four research tools, the loop exposes its own internal ones
(`_INTERNAL_TOOL_NAMES` in `tool_schemas.py`): `set_research_plan` registers the dry
run's ranked gaps, and external tool calls come back as a nudge to plan first
until it has run; `record_findings` banks findings mid-run; `conclude` finishes
the loop, optionally banking final findings and leaving pending leads. The two
findings tools run their input through the same validation and detachment lint.
Internal calls don't count against the tool-call budget.

### The findings gates

Three checks decide what the internal tools accept, and each one sits where the
naive behaviour would quietly cost the run.

`set_research_plan` rejects a plan with zero valid gaps (F3a) rather than storing
it. Storing it would flip W1's `plan_gate_active` off, which opens the external
tools, while `gates._evaluate_conclude_gate` returns `None` on `not plan.gaps`,
which disables the W2 gate outright. Between them that lets a driver conclude
with zero research. Leaving `state.research_plan` untouched (`None`, or a prior
valid plan) keeps the W1 gate armed and stops a re-plan to empty from clobbering
a plan that was already good, and the rejection nudges the driver to register
real gaps.

`run_agentic_loop` seeds the provenance sets from the frozen user brief, because
the URLs embedded in it (the resolution-source snapshot, the market snapshot, the
AskNews digests) are things the driver really saw, so a non-discrepancy finding
may cite them. The system prompt is a fixed template that embeds no question
URLs, so nothing is seeded from it. A discrepancy finding may not lean on a
briefing URL at all: see `gates._check_url_provenance`.

Each finding has one source URL. The driver is instructed to record evidence from
different sources as separate findings, keeping each source's evidence in its own
`claim` and `quote` or `visual_observation`, so every excerpt retains its own link.

The quote spot-check in `_validate_findings_payload` applies only to text
findings and is warn-only. A quote that is not found verbatim in the run's tool
contents is logged and counted in `quote_mismatch_warnings`, but the finding is
still banked, because `read_document` paraphrases and joins passages with
ellipses. The warning is deduped per run on `(source_url, quote)`, so a finding
re-listed in `conclude`'s `final_findings` counts once rather than once per
submission. Image findings have no quote: the finding validator instead
requires an image ID already delivered on a preceding driver turn and checks
that its `source_url` is the image's registered source or final URL. The visual
observation is scanned by detachment lint, but the code cannot verify that it
matches the pixels.

Two `Finding` fields carry their own rules. `derivation` (W3) is arithmetic-only
synthesis over the finding's own quoted numbers: a derived table, bound or rate
whose every input appears as a quoted value with a URL in that finding's quote and
source fields. It is exempt from the detachment lint (arithmetic plus its result,
no likelihood language, no new facts; `artifact.detachment_lint`) and rendered
under a "Derived analysis" label so the panel weights it as our synthesis rather
than a source claim. `verification_tier` (W4) is stamped by the loop at banking
time from the URL-to-best-method-seen map, never driver-claimed (the free-text
(A) to (D) tags in `claim` stay advisory): `fetched` when the URL was seen through
a fetch or read (document, rendered, plain or cache), `snippet` when only through
a search or news result, None until stamped (a briefing-only URL is never seen
through a tool). A discrepancy finding must be `fetched` to keep the supersede
banner; a snippet-tier discrepancy is demoted to "possible corrections", the
131.3 failure mode (`gates._stamp_verification_tier`, `artifact.render_findings`).

`GapAccountingEntry.status` (W2) is a plan gap's terminal disposition at conclude
time: `resolved` (the fact was found), `unresolved_parked` (attempted,
unresolvable this run, a pending lead) or `not_decision_relevant_on_inspection`
(it turned out not to move the forecast). All three are honest outcomes; the
conclude gate needs an entry with some action for every gap, not any particular
status.

## The ghost forecast (telemetry only)

After the driver concludes, the loop asks it to privately complete the forecast
itself using the panel's template and its own findings, and to output only the
structured forecast block. This is the "ghost forecast." It is never shown to
the panel and never published. It exists so the run logs carry a signal of what
a forecast built purely on v2's research would have looked like, which is useful
for evaluating driver quality. The ghost phase runs only when the driver
concluded explicitly (not when the deadline cut it off) and is bounded by its own
`asyncio.wait_for` in `loop.py`, so a slow ghost call can't eat into the run.

The ghost request offers the same tool list the last research turn offered and
forbids tool use with `tool_choice="none"`, instead of sending no tools. OpenAI's
prompt cache keys on the rendered prefix, and that prefix includes the tool
definitions ("cache reuse requires the entire rendered prefix to match"; the
settings that change the prefix are `model`, `tools`, `parallel_tool_calls`,
`text.format`, `reasoning.effort`, `text.verbosity` and `context_management`, and
`tool_choice` is not among them; OpenAI's own guidance is "set tool_choice to
none instead of removing the tool definitions"). Every research turn re-sends the
whole transcript and bills 87% of its input at the cached-read rate, so a ghost
sent with `tools=None` was the one call that re-paid full input price on the
entire history: about 41,000 tokens, $0.09 a question, 29% of the
`gap_fill_v2_driver` line in the 2026-09-09 cost pass
(`scratch/cost_pass_2026-09-09/v2_cost_anatomy.md`). The `LlmCall` protocol in
`agentic/llm.py` carries `tool_choice` for exactly this call; the research turns
leave it unset. OpenRouter forwards `tool_choice` unchanged and litellm lists it
among the OpenRouter provider's supported params, so nothing strips it. The
ghost's output and both of its markers are unchanged; `cached_tokens` on the
driver's `CREDIT_ROLE_SPEND` row is how the saving shows up on the next run.

`_run_ghost_phase` logs the parsed result twice: a lossy human-readable
`GHOST_FORECAST` line kept byte-identical for the already-harvested archive, and
an additive `GHOST_FORECAST_JSON` line carrying the complete forecast (every
percentile, not just the median) so `scripts/score_ghosts.py` can score numeric
ghosts. A date ghost's percentiles are written in epoch seconds, the axis the date
pipeline forecasts on; the scorer counts date ghosts by type and reports that none
can be scored while the residual dataset excludes date questions
(`docs/performance_analysis.md`). The JSON line is suppressed when no structured
block parsed, and it carries its question id the same way `GHOST_FORECAST` does,
through `log_prefix`, so the harvester derives it identically. The turn-one
plan emits the same pair as `GHOST_PRE` / `GHOST_PRE_JSON`
(`_set_research_plan_tool`) from the driver's pre-research dry run, so the
pre-versus-post delta measures whether v2's own research moved its own view.

### The v1 ghost

Since 2026-09-09 a second private forecast, the v1 ghost, is asked of the same
driver at the same effort whenever gap-fill v1 produced a section on the
question. Its brief is the plain ghost's plus that section, rendered under the
same `## Targeted Gap-Fill (second pass)` header the panel reads it under
(`driver_prompt.build_ghost_v1_prompt`). v1 and v2 run concurrently in one
gather (`gap_fill_stages.run_gap_fill_passes`), so the loop itself never sees
v1's section; the loop therefore hands out a `GhostContext` (the transcript up
to, not including, the plain ghost's prompt, the tool list the last research
turn offered, and the transport it used) through the seam's `ghost_context_sink`,
and the stage issues the v1 ghost after the gather, once both sections exist
(`agentic_gap_fill.run_gap_fill_v2_ghost_v1`, `loop.run_ghost_v1`). It runs
only when both the plain ghost ran and v1's section exists, so every
`GHOST_FORECAST_V1` has a `GHOST_FORECAST` partner in the same run; with
gap-fill v1 off there is no second call and no marker.

The request is cache-aligned the same way as the plain ghost: it re-sends the
prefix the last research turn and the plain ghost just sent, with the same tool
list and `tool_choice="none"`, so only v1's section and the instruction are new
input, about a cent or two a question. It branches off the transcript from
before the plain ghost's prompt rather than appending to it, so the driver
answers without seeing its own first ghost and the pair measures v1's section
alone. It never touches the loop's own transcript or its findings, never
publishes, and never raises: a failure or a timeout (the same 60 s bound, plus
the research phase's remaining budget) logs a WARNING and leaves the pair
half-empty. It costs one driver call of research-phase latency after the gather,
which the close-derived time budget bounds like everything else in the phase.
Markers `GHOST_FORECAST_V1` / `GHOST_FORECAST_V1_JSON` have exactly the plain
ghost's shapes; the archive payload carries it as `ghost_v1` beside `ghost`.

What each pair measures, all on the same cheap driver and never as a panel proxy:

- `GHOST_PRE_JSON` to `GHOST_FORECAST_JSON` (same run): what v2's own research
  did to the driver. Positive delta = the loop moved the driver toward the truth.
  This is the v2 instrument (`scripts/score_ghosts.py`, the `pre_post` read).
- `GHOST_FORECAST_JSON` to `GHOST_FORECAST_V1_JSON` (same run): what v1's
  section adds on top of v2's findings, the mirror read for v1 (`v1_pairs`).
- `GHOST_FORECAST_JSON` versus the published forecast: the driver against the
  ensemble, the original retire-v1 gate, read on the loop-moved subset only.

### Scoring the ghosts

`scripts/score_ghosts.py` (`make score_ghosts`) joins the harvested markers to the
resolved-question dataset on `post_id` and log-scores each ghost the way Metaculus
scores the published forecast. Two prod mechanisms decide how a numeric ghost is
built. Native-discrete questions (Metaculus `type == "discrete"`) publish a CDF on
a reduced grid (`cdf_size != 201`); prod builds every member directly on that grid
(`numeric/pipeline._build_discrete_distribution`) and aggregates positionally, so
the scorer builds the ghost with `num_points=len(published_cdf)` and step bounds
scaled to that length (`grid_step_constraints`), and both sides share the native
grid; no integer snap is involved (`discrete_snap` skips `cdf_size != 201`).
Continuous questions (`cdf_size == 201`) are integer-snapped by prod only when a
strict majority of the forecasters vote the outcome integer-valued; that vote is
prod-side state absent from both the record and the ghost payload, and the snap
is not reliably recoverable from the published CDF's shape, so the scorer scores
the ghost as the smooth distribution it declared. On that integer-outcome
minority the snapped published forecast holds a little extra mass on the
resolution bucket, so those deltas are mildly biased against the ghost; it is
bounded and does not touch the continuous questions that make up the bulk of the
gate. Date ghosts are counted but unscoreable while the residual dataset excludes
date questions.

When the driver supplies a `dry_run_forecast` that is not a dict, or one that
fails schema validation (the observed case is flat declared percentiles, run
30718626314), `GHOST_PRE_JSON` is suppressed and `_set_research_plan_tool` logs a
WARN saying this question's ghost pair will have no pre-research half. That line
exists because the loss is not random: it drops exactly the flattest
pre-research views, the ones whose later sharpening would be the strongest
evidence that research moved the driver, so the archived zero-move rate reads
slightly high.

## The bounds

The loop is anytime: it always emits whatever it has banked, even when it runs
out of budget. Three limits bound it:

- **Wall deadline** `GAP_FILL_V2_WALL_DEADLINE` (`constants.py`, env-overridable).
  A hard ceiling on the whole loop, enforced by an outer `asyncio.wait_for`. It
  sits inside v1's worst-case timing envelope, so running v2 concurrently with v1
  adds no research-phase wall-clock.
- **Max tool calls** `GAP_FILL_V2_MAX_TOOL_CALLS` (`constants.py`,
  env-overridable). Parallel calls each count against this cap. Steps, not calls,
  are where latency lives, so batching is encouraged.
- **Max steps** `LoopConfig.max_steps`, which the seam doesn't override, so
  this one lives on the dataclass rather than in `constants.py` and takes no env
  var. A step is one driver turn.
- **The gate caps**, also on the dataclass. `max_gaps` is how many ranked gaps
  `set_research_plan` keeps (the driver ranks them, so the dropped tail is the
  least valuable; `GAP_FILL_V2_MAX_GAPS` feeds it). `max_plan_nudges` is how many
  times the W1 plan gate may reject an external tool call before the loop
  soft-continues without a plan, so a driver that never plans cannot wedge it.
  `max_conclude_gate_rejections` mirrors it for W2: how many early conclusions the
  conclude gate may send back before accepting one unconditionally; a
  budget-exhaustion conclusion bypasses the gate and never counts.

There is also a **conclude threshold** `GAP_FILL_V2_CONCLUDE_THRESHOLD`. Once
fewer than that many seconds remain, or the tool-call cap is hit, `_tool_schemas`
stops offering the research tools and exposes only the internal ones
(`_INTERNAL_TOOL_NAMES`), which forces the driver to wrap up inside the wall
deadline. A budget line is appended to every tool result so the driver always
knows how much room it has left.

The loop also does light stuck-detection: an exact-duplicate tool call (same
tool, same normalized arguments) bumps a `dup_tool_calls` counter and gets a
gentle warning appended to its result telling the driver the result won't have
changed. There's no hard enforcement, just the nudge. One exception: a fetch that
came back throttled has its call key forgotten again as its tool message is
written, because a throttle outcome is never cached and its message asks the
driver to retry the same URL later in the run, so that retry really can return
something different and must not be told otherwise.

## Soft-fail and isolation (load-bearing safety property)

This is the most important property for anyone operating the bot: **gap-fill v2
can never crash a forecast.** A forecast built on first-pass research alone is
strictly better than no forecast, so every boundary in this subsystem degrades
to an empty string instead of raising. There are four layers of this:

1. **The seam** (`agentic_gap_fill.py` `run_gap_fill_v2`) returns `""` and makes
   zero LLM calls when the flag is off, when benchmarking, or for an unsupported
   question type, and it wraps the whole run in a broad `except` that logs and
   returns `""` on any error.
2. **The loop** (`loop.py` `run_agentic_loop`) wraps its body in
   `asyncio.wait_for` at the wall deadline. On timeout it marks `deadline_hit`
   and returns whatever findings it had banked. On any other exception it logs
   and returns the banked findings. `CancelledError` is the one thing it
   re-raises, so cancellation propagates cleanly.
3. **Each tool execution** is wrapped so that a tool timeout, a bad outcome, or
   any tool exception becomes an error `ToolOutcome` fed back to the driver,
   never a loop crash. The driver sees the error and can route around it.
4. **The orchestrator** (`orchestrator.py`) runs v1 and v2 in their own
   independent guards inside the gather, so a v2 defect (an import error in the
   agentic package, an unhandled raise) can never zero out v1's addendum, and
   vice versa.

Layer 2 has to tell two different timeouts apart. On Python 3.11 and newer
`asyncio.TimeoutError` is the builtin `TimeoutError`, so a connection-level
timeout raised inside the unguarded driver call arrives in the same `except` as a
real outer `wait_for` deadline. The loop classifies by elapsed wall time: a
genuine deadline hit has elapsed roughly `wall_deadline_s` (within
`_DEADLINE_SLOP_S`, which absorbs scheduling jitter), while an inner timeout
fires earlier and counts as a crash, stamping `error` and bumping the
orchestrator's alertable counter like any other soft-fail. Both stamped strings
are newline-sanitized, because the `GAP_FILL_V2` marker regex captures `error=`
to end-of-line and an embedded newline would truncate the harvest.

Layer 3 depends on where the handler coroutine is created. External tool handlers
have concrete signatures and `async def` binds its kwargs eagerly, so a missing,
misspelled or extra key in the LLM-emitted `arguments` raises `TypeError` at bind
time, before any `await`. `_run_tool_handler` therefore instantiates the
coroutine inside its own `try`, which turns that failure into a `status="error"`
outcome instead of letting it escape the batch `gather` and abort the whole pass,
matching what an unknown tool name does. The three internal tools bind
positionally and cannot hit this.

The benchmarking guard deserves its own mention. When `is_benchmarking=True`,
v2 returns `""` before doing anything. Live search on a resolved question sees
post-resolution information, which would leak the answer, so v2 is hard-off in
backtests for the same reason the prediction-market provider is.

## Telemetry

Every run that reaches the loop emits one INFO line to the run logs
(`loop.py` `_log_completion`):

```
GAP_FILL_V2: model=... steps=... tool_calls=... searches=... fetches=... rendered=... reads=... dup_tool_calls=... deadline_hit=... concluded_early=... wall_s=... findings=... pending_leads=... lint_rejections=... provenance_rejections=... quote_mismatch_warnings=... plan_gaps=... plan_skipped=... conclude_gate_rejections=... error=...
```

`searches` sums `search_news` and `search_web`; `rendered` counts fetches that
went all the way to the headless-Chromium rung; `reads` counts `read_document`
calls; `concluded_early` is true when the driver called `conclude` before the
deadline. `error` carries the `repr` of whatever tripped the loop's catch-all
soft-fail and is `None` on both a healthy run and a deadline hit, which makes it
the one field that separates a step-zero crash from an idle run: the two emit
otherwise byte-identical `steps=0 tool_calls=0 findings=0` lines. Everything from
`provenance_rejections` onward postdates the original marker, so
`scripts/telemetry/markers.py` wraps that tail in optional regex groups and still
harvests pre-branch archived logs that end at `lint_rejections`. This marker is
grep-able in the durable `run_logs/` artifacts every workflow tees, so the driver
can be vibe-evaluated after the fact without pulling any research-archive JSON.

One event outside that line has its own marker, because it is invisible in the counters:
a fetch whose 200-OK body was the host's rate-limit interstitial rather than the page logs

```
AGENTIC_FETCH_THROTTLED: url=... method=... chars=... phrase=...
```

as a WARN from `tools.py`, harvested as `agentic_fetch_throttled`. It carries no `question=`
(the tool handlers run below the loop's `log_prefix`), so a join goes through the run id.
Such a fetch returns `status=throttled` and is never cached, so the driver's retry of the
same URL is a real request; `chars` and `phrase` are the two fields that say whether a fire
was a true throttle or the rule over-reaching. Receipt: q45191, where two throttled
ogimet.com fetches reached the driver as successful ones and its own retry was served the
cached refusal.

A second event outside the counters is a document read for free, which is what the
local-document rung exists to produce:

```
AGENTIC_FETCH_LOCAL_DOC: url=... method=pdf_local|digest_local chars=... pages=... passages=...
```

as an INFO from `local_document.py`, harvested as `agentic_fetch_local_doc` and likewise
without a `question=`. `pdf_local` is a `fetch` serving a PDF's own extracted text, which
paginates like a long page and therefore selects nothing (`passages=n/a`); `digest_local` is a
`read_document` answering the ask from BM25-selected passages of text we hold, where
`passages=0` is the reading that matters: the document does not discuss what was asked, which
in the block itself reads exactly like a successful read. `chars` is the text we HELD, not the
window handed to the driver, so it is comparable across both routes and against
`URL_CONTEXT_SIZE_GATE_TOKENS` (chars / 4).

The line fires only where a digest or a PDF's text was actually SERVED, so its absence is not
a measurement: a `read_document` whose digest was refused (the one shape above) or whose
ladder held nothing leaves no line at all, and the paid read that followed is visible only in
the reader's own spend. Count fires, never non-fires.

A third is the pre-check that skips a paid read the host would refuse anyway:

```
AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=... host=...
```

as an INFO from `tools.py`, harvested as `agentic_urlcontext_robots_skip` and, like the two
above, with no `question=`. Non-alertable: a fire is a paid call NOT billed, not a defect.
`host` rides beside `url` because the robots verdict is cached and applied per host, so the
host is the unit any rate is computed over, and a suspiciously high rate is the signal that
the group parser is over-matching and withholding reads we could have had.

Field notes on the counters whose names do not say everything. `rendered_fetches`
counts fetches served by the headless-Chromium rung, a per-method count
`per_tool_counts` cannot see. `dup_tool_calls` counts exact-duplicate (tool,
normalized arguments) repeats, the stuck-detection nudge described under "The
bounds". `provenance_rejections` counts findings dropped because their cited
`source_url` never appeared in a tool result this run (the hard W3 gate), while
`quote_mismatch_warnings` counts findings ACCEPTED despite a quote not found
verbatim in the tool contents (warn-only, see "The findings gates").
`plan_gaps` is the number of ranked gaps the driver registered in
`set_research_plan`; `plan_skipped` is true when it never planned and the
plan-nudge cap was hit, so the loop soft-continued unplanned (W1, a degraded run
worth flagging); `conclude_gate_rejections` counts early conclusions the W2 gate
sent back before accepting one, and persistent 2s in prod flag a gate that is too
strict or a prompt that is unclear.

The existing completion line is unchanged by image support: `tool_calls` includes
`view_image`, while `fetches` and `reads` continue to count only `fetch` and
`read_document`. The archive's `telemetry.per_tool_counts` retains the separate
`view_image` count. There is no additional image marker.

For a richer trace, the seam accepts an `archive_sink` callback. When the loop
actually ran, the orchestrator captures `{transcript, telemetry, ghost, ghost_v1}`
through it and writes it into the research archive (`persistence.py`), including
empty-findings runs, whose telemetry is still worth keeping; `ghost` and
`ghost_v1` are the serialized ghost forecasts, None when that ghost did not run.

## Module layout

Everything lives under `metaculus_bot/research/agentic/`, with one seam file one
level up:

| File | What's in it |
| --- | --- |
| `agentic_gap_fill.py` (one level up) | The seam. `run_gap_fill_v2` owns prompt/tool/config construction and the outermost soft-fail boundary, keeping the orchestrator thin, and hands the loop's `GhostContext` out through `ghost_context_sink`; `run_gap_fill_v2_ghost_v1` is the v1 ghost the stage (`research/gap_fill_stages.py`) issues once both passes have landed. |
| `agentic/loop.py` | `run_agentic_loop` and the turn loop: message management, the three internal tool handlers, per-call handler dispatch, image materialization at the LLM boundary, the ghost phase, `run_ghost_v1`, the `GAP_FILL_V2` completion marker, and the timeout/soft-fail wrapper. Everything that logs one of this loop's telemetry markers stays here so the markers keep their `...agentic.loop` logger. |
| `agentic/tool_schemas.py` | `_INTERNAL_TOOL_NAMES` (the loop's own tools, and their timeout) plus the JSON-schema builders for the tool list advertised each turn. |
| `agentic/loop_state.py` | `_LoopState` (the one mutable per-run record), the `_ToolCall` / `_ToolExecutionResult` per-turn records, the assistant-message parsers that produce them, and the budget arithmetic. |
| `agentic/provenance.py` | URL and quote normalization, the quote-grounding span logic, and the per-call harvesters behind the provenance gate and the W4 verification tiers. |
| `agentic/gates.py` | The W1 plan gate's nudge and gap coercion, the W2 conclude gate, the W3 `source_url` check, and W4 tier stamping plus idempotent findings banking. |
| `agentic/dispatch.py` | One assistant turn's tool calls in, one tool message each out: batch admission (plan gate, call budget, duplicate detection), provenance absorption, byte-free image references appended after tool replies, and the tool-message/rejection rendering. |
| `agentic/tools.py` | `build_gap_fill_tools`, `question_ladder_context` and the search, fetch, document, image, and known-API handlers, plus `_fetch_via_ladder`, the one seam onto the shared fetch ladder, source selectors, window presentation, and throttle outcome. |
| `agentic/ladder_adapter.py` | One `FetchResult` read as this ladder's own `PlainFetchResult`: the status and method tables, and the message the driver is told for every non-read. |
| `agentic/local_document.py` | What the free ladder holds for one URL (`HeldDocument`), the passage digest `read_document` serves, the url_context size gate, and the `AGENTIC_FETCH_LOCAL_DOC` marker. The parses themselves are held in `research/document_cache.py`, shared with the ladder's document verdict. |
| `agentic/fetch_outcomes.py` | This ladder's result type (`PlainFetchResult`), the question-platform self-reference refusal (metaculus.com and `competitions.mantic.com`), and the escalate-to-a-reader outcome. Its per-body-shape builders are dead since the loop moved onto the shared classifier and are deleted with the rest of the loop's own rungs. |
| `agentic/tool_backends.py` | The outbound half of the tools: the AskNews and Exa clients with their retry ladders and concurrency caps, the Gemini `url_context` document read and its fixed in-thread ceiling, and the markdown formatting of what comes back. |
| `agentic/tool_descriptions.py` | The driver-facing tool descriptions and JSON parameter schemas: behavioral text, so a change here changes what the driver does. |
| `agentic/image_messages.py` | Byte-free image references in archived messages and their multimodal materialization for the next driver request. |
| `research/source_documents.py`, `research/source_presentation.py` | Bounded local source parsers and caller-time member/sheet selection, inventories, and query digests. |
| `research/image_leads.py`, `research/image_assets.py`, `research/image_persistence.py` | HTML image-lead metadata, bounded image normalization, and content-addressed sidecar persistence and archive verification. |
| `research/robots_policy.py` (outside `agentic/`, shared with the Tier-1 url_context rung) | The `Google-Extended` robots.txt group parser and per-host cache behind the pre-check on every paid read, written because `urllib.robotparser` falls back to `User-agent: *`. |
| `agentic/driver_prompt.py` | The four prompt builders: `build_system_prompt`, `build_user_brief`, `build_ghost_prompt`, `build_ghost_v1_prompt`, plus the `SupportedQuestion` type. |
| `agentic/artifact.py` | `render_findings` (the output section) and `detachment_lint`. |
| `agentic/types.py` | The dataclasses and Pydantic models: `ToolOutcome`, `ToolSpec`, `Finding`, `GhostForecast`, `LoopConfig`, `LoopTelemetry`, `GhostContext`, `LoopResult`. |
| `agentic/llm.py` | `build_default_llm_call`, the litellm/OpenRouter binding with donated-key-first routing and personal-key fallback. |
| `agentic/__init__.py` | Package exports. |

## Configuration

All flags are read in `constants.py`. The enable flag uses the standard
`env_flag_enabled` helper, so it is off unless explicitly set to
`true`/`1`/`yes`.

Defaults are deliberately not reproduced here. Read them off the definitions in
`constants.py`, which is the only copy that cannot go stale.

| Env var | What it controls |
| --- | --- |
| `GAP_FILL_V2_ENABLED` | Master switch. Off unless set; on in all four workflow yamls, live in prod since 2026-07-21 (`b4e9df0`). |
| `GAP_FILL_V2_DRIVER_MODEL` | The driver LLM. Picked by the 2026-07-17 blind 5-arm replay eval. |
| `GAP_FILL_V2_DRIVER_EFFORT` | Driver reasoning effort. |
| `GAP_FILL_V2_READER_MODEL` | The `read_document` backend model on the native google-genai path. |
| `GAP_FILL_V2_MAX_TOOL_CALLS` | Tool-call budget. |
| `GAP_FILL_V2_MAX_GAPS` | Ranked gaps `set_research_plan` accepts; the excess is dropped from the low-relevance end. Independent of v1's `GAP_FILL_MAX_GAPS`. |
| `GAP_FILL_V2_WALL_DEADLINE` | Hard wall for the whole loop, in seconds. |
| `GAP_FILL_V2_CONCLUDE_THRESHOLD` | Seconds-remaining threshold below which only `conclude` is offered. |
| `GAP_FILL_V2_MIN_CONTENT_CHARS` | Extracted-char floor below which `fetch` escalates plain HTTP to headless Chromium. |

The driver and reader run on separate credentials. The driver goes through
litellm/OpenRouter with donated-key-first routing (all eval candidates were
OpenAI/Anthropic models). The `read_document` reader uses the personal
`GOOGLE_API_KEY` on the native google-genai SDK, and `search_web` (Exa) and
`search_news` (AskNews) use their own personal keys.

One caveat worth flagging for operators: `GAP_FILL_V2_READER_MODEL`'s default id
(`gemini-3.8-flash`) was verified live on the native AI Studio SDK 2026-09-03, so
the constants file no longer carries an unverified-id caution. A wrong id still
soft-fails `read_document` (model-not-found becomes an error outcome), which
silently disables the directed-reading rung without breaking anything else, and
url_context retrieval fails the same quiet way on a host whose robots.txt
disallows `Google-Extended`. If `read_document` never seems to work, check the
reader model id first, then the target host's robots.txt.

### The driver transport

The driver's completions go through raw `litellm.acompletion` rather than a
`GeneralLlm` wrapper, because the tool loop needs the raw request. Five details
of that call are load-bearing (`agentic/llm.py`).

The `messages` list is passed as a shallow copy, since litellm mutates the
caller's list in place on some code paths and the loop's prefix has to stay
append-only. Copying the container and not the dicts keeps dict identity intact,
which is what providers cache on.

`metadata` carries the `CREDIT_ROLE_SPEND` tag. The `GeneralLlm` builders stamp
that once at construction; this path stamps it per call, so the alias names the
key the attempt actually bills.

`allowed_openai_params: ["reasoning_effort"]` is what gets the reasoning effort
to OpenRouter. litellm's `OpenrouterConfig` does not map `reasoning_effort`, so
without the whitelist the param survives only because `forecasting_tools` sets
`litellm.drop_params=True` globally, which silently strips it. Whitelisting
passes the raw param through, validated live by
`scratch/driver_replay_2026-07-17`.

`_skip_mcp_handler: True` is a private litellm kwarg, popped before the provider
sees it. litellm 1.92 and newer eagerly import the proxy MCP-gateway handler,
which requires fastapi (a proxy-only extra we do not install), whenever `tools`
is passed, even for plain function tools that never touch the gateway. We run our
own tool dispatch, so the import is skipped. Both the eager-import defect and
this skip kwarg are 1.92-era, verified against the locked litellm 1.92; if a
future litellm drops the kwarg, the call crashes loudly rather than regressing
quietly.

The donated-key fallback records itself. When the donated key fails with a
key-scoped error, `record_donated_key_fallback` counts the event once in the
generic total (plus at most one subset) and logs it as a paid personal-key
fallback, the same accounting `FallbackOpenRouterLlm.invoke` does. Without it the
bot's highest-volume donated-key path, a v2 run on every question in all four
prod workflows, failed over to the paid key completely silently. The
counted-and-logged decision is shared with `fallback_openrouter`; only the
transport differs, and if this path ever grows a retry ladder the transport
should be shared too.
