"""Driver-facing text for the public agentic tools: descriptions + JSON parameter schemas.

This is what the gap-fill v2 driver LLM reads when it decides which tool to call, so the
wording is behavioral rather than documentation: the descriptions steer
search_news-vs-search_web routing, promise that ``fetch`` handles PDFs / JS pages / images
so the driver does not avoid a URL by format, and spell out the ``start_char`` pagination
contract. Split out of ``tools.py`` so the prose sits in one place and the module holding
the fetch ladder stays readable; ``build_gap_fill_tools`` (``tools.py``) pairs each
description + schema with its handler.
"""

from __future__ import annotations

from metaculus_bot.constants import MANTIC_HOST, METACULUS_HOST

SEARCH_NEWS_DESCRIPTION = (
    "Search recent and historical NEWS coverage (AskNews). Use for: events,\n"
    "announcements, things that happened, ongoing-situation updates. Query with a\n"
    "short natural-language phrase, not keywords. Returns a digest of matching\n"
    "articles with dates and URLs. Use search_web instead for: reports, datasets,\n"
    "official documents, niche/technical facts, or anything where the best source\n"
    "is not a news article.\n"
    'Example: search_news(query="Nauru parliament treaty ratification vote")'
)

SEARCH_WEB_DESCRIPTION = (
    "Semantic web search (Exa). Use for: official documents, datasets, reports,\n"
    "organizational pages, technical/niche facts, finding a primary source you\n"
    "believe exists. Returns results with URLs and relevant excerpts. Follow up\n"
    "promising results with fetch(url) — excerpts are often not enough to verify\n"
    "a claim. Use search_news instead for event/news coverage.\n"
    'Example: search_web(query="IAEA safeguards report Iran enrichment June 2026 pdf")'
)

FETCH_DESCRIPTION = (
    "Fetch a URL and return its main content as concise markdown, plus a list of\n"
    "outbound links. Handles ordinary pages, JavaScript-heavy pages, PDFs, archives,\n"
    "workbooks, Word documents, and raster-image discovery. A PDF is read here, in\n"
    "full text, with `method=pdf_local`, and paginates exactly like a long HTML\n"
    "page. Content over the size cap is truncated, ending with `[truncated at N\n"
    "of M chars — call again with start_char=N]`; pass start_char to read the\n"
    "next window (continuations are served from cache — they are cheap and do not\n"
    "refetch). Archive and workbook inventories are navigation only: pass the\n"
    "exact member and sheet names shown to read their content. A direct raster-image\n"
    "URL delivers its pixels automatically. HTML image leads have pixels-not-read\n"
    "metadata; call view_image on a lead URL when its pixels matter.\n"
    "Links in the result are leads you can fetch next.\n"
    "Use read_document instead only when you need a specific question answered\n"
    "from inside a long/complex document.\n"
    f"Do NOT fetch {METACULUS_HOST} or {MANTIC_HOST} URLs — the question brief already reflects them.\n"
    "FRED economic series, Kalshi and Polymarket market prices, and Yahoo Finance quotes are\n"
    "already in the research briefing (its financial-data and prediction-market sections) via\n"
    "their APIs — cite those rather than fetching these hosts as web pages.\n"
    "Use fred_series or yahoo_history for a different date window, and market_snapshot for current prices.\n"
    'Example: fetch(url="https://www.ons.gov.uk/releases/gdpquarterly")\n'
    'Example: fetch(url="https://example.gov/long-report", start_char=12000)'
)

READ_DOCUMENT_DESCRIPTION = (
    "Ask a specific question of a specific document, and get back the passages of\n"
    "it that bear on your `ask`, quoted verbatim with page numbers where the\n"
    "document has pages (`method=digest_local`). Use it for targeted extraction\n"
    "from a long or complex document — a 200-page report, a filing, or a parsed\n"
    "archive/workbook. Use exact member/sheet selectors to narrow local sources.\n"
    "and for an ordinary web URL where fetch returned status=blocked/js_wall/error and you\n"
    "still need the content: this tool fetches the page itself, and where it\n"
    "cannot, a model reads the URL for you instead (`method=document`, slower).\n"
    "A local archive/Office parser refusal is terminal and never uses that model fallback.\n"
    "For an archive, select its exact member before a sheet; sheet names are exact.\n"
    "A direct raster-image URL returns its pixels locally and never uses the model fallback.\n"
    "A digest with no matching passage means the document does not discuss what\n"
    "you asked, not that the read failed — try a different ask or another source.\n"
    "Always pass a precise `ask`: it is what selects the passages.\n"
    'Example: read_document(url="https://example.gov/report-q2.pdf",\n'
    '                       ask="What is the reported unemployment rate for May 2026, and what revision to April is stated?")'
)

VIEW_IMAGE_DESCRIPTION = (
    "View a supported raster image using bytes fetched and retained by the shared\n"
    "fetch ladder. Returns bounded normalized pixels to your next turn plus an\n"
    "image_id and metadata. Optional crop is [left, top, right, bottom] in the\n"
    "displayed orientation of the original image. Use the image_id when recording\n"
    "visual evidence. SVG and animated images are unsupported.\n"
    'Example: view_image(url="https://example.gov/chart.png")\n'
    'Example: view_image(url="https://example.gov/chart.png", crop=[100, 40, 900, 640])'
)

_SEARCH_NEWS_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

_SEARCH_WEB_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "end_published_date": {"type": ["string", "null"]},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_FETCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "start_char": {"type": "integer", "minimum": 0},
        "member": {"type": ["string", "null"]},
        "sheet": {"type": ["string", "null"]},
    },
    "required": ["url"],
    "additionalProperties": False,
}

_READ_DOCUMENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "ask": {"type": "string"},
        "member": {"type": ["string", "null"]},
        "sheet": {"type": ["string", "null"]},
    },
    "required": ["url", "ask"],
    "additionalProperties": False,
}

_VIEW_IMAGE_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "crop": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}
