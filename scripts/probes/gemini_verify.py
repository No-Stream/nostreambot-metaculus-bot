"""Verify a candidate Gemini model id, and test the Google-Extended robots hypothesis.

This probe answers two questions the repo cannot answer offline, and it SPENDS the
operator's own Google AI Studio credit to do it, which is why it refuses to run
without ``--i-accept-spend``.

1. Does the candidate model id work on the native google-genai SDK, with grounding
   and with the ``thinking_level`` knob? A bad id fails as a 404 from the SDK rather
   than as a quiet fallback, so call 1 either prints a grounded answer with real
   search queries behind it or crashes naming the id.

2. Does Gemini's ``url_context`` tool honour a robots.txt ``Google-Extended:
   Disallow``? The hypothesis comes from prod: two internationalaisafetyreport.org
   URLs came back with zero retrievals while the answer text stayed fluent, which is
   the failure mode ``research/url_context_telemetry`` exists to catch — and that
   site disallows Google-Extended. Calls 2 and 3 read one URL each, matched on
   everything except that robots directive: Wikipedia allows Google-Extended, the AI
   Safety Report does not. If the allowed URL retrieves and the disallowed one does
   not, the directive is the mechanism, and no amount of retrying will fetch that
   host through url_context.

Run it locally, with GOOGLE_API_KEY set (the operator's personal AI Studio key —
there is no Metaculus-donated key on the google-genai side):

    uv run python scripts/probes/gemini_verify.py --i-accept-spend
"""

from __future__ import annotations

import argparse
import os
from typing import Any, NamedTuple

from google import genai
from google.genai import types as genai_types

from metaculus_bot.constants import GOOGLE_API_KEY_ENV
from metaculus_bot.research.url_context_telemetry import URL_RETRIEVAL_SUCCESS, extract_url_context_telemetry

# The id under test. Deliberately a literal here rather than a read of
# GEMINI_SEARCH_DEFAULT_MODEL: the point of this probe is to verify an id the repo
# has NOT adopted yet, so reading the adopted one would test nothing.
CANDIDATE_MODEL = "gemini-4-hydrogen"

# Matched pair for the robots test. Everything about the two calls is identical
# except the target host's Google-Extended directive.
ROBOTS_ALLOWED_URL = "https://en.wikipedia.org/wiki/Nuri_(rocket)"
ROBOTS_DISALLOWED_URL = "https://internationalaisafetyreport.org/publication/international-ai-safety-report-2026"

_GROUNDED_PROMPT = (
    "In two sentences, what is the most recent US unemployment rate print and its release date? Cite sources."
)
_QUOTE_ASK = "Quote the first sentence of this page verbatim."

# Client-side ceiling in milliseconds, same shape as gap-fill v2's reader
# (``research/agentic/tool_backends._run_document_read_sync``): a hung endpoint
# returns rather than pinning the probe forever.
_HTTP_TIMEOUT_MS = 90_000

_ANSWER_PREVIEW_CHARS = 400

# Both url_context calls run at the same thinking level, because the pair is a
# controlled comparison: the only difference allowed between them is the target host.
_URL_CONTEXT_THINKING_LEVEL = genai_types.ThinkingLevel.LOW


def build_probe_client() -> genai.Client:
    """Construct the google-genai client, mirroring the bot's own construction shape."""
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{GOOGLE_API_KEY_ENV} must be set to run this probe")
    return genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=_HTTP_TIMEOUT_MS))


def print_cost_estimate(model: str) -> None:
    """State what this run spends, and on which assumptions, before anything is called."""
    print("Estimated cost of this run")
    print(f"  Three live calls to {model} on the operator's personal Google AI Studio key:")
    print("    call 1  google_search grounding, thinking_level MEDIUM, one-sentence prompt")
    print(f"    call 2  url_context read of {ROBOTS_ALLOWED_URL}")
    print(f"    call 3  url_context read of {ROBOTS_DISALLOWED_URL}")
    print(
        "  Token cost is dominated by whatever url_context retrieves, which is single-digit thousands to a\n"
        "  few tens of thousands of input tokens across calls 2 and 3. At flash-tier token prices that is a\n"
        "  fraction of a cent to a few cents. This model's own published price is NOT verified here, so treat\n"
        "  the figure as an order of magnitude; the usage_metadata printed after each call is the real number."
    )
    print(
        "  Call 1 also consumes one grounded prompt from the project's 5,000-per-month grounded allowance,\n"
        "  shared across Gemini 3 models. If that allowance is already spent this month, a grounded prompt\n"
        "  bills per search query on overage (order $14 per 1,000 queries), so call 1 alone could cost a few\n"
        "  cents. Check the AI Studio credit balance first if that matters."
    )
    print()


def _print_response_header(label: str, response: genai_types.GenerateContentResponse) -> None:
    """Print the identity + accounting block every call reports."""
    print(f"{label}")
    print(f"  model_version: {response.model_version}")
    usage = response.usage_metadata
    if usage is None:
        print("  usage_metadata: absent")
        return
    print(
        f"  tokens: prompt={usage.prompt_token_count} candidates={usage.candidates_token_count} "
        f"thoughts={usage.thoughts_token_count} tool_use_prompt={usage.tool_use_prompt_token_count} "
        f"total={usage.total_token_count}"
    )


def _print_answer(response: genai_types.GenerateContentResponse) -> None:
    text = (response.text or "").strip()
    preview = text[:_ANSWER_PREVIEW_CHARS]
    suffix = " […]" if len(text) > _ANSWER_PREVIEW_CHARS else ""
    print(f"  answer ({len(text)} chars): {preview!r}{suffix}")


def run_grounded_call(client: genai.Client, model: str) -> None:
    """Call 1: does this id serve grounded search, and does thinking_level take?"""
    tools: list[Any] = [{"google_search": {}}]
    config = genai_types.GenerateContentConfig(
        tools=tools,
        thinking_config=genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.MEDIUM),
    )
    response = client.models.generate_content(model=model, contents=_GROUNDED_PROMPT, config=config)

    _print_response_header("Call 1 — google_search grounding, thinking_level MEDIUM", response)
    candidates = response.candidates
    metadata = candidates[0].grounding_metadata if candidates else None
    queries = list(metadata.web_search_queries or []) if metadata is not None else []
    n_chunks = len(metadata.grounding_chunks or []) if metadata is not None else 0
    print(f"  web_search_queries: {len(queries)} {queries}")
    print(f"  grounding_chunks: {n_chunks}")
    _print_answer(response)
    print()


class UrlContextResult(NamedTuple):
    """One url_context call's retrieval telemetry, which is the whole point of calls 2 and 3."""

    url: str
    reported: bool
    n_total: int
    n_success: int
    entries: list[tuple[str, str]]


def run_url_context_call(client: genai.Client, model: str, url: str, label: str) -> UrlContextResult:
    """Calls 2 and 3: read one URL and report whether Gemini actually retrieved it."""
    tools: list[Any] = [{"url_context": {}}]
    config = genai_types.GenerateContentConfig(
        tools=tools,
        thinking_config=genai_types.ThinkingConfig(thinking_level=_URL_CONTEXT_THINKING_LEVEL),
    )
    response = client.models.generate_content(model=model, contents=f"{_QUOTE_ASK}\n\nURL: {url}", config=config)

    _print_response_header(label, response)
    reported, n_total, n_success, entries = extract_url_context_telemetry(response)
    print(f"  url_context reported: {reported}; retrievals: {n_success}/{n_total} succeeded")
    for status_name, retrieved_url in entries:
        marker = "ok " if status_name == URL_RETRIEVAL_SUCCESS else "FAIL"
        print(f"    {marker} {status_name} — {retrieved_url or '(no url reported)'}")
    if not entries:
        print("    (no url_metadata entries — the tool reported nothing it tried to fetch)")
    _print_answer(response)
    print()
    return UrlContextResult(url, reported, n_total, n_success, entries)


def print_robots_verdict(allowed: UrlContextResult, disallowed: UrlContextResult) -> None:
    """Decide the Google-Extended question from the matched pair."""
    print("Google-Extended hypothesis")
    if allowed.n_success > 0 and disallowed.n_success == 0:
        print("  Google-Extended hypothesis SUPPORTED")
        print(
            "  The allowed host retrieved and the disallowed host did not, on identical calls, so the robots\n"
            "  directive is the mechanism: url_context will not fetch that host, and retrying cannot help.\n"
            "  Any question resolving on it needs a different route (Tier-1 fetch, or the archived copy)."
        )
        return
    if allowed.n_success > 0 and disallowed.n_success > 0:
        print("  Google-Extended hypothesis REFUTED")
        print(
            "  Both hosts retrieved, so the directive is not what suppressed the prod reads and the zero-retrieval\n"
            "  observation needs another explanation (transient failure, page size, or the specific URLs used)."
        )
        return
    print("  Google-Extended hypothesis INCONCLUSIVE")
    print(
        "  The allowed-host control did not retrieve either, so this run says nothing about the directive — the\n"
        "  control is what the comparison rests on. Re-run and check the key, the quota and the model id first."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a candidate Gemini model id and test the Google-Extended robots hypothesis. SPENDS MONEY."
    )
    parser.add_argument(
        "--i-accept-spend",
        action="store_true",
        help="Required. Confirms you accept three live, billed calls on your own Google AI Studio key.",
    )
    parser.add_argument(
        "--model",
        default=CANDIDATE_MODEL,
        help=f"Model id to verify (default: {CANDIDATE_MODEL}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_cost_estimate(args.model)
    if not args.i_accept_spend:
        print("Refusing to run: this probe makes three billed calls. Re-run with --i-accept-spend to proceed.")
        raise SystemExit(2)

    client = build_probe_client()
    run_grounded_call(client, args.model)
    allowed = run_url_context_call(
        client, args.model, ROBOTS_ALLOWED_URL, "Call 2 — url_context, robots ALLOWS Google-Extended (control)"
    )
    disallowed = run_url_context_call(
        client, args.model, ROBOTS_DISALLOWED_URL, "Call 3 — url_context, robots DISALLOWS Google-Extended (test)"
    )
    print_robots_verdict(allowed, disallowed)


if __name__ == "__main__":
    main()
