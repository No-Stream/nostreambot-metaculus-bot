"""The two fetch markers' one spelling, for both callers of the shared ladder.

``RESOLUTION_SOURCE_FETCH`` and ``RESOLUTION_SOURCE_ESCALATION`` are data contracts: the research
archive matches them by regex on the exact field order and spelling
(``scripts/telemetry/markers.py``). Two callers now emit them — the resolution-source fetcher at its
per-question aggregation point, and the gap-fill v2 loop after each tool call — so the format string
lives here rather than once per emitter, where the two would drift a field apart and the archive
would silently stop parsing one of them.

Each caller still owns its own ``logger``, so a line keeps the module prefix of the code that
produced it, and its own ``question=`` ref: the fetcher has the question id at that point and the
loop does not (its three event markers carry ``question=None`` for the same reason, so a join to a
question goes through the run id). Field semantics and receipts:
``docs/telemetry_markers.md``.
"""

from __future__ import annotations

from metaculus_bot.research.resolution_fetch_result import FetchResult, fetch_outcome_token


def fetch_marker_line(result: FetchResult, *, qid: int | None, caller: str) -> str:
    """The ``RESOLUTION_SOURCE_FETCH`` line for one fetched URL.

    Every optional field is appended only when present and in one fixed order, so a line carrying
    some but none of the others parses without a group claiming its neighbour's value, and every
    line the archive already holds stays byte-identical.
    """
    reason = f" reason={result.status_reason}" if result.status_reason else ""
    route = f" route={result.route}" if result.route != "direct" else ""
    failure_class = f" failure_class={result.failure_class}" if result.failure_class else ""
    exc = f" exc={result.exc}" if result.exc else ""
    server = f" server={result.server}" if result.server else ""
    digest = ""
    if result.passages_returned is not None:
        if (
            result.passages_grounded is None
            or result.fallback_used is None
            or result.digest_outcome is None
            or result.digest_scope is None
        ):
            raise ValueError("digest marker fields must be provided together")
        digest = (
            f" passages_returned={result.passages_returned} passages_grounded={result.passages_grounded}"
            f" fallback_used={result.fallback_used} digest_outcome={result.digest_outcome}"
            f" digest_scope={result.digest_scope}"
        )
    embeds = ",".join(result.unreadable_embeds) if result.unreadable_embeds else "none"
    return (
        f"RESOLUTION_SOURCE_FETCH: question={qid} url={result.url} status={fetch_outcome_token(result)} "
        f"http={result.http_status if result.http_status is not None else 'n/a'} "
        f"embeds={embeds}{reason}{route}{failure_class}{exc}{server}{digest} caller={caller}"
    )


def escalation_marker_lines(result: FetchResult, *, qid: int | None, caller: str) -> list[str]:
    """One ``RESOLUTION_SOURCE_ESCALATION`` line per rung that FIRED on this URL.

    A skipped rung emits none: the marker means "a rung fired and finished", and skips ride the
    provider's ``details["counts"]`` instead, where a zero renders nothing but survives into the
    archive.
    """
    return [
        f"RESOLUTION_SOURCE_ESCALATION: question={qid} url={attempt.url} "
        f"from_status={attempt.from_status} rung={attempt.rung} outcome={attempt.outcome} "
        f"wall_s={attempt.wall_s if attempt.wall_s is not None else 0.0:.2f} caller={caller}"
        for attempt in result.rung_attempts
        if not attempt.skipped_reason
    ]
