"""Raw research-provider payload logging — a durable "tape recorder" for GHA artifacts.

Each research provider formats its raw API return into markdown/prose before the
orchestrator combines it into the research bundle; only that combined text (plus a
post-summarizer briefing) reaches the research archive. That makes the raw evidence
behind a forecast — the exact AskNews articles, the grounding sources Gemini used,
the prediction-market contracts — unrecoverable after the fact, and the summarizer
relevance gate un-auditable (you can't see what it dropped).

``record_raw_research`` captures each provider's raw payload as one JSONL record,
appended to ``<RAW_RESEARCH_LOG_DIR>/raw_research_<run_id>.jsonl``. The four workflow
yamls tee stdout to ``run_logs/`` and upload it wholesale as an artifact, so the raw
log rides along and survives the 90-day window without depending on published
comments. ``scripts/download_raw_research.py`` archives it locally.

Contract: this MUST NEVER break a forecast. It is a best-effort side channel —
serialization failures and IO errors are caught, logged, and swallowed. When the
``RAW_RESEARCH_LOG_ENABLED`` env flag is unset (tests, local runs) every call is a
cheap no-op.
"""

import dataclasses
import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import NotRequired, TypedDict
from urllib.parse import urlsplit, urlunsplit

from metaculus_bot.constants import (
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    LOCAL_SOURCE_MAX_CHARS,
    RAW_RESEARCH_LOG_DIR_DEFAULT,
    RAW_RESEARCH_LOG_DIR_ENV,
    RAW_RESEARCH_LOG_ENABLED_ENV,
    RAW_RESEARCH_MAX_PAYLOAD_CHARS,
    RENDERED_DOM_MAX_CHARS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    env_flag_enabled,
)

logger = logging.getLogger(__name__)

RAW_RESEARCH_SCHEMA_VERSION = 1
SOURCE_CAPTURE_SCHEMA_VERSION = 1
SOURCE_CAPTURE_PROVIDER = "source_capture"
SOURCE_CAPTURE_RETENTION_DAYS = 90
SOURCE_CAPTURE_DIR = "source_captures"


class _SourceCaptureAsset(TypedDict):
    status: str
    asset_path: NotRequired[str]
    sha256: NotRequired[str]
    byte_count: NotRequired[int]
    unit_count: NotRequired[int]
    refusal_reason: NotRequired[str]


def _json_default(obj: object) -> object:
    """Coerce provider payloads to JSON-native form for ``json.dumps(default=...)``.

    Handles the concrete types real providers pass — pydantic models (AskNews
    ``Article``, the Gemini ``GenerateContentResponse``), dataclasses (the
    prediction-market ``MarketSnapshot``/``MarketMatch`` and resolution-source
    ``FetchResult``), and stray datetimes nested inside those. Anything else falls
    back to ``str`` so a payload with an exotic object degrades to its repr rather
    than raising. ``json.dumps`` calls this recursively for nested values.
    """
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _log_path() -> Path:
    log_dir = os.getenv(RAW_RESEARCH_LOG_DIR_ENV, RAW_RESEARCH_LOG_DIR_DEFAULT)
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    return Path(log_dir) / f"raw_research_{run_id}.jsonl"


def source_capture_enabled() -> bool:
    """Whether source sidecars share the existing raw-research artifact lifecycle."""
    return env_flag_enabled(RAW_RESEARCH_LOG_ENABLED_ENV)


def record_raw_research(
    *,
    qid: int | None,
    provider: str,
    phase: str | None = None,
    payload: object,
) -> None:
    """Append one raw provider payload to the run's raw-research JSONL log.

    ``phase`` distinguishes multiple raw payloads from one provider (AskNews's
    ``hot``/``historical`` searches); ``None`` for single-payload providers.
    No-op when ``RAW_RESEARCH_LOG_ENABLED`` is unset. Never raises.
    """
    if not source_capture_enabled():
        return

    try:
        payload_json = json.dumps(payload, default=_json_default, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # broad by contract: side-channel must never break a forecast (see module docstring)
        # Serialization guard: a non-str dict key `default=` can't rescue (TypeError),
        # or an arbitrary error raised inside a payload's model_dump/__str__
        # (RuntimeError, AttributeError, ...) must not propagate into the forecast.
        logger.warning(
            "raw_research: could not serialize payload for qid=%s provider=%s phase=%s: %s",
            qid,
            provider,
            phase,
            exc,
        )
        return

    payload_chars = len(payload_json)
    truncated = payload_chars > RAW_RESEARCH_MAX_PAYLOAD_CHARS
    if truncated:
        payload_field: object = {
            "_truncated": True,
            "_original_chars": payload_chars,
            "_preview": payload_json[:RAW_RESEARCH_MAX_PAYLOAD_CHARS],
        }
    else:
        # Round-trip so the outer dump sees JSON-native values (no second default=).
        payload_field = json.loads(payload_json)

    record = {
        "schema_version": RAW_RESEARCH_SCHEMA_VERSION,
        "qid": qid,
        "provider": provider,
        "phase": phase,
        "fetched_at": datetime.now(UTC).isoformat(),
        "payload_chars": payload_chars,
        "truncated": truncated,
        "payload": payload_field,
    }

    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # broad by contract: side-channel must never break a forecast (see module docstring)
        # Write guard: OSError from a bad path, but also UnicodeEncodeError (a
        # ValueError, not OSError) when a payload carries a lone surrogate that
        # survives json.dumps(ensure_ascii=False) and only fails at utf-8 encode.
        logger.warning("raw_research: failed to append record to %s: %s", path, exc)


def record_source_capture(
    *,
    qid: int | None,
    source_url: str,
    retrieval_method: str,
    content_type: str | None,
    parser_version: str,
    raw_body: bytes | None,
    parsed_text: str | None,
    truncated_by: str = "",
    refusal_reason: str = "",
    page_count: int | None = None,
    pages_read: int | None = None,
) -> dict[str, object] | None:
    """Save bounded source bytes and parsed text beside the raw research log.

    Only the small manifest is serialized into the existing raw-research collector. Source
    bytes and parsed text are separate content-addressed sidecars, kept under the same
    ``run_logs/`` artifact retention as the collector. Oversized inputs are refused whole;
    the manifest records the refusal instead of silently saving a prefix.
    """
    if not source_capture_enabled():
        return None

    output_dir = _log_path().parent
    is_rendered = retrieval_method == "rendered"
    if (raw_body or b"").startswith(b"%PDF-") or "pdf" in (content_type or "").lower():
        raw_max_units = DOCUMENT_TEXT_PDF_MAX_BYTES
    elif is_rendered:
        raw_max_units = RENDERED_DOM_MAX_CHARS
    else:
        raw_max_units = RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
    raw_capture = _source_capture_asset(
        raw_body,
        max_units=raw_max_units,
        unit_count=_utf8_character_count if is_rendered else len,
        extension=_source_capture_extension(raw_body, content_type),
        output_dir=output_dir,
    )

    parsed_bytes = None
    parsed_refusal = ""
    if parsed_text is not None:
        try:
            parsed_bytes = parsed_text.encode("utf-8")
        except UnicodeEncodeError:
            parsed_refusal = "parsed_text_not_utf8"
    parsed_capture = _source_capture_asset(
        parsed_bytes,
        max_units=LOCAL_SOURCE_MAX_CHARS,
        unit_count=_utf8_character_count,
        extension="txt",
        output_dir=output_dir,
    )

    raw_capture_status = raw_capture["status"]
    parsed_capture_status = parsed_capture["status"]
    refusals = [
        reason
        for reason in (
            refusal_reason,
            raw_capture.get("refusal_reason"),
            parsed_refusal,
            parsed_capture.get("refusal_reason"),
        )
        if reason
    ]
    if raw_capture_status == "captured" and parsed_capture_status == "captured" and not truncated_by and not refusals:
        capture_status = "complete"
    elif raw_capture_status == "captured" or parsed_capture_status == "captured":
        capture_status = "partial"
    elif raw_capture_status == "refused" or parsed_capture_status == "refused" or refusals:
        capture_status = "refused"
    else:
        capture_status = "unavailable"

    manifest: dict[str, object] = {
        "schema_version": SOURCE_CAPTURE_SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "qid": qid,
        "source_url": _source_capture_safe_url(source_url),
        "source_identity_sha256": hashlib.sha256(source_url.encode("utf-8")).hexdigest(),
        "retrieval_method": retrieval_method,
        "content_type": content_type,
        "parser_version": parser_version,
        "page_count": page_count,
        "pages_read": pages_read,
        "raw_capture_status": raw_capture_status,
        "raw_asset": raw_capture.get("asset_path"),
        "raw_sha256": raw_capture.get("sha256"),
        "raw_byte_count": raw_capture.get("byte_count"),
        "parsed_capture_status": parsed_capture_status,
        "parsed_asset": parsed_capture.get("asset_path"),
        "parsed_sha256": parsed_capture.get("sha256"),
        "parsed_byte_count": parsed_capture.get("byte_count"),
        "parsed_character_count": parsed_capture.get("unit_count"),
        "truncated_by": truncated_by,
        "refusal_reason": "; ".join(refusals),
        "capture_status": capture_status,
        "retention_days": SOURCE_CAPTURE_RETENTION_DAYS,
    }
    capture_identity_sha256 = _source_capture_identity_sha256(manifest)
    manifest["capture_identity_sha256"] = capture_identity_sha256
    record_raw_research(qid=qid, provider=SOURCE_CAPTURE_PROVIDER, phase=capture_identity_sha256, payload=manifest)
    return manifest


def _source_capture_identity_sha256(manifest: Mapping[str, object]) -> str:
    identity_fields = (
        "schema_version",
        "source_identity_sha256",
        "retrieval_method",
        "content_type",
        "parser_version",
        "page_count",
        "pages_read",
        "raw_capture_status",
        "raw_sha256",
        "raw_byte_count",
        "parsed_capture_status",
        "parsed_sha256",
        "parsed_byte_count",
        "truncated_by",
        "refusal_reason",
    )
    identity = {field: manifest[field] for field in identity_fields}
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()


def _source_capture_asset(
    content: bytes | None,
    *,
    max_units: int,
    unit_count: Callable[[bytes], int],
    extension: str,
    output_dir: Path,
) -> _SourceCaptureAsset:
    if content is None:
        return {"status": "unavailable"}
    count = unit_count(content)
    if count > max_units:
        return {"status": "refused", "refusal_reason": "over_capture_limit", "unit_count": count}
    digest = hashlib.sha256(content).hexdigest()
    asset_path = f"{SOURCE_CAPTURE_DIR}/{digest}.{extension}"
    destination = output_dir / asset_path
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if existing_digest != digest:
                logger.error("source_capture: existing content-addressed asset is corrupt: %s", destination)
                return {"status": "refused", "refusal_reason": "existing_asset_hash_mismatch"}
        else:
            destination.write_bytes(content)
    except OSError as exc:
        logger.warning("source_capture: could not persist asset %s: %s", asset_path, exc)
        return {"status": "refused", "refusal_reason": "asset_write_failed"}
    return {
        "status": "captured",
        "asset_path": asset_path,
        "sha256": digest,
        "byte_count": len(content),
        "unit_count": count,
    }


def _source_capture_extension(raw_body: bytes | None, content_type: str | None) -> str:
    if raw_body is not None and raw_body.startswith(b"%PDF-"):
        return "pdf"
    media_type = (content_type or "").partition(";")[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"} or (raw_body or b"").lstrip().lower().startswith(
        b"<!doctype html"
    ):
        return "html"
    return "bin"


def _utf8_character_count(content: bytes) -> int:
    return len(content.decode("utf-8", errors="replace"))


def _source_capture_safe_url(source_url: str) -> str:
    parts = urlsplit(source_url)
    if parts.scheme not in {"http", "https"} or parts.hostname is None:
        raise ValueError("Source capture URLs must be absolute HTTP or HTTPS URLs")
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))
