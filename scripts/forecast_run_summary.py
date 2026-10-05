#!/usr/bin/env python3
"""Render a safe GitHub summary and job title from forecast run status JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn

MAX_STATUS_BYTES = 64_000
MAX_TITLE_LENGTH = 100

OUTCOMES = {"clean", "degraded", "hard_failure", "unknown"}
COMPONENT_LABELS = {
    "asknews": "AskNews",
    "aggregation": "Aggregation",
    "credit": "Credits",
    "custom": "Custom research",
    "exa": "Exa",
    "financial_data": "Financial data",
    "gap_fill_v1": "Gap-fill analyzer",
    "gap_fill_v2": "Agentic gap-fill",
    "gemini_search": "Gemini search",
    "kalshi": "Kalshi",
    "kalshi_catalogue": "Kalshi catalogue",
    "manifold": "Manifold",
    "manifold_detail": "Manifold detail",
    "models": "Models",
    "native_search": "Native search",
    "openrouter": "OpenRouter",
    "perplexity": "Perplexity",
    "polymarket": "Polymarket",
    "provider_health": "Provider health",
    "publishing": "Publishing",
    "predictit": "PredictIt",
    "prediction_market": "Prediction market",
    "publish_gate": "Publication gate",
    "query_author": "Query author",
    "ranking": "Ranking",
    "research": "Research",
    "resolution_source": "Resolution source",
    "runtime": "Runtime",
    "setup": "Setup",
    "snapshot": "Snapshot",
    "summarizer": "Summarizer",
    "timeseries_anchor": "Timeseries anchor",
    "tournament": "Tournament status",
    "unknown": "Unknown component",
}
ERROR_LABELS = {
    "aggregation_failure": "aggregation failure",
    "authentication_error": "authentication error",
    "connection_error": "connection error",
    "configuration_reminder": "configuration reminder",
    "credit_floor": "credit floor reached",
    "deprecated_model": "deprecated model",
    "forecast_drop": "forecast dropped",
    "http_error": "HTTP error",
    "provider_failure": "provider failure",
    "publish_failure": "publication failure",
    "summary_error": "summary error",
    "stale_tournament": "stale tournament status",
    "time_budget": "limited time budget",
    "timeout": "timeout",
    "unknown": "unknown error",
}
DROP_CAUSES = {
    "error_other",
    "parse_extraction",
    "timeout_soft_deadline",
    "timeout_wall_clock",
    "zero_output",
}
COUNT_UNIT_LABELS = {
    "alerts": ("alert", "alerts/events"),
    "lookups": ("lookup", "lookups"),
    "models": ("model", "models"),
    "provider_calls": ("provider call", "provider calls"),
    "publications": ("publication", "publications"),
    "source_checks": ("source check", "source checks"),
}
UNKNOWN_COUNT_UNIT = "unknown"


class StatusError(ValueError):
    """The status file is missing, malformed, or outside the supported schema."""


@dataclass(frozen=True)
class Cause:
    component: str
    unit: str
    affected: int | None
    attempted: int | None
    error_type: str | None
    http_status: int | None


@dataclass(frozen=True)
class Counts:
    attempted: int | None
    succeeded: int | None
    failed: int | None


@dataclass(frozen=True)
class Models:
    attempted: int | None
    succeeded: int | None
    timed_out: int | None
    dropped: int | None
    drop_causes: dict[str, int | None]


@dataclass(frozen=True)
class RunStatus:
    outcome: str
    causes: tuple[Cause, ...]
    models: Models
    forecasts: Counts
    comments: Counts


def _fail(message: str) -> NoReturn:
    raise StatusError(message)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(f"{name} must be an object")
    return value


def _required(mapping: dict[str, object], key: str, name: str) -> object:
    if key not in mapping:
        _fail(f"{name}.{key} is missing")
    return mapping[key]


def _count(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(f"{name} must be a non-negative integer or null")
    return value


def _enum(value: object, allowed: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{name} has an unsupported value")
    return value


def _safe_enum(value: object, allowed: set[str], fallback: str) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def _known_count(value: int | None, name: str) -> int:
    if value is None:
        _fail(f"{name} is missing from a clean outcome")
    return value


def _cause(value: object, index: int) -> Cause:
    name = f"causes[{index}]"
    entry = _mapping(value, name)
    component = _safe_enum(_required(entry, "component", name), set(COMPONENT_LABELS), "unknown")
    unit = _safe_enum(entry.get("unit"), set(COUNT_UNIT_LABELS), UNKNOWN_COUNT_UNIT)
    affected = _count(_required(entry, "affected", name), f"{name}.affected")
    attempted = _count(_required(entry, "attempted", name), f"{name}.attempted")
    error_type = _safe_enum(_required(entry, "error_type", name), set(ERROR_LABELS), "unknown")
    http_status_value = _required(entry, "http_status", name)
    http_status = (
        http_status_value
        if isinstance(http_status_value, int)
        and not isinstance(http_status_value, bool)
        and 100 <= http_status_value <= 599
        else None
    )
    if affected is not None and attempted is not None and affected > attempted:
        _fail(f"{name}.affected exceeds attempted")
    return Cause(component, unit, affected, attempted, error_type, http_status)


def _models(value: object) -> Models:
    entry = _mapping(value, "models")
    attempted = _count(_required(entry, "attempted", "models"), "models.attempted")
    succeeded = _count(_required(entry, "succeeded", "models"), "models.succeeded")
    timed_out = _count(_required(entry, "timed_out", "models"), "models.timed_out")
    dropped = _count(_required(entry, "dropped", "models"), "models.dropped")
    raw_drop_causes = _mapping(_required(entry, "drop_causes", "models"), "models.drop_causes")
    drop_causes: dict[str, int | None] = {}
    for cause, count in raw_drop_causes.items():
        parsed_count = _count(count, "models.drop_causes count")
        if cause in DROP_CAUSES:
            drop_causes[cause] = parsed_count
    if attempted is not None and succeeded is not None and dropped is not None and succeeded + dropped > attempted:
        _fail("models succeeded plus dropped exceeds attempted")
    if timed_out is not None and dropped is not None and timed_out > dropped:
        _fail("models timed out exceeds dropped")
    if (
        dropped is not None
        and all(count is not None for count in drop_causes.values())
        and sum(count for count in drop_causes.values() if count is not None) > dropped
    ):
        _fail("models drop causes exceed dropped")
    return Models(attempted, succeeded, timed_out, dropped, drop_causes)


def _publishing_counts(value: object, name: str) -> Counts:
    entry = _mapping(value, f"publishing.{name}")
    counts = Counts(
        _count(_required(entry, "attempted", f"publishing.{name}"), f"publishing.{name}.attempted"),
        _count(_required(entry, "succeeded", f"publishing.{name}"), f"publishing.{name}.succeeded"),
        _count(_required(entry, "failed", f"publishing.{name}"), f"publishing.{name}.failed"),
    )
    if (
        counts.attempted is not None
        and counts.succeeded is not None
        and counts.failed is not None
        and counts.succeeded + counts.failed > counts.attempted
    ):
        _fail(f"publishing.{name} successes plus failures exceeds attempts")
    return counts


def _parse_status(raw: object) -> RunStatus:
    status = _mapping(raw, "status")
    schema_version = _required(status, "schema_version", "status")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        _fail("status schema version is unsupported")
    outcome = _enum(_required(status, "outcome", "status"), OUTCOMES, "status.outcome")
    raw_causes = _required(status, "causes", "status")
    if not isinstance(raw_causes, list):
        _fail("status.causes must be an array")
    causes = tuple(_cause(value, index) for index, value in enumerate(raw_causes))
    models = _models(_required(status, "models", "status"))
    publishing = _mapping(_required(status, "publishing", "status"), "publishing")
    forecasts = _publishing_counts(_required(publishing, "forecasts", "publishing"), "forecasts")
    comments = _publishing_counts(_required(publishing, "comments", "publishing"), "comments")
    if outcome == "clean":
        _validate_clean_outcome(models, forecasts, comments, causes)
    return RunStatus(outcome, causes, models, forecasts, comments)


def _validate_clean_outcome(models: Models, forecasts: Counts, comments: Counts, causes: tuple[Cause, ...]) -> None:
    model_attempted = _known_count(models.attempted, "models.attempted")
    model_succeeded = _known_count(models.succeeded, "models.succeeded")
    model_timed_out = _known_count(models.timed_out, "models.timed_out")
    model_dropped = _known_count(models.dropped, "models.dropped")
    forecast_attempted = _known_count(forecasts.attempted, "forecasts.attempted")
    forecast_succeeded = _known_count(forecasts.succeeded, "forecasts.succeeded")
    forecast_failed = _known_count(forecasts.failed, "forecasts.failed")
    comment_attempted = _known_count(comments.attempted, "comments.attempted")
    comment_succeeded = _known_count(comments.succeeded, "comments.succeeded")
    comment_failed = _known_count(comments.failed, "comments.failed")
    if model_succeeded + model_dropped != model_attempted:
        _fail("clean outcome has unsettled model attempts")
    if model_dropped != 0 or model_timed_out != 0:
        _fail("clean outcome includes dropped or timed out models")
    if forecast_succeeded + forecast_failed != forecast_attempted:
        _fail("clean outcome has unsettled forecast publications")
    if comment_succeeded + comment_failed != comment_attempted:
        _fail("clean outcome has unsettled comment publications")
    if forecast_failed != 0 or comment_failed != 0:
        _fail("clean outcome includes failed publications")
    if any(
        cause.unit not in {"lookups", "source_checks", "provider_calls"}
        and (cause.affected != 0 or cause.attempted is None)
        for cause in causes
    ):
        _fail("clean outcome includes an alertable or unknown cause")


def _read_status(path: Path) -> RunStatus:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise StatusError("status file is unavailable") from error
    if len(raw_bytes) > MAX_STATUS_BYTES:
        _fail("status file exceeds the supported size")
    try:
        raw = json.loads(raw_bytes)
    except (ValueError, RecursionError) as error:
        raise StatusError("status file is malformed") from error
    return _parse_status(raw)


def _number(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _unknown_outcomes(count: int) -> str:
    noun = "outcome" if count == 1 else "outcomes"
    return f"{count} {noun} unknown"


def _cause_reason(cause: Cause) -> str:
    error_label = ERROR_LABELS[cause.error_type] if cause.error_type else None
    if cause.http_status is None:
        return error_label or "cause unknown"
    if cause.error_type in {"http_error", "unknown"}:
        return f"HTTP {cause.http_status}"
    return f"{error_label or 'cause unknown'} (HTTP {cause.http_status})"


def _cause_description(cause: Cause) -> str:
    component = COMPONENT_LABELS[cause.component]
    reason = _cause_reason(cause)

    unit_labels = COUNT_UNIT_LABELS.get(cause.unit)
    if unit_labels is None:
        if cause.attempted is None:
            return f"{component} {reason}; {_number(cause.affected)} affected; attempt count and unit unknown"
        return (
            f"{component} {reason}; {_number(cause.affected)} affected; attempted count {cause.attempted}, unit unknown"
        )
    singular_unit, plural_unit = unit_labels
    if cause.attempted is not None and cause.affected is not None:
        other_outcomes = "; other outcomes unknown" if cause.affected < cause.attempted else ""
        return f"{component} {reason}; {cause.affected}/{cause.attempted} {plural_unit} affected{other_outcomes}"
    if cause.attempted is None:
        affected_unit = singular_unit if cause.affected == 1 else plural_unit
        return f"{component} {reason}; {_number(cause.affected)} {affected_unit} affected; attempt count unknown"
    return f"{component} {reason}; affected {plural_unit} count unknown of {cause.attempted} attempted"


def _models_line(models: Models) -> str:
    if models.attempted is not None and models.succeeded is not None:
        completed = f"{models.succeeded}/{models.attempted} succeeded"
    else:
        completed = f"{_number(models.succeeded)} succeeded of {_number(models.attempted)} attempted"
    details = [completed, f"{_number(models.timed_out)} timed out", f"{_number(models.dropped)} dropped"]
    if models.attempted is not None and models.succeeded is not None and models.dropped is not None:
        unclassified = models.attempted - models.succeeded - models.dropped
        if unclassified:
            details.append(_unknown_outcomes(unclassified))
    elif models.attempted is None or models.succeeded is None or models.dropped is None:
        details.append("some outcomes unknown")
    if models.drop_causes:
        labels = ", ".join(
            f"{cause.replace('_', ' ')}: {_number(count)}" for cause, count in sorted(models.drop_causes.items())
        )
        details.append(f"drop reasons: {labels}")
    return "Models: " + ", ".join(details)


def _publishing_line(label: str, counts: Counts) -> str:
    if counts.attempted is not None and counts.succeeded is not None:
        summary = f"{counts.succeeded}/{counts.attempted} succeeded"
    else:
        summary = f"{_number(counts.succeeded)} succeeded of {_number(counts.attempted)} attempted"
    details = [summary, f"{_number(counts.failed)} failed"]
    if counts.attempted is not None and counts.succeeded is not None and counts.failed is not None:
        unclassified = counts.attempted - counts.succeeded - counts.failed
        if unclassified:
            details.append(_unknown_outcomes(unclassified))
    return f"{label}: " + ", ".join(details)


def _publication_succeeded(status: RunStatus) -> bool:
    return all(
        counts.attempted is not None
        and counts.attempted > 0
        and counts.succeeded == counts.attempted
        and counts.failed == 0
        for counts in (status.forecasts, status.comments)
    )


def _model_issue(status: RunStatus) -> str | None:
    if status.models.timed_out:
        return f"{status.models.timed_out} model timed out"
    if status.models.dropped:
        return f"{status.models.dropped} model dropped"
    if (
        status.models.attempted is None
        or status.models.succeeded is None
        or status.models.dropped is None
        or status.models.timed_out is None
    ):
        return "model outcomes unknown"
    unclassified = status.models.attempted - status.models.succeeded - status.models.dropped
    if unclassified:
        noun = "outcome" if unclassified == 1 else "outcomes"
        return f"{unclassified} model {noun} unknown"
    return None


def _publication_issue(status: RunStatus) -> str | None:
    failed_forecasts = status.forecasts.failed
    failed_comments = status.comments.failed
    if (failed_forecasts or 0) + (failed_comments or 0) == 0:
        return None
    return f"Publication failed; {_number(failed_forecasts)} forecast, {_number(failed_comments)} comments"


def _publication_unknown(status: RunStatus) -> bool:
    for counts in (status.forecasts, status.comments):
        if counts.attempted is None or counts.succeeded is None or counts.failed is None:
            return True
        if counts.attempted != counts.succeeded + counts.failed:
            return True
    return False


def _clean_title(status: RunStatus) -> str:
    if status.models.attempted == 0 and status.forecasts.attempted == 0 and status.comments.attempted == 0:
        title = "No forecasts attempted"
    else:
        title = "Published OK" if _publication_succeeded(status) else "Forecast run clean"
    if status.causes:
        cause = status.causes[0]
        return f"{title}; {COMPONENT_LABELS[cause.component]} {_cause_reason(cause)}"
    return title


def _title(status: RunStatus) -> str:
    if status.outcome == "clean":
        return _clean_title(status)
    if status.outcome == "unknown":
        return "Forecast status unknown"
    publication_issue = _publication_issue(status)
    if publication_issue:
        return publication_issue
    model_issue = _model_issue(status)
    if model_issue and (not status.causes or status.causes[0].component == "models"):
        prefix = "Published OK" if _publication_succeeded(status) else "Degraded"
        if status.outcome == "hard_failure":
            prefix = "Failed"
        return f"{prefix}; {model_issue}"
    if status.causes:
        cause = status.causes[0]
        if status.outcome == "degraded" and _publication_succeeded(status):
            prefix = "Published OK"
        else:
            prefix = "Failed" if status.outcome == "hard_failure" else "Degraded"
        return f"{prefix}; {COMPONENT_LABELS[cause.component]} {_cause_reason(cause)}"
    if _publication_unknown(status):
        return "Degraded; publication status unknown"
    return "Forecast run failed" if status.outcome == "hard_failure" else "Run degraded"


def _summary(status: RunStatus | None, fallback: str | None) -> str:
    if status is None:
        lines = ["### Forecast run: Unknown", f"- Status: {fallback or 'status evidence is unavailable.'}"]
        return "\n".join(lines) + "\n"

    outcome_title = status.outcome.replace("_", " ").title()
    if status.outcome == "clean" and status.causes:
        outcome_title = "Clean under existing alert policy"
    lines = [f"### Forecast run: {outcome_title}"]
    if status.causes:
        lines.extend(f"- Cause: {_cause_description(cause)}." for cause in status.causes)
    else:
        lines.append("- Cause: none recorded.")
    lines.append(f"- {_models_line(status.models)}.")
    lines.append(f"- {_publishing_line('Forecasts', status.forecasts)}.")
    lines.append(f"- {_publishing_line('Comments', status.comments)}.")
    return "\n".join(lines) + "\n"


def _annotation_message(status: RunStatus | None, fallback: str | None) -> str:
    if status is None:
        return fallback or "Forecast status evidence is unavailable."
    details: list[str] = []
    if status.causes:
        details.append(_cause_description(status.causes[0]))
    issue = _model_issue(status)
    if issue:
        details.append(issue)
    issue = _publication_issue(status)
    if issue:
        details.append(issue)
    return "; ".join(details) if details else "Forecast run completed with a non-clean outcome."


def _gh_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _write_outputs(summary: str, title: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not summary_path or not output_path:
        raise OSError("GITHUB_STEP_SUMMARY and GITHUB_OUTPUT must be set")
    with Path(summary_path).open("a", encoding="utf-8") as summary_file:
        summary_file.write(summary)
    safe_title = title[:MAX_TITLE_LENGTH].rstrip()
    with Path(output_path).open("a", encoding="utf-8") as output_file:
        output_file.write(f"title={safe_title}\n")


def _annotation(title: str, message: str) -> None:
    print(f"::error title={_gh_escape(title)}::{_gh_escape(message)}")


def _reconcile_workflow_outcome(status: RunStatus, bot_outcome: str, job_status: str) -> RunStatus:
    if status.outcome == "clean" and (bot_outcome != "success" or job_status in {"failure", "cancelled"}):
        runtime_cause = Cause("runtime", "alerts", 1, None, "unknown", None)
        return replace(status, outcome="hard_failure", causes=(*status.causes, runtime_cause))
    return status


def _fallback_message(bot_outcome: str, job_status: str, setup_outcome: str) -> str:
    parts = ["Status evidence is unavailable."]
    if setup_outcome == "failure":
        parts.append("Setup step failed.")
    if bot_outcome == "failure":
        parts.append("Bot step failed.")
    elif bot_outcome == "skipped":
        parts.append("Bot step was skipped.")
    if job_status == "failure" and bot_outcome not in {"failure", "skipped"}:
        parts.append("Forecast job failed before a status was recorded.")
    elif job_status == "cancelled":
        parts.append("Forecast job was cancelled before a status was recorded.")
    return " ".join(parts)


def _load_status(
    path: Path | None, bot_outcome: str, job_status: str, setup_outcome: str
) -> tuple[RunStatus | None, str | None]:
    try:
        if path is None:
            raise StatusError("status file path is unavailable")
        status = _read_status(path)
    except StatusError:
        return None, _fallback_message(bot_outcome, job_status, setup_outcome)
    if status.outcome == "unknown" and bot_outcome == "success":
        return None, "Status evidence is unavailable despite a successful bot step."
    return _reconcile_workflow_outcome(status, bot_outcome, job_status), None


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        type=Path,
        default=os.environ.get("FORECAST_RUN_STATUS_PATH"),
        help="path to the structured status JSON (defaults to FORECAST_RUN_STATUS_PATH)",
    )
    return parser


def main() -> int:
    args = _argument_parser().parse_args()
    bot_outcome = os.environ.get("BOT_STEP_OUTCOME", "unknown")
    job_status = os.environ.get("FORECAST_JOB_STATUS", "unknown")
    setup_outcome = os.environ.get("PLAYWRIGHT_STEP_OUTCOME", "unknown")
    status, fallback = _load_status(args.status, bot_outcome, job_status, setup_outcome)

    title = _title(status) if status is not None else "Forecast status unknown"
    summary = _summary(status, fallback)
    annotation_needed = status is None or status.outcome != "clean"
    if status is not None:
        annotation_title = f"Forecast run {status.outcome.replace('_', ' ')}"
        annotation_message = _annotation_message(status, fallback)
    else:
        annotation_title = "Forecast status unknown"
        annotation_message = fallback or "Forecast status evidence is unavailable."

    try:
        _write_outputs(summary, title)
    except OSError as error:
        print(f"forecast-run-summary: {error}", file=sys.stderr)
        return 1

    if annotation_needed:
        _annotation(annotation_title, annotation_message)
    if status is None and bot_outcome == "success":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
