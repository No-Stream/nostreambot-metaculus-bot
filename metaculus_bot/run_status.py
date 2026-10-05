"""Privacy-safe, structured evidence for one forecasting run.

This is a second readout of existing run outcomes, not an exit policy. The CLI
continues to own the exit code; this module records observed counts and emits a
small JSON artifact for workflow reporting.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Literal, TypedDict

from metaculus_bot.degradation_counters import DegradationSnapshot
from metaculus_bot.http_status import http_status_from_exception

logger = logging.getLogger(__name__)

RunOutcome = Literal["clean", "degraded", "hard_failure", "unknown"]
PublishKind = Literal["forecasts", "comments"]
CountUnit = Literal["lookups", "source_checks", "provider_calls", "models", "publications", "alerts"]


class RunCause(TypedDict):
    component: str
    unit: CountUnit
    affected: int
    attempted: int | None
    error_type: str
    http_status: int | None


class ModelCounts(TypedDict):
    attempted: int
    succeeded: int
    timed_out: int
    dropped: int
    drop_causes: dict[str, int]


class PublishCounts(TypedDict):
    attempted: int
    succeeded: int
    failed: int


class RunStatusPayload(TypedDict):
    schema_version: int
    outcome: RunOutcome
    causes: list[RunCause]
    models: ModelCounts
    publishing: dict[PublishKind, PublishCounts]


COUNT_UNITS: frozenset[str] = frozenset(
    {"lookups", "source_checks", "provider_calls", "models", "publications", "alerts"}
)

STATUS_COMPONENTS: frozenset[str] = frozenset(
    {
        "setup",
        "runtime",
        "research",
        "models",
        "publishing",
        "unknown",
        "custom",
        "asknews",
        "exa",
        "perplexity",
        "openrouter",
        "native_search",
        "gemini_search",
        "summarizer",
        "financial_data",
        "timeseries_anchor",
        "prediction_market",
        "resolution_source",
        "polymarket",
        "manifold",
        "manifold_detail",
        "predictit",
        "kalshi",
        "kalshi_catalogue",
        "snapshot",
        "query_author",
        "ranking",
        "gap_fill_v1",
        "gap_fill_v2",
        "provider_health",
        "publish_gate",
        "aggregation",
        "credit",
        "tournament",
    }
)
STATUS_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "http_error",
        "timeout",
        "connection_error",
        "authentication_error",
        "provider_failure",
        "forecast_drop",
        "publish_failure",
        "summary_error",
        "time_budget",
        "credit_floor",
        "stale_tournament",
        "configuration_reminder",
        "deprecated_model",
        "aggregation_failure",
        "unknown",
    }
)
STATUS_DROP_CAUSES: frozenset[str] = frozenset(
    {
        "timeout_wall_clock",
        "timeout_soft_deadline",
        "zero_output",
        "parse_extraction",
        "error_other",
    }
)
_HTTP_STATUS_MIN = 100
_HTTP_STATUS_MAX = 599
_STATUS_PATH_ENV = "FORECAST_RUN_STATUS_PATH"
_PARTIAL_SOURCE_TOKEN = re.compile(r"partial\((?P<succeeded>\d+)/(?P<attempted>\d+)\)")
_HTTP_SOURCE_TOKEN = re.compile(r"http_(?P<status>\d{3})\Z")


def _safe_component(component: str) -> str:
    return component if component in STATUS_COMPONENTS else "unknown"


def _safe_error_type(error_type: str) -> str:
    return error_type if error_type in STATUS_ERROR_TYPES else "unknown"


def _safe_http_status(status: int | None) -> int | None:
    if status is None or not _HTTP_STATUS_MIN <= status <= _HTTP_STATUS_MAX:
        return None
    return status


def _error_type_from_name(error_type: str | None) -> str:
    if error_type in {"ApiIdentityError", "AuthenticationError", "Unauthorized", "InvalidToken"}:
        return "authentication_error"
    if error_type in {"Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return "timeout"
    if error_type in {"ConnectError", "ConnectionError", "ClientConnectorError", "ConnectionResetError"}:
        return "connection_error"
    return "provider_failure"


def _source_token_evidence(source_status: str) -> tuple[bool, str, int | None]:
    partial_match = _PARTIAL_SOURCE_TOKEN.fullmatch(source_status)
    if partial_match:
        succeeded = int(partial_match.group("succeeded"))
        attempted = int(partial_match.group("attempted"))
        if succeeded < attempted:
            return True, "provider_failure", None
        return True, "unknown", None
    if source_status == "none" or source_status.startswith("ok("):
        return False, "unknown", None

    reason = (
        source_status[6:-1] if source_status.startswith("error(") and source_status.endswith(")") else source_status
    )
    source_http = _HTTP_SOURCE_TOKEN.fullmatch(reason)
    if source_http:
        return True, "http_error", int(source_http.group("status"))
    if reason in {"timeout", "wall_timeout"}:
        return True, "timeout", None
    return True, _error_type_from_name(reason), None


class RunStatus:
    """Thread-safe accumulator for only allowlisted labels and integer counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self, *, stage: Literal["setup", "runtime"] = "setup") -> None:
        with self._lock:
            self._attempted_by_component: Counter[tuple[str, CountUnit]] = Counter()
            self._failures: Counter[tuple[str, CountUnit, str, int | None]] = Counter()
            self._models_attempted = 0
            self._models_succeeded = 0
            self._drop_causes: Counter[str] = Counter()
            self._publish_attempted: Counter[PublishKind] = Counter()
            self._publish_succeeded: Counter[PublishKind] = Counter()
            self._publish_failed: Counter[PublishKind] = Counter()
            self._outcome: RunOutcome = "unknown"
            self._stage: Literal["setup", "runtime"] = stage

    def set_outcome(self, outcome: RunOutcome) -> None:
        with self._lock:
            self._outcome = outcome

    def set_stage(self, stage: Literal["setup", "runtime"]) -> None:
        with self._lock:
            self._stage = stage

    @property
    def stage(self) -> Literal["setup", "runtime"]:
        with self._lock:
            return self._stage

    def record_research_attempt(self, component: str, count: int = 1, *, unit: CountUnit = "provider_calls") -> None:
        if count < 0:
            raise ValueError("research attempt count cannot be negative")
        safe_component = _safe_component(component)
        with self._lock:
            self._record_attempt(safe_component, unit, count)

    def record_source_lookup_attempt(self, component: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError("source lookup attempt count cannot be negative")
        safe_component = _safe_component(component)
        with self._lock:
            self._record_attempt(safe_component, "lookups", count)

    def _record_attempt(self, component: str, unit: CountUnit, count: int) -> None:
        if count:
            self._attempted_by_component[(component, unit)] += count

    def record_research_failure(
        self,
        component: str,
        *,
        error_type: str,
        http_status: int | None = None,
        affected: int = 1,
        unit: CountUnit = "provider_calls",
    ) -> None:
        if affected < 0:
            raise ValueError("affected count cannot be negative")
        safe_component = _safe_component(component)
        signature = (safe_component, unit, _safe_error_type(error_type), _safe_http_status(http_status))
        with self._lock:
            self._failures[signature] += affected

    def record_source_http_failure(
        self,
        component: str,
        *,
        http_status: int,
        affected: int = 1,
    ) -> None:
        """Record a terminal source HTTP failure at the logical lookup seam."""
        self.record_source_failure(
            component,
            error_type="http_error",
            http_status=http_status,
            affected=affected,
        )

    def record_source_failure(
        self,
        component: str,
        *,
        error_type: str,
        http_status: int | None = None,
        affected: int = 1,
    ) -> None:
        """Record a terminal failure for an instrumented logical source lookup."""
        if affected < 0:
            raise ValueError("affected count cannot be negative")
        safe_component = _safe_component(component)
        signature = (safe_component, "lookups", _safe_error_type(error_type), _safe_http_status(http_status))
        with self._lock:
            self._failures[signature] += affected

    def record_source_check(
        self,
        component: str,
        *,
        lost: bool,
        error_type: str,
        http_status: int | None = None,
    ) -> None:
        """Record one per-question source token, separate from transport lookup counts."""
        safe_component = _safe_component(component)
        with self._lock:
            self._record_attempt(safe_component, "source_checks", 1)
            if lost:
                signature = (
                    safe_component,
                    "source_checks",
                    _safe_error_type(error_type),
                    _safe_http_status(http_status),
                )
                self._failures[signature] += 1

    def record_provider_result(
        self,
        *,
        name: str,
        status: str,
        error_type: str | None,
        details: dict[str, object],
    ) -> None:
        """Consume only allowlisted counts/statuses from one existing provider result."""
        provider = _safe_component(name)
        self.record_research_attempt(provider, unit="provider_calls")
        self._record_provider_status(provider, status, error_type)

        sources = details.get("sources")
        if isinstance(sources, dict):
            for source_name, token in sources.items():
                if isinstance(source_name, str) and isinstance(token, str):
                    self._record_source_token(provider, source_name, token)

    def _record_provider_status(self, provider: str, status: str, error_type: str | None) -> None:
        if status == "deadline":
            self.record_research_failure(provider, error_type="timeout", unit="provider_calls")
        elif status == "errored":
            self.record_research_failure(provider, error_type=_error_type_from_name(error_type), unit="provider_calls")

    def _record_source_token(self, provider: str, source_name: str, token: str) -> None:
        safe_source = _safe_component(source_name)
        component = safe_source if safe_source != "unknown" else provider
        lost, error_type, http_status = _source_token_evidence(token)
        self.record_source_check(
            component,
            lost=lost,
            error_type=error_type,
            http_status=http_status,
        )

    def record_model_attempt(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("model attempt count cannot be negative")
        with self._lock:
            self._models_attempted += count
            self._record_attempt("models", "models", count)

    def record_legacy_cause(
        self,
        component: str,
        *,
        error_type: str,
        affected: int,
        http_status: int | None = None,
    ) -> None:
        """Add a safe summary for a legacy alert counter with no finer event seam."""
        if affected < 0:
            raise ValueError("affected count cannot be negative")
        safe_component = _safe_component(component)
        signature = (
            safe_component,
            "alerts",
            _safe_error_type(error_type),
            _safe_http_status(http_status),
        )
        with self._lock:
            self._failures[signature] += affected

    def record_degradation_snapshot(self, snapshot: DegradationSnapshot) -> None:
        """Fill status gaps from the existing end-of-run counters without duplicating detail."""
        self._record_snapshot_cause(
            snapshot.stacker_primary_failed + snapshot.stacker_fallback_used + snapshot.stacker_fallback_failed,
            component="aggregation",
            error_type="aggregation_failure",
        )
        self._record_snapshot_cause(
            snapshot.gap_fill_v1_errors,
            component="gap_fill_v1",
            error_type="provider_failure",
        )
        self._record_snapshot_cause(
            snapshot.gap_fill_v2_errors,
            component="gap_fill_v2",
            error_type="provider_failure",
        )
        self._record_snapshot_cause(
            snapshot.provider_degradation,
            component="provider_health",
            error_type="provider_failure",
        )
        self._record_snapshot_cause(
            snapshot.questions_failed_to_publish + snapshot.publish_skipped_closed,
            component="publish_gate",
            error_type="publish_failure",
        )
        self._record_snapshot_cause(
            snapshot.time_budget_fast_path + snapshot.research_budget_cuts,
            component="research",
            error_type="time_budget",
        )
        self._record_snapshot_cause(
            snapshot.publish_attempt_failures,
            component="publishing",
            error_type="publish_failure",
        )
        market_source_count = snapshot.prediction_market_degraded + snapshot.prediction_market_source_losses
        if market_source_count and not self._has_market_source_cause():
            self.record_legacy_cause(
                "prediction_market",
                error_type="provider_failure",
                affected=market_source_count,
            )
        if snapshot.research_provider_failures and not self._has_provider_cause():
            self.record_legacy_cause(
                "research",
                error_type="provider_failure",
                affected=snapshot.research_provider_failures,
            )
        if snapshot.summarizer_failures and not self._has_component_cause("summarizer"):
            self.record_legacy_cause(
                "summarizer",
                error_type="summary_error",
                affected=snapshot.summarizer_failures,
            )

    def _record_snapshot_cause(self, affected: int, *, component: str, error_type: str) -> None:
        if affected and not self._has_component_cause(component):
            self.record_legacy_cause(component, error_type=error_type, affected=affected)

    def _has_component_cause(self, component: str) -> bool:
        with self._lock:
            safe_component = _safe_component(component)
            return any(key[0] == safe_component and count for key, count in self._failures.items())

    def _has_provider_cause(self) -> bool:
        with self._lock:
            return any(
                component in {"asknews", "exa", "perplexity", "openrouter", "native_search"} and count
                for (component, _, _, _), count in self._failures.items()
            )

    def _has_market_source_cause(self) -> bool:
        with self._lock:
            return any(
                component in {"polymarket", "manifold", "manifold_detail", "predictit", "kalshi"} and count
                for (component, _, _, _), count in self._failures.items()
            )

    def record_model_success(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("model success count cannot be negative")
        with self._lock:
            self._models_succeeded += count

    def record_model_drop(self, cause: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError("model drop count cannot be negative")
        safe_cause = cause if cause in STATUS_DROP_CAUSES else "error_other"
        with self._lock:
            self._drop_causes[safe_cause] += count
            self._failures[("models", "models", "forecast_drop", None)] += count

    def record_publish_attempt(self, kind: PublishKind) -> None:
        with self._lock:
            self._publish_attempted[kind] += 1
            self._record_attempt("publishing", "publications", 1)

    def record_publish_success(self, kind: PublishKind) -> None:
        with self._lock:
            self._publish_succeeded[kind] += 1

    def record_publish_failure(
        self,
        kind: PublishKind,
        *,
        error_type: str,
        http_status: int | None = None,
    ) -> None:
        with self._lock:
            self._publish_failed[kind] += 1
            signature = (
                "publishing",
                "publications",
                _safe_error_type(error_type),
                _safe_http_status(http_status),
            )
            self._failures[signature] += 1

    def payload(self) -> RunStatusPayload:
        with self._lock:
            causes: list[RunCause] = [
                {
                    "component": component,
                    "unit": unit,
                    "affected": affected,
                    "attempted": self._attempted_by_component.get((component, unit)),
                    "error_type": error_type,
                    "http_status": status,
                }
                for (component, unit, error_type, status), affected in sorted(
                    self._failures.items(),
                    key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3] or 0),
                )
                if affected
            ]
            timed_out = self._drop_causes["timeout_wall_clock"] + self._drop_causes["timeout_soft_deadline"]
            publishing: dict[PublishKind, PublishCounts] = {
                "forecasts": {
                    "attempted": self._publish_attempted["forecasts"],
                    "succeeded": self._publish_succeeded["forecasts"],
                    "failed": self._publish_failed["forecasts"],
                },
                "comments": {
                    "attempted": self._publish_attempted["comments"],
                    "succeeded": self._publish_succeeded["comments"],
                    "failed": self._publish_failed["comments"],
                },
            }
            return {
                "schema_version": 1,
                "outcome": self._outcome,
                "causes": causes,
                "models": {
                    "attempted": self._models_attempted,
                    "succeeded": self._models_succeeded,
                    "timed_out": timed_out,
                    "dropped": sum(self._drop_causes.values()),
                    "drop_causes": dict(sorted(self._drop_causes.items())),
                },
                "publishing": publishing,
            }

    def write(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.payload(), sort_keys=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
        logger.info("RUN_STATUS_JSON: %s", serialized)


RUN_STATUS = RunStatus()


def safe_error_type_from_exception(exc: BaseException) -> str:
    """Reduce an exception to a safe category without reading its message."""
    if http_status_from_exception(exc) is not None:
        return "http_error"

    if isinstance(exc, TimeoutError):
        return "timeout"
    class_name = type(exc).__name__
    if class_name in {
        "ConnectError",
        "ConnectionError",
        "ClientConnectorError",
        "ConnectionResetError",
    }:
        return "connection_error"
    if class_name in {"AuthenticationError", "ApiIdentityError", "Unauthorized", "InvalidToken"}:
        return "authentication_error"
    return "unknown"


def record_cli_failure(exc: BaseException, *, component: str | None = None) -> None:
    failure_component = component or RUN_STATUS.stage
    RUN_STATUS.record_legacy_cause(
        failure_component,
        error_type=safe_error_type_from_exception(exc),
        http_status=http_status_from_exception(exc),
        affected=1,
    )


def write_run_status_from_environment() -> None:
    """Write the status artifact when a workflow configured its destination."""
    destination = os.environ.get(_STATUS_PATH_ENV)
    if not destination:
        return
    RUN_STATUS.write(Path(destination))
