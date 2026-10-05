from __future__ import annotations

import json
from dataclasses import fields

from metaculus_bot.degradation_counters import DegradationSnapshot
from metaculus_bot.run_status import COUNT_UNITS, RunStatus


def test_run_status_serializes_counts_by_unit_without_error_text(tmp_path) -> None:
    status = RunStatus()
    status.record_source_lookup_attempt("polymarket", count=3)
    status.record_source_http_failure("polymarket", http_status=403)
    status.record_source_check("polymarket", lost=True, error_type="provider_failure")
    status.record_source_check("polymarket", lost=False, error_type="unknown")
    status.record_model_attempt(3)
    status.record_model_success(2)
    status.record_model_drop("timeout_soft_deadline")
    status.record_publish_attempt("forecasts")
    status.record_publish_success("forecasts")
    status.record_publish_attempt("comments")
    status.record_publish_failure("comments", error_type="http_error", http_status=503)
    status.set_outcome("degraded")

    destination = tmp_path / "run_logs" / "status.json"
    status.write(destination)
    payload = json.loads(destination.read_text())

    assert payload == {
        "schema_version": 1,
        "outcome": "degraded",
        "causes": [
            {
                "component": "models",
                "unit": "models",
                "affected": 1,
                "attempted": 3,
                "error_type": "forecast_drop",
                "http_status": None,
            },
            {
                "component": "polymarket",
                "unit": "lookups",
                "affected": 1,
                "attempted": 3,
                "error_type": "http_error",
                "http_status": 403,
            },
            {
                "component": "polymarket",
                "unit": "source_checks",
                "affected": 1,
                "attempted": 2,
                "error_type": "provider_failure",
                "http_status": None,
            },
            {
                "component": "publishing",
                "unit": "publications",
                "affected": 1,
                "attempted": 2,
                "error_type": "http_error",
                "http_status": 503,
            },
        ],
        "models": {
            "attempted": 3,
            "succeeded": 2,
            "timed_out": 1,
            "dropped": 1,
            "drop_causes": {"timeout_soft_deadline": 1},
        },
        "publishing": {
            "forecasts": {"attempted": 1, "succeeded": 1, "failed": 0},
            "comments": {"attempted": 1, "succeeded": 0, "failed": 1},
        },
    }
    assert "error message" not in destination.read_text()


def test_status_count_units_are_explicit_and_stable() -> None:
    assert frozenset({"lookups", "source_checks", "provider_calls", "models", "publications", "alerts"}) == COUNT_UNITS


def test_unobserved_outcome_and_counts_remain_unknown() -> None:
    payload = RunStatus().payload()

    assert payload["outcome"] == "unknown"
    assert payload["causes"] == []
    assert payload["models"] == {
        "attempted": 0,
        "succeeded": 0,
        "timed_out": 0,
        "dropped": 0,
        "drop_causes": {},
    }
    assert payload["publishing"] == {
        "forecasts": {"attempted": 0, "succeeded": 0, "failed": 0},
        "comments": {"attempted": 0, "succeeded": 0, "failed": 0},
    }


def test_unknown_components_and_error_types_are_redacted() -> None:
    status = RunStatus()
    status.record_research_attempt("customer supplied query text")
    status.record_research_failure(
        "customer supplied query text",
        error_type="HTTPError: token=secret query=private",
        http_status=403,
    )

    cause = status.payload()["causes"][0]
    assert cause == {
        "component": "unknown",
        "unit": "provider_calls",
        "affected": 1,
        "attempted": 1,
        "error_type": "unknown",
        "http_status": 403,
    }


def test_invalid_http_status_is_not_emitted() -> None:
    status = RunStatus()
    status.record_research_attempt("polymarket")
    status.record_research_failure("polymarket", error_type="http_error", http_status=9999)

    assert status.payload()["causes"][0]["http_status"] is None


def test_provider_tokens_and_transport_failures_keep_separate_units() -> None:
    status = RunStatus()
    status.record_source_lookup_attempt("predictit")
    status.record_source_http_failure("predictit", http_status=403)

    for _ in range(3):
        status.record_provider_result(
            name="prediction_market",
            status="ok",
            error_type=None,
            details={"sources": {"predictit": "error(all_queries_failed)"}},
        )

    assert status.payload()["causes"] == [
        {
            "component": "predictit",
            "unit": "lookups",
            "affected": 1,
            "attempted": 1,
            "error_type": "http_error",
            "http_status": 403,
        },
        {
            "component": "predictit",
            "unit": "source_checks",
            "affected": 3,
            "attempted": 3,
            "error_type": "provider_failure",
            "http_status": None,
        },
    ]


def test_partial_token_counts_one_question_check_not_embedded_lookup_denominator() -> None:
    status = RunStatus()
    status.record_provider_result(
        name="prediction_market",
        status="ok",
        error_type=None,
        details={"sources": {"polymarket": "partial(12/13)"}},
    )

    assert status.payload()["causes"] == [
        {
            "component": "polymarket",
            "unit": "source_checks",
            "affected": 1,
            "attempted": 1,
            "error_type": "provider_failure",
            "http_status": None,
        }
    ]


def test_malformed_partial_token_is_not_treated_as_a_healthy_check() -> None:
    status = RunStatus()
    status.record_provider_result(
        name="prediction_market",
        status="ok",
        error_type=None,
        details={"sources": {"polymarket": "partial(13/12)"}},
    )

    assert status.payload()["causes"] == [
        {
            "component": "polymarket",
            "unit": "source_checks",
            "affected": 1,
            "attempted": 1,
            "error_type": "unknown",
            "http_status": None,
        }
    ]


def test_free_text_inside_source_token_never_reaches_the_status_payload() -> None:
    status = RunStatus()
    status.record_provider_result(
        name="prediction_market",
        status="ok",
        error_type=None,
        details={"sources": {"polymarket": "error(token=secret query=private text)"}},
    )

    assert status.payload()["causes"] == [
        {
            "component": "polymarket",
            "unit": "source_checks",
            "affected": 1,
            "attempted": 1,
            "error_type": "provider_failure",
            "http_status": None,
        }
    ]
    assert "secret" not in json.dumps(status.payload())
    assert "private text" not in json.dumps(status.payload())


def test_legacy_degradation_cause_has_unknown_denominator_even_with_other_unit() -> None:
    status = RunStatus()
    status.record_research_attempt("gap_fill_v1")
    status.record_legacy_cause("gap_fill_v1", error_type="provider_failure", affected=2)

    assert status.payload()["causes"] == [
        {
            "component": "gap_fill_v1",
            "unit": "alerts",
            "affected": 2,
            "attempted": None,
            "error_type": "provider_failure",
            "http_status": None,
        }
    ]


def test_snapshot_fills_legacy_cause_gaps_without_repeating_source_failures() -> None:
    values = {field.name: 0 for field in fields(DegradationSnapshot)}
    values.update(
        gap_fill_v1_errors=2,
        prediction_market_source_losses=1,
        provider_degradation=1,
        stacker_fallback_used=1,
    )
    snapshot = DegradationSnapshot(**values)
    status = RunStatus()
    status.record_source_http_failure("polymarket", http_status=403)

    status.record_degradation_snapshot(snapshot)

    assert status.payload()["causes"] == [
        {
            "component": "aggregation",
            "unit": "alerts",
            "affected": 1,
            "attempted": None,
            "error_type": "aggregation_failure",
            "http_status": None,
        },
        {
            "component": "gap_fill_v1",
            "unit": "alerts",
            "affected": 2,
            "attempted": None,
            "error_type": "provider_failure",
            "http_status": None,
        },
        {
            "component": "polymarket",
            "unit": "lookups",
            "affected": 1,
            "attempted": None,
            "error_type": "http_error",
            "http_status": 403,
        },
        {
            "component": "provider_health",
            "unit": "alerts",
            "affected": 1,
            "attempted": None,
            "error_type": "provider_failure",
            "http_status": None,
        },
    ]
