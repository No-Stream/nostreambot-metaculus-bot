"""Public helper tests for archived ghost forecasts."""

import json
from itertools import pairwise

from metaculus_bot.performance_analysis.ghosts import (
    ghost_binary_prob,
    ghost_cdf_on_published_grid,
    ghost_mc_probs,
    ghost_numeric_percentiles,
    load_concluding_ghosts,
    parse_ghost_summary,
    records_by_post_id,
    select_ghost_per_qid,
)


def _json_marker(qid: int, prob: float, run_date: str) -> dict:
    return {
        "qid": qid,
        "run_date": run_date,
        "seq": 0,
        "forecast_json": json.dumps({"qtype": "binary", "prob": prob}),
    }


def test_ghost_values_parse_from_both_marker_shapes() -> None:
    json_ghost = {
        "source": "json",
        "payload": {
            "prob": 0.7,
            "option_probs": {"Blue": 0.3, "Red": 0.7},
            "declared_percentiles": {"0.1": 10, "0.9": 90},
        },
    }
    legacy_ghost = {"source": "legacy", "summary": "posterior_prob=0.4"}

    assert ghost_binary_prob(json_ghost) == 0.7
    assert ghost_binary_prob(legacy_ghost) == 0.4
    assert ghost_mc_probs(json_ghost) == {"Blue": 0.3, "Red": 0.7}
    assert ghost_numeric_percentiles(json_ghost) == {0.1: 10.0, 0.9: 90.0}
    assert parse_ghost_summary("numeric", "median=42.5") == {"median": 42.5}


def test_selection_prefers_latest_json_to_latest_legacy() -> None:
    selected = select_ghost_per_qid(
        [
            _json_marker(17, 0.2, "2026-01-01T00:00:00Z"),
            _json_marker(17, 0.8, "2026-01-02T00:00:00Z"),
        ],
        [
            {"qid": 17, "qtype": "binary", "summary": "posterior_prob=0.1", "run_date": "2026-01-03T00:00:00Z"},
        ],
    )

    assert selected[17]["source"] == "json"
    assert selected[17]["payload"]["prob"] == 0.8


def test_records_are_indexed_by_post_id() -> None:
    record = {"post_id": 17, "question_id": 90017}

    indexed = records_by_post_id([record])

    assert indexed[17] is record
    assert 90017 not in indexed


def test_ghost_cdf_uses_published_length_and_returns_none_for_missing_inputs() -> None:
    record = {
        "resolution_parsed": 50.0,
        "our_forecast_values": [index / 10 for index in range(11)],
        "scaling": {"range_min": 0, "range_max": 100, "zero_point": None},
        "open_lower_bound": False,
        "open_upper_bound": False,
    }

    cdf = ghost_cdf_on_published_grid({0.1: 10, 0.5: 50, 0.9: 90}, record)

    assert cdf is not None
    assert len(cdf) == len(record["our_forecast_values"])
    assert all(left <= right for left, right in pairwise(cdf))
    assert ghost_cdf_on_published_grid({0.1: 10, 0.9: 90}, {**record, "our_forecast_values": None}) is None
    assert (
        ghost_cdf_on_published_grid({0.1: 10, 0.9: 90}, {**record, "scaling": {"range_min": None, "range_max": 100}})
        is None
    )


def test_load_concluding_ghosts_reads_archive_markers(tmp_path) -> None:
    (tmp_path / "ghost_forecast_json.jsonl").write_text(
        json.dumps(_json_marker(21, 0.7, "2026-01-01T00:00:00Z")) + "\n"
    )
    (tmp_path / "ghost_forecast.jsonl").write_text(
        json.dumps({"qid": 22, "qtype": "binary", "summary": "posterior_prob=0.3"}) + "\n"
    )

    ghosts = load_concluding_ghosts(tmp_path)

    assert set(ghosts) == {21, 22}
    assert ghosts[21]["source"] == "json"
    assert ghosts[22]["source"] == "legacy"
