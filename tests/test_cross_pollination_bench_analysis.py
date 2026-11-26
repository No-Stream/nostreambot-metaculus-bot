from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from metaculus_bot.performance_analysis.paired_comparison import PairedScoreRow, compare_paired
from scripts.probes.cross_pollination_bench import analysis, cli
from scripts.probes.cross_pollination_bench import analysis as analysis_module


@pytest.fixture(scope="module")
def real_comparison() -> dict:
    rows = [
        PairedScoreRow(
            f"q-{index // 2}",
            f"event-{index // 4}",
            ("binary", "numeric")[(index // 2) % 2],
            index % 2,
            float(index % 5 - 2),
        )
        for index in range(12)
    ]
    return compare_paired(rows, draws=100, tune=100, chains=2, seed=42, allow_diagnostics_warnings=True)


def _rows() -> dict[str, list[dict]]:
    names = [*analysis.CONTRASTS, analysis.MEMBER_CONTRAST]
    return {
        name: [
            {
                "contrast": name,
                "unit_id": f"q-{index // 2}",
                "cluster_id": f"event-{index // 4}",
                "group": ("binary", "numeric")[(index // 2) % 2],
                "replicate": index % 2,
                "difference": float(index % 5 - 2),
                "member": "haiku" if name == analysis.MEMBER_CONTRAST else None,
                "complete_panel": index % 2 == 0,
                "after_release": index % 3 == 0,
            }
            for index in range(12)
        ]
        for name in names
    }


def test_run_analysis_covers_every_contrast_the_subsets_and_the_members(real_comparison: dict) -> None:
    rows = _rows()
    with patch.object(analysis, "compare_paired", return_value=real_comparison) as compare:
        result = analysis.run_analysis(rows, draws=100, tune=100, chains=2, seed=42, allow_diagnostics_warnings=True)
    assert set(result["contrasts"]) == set(analysis.CONTRASTS)
    assert result["primary_complete_panels"] == real_comparison
    assert result["primary_after_release"] == real_comparison
    assert result["members"] == real_comparison
    assert compare.call_count == len(analysis.CONTRASTS) + 3
    assert len(result["paired_rows_sha256"]) == 64


def _decision(
    *,
    typical: float = 2,
    primary_probability: float = 0.95,
    gain_probability: float = 0.95,
    first: float = 1,
    pooled: float = 1,
    blend: float = 1,
    member: float = 0,
    first_draw_noise: float = 0,
) -> tuple[dict, dict]:
    results = {
        "movement": {
            "peers_full": {"typical_change": typical},
            "second_look": {"typical_change": 1.5},
            "first_draw_noise": first_draw_noise,
        }
    }
    fit = {
        "student_t": {
            "effect": {"mean": 1, "p_above_zero": primary_probability, "p_gain_above": {"2.0": gain_probability}}
        },
        "normal": {"effect": {"mean": 1}},
        "fits_disagree": False,
        "members": {},
    }
    contrasts = {"peers_full_vs_second_look": fit}
    for name, mean in (
        ("peers_full_vs_first", first),
        ("peers_full_vs_pooled_draws", pooled),
        ("peers_full_vs_blend", blend),
    ):
        contrasts[name] = {"student_t": {"effect": {"mean": mean}}}
    return results, {"contrasts": contrasts, "members": {"student_t": {"members": {"haiku": {"mean": member}}}}}


@pytest.mark.parametrize(
    ("typical", "expected"), [(1.5, "stop"), (1.0, "stop"), (2.0, "price_production_confirmation")]
)
def test_decision_stops_when_the_median_moves_no_further_than_under_the_second_look(
    typical: float, expected: str
) -> None:
    results, fits = _decision(typical=typical)
    assert analysis.decide(results, fits)["outcome"] == expected


def test_first_draw_noise_does_not_enter_the_decision() -> None:
    outcomes = [analysis.decide(*_decision(first_draw_noise=value))["outcome"] for value in (0, 1, 1000)]
    assert outcomes == ["price_production_confirmation"] * 3


def test_decision_stops_when_a_two_point_gain_is_unlikely() -> None:
    results, fits = _decision(gain_probability=0.09)
    assert analysis.decide(results, fits)["outcome"] == "stop"


def test_stop_is_checked_before_confirmation() -> None:
    results, fits = _decision(typical=1.5)
    assert analysis.decide(results, fits)["outcome"] == "stop"


def test_decision_prices_a_confirmation_when_all_four_hold() -> None:
    results, fits = _decision()
    assert analysis.decide(results, fits)["outcome"] == "price_production_confirmation"


@pytest.mark.parametrize("failed", ["probability", "first", "pooled", "blend", "member"])
def test_each_failed_condition_alone_is_inconclusive(failed: str) -> None:
    changes = {
        "probability": {"primary_probability": 0.89},
        "first": {"first": -0.1},
        "pooled": {"pooled": -0.1},
        "blend": {"blend": 0},
        "member": {"member": -2.1},
    }
    results, fits = _decision(**changes[failed])
    assert analysis.decide(results, fits)["outcome"] == "inconclusive"


def test_summary_prints_the_three_fits_side_by_side(real_comparison: dict) -> None:
    from scripts.probes.cross_pollination_bench.report import render_markdown

    fits = {**real_comparison, "fits_disagree": False}
    results = {
        "question_set": {"n": 12, "after_release": 4},
        "contrasts": {},
        "paired_rows": {},
        "movement": {"peers_full": {"typical_change": 2}, "second_look": {"typical_change": 1}},
        "failures": {},
        "spend": {},
        "spread": {},
        "member_changes": {},
        "corrections": {},
        "production_proxy": {},
        "analysis": {
            "contrasts": {analysis.PRIMARY_CONTRAST: fits},
            "primary_complete_panels": fits,
            "primary_after_release": fits,
            "members": fits,
        },
        "decision": {"outcome": "inconclusive", "criteria": {"example": True}},
    }
    text = render_markdown(results)
    assert "Student-t model" in text
    assert "Normal model" in text
    assert "Bayesian bootstrap by event" in text
    assert "questions per event and two draws per question, a follow-up needs" in text
    assert "Influential questions" not in text
    disagree = json.loads(json.dumps(fits))
    disagree["fits_disagree"] = True
    results["analysis"]["contrasts"][analysis.PRIMARY_CONTRAST] = disagree
    text = render_markdown(results)
    assert "Influential questions" in text


def test_load_paired_rows_hashes_and_groups(tmp_path: Path) -> None:
    path = tmp_path / "paired_rows.jsonl"
    path.write_text(json.dumps(_rows()[analysis.PRIMARY_CONTRAST][0]) + "\n", encoding="utf-8")
    grouped = analysis.load_paired_rows(path)
    assert analysis.PRIMARY_CONTRAST in grouped
    assert grouped[analysis.PRIMARY_CONTRAST][0]["unit_id"] == "q-0"


def test_analyze_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_comparison: dict) -> None:
    from scripts.probes.cross_pollination_bench.manifest import build_manifest, write_manifest
    from scripts.probes.cross_pollination_bench.models import BENCH_MODELS
    from tests.cross_pollination_bench_fakes import FakeBenchModel, write_fake_round
    from tests.test_cross_pollination_bench_cli import _patch_cli

    fake = write_fake_round(tmp_path)
    manifest = build_manifest(fake.perf_path, fake.archive_dir, event_map=None, fetch=fake.fetch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_manifest(manifest, run_dir / "manifest.json")
    _patch_cli(monkeypatch, {model.label: FakeBenchModel(model.label) for model in BENCH_MODELS})
    monkeypatch.setattr(cli, "fetch_question", fake.fetch)
    assert cli.main(["--run-dir", str(run_dir), "--i-accept-spend"]) == 0
    monkeypatch.setattr(analysis_module, "compare_paired", lambda *args, **kwargs: real_comparison)
    assert cli.main(["--run-dir", str(run_dir), "--analyze"]) == 0
    assert (run_dir / "analysis.json").exists()
    assert json.loads((run_dir / "results.json").read_text())["decision"]["outcome"] in {
        "stop",
        "price_production_confirmation",
        "inconclusive",
    }
    event_map_path = tmp_path / "event-map.json"
    event_map_path.write_text(json.dumps({"101": "changed", "105": "changed"}), encoding="utf-8")
    assert cli.main(["--run-dir", str(run_dir), "--rescore", "--event-map", str(event_map_path)]) == 0
    results = json.loads((run_dir / "results.json").read_text())
    assert results["analysis_stale"] is True
    assert (run_dir / "analysis.json").exists()


def test_analyze_without_paired_rows_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli.main(["--run-dir", str(tmp_path), "--analyze"])
