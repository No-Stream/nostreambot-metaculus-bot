from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.probes.cross_pollination_bench import cli, estimate, report
from scripts.probes.cross_pollination_bench.manifest import build_contexts, build_manifest, write_manifest
from scripts.probes.cross_pollination_bench.models import BENCH_MODELS
from scripts.probes.cross_pollination_bench.rows import BenchCallRow
from scripts.probes.cross_pollination_bench.schedule import MAIN_RUN
from tests.cross_pollination_bench_fakes import FakeBenchModel, write_fake_round


def test_the_estimate_arithmetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = write_fake_round(tmp_path)
    manifest = build_manifest(fake.perf_path, fake.archive_dir, event_map=None, fetch=fake.fetch)

    contexts = build_contexts(manifest, fake.archive_dir)[:2]
    monkeypatch.setattr(estimate, "count_tokens", lambda model, prompt: 100)
    result = estimate.estimate_run(contexts, BENCH_MODELS[:1], MAIN_RUN, output_tokens=10, answer_tokens=5)
    assert result.n_questions == 2
    assert result.n_calls == 16
    assert result.per_model["haiku"]["calls"] == 16
    assert result.uncached_usd > result.cached_usd


def test_cache_summary_with_no_cached_tokens_and_error_rows() -> None:
    first = BenchCallRow(101, "binary", "haiku", 1, "first", "scored", "x", prompt_tokens=100)
    update = BenchCallRow(
        101, "binary", "haiku", 1, "peers_full", "scored", "x", prompt_tokens=100, cached_tokens=0, cost_usd=0.1
    )
    error = replace(update, question_id=102, status="api_error", cost_usd=None)
    result = report.cache_summary([first, update, error], BENCH_MODELS[:1])
    assert result["haiku"]["cached_share"] == 0.0
    assert result["haiku"]["updates"]["cached_share"] == 0.0


def test_smoke_fails_on_a_repaired_block() -> None:
    first = BenchCallRow(
        101,
        "binary",
        "haiku",
        1,
        "first",
        "scored",
        "x",
        prompt_tokens=100,
        rung="block",
        upstream_provider="Anthropic",
        reply_text="ok",
    )
    update = BenchCallRow(
        101,
        "binary",
        "haiku",
        1,
        "peers_full",
        "scored",
        "x",
        prompt_tokens=100,
        cached_tokens=90,
        cost_usd=0.0001,
        completion_tokens=1,
        rung="repair",
        upstream_provider="Anthropic",
        reply_text="ok",
    )
    checks = report.smoke_checks([first, update], BENCH_MODELS[:1])
    assert any(not check["passed"] and "rung" in check["check"] for check in checks)


def test_default_caps_are_twenty_and_one() -> None:
    assert cli.parse_args(["--run-dir", "run"]).max_spend_usd == 20.0
    assert cli.parse_args(["--run-dir", "run", "--smoke"]).max_spend_usd == 1.0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--run-dir", "run", "--max-spend-usd", "nan"],
        ["--run-dir", "run", "--question-concurrency", "0"],
        ["--dry-run"],
        ["--build-manifest"],
    ],
)
def test_arguments_that_would_disable_the_run_are_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(arguments)


def _fake_manifest(tmp_path: Path):
    fake = write_fake_round(tmp_path)
    manifest = build_manifest(fake.perf_path, fake.archive_dir, event_map=None, fetch=fake.fetch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_manifest(manifest, run_dir / "manifest.json")
    return fake, manifest, run_dir


def _patch_cli(monkeypatch: pytest.MonkeyPatch, fake_models: dict[str, FakeBenchModel]) -> None:
    def dummy_keys():
        return object()

    monkeypatch.setattr(cli, "split_key_environment", dummy_keys)
    monkeypatch.setattr(cli, "install_role_spend_tracker", lambda: None)

    async def drain() -> None:
        return None

    monkeypatch.setattr(cli, "drain_litellm_callbacks", drain)
    monkeypatch.setattr(cli, "build_parser_llm", lambda model: MagicMock(model=model))
    monkeypatch.setattr(cli, "build_bench_model_call", lambda model, keys: fake_models[model.label])
    monkeypatch.setattr(estimate, "count_tokens", lambda model, prompt: 100)


def test_dry_run_prints_the_plan_and_calls_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    fake, _, run_dir = _fake_manifest(tmp_path)
    calls = {model.label: FakeBenchModel(model.label) for model in BENCH_MODELS}
    _patch_cli(monkeypatch, calls)
    monkeypatch.setattr(cli, "fetch_question", fake.fetch)
    assert cli.main(["--run-dir", str(run_dir), "--dry-run"]) == 0
    assert "144 calls" in capsys.readouterr().out
    assert not any(call.counter for call in calls.values())
    assert cli.main(["--run-dir", str(run_dir), "--dry-run", "--smoke"]) == 0
    assert "48 calls" in capsys.readouterr().out


def test_a_bare_invocation_prints_the_estimate_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _, _, run_dir = _fake_manifest(tmp_path)
    calls = {model.label: FakeBenchModel(model.label) for model in BENCH_MODELS}
    _patch_cli(monkeypatch, calls)
    monkeypatch.setattr(cli, "split_key_environment", lambda: pytest.fail("key split must not run"))
    assert cli.main(["--run-dir", str(run_dir)]) == 2
    output = capsys.readouterr().out
    assert "144 calls" in output
    assert "Refusing" in output


def test_end_to_end_with_a_fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _manifest, run_dir = _fake_manifest(tmp_path)
    calls = {model.label: FakeBenchModel(model.label) for model in BENCH_MODELS}
    order: list[str] = []
    _patch_cli(monkeypatch, calls)
    monkeypatch.setattr(cli, "split_key_environment", lambda: order.append("keys") or object())

    def build_call(model, keys):
        order.append("client")
        return calls[model.label]

    monkeypatch.setattr(cli, "build_bench_model_call", build_call)
    monkeypatch.setattr(cli, "fetch_question", fake.fetch)
    monkeypatch.setattr(estimate, "count_tokens", lambda model, prompt: 100)
    assert cli.main(["--run-dir", str(run_dir), "--i-accept-spend"]) == 0
    rows = __import__("scripts.probes.cross_pollination_bench.rows", fromlist=["load_rows"]).load_rows(
        run_dir / "calls.jsonl"
    )
    assert len(rows) == 144
    assert all(row.status == "scored" for row in rows)
    assert order[0] == "keys"
    assert json.loads((run_dir / "run.json").read_text())["measured_usd"] == pytest.approx(0.144)
    results = json.loads((run_dir / "results.json").read_text())
    assert len(results["contrasts"]) == 6
    assert {"movement", "spread", "member_changes", "corrections", "production_proxy"} <= results.keys()
    assert all(
        {"likelihood_ratio_binary_and_mc", "by_type", "by_first_spread"} <= values.keys()
        for values in results["contrasts"].values()
    )
    assert (run_dir / "paired_rows.jsonl").exists()
    summary = (run_dir / "SUMMARY.md").read_text()
    assert summary.startswith("# Cross-pollination bench")
    assert "specific to three members" in summary
    before = sum(call.counter for call in calls.values())
    assert cli.main(["--run-dir", str(run_dir), "--i-accept-spend"]) == 0
    assert sum(call.counter for call in calls.values()) == before
    original = (run_dir / "results.json").read_bytes()
    assert cli.main(["--run-dir", str(run_dir), "--rescore"]) == 0
    assert (run_dir / "results.json").read_bytes() == original


def _run_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cached_share: float = 0.9,
    repair_only_block: bool = False,
    upstream_provider: str | None = None,
    double_deepseek: bool = False,
    fixed_charge: bool = False,
):
    fake, _, run_dir = _fake_manifest(tmp_path)
    calls = {
        model.label: FakeBenchModel(
            model.label,
            price_from=None if fixed_charge else model,
            charged_usd=0.001,
            price_multiplier=2.0 if double_deepseek and model.label == "deepseek" else 1.0,
            cached_share=cached_share,
            repair_only_block=repair_only_block,
            upstream_provider=upstream_provider if model.label == "haiku" and upstream_provider else None,
        )
        for model in BENCH_MODELS
    }
    _patch_cli(monkeypatch, calls)
    monkeypatch.setattr(cli, "fetch_question", fake.fetch)
    result = cli.main(["--run-dir", str(run_dir), "--smoke", "--i-accept-spend"])
    return result, run_dir, calls


def test_smoke_passes_with_cached_replies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, run_dir, _ = _run_smoke(tmp_path, monkeypatch)
    assert result == 0
    from scripts.probes.cross_pollination_bench.rows import load_rows

    assert len(load_rows(run_dir / "smoke" / "calls.jsonl")) == 48


def test_smoke_fails_without_cached_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    result, _, _ = _run_smoke(tmp_path, monkeypatch, cached_share=0.0)
    assert result == 1
    assert "haiku" in capsys.readouterr().out


def test_smoke_fails_on_a_repaired_block_from_the_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _, _ = _run_smoke(tmp_path, monkeypatch, repair_only_block=True)
    assert result == 1


def test_smoke_fails_on_the_wrong_upstream_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _, _ = _run_smoke(tmp_path, monkeypatch, upstream_provider="Other")
    assert result == 1


def test_smoke_cost_check_accepts_the_deepseek_double_price(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _, _ = _run_smoke(tmp_path, monkeypatch, double_deepseek=True)
    assert result == 0


def test_smoke_restates_the_main_run_estimate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "smoke_checks", lambda rows, models: [])
    result, run_dir, _ = _run_smoke(tmp_path, monkeypatch, fixed_charge=True)
    assert result == 0
    results = json.loads((run_dir / "smoke" / "results.json").read_text())
    assert results["restated_main_run_usd"] == pytest.approx(0.144)


def test_an_estimate_over_the_cap_aborts_before_the_first_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, run_dir = _fake_manifest(tmp_path)
    calls = {model.label: FakeBenchModel(model.label) for model in BENCH_MODELS}
    _patch_cli(monkeypatch, calls)
    monkeypatch.setattr(cli, "split_key_environment", lambda: pytest.fail("no key access above estimate cap"))
    assert cli.main(["--run-dir", str(run_dir), "--max-spend-usd", "0.0000001", "--i-accept-spend"]) == 2
    assert not any(call.counter for call in calls.values())


def test_build_manifest_writes_a_run_dir_and_prints_the_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    fake = write_fake_round(tmp_path)
    monkeypatch.setattr(cli, "fetch_question", fake.fetch)
    assert (
        cli.main(
            [
                "--build-manifest",
                "--perf-json",
                str(fake.perf_path),
                "--archive-dir",
                str(fake.archive_dir),
                "--out-root",
                str(tmp_path / "outputs"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "same title" in output
    assert "101" in output
    assert "105" in output
    run_dirs = list((tmp_path / "outputs").glob("cross_pollination_bench_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.json").exists()
