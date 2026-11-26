"""Command line entry point for the offline cross-pollination bench."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from metaculus_bot.credit_telemetry import drain_litellm_callbacks, install_role_spend_tracker
from scripts.probes.bench_common.cli_support import positive_finite_float, positive_int, run_logging
from scripts.probes.bench_common.paid_call import SpendMeter, build_parser_llm, parser_ledger_usd
from scripts.probes.cross_pollination_bench.analysis import load_paired_rows, run_analysis
from scripts.probes.cross_pollination_bench.estimate import RunEstimate, estimate_run, print_plan
from scripts.probes.cross_pollination_bench.manifest import (
    Manifest,
    QuestionContext,
    build_contexts,
    build_manifest,
    fetch_question,
    load_manifest,
    write_manifest,
)
from scripts.probes.cross_pollination_bench.models import BENCH_MODELS
from scripts.probes.cross_pollination_bench.report import write_analysis_outputs, write_outputs
from scripts.probes.cross_pollination_bench.request import build_bench_model_call, split_key_environment
from scripts.probes.cross_pollination_bench.rows import BenchCallRow, append_row, load_rows, saved_replies
from scripts.probes.cross_pollination_bench.schedule import MAIN_RUN, SMOKE_RUN, RunConfig, run_questions

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "backtests" / "research_archive" / "latest"
DEFAULT_OUT_ROOT = REPO_ROOT / "scratch" / "probes"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline cross-pollination bench; paid calls require explicit approval."
    )
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--perf-json", type=Path)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--event-map", type=Path, help="JSON object mapping question ids to event labels")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-accept-spend", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--mcmc-draws", type=positive_int, default=2000)
    parser.add_argument("--mcmc-tune", type=positive_int, default=2000)
    parser.add_argument("--mcmc-chains", type=positive_int, default=4)
    parser.add_argument("--mcmc-seed", type=int, default=20261125)
    parser.add_argument("--allow-diagnostics-warnings", action="store_true")
    parser.add_argument("--max-spend-usd", type=positive_finite_float)
    parser.add_argument("--output-tokens", type=positive_int, default=6000)
    parser.add_argument("--answer-tokens", type=positive_int, default=1500)
    parser.add_argument("--questions", type=positive_int, nargs="+")
    parser.add_argument("--question-concurrency", type=positive_int, default=1)
    parser.add_argument("--peer-order-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_spend_usd is None:
        args.max_spend_usd = 1.0 if args.smoke else 20.0
    if args.build_manifest and args.perf_json is None:
        parser.error("--build-manifest requires --perf-json")
    if args.build_manifest and (args.run_dir or args.rescore or args.i_accept_spend):
        parser.error("--build-manifest cannot be combined with run modes")
    if not args.build_manifest and args.run_dir is None:
        parser.error("run modes require --run-dir")
    if args.rescore and (args.dry_run or args.i_accept_spend or args.build_manifest):
        parser.error("--rescore cannot be combined with run modes")
    incompatible_analysis_mode = any((args.rescore, args.dry_run, args.i_accept_spend, args.build_manifest, args.smoke))
    if args.analyze and incompatible_analysis_mode:
        parser.error("--analyze cannot be combined with other run modes")
    if args.dry_run and args.i_accept_spend:
        parser.error("--dry-run and --i-accept-spend are mutually exclusive")
    if args.questions and args.build_manifest:
        parser.error("--questions is available only when running a frozen manifest")
    return args


def _read_event_map(path: Path | None) -> dict[int, str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("event map must be a JSON object")
    return {int(question_id): str(event) for question_id, event in payload.items()}


def _new_run_dir(root: Path) -> Path:
    path = root / f"cross_pollination_bench_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _select_contexts(
    contexts: list[QuestionContext], args: argparse.Namespace, config: RunConfig
) -> list[QuestionContext]:
    if args.questions:
        requested = set(args.questions)
        selected = [context for context in contexts if context.entry.question_id in requested]
        missing = requested - {context.entry.question_id for context in selected}
        if missing:
            raise ValueError(f"questions are absent from manifest: {sorted(missing)}")
        return selected
    if args.smoke:
        first_by_type: dict[str, QuestionContext] = {}
        for context in sorted(contexts, key=lambda item: item.entry.question_id):
            first_by_type.setdefault(context.entry.qtype, context)
        return list(first_by_type.values())
    return contexts


def _manifest_mode(args: argparse.Namespace) -> int:
    archive_dir = args.archive_dir or DEFAULT_ARCHIVE_DIR
    manifest = build_manifest(
        args.perf_json, archive_dir, event_map=_read_event_map(args.event_map), fetch=fetch_question
    )
    run_dir = _new_run_dir(args.out_root)
    write_manifest(manifest, run_dir / "manifest.json")
    print(
        f"Selected {len(manifest.questions)} questions; after release: {sum(q.after_release for q in manifest.questions)}"
    )
    print(f"By type: {dict(Counter(question.qtype for question in manifest.questions))}")
    event_questions: dict[str, list[int]] = {}
    for question in manifest.questions:
        event_questions.setdefault(question.event_key, []).append(question.question_id)
    for event, question_ids in sorted(event_questions.items()):
        if len(question_ids) > 1:
            print(f"Shared event {event}, questions {sorted(question_ids)}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Run directory: {run_dir}")
    return 0


def _run_meta(
    estimate: RunEstimate, models: tuple[Any, ...], config: RunConfig, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "models": [
            {
                "label": model.label,
                "slug": model.slug,
                "provider_pin": model.provider_only,
                "upstream_provider": model.upstream_provider,
                "key_route": model.key_route,
            }
            for model in models
        ],
        "config": {"draws": config.draws, "variants": list(config.variants), "peer_order_seed": args.peer_order_seed},
        "estimate": estimate.as_dict(),
        "started_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
    }


@dataclass(frozen=True)
class RunInputs:
    contexts: list[QuestionContext]
    manifest: Manifest
    estimate: RunEstimate
    run_dir: Path
    args: argparse.Namespace
    config: RunConfig
    event_map: dict[int, str] | None


def _latest_rows(previous: list[BenchCallRow], current: list[BenchCallRow]) -> list[BenchCallRow]:
    return list({(row.question_id, row.model, row.draw, row.stage): row for row in [*previous, *current]}.values())


def _run_totals(rows: list[BenchCallRow]) -> tuple[dict[str, float], dict[str, float]]:
    model_totals: dict[str, float] = {}
    key_totals: dict[str, float] = {}
    for row in rows:
        if row.cost_usd is not None:
            model_totals[row.model] = model_totals.get(row.model, 0.0) + row.cost_usd
            key_totals[row.key_alias] = key_totals.get(row.key_alias, 0.0) + row.cost_usd
    return model_totals, key_totals


def _finalize_run(run: RunInputs, old_rows: list[BenchCallRow], rows: list[BenchCallRow], meter: SpendMeter) -> int:
    latest = _latest_rows(old_rows, rows)
    parser_usd = parser_ledger_usd()
    dollars_by_model, dollars_by_key = _run_totals(latest)
    run_meta = _run_meta(run.estimate, BENCH_MODELS, run.config, run.args)
    run_meta.update(
        {
            "saved_replies_reused": len(saved_replies(old_rows)),
            "finished_utc": datetime.now(UTC).isoformat(),
            "dollars_by_model": dollars_by_model,
            "dollars_by_key_alias": dollars_by_key,
            "parser_usd": parser_usd,
            "measured_usd": sum(dollars_by_model.values()) + parser_usd,
            "unpriced_calls": sum(row.reply_text is not None and row.cost_usd is None for row in latest),
            "stop_reason": meter.stop_reason,
        }
    )
    (run.run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    write_outputs(run.run_dir, latest, run.manifest, run_meta, event_map=run.event_map)
    smoke_checks = json.loads((run.run_dir / "results.json").read_text()).get("smoke_checks", [])
    failed_smoke = run.args.smoke and any(not check["passed"] for check in smoke_checks)
    for check in smoke_checks:
        if not check["passed"]:
            print(f"Smoke check failed for {check['model']}: {check['check']} (question {check['question_id']}).")
    return 1 if failed_smoke else 0


async def run_bench(
    run: RunInputs,
) -> int:
    keys = split_key_environment()
    install_role_spend_tracker()
    calls = {model.label: build_bench_model_call(model, keys) for model in BENCH_MODELS}
    parsers = {model.label: build_parser_llm(model.slug) for model in BENCH_MODELS}
    calls_path = run.run_dir / "calls.jsonl"
    old_rows = load_rows(calls_path) if calls_path.exists() else []
    saved = saved_replies(old_rows)
    print(f"Reusing {len(saved)} saved replies.")
    meter = SpendMeter(run.args.max_spend_usd)
    with run_logging(run.run_dir):
        rows = await run_questions(
            run.contexts,
            models=BENCH_MODELS,
            calls=calls,
            parsers=parsers,
            saved=saved,
            meter=meter,
            config=run.config,
            on_row=lambda row: append_row(calls_path, row),
            question_concurrency=run.args.question_concurrency,
        )
        await drain_litellm_callbacks()
    return _finalize_run(run, old_rows, rows, meter)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_manifest:
        return _manifest_mode(args)
    root = args.run_dir
    manifest_path = root / "manifest.json"
    if args.analyze:
        paired_by_contrast = load_paired_rows(root / "paired_rows.jsonl")
        result = run_analysis(
            paired_by_contrast,
            draws=args.mcmc_draws,
            tune=args.mcmc_tune,
            chains=args.mcmc_chains,
            seed=args.mcmc_seed,
            allow_diagnostics_warnings=args.allow_diagnostics_warnings,
        )
        write_analysis_outputs(root, result)
        print(f"Analyzed paired rows in {root}")
        return 0
    if args.rescore:
        manifest = load_manifest(manifest_path)
        rows = load_rows(root / "calls.jsonl")
        run_meta = json.loads((root / "run.json").read_text(encoding="utf-8"))
        write_outputs(root, rows, manifest, run_meta, event_map=_read_event_map(args.event_map))
        print(f"Rescored {len(rows)} calls in {root}")
        return 0
    manifest = load_manifest(manifest_path)
    contexts = build_contexts(manifest, args.archive_dir or Path(manifest.archive_dir))
    config = SMOKE_RUN if args.smoke else MAIN_RUN
    if args.peer_order_seed != config.peer_order_seed:
        config = RunConfig(config.draws, config.variants, args.peer_order_seed, config.full_answers_first)
    contexts = _select_contexts(contexts, args, config)
    estimate = estimate_run(
        contexts, BENCH_MODELS, config, output_tokens=args.output_tokens, answer_tokens=args.answer_tokens
    )
    print_plan(contexts, manifest, estimate, cap_usd=args.max_spend_usd, smoke=args.smoke)
    if args.dry_run:
        return 0
    if not args.i_accept_spend:
        print("Refusing to run: add --i-accept-spend after reviewing the estimate.")
        return 2
    if estimate.uncached_usd > args.max_spend_usd:
        print(
            f"Aborting before the first call: uncached estimate ${estimate.uncached_usd:.2f} exceeds cap ${args.max_spend_usd:.2f}."
        )
        return 2
    run_dir = root / "smoke" if args.smoke else root
    if args.smoke:
        write_manifest(manifest, run_dir / "manifest.json")
    request = RunInputs(contexts, manifest, estimate, run_dir, args, config, _read_event_map(args.event_map))
    return asyncio.run(run_bench(request))
