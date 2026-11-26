"""Frozen, reproducible inputs for cross-pollination bench calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from forecasting_tools import BinaryQuestion, MetaculusApi, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, MetaculusQuestion

from metaculus_bot.backtest.question_prep import _prepare_question_for_backtest
from metaculus_bot.performance_analysis.id_mapping import record_platform
from metaculus_bot.prompts import forecasting_window_at
from scripts.probes.bench_common.prompting import prompt_clock, render_prompt
from scripts.probes.bench_common.records import typed_resolution
from scripts.probes.bench_common.scoring import Resolution

LEAK_CUTOFF_DATE = date(2026, 9, 17)
NEWEST_MODEL_RELEASE_DATE = date(2026, 10, 17)
FALL_2026_EVENT_MAP: dict[int, str] = {
    46184: "brazil-2026-presidential-election",
    46185: "brazil-2026-presidential-election",
    45920: "brazil-2026-presidential-election",
    46423: "alaska-2026-elections",
    46426: "alaska-2026-elections",
}


@dataclass(frozen=True)
class ManifestQuestion:
    question_id: int
    post_id: int
    qtype: str
    title: str
    event_key: str
    archive_timestamp: str
    after_release: bool
    n_options: int | None
    resolution_raw: str
    resolution_parsed: bool | float | str
    api_resolution_string: str | None
    resolution_criteria: str
    fine_print: str
    question_json: dict[str, Any]
    research_sha256: str
    research_chars: int
    production_forecast: dict[str, Any]
    production_spot_peer_score: float | None


@dataclass
class Manifest:
    created_utc: str
    perf_json: str
    archive_dir: str
    leak_cutoff_date: str
    newest_model_release_date: str
    questions: list[ManifestQuestion]


@dataclass(frozen=True)
class QuestionContext:
    entry: ManifestQuestion
    question: MetaculusQuestion
    resolution: Resolution
    prompt: str


QuestionFetch = Callable[[int, int], MetaculusQuestion | list[MetaculusQuestion]]


def default_event_key(title: str) -> str:
    return " ".join(title.casefold().replace("\u2019", "'").split())


def assign_event_keys(titles_by_question_id: Mapping[int, str], event_map: Mapping[int, str] | None) -> dict[int, str]:
    selected_event_map = (
        {qid: key for qid, key in FALL_2026_EVENT_MAP.items() if qid in titles_by_question_id}
        if event_map is None
        else dict(event_map)
    )
    unknown_ids = set(selected_event_map) - set(titles_by_question_id)
    if unknown_ids:
        raise ValueError(f"event map contains question ids not in selection: {sorted(unknown_ids)}")
    return {qid: selected_event_map.get(qid, default_event_key(title)) for qid, title in titles_by_question_id.items()}


def _timestamp_date(timestamp: str) -> date:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"archive timestamp must include a timezone: {timestamp!r}")
    return parsed.astimezone(UTC).date()


def select_records(
    perf_rows: list[dict[str, Any]], archive_dir: Path, *, cutoff: date = LEAK_CUTOFF_DATE
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in perf_rows:
        if row["platform"] != "metaculus" or row["resolution_raw"] is None:
            continue
        question_id = int(row["question_id"])
        archive_path = archive_dir / f"{question_id}.json"
        if not archive_path.exists():
            continue
        archive = json.loads(archive_path.read_text(encoding="utf-8"))
        if _timestamp_date(archive["timestamp"]) <= cutoff:
            continue
        if record_platform(archive) != row["platform"] or int(archive["post_id"]) != int(row["post_id"]):
            raise ValueError(f"question {question_id}: archive platform or post_id differs from dataset row")
        selected.append((row, archive))
    return sorted(selected, key=lambda pair: int(pair[0]["question_id"]))


def fetch_question(post_id: int, question_id: int) -> MetaculusQuestion:
    fetched = cast(
        MetaculusQuestion | list[MetaculusQuestion],
        MetaculusApi.get_question_by_post_id(post_id, group_question_mode="unpack_subquestions"),
    )
    questions = fetched if isinstance(fetched, list) else [fetched]
    for question in questions:
        if question.id_of_question == question_id:
            return question
    raise ValueError(f"post {post_id} did not contain question {question_id}")


def prepare_question(question: MetaculusQuestion) -> MetaculusQuestion:
    prepared = _prepare_question_for_backtest(question)
    prepared.api_json = {"question": {}}
    return prepared


def _question_type(question: MetaculusQuestion) -> str:
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    if isinstance(question, DiscreteQuestion):
        return "discrete"
    if isinstance(question, NumericQuestion):
        return "numeric"
    raise TypeError(f"unsupported question class {type(question).__name__}")


def _read_perf_rows(perf_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(perf_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"performance dataset {perf_path} must contain a JSON array")
    return rows


def build_manifest(
    perf_path: Path,
    archive_dir: Path,
    *,
    event_map: dict[int, str] | None,
    fetch: QuestionFetch,
) -> Manifest:
    selected = select_records(_read_perf_rows(perf_path), archive_dir)
    event_keys = assign_event_keys({int(row["question_id"]): row["title"] for row, _ in selected}, event_map)
    entries: list[ManifestQuestion] = []
    for row, archive in selected:
        question_id = int(row["question_id"])
        fetched = cast(MetaculusQuestion | list[MetaculusQuestion], fetch(int(row["post_id"]), question_id))
        fetched_questions = fetched if isinstance(fetched, list) else [fetched]
        matching_questions = [question for question in fetched_questions if question.id_of_question == question_id]
        if len(matching_questions) != 1:
            raise ValueError(f"post {row['post_id']} did not contain exactly one question {question_id}")
        raw_question = matching_questions[0]
        actual_type = _question_type(raw_question)
        if actual_type != row["type"]:
            raise ValueError(
                f"question {question_id}: dataset type {row['type']!r} differs from API type {actual_type!r}"
            )
        api_resolution_string = raw_question.resolution_string
        question = prepare_question(raw_question)
        research = archive["research_text"]
        entries.append(
            ManifestQuestion(
                question_id=question_id,
                post_id=int(row["post_id"]),
                qtype=row["type"],
                title=row["title"],
                event_key=event_keys[question_id],
                archive_timestamp=archive["timestamp"],
                after_release=_timestamp_date(archive["timestamp"]) > NEWEST_MODEL_RELEASE_DATE,
                n_options=len(question.options) if isinstance(question, MultipleChoiceQuestion) else None,
                resolution_raw=str(row["resolution_raw"]),
                resolution_parsed=row["resolution_parsed"],
                api_resolution_string=api_resolution_string,
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                question_json=question.to_json(),
                research_sha256=hashlib.sha256(research.encode("utf-8")).hexdigest(),
                research_chars=len(research),
                production_forecast={
                    "prob_yes": row["our_prob_yes"],
                    "forecast_values": row["our_forecast_values"],
                },
                production_spot_peer_score=(
                    float(row["metaculus_scores"]["spot_peer_score"])
                    if row["metaculus_scores"].get("spot_peer_score") is not None
                    else None
                ),
            )
        )
    return Manifest(
        created_utc=datetime.now(UTC).isoformat(),
        perf_json=str(perf_path),
        archive_dir=str(archive_dir),
        leak_cutoff_date=LEAK_CUTOFF_DATE.isoformat(),
        newest_model_release_date=NEWEST_MODEL_RELEASE_DATE.isoformat(),
        questions=entries,
    )


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        created_utc=payload["created_utc"],
        perf_json=payload["perf_json"],
        archive_dir=payload["archive_dir"],
        leak_cutoff_date=payload["leak_cutoff_date"],
        newest_model_release_date=payload["newest_model_release_date"],
        questions=[ManifestQuestion(**entry) for entry in payload["questions"]],
    )


def rebuild_question(entry: ManifestQuestion) -> MetaculusQuestion:
    question_class: type[MetaculusQuestion] = {
        "binary": BinaryQuestion,
        "multiple_choice": MultipleChoiceQuestion,
        "numeric": NumericQuestion,
        "discrete": DiscreteQuestion,
    }[entry.qtype]
    return question_class.from_json(entry.question_json)


def bench_resolution(entry: ManifestQuestion, question: MetaculusQuestion) -> Resolution:
    return typed_resolution({"question_id": entry.question_id, "resolution_parsed": entry.resolution_parsed}, question)


def research_text(entry: ManifestQuestion, archive_dir: Path) -> str:
    archive = json.loads((archive_dir / f"{entry.question_id}.json").read_text(encoding="utf-8"))
    research = archive["research_text"]
    digest = hashlib.sha256(research.encode("utf-8")).hexdigest()
    if digest != entry.research_sha256:
        raise ValueError(f"question {entry.question_id}: archived research changed since the manifest was built")
    return research


def build_contexts(manifest: Manifest, archive_dir: Path) -> list[QuestionContext]:
    contexts: list[QuestionContext] = []
    for entry in manifest.questions:
        question = rebuild_question(entry)
        timestamp = datetime.fromisoformat(entry.archive_timestamp.replace("Z", "+00:00"))
        research = research_text(entry, archive_dir)
        with prompt_clock(
            lambda current_question, archive_time=timestamp: forecasting_window_at(current_question, archive_time),
            timestamp.astimezone(UTC).strftime("%Y-%m-%d"),
        ):
            prompt = render_prompt(question, research)
        contexts.append(QuestionContext(entry, question, bench_resolution(entry, question), prompt))
    return contexts
