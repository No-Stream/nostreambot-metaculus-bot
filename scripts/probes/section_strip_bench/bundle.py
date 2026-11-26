"""The bench's inputs: the archived bundle cut into arms, and the question each pair is scored as.

Stripping is exact-header: the bundle is cut at the two section headers, each block keeps its own
leading separator, and every stripped arm is checked to have lost exactly that block. The question
object is rebuilt from the pair record and the tagged dataset's row, so the production prompt
builders and CDF build read the same fields they read on a live question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, MetaculusQuestion

from metaculus_bot.performance_analysis.research_tags import GFV2_SECTION_TITLE
from scripts.probes.bench_common.records import (
    read_jsonl,
    typed_resolution,
)
from scripts.probes.bench_common.records import (
    resolution_label as _resolution_label,
)
from scripts.probes.bench_common.scoring import Resolution, score_published

# The two appended sections as research/gap_fill_stages.run_gap_fill_passes writes them; v2 heads its own (agentic/artifact.render_findings).
V1_SECTION_HEADER = "## Targeted Gap-Fill (second pass)"
V2_SECTION_HEADER = f"## {GFV2_SECTION_TITLE}"
SECTION_SEPARATOR = "\n\n---\n\n"

FULL_ARM = "full"
ARMS: tuple[str, ...] = (FULL_ARM, "minus_v1", "minus_v2", "minus_both")
STRIPPED_ARMS: tuple[str, ...] = ARMS[1:]

_TODAY_LINE = re.compile(r"^Today: (\d{4}-\d{2}-\d{2})$", re.MULTILINE)
_UNITS_LINE = re.compile(r"^Units: (.+)$", re.MULTILINE)
# The driver brief's placeholder for a question with no unit (research/agentic/driver_prompt._question_header).
_UNITLESS_HEADER = "unspecified (assume unitless)"


@dataclass(frozen=True)
class BundleSections:
    """A published bundle cut at its two appended sections; ``first_pass + v1_block + v2_block`` is the bundle."""

    first_pass: str
    v1_block: str
    v2_block: str


def _sole_header_offset(bundle: str, header: str) -> int:
    count = bundle.count(header)
    if count != 1:
        raise ValueError(f"expected exactly one {header!r} in the bundle, found {count}")
    offset = bundle.index(header)
    if bundle[offset - len(SECTION_SEPARATOR) : offset] != SECTION_SEPARATOR:
        raise ValueError(f"{header!r} is not preceded by the section separator")
    return offset


def split_bundle(bundle: str) -> BundleSections:
    """Cut the bundle at the two headers; each block carries its own leading separator and header."""
    v1_at = _sole_header_offset(bundle, V1_SECTION_HEADER)
    v2_at = _sole_header_offset(bundle, V2_SECTION_HEADER)
    if not v1_at < v2_at:
        raise ValueError("the v1 section must precede the v2 section")
    v1_start = v1_at - len(SECTION_SEPARATOR)
    v2_start = v2_at - len(SECTION_SEPARATOR)
    sections = BundleSections(
        first_pass=bundle[:v1_start], v1_block=bundle[v1_start:v2_start], v2_block=bundle[v2_start:]
    )
    assert sections.first_pass + sections.v1_block + sections.v2_block == bundle
    return sections


def arm_texts(sections: BundleSections) -> dict[str, str]:
    """The four arms' research texts, checked to have lost exactly the named section and nothing else."""
    arms = {
        FULL_ARM: sections.first_pass + sections.v1_block + sections.v2_block,
        "minus_v1": sections.first_pass + sections.v2_block,
        "minus_v2": sections.first_pass + sections.v1_block,
        "minus_both": sections.first_pass,
    }
    for arm, lost, kept in (
        ("minus_v1", V1_SECTION_HEADER, V2_SECTION_HEADER),
        ("minus_v2", V2_SECTION_HEADER, V1_SECTION_HEADER),
    ):
        block = sections.v1_block if arm == "minus_v1" else sections.v2_block
        assert lost not in arms[arm], arm
        assert kept in arms[arm], arm
        assert len(arms[FULL_ARM]) - len(arms[arm]) == len(block), arm
    assert V1_SECTION_HEADER not in arms["minus_both"]
    assert V2_SECTION_HEADER not in arms["minus_both"]
    return arms


@dataclass(frozen=True)
class BenchQuestion:
    """One resolved pair, ready to forecast: the question object the prompts read, the four arm texts,
    the resolution to score against, and the clock the archive recorded."""

    question_id: int
    qtype: str
    title: str
    question: MetaculusQuestion
    resolution: Resolution
    arms: dict[str, str]
    forecasting_window: str
    today: str
    published_score: float | None
    spot_peer_score: float | None

    @property
    def n_options(self) -> int | None:
        return len(self.question.options) if isinstance(self.question, MultipleChoiceQuestion) else None

    def summary(self) -> dict[str, Any]:
        """The fields the aggregation needs, so a rescore never has to rebuild the question."""
        return {
            "question_id": self.question_id,
            "qtype": self.qtype,
            "title": self.title,
            "resolution": _resolution_label(self.resolution),
            "n_options": self.n_options,
            "published_score": self.published_score,
            "spot_peer_score": self.spot_peer_score,
        }


def _unit_from_header(question_header: str) -> str | None:
    """The unit the driver brief rendered (``Units: Cases``); its unitless placeholder reads back as None."""
    match = _UNITS_LINE.search(question_header)
    if match is None:
        raise ValueError("numeric question header carries no Units line")
    unit = match.group(1).strip()
    return None if unit == _UNITLESS_HEADER else unit


def _today_from_window(forecasting_window: str) -> str:
    match = _TODAY_LINE.search(forecasting_window)
    if match is None:
        raise ValueError("forecasting window carries no Today line")
    return match.group(1)


def build_question(pair: dict[str, Any], perf: dict[str, Any]) -> MetaculusQuestion:
    """The question object production's prompt builders read, from a pair record and its tagged-dataset row.

    The archive keeps no background text and the backtest blanks it anyway (``_prepare_question_for_backtest``),
    so it is empty here; it is constant across arms either way.
    """
    metadata = perf["metadata"]
    common: dict[str, Any] = {
        "question_text": pair["title"],
        "id_of_question": int(pair["question_id"]),
        "id_of_post": int(pair["post_id"]),
        "page_url": f"https://www.metaculus.com/questions/{pair['post_id']}/",
        "background_info": "",
        "resolution_criteria": pair["resolution_criteria"],
        "fine_print": pair["fine_print"] or "",
        "open_time": datetime.fromisoformat(metadata["open_time"]),
        "scheduled_resolution_time": datetime.fromisoformat(metadata["scheduled_resolve_time"]),
        "api_json": {"question": {}},
    }
    qtype = pair["type"]
    if qtype == "binary":
        return BinaryQuestion(**common)
    if qtype == "multiple_choice":
        return MultipleChoiceQuestion(options=list(perf["options"]), **common)
    if qtype not in ("numeric", "discrete"):
        raise ValueError(f"unsupported question type {qtype!r} for question {pair['question_id']}")
    scaling = perf["scaling"]
    continuous: dict[str, Any] = {
        "lower_bound": float(scaling["range_min"]),
        "upper_bound": float(scaling["range_max"]),
        "open_lower_bound": bool(perf["open_lower_bound"]),
        "open_upper_bound": bool(perf["open_upper_bound"]),
        "zero_point": scaling["zero_point"],
        "cdf_size": int(scaling["inbound_outcome_count"]) + 1,
        "nominal_lower_bound": scaling["nominal_min"],
        "nominal_upper_bound": scaling["nominal_max"],
        "unit_of_measure": _unit_from_header(pair["question_header"]),
    }
    cls = DiscreteQuestion if qtype == "discrete" else NumericQuestion
    return cls(**continuous, **common)


def load_bench_questions(
    pairs_path: Path, perf_path: Path, archive_dir: Path, *, only: set[int] | None = None
) -> list[BenchQuestion]:
    """Every resolved pair (``resolution_raw`` present, a NO included), joined to its archive bundle and
    tagged-dataset row. Any pair that cannot be built raises: a silent skip would shrink the paired set."""
    perf_by_qid = {int(row["question_id"]): row for row in json.loads(perf_path.read_text(encoding="utf-8"))}
    questions: list[BenchQuestion] = []
    for pair in read_jsonl(pairs_path):
        qid = int(pair["question_id"])
        if pair["resolution_raw"] is None or (only is not None and qid not in only):
            continue
        questions.append(_bench_question(pair, perf_by_qid[qid], archive_dir))
    if only is not None and (missing := only - {q.question_id for q in questions}):
        raise ValueError(f"--questions named ids with no resolved pair: {sorted(missing)}")
    return questions


def _bench_question(pair: dict[str, Any], perf: dict[str, Any], archive_dir: Path) -> BenchQuestion:
    qid = int(pair["question_id"])
    record = json.loads((archive_dir / f"{qid}.json").read_text(encoding="utf-8"))
    sections = split_bundle(record["research_text"])
    if sections.first_pass.strip() != pair["first_pass_research"].strip():
        raise ValueError(f"question {qid}: the archive's first-pass text disagrees with the pair record")
    question = build_question(pair, perf)
    resolution = typed_resolution(pair, question)
    return BenchQuestion(
        question_id=qid,
        qtype=pair["type"],
        title=pair["title"],
        question=question,
        resolution=resolution,
        arms=arm_texts(sections),
        forecasting_window=pair["forecasting_window"],
        today=_today_from_window(pair["forecasting_window"]),
        published_score=score_published(pair["published_forecast"], question, resolution),
        spot_peer_score=pair["spot_peer_score"],
    )
