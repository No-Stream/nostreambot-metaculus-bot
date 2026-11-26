from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion
from forecasting_tools.data_models.questions import MetaculusQuestion, OutOfBoundsResolution

from scripts.probes.bench_common.scoring import Resolution

STATUS_SCORED = "scored"
STATUS_EXTRACTION_FAILED = "extraction_failed"
STATUS_BUILD_FAILED = "build_failed"
STATUS_UNIT_MISMATCH = "unit_mismatch"
STATUS_API_ERROR = "api_error"
STATUS_SKIPPED_SPEND_CAP = "skipped_spend_cap"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def typed_resolution(pair: dict[str, Any], question: MetaculusQuestion) -> Resolution:
    """Convert the parsed dataset resolution to the representation used by production scoring."""
    parsed = pair["resolution_parsed"]
    question_id = pair["question_id"]
    if isinstance(question, BinaryQuestion):
        if not isinstance(parsed, bool):
            raise ValueError(f"binary question {question_id} resolution is {parsed!r}, not a bool")
        return parsed
    if isinstance(question, MultipleChoiceQuestion):
        if parsed not in question.options:
            raise ValueError(f"question {question_id} resolved {parsed!r}, not one of {question.options}")
        return str(parsed)
    if isinstance(parsed, bool) or not isinstance(parsed, (int, float, str)):
        raise ValueError(f"numeric question {question_id} resolution is {parsed!r}")
    if isinstance(parsed, str):
        return OutOfBoundsResolution(parsed)
    return float(parsed)


def resolution_label(resolution: Resolution) -> str:
    return resolution.value if isinstance(resolution, OutOfBoundsResolution) else str(resolution)
