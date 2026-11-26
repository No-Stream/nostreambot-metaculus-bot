from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {text}")
    return value


def positive_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {text}")
    return value


@contextmanager
def run_logging(run_dir: Path) -> Iterator[None]:
    """Write INFO pipeline output to bench.log while leaving warnings visible in the console."""
    root = logging.getLogger()
    file_handler = logging.FileHandler(run_dir / "bench.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    try:
        yield
    finally:
        root.removeHandler(file_handler)
        root.removeHandler(console)
        file_handler.close()
        root.setLevel(previous_level)
