"""Shared posterior summaries and sampler diagnostics for offline analyses."""

from __future__ import annotations

import logging
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def interval_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "interval_90": np.quantile(values, [0.05, 0.95]).tolist(),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def sampling_diagnostics(inference: Any, *, allow_diagnostics_warnings: bool) -> dict[str, Any]:
    summary = cast(pd.DataFrame, az.summary(inference, kind="diagnostics", round_to="none"))
    rhat = summary["r_hat"]
    ess_bulk = summary["ess_bulk"]
    ess_tail = summary["ess_tail"]
    divergences = int(inference.sample_stats["diverging"].sum())
    sample_count = int(inference.sample_stats["diverging"].size)
    invalid = bool(rhat.isna().any() or (rhat > 1.01).any() or divergences / sample_count > 0.01)
    if invalid:
        message = "Sampling diagnostics failed: R-hat missing or above 1.01, or divergences above 1%"
        if not allow_diagnostics_warnings:
            raise ValueError(message)
        logger.warning(message)
    clean_summary = summary.replace([np.inf, -np.inf], np.nan).reset_index()
    return {
        "max_r_hat": finite_or_none(rhat.max()),
        "min_ess_bulk": finite_or_none(ess_bulk.min()),
        "min_ess_tail": finite_or_none(ess_tail.min()),
        "divergences": divergences,
        "draws": sample_count,
        "passed": not invalid,
        "parameters": clean_summary.astype(object).where(clean_summary.notna(), None).to_dict(orient="records"),
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
