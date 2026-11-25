"""Cached readers of the trace-review bundle for the app.

The forecasts and members tables load whole (a few MB); texts and telemetry load one forecast's
rows at a time through a parquet filter, except the two text columns the full-text search needs.
Every read is cast back to ``schema.py``, so a bundle from an older builder fails here, loudly.
The cache key includes the manifest's build time, so a rebuild is picked up on the next rerun.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from metaculus_bot.trace_review.config import (
    BUNDLE_DIR_ENV,
    DEFAULT_BUNDLE_DIR,
    FORECASTS_FILE,
    MANIFEST_FILE,
    MEMBERS_FILE,
    TELEMETRY_FILE,
    TEXTS_FILE,
)
from metaculus_bot.trace_review.schema import (
    FORECAST_COLUMNS,
    MEMBER_COLUMNS,
    SCHEMA_VERSION,
    TELEMETRY_COLUMNS,
    TEXT_COLUMNS,
    ColumnTypes,
    cast_frame,
)

SEARCH_TEXT_COLUMNS: tuple[str, ...] = ("forecast_key", "comment_text", "research_text")


class BundleUnavailableError(RuntimeError):
    """No bundle at the configured directory, or one written by a different schema version."""


def bundle_dir() -> Path:
    return Path(os.environ.get(BUNDLE_DIR_ENV) or DEFAULT_BUNDLE_DIR)


def load_manifest(directory: Path) -> dict:
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise BundleUnavailableError(f"No trace index at {directory}. Build it with: make trace_index")
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise BundleUnavailableError(
            f"The trace index at {directory} has schema version {manifest['schema_version']}, this app reads "
            f"{SCHEMA_VERSION}. Rebuild it with: make trace_index"
        )
    return manifest


def _read(directory: Path, name: str, columns: ColumnTypes, key: str | None = None) -> pd.DataFrame:
    filters = [("forecast_key", "==", key)] if key is not None else None
    frame = pd.read_parquet(directory / name, filters=filters)
    return cast_frame(frame.reset_index(drop=True), columns)


@st.cache_data(show_spinner="Loading the trace index")
def load_forecasts(directory: str, built_at: str) -> pd.DataFrame:
    del built_at
    return _read(Path(directory), FORECASTS_FILE, FORECAST_COLUMNS)


@st.cache_data(show_spinner=False)
def load_members(directory: str, built_at: str) -> pd.DataFrame:
    del built_at
    return _read(Path(directory), MEMBERS_FILE, MEMBER_COLUMNS)


@st.cache_data(show_spinner="Loading comment and research text for the search")
def load_search_texts(directory: str, built_at: str) -> pd.DataFrame:
    del built_at
    frame = pd.read_parquet(Path(directory) / TEXTS_FILE, columns=list(SEARCH_TEXT_COLUMNS))
    return cast_frame(frame, {column: TEXT_COLUMNS[column] for column in SEARCH_TEXT_COLUMNS})


@st.cache_data(show_spinner=False, max_entries=64)
def load_texts_for(directory: str, built_at: str, key: str) -> dict[str, object]:
    del built_at
    rows = _read(Path(directory), TEXTS_FILE, TEXT_COLUMNS, key)
    if len(rows) != 1:
        raise KeyError(f"{key} has {len(rows)} text rows; expected exactly one")
    return {str(column): (None if pd.isna(value) else value) for column, value in rows.iloc[0].items()}


@st.cache_data(show_spinner=False, max_entries=64)
def load_telemetry_for(directory: str, built_at: str, key: str) -> pd.DataFrame:
    del built_at
    return _read(Path(directory), TELEMETRY_FILE, TELEMETRY_COLUMNS, key)
