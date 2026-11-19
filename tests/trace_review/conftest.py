"""Module-scoped fixtures over the trace-review workspace (builders in workspace_fixtures.py)."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest

from metaculus_bot.performance_analysis import collector
from metaculus_bot.trace_review.build import build_index
from metaculus_bot.trace_review.config import FORECASTS_FILE, MEMBERS_FILE
from metaculus_bot.trace_review.schema import FORECAST_COLUMNS, MEMBER_COLUMNS, cast_frame
from tests.trace_review.workspace_fixtures import (
    BuiltBundle,
    FetcherDoubles,
    Workspace,
    install_doubles,
    write_workspace,
)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(collector, "api_get", lambda *args, **kwargs: pytest.fail("api_get reached"))
        return write_workspace(tmp_path_factory.mktemp("trace_review_workspace"))


@pytest.fixture(scope="module")
def built_bundle(workspace: Workspace, tmp_path_factory: pytest.TempPathFactory) -> Iterator[BuiltBundle]:
    """One full build of the fixture workspace (both platforms), shared by the read-only tests."""
    doubles = FetcherDoubles()
    with pytest.MonkeyPatch.context() as monkeypatch:
        preflights = install_doubles(monkeypatch, doubles)
        result = build_index(workspace.config(tmp_path_factory.mktemp("bundle")))
        yield BuiltBundle(workspace, result, doubles, preflights)


@pytest.fixture(scope="module")
def bundle_forecasts(built_bundle: BuiltBundle) -> pd.DataFrame:
    """The built forecasts table as the app reads it (cast back to the schema)."""
    return cast_frame(pd.read_parquet(built_bundle.output_dir / FORECASTS_FILE), FORECAST_COLUMNS)


@pytest.fixture(scope="module")
def bundle_members(built_bundle: BuiltBundle) -> pd.DataFrame:
    return cast_frame(pd.read_parquet(built_bundle.output_dir / MEMBERS_FILE), MEMBER_COLUMNS)
