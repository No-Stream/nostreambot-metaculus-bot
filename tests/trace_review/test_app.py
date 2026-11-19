"""Streamlit AppTest smoke tests of every page over the fixture bundle (in process, no socket)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from metaculus_bot.constants import PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.trace_review.config import BUNDLE_DIR_ENV
from metaculus_bot.trace_review.filters import KEY_PARAM
from tests.trace_review.workspace_fixtures import (
    COLLIDING_BINARY,
    MANTIC_DISCRETE,
    NUMERIC_BACKFILLED,
    BuiltBundle,
    forecast_key,
)

APP_PATH = str(Path(__file__).parents[2] / "metaculus_bot" / "trace_review" / "app.py")
APP_TIMEOUT_S = 60
DETAIL_TABS = 5


def _app(monkeypatch: pytest.MonkeyPatch, bundle_dir: Path, page: str | None = None, **params: str) -> AppTest:
    monkeypatch.setenv(BUNDLE_DIR_ENV, str(bundle_dir))
    app = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_S)
    for name, value in params.items():
        app.query_params[name] = value
    if page is not None:
        app.switch_page(page)
    return app.run()


def _no_failures(app: AppTest) -> None:
    assert not app.exception, [exception.value for exception in app.exception]
    assert not app.error, [error.value for error in app.error]


def _forecast_table_rows(app: AppTest) -> int:
    return max(len(frame.value) for frame in app.dataframe)


class TestForecastList:
    def test_the_list_shows_every_forecast_and_one_block_per_platform(
        self, monkeypatch, built_bundle: BuiltBundle, bundle_forecasts
    ):
        app = _app(monkeypatch, built_bundle.output_dir)

        _no_failures(app)
        assert _forecast_table_rows(app) == len(bundle_forecasts)
        headers = [header.value for header in app.subheader]
        assert headers.count("Metaculus") == 1
        assert headers.count("Mantic") == 1
        assert len(app.metric) == 2

    def test_a_query_string_filter_narrows_the_table(self, monkeypatch, built_bundle: BuiltBundle, bundle_forecasts):
        app = _app(monkeypatch, built_bundle.output_dir, type="binary")

        _no_failures(app)
        assert _forecast_table_rows(app) == int((bundle_forecasts["question_type"] == "binary").sum())


class TestForecastDetail:
    @pytest.mark.parametrize(
        "key",
        [
            forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY),
            forecast_key(PLATFORM_METACULUS, NUMERIC_BACKFILLED),
            forecast_key(PLATFORM_MANTIC, MANTIC_DISCRETE),
        ],
    )
    def test_each_question_type_renders(self, monkeypatch, built_bundle: BuiltBundle, key: str):
        app = _app(monkeypatch, built_bundle.output_dir, "pages/detail.py", **{KEY_PARAM: key})

        _no_failures(app)
        assert len(app.tabs) == DETAIL_TABS
        assert len(app.get("plotly_chart")) == 1

    def test_next_moves_to_the_neighbour_in_the_filtered_order(self, monkeypatch, built_bundle: BuiltBundle):
        first = forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY)
        app = _app(monkeypatch, built_bundle.output_dir, "pages/detail.py", **{KEY_PARAM: first})
        caption_before = next(caption.value for caption in app.caption if " in the filtered list" in caption.value)

        app.button[1].click().run()

        _no_failures(app)
        assert app.query_params[KEY_PARAM] != [first]
        caption_after = next(caption.value for caption in app.caption if " in the filtered list" in caption.value)
        assert int(caption_after.split()[0]) == int(caption_before.split()[0]) + 1

    def test_an_unknown_key_is_an_error_not_a_crash(self, monkeypatch, built_bundle: BuiltBundle):
        app = _app(monkeypatch, built_bundle.output_dir, "pages/detail.py", **{KEY_PARAM: "metaculus:999999"})

        assert not app.exception
        assert any("metaculus:999999" in error.value for error in app.error)


def test_the_models_page_renders_one_table_per_platform(monkeypatch, built_bundle: BuiltBundle):
    app = _app(monkeypatch, built_bundle.output_dir, "pages/models.py")

    _no_failures(app)
    assert [header.value for header in app.subheader] == ["Mantic", "Metaculus", "Member rows"]


@pytest.mark.parametrize("page", [None, "pages/detail.py", "pages/models.py"])
def test_a_missing_bundle_names_the_build_command(monkeypatch, tmp_path: Path, page: str | None):
    app = _app(monkeypatch, tmp_path, page)

    assert not app.exception
    assert any("make trace_index" in error.value for error in app.error)
