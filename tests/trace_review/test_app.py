"""Streamlit AppTest smoke tests of every page over the fixture bundle (in process, no socket)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, unquote

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from metaculus_bot.constants import PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.trace_review.config import BUNDLE_DIR_ENV
from metaculus_bot.trace_review.filters import KEY_PARAM
from tests.trace_review.workspace_fixtures import (
    COLLIDING_BINARY,
    MANTIC_DISCRETE,
    NAMED_MC,
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
        captions = [caption.value for caption in app.caption]
        assert sum(caption.startswith("Mean log score: ") for caption in captions) == 2
        assert not any("{" in caption for caption in captions), "a raw dict leaked into the summary"

    def test_a_query_string_filter_narrows_the_table(self, monkeypatch, built_bundle: BuiltBundle, bundle_forecasts):
        app = _app(monkeypatch, built_bundle.output_dir, type="binary")

        _no_failures(app)
        assert _forecast_table_rows(app) == int((bundle_forecasts["question_type"] == "binary").sum())

    def test_each_question_links_to_its_detail_page_with_the_filters_and_its_title(
        self, monkeypatch, built_bundle: BuiltBundle, bundle_forecasts
    ):
        app = _app(monkeypatch, built_bundle.output_dir, type="binary")

        _no_failures(app)
        table = max(app.dataframe, key=lambda frame: len(frame.value)).value
        titles = bundle_forecasts.set_index("forecast_key")["title"]
        for link in table["title"]:
            path, _, title = link.partition("#")
            query = parse_qs(path.removeprefix("detail?"))
            assert path.startswith("detail?")
            assert query["type"] == ["binary"]
            assert unquote(title) == titles[query[KEY_PARAM][0]]


class TestForecastDetail:
    @pytest.mark.parametrize(
        "key",
        [
            forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY),
            forecast_key(PLATFORM_METACULUS, NUMERIC_BACKFILLED),
            forecast_key(PLATFORM_MANTIC, MANTIC_DISCRETE),
        ],
    )
    def test_each_question_type_renders(
        self, monkeypatch, built_bundle: BuiltBundle, bundle_forecasts, bundle_members, key: str
    ):
        app = _app(monkeypatch, built_bundle.output_dir, "app_pages/detail.py", **{KEY_PARAM: key})

        _no_failures(app)
        assert len(app.tabs) == DETAIL_TABS
        assert len(app.get("plotly_chart")) == 1
        # Traits are rounded "label: value" badges, and a flag reads yes or no, never True/False.
        text = " ".join([*(caption.value for caption in app.caption), *(block.value for block in app.markdown)])
        badges = re.findall(r":(?:gray|violet)-badge\[([^\]]+)\]", text)
        has_rationale = bundle_members[bundle_members["forecast_key"] == key]["has_rationale"].fillna(value=False)
        assert any(badge.startswith("Cites a market price: ") for badge in badges) == bool(has_rationale.any())
        assert all(not re.search(r"\d\.\d{2,}", badge) for badge in badges)
        assert "True" not in text
        assert "False" not in text
        assert "Extraction rung" not in text
        fine_era = bundle_forecasts.set_index("forecast_key").loc[key, "triple_subera_fine"]
        if pd.notna(fine_era):
            assert any(fine_era in block.value for block in app.markdown)

    def test_rationale_markdown_escapes_tildes_so_they_are_not_strikethrough(
        self, monkeypatch, built_bundle: BuiltBundle
    ):
        key = forecast_key(PLATFORM_METACULUS, NAMED_MC)
        app = _app(monkeypatch, built_bundle.output_dir, "app_pages/detail.py", **{KEY_PARAM: key})

        _no_failures(app)
        rendered = [block.value for block in app.markdown if "41%" in block.value]
        assert rendered
        assert all("\\~41%" in block and not re.search(r"(?<!\\)~", block) for block in rendered)

    def test_the_detail_page_without_a_key_points_to_the_list(self, monkeypatch, built_bundle: BuiltBundle):
        app = _app(monkeypatch, built_bundle.output_dir, "app_pages/detail.py")

        _no_failures(app)
        assert any("from the list" in info.value for info in app.info)

    def test_rationales_can_be_read_one_model_at_a_time(self, monkeypatch, built_bundle: BuiltBundle):
        key = forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY)
        app = _app(monkeypatch, built_bundle.output_dir, "app_pages/detail.py", **{KEY_PARAM: key})
        assert app.radio(key=f"view_{key}").value == "Side by side"

        app.radio(key=f"view_{key}").set_value("One model at a time").run()

        _no_failures(app)
        chosen = app.radio(key=f"model_{key}")
        assert chosen.value == chosen.options[0]
        assert [header.value for header in app.subheader] == [chosen.value]

    def test_next_moves_to_the_neighbour_in_the_filtered_order(self, monkeypatch, built_bundle: BuiltBundle):
        first = forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY)
        app = _app(monkeypatch, built_bundle.output_dir, "app_pages/detail.py", **{KEY_PARAM: first})
        caption_before = next(caption.value for caption in app.caption if " in the filtered list" in caption.value)

        app.button[1].click().run()

        _no_failures(app)
        assert app.query_params[KEY_PARAM] != [first]
        caption_after = next(caption.value for caption in app.caption if " in the filtered list" in caption.value)
        assert int(caption_after.split()[0]) == int(caption_before.split()[0]) + 1

    def test_the_back_link_keeps_every_value_of_a_repeated_filter(self, monkeypatch, built_bundle: BuiltBundle):
        key = forecast_key(PLATFORM_METACULUS, COLLIDING_BINARY)
        monkeypatch.setenv(BUNDLE_DIR_ENV, str(built_bundle.output_dir))
        app = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT_S)
        app.query_params[KEY_PARAM] = key
        app.query_params["type"] = ["binary", "numeric"]
        app.switch_page("app_pages/detail.py").run()

        _no_failures(app)
        back = next(link for link in app.get("page_link") if link.proto.label == "Back to the list")
        assert parse_qs(back.proto.query_string) == {"type": ["binary", "numeric"]}

    def test_an_unknown_key_is_an_error_not_a_crash(self, monkeypatch, built_bundle: BuiltBundle):
        app = _app(monkeypatch, built_bundle.output_dir, "app_pages/detail.py", **{KEY_PARAM: "metaculus:999999"})

        assert not app.exception
        assert any("metaculus:999999" in error.value for error in app.error)


def test_the_models_page_renders_one_table_per_platform(monkeypatch, built_bundle: BuiltBundle):
    app = _app(monkeypatch, built_bundle.output_dir, "app_pages/models.py")

    _no_failures(app)
    assert [header.value for header in app.subheader] == ["Mantic", "Metaculus", "Member rows"]


@pytest.mark.parametrize("page", [None, "app_pages/detail.py", "app_pages/models.py"])
def test_a_missing_bundle_names_the_build_command(monkeypatch, tmp_path: Path, page: str | None):
    app = _app(monkeypatch, tmp_path, page)

    assert not app.exception
    assert any("make trace_index" in error.value for error in app.error)
