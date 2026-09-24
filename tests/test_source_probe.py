"""The maintained single-source smoke path cannot invoke a paid ladder rung."""

from typing import Any

import pytest

from metaculus_bot.research.fetch_ladder.digest import bm25_digest
from metaculus_bot.research.resolution_fetch_result import FetchResult
from scripts.probes import fetch_diagnostic


@pytest.mark.asyncio
async def test_source_probe_uses_query_and_structurally_free_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fetch(url: str, **kwargs: Any) -> FetchResult:
        calls.append(kwargs)
        return FetchResult(
            url=url,
            status="success",
            http_status=200,
            content_type="application/zip",
            text="Regional drought measurements",
            route="direct",
        )

    monkeypatch.setattr(fetch_diagnostic, "fetch_url", fetch)
    result = await fetch_diagnostic.probe_source("https://example.org/data.zip", "regional drought")
    assert result.text == "Regional drought measurements"
    assert len(calls) == 1
    assert calls[0]["ctx"].query == "regional drought"
    assert calls[0]["policy"].digest is bm25_digest
    assert "url_context" not in calls[0]["policy"].rungs_enabled


def test_source_probe_cli_requires_query() -> None:
    with pytest.raises(SystemExit):
        fetch_diagnostic.parse_args(["--source-url", "https://example.org/data.zip"])
