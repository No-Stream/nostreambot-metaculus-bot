"""Tests for backfill_research_from_comments.py — extracting research from bot comments."""

import json
from pathlib import Path

# --- Sample comment bodies mimicking real bot output ---

SAMPLE_COMMENT_FULL = """\
# SUMMARY

*Question*: Will X happen by 2026?
*URL*: https://www.metaculus.com/questions/43613/

Brief summary of the forecast.

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Here are the relevant news articles about topic X.

Source 1: Article about X happening.
Source 2: Analysis of X trends.

---

## Web Research (Native Search)
Web search found additional context about X.

Multiple sources confirm the trend is accelerating.

---

## Web Research (Google Search via Gemini)
Gemini grounded search results here.

Key finding: the timeline is shorter than expected.

---

## Targeted Gap-Fill (second pass)
Gap 1: What is the current regulatory status?
Answer: Regulations were passed in March 2026.

# FORECASTS
## R1: Forecaster 1 Reasoning
I think the probability is 75% because...

## R2: Forecaster 2 Reasoning
I estimate 60% based on...
"""

SAMPLE_COMMENT_NO_RESEARCH = """\
# SUMMARY

*Question*: Will Y happen?

This question could not be forecasted due to errors.

# FORECASTS
No forecasts available.
"""


def test_comment_backfill_expands_archived_list_comments(monkeypatch):
    import scripts.backfill_research_from_comments as mod

    fixture_path = Path(__file__).parent / "data" / "truncated_comment_payloads.json"
    list_comment = json.loads(fixture_path.read_text())["list_comment"]
    calls: list[tuple[str, dict | None]] = []

    def fake_api_get(endpoint: str, token: str, params: dict | None = None) -> dict:
        calls.append((endpoint, params))
        return {"results": [list_comment], "next": None}

    def expand(comment: dict, token: str) -> dict:
        assert token == "token"
        return {**comment, "text": SAMPLE_COMMENT_FULL}

    monkeypatch.setattr(mod, "api_get", fake_api_get)
    monkeypatch.setattr(mod, "expand_truncated_comment", expand)

    comments = mod.fetch_all_comments("token")

    assert len(comments) == 1
    assert comments[0]["text"] == SAMPLE_COMMENT_FULL
    assert [params and params.get("is_private") for _, params in calls] == [None, "true"]


# Mirrors SAMPLE_COMMENT_FULL but carries the research blocks ADDED this session:
# ``## Prediction Market Snapshot`` (a NEW provider header wired into
# PROVIDER_HEADERS), ``## Provider Diagnostics`` (pipe-delimited per-provider
# status lines), a ``### URL Context Fetches`` / ``_url_context: none_`` block,
# and the ``<!-- financial_routing ... -->`` HTML comment. These exercise the
# regression that detect_providers / extract_research_text must (a) recognize the
# new prediction-market header, (b) span the whole research blob including the new
# blocks, and (c) NOT emit spurious providers from the diagnostics / routing /
# url-context telemetry lines (detect_providers is a substring scan over header
# strings, so a stray ``native_search`` token in a diagnostics line must not count
# as a second provider header match).
SAMPLE_COMMENT_WITH_NEW_BLOCKS = """\
# SUMMARY

*Question*: Will X happen by 2026?
*URL*: https://www.metaculus.com/questions/43613/

Brief summary of the forecast.

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Here are the relevant news articles about topic X.

---

## Provider Diagnostics
- asknews: ok | 4210 chars | 8120 ms
- native_search: ok | 9044 chars | 41210 ms
- gemini_search: ok | 2210 chars | 15330 ms

---

## Web Research (Native Search)
Web search found additional context about X.

---

## Prediction Market Snapshot
| platform | title | prob | vol | close | conf |
|---|---|---|---|---|---|
| Polymarket | Will X happen by 2026 | 0.55 | 12000 | 2026-07-01 | 0.80 |

##### Resolution criteria / rules
- **Polymarket** <https://polymarket.com/event/x>: resolves YES if X happens: 55%

---

## Web Research (Google Search via Gemini)
Gemini grounded search results here.

### URL Context Fetches
_url_context: none_

<!-- financial_routing: fred=[] tickers=['ACME'] extracted=['ACME'] unknown=[] -->

# FORECASTS
## R1: Forecaster 1 Reasoning
I think the probability is 75% because...
"""

SAMPLE_COMMENT_TRIMMED = """\
# SUMMARY

*Question*: Will Z happen?

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Very long research text here...

[... trimmed for length]

---

## Web Research (Native Search)
More research content.

# FORECASTS
## R1: Forecaster 1
Prediction: 50%
"""

SAMPLE_COMMENT_NO_FORECASTS_MARKER = """\
# SUMMARY

*Question*: Will W happen?

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Research with no forecasts section following.

## Web Research (Native Search)
More content at the end.
"""

SAMPLE_COMMENT_MINIMAL_RESEARCH = """\
# SUMMARY

Summary text.

# RESEARCH
## Report 1 Research
## News Articles (AskNews)
Just one provider.

# FORECASTS
## R1: Reasoning
Done.
"""


class TestExtractResearchText:
    """Test extracting research text from structured bot comments."""

    def test_extracts_full_research_section(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        assert "News Articles (AskNews)" in result
        assert "Web Research (Native Search)" in result
        assert "Web Research (Google Search via Gemini)" in result
        assert "Targeted Gap-Fill (second pass)" in result
        assert "Here are the relevant news articles about topic X." in result
        assert "Regulations were passed in March 2026." in result

    def test_excludes_forecasts_section(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        assert "Forecaster 1 Reasoning" not in result
        assert "I think the probability is 75%" not in result

    def test_excludes_summary_section(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        assert "Brief summary of the forecast" not in result
        assert "*Question*: Will X happen" not in result

    def test_returns_none_when_no_research_marker(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_NO_RESEARCH)
        assert result is None

    def test_handles_missing_forecasts_marker(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_NO_FORECASTS_MARKER)
        assert result is not None
        assert "Research with no forecasts section following." in result
        assert "More content at the end." in result

    def test_strips_report_1_research_header(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_FULL)
        assert result is not None
        # The "## Report 1 Research" line itself should be stripped as a header
        assert not result.startswith("## Report 1 Research")

    def test_minimal_research_single_provider(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text(SAMPLE_COMMENT_MINIMAL_RESEARCH)
        assert result is not None
        assert "Just one provider." in result

    def test_empty_string_returns_none(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text("")
        assert result is None

    def test_research_marker_only_returns_none(self):
        from scripts.backfill_research_from_comments import extract_research_text

        result = extract_research_text("# RESEARCH\n\n# FORECASTS\n")
        assert result is None


class TestDetectTrimmed:
    """Test detection of trimmed research content."""

    def test_detects_standard_trim_marker(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("Some text [... trimmed for length] more text") is True

    def test_detects_alternate_trim_marker(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("Content [...trimmed for length]") is True

    def test_no_trim_marker_returns_false(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("Normal research text without any trimming.") is False

    def test_empty_string_returns_false(self):
        from scripts.backfill_research_from_comments import detect_trimmed

        assert detect_trimmed("") is False

    def test_trimmed_in_sample_comment(self):
        from scripts.backfill_research_from_comments import detect_trimmed, extract_research_text

        research = extract_research_text(SAMPLE_COMMENT_TRIMMED)
        assert research is not None
        assert detect_trimmed(research) is True


class TestProviderDetectionFromComments:
    """Test that detect_providers works on comment-extracted research."""

    def test_detects_all_providers_in_full_comment(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_providers

        research = extract_research_text(SAMPLE_COMMENT_FULL)
        assert research is not None
        providers = detect_providers(research)
        assert "asknews" in providers
        assert "native_search" in providers
        assert "gemini_search" in providers

    def test_detects_gap_fill_in_full_comment(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_gap_fill

        research = extract_research_text(SAMPLE_COMMENT_FULL)
        assert research is not None
        assert detect_gap_fill(research) is True

    def test_no_gap_fill_in_minimal_comment(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_gap_fill

        research = extract_research_text(SAMPLE_COMMENT_MINIMAL_RESEARCH)
        assert research is not None
        assert detect_gap_fill(research) is False


class TestNewResearchBlocksFromComments:
    """Regression guards for the research blocks added this session.

    The Prediction Market Snapshot, Provider Diagnostics, URL Context Fetches,
    and financial_routing blocks all live INSIDE the ``# RESEARCH`` span. These
    tests lock down that (a) the new prediction-market header is recognized by
    detect_providers, (b) the research span still bounds correctly with the new
    blocks present, and (c) the diagnostics / routing / url-context telemetry
    lines don't trip detect_providers' substring scan into emitting spurious
    providers.
    """

    def test_detects_prediction_market_provider(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_providers

        research = extract_research_text(SAMPLE_COMMENT_WITH_NEW_BLOCKS)
        assert research is not None
        providers = detect_providers(research)
        assert "prediction_market" in providers
        # The other real provider headers still resolve correctly.
        assert "asknews" in providers
        assert "native_search" in providers
        assert "gemini_search" in providers

    def test_extract_research_text_spans_new_blocks_without_bleeding(self):
        from scripts.backfill_research_from_comments import extract_research_text

        research = extract_research_text(SAMPLE_COMMENT_WITH_NEW_BLOCKS)
        assert research is not None

        # Begins at the first provider section (## Report 1 Research header is
        # stripped), includes every new block...
        assert research.startswith("## News Articles (AskNews)")
        assert "## Provider Diagnostics" in research
        assert "- native_search: ok | 9044 chars | 41210 ms" in research
        assert "## Prediction Market Snapshot" in research
        assert "- **Polymarket** <https://polymarket.com/event/x>: resolves YES if X happens: 55%" in research
        assert "### URL Context Fetches" in research
        assert "_url_context: none_" in research
        assert "<!-- financial_routing: fred=[] tickers=['ACME'] extracted=['ACME'] unknown=[] -->" in research

        # ...and ends before # FORECASTS — no forecast prose bleeds in.
        assert "# FORECASTS" not in research
        assert "Forecaster 1 Reasoning" not in research
        assert "I think the probability is 75%" not in research

    def test_telemetry_lines_do_not_emit_spurious_providers(self):
        from scripts.backfill_research_from_comments import extract_research_text
        from scripts.backfill_research_from_logs import detect_providers

        research = extract_research_text(SAMPLE_COMMENT_WITH_NEW_BLOCKS)
        assert research is not None
        providers = detect_providers(research)

        # detect_providers is a substring scan over the ## header strings. The
        # diagnostics block mentions provider NAMES (``native_search: ok | ...``)
        # but never their ``## ...`` header form, so each provider must be
        # counted exactly once — no duplicates from the telemetry lines.
        assert len(providers) == len(set(providers))
        # Providers whose ## header is absent must NOT appear despite their name
        # showing up in diagnostics / routing telemetry.
        assert "financial_data" not in providers
        assert "exa" not in providers
        assert "perplexity" not in providers


class TestBuildRecord:
    """Test record construction from a comment dict."""

    def test_builds_valid_record(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 12345,
            "on_post": 99999,
            "text": SAMPLE_COMMENT_FULL,
            "created_at": "2026-05-20T15:48:05Z",
        }
        record = build_record(comment, question_id=43613)
        assert record is not None
        assert record["schema_version"] == 1
        assert record["qid"] == 43613
        assert record["on_post"] == 99999
        assert record["run_id"] == "comment-12345"
        assert record["timestamp"] == "2026-05-20T15:48:05Z"
        assert record["run_mode"] == "tournament"
        assert record["research_chars"] == len(record["research_text"])
        assert record["is_trimmed"] is False
        assert "asknews" in record["providers_used"]
        assert record["gap_fill_used"] is True
        assert record["page_url"] == "https://www.metaculus.com/questions/43613/"

    def test_builds_record_with_none_qid(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 111,
            "on_post": 222,
            "text": SAMPLE_COMMENT_MINIMAL_RESEARCH,
            "created_at": "2026-05-01T00:00:00Z",
        }
        record = build_record(comment, question_id=None)
        assert record is not None
        assert record["qid"] is None
        assert record["page_url"] == ""

    def test_returns_none_for_no_research(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 111,
            "on_post": 222,
            "text": SAMPLE_COMMENT_NO_RESEARCH,
            "created_at": "2026-05-01T00:00:00Z",
        }
        record = build_record(comment, question_id=43613)
        assert record is None

    def test_detects_trimmed_in_record(self):
        from scripts.backfill_research_from_comments import build_record

        comment = {
            "id": 333,
            "on_post": 444,
            "text": SAMPLE_COMMENT_TRIMMED,
            "created_at": "2026-05-15T10:00:00Z",
        }
        record = build_record(comment, question_id=55555)
        assert record is not None
        assert record["is_trimmed"] is True


def _run_main(tmp_path, monkeypatch, comments, *, extra_argv=()):
    """Drive the CLI end to end with the network, token load and rate-gate stubbed out.

    Everything below the API boundary is real: argument parsing, the post_id -> question_id
    cache, the dedup read of an existing output file, record building and the JSONL write.
    Returns the parsed records the run appended to the output file.
    """
    import scripts.backfill_research_from_comments as mod

    resolved: list[int] = []

    monkeypatch.setattr(mod, "get_token", lambda: "token")
    monkeypatch.setattr(mod, "verify_metaculus_api_identity", lambda: None)
    monkeypatch.setattr(mod, "fetch_all_comments", lambda token: comments)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    def _resolve(post_id: int, token: str) -> int | None:
        resolved.append(post_id)
        return post_id + 1000 if post_id != 777 else None

    monkeypatch.setattr(mod, "get_question_id_for_post", _resolve)
    monkeypatch.setattr("sys.argv", ["backfill", "--output-dir", str(tmp_path), *[str(a) for a in extra_argv]])

    mod.main()

    output_file = tmp_path / "comments_backfill.jsonl"
    written = (
        [json.loads(line) for line in output_file.read_text().splitlines() if line.strip()]
        if output_file.exists()
        else []
    )
    return written, resolved


class TestMainWritesTheBackfillFile:
    """End-to-end CLI behavior: what lands in comments_backfill.jsonl and what is skipped.

    ``main`` carries the parts no unit test reaches — the dedup read, the post->question
    cache, the dry-run gate and the per-record counters — so these pin them before any
    restructuring of the function.
    """

    def _comments(self):
        return [
            {"id": 1, "on_post": 101, "text": SAMPLE_COMMENT_FULL, "created_at": "2026-05-20T15:48:05Z"},
            {"id": 2, "on_post": 102, "text": SAMPLE_COMMENT_NO_RESEARCH, "created_at": "2026-05-21T00:00:00Z"},
            {"id": 3, "on_post": None, "text": SAMPLE_COMMENT_FULL, "created_at": "2026-05-22T00:00:00Z"},
            {"id": 4, "on_post": 777, "text": SAMPLE_COMMENT_TRIMMED, "created_at": "2026-05-23T00:00:00Z"},
        ]

    def test_writes_one_record_per_comment_with_research(self, tmp_path, monkeypatch):
        written, resolved = _run_main(tmp_path, monkeypatch, self._comments())

        assert [r["on_post"] for r in written] == [101, 777]
        assert written[0]["qid"] == 1101
        assert written[0]["run_id"] == "comment-1"
        assert written[0]["is_trimmed"] is False
        # An unresolvable post still gets a record, with qid None and an empty page_url.
        assert written[1]["qid"] is None
        assert written[1]["page_url"] == ""
        assert written[1]["is_trimmed"] is True
        # The comment with no research section and the one with no post_id are both skipped,
        # and only real post_ids are ever resolved.
        assert resolved == [101, 102, 777]

    def test_rerun_deduplicates_against_the_existing_file(self, tmp_path, monkeypatch):
        _run_main(tmp_path, monkeypatch, self._comments())
        written, resolved = _run_main(tmp_path, monkeypatch, self._comments())

        assert [r["on_post"] for r in written] == [101, 777]  # unchanged: nothing appended
        # Dedup keys on what was WRITTEN, so only the research-bearing posts are skipped;
        # a comment with no research section is re-resolved on every re-run.
        assert resolved == [102]

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        written, resolved = _run_main(tmp_path, monkeypatch, self._comments(), extra_argv=["--dry-run"])

        assert written == []
        assert resolved == [101, 102, 777]

    def test_limit_truncates_the_comment_list(self, tmp_path, monkeypatch):
        written, resolved = _run_main(tmp_path, monkeypatch, self._comments(), extra_argv=["--limit", 1])

        assert [r["on_post"] for r in written] == [101]
        assert resolved == [101]
