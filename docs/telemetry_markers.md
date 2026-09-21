# Telemetry marker registry

This is the receipts and design-detail home for the marker registry in
`scripts/telemetry/markers.py`. Every entry in that module's `MARKER_SPECS` list is a
`MarkerSpec`: a name, a compiled regex that reads the marker's fields off a bot run-log line (or,
for the three HTML-comment markers, off a published Metaculus comment), and `qid_kind`, the
Metaculus id space its `question=` reference lives in (`post_id` or `question_id`, or `None` for
markers with no question ref). `parse_log_text` walks every line of a run log and matches it
against every spec's regex with `re.search`, so a spec does not care what prefix the logging
formatter puts before the message: the production format is
`%(asctime)s - %(name)s - %(levelname)s - %(message)s` and the ablation harness's is
`%(asctime)s %(levelname)s %(name)s | %(message)s`, and both parse against the same regexes.
Marker tokens are mutually exclusive on one line: the parser breaks on the first spec that
matches, so no line is double-counted.

Each section under "Marker specs" documents one `MarkerSpec`, headed exactly by `spec.name` in
upper case (`MEMBER_FORECAST` documents the spec named `"member_forecast"`). The code file keeps
only a one-line `# Why: ... Receipt: docs/telemetry_markers.md "TOKEN".` comment above each spec;
this file carries the emitter location, what each field means, and the receipts, dates, and
incidents behind the design.

## Marker index

| Token | Emitter | What it records |
|---|---|---|
| `EXTRACTION_RUNG` | `value_extraction.py:_log_extraction` | Per-forecast extraction-rung outcome. `qtype` is the block type: a question type, or `pmf` for the per-bin block a Mantic enumerable grid is elicited with since 2026-09-09. The question's own type rides the `MEMBER_FORECAST` line with the same `question` and `model`. |
| `BLOCK_FALLBACK` | `value_extraction.py:_run_ladder` | Per-forecast fallback record, only when the winning value came from a candidate other than the first the best-first walk tried. |
| `GAP_FILL_V2` | `research/agentic/loop.py:_log_completion` | Per-question gap-fill v2 agentic loop completion counters, including image tool calls in `tool_calls`. |
| `GHOST_PRE` / `GHOST_PRE_JSON` | `research/agentic/loop.py:_set_research_plan_tool` | Pre-research ghost snapshot (the counterpart to `GHOST_FORECAST` taken before research starts) and its JSON companion. |
| `GHOST_FORECAST` / `GHOST_FORECAST_JSON` | `research/agentic/loop.py:_run_ghost_phase` | Concluding ghost-forecast summary and its full-fidelity JSON companion. |
| `GHOST_FORECAST_V1` / `GHOST_FORECAST_V1_JSON` | `research/agentic/loop.py:run_ghost_v1`, issued from `research/gap_fill_stages.py` | The same ghost re-asked with gap-fill v1's section in its brief, once both passes have landed; the plain ghost's shapes under a `_V1` token. |
| `AGENTIC_FETCH_THROTTLED` | `research/agentic/tools.py:_throttled_fetch_outcome` | Per-fetch: a host answered the gap-fill v2 ladder with a rate-limit interstitial under HTTP 200. |
| `AGENTIC_FETCH_LOCAL_DOC` | `research/agentic/local_document.py:log_local_document_read` | Per-document: the gap-fill v2 loop served held PDF or page text locally instead of paying for a Gemini `url_context` call. |
| `AGENTIC_URLCONTEXT_ROBOTS_SKIP` | `research/agentic/tools.py` | Per-URL: the gap-fill v2 paid document read was skipped before spending anything, because robots.txt disallows `Google-Extended`. |
| `OPEN_BOUND_PILING` | `numeric/diagnostics.py:log_open_bound_piling_diagnostics` | Per-forecaster open-bound piling on a numeric declaration. |
| `CLOSE_MARGIN` | `close_margin.py:format_close_margin_marker`, emitted from `forecaster.py` at submit time | Per-question submit-time close margin. |
| `MARKET_RANKING` | `research/prediction_market.py:_log_ranking_telemetry` | Per-question ranked market-retrieval outcome. |
| `MARKET_CHILD_RENDER` | `research/prediction_market.py:_log_child_render_telemetry` | Per-question multi-outcome child-render accounting. |
| `MARKET_RANKING_DEGRADED` | `research/prediction_market.py:_rank_pool` | Per-question ranker fail-open, and why. |
| `MARKET_TIER_CAPPED` | `research/prediction_market.py:_log_tier_caps` | Per-question staleness tier cap. |
| `NUMERIC_DEGENERATE_DECLARATION` | `numeric/pipeline.py:_apply_jitter_and_clamp` | Per-forecaster point-mass numeric declaration. |
| `NUMERIC_AGGREGATE_GRID_MISMATCH` | `numeric/utils.py:aggregate_numeric` | Per-model CDF whose grid length disagreed with the question's. |
| `CDF_MAXSTEP_CLIP` | `numeric/pchip_cdf.py:safe_cdf_bounds` | Per-CDF-build max-step clip. |
| `NUMERIC_PCHIP_FALLBACK` (matches the log line `PCHIP CDF construction failed`) | `numeric/diagnostics.py:log_pchip_fallback` | Per-question PCHIP build failure that fell back to forecasting-tools' own CDF builder. |
| `SPREAD_UNDEFINED` | `spread_metrics.py:numeric_percentile_spread` | Per-question unmeasurable disagreement spread. |
| `TS_ANCHOR_ROUTE` | `research/ts_routing.py:route_question` | Per-question timeseries-anchor routing decision. |
| `FINANCIAL_STALE_LATEST` | `research/financial_data.py:_fetch_yfinance_data` and `research/ts_render.py:_render_single` | Per-identifier stale "latest" disclosure. |
| `FINANCIAL_NOISE_FLAG` | `research/financial_data.py:_volatility_lines` and `research/ts_render.py:_realized_vol_lines` | Per-identifier vendor-noise disclosure. |
| `FRED_UNKNOWN_SERIES` | `research/fred_rendering.py:_fetch_fred_data` and `research/known_api/backends.py:fred_series` | A FRED series id that does not exist, as FRED itself reports it. |
| `ASKNEWS_NO_ARTICLES` | `research/providers.py:_asknews_provider` | Per-question: both AskNews phases came back empty. |
| `RESOLUTION_SOURCE_FETCH` | `research/resolution_source.py:_log_fetch_outcome_markers` | Per-URL Tier-1 page fetch and Tier-2 Datawrapper dataset-hop outcome. |
| `RESOLUTION_SOURCE_ESCALATION` | `research/resolution_source.py` | Per escalated-URL rung attempt. |
| `RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP` / `_UNGROUNDED_SUPPRESSED` / `_NOT_ADDRESSED` | `research/resolution_source.py:_url_context_admission` and `_url_context_rung` | Per-URL accounting for the resolution-source ladder's one paid rung. |
| `RENDERED_FETCH_OFF_HOST` | `research/rendered_fetch.py:render_page` | Per refused render: headless Chromium's main frame landed on a host the DNS pin does not cover. |
| `FORECASTER_DROPS` | `drop_telemetry.py:emit_drop_telemetry` | Per-run summary of which models dropped and why. |
| `SYSTEMATIC_FORECASTER_FAILURE` | `drop_telemetry.py:emit_drop_telemetry` | Per-run, per-model WARN, one line for each model the drops summary lists as systematic. |
| `FORECASTERS_SURVIVED` | `forecaster.py:_research_and_make_predictions` | Per-question positive survivor count. |
| `EXTREME_CALL` | `extreme_call.py:format_extreme_call_markers`, emitted from `forecaster.py` | Per-member extreme binary call. |
| `THIN_PUBLISH_FLOOR` | `aggregation_pipeline.py:_floor_single_survivor_binary` | Per-question single-survivor binary publish floor. |
| `MEMBER_FORECAST` | `member_forecast.py:format_member_forecast_marker`, emitted from `forecaster_runners.py`, `stacking.py`, `aggregation_pipeline.py` | Per-value record of every forecast that leaves a runner. |
| `NUMERIC_AGGREGATE` | `member_forecast.py:format_numeric_aggregate_marker`, emitted from `forecaster.py:_aggregate_predictions` | Per-question record of the published numeric or date distribution. |
| `DEGRADATION_COUNTERS` (matches the log line `Degradation counters:`) | `degradation_counters.py:format_degradation_summary` | The per-run counter set that decides CI color. |
| `PROVIDER_DEGRADATION` | `research/provider_health.py:log_provider_degradation_summary` | Per-run provider-degradation summary. |
| `PUBLISH_HARDENING` | `publish_hardening.py:_wrap_with_timeout_retry` | Per-attempt publish-failure WARN. |
| `PUBLISH_SKIPPED_CLOSED` | `publish_gate.py:skip_publish_if_closed` | Per-question pre-publish skip. |
| `MANTIC_QUESTION` | `mantic.py:_log_mantic_question` | Per-question INFO, one line for every question the Mantic client parses. |
| `MANTIC_POST_DROPPED` | `mantic.py:_count_dropped_post` | Per-post ERROR for a post the framework could not parse. |
| `MANTIC_TOURNAMENTS` | `mantic.py:preflight_mantic_tournaments` | Per-run startup line naming the ongoing bots-only Mantic tournaments. |
| `TIME_BUDGET` | `time_budget.py` | Per-question budget-grant INFO, emitted for every question. |
| `TIME_BUDGET_FAST_PATH` | `forecaster.py` | Per-question WARN when the close-derived budget dropped the optional research stages. |
| `WALLCLOCK_ABORT` | `forecaster.py:_gather_predictions_with_wall_clock` | Per-question WARN when the close-derived budget ran out with forecasters still running. |
| `RESEARCH_PHASE_DEADLINE` | `research/provider_fanout.py:await_providers_within_deadline` | Research-phase deadline WARN. |
| `GAP_FILL_SKIPPED_FOR_BUDGET` | `research/gap_fill_stages.py` | Per-question gap-fill skip when both passes were dropped up front. |
| `GAP_FILL_CUT_FOR_BUDGET` (matches `GAP_FILL_V1_CUT_FOR_BUDGET` / `GAP_FILL_V2_CUT_FOR_BUDGET`) | `research/gap_fill_stages.py` | Per-question mid-phase gap-fill cut. |
| `PAID_PERSONAL_KEY_FALLBACK` (matches the log line `PAID PERSONAL-KEY FALLBACK`) | `fallback_openrouter.py:record_donated_key_fallback` | Per-call donated-to-personal key fallback WARN. |
| `RUN_ALERTABLE_SUMMARY` (matches the log line `Run completed ... with N alertable ...`) | `cli.py` | The end-of-run alertable breakdown, emitted on every path. |
| `ONLY_POSTS` | `cli.py:_tournament_source` | The `--only-posts` smoke filter, one line per run that set it. |
| `QUESTION_CAP_FORFEIT` | `forecaster.py:forecast_questions` | The `max_questions_per_run` cap. |
| `SKIP_GUARD_UNREADABLE` | `forecaster.py:_drop_questions_with_unreadable_forecast_history` | Per-question WARNING: the re-spend guard could not read `my_forecasts`, so the question was dropped rather than treated as never forecast. |
| `GEMINI_UNGROUNDED_SUPPRESSED` | `research/gemini_search.py:_format_grounded_response` | Gemini grounded-search suppression. |
| `GEMINI_GROUNDING_DENSITY` | `research/gemini_search.py:_format_grounded_response` | The floor's complement: one row per response that passed the grounded-chunk floor. |
| `GEMINI_UNSUPPORTED_ATTRIBUTION` | `research/gemini_search.py:_check_attributions` | The embellishment channel, per response. |
| `GEMINI_USAGE` | `research/gemini_search.py`, `research/agentic/tool_backends.py`, `research/resolution_source.py` | Per-call google-genai token and grounded-query accounting for all three Gemini surfaces. |
| `AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` | `research/agentic/tools.py:read_document` | The `read_document` twin of `GEMINI_UNGROUNDED_SUPPRESSED`. |
| `GAP_FILL_ANALYZER_FAILED` | `research/targeted.py:run_gap_fill_pass` | Gap-fill v1's analyzer died. |
| `GAP_FILL_V1_TRIAGE` | `research/targeted.py:run_gap_fill_pass` | Per-question gap-fill v1 triage: how many gaps the analyzer listed, how many the resolver searched, and the count dropped per reason before any spend. |
| `CREDIT_BALANCE` / `CREDIT_SPEND` / `CREDIT_ROLE_SPEND` / `CREDIT_FLOOR_BREACH` | `credit_telemetry.py` | OpenRouter credit balance, spend, per-role spend, and floor-breach markers. |
| `CREDIT_RUN_SUMMARY` | `credit_telemetry.py:log_run_summary` | Per-run, on every path: the role ledger folded to dollars per question, by key, with the run's token totals and largest prompt. |
| `PROMPT_SIZE_ALERT` | `credit_telemetry.py:_alert_on_oversized_prompt` | Per-call WARN: one LLM call's prompt exceeded `PROMPT_TOKENS_ALERT_THRESHOLD`. |
| `DONATED_KEY_STATE` | `credit_telemetry.py:classify_donated_key_state` | Per-run, at most once: the `/auth/key` probe's verdict on the donated key. |
| `LITELLM_CALLBACK_DRAIN_TIMEOUT` | `credit_telemetry.py:drain_litellm_callbacks` | Per-run completeness flag on that run's `CREDIT_ROLE_SPEND` rows. |
| `STACKER_OUTCOME` / `STACKER_SKIP_REASON` / `TOOLS_USED` / `FORECASTERS_USED` | `comment/markers.py` | HTML-comment markers injected into the published Metaculus comment; see "HTML-comment markers" below. |

## Numeric repair tiers, deliberately unharvested

The sentinel_value_audit M16 review asked for one `numeric_repair` marker spanning five repair
surfaces. The repair-tier WARNs in `numeric/bounds_clamping.py` (`Corrected numeric
distribution`, `Heavy bound clamping`, `Cluster spread applied`) never fire on real model output,
because `generate_pchip_cdf`'s uniform-mixture construction pre-enforces the min-step before any
repair tier is reached (0 of 1182 archived numeric forecasts; see `AGENTS.md` "Repair-tier WARN
signals are effectively dead code"). Registering specs for them would archive permanently-empty
files that read as signal, so the omission is a decision, not a miss. M16's incidence question is
answered by the narrower markers already registered: `NUMERIC_DEGENERATE_DECLARATION`,
`NUMERIC_AGGREGATE_GRID_MISMATCH`, `SPREAD_UNDEFINED`, plus `NUMERIC_PCHIP_FALLBACK` for the
surface that does fire. The unit-mismatch withhold rides `FORECASTER_DROPS` rather than its own
marker.

## Parsing notes

The parser matches on the marker token via `re.search`, so it is agnostic to the log-line prefix
(the prod `%(asctime)s - %(name)s - %(levelname)s - %(message)s` format and the ablation
`%(asctime)s %(levelname)s %(name)s | %(message)s` format both work).

`question=` on `GAP_FILL_V2` / `GHOST_PRE` / `GHOST_PRE_JSON` / `GHOST_FORECAST` /
`GHOST_FORECAST_JSON` comes from `log_prefix` (see `agentic_gap_fill.py`'s
`f"question={ref} "`) and is prepended BEFORE the marker token, so it is an optional leading
group on those specs' regexes. On `EXTRACTION_RUNG` / `OPEN_BOUND_PILING` / `CLOSE_MARGIN` /
`MARKET_RANKING`, and most other specs, `question=` is a normal field AFTER the token.

`_RAW_FIELDS` (module-level in `markers.py`) names fields captured as free text or references and
never numerically coerced: `question` is a raw ref (a URL or a bare id), `summary` is the
ghost-forecast free-text summary, `forecast_json` is the compact ghost-forecast JSON blob (kept
verbatim so the scorer's `json.loads` sees it unmangled), and `detail` is free text or a JSON
blob depending on the spec. `MarkerSpec.raw_fields` extends this set per spec (for example
`member_forecast`'s `raw` and `published`, which are compact JSON literals that must not coerce
to a bare float).

`_NONE_SENTINELS` (`{"none", "n/a", "null"}`) are the strings `coerce_value` reads as "no data":
`_fmt` renders `None` as `"n/a"`, `question_id` renders as the string `"None"`, and a stray
`"null"` is handled defensively even though no emitter writes it.

`_KV_PAIR_RE` matches one `key=value` token of a generic counter tail (see `DEGRADATION_COUNTERS`
above): keys are Python identifiers, and values run to the next comma or whitespace.

## Post ID vs question ID (`qid_kind`)

Metaculus posts contain questions, and the two ids diverge on newer posts (post 38880 wraps
question 38195). Marker types are keyed in different spaces, so each `MarkerSpec` declares its
own `qid_kind` and every harvested record carries it: a residual join keyed on one id can then
translate into the record's own space instead of silently dropping the records keyed on the
other (see `metaculus_bot.performance_analysis.id_mapping`). The split is mechanical: a marker
that logs `question.id_of_question` is `question_id`, and one that logs `question.page_url` is
`post_id` (the gap-fill v2 and ghost family, whose `log_prefix` carries the page URL).
`_build_record` stamps `qid` and `qid_kind` onto a record only when the spec captured a
`question` field, so records from the markers with no question ref (the credit and run-level
markers) carry neither key.

`QID_KIND_POST_ID` and `QID_KIND_QUESTION_ID` are local literals in `markers.py` rather than
imports from `metaculus_bot`, so that the parser and the whole `scripts.telemetry` archive stack
stay pure-stdlib: `make sync_telemetry` must not drag in forecasting_tools, numpy or streamlit
just to grep logs. The canonical definitions live in
`metaculus_bot.performance_analysis.id_mapping`, and
`tests/test_id_mapping.py::test_qid_kind_constants_match_markers` pins the two pairs equal.

That per-spec field is the only membership statement. The module docstring in `markers.py` used to
also enumerate which markers were in which space; the list rotted to 12 of the 26 question-keyed
specs, which made the partiality read as if the unlisted ones were keyed some third way. The
current membership is one filter over the registry:

```python
[spec.name for spec in MARKER_SPECS if spec.qid_kind == QID_KIND_QUESTION_ID]
```

## HTML-comment markers

`STACKER_OUTCOME`, `STACKER_SKIP_REASON`, and `TOOLS_USED` (with `FORECASTERS_USED`) are
`<!-- ... -->` markers injected into the published Metaculus comment, not logged to
stdout/stderr; the framework logs only `Posted comment on post N`, never the comment body. They
are therefore almost never present in run logs. Their durable source is the comment itself, which
`metaculus_bot.performance_analysis` already parses. Their specs live in the registry so the
run-log parser stays complete if a comment body is ever logged.

Their zero archived rows in the run-log telemetry archive are a harvesting gap, not dormancy. The
harvester shipped 2026-07-18, after stacking ended (2026-05-29) and `PROBABILISTIC_TOOLS_ENABLED`
went off (2026-07-11). 137 published comments from that window carry `TOOLS_USED=true`. Do not
read their absence from the telemetry archive as signal.

The `ANCHOR_OVERSHOOT_PP` and `CLAUSE_PRODUCT_DIVERGENCE_PP` specs were deleted 2026-09-02 with
the fields that fed them; both had 0 archived rows.

## Marker specs

### EXTRACTION_RUNG

`metaculus_bot/value_extraction.py:_log_extraction`. `qtype` is the BLOCK type: a question type,
or `pmf` for the per-bin block a Mantic enumerable grid is elicited with since 2026-09-09. The
question's own type is on the `MEMBER_FORECAST` line with the same `question` and `model`.
`qid_kind` is `question_id` because `value_extraction.py` emits `question.id_of_question`.

### BLOCK_FALLBACK

`metaculus_bot/value_extraction.py:_run_ladder`, per-forecast, only when the value came from a
candidate other than the first one the best-first walk tried: `skipped` is how many
higher-ranked candidates failed first, `rung` the mechanism that read the winner, and `reasons`
the `" | "`-joined failure list to end of line. The observed shape is a benign trailing
schema-example block, so the rate is the signal, not any one record: a rising rate means the
prompt's block-last contract is eroding.

`reasons` is NOT in `raw_fields`: every entry carries a `block:` or `repair:` prefix, so
`coerce_value` can never read it as a number, bool, or sentinel, and hands the string back
unchanged. `qid_kind` is `question_id` because `value_extraction.py` emits
`question.id_of_question`.

### GAP_FILL_V2

`metaculus_bot/research/agentic/loop.py:_log_completion`. The trailing counter group
(`provenance_rejections` through `conclude_gate_rejections`, added 2026-07-21) is optional:
re-harvesting replays pre-branch logs whose lines end at `lint_rejections`, and a mandatory tail
would drop every one of those records on the next replace-by-run sync. Missing groups coerce to
None. `error` (added 2026-07-23) is nested one level deeper: it is always emitted alongside the
2026-07-21 counters, so it can only appear after `conclude_gate_rejections`, and it captures
greedily to end-of-line because it holds `repr(exc)`, which contains spaces (`error=None` on
healthy runs). `qid_kind` is `post_id` because `agentic_gap_fill.py` emits `question.page_url`.

### GHOST_PRE

Pre-research counterpart to the `GHOST_FORECAST` pair, emitted by
`metaculus_bot/research/agentic/loop.py:_set_research_plan_tool` at plan-set time. `question=`
comes from the same `log_prefix` leading group as `GAP_FILL_V2`, and the `GHOST_PRE:` token
requires the colon so it can't collide with `GHOST_PRE_JSON` under the one-marker-per-line
`break` in `parse_log_text` (the same mechanism that keeps `GHOST_FORECAST` and
`GHOST_FORECAST_JSON` apart). The `GHOST_PRE` to `GHOST_FORECAST` delta measures whether v2's
research moved its own view. `qid_kind` is `post_id` (`agentic_gap_fill.py`'s `log_prefix` is
`question.page_url`).

### GHOST_PRE_JSON

`metaculus_bot/research/agentic/loop.py:_set_research_plan_tool`. `qid_kind` is `post_id`
(`agentic_gap_fill.py`'s `log_prefix` is `question.page_url`).

### GHOST_FORECAST

`metaculus_bot/research/agentic/loop.py:_run_ghost_phase`. `qid_kind` is `post_id`
(`agentic_gap_fill.py`'s `log_prefix` is `question.page_url`).

### GHOST_FORECAST_JSON

Additive full-fidelity companion to `GHOST_FORECAST` (`metaculus_bot/research/agentic/loop.py:_run_ghost_phase`).
`question=` comes from `log_prefix`, the same leading-group mechanism as `GAP_FILL_V2` and
`GHOST_FORECAST`, and `forecast_json` greedily captures the compact single-line JSON payload to
the final `}` (it is the ghost scorer's `json.loads` input, kept verbatim since coercion would
mangle it). The `GHOST_FORECAST_JSON` token can't collide with `GHOST_FORECAST:` (the latter
requires a `:` immediately after `GHOST_FORECAST`), so the two specs stay mutually exclusive
under the one-marker-per-line `break`. `qid_kind` is `post_id`.

### GHOST_FORECAST_V1

`metaculus_bot/research/agentic/loop.py:run_ghost_v1`, issued by `research/gap_fill_stages.py` after
both gap-fill passes have landed. Since 2026-09-09 the driver is asked a second private forecast whose
brief is the plain ghost's plus gap-fill v1's section (`## Targeted Gap-Fill (second pass)`), sent on
the loop's transcript up to, not including, the plain ghost's prompt, with the same tool list and
`tool_choice="none"`, so it re-reads the cached prefix and never sees its own first ghost. The line
has exactly the `GHOST_FORECAST` shape (`qtype=`, `summary=`) under the `GHOST_FORECAST_V1` token, so
`scripts/score_ghosts.py` reads both with one parser. It is emitted only when v1's section exists on
the question AND the plain ghost ran, so every record has a `GHOST_FORECAST` partner in the same run;
with gap-fill v1 off there is no line. Paired with the plain ghost it measures v1's marginal value on
the driver, the mirror of the `GHOST_PRE` to `GHOST_FORECAST` read that measures v2
(docs/agentic_gap_fill.md "The ghost forecast"). `question=` comes from the same `log_prefix`
mechanism, so `qid_kind` is `post_id`. The token cannot collide with `GHOST_FORECAST:` (that spec
requires the colon right after `GHOST_FORECAST`) or with `GHOST_FORECAST_JSON:`.

### GHOST_FORECAST_V1_JSON

The v1 ghost's full-fidelity companion, `GHOST_FORECAST_JSON`'s shape under the `_V1` token, same
emitter; suppressed when no structured block parsed. Together with `GHOST_FORECAST_JSON` from the same
run it is the input of the `v1_pairs` read in `scripts/score_ghosts.py`. Also written into the research
archive as the `ghost_v1` key of the `gap_fill_v2` payload (None when the v1 ghost did not run), beside
`ghost`. `qid_kind` is `post_id`.

### AGENTIC_FETCH_THROTTLED

Per-fetch: the gap-fill v2 fetch ladder read a 200-OK body that was the host's rate-limit
interstitial rather than the page it asked for. The shared ladder detects the body; the tool handler
emits the marker through `research/agentic/tools.py:_throttled_fetch_outcome`.
Registered because the event had no trace at all before it, and its whole failure mode is looking
like a success: on q45191, two throttled ogimet.com fetches were served to the driver as
`status: ok`, cached, and replayed on its own retry, so the exact-date reference class it
published came to 4 years instead of 6. No `question=`: the tool handlers run below the loop's
`log_prefix` and have no question id, exactly like the credit markers, so a join goes through the
run id.

`phrase` is the entry of `fetch_ladder.throttle.FETCH_THROTTLE_PHRASES` that fired and is last in the
regex because it contains spaces; with `chars` (the body's length) it is what lets a prod fire be
graded a true or false positive, and the phrase list and the `FETCH_THROTTLE_PAGE_MAX_CHARS` cap
are retuned on evidence rather than taste.

### AGENTIC_FETCH_LOCAL_DOC

Per-document: the gap-fill v2 loop served held PDF or page text without paying for it
(`research/agentic/local_document.py:log_local_document_read`). Registered because it is how the
PDF-first local-read change is measured: before it, every PDF the driver met went to a paid Gemini
`url_context` read, and the only trace of one was the spend. `method` separates the two local
routes: `pdf_local` is a fetch serving a PDF's extracted text (which paginates, so it selects
nothing); `digest_local` is a `read_document` answering an ask from BM25-selected passages of PDF
or page text already held. The later ZIP/workbook/Word readers and image views do not emit this
marker; their tool outcomes, selectors, and image IDs remain in the archived research transcript,
and `GAP_FILL_V2` keeps its existing line shape.

`chars` is the local text held, not the window or digest block handed to the driver, so one
figure is comparable across both routes and against `URL_CONTEXT_SIZE_GATE_TOKENS` (`chars / 4`).
`pages` is n/a for a page with no page structure; `passages` is n/a on a `pdf_local` line and, on a
`digest_local` line, is the field that says whether the digest actually answered: 0 means the
document does not discuss what was asked, which reads in the block itself as an ordinary
successful read. No `question=`: the tool handlers run below the loop's `log_prefix` and have no
question id, exactly like the throttle marker above, so a join goes through the run id.

### AGENTIC_URLCONTEXT_ROBOTS_SKIP

Per-URL: the gap-fill v2 paid document read was skipped before it spent anything, because the
host's `robots.txt` disallows `Google-Extended`, the product token Gemini's `url_context`
retrieval identifies as, so that read is refused at the host and returns nothing whatever it
costs (proven live 2026-09-03 on internationalaisafetyreport.org, against a robots-allowed host
that retrieved on the identical call). Registered because the pre-check spends one free request
per host to save a paid call, and only these lines say how often it fires: a fire is a call not
billed, and a suspiciously high rate would mean the group parser is over-matching and withholding
reads we could have had.

`host` rides beside `url` because the verdict is cached and applied per host, so the host is the
unit any rate is computed over. No `question=`: the tool handlers run below the loop's
`log_prefix` and have no question id, exactly like the two markers above, so a join goes through
the run id.

### OPEN_BOUND_PILING

`metaculus_bot/numeric/diagnostics.py`. `qid_kind` is `question_id` (`numeric/diagnostics.py`
emits `question.id_of_question`).

### CLOSE_MARGIN

`metaculus_bot/close_margin.py`, emitted at submit time in `forecaster.py`. `qid_kind` is
`question_id` (`close_margin.py` emits `question.id_of_question`).

### MARKET_RANKING

Per-question ranked market-retrieval outcome (`research/prediction_market.py:_log_ranking_telemetry`).
This is the port's own post-ship instrument and the reason it needs a spec rather than a manual
grep: `rendered`'s pool indices answer whether the ranker's attention decays down a
~400-candidate prompt and whether Manifold detail enrichment shifts which rows get picked, and
`prompt_chars` gives the free prod distribution against the ranker's prompt ceiling. Run logs
expire from GHA at 90 days, so an unharvested line is an unanswerable question later.

`rendered` is a comma-joined `venue:pool_index@rank` list with no spaces, so `\S+` takes the
whole field; the "none" sentinel (no rows rendered) coerces to None, which reads correctly
alongside `rows=0`. A pool index of -1 means the row could not be traced back to a pool entry.
`qid_kind` is `question_id` (`prediction_market.py` emits `question.id_of_question`).

### MARKET_CHILD_RENDER

Per-question multi-outcome child render accounting (`research/prediction_market.py:_log_child_render_telemetry`).
A separate line rather than extra fields on `MARKET_RANKING`, because that regex is not
end-anchored and a separate spec keeps this harvester change purely additive.

Two fields carry the questions this exists to answer. `withheld` counts the prices the venue
parsers refused as manufactured: an empty Kalshi book, a Polymarket placeholder leg at Gamma's
`["0.5","0.5"]` default, or a Manifold answer at its untouched prior. The Kalshi half of that is
gated on `KALSHI_NO_PRICE_SPREAD`, a threshold calibrated on eleven fixture strikes, so its prod
incidence has to be a query rather than a guess. `max_stage` and `ladder_chars` say whether the
ladder's section allowance binds on real slates (0 means every outcome named, 99 the per-family
hard bound).

`named` plus `collapsed` equals `outcomes` is the completeness invariant the render guarantees, so
a harvested line where those disagree is a render bug and not a tuning signal. `qid_kind` is
`question_id`.

### MARKET_RANKING_DEGRADED

Per-question ranker fail-open (`research/prediction_market.py:_rank_pool`), the discriminating
sibling of `MARKET_RANKING`'s `outcome=failopen`. That field says a fail-open happened; only this
line says which failure, and the distinction is the whole finding: `reason=shape_regression`
means the ranker's output was structurally unreadable to us (a renamed index key, indices all out
of range), which before 2026-08-25 was reported as `ok(0)`, indistinguishable from the model
deliberately answering "none of these markets bear on the question", which the render turns into
an affirmative forecaster-facing sentence. `reason=unreadable` means the completion was not a
ranking array at all.

Not end-anchored: `detail=` is free text holding the exception's `str` (spaces, semicolons,
quotes), captured verbatim via `_RAW_FIELDS` and optional so a terser future form still harvests
rather than dropping the record. `qid_kind` is `question_id`.

### MARKET_TIER_CAPPED

Per-question staleness tier cap (`research/prediction_market.py:_log_tier_caps`, over
`market_retrieval.ranking.cap_stale_top_tier`). Silent on the no-cap case, so a harvested record
means the ranker graded a market that stopped trading more than
`MARKET_STALENESS_TIER_CAP_DAYS` before the question opened as `same_quantity_same_date`, the
claim a long-closed market cannot make. The demotion also rides the archived snapshot as
`MarketMatch.tier_cap_note`, so the incidence is answerable offline too; this line is the prod-log
half and the one that survives a snapshot the research archive never captured.

It fires on nothing in the 102 archived snapshots, and would not have fired on q45163 either (that
row was graded one tier lower; see `AGENTS.md`'s prediction-market paragraph), so a first record
is itself the finding. `capped` is a comma-joined `venue@rank` list with no spaces, so `\S+` takes
the whole field. `qid_kind` is `question_id`.

### NUMERIC_DEGENERATE_DECLARATION

Per-forecaster point-mass numeric declaration (`numeric/pipeline.py:_apply_jitter_and_clamp`): the
model put (near-)identical values at every percentile. What happens next depends on the grid, and
`spread_applied` says which. On the 201-point continuous grid the cluster spreader is deliberately
NOT applied (`spread_applied=false`) and the honest zero span reaches the unit-mismatch guard,
which withholds that forecaster, so there the count is a per-model fabrication-attempt rate:
before 2026-08-25 the spreader manufactured a +/-6-unit distribution from it, and that width was
exactly what let it pass the guard, so the published forecast stated a width nobody declared.
Where the published bins are the outcome space (a discrete question, or any non-201 grid such as
a day-granularity date question; see `numeric/config.py`'s `grid_is_outcome_space`), the collapse
IS spread like any other plateau, under the one-bin cap (`spread_applied=true`): "100% on
2026-09-16" is fully expressible on twelve one-day bins, so the member publishes with its mass
inside the bin it named. The 201-grid drop shows up as `UnitMismatchError` in `FORECASTER_DROPS`;
this line is the only place the cause is named.

`model` names the forecaster (or the stacker, on the aggregation path). All three
`sanitize_percentiles` callers pass it, so "unknown" now means a NEW caller forgot to; it is not
in `_NONE_SENTINELS`, so it stays a readable string rather than coercing into an absent field.
`qid_kind` is `question_id` (`numeric/pipeline.py` emits `question.id_of_question`).

### NUMERIC_AGGREGATE_GRID_MISMATCH

Per-model grid-length disagreement inside ensemble aggregation (`numeric/utils.py:aggregate_numeric`).
Expect zero records in prod: every model's CDF is built on the question's own point count, so a
record means a length drifted, which used to matter far more than a resample, because the old
group-by-value aggregation silently medianed over a rotating subset of the ensemble whenever an
ft-fallback distribution mixed with PCHIP ones (their x-axes differ by float rounding, and on a
log-scaled question by construction). Aggregation is positional now, so the mismatch is handled
rather than silent; this marker is what keeps "handled" from meaning "unnoticed". `qid_kind` is
`question_id` (`numeric/utils.py` emits `question.id_of_question`).

### CDF_MAXSTEP_CLIP

Per-CDF-build max-step clip (`numeric/pchip_cdf.py:safe_cdf_bounds`): a bin whose mass exceeded
the server's per-bin cap (`0.2 * 200 / N`) was clipped and the excess moved elsewhere. Not a bot
defect and not alertable: the cap is the platform's, so a spike above it is simply unpublishable.
What this measures is how much of a published forecast's shape the repair owns: q45065
(2026-08-01) capped three forecasters who all declared ~0.72 on the resolving count and, under the
old slack-proportional policy, scattered 47% of the mass past 35 deaths where the ensemble had put
~2%. It logged at DEBUG, so the 2026-07-15 "repair-tier WARNs never fire" audit never saw it, and
the reshaping left no trace in any run log.

`bins_displaced` and `max_offset_bins` are the fields that make the policy itself auditable:
nearest-first packing puts the excess a bin or two away (q45065: 4 bins, offset 2), while the
retired policy touched nearly every bin on the grid. A record whose `max_offset_bins` runs into
the tens means the neighbours were already at cap, i.e. a genuinely wide declaration, not a
scattered spike.

`model` is the forecaster whose declaration was clipped, or an `ensemble_*` label
(`ensemble_median` / `ensemble_mean` / `ensemble_discrete_snap`) for the aggregation stages; the
ablation/pooling callers pass none and read "unknown". `qid_kind` is `question_id`
(`numeric/pchip_cdf.py` is handed `question.id_of_question`).

### NUMERIC_PCHIP_FALLBACK

Matches the log line `Question <id>: PCHIP CDF construction failed (<error>), falling back to
forecasting-tools default`, per question (`numeric/diagnostics.py:log_pchip_fallback`): the
distribution the forecasters see came from forecasting-tools' fallback CDF builder, not our PCHIP
pipeline. The one numeric repair surface with a confirmed prod fire (the repair-tier WARNs
upstream of it are dead code on real output; see "Numeric repair tiers, deliberately unharvested"
above), so its absence from the archive was the one genuine blind spot in that family. The
question ref is `id_of_question`, rendered "N/A" when absent (a `_NONE_SENTINELS` member, coerces
to None). `qid_kind` is `question_id`.

### SPREAD_UNDEFINED

Per-question unmeasurable disagreement spread (`spread_metrics.py:numeric_percentile_spread`): the
normalizing denominator was non-positive, so no spread could be computed. It returns `inf` now;
before 2026-08-25 it returned `0.0`, which `route_after_forecasts` reads as "the models agree", and
the comment marker then claimed `spread_below_threshold`, a measurement failure read as an
affirmative agreement signal.

Latent in prod while the three per-type stacking gates are off, but it fires in backtests and
ablation, where it marks a question whose routing decision was made on no measurement at all.
`qtype` is a field rather than a literal so a future binary/MC variant of the same shape harvests
with no spec change. `qid_kind` is `question_id` (`spread_metrics.py` emits `question.id_of_question`).

### TS_ANCHOR_ROUTE

Per-question timeseries-anchor routing decision (`research/ts_routing.py:route_question`). This
exists because routing was near-unauditable from telemetry: `route_question` logged only the
ambiguous/guard branches, so of the triple era's 30 route-level misses exactly 2 left any line in
1,800 run logs, and 27 were the silent `kw_no_keyword_hit` return that left no log line at all;
part of the research-archive-qa dimension could only be written by re-running the router offline.
One line per numeric/discrete question makes anchor coverage a query.

`decision` is `routed` or `skipped`; `step` names the deciding branch on a route (`url_single` /
`url_spread` / `kw_single`) or the reject reason on a skip (`url_ambiguous`, `url_quantity_gate`,
`url_change_vs_level_guard`, `url_no_relative_return_wording`, `kw_no_keyword_hit`,
`kw_derivation_gate`, `kw_ambiguous`, `kw_change_vs_level_guard`). `series` is the series involved
where one is known: comma-joined on ambiguity, slash-joined on a spread, the "none" sentinel
(-> None) on a plain keyword miss. All values are spaceless, so `\S+` takes each.

Denominator caveat: the marker covers routing-eligible questions, not every numeric/discrete
question. `build_anchor_section` returns before `route_question` when `scheduled_resolution_time`
is missing or non-datetime, and a disabled `TS_ANCHOR_ENABLED` run emits nothing. A coverage query
must reconcile against the run's question list rather than treat absent lines as skips. `qid_kind`
is `question_id` (`ts_routing.py` emits `question.id_of_question`).

### FINANCIAL_STALE_LATEST

Stale "latest" observation behind a rendered anchor value, WARNING-level and informational, NOT
alertable (the render already tells the forecaster to treat the value as stale; this line makes
each surface's prod incidence a query instead of a guess). Two emitters share one shape because
they share the estimator (`ts_estimators.stale_latest_age_days`): `surface=financial_data` is
`financial_data.py`'s `_fetch_yfinance_data`, `surface=ts_anchor` is `ts_render.py`'s
`_render_single`.

`symbol` is a ticker or FRED series id: carets and dots (`^GSPC`, `BRK.B`) are spaceless, so `\S+`
takes it and `coerce_value` keeps it a string. `cadence` is the daily-step unit the age was judged
against (trading-day / calendar-day). No question ref: the fetch is per-identifier, and one
question can fire several, so `qid_kind` stays None.

### FINANCIAL_NOISE_FLAG

Vendor-noise flag on a rendered volatility, the sibling of `FINANCIAL_STALE_LATEST` and
non-alertable for the same reason: the render already tells the forecaster the one-day-return
volatility is inflated, so this line exists to make each surface's prod incidence a query rather
than a guess. The two emitters again share one shape because they share the estimator
(`ts_estimators.variance_ratio` / `multi_period_annualized_vol_pct`): `surface=financial_data` is
`financial_data.py`'s `_volatility_lines`, `surface=ts_anchor` is `ts_render.py`'s
`_realized_vol_lines`.

`vr` is the Lo-MacKinlay overlapping variance ratio at lag `vr_lag` over the provider's full held
history; a random walk reads ~1.0 and the flag fires below `floor`. `robust_vol` is the volatility
measured on overlapping `vr_lag`-step returns (the flagged block's headline figure) and reads
"None" when the estimator refused the sample. `long_vol` is the long-horizon one-day-return
volatility and reads "None" on the `ts_anchor` surface, which computes no long window at all,
exactly as it does on a yfinance series too short to hold one; read `surface` to tell those apart.
Every field is required: one shared emitter (`research/noise_flag.py:noise_flag_line`) means one
shape, so a future field reorder harvests as a clean zero rather than recording None for a value
that WAS emitted, which an optional group in the middle of same-shaped `\S+` fields would do.

`symbol` is the ticker or FRED series id the flagged volatility was computed on, in the same field
position its stale-latest sibling carries it. It is required, not optional-wrapped: the marker
ships in the same diff as this spec, so no archived record predates the field. Without it every
record was anonymous, and the fan-out is one thread per ticker up to
`MAX_FINANCIAL_IDENTIFIERS` with nondeterministic line order, so two flagged tickers in one run
were byte-identical apart from `seq`, with no join to the stale-latest record for the same series
and no way to tell a pegged-cross true positive from a `^GSPC` false positive at n=1.

No question ref: like its stale-latest sibling the flag is per-identifier (one question can fire
several) and neither call site has the question in scope, so `qid_kind` stays None.

### FRED_UNKNOWN_SERIES

A FRED series id that does not exist, as FRED itself reports it (`400 "The series does not
exist"`, surfaced by fredapi as a `ValueError` carrying that body). Emitted by
`fred_rendering.py:_fetch_fred_data` on the live path and by the known-API FRED backend; the
keyless benchmarking fetcher cannot tell a bad id from a vintage predating the series, so it stays
silent rather than guessing.

Not alertable: a hallucinated id is the classifier's habit, not a bot crash, and the provider
degrades to whatever its other identifiers returned. What the marker buys is the incidence: q45363
lost its whole financial block to `DEXBOUS` with only the ambiguous `DEXBOUS:empty` source token
to show for it, which reads identically to a live series with no observations.

`proposed_by` splits the causes, which want different responses: `classifier` means an LLM invented
the id (the prompt's FX routing rule is the fix point), `resolution_url` means the question's own
resolution criteria link a dead FRED page, and `gap_fill_driver` means the deterministic known-API
gap-fill path was given an unknown id.

No question ref: the fetch runs in a per-identifier `to_thread` worker with no question in scope,
the same limitation its `FINANCIAL_STALE_LATEST` / `FINANCIAL_NOISE_FLAG` siblings carry, so one
question can fire several lines and `qid_kind` stays None.

### ASKNEWS_NO_ARTICLES

Per-question WARN (`research/providers.py:_asknews_provider`): both AskNews phases came back
empty, so the provider returned `""` rather than a prose "no articles" sentence, with the hot and
historical counts. The `articles: empty(no_articles)` loss token it records beside this line
travels only with the question's `provider_results` row in the research archive; this is the
run-log record. `qid_kind` is `question_id` (`providers.py` logs `question.id_of_question`).

### RESOLUTION_SOURCE_FETCH

One line per fetched URL, emitted at the per-question aggregation point in the provider (that is
where the question id exists; threading it down through the monkeypatched fetch surface would
change every signature for a log line). Before this marker the per-URL outcomes lived only in
free-text log lines and the comment's provider-diagnostics block, so a cut like "cdc.gov is 0
successes in 1,069 fetch records" meant re-scraping GitHub Actions logs that expire at 90 days.
The free-text `resolution_source fetched <netloc> (<status>)` lines it replaces were deleted, so no
fetch is recorded twice.

`status` is `ok` for a success and the verbatim `FetchStatus` otherwise (`blocked` / `js_wall` /
`no_resolving_content` / `stale_data` / ...), the same token the provider-diagnostics source map
uses, minus that map's dataset-`stale_data`-to-`none` amnesty: telemetry keeps the reason
verbatim. Since the escalation ladder (2026-09-03) it may be a rung's verdict rather than the
direct fetch's: the Wayback rung's `stale_data` where the direct fetch said `blocked` / `error` /
`not_found`, or the paid reader's `ungrounded` where it said `blocked` / `js_wall` / `error` /
`no_resolving_content`. An era-bucketed `blocked` rate off this field alone shows a drop at that
merge that is bookkeeping, not hosts refusing us less; the direct outcome is `from_status` on the
sibling escalation line, and `route` partitions the two populations. `http` is `n/a` when no
response ever arrived (timeout, client error, SSRF rejection). `embeds` names the routeless
data-embed providers found in the page's raw HTML, or the `none` sentinel (which harvests as
None); it is what makes an unreadable-embed page queryable on the qids 44554/44556 shape, where
the page carried real prose and the fetch was a legitimate `ok`.

Tier-2 dataset hops ride this marker too and are told apart by `url`: every dataset is
`static.dwcdn.net/data/<chart_id>.csv`, a host reachable no other way, so a query partitions cited
pages from hop artifacts on it.

`reason` (optional, 2026-09-02) disambiguates a status that has more than one rule behind it:
`no_resolving_content` is `embed_shell` when the page named a routeless data embed, `thin_page`
when the extraction was simply under the chrome floor (the population the floor gained when it
stopped being gated on a named provider), and `no_matching_passage` when a cited document read in
full discusses nothing the question asks about, the one member that is a document rather than a
page. `unreadable_document` splits into `no_text_layer` / `encrypted` / `malformed`, and
`unsupported_type` carries `budget_skipped` / `parse_contention` when it was a document we were
holding and declined to parse. `blocked` carries `metaculus_self_ref` when the refusal was ours, a
redirect onto the question platform's own site, rather than the host's. The provider appends it
only where it applies, so the group is optional in both directions: absent on every line the
archive already holds, and absent on a fresh line whose status carries no reason.

`route` (optional) names which rung of the escalation ladder produced the recorded outcome:
`direct` for the plain fetch, and `meta_refresh` / `impersonate` / `pdf_local` / `derived_api` /
`rendered` / `wayback` / `url_context` for an escalated one. Without it a rescued page is
indistinguishable from one the direct route read, so "what did the ladder actually buy" is not a
query. Both optional groups are keyed and at the tail, in that order: an optional group sitting
between same-shaped `\S+` fields silently records None for a value that WAS emitted, and a keyed
tail group cannot mis-claim its neighbour's value, so a line carrying `route` but no `reason`
parses correctly.

`failure_class` / `exc` / `server` (all optional, 2026-09-03) are the failure diagnostics that
separate an egress-reputation refusal from a host fault: a small token vocabulary (`http_403` /
`http_4xx` / `http_5xx` off the response; `tls` / `dns` / `timeout` / `connection` / `decode` /
`malformed_response` off the transport exception, the last added 2026-09-04 for a response
aiohttp's parser refused, such as an undecodable Content-Encoding or an oversized header, which
recorded as `connection` before), the exception class name, and the `Server` header lower-cased
with internal spaces collapsed to `_`. Keyed and tail-positioned after `route` in that fixed
order, each emitted only when present, so an old parser and every archived line still parse and a
line carrying a later field but not an earlier one cannot mis-claim a neighbour's value. `qid_kind`
is `question_id` (`resolution_source.py` emits `question.id_of_question`).

`passages_returned` / `passages_grounded` / `fallback_used` (all optional, 2026-09-09) are the page
digest's counters (`research/page_digest.py`; docs/research.md "Page digest"): how many passages the
`page_digest_extractor` model answered with, how many of those were literal substrings of the page
after whitespace and quote-glyph normalisation, and whether the passages served after the page's
opening came from the BM25 fallback instead (`True` / `False`, harvested as a bool). The difference of
the two counts is the model's fabrication count for that page. Both counts are `0` with
`fallback_used=True` on every fallback that has no answer to count: no call was made (the remaining
wall was below the floor, or the page was empty), a call was made and failed before answering (the
timeout, a provider error, an off-schema or empty completion), or the model answered with an empty
list. The three fields therefore cannot separate a skipped call from a failed one; the run log's
`PAGE_DIGEST` lines, one per fallback naming its reason, carry that distinction. Keyed and
tail-positioned after `server` in that fixed order, appended by the callers only on a fetch that ran
the digest, so a page under the digest threshold and every archived line still parse with the three
harvested as None. The spec was added ahead of its emitter: the fetch-ladder unification's callers
append the three values off the `PageDigest` the digest returns.

`caller` (optional, 2026-09-10) names which of the two callers of the shared fetch ladder emitted
the line: `resolution_source` for the resolution-source fetcher, `gap_fill_v2` for the gap-fill v2
agentic loop, whose `fetch` and `read_document` tools moved onto the same ladder and gained this
per-URL record they never had. Last in the keyed tail, after the optional digest fields, for the same reason every
group before it is keyed and tail-positioned: every archived line parses unchanged, and an absent
value keeps meaning "does not apply" rather than "old record". A `gap_fill_v2` line carries
`question=None` (the loop's three event markers do the same, because the tool call has no question
id in hand), so a join to a question goes through the run id. The marker NAME reads as a slight
misnomer for a loop fetch, and names are contracts here, so the misnomer is the correct price.

### RESOLUTION_SOURCE_ESCALATION

One line per escalated-URL rung attempt: the direct fetch could not read the page, so the ladder
tried a heavier route. Its sibling `RESOLUTION_SOURCE_FETCH` records only the final per-URL
outcome, so on its own it cannot say whether a rung rescued the page, how many rungs were spent,
or what the attempt cost; `wall_s` is the field that decides whether a rung earns its place on a
question under a close-derived time budget.

`from_status` is the verbatim `FetchStatus` that triggered the escalation, so the trigger
population is queryable without joining back to the fetch marker. Its domain is per rung, and the
pairs are disjoint by construction: `js_wall` / `no_resolving_content` for `meta_refresh`,
`derived_api`, and `rendered` (a page that answered 200 with nothing readable); `blocked` for
`impersonate` (a direct 403 re-dialed under a browser's TLS fingerprint, since 2026-09-04; its
`outcome` is `blocked` again for a host that refused the impersonated client too, `success` for a
rescue, `js_wall` for one while the direct status stands, `not_found` for a 404 or 410 under
impersonation, or `error` for any other non-200 answer, the two `_NON_OK_FETCH_STATUS` verdicts
and its default, so a `rung=impersonate outcome=error` row is an edge answering the impersonated
client 5xx rather than an emitter bug); `unsupported_type` for `pdf_local` (the content-type
router's verdict before the `%PDF-` sniff, including a document the impersonated retry fetched,
which pairs its own `pdf_local` line with the `impersonate` one); `blocked` / `error` / `not_found`
for `wayback` (a page our address never read); and `blocked` / `js_wall` / `error` /
`no_resolving_content` for `url_context`. `blocked` never pairs with a browser rung, since
Chromium dials from the same address. `rung` names the route tried. `outcome` and `wall_s` are
that rung's own, stamped as it closes (`RungAttempt`): `outcome` is the status that stood once the
rung was over, its rescue, its verdict (`stale_data`, `ungrounded`), or the direct status it left
standing when it declined, and `wall_s` is what that rung alone cost. So on a page where a dead
feed GET was followed by a rescuing render, the first line reads the direct status and the second
reads `success`, which is what keeps a rung that fires often but rescues nothing distinguishable
from one that never fires at all. Two rungs measure `wall_s` narrower than their whole footprint:
the local PDF read stamps it inside the parse gate, so queueing for a slot is not billed to the
parse, and the paid rung opens its attempt only after its `Google-Extended` robots.txt pre-check
(a real request, bounded at `ROBOTS_FETCH_TIMEOUT_S`), so that pre-check is not in its `wall_s`, a
15-30% under-count against the rung's 15s floor when the pre-check has to fetch. Skipped attempts
emit no line at all and ride `details["counts"]` instead.

`caller` (optional, 2026-09-10) is the same field the fetch marker carries, with the same two
values and the same reason for sitting keyed at the tail; see that section.

The token cannot collide with `RESOLUTION_SOURCE_FETCH`: both specs match on their own full marker
word plus the colon, and neither word is a prefix of the other, so the one-marker-per-line `break`
in `parse_log_text` cannot mis-route either line whichever order they sit in. `qid_kind` is
`question_id`.

### RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP

Per-URL: the resolution-source ladder's paid `url_context` read was skipped before it spent
anything, because the host's `robots.txt` disallows `Google-Extended`, the product token Gemini's
retrieval identifies as, so the read would have been spend with a known-zero return
(`research/resolution_source.py:_url_context_admission`). The resolution-source twin of
`AGENTIC_URLCONTEXT_ROBOTS_SKIP`: the same pre-check and the same per-host cache
(`research/robots_policy.py`), kept as a separate spec because the two surfaces have different
trigger populations, and this one fires only for a cited resolution URL every free rung failed to
read. Registered on 2026-09-04, when `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` went on in every bot
workflow; until then the line could not fire in production, and a spec would have archived an
always-empty column, so no run from before that merge carries a record.

A fire is a paid call not billed, so it must not read as a failure; a rate far above the handful
of hosts publishing the directive would mean the group parser is over-matching and withholding
reads we could have had. `host` rides beside `url` because the verdict is cached and applied per
host, so the host is the unit any rate is taken over. No `question=`: the rung runs per cited URL
inside its provider with no question in scope, exactly like the `GEMINI_USAGE` row for the same
read, so a join goes through the run id.

### RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED

Per-URL: a paid `url_context` read on the resolution-source ladder came back with zero successful
retrievals, so its text was discarded as `ungrounded` rather than rendered under the
primary-grading-evidence caption (`research/resolution_source.py:_url_context_rung`). Gemini
answers fluently out of parametric memory when every retrieval failed, and this section tells the
forecasters what the resolution source says, so the suppression is the same floor
`GEMINI_UNGROUNDED_SUPPRESSED` and `AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED` apply, and the three
rates read as one family. The read WAS billed, so each record is money spent on nothing served,
the figure that says whether the rung's trigger population earns its cost. Registered with its two
siblings on 2026-09-04, for the same reason.

`statuses` is the comma-joined list of `url_retrieval_status` values the SDK reported, or the
`none` sentinel when it attached no `url_metadata` at all (which harvests as None through
`coerce_value`). It splits a retrieval that was attempted and failed for a nameable reason from
one the tool never made. Required rather than optional (the v2 twin's tail is optional for
archived pre-field lines): the emitter has always written it, and no production line exists from
before this spec. No `question=`, for the same reason as the robots-skip line above.

### RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED

Per-URL: a paid `url_context` read retrieved the page but answered with the prompt's
`NOT_ADDRESSED` sentinel, the model's designed reply when the page does not discuss the ask, so
the read was withheld as `no_resolving_content` / `not_addressed` instead of rendered
(`research/resolution_source.py:_url_context_rung`). Rendered, it was prose standing in for an
absent section, the shape the PDF digest closes with `no_matching_passage`. Distinct from the
ungrounded line above: the page WAS retrieved, so Gemini can reach the host, and the money bought
a true negative rather than nothing. `host` because the rollout question is which hosts Gemini
reaches but finds nothing on, the population a sharper ask or a different rung would recover.
Registered with its two siblings on 2026-09-04; no `question=`, as on both of them.

### RENDERED_FETCH_OFF_HOST

Per refused render: headless Chromium's main frame ended up on a host other than the one its DNS
pin covers, so the DOM was refused unread, or discarded unpublished when the navigation committed
during the read itself (`research/rendered_fetch.py:render_page`; the landing is checked before
and after `page.content()`). Read fail-shut, so Chromium's own error document after a failed
navigation (`landed_host=chromewebdata`) is refused too, which makes the record an upper bound on
hostile landings. A server-side redirect hop is dialed by the browser with no route handler of
ours involved, so the landing is reached through Chromium's own resolver, outside every check the
transport makes; a `same_publisher=false` record therefore says a cited page tried to send us
somewhere the pin does not cover, a security-relevant event rather than a data-quality one. A
`same_publisher=true` record is the other population strict hostname equality refuses: a benign
client-side hop inside the publisher's own registrable domain (`example.com` to `www.example.com`),
and its rate prices the strictness of the rule rather than measuring hostile landings.

This is the only per-event record of it. The resolution-source caller counts the refusal under
`render_off_host_skips` in its `details["counts"]`, a per-question total that names neither the
pinned host nor the landing, and a skipped rung emits no `RESOLUTION_SOURCE_ESCALATION` line at
all; the gap-fill v2 caller has no count of its own. Registered 2026-09-04 with the check itself,
so no archived run carries a record and a first one is itself the finding.

`scope` is the transport's `memo_scope`, `resolution_source` or `gap_fill_v2`: the render path is
shared by both callers, whose URL populations differ, and the count only one of them keeps cannot
tell them apart. `landed_host` is a hostname and never the landing URL, which can carry a session
token or a credential, and it is `None` (harvested as no data) for any landing with no hostname at
all: an http(s) URL with an empty authority, or a non-http(s) scheme the fail-shut guard refuses
(`data:`, `file:`, `blob:`), every one of which matches no pin and so fails closed into this
refusal. Chromium's own error document is the one non-http(s) shape that does carry a hostname,
`chromewebdata`. `same_publisher` is `registrable_domain(landed_host) == registrable_domain(pinned_host)`
over the vendored public-suffix list (`research/public_suffix.py`), the same judgment the XHR
harvest makes about a page's own JSON, with IP literals compared exactly and `false` whenever the
landing has no hostname. It was added on 2026-09-04, the day the marker was registered and before
any run emitted it, so no archived record lacks the field and extending the line is contract-safe.
No `question=`: the transport runs per URL with no question in scope, so a join goes through the
run id.

### FORECASTER_DROPS

Per-run ensemble-drop summary emitted by `drop_telemetry.py:emit_drop_telemetry`. No per-question
ref (it aggregates a whole run), so `qid_kind` stays None. `detail` is a compact
model-to-cause-to-count JSON blob captured verbatim (it is in `_RAW_FIELDS`) so the `/`-laden
OpenRouter slugs and nested counts survive; `systematic` is a comma-joined model list (or the
"none" sentinel, coercing to None).

### SYSTEMATIC_FORECASTER_FAILURE

Per-run, per-model WARN from `drop_telemetry.py:emit_drop_telemetry`, one line for each model the
`FORECASTER_DROPS` summary lists as `systematic`. `dropped_on_questions` is how many questions the
model dropped on, `qids` is the comma-joined QUESTION ids, and `causes` is the `cause:count` pairs.
`qids` is the field the summary line does not carry; it always holds two or more ids, so it stays
one string. There is no `question=` ref, so `qid_kind` is None. The regex ends at the `causes`
value; the emitter appends a prose tail after an em dash, which the pattern deliberately does not
try to capture.

### FORECASTERS_SURVIVED

The positive per-question counterpart to `FORECASTER_DROPS`, emitted by
`forecaster.py:_research_and_make_predictions` once the survivor set is known. Unlike
`FORECASTERS_USED` (the HTML marker further down, which carries the same count but lives in the
published comment and so is effectively never in a run log), this one is on stdout, which makes
historical survivor counts queryable from the telemetry archive alone.

`models` is a comma-joined slug list; OpenRouter slugs contain no spaces, so `\S+` takes the whole
field. It is deliberately NOT in `_RAW_FIELDS`: a comma-joined string coerces to itself
(`coerce_value` only converts numeric-looking text), and the "unknown" sentinel is not in
`_NONE_SENTINELS`, so it survives verbatim either way. `qid_kind` is `question_id`
(`forecaster.py` emits `question.id_of_question`).

### EXTREME_CALL

Per-member extreme binary call, emitted by `extreme_call.py:format_extreme_call_markers` from
`forecaster.py:_research_and_make_predictions`, right after `FORECASTERS_SURVIVED` (which is this
marker's denominator: only extreme members get a line, so a rate needs the survivor list from
`FORECASTERS_SURVIVED` plus the question's type).

`lone` is the finding, not `p`: a member at 0.03 with nobody else in the extreme band on the same
side behaves nothing like one whose neighbour agrees (4 of 9 right versus 21 of 23;
`scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` section 2), and it was re-derived by
hand from parsed comments every residual round before this line existed. `survivors` rides along
because "lone" is vacuous at k=1, so a cut can drop those records instead of joining out to
another marker to find them. Those 4-of-9 / 21-of-23 counts come from the memo's own scripts,
which read "lone" as "no other extreme member on either side"; this marker uses the same-side rule
the memo's prose states, and `extreme_call.py`'s own docstring measures where the two part company
before anyone pools old and new counts.

Binary questions only; a dominant MC option is a different measurement. `model` is a bare
display-name slug (spaceless), "unknown" when the forecaster's reasoning carried no `Model:`
prefix; `lone` is rendered lowercase and `coerce_value` lowercases before its bool test, so it
harvests as a bool. `qid_kind` is `question_id`.

### THIN_PUBLISH_FLOOR

Per-question single-survivor binary publish floor, emitted by
`aggregation_pipeline.py:_floor_single_survivor_binary` from the base-combine re-entry, only when
the lone survivor's value actually moved: `raw` is the member's declared probability (what the
comment's summary bullet still carries) and `clamped` is what was published
(`THIN_PUBLISH_BINARY_FLOOR` / `_CEIL` in `constants.py`). A lone value already inside the band
leaves no line, so this marker's count IS the floor's prod incidence; the single-survivor event
itself is `FORECASTERS_SURVIVED`'s `survived=1`.

`survivors` is always 1 today and rides along so the record stays self-describing if the k<=2
generalisation the receipt discusses
(`scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` section 3, "1=") is ever enabled: a
cut can then split the two regimes without a join. `raw` and `clamped` are `%.4f`, so
`coerce_value` reads them as floats. `qid_kind` is the same id space as `FORECASTERS_SURVIVED` /
`EXTREME_CALL`.

### MEMBER_FORECAST

Per-value record of every forecast that leaves a runner (`member_forecast.py:format_member_forecast_marker`):
`raw` is what the extraction ladder read off the rationale, `published` what the runner returned
after its clamp (binary), clamp-and-renormalise (MC), or sanitise (numeric). `role` is `member`
for an ensemble forecaster and `stacker` for the meta-forecaster, whose numeric line is emitted by
`aggregation_pipeline.py` where its percentiles are sanitised.

This is the only marker that carries a member's value on every question. Before it (2026-09-02)
the raw value existed solely inside the published comment's per-rationale fenced block,
middle-trimmed at `COMMENT_CHAR_LIMIT`, only present since 2026-05, and recoverable for 74 of 451
resolved binaries when the clip-threshold re-read needed it. `EXTREME_CALL`'s `p` covers only
members past the extreme band, and `THIN_PUBLISH_FLOOR` only the lone-survivor case.

`raw` and `published` are compact JSON literals with no whitespace (a float, `[p1,p2,...]` in
`question.options` order, or `[[percentile,value],...]` with the percentile as a decimal), so
`\S+` takes each whole; they are in this spec's `raw_fields` so the archive holds them verbatim
and a consumer always `json.loads`, otherwise a binary line would coerce to a float while the MC
and numeric vectors stayed strings. `model` is `.+?` like `EXTRACTION_RUNG`'s, since the same
`forecaster_llm.model` feeds both.

`oor_low` / `oor_high`: the out-of-range mass of the CDF the runner built from `published`,
`cdf[0]` and `1 - cdf[-1]`, on numeric and date lines only. Optional in the regex so every line
that predates the fields and every binary / MC line (no CDF, no fields) still harvests, with both
fields None on those records.

`elicitation`: `pmf` when the member declared one probability per bin, on which `raw` and
`published` are the platform's `N + 2` PMF vector. Its own optional group; None means percentiles.
`qid_kind` is `question_id` (every emitter passes `question.id_of_question`).

### NUMERIC_AGGREGATE

Per-question record of the published numeric or date distribution
(`member_forecast.py:format_numeric_aggregate_marker`, emitted from `forecaster.py:_aggregate_predictions`,
the one seam every aggregation path returns through): the grid it was submitted on and its
out-of-range mass, `cdf[0]` below the lower bound and `1 - cdf[-1]` above the upper. The platform
scores an out-of-range resolution against a fixed 0.05 reference (a 1% tail scores -80.5, 5%
scores 0); Mantic's Series 1 resolved half its date questions and a quarter of its discrete ones
outside the displayed range, and this pipeline builds exactly 1% there whenever every percentile
sits inside, so on a Mantic question the published aggregate's open tails are raised to
`MANTIC_OUT_OF_RANGE_TAIL_FLOOR` (`numeric/out_of_range_floor.py`).

`oor_low` / `oor_high` are the published tails, after that floor. The trailing `oor_low_raw` /
`oor_high_raw` are the aggregate's own tails before it, and `tail_floor` the level the moved tails
were actually raised to (the floor, or less where the other tail's mass left the interior only its
min-step minimum; 0 on Metaculus, on a closed-bound question, or when nothing moved), so the
floor's cost and gain can be replayed on this bot's own forecasts; joined with the per-member
`oor_*` fields on `MEMBER_FORECAST` they also say whether the models place mass beyond the bounds
on their own. The three are one optional group at the tail, so lines that predate them harvest
with all three None.

`method` (2026-09-09): the rule the members were combined by, on every numeric and date question,
authoritative over the configured strategy and the comment's `STACKER_OUTCOME`: `mean` (the
pointwise mean of the members' CDFs: every per-bin question, and any run whose strategy is MEAN),
`median`, `stacked`, `single`, or `unrecorded` (a bug signal). `elicitation=pmf` on
`MEMBER_FORECAST`, not `method`, says a question was elicited per bin. Its own optional group;
earlier lines read None. `qid_kind` is `question_id` (`forecaster.py` passes `question.id_of_question`).

### DEGRADATION_COUNTERS

The per-run summary that decides CI color (`degradation_counters.py:format_degradation_summary`,
matching the log line `Degradation counters:`; `cli.py` exits non-zero on a positive
`alertable_count`), emitted by `forecaster.py:forecast_questions`. No per-question ref (it
aggregates a whole run), so `qid_kind` stays None.

The tail is parsed generically: `kv_pairs` captures the whole `key=value, key=value` list and
`_build_record` tokenizes it, so a new counter in `format_degradation_summary` harvests with no
change here (the old 17-named-group `$`-anchored regex needed a coordinated two-file edit per
counter, and getting it wrong dropped the whole line's harvest). Historic renames survive as
their own keys exactly as before (`research_provider_timeouts` era-records keep that spelling).
One deliberate delta from the old spec: a key absent from an era's line is now absent from the
record rather than explicitly None; read `record.get(key)` and treat both as "this era didn't
emit it", never as a measured zero.

`gap_fill_v1_errors` counts analyzer, schema and resolver failures reported by the v1 gap-fill
pass. A valid empty analysis and gaps dropped by legitimate triage grades leave it at zero; the
counter is separate from `gap_fill_v2_errors` so a v1 failure remains attributable. Both are
alertable and the forecast still publishes before the CLI exits non-zero.

`tests/test_degradation_counters.py` pins the shape of this line, and three of its pins are worth
knowing before you edit either side. It drives fifteen of the alertable terms with a distinct power
of two each, so the resulting sum names exactly which subset was counted and a missing or
double-counted term is visible rather than merely wrong; `publish_skipped_closed` is the one term
that sum leaves at zero, and the same file pins its presence on the line separately. Four terms are
not bot attributes, so the test stubs what the property reads rather than driving the counter: three
are read-only accessors the bot property imports (`kalshi_catalogue_fetch_failures` and
`prediction_market_source_losses` on `prediction_market`, `provider_degradation_count` on
`provider_health`) and the fourth is the `publish_hardening._PUBLISH_ATTEMPT_FAILURES` module global.
And the suite asserts the line ENDS with the newest counter, `research_budget_cuts`. That last pin
dates from the `$`-anchored regex, where a key appended past the optional-group tail dropped the whole
line's harvest; under the generic `kv_pairs` parse the order no longer decides whether a record
harvests, so what the assertion records now is which counter is newest.

What the counters mean, moved here from the `format_degradation_summary` docstring.
`research_budget_cuts` is the off-fast-path complement of `time_budget_fast_path`: a question whose
window cleared the fast-path threshold but whose research window still cut a provider or a gap-fill
pass (deduplicated per question, orchestrator-side); without it that band's degradation was
invisible to the counter contract and the all-clear census. The three publish-side counters mean
three different things, which is why they are three keys: `questions_failed_to_publish` counts
questions the min-forecasters floor kept from attempting publication; `publish_attempt_failures`
counts attempted POSTs that exhausted the publish-hardening retry budget (the q45085 405 shape the
old counter could not see); `publish_skipped_closed` counts questions whose publish the close-time
gate skipped before any POST, so latency cost the question. The oldest key's name is misleading but
stays, because renaming it would silently drop the field from every historical record the parser
replays. `time_budget_fast_path` is the fourth member of that family and the earliest of them: it
counts questions whose close time was too near for the full pipeline's worst case, so the optional
research stages were dropped to protect the prediction POST. The three above fire once a publish
has already failed or been withheld; this one fires while the question is still savable, which is
why it is worth alerting on separately.

### PROVIDER_DEGRADATION

Per-run provider-degradation summary (`research/provider_health.py:log_provider_degradation_summary`),
the positive/negative counterpart to the `provider_degradation` counter on `DEGRADATION_COUNTERS`.
Aggregates a whole run, so `qid_kind` stays None. Emitted even at `findings=0`, which makes a
measured zero a recorded fact rather than an absent line.

Two live signals feed `findings`, and both describe a provider that POPULATED but degraded rather
than one that failed: `market_field_contract` (a declared liquidity field dead across 100% of a
venue's pool rows) and `catalogue_empty` (a prefetch reported success and handed the local matcher an
empty catalogue). A third rule, `venue_no_contribution`, was deleted 2026-08-04 with its receipts in
`research/provider_health.py`'s module docstring, so a venue contributing nothing is now counted only
when its own prefetch catalogue came back empty.

`detail` is a compact JSON array of findings captured verbatim (it is in `_RAW_FIELDS`): venue and
field names are delimiter-hostile, and residual analysis `json.loads` it. The trailing suppression
clause is free text after the JSON, so `detail` stops at the array's closing bracket.

The observation denominators (`venues_observed` / `catalogues_observed` / `pool_rows`, added
2026-08-24) are optional-group wrapped: re-harvesting replays the ~1039 archived lines that
predate them, and on those a missing group coerces to None, which reads correctly as "not
recorded", never as a measured zero. They exist because `findings=0` alone is byte-identical
between a run that evaluated 400 pool rows and one that evaluated nothing. All three at zero is
therefore the "nothing was recorded" reading, distinct from the healthy zero of a run that evaluated
a populated pool, and `tests/test_provider_health.py` pins both shapes.

### PUBLISH_HARDENING

Per-attempt publish failure WARN (`publish_hardening.py:_wrap_with_timeout_retry`). Two emitted
shapes share the prefix: `"PUBLISH_HARDENING: <method> attempt N/M timed out after Ts"` and
`"PUBLISH_HARDENING: <method> attempt N/M failed (<ExcType>: <msg>)"`. The failed shape in the
tests is copied from question 45085's real run on 2026-08-03, the 405 "already closed to
forecasting" rejection whose crash also took out that run's end-of-run summary, so the publish
failure left no harvestable trace at all. The `attempt N/M` clause is what keeps this spec off the
other `PUBLISH_HARDENING`-prefixed strings in that module (the applied-INFO line, the seam-moved
`AttributeError`s, the loop-exited `RuntimeError`). Exactly one of `timeout_s` / (`error_type`,
`error`) is populated per record; the counter it complements is `publish_attempt_failures` in the
degradation line, but only this marker names which method died and with what. No question ref
(the wrapper sees only the POST), so `qid_kind` stays None.

### PUBLISH_SKIPPED_CLOSED

Per-question pre-publish skip WARN (`publish_gate.py:skip_publish_if_closed`). The counterpart to
`PUBLISH_HARDENING`: that marker fires when a POST was attempted and died, this one when the gate
saw the window had closed and made no POST at all. Both point at the same underlying problem
(latency against a question's close deadline), and this is the one that names the question and by
how many seconds it missed, which is what a latency analysis needs and what `CLOSE_MARGIN` alone
cannot say (a negative margin there does not distinguish "published late but accepted" from
"never published"). `overdue_s` can be negative under `reason=state_closed`, meaning the question
was shut ahead of its scheduled close. `qid_kind` is `question_id` (`publish_gate.py` emits
`question.id_of_question`).

### MANTIC_QUESTION

Per-question INFO from `mantic.py`, one line for every question the Mantic client parses
(competitions.mantic.com runs a fork of the Metaculus platform). Mantic adds question fields the
framework does not model: `multi_resolution` (scored against several resolution values, e.g.
eleven daily prices averaged), `date_granularity` (day/week), and `precision` (bin width), so
without this line they are gone at the 90-day GHA log expiry. `type` is the type as it arrived on
the wire: `quantitative` is Mantic's Series 2 merger of numeric and discrete, which the client
rewrites to `discrete` before parsing, so this field is the only record of which questions were
rewritten. `cdf_size` is the parsed grid (Mantic allows up to 2,001 points against Metaculus's
201); `post` rides along because a group subquestion's post id differs from its question id.
Absent values render `n/a` and harvest as None; the booleans render lowercase. `qid_kind` is
`question_id` (`mantic.py` emits `question.id_of_question` as `question=`). The test file's example
lines come from `tests/data/mantic_preseason2_posts_2026_09_08.json`, the real Preseason 2 posts
recorded on 2026-09-08.

### MANTIC_POST_DROPPED

Per-post ERROR from `mantic.py:_count_dropped_post`: a post the framework could not parse into a
question. Before this line the framework's per-post loop logged one warning and moved on, so a
new Mantic type string or a missing field forfeited a whole question class on every run with
nothing harvestable. Now the drop is counted and logged before the error is re-raised into that
loop, which still swallows it as a warning, and the same counter reddens the run
(`docs/operations.md` "Parse drops are counted"). `post` is the post id as it arrived, `type` the
wire `question.type` read with `.get` (`n/a` when absent), and `error` the exception class. A
post-level census rather than a per-question record: the post never became a question, so there is
no question id and no qid is stamped.

### MANTIC_TOURNAMENTS

Per-run startup line from `mantic.py:preflight_mantic_tournaments`, off the one authenticated GET
of `/api/projects/tournaments/` that also confirms the token's forecast permission. `ongoing` is
the comma-joined sorted slugs with `is_ongoing` true, `configured` is `MANTIC_TOURNAMENT_ID`, and
`new` the ongoing bots-only slugs that are NOT the configured one (WARNING when non-empty, INFO
otherwise; `none` for an empty list). It exists because a zero-question run is green: without it a
Series 2 slug could open and every hourly run would keep fetching the ended preseason silently.
Run-level, so no question ref; several slugs stay one comma-separated string.

### TIME_BUDGET

Per-question budget grant INFO (`time_budget.py`), emitted for every question including the roomy
ones. That is the point: `CLOSE_MARGIN`, the only other close-time telemetry, is emitted after a
successful submission, so it is censored on exactly the thin-window questions the budget exists
for (q45085 had 22 seconds of headroom and appears in no `CLOSE_MARGIN` record). This marker is
the uncensored denominator: how often a window is actually thin, and how often the fast path
fires. The test's thin line replays question 45085's real close time against a fetch 20 minutes
out, which less the 60-second publish reserve is the 1140-second budget it carries.

`close_limited` says the close time, not the static `PER_QUESTION_WALL_CLOCK_DEADLINE`, set the
budget; `fast_path` says the optional research stages were dropped, and is the per-question detail
behind the `time_budget_fast_path` counter in the degradation line. `qid_kind` is `question_id`
(`time_budget.py` emits `question.id_of_question`).

### TIME_BUDGET_FAST_PATH

Per-question WARN (`forecaster.py`) when the close-derived budget dropped the optional research
stages. The INFO `TIME_BUDGET` line carries the same fact as a field; this is the loud half
`docs/operations.md` tells the operator to grep, and without a spec it vanished at the 90-day GHA
log expiry. `close_time` is a datetime repr with an internal space, so it captures up to the
semicolon rather than as one `\S+` token. `qid_kind` is `question_id` (`forecaster.py` emits
`question.id_of_question`).

### WALLCLOCK_ABORT

Per-question WARN when the close-derived budget ran out with forecasters still running
(`forecaster.py:_gather_predictions_with_wall_clock`): seconds elapsed, how many of the configured
forecasters had finished, how many were cancelled, and the budget left, negative once overrun.
Each cancelled member also lands in `FORECASTER_DROPS` as `timeout_wall_clock`, but that is a
per-run count; this is the per-question record of a missed deadline. The emitter spells the ref
`qid=` and the regex names the group `question` so the record is stamped like every other
question-keyed marker. `stacking_route.py` reuses the token on a prose line, `WALLCLOCK_ABORT:
skipping stacking for Q ...`, which the `qid=` anchor in this spec's regex deliberately excludes;
that skip is archived as the `wall_clock_budget` `STACKER_SKIP_REASON` instead. `qid_kind` is
`question_id` (`forecaster.py` emits `question.id_of_question`).

### RESEARCH_PHASE_DEADLINE

Research-phase deadline WARN (`research/provider_fanout.py:await_providers_within_deadline`): the
outer budget bound cancelled straggler providers. Carries no question ref (the line names counts
and provider names only), so `qid_kind` stays None; the cancelled providers also survive as
`status="deadline"` rows in the archive's `provider_results`, which is where per-question
attribution lives.

### GAP_FILL_SKIPPED_FOR_BUDGET

Per-question gap-fill skip (`research/gap_fill_stages.py`): both passes dropped up front, either
on the fast path or because the research phase had no budget left. `research_phase_remaining` is
"n/a" (fast path, never computed) or "NNNs". `qid_kind` is `question_id` (the orchestrator logs
`question.id_of_question`).

### GAP_FILL_CUT_FOR_BUDGET

Per-question mid-phase gap-fill cut (`research/gap_fill_stages.py`): the pass started and was then
cancelled at the research-phase deadline, the one budget event recoverable from nothing else once
GHA logs expire (the up-front skip above and the fast path both have their own records).
`gap_fill_pass` is `V1` or `V2` (the spec matches both `GAP_FILL_V1_CUT_FOR_BUDGET` and
`GAP_FILL_V2_CUT_FOR_BUDGET`). `qid_kind` is `question_id` (the orchestrator logs
`question.id_of_question`).

### PAID_PERSONAL_KEY_FALLBACK

Matches the log line `PAID PERSONAL-KEY FALLBACK`, per-call donated-to-personal key fallback WARN
(`fallback_openrouter.py:record_donated_key_fallback`, the shared accounting seam for every
donated-first call path, whose non-404 branch emits this warning). The counters it feeds are
already in `DEGRADATION_COUNTERS` and the cli summary, but only this line names which model fell
back and with what error, which is what separates "one flaky Gemini call" from "every forecaster
ran on the paid key". `error` captures greedily to end-of-line because it holds the exception's
`str`.

### RUN_ALERTABLE_SUMMARY

The end-of-run alertable breakdown, matching the log line `Run completed ... with N alertable
degradation event(s) ...` (`cli.py`), emitted on every path: the green fully-suppressed case is
exactly the one that would otherwise leave no record (the 2026-07-26 drained-key run read
`alertable=0` alongside real degradation). `donated_key` is the `/auth/key` probe verdict and is
optional-group wrapped twice over: it is omitted entirely when no spend-cap failure made the
wrapper probe, and the suppression clause between it and `credit` only appears mid-window.

`outcome` captures the literal "clean" that `cli.py` adds when nothing degraded at all
(2026-08-25; before then such a run logged no line, so the census counted only degraded runs, a
gap the drained-key window hid because every run in that window fell back at least once and so
always emitted a line). It is None on every other shape, including all pre-2026-08-25 records; the
alternation is purely additive. The token, rather than all-zero fields, is what marks a run clean:
a run that lost a question to a raising `log_report_summary` emits all zeros too (q45085's shape)
and deliberately keeps the plain phrase.

`mantic_post_drops`: posts the Mantic client could not parse, folded into `alertable` and rendered
only when non-zero, so absent harvests as None.

### ONLY_POSTS

The `--only-posts` smoke filter (`cli.py:_tournament_source`), one line per run that set it: the
post ids asked for, the open ones the tournament actually held among them, and how many open
questions the filter left out. It is what says which question a one-question paid run spent on.
`requested` / `matched` are comma-separated post ids (a lone id coerces to int, several stay one
string) and an empty match is `none`.

### QUESTION_CAP_FORFEIT

The `max_questions_per_run` cap (`forecaster.py:forecast_questions`), one WARNING per run that cut
the tightest-close-first list: which platform's questions, the cap, how many were open after the
skip filter, how many were left behind and their post ids. Run-level, so no question ref. `posts`
is comma-separated post ids in the order the cap dropped them (a lone id coerces to int, several
stay one string). On Mantic the fetch ceiling is 500 against a default cap of 10 and a whole
hour's batch opens at once, so each row is a paid forecast the run never made; the only trace
before this spec was a free-text WARNING gone with the 90-day GitHub Actions log expiry.

### SKIP_GUARD_UNREADABLE

The fail-shut leg of the skip-previously-forecasted guard
(`forecaster.py:_drop_questions_with_unreadable_forecast_history`, called from
`forecast_questions` whenever `skip_previously_forecasted_questions` is on, which `cli.py` pins for
every tournament-shaped mode). One WARNING per dropped question: the question id, the post id, the
platform (`metaculus` or `mantic`) and `reason`, today always `my_forecasts_missing`. The framework
derives `already_forecasted` inside a blanket except that answers False, so a payload with no
`my_forecasts` field (a list GET without `with_cp=true`, a Mantic read that lost its token, an API
change) would read as never forecast and an hourly run would re-forecast and re-publish the whole
tournament. With an external dispatcher adding firings the guard has to fail shut instead, so such a
question is dropped before any spend; a present field with an empty `history` stays eligible. The
drop is otherwise invisible, since the framework's own skip logs nothing, and the free-text count
line that follows it (`Dropped N question(s) with no readable my_forecasts field`) is not a marker.
Observed live 2026-09-09: authenticated `with_cp=true` reads carry the field on every question on
both platforms, and the guard has never fired historically (1,054 bot comments on 1,054 distinct
posts), so this is prevention. `qid_kind` is `question_id`; `post_id` rides beside it so the
dropped post can be named without a join. What to do when it fires is in `docs/operations.md`
"Scheduling reliability".

### GEMINI_UNGROUNDED_SUPPRESSED

Gemini grounded-search suppression (`research/gemini_search.py:_format_grounded_response`):
`google_search` returned no grounding chunks and no `url_context` read succeeded, so the section
is dropped as ungrounded parametric output. The orchestrator then records `status="empty"`, which
is NOT alertable and bumps no counter, so this WARN is the only signal, and without a spec the
suppression rate was unmeasurable from the archive. `qid_kind` is `question_id`
(`gemini_search.py` passes `question.id_of_question`).

### GEMINI_GROUNDING_DENSITY

The floor's complement (`research/gemini_search.py:_format_grounded_response`): one row per
response that passed the grounded-chunk floor, carrying how thinly the passing text is
attributed. Post-floor the median response has one grounding support per ~872 chars and 41% of
passers carry <=3 supports, which is the surface the floor cannot see and where the embellishment
rate lives. Deliberately telemetry and never a gate: a decisive true figure once came out of a
1-support response, so nothing keys on these values; they exist so "did embellishment move" is a
query over the archive. `chars` is the raw model text, so `supports / chars` reproduces the
audit's density denominator. `qid_kind` is `question_id`.

### GEMINI_UNSUPPORTED_ATTRIBUTION

The embellishment channel, per response (`research/gemini_search.py:_check_attributions`):
outlet-named source-tier tags (`[A: NASA]`, `[B: Reuters]`) that the same response's own
grounded-domain list does not name, rewritten to `[unverified attribution]` at format time. 70%
(478 of 681) of the outlet-named tier attributions in the 323 archived Gemini sections are that
shape under the shipped keep-biased matcher (86% under the audit's looser rule; receipts in
`scratch/next_season_bundle_2026-09/item4_attribution_check/VALIDATION.md`), and the zero-chunk
floor cannot see any of them (it fires only when nothing grounded at all), so before this the rate
was a hand audit. `labels` is load-bearing context, not decoration: the same `unsupported` count
reads completely differently against it (q38195 named 21 outlets over one grounded domain), and
`groups` is the render footprint, below `unsupported` because several unsupported names in one
bracket collapse to a single marker. Emitted only when `unsupported` > 0; a checked response with
none logs nothing and carries its zero in the research archive's provider details instead. Not
alertable: an absent outlet is the model's habit, not a bot defect. `qid_kind` is `question_id`.

### GEMINI_USAGE

Per-call google-genai accounting for all three Gemini surfaces: grounded search
(`research/gemini_search.py`, role `grounded_search`), gap-fill v2's `read_document`
(`research/agentic/tool_backends.py`, role `read_document`), and the resolution-source ladder's
paid `url_context` rung (`research/resolution_source.py`, role `resolution_source`, which emits
only from runs with `RESOLUTION_SOURCE_URL_CONTEXT_ENABLED` on: every bot workflow since
2026-09-04, and no run from before that merge). None routes through OpenRouter, so none shows up
in `CREDIT_ROLE_SPEND`, and the whole Google AI Studio side of a run's spend was unmeasurable from
the archive, which matters because grounding is metered against a monthly grounded-prompt
allowance per project, billed per query on overage, and any feature that multiplies grounded calls
re-eats that pool (the spring-2026 billing arc). `role` names the surface, so they are separable
without keying on the model.

Every token field can read `n/a`: the SDK's `usage_metadata` fields are individually optional, and
a missing count must harvest as None rather than as a measured zero, which is exactly what the
`n/a` sentinel does through `coerce_value`. `search_queries` is the grounded-query count (the
billable unit on overage). On the `read_document` surface it reads a genuine `0`, not `n/a`: the
SDK omits `web_search_queries` when the search tool issued none, and an absent list IS a count of
none, the honest reading for a `url_context`-only read. The resolution-source rung reads it the
same way, being `url_context`-only too. It reads `n/a` only when the grounding metadata could not
be walked at all. So a spend query filters the surfaces on `role`, never on 0-versus-n/a in this
field.

The ledger covers completed responses only. `log_gemini_usage` runs after the SDK returns, so a
Gemini call that timed out or raised billed unknown tokens and emitted no row (14 of 154 archived
`read_document` calls, 9.1%, hit that handler). A spend total from these rows is therefore a lower
bound, biased toward undercounting the largest calls; the denominator is
`provider_results['gemini_search'].status` per question plus `research_provider_failures`, never
this marker's row count.

`question` is optional and last: the grounded-search call site has the question in scope and
passes `question.id_of_question` (hence `qid_kind`, matching its `GEMINI_GROUNDING_DENSITY`
sibling), while `read_document` runs as a per-URL tool below the loop's log prefix with no
question at all, and the resolution-source rung runs per cited URL inside its provider with none
either. A keyed tail group is what lets one spec serve all three without recording None for a
field that WAS emitted. `qid_kind` is `question_id`.

### AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED

The `read_document` twin of `GEMINI_UNGROUNDED_SUPPRESSED` (`research/agentic/tools.py:read_document`):
Gemini's `url_context` tool retrieved nothing, so the answer would be unsourced recall and the
"fetched" verification tier is withheld. Worth measuring separately because a "fetched" document
discrepancy is the only kind that enters the artifact's SUPERSEDE block, i.e. the one that tells
every forecaster to override the briefing. Carries no question id (`read_document` is a per-URL
tool with no question in scope), so the URL was for a long time its only field.

`statuses` (optional) is the comma-joined list of `url_context` retrieval statuses the SDK
reported for that call, or the `none` sentinel when it reported none at all (which harvests as
None, the same reading an archived pre-field line gets). It splits the two causes a bare
suppression cannot: the tool tried and the fetch failed for a nameable reason, versus the tool
never retrieved anything to report on. Optional and at the tail so every line the archive already
holds parses byte-identically.

### GAP_FILL_ANALYZER_FAILED

Gap-fill v1's analyzer (`research/targeted.py:run_gap_fill_pass`) died, which gates the whole
pass: the addendum is silently `""` and the run looks identical to a question that legitimately
had no gaps. Gap-fill isn't one of the orchestrator's `_run_one` providers, so it has no
`ProviderResult` and no `lost=` token; this marker is the only durable signal, and v1's searches
are one of the largest research spend lines (~44%). `detail` captures greedily to end-of-line
because it holds the exception's `str`. `qid_kind` is `question_id` (`targeted.py` passes
`question.id_of_question`).

### GAP_FILL_V1_TRIAGE

Per-question gap-fill v1 triage (`research/targeted.py:run_gap_fill_pass`, added 2026-09-09), emitted
once for every question whose analyzer answered, so `listed=0` is "the analyzer answered and produced
no parsable gaps" (an unparseable reply lands there too, traced only by the `GapFill: could not parse
analyzer JSON` warning in the run log) and a dead analyzer is `GAP_FILL_ANALYZER_FAILED` alone.
`listed` is the analyzer's slot count (a malformed item keeps its slot), `kept` the number the
resolver searched, and the five `dropped_*` counts partition the rest, one reason per gap in the
order the triage applies them: `dropped_not_answerable` (the analyzer graded `answerable_now` false:
the gap can only be answered by an observation not yet made or a result not yet published),
`dropped_in_first_pass` (`already_in_first_pass` true: the first pass already states the value with
its date), `dropped_same_need` (`same_need_as` named an earlier gap whose need a kept gap is
searching or the first pass answers), `dropped_schema` (one of the two required booleans was omitted
or mistyped, or a typed `same_need_as` named no earlier position, or the slot was malformed, and the
gap is dropped rather than read as passing; a missing `same_need_as` key reads as null and does not
drop the gap) and `dropped_over_cap` (a survivor past `GAP_FILL_MAX_GAPS`, which applies after the
filter). `listed = kept + sum(dropped_*)` on every line. The line is INFO, except when every listed
slot dropped as `dropped_schema`, which is WARNING: the analyzer has stopped emitting the grades and
v1 has gone dark while it still bills, the same outcome `GAP_FILL_ANALYZER_FAILED` warns about; the
level is not part of the marker contract, so the harvester reads both. Receipt: about a third of
v1's resolver calls bought nothing on the archive
(`scratch/cost_pass_2026-09-09/v1_gap_redundancy/REDUNDANCY.md`), and this marker is how that share
is measured once the filter is live; the dropped gaps themselves, with position and reason, ride the
raw research record (`provider="gap_fill"`, key `dropped`). The rules and the receipts:
`docs/research.md` "v1 triage". `qid_kind` is `question_id` (`targeted.py` passes
`question.id_of_question`).

### CREDIT_BALANCE

`credit_telemetry.py`.

### CREDIT_SPEND

`credit_telemetry.py`. `source` (added 2026-07-27) names which branch produced the delta and so
how much to trust it: `remaining_delta` is reliable, `usage_delta_unsettled` is a lower bound
(settlement lag; see `credit_telemetry.py`'s module docstring), `unavailable` means no delta. The
group is optional because re-harvesting replays pre-2026-07-27 logs whose lines end at
`remaining=`; a mandatory tail would drop every one of those records on the next replace-by-run
sync. Missing coerces to None, which reads correctly as "this run predates the field".

### CREDIT_ROLE_SPEND

Per-(role, key) decomposition of the run's OpenRouter spend, read off OpenRouter's own per-call
usage accounting (`credit_telemetry.py`, "Per-role dollar attribution"; the field semantics and
their receipts are in `docs/operations.md` "Per-role spend"). `usd` is `n/a` when no call of that
row carried cost data, never a fabricated zero, and `costed_calls` says how many of `calls` the
sum covers. `byok_usd` is the sum of `usage.cost_details.upstream_inference_cost`, the
upstream-provider charge OpenRouter reports beside its own `usage.cost`. Roles are the string literals
each builder call site passes through `credit_telemetry.llm_call_metadata` (`forecaster:<vendor>`,
`parser`, `native_search`, ...), enumerated only in `docs/operations.md` "Per-role spend";
`untagged` means a completion nobody stamped.

`usd` is `usage.cost + upstream_inference_cost` summed over the row, the definition it has had
since the marker shipped on 2026-09-03, and it is kept in place because the field's meaning may
not change. It DOUBLE COUNTS a non-BYOK call: OpenRouter's docs say `upstream_inference_cost` is
0 or null off BYOK, but since at least 2026-09-03 it is reported on non-BYOK calls too, equal to
`cost`, while only `cost` is drawn from the key. The 2026-09-09 cost pass proved it on the three
production runs whose only personal-key row was the Google forecaster slot: the key's settled
usage moved by that row's `byok_usd` to the cent (0.57 against `usd=1.1433`, 0.14 against
0.2787, 0.67 against 1.3268), so the slot costs $0.13 a question, not $0.27
(`scratch/cost_pass_2026-09-09/cost_anatomy.md` section 2). A pre-2026-09-09 personal-key row
whose `usd` is twice its `byok_usd` is that double count.

`charged_usd` and `byok_calls`, added 2026-09-09, are the correction: `charged_usd` sums `cost`
plus, on calls whose `usage.is_byok` is true, `upstream_inference_cost`, so it is the money
actually charged across both payers (OpenRouter credits, and the BYOK account's owner for the
upstream part); `byok_calls` is how many of `calls` routed BYOK. On a BYOK row `charged_usd ==
usd`; on a non-BYOK row `charged_usd == usd - byok_usd`. A row with `byok_calls=0` but
`byok_usd > charged_usd` on the donated key would mean OpenRouter stopped sending `is_byok`,
which is then visible rather than silently zeroed. Consumers sum `charged_usd`
(`scripts/reconcile_credit_spend.py` falls back to `usd` on the 44 older rows). Both 2026-09-09
tails are optional regex groups, so those older rows still harvest with the new fields as None.

Token fields, added 2026-09-09 and summed over every call of the row (costed or not):
`prompt_tokens` and `completion_tokens` are the base counts; `cached_tokens` is
`usage.prompt_tokens_details.cached_tokens`, the prompt tokens the provider served from its prompt
cache, and `reasoning_tokens` is `usage.completion_tokens_details.reasoning_tokens`, the hidden
reasoning output billed at the completion rate. Each detail count is 0 when the provider reports
nothing, so `cached_tokens / prompt_tokens` is the row's cache hit rate and
`reasoning_tokens / completion_tokens` its reasoning share. They exist because the 2026-09-09 cost
pass had to FIT the gap-fill v2 driver's output tokens and could only infer that its prompt cache
was active (`scratch/cost_pass_2026-09-09/v2_cost_anatomy.md`); with these fields both are read
straight off the ledger. The tail is one optional regex group, so the 44 rows archived before it
still harvest, with the four token fields coerced to None ("this run predates the field").

`max_prompt_tokens`, added later on 2026-09-09 as a third optional tail, is the largest single
prompt among the row's calls. It exists because `prompt_tokens` is a sum: the gap-fill v2 driver's
41k-token last research turn is invisible inside its 300k-token row total, and the packet-size
question ("is any single prompt approaching the size that degrades the model?") needs the maximum,
not the sum. `PROMPT_SIZE_ALERT` below is the same measurement fired per call when it crosses the
threshold; this field is how a run that never fired still reports how close it came.

### CREDIT_RUN_SUMMARY

One INFO line per run from `credit_telemetry.py:log_run_summary`, emitted from the same `finally`
as the `CREDIT_ROLE_SPEND` rows (`cli.main`), after the callback drain, on every path including a
crash: the role ledger folded down to cost per question. It exists because the per-role rows carry
no question denominator, which is how a five-fold-wrong per-question figure ($0.38 to $0.41 against
a measured $2.07 to $2.21) stood in the docs for two months (2026-09-09 cost pass, section 6).

`n_questions` is the number of `ForecastReport` objects the run returned, one per question that
reached the publish step; exceptions beside them are not counted, and a run that crashed before its
reports came back reads `n_questions=0`. `charged_usd` sums every row's `charged_usd` (the money
actually charged; see `CREDIT_ROLE_SPEND` above) and `usd_per_question` divides it by `n_questions`;
both read `n/a` when no row carried cost data or when `n_questions` is 0, never a fabricated rate.
`donated_usd` and `personal_usd` are the same sum restricted to each `KEY_SPECS` key, so they join
onto `CREDIT_SPEND key=`; a key with no rows at all is a true `0.0000` (a Mantic run never touches
the donated key), while a key whose every call OpenRouter left uncosted reads `n/a`. `prompt_tokens`
and `cached_tokens` are the run totals and `cached_share` their ratio (`n/a` on zero prompt tokens).
`max_prompt_tokens` is the largest single prompt any role sent this run and `max_prompt_role` names
that role (`none` on an empty ledger; the sentinel coerces to None like `n/a`).

Consumers: `scripts/cost_report.py` (`make cost_report`) takes a run's question count from this line
and falls back to counting its `FORECASTERS_SURVIVED` lines on the runs archived before it.

### PROMPT_SIZE_ALERT

Per-call WARNING from `credit_telemetry.py:_alert_on_oversized_prompt`, called by the
`RoleSpendTracker` success callback whenever one completion's `usage.prompt_tokens` exceeds
`PROMPT_TOKENS_ALERT_THRESHOLD` (150k; docs/constants.md has the sizing against the measured 17k
forecaster prompt and 41k v2 peak). `role` is the ledger role of the call; `question` is the
question ref when the call site stamped one and `n/a` otherwise. Today only the gap-fill v2 driver
stamps it (`llm_call_metadata(..., question_ref=)` through `LoopConfig.question_ref`, the same
`page_url` ref its `log_prefix` carries, so `qid_kind` is `post_id`); the roster `GeneralLlm`
objects are built once per process and cannot know their question, so a forecaster alert reads
`question=n/a` and is placed by the log lines around it. `prompt_tokens` and `threshold` are both
on the line so the archive records what the bar was when it fired. Non-alertable: the call has
already been billed, so the line reads and never gates, and it enters no degradation counter.
Registered 2026-09-09 and has never fired; a first record is itself the finding.

### CREDIT_FLOOR_BREACH

`credit_telemetry.py`.

### DONATED_KEY_STATE

Per-run, at most once, at the first credit-shaped donated-key failure: the `/auth/key` probe's
verdict, `drained` / `zeroed` / `revoked` / `funded` / `unknown`, INFO for `drained` and WARNING
otherwise (`credit_telemetry.py:classify_donated_key_state`). The end-of-run summary echoes the
same verdict as `donated_key=`; this line is the primary record, timestamped at the failure. The
prose `DONATED_KEY_STATE: /auth/key probe failed` line that precedes an `unknown` verdict has no
`state=` and is not harvested, which is why the regex anchors on `state=` rather than the bare
token.

### LITELLM_CALLBACK_DRAIN_TIMEOUT

Per-run completeness flag on the `CREDIT_ROLE_SPEND` rows above, emitted at most once per run from
`credit_telemetry.py:drain_litellm_callbacks`: litellm's logging worker did not deliver every
queued success callback inside the drain's bound, so the role ledger logged beside it is a lower
bound, missing that run's last few completions. Without this row a low
`reconcile_credit_spend.py --roles` coverage ratio has two readings, a genuine gap in OpenRouter's
per-call cost data, or a drain that gave up, and the archive cannot tell them apart, because no
field on the `CREDIT_ROLE_SPEND` row carries the caveat. Registered 2026-09-04; the WARN is new in
the 2026-09 bundle and has never fired, so a first record is itself the finding.

`timeout_s` is the bound the run used, i.e. the `LITELLM_CALLBACK_DRAIN_TIMEOUT_S` constant unless
a caller overrode it, so the row's value is its presence rather than that near-constant. The same
field name sits on the `PUBLISH_HARDENING` spec meaning the per-attempt POST timeout; one JSONL
per marker keeps the two apart, but a pooled cross-marker query has to key on `marker`.

The regex stays loose either side of the `within <n>s` clause so a reword of the surrounding prose
keeps harvesting, but that clause is now part of the contract: rewording it would zero the
harvest, which the seam pin in `tests/test_credit_telemetry.py` catches. The line names
`CREDIT_ROLE_SPEND` in its own prose and cannot be stolen by that spec, which demands
`CREDIT_ROLE_SPEND:\s*role=`, nor by `CREDIT_SPEND`.

### STACKER_OUTCOME

`metaculus_bot/comment/markers.py`. See "HTML-comment markers" above.

### STACKER_SKIP_REASON

Additive skip-reason companion to `STACKER_OUTCOME` (`metaculus_bot/comment/markers.py`):
separates the mechanisms the plain "skipped" outcome conflates: spread below threshold, per-type
config gate off, single-forecaster short-circuit (computes no spread at all), wall-clock budget,
and `spread_undefined` (the spread could not be measured; it must not read as an affirmative
agreement). HTML-comment marker like its parent, so the same rarely-in-run-logs caveat applies;
see "HTML-comment markers" above.

### TOOLS_USED

`metaculus_bot/comment/markers.py`. See "HTML-comment markers" above.

### FORECASTERS_USED

Ensemble-size disclosure (`metaculus_bot/comment/markers.py`). Like the other HTML-comment markers
this lives in the published comment, not stdout; its durable consumer is
`performance_analysis.parsing`, and the spec is here so the run-log parser stays complete if a
comment body is ever logged. `used` / `configured` are the contributed / configured forecaster
counts. See "HTML-comment markers" above.
