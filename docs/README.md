# Documentation

This directory holds the human-facing guides for the Metaculus forecasting bot.
Each file covers one area of the system. Start with whichever matches what you
need; the repo-root pointers at the bottom cover quick-start and design history.

## Guides

| Doc | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | How a question flows through the bot end to end: research fan-out, the forecaster ensemble, the min-forecasters guard, and aggregation (CONDITIONAL_STACKING by default, MEDIAN in prod since stacking is disabled). |
| [research.md](research.md) | The research providers that build each question's briefing: AskNews (primary), OpenAI native search, Gemini grounded search, financial data, the prediction-market snapshot, and the resolution-source fetcher, plus the always-on gap-fill passes. All fan out in parallel and are independently env-gated. |
| [agentic_gap_fill.md](agentic_gap_fill.md) | Gap-fill v2, the bounded agentic research loop: a driver LLM dry-runs the forecast, picks fill/verify targets, then iterates over search/fetch/read tools to produce a citation-only findings artifact. Runs alongside the v1 gap-fill pass. |
| [numeric_pipeline.md](numeric_pipeline.md) | How numeric forecasts turn the declared percentile set into a PCHIP CDF: value extraction, sanitizing, tail widening, min/max-step enforcement, bound pinning, discrete-integer snapping, the unit-mismatch guard, and the binary and MC clamps. |
| [value_extraction.md](value_extraction.md) | The four-rung ladder that reads a forecast value out of a model's rationale (block, JSON repair, LLM salvage, raise), the fidelity checks each rung must pass, and the `EXTRACTION_RUNG` and `MEMBER_FORECAST` markers with their raw-versus-published convention. |
| [prompts.md](prompts.md) | Every rule the forecasting prompts carry, the named constant that holds it, the measured failure it corrects, and the size accounting from the 2026-09 de-bloat. Read it before adding, removing or rewording a prompt rule. |
| [roster_history.md](roster_history.md) | The ensemble roster's design rule (latest per vendor, resolved only from a live model-list read), the dated history of every roster change and the merge that made it live, the support-model roles, and the two subsystems that are wired but dormant in prod. |
| [performance_analysis.md](performance_analysis.md) | Residual-analysis conventions and receipts: era bucketing with the merge-to-main rule, the exclusion cohorts, the research archive's three record classes, the PIT and spot-peer conventions, `spot_peer_delta`, the starved outer tail, per-model recovery, and the clip-threshold sweep. |
| [forecast_alerts.md](forecast_alerts.md) | GitHub failure summaries, publication evidence, the synthetic notification workflow, and native-email verification limits. |
| [operations.md](operations.md) | Running the bot: environment setup with `uv`, the shared vs. personal API keys, Google AI Studio billing, the production workflows and their env flags, credit telemetry, backtesting, and the cost gate (any live or paid run needs operator approval). |
| [constants.md](constants.md) | One entry per value in `constants.py`, in file order: the measurement, incident or decision behind it. The code keeps one line of why and a pointer here. |
| [telemetry_markers.md](telemetry_markers.md) | The run-log marker registry: every `MarkerSpec` with its fields and receipts, how the harvester matches them, the qid_kind convention, and the HTML-comment markers. |
| [supply_probe.md](supply_probe.md) | The question-supply probe: why it exists, what each report block means, the Mantic mode and its public-snapshot caveat, and the API facts it is built around. |

## Repo-root references

- [README.md](../README.md): the human quick-start (install, configure, run).
- [AGENTS.md](../AGENTS.md): the terse starting point for coding agents, also symlinked as
  `CLAUDE.md`. It carries the cost gate, the repo's overrides, the layout, the pipeline
  outline, and the standing rules, and points here for depth rather than repeating it.
- [FUTURE.md](../FUTURE.md): the branch design log. Records intent and history,
  including shipped, planned, and rejected ideas. Read it for the "why," not for
  the current state of the code.

For the authoritative model roster, see `metaculus_bot/llm_configs.py`. It is the
single source of truth for the ensemble and rotates often, so the guides describe
it rather than pin exact model names.
