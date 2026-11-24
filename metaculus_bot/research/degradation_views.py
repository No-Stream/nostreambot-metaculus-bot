"""Read-only views onto the research side's module-global degradation counters.

Split out of ``orchestrator.py``: none of this touches orchestrator state or the
research phase at all. The prediction-market and provider-health counters are module
globals (they have to be — the providers soft-fail internally, deep inside fan-outs
the orchestrator never sees), and these views are how they reach the forecaster's
``alertable_count`` and the end-of-run summary. Grouping them keeps five pass-throughs
and their long "why is this alertable" rationales out of the live orchestration path.

``ResearchOrchestrator`` re-exposes all five as its own properties/methods, so every
consumer (``forecaster.py``, ``cli.py``, ``degradation_counters.py``) keeps reaching
them as attributes of the orchestrator.

``provider_health`` is imported at module scope and read through the module at call
time, which both keeps the cheap import off the function bodies and preserves the
patch surface tests use (``monkeypatch.setattr(provider_health, ...)``).
``prediction_market`` stays behind function-level imports: it pulls the optional
market stack (rapidfuzz, aiohttp), which has no business loading for a run that never
enables the provider.
"""

from metaculus_bot.research import provider_health


def prediction_market_degraded_count() -> int:
    """Per-run Kalshi CATALOGUE fetch failures, read from the prediction-market
    module counter and folded into the forecaster's alertable_count.

    The prediction-market provider soft-fails internally (a lost catalogue pull still
    returns whatever the venue-search channel found), so this sub-path failure never
    raises and never bumps provider_failure_count. Reading the module counter here is
    the only way it reddens CI (the 2026-07-25 hole where
    research_provider_failures=0 while the path was dead). The function and marker
    names predate the ranked pipeline, where the counter moved from the retired /series
    index to the events catalogue — a strictly more load-bearing thing, since the
    catalogue feeds both the settlement-source join and the fuzzy channel."""
    from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # optional market deps stay off cold path
        kalshi_catalogue_fetch_failures,
    )

    return kalshi_catalogue_fetch_failures()


def prediction_market_source_loss_count() -> int:
    """Per-run count of LOST prediction-market sources, read from the module
    counter and folded into alertable_count.

    A "source" is anything the snapshot depends on: one per venue whose
    search/prefetch fan-out lost a sub-fetch, one per whole-provider failure, and
    one each when the query author or the RANKING call comes back unusable. Those
    last two are why this counts sources rather than venues — a dead ranker
    degrades every venue's contribution without any venue going down. The
    distinguishing detail is durable per-source in ``MarketSnapshot.sources``
    (``ranking:error(...)`` vs ``polymarket:error(...)``), which rides the
    published comment and the schema-v2 research archive; this scalar
    deliberately stays one number.

    Operator decision 2026-07-25: alert on ANY source loss, not only a total
    blackout. The provider soft-fails every venue internally, so without this the
    forecasters silently run on zero market data while CI stays green."""
    from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # optional market deps stay off cold path
        prediction_market_source_losses,
    )

    return prediction_market_source_losses()


def provider_degradation_count() -> int:
    """Per-run count of ALERTABLE provider-degradation findings, folded into
    alertable_count.

    One finding per (signal, venue), over the two signals provider_health
    defines: a declared liquidity field dead across 100% of the pool rows a
    venue produced (``market_field_contract``), or a prefetch reporting success
    while returning an empty catalogue (``catalogue_empty``). Each is a
    100%-of-denominator conjunction over the whole run, so a single question
    with no matching market stays silent — the denominators are a venue's own
    pool rows and a catalogue's own size, never questions-in-a-run (prod runs
    carry 1-2 questions, so a rate over those IS a per-question flag).

    The first is the signal that would have caught Kalshi's liquidity labels
    blank on 100% of rows for weeks in prod while every counter read zero; the
    second closes its blind spot, since a catalogue that silently empties out
    looks to it like a venue with nothing to say. A third rule (Signal B, a
    venue contributing nothing while >=2 siblings answered) was deleted
    2026-08-04 as unsound under ranked retrieval — see provider_health's module
    docstring and backlog decision-26 "venue_no_contribution provider-health rule"; the surviving cross-run intent is unjudgeable
    inside one question.

    Suppressed findings are excluded here but still logged in full and still
    ride the PROVIDER_DEGRADATION marker (see
    constants.provider_degradation_alerts_active)."""
    return provider_health.provider_degradation_count()


def log_provider_degradation_summary() -> None:
    """Emit the per-run PROVIDER_DEGRADATION marker + one WARN per finding.

    Called from forecast_questions after publishing completes, alongside the
    other end-of-run summaries. Fires even at zero findings — a measured zero is
    a positive statement of provider health, the same reasoning behind
    FORECASTERS_SURVIVED existing next to FORECASTER_DROPS."""
    provider_health.log_provider_degradation_summary()


def reset_run_degradation_counters() -> None:
    """Zero per-run degradation counters at run start (called by
    forecast_questions alongside reset_pchip_stats). The prediction-market
    series and source-loss counters, and the provider-health observation store,
    are module globals — resetting them here keeps them clean per-run metrics
    instead of leaking across runs/tests that share a process. The
    orchestrator's own instance counters are fresh per bot, so they need no
    reset here."""
    from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
        reset_series_degradation_counter,
        reset_source_loss_counter,
    )

    reset_series_degradation_counter()
    reset_source_loss_counter()
    provider_health.reset_provider_health()
