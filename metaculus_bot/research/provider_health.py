"""Per-run provider-degradation signals: publish normally, then redden CI.

Why this exists: two real defects — Kalshi's liquidity labels blank on 100% of
rows since 2026-07-12, and Manifold contributing zero rows since before
2026-07-17 — each survived for weeks because a degraded provider is
byte-identical to a healthy one in every observability channel the bot has. The
prediction-market provider soft-fails internally, the snapshot still renders, the
run exits zero, and only a residual round digs the loss out.

The design constraint that shapes everything here: **prod runs carry ONE OR TWO
questions** (median 1, verified two ways off the telemetry archive plus a
1011-run research-archive histogram of ``{1: 1008, 2: 3}``). So "a rate over the
questions in a run" IS a per-question flag, and a per-question flag is exactly
what must not fire — a single question with no matching prediction market is
normal. Every rule below therefore uses a denominator that exists INSIDE one
question:

* the pool candidates a venue produced (Signal A),
* a prefetch catalogue's own size (Signal C).

A third rule (Signal B, ``venue_no_contribution``) was deleted 2026-08-04. Under
ranked retrieval the enumerable venues enumerate whole catalogues into the pool,
so their candidate count is never zero — they could never be the flagged venue
but always supplied its ">=2 live siblings" leg for free, which left a search
venue whose index legitimately returned ``[]`` satisfying the one leg meant to
exclude correct behaviour. Replaying all 59 archived snapshots, the rule fired on
45 healthy manifold runs and 26 polymarket ones, and the narrowest form that
still fires at all fired on 20. The surviving intent — a venue contributing zero
across many CONSECUTIVE runs — is inherently cross-run and unjudgeable inside one
question; it is recorded in backlog decision-26 "venue_no_contribution provider-health rule" as a
check over the telemetry archive.

Each rule is a 100%-of-denominator conjunction with no tunable float. A longer
run makes every signal strictly HARDER to trip, which is the correct direction
and means no retuning when a ``test_bot`` run carries four questions.

The store is module-scoped for the same reason ``prediction_market``'s
``_SOURCE_LOSSES`` is: the provider is a stateless callable with no handle back
to the bot. ``reset_provider_health`` is called per run from
``ResearchOrchestrator.reset_run_degradation_counters``; without that reset the
state leaks across runs sharing a process and across tests, poisoning every later
``alertable_count == 0`` assertion.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from metaculus_bot.constants import (
    PROVIDER_DEGRADATION_SUPPRESSED_UNTIL,
    provider_degradation_alerts_active,
)

logger = logging.getLogger(__name__)

# Liquidity/participation fields each venue is DECLARED to supply, i.e. the ones
# whose absence is a defect rather than a fact about the venue. This map is the
# design's one piece of declared knowledge, so ``tests/test_provider_health.py``
# pins it against what each match builder actually populates from the captured
# live payloads — a builder that starts or stops reading a field without moving
# the map fails there.
#
# PredictIt is deliberately EMPTY: its ``/marketdata/all/`` dump carries no
# volume, liquidity, or open-interest field anywhere, so its blank ``signal``
# column is honest and must never alert. Manifold is play-money, so its
# informativeness signal is the unique-bettor count rather than dollars.
VENUE_EXPECTED_LIQUIDITY_FIELDS: dict[str, tuple[str, ...]] = {
    "polymarket": ("total_volume", "open_interest"),
    "kalshi": ("total_volume", "open_interest"),
    "manifold": ("num_bettors",),
    "predictit": (),
}

# Signal names, also the ``signal=`` value in the marker's JSON detail.
SIGNAL_MARKET_FIELD_CONTRACT = "market_field_contract"
SIGNAL_CATALOGUE_EMPTY = "catalogue_empty"

# Where a finding's remedy sends the operator. The match builders and the prefetch parsing live
# one module per venue in the venues package; `prediction_market.py` is only the seam that calls
# them, so naming it sent the reader to a file with no parsing in it. Signal C's key is a
# catalogue SOURCE (`kalshi_events`) rather than a venue, hence both spellings here.
_VENUES_PACKAGE_PATH = "metaculus_bot/research/market_retrieval/venues/"
_VENUE_MODULES: dict[str, str] = {
    "polymarket": "polymarket.py",
    "kalshi": "kalshi.py",
    "kalshi_events": "kalshi.py",
    "manifold": "manifold.py",
    "predictit": "predictit.py",
    "predictit_markets": "predictit.py",
}


def _venue_module_path(venue: str) -> str:
    """The file to open for ``venue``, or the package when the name is not one we know."""
    module = _VENUE_MODULES.get(venue)
    return f"{_VENUES_PACKAGE_PATH}{module}" if module else _VENUES_PACKAGE_PATH


@dataclass(frozen=True, slots=True)
class VenueObservation:
    """One venue's outcome on one question, at the granularity the rules need.

    ``candidates_pre_filter`` counts the rows the venue put into the POOL, i.e. what
    its fan-out or catalogue produced before the ranker selected anything. It is
    Signal A's gate and denominator, and reading the pool rather than the render is
    what makes Signal A immune to the ranker's judgment: a zero-row render is
    routine, so a rendered denominator would let the same dead parser alert or stay
    silent on which rows the model picked.

    ``rows_post_filter`` is the RENDERED count. No rule reads it — it is recorded so
    the recording tests can pin the pool-versus-render split, and so a cross-run
    analysis over the telemetry archive can ask how often a venue's candidates reach
    a forecaster. Anything alerting on it inside one question would be alerting on a
    ranker decision.

    ``liquidity_fields_present`` holds the declared fields that parsed non-``None``
    on at least one POOL row — the same rows ``candidates_pre_filter`` counts. It is
    computed off the parsed ``MarketMatch`` objects rather than the upstream payload,
    so Signal A cannot disagree with the ``signal`` column a forecaster reads.
    """

    qid: int
    venue: str
    candidates_pre_filter: int
    rows_post_filter: int
    liquidity_fields_present: frozenset[str]


@dataclass(frozen=True, slots=True)
class CatalogueObservation:
    """One prefetch catalogue's size on one question, plus whether its fetch succeeded.

    Kalshi and PredictIt prefetch a catalogue and fuzzy-match locally, so their
    fetch tally reports success even when the catalogue is empty and the local
    matcher therefore has nothing to match against. A successful fetch returning
    zero entries is a deterministic contradiction, which is why Signal C needs no
    rate. A FAILED fetch returning zero is already owned by
    ``prediction_market_source_losses``; ``fetch_ok`` is what keeps the two from
    double-counting the same outage.
    """

    qid: int
    source: str
    entries: int
    fetch_ok: bool


@dataclass(frozen=True, slots=True)
class DegradationFinding:
    """One degradation, deduped per (signal, venue) — never per row or per question.

    A run where both of Kalshi's liquidity fields are dead across three rows is
    ONE defect, not six events, so the count carries no diagnostic weight and the
    log line carries all of it. ``suppressed_until`` is set when the operator has
    a dated acceptance on record for this venue, in which case the finding is
    still logged and still rides the marker but drops out of ``alertable``.
    """

    signal: str
    venue: str
    detail: dict[str, object]
    questions: int
    remedy: str
    suppressed_until: date | None = None

    @property
    def is_alertable(self) -> bool:
        return self.suppressed_until is None

    def as_marker_detail(self) -> dict[str, object]:
        """The finding as one entry of the marker's ``detail=`` JSON array."""
        out: dict[str, object] = {"signal": self.signal, "venue": self.venue, "questions": self.questions}
        out.update(self.detail)
        if self.suppressed_until is not None:
            out["suppressed_until"] = self.suppressed_until.isoformat()
        return out

    def human_line(self) -> str:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(self.detail.items()))
        return (
            f"{self.venue} {self.signal} across {self.questions} question(s) ({rendered}) — {self.remedy}"
            if rendered
            else f"{self.venue} {self.signal} across {self.questions} question(s) — {self.remedy}"
        )


@dataclass
class _RunState:
    venue_observations: list[VenueObservation] = field(default_factory=list)
    catalogue_observations: list[CatalogueObservation] = field(default_factory=list)


_RUN = _RunState()


def record_venue_observation(observation: VenueObservation) -> None:
    """Record one venue's per-question outcome. Pure module-state write.

    No I/O and no ``await``, so it cannot raise, block, or alter the snapshot it
    is recorded from. Single-threaded asyncio with distinct ``(qid, venue)`` keys,
    the same safety argument already documented for
    ``provider_diagnostics._PROVIDER_DETAIL_REGISTRY``.
    """
    _RUN.venue_observations.append(observation)


def record_catalogue_size(*, qid: int, source: str, entries: int, fetch_ok: bool) -> None:
    """Record one prefetch catalogue's size + fetch outcome for this question."""
    _RUN.catalogue_observations.append(CatalogueObservation(qid=qid, source=source, entries=entries, fetch_ok=fetch_ok))


def reset_provider_health() -> None:
    """Drop all observations at run start (see the module docstring on leakage)."""
    _RUN.venue_observations.clear()
    _RUN.catalogue_observations.clear()


def recorded_observations() -> tuple[tuple[VenueObservation, ...], tuple[CatalogueObservation, ...]]:
    """This run's raw observations, for asserting on what the RECORDING site produced.

    Exists so the recording tests can pin the numbers the rules are fed without
    reaching into module privates: the pre-filter/post-filter split and the
    liquidity-field presence are the values whose derivation actually carries risk,
    and a rule-level assertion cannot distinguish a correct rule fed wrong inputs
    from a wrong rule fed correct ones. Returns immutable copies so a caller cannot
    mutate the run's state.
    """
    return tuple(_RUN.venue_observations), tuple(_RUN.catalogue_observations)


def _field_contract_findings(observations: list[VenueObservation], today: date | None) -> list[DegradationFinding]:
    """Signal A — a declared liquidity field dead across 100% of a venue's POOL rows.

    Gated and denominated on ``candidates_pre_filter``, NOT on the rendered count,
    because the rule exists to catch a PARSER whose field names went stale and the
    ranker's selection has nothing to do with that. ``RENDER_BUDGET`` is a global
    ceiling of 8 across four venues against pool widths of 100/60/60/~197, and an
    empty ranker answer is explicitly valid, so a venue rendering zero rows is the
    routine case rather than an edge — 42% of question-runs for kalshi, 39%
    polymarket, 25% manifold in the bake-off's own diagnostics. Gating on the
    rendered count let the same dead parser alert or stay silent purely on which
    rows the model happened to pick, which is the 2026-07-12 hole (Kalshi labels
    blank on 100% of rows for weeks) reopened. The recording site measures presence
    over the same pool rows, so numerator and denominator now come from one place.

    A legitimately market-less question stays silent because a venue with no pool
    candidates is never evaluated. ``_liquidity_label`` renders ``no-liquidity-data``
    if and only if the parsed value is ``None``, while a genuinely zero-volume
    brand-new market parses to ``0.0`` and renders ``thin`` — absent and zero are
    already distinct in the code, which is what makes a small pool a conclusive
    denominator rather than a noisy sample.
    """
    findings: list[DegradationFinding] = []
    for venue in sorted(VENUE_EXPECTED_LIQUIDITY_FIELDS):
        expected = VENUE_EXPECTED_LIQUIDITY_FIELDS[venue]
        if not expected:
            continue
        with_rows = [obs for obs in observations if obs.venue == venue and obs.candidates_pre_filter > 0]
        if not with_rows:
            continue
        dead = [f for f in expected if not any(f in obs.liquidity_fields_present for obs in with_rows)]
        if not dead:
            continue
        pool_rows = sum(obs.candidates_pre_filter for obs in with_rows)
        findings.append(
            DegradationFinding(
                signal=SIGNAL_MARKET_FIELD_CONTRACT,
                venue=venue,
                detail={"fields": ",".join(dead), "pool_rows": pool_rows},
                questions=len(with_rows),
                remedy=(
                    "the rendered `signal` column read no-liquidity-data on every row, and the forecaster "
                    "prompt tells models to weight market signals by it. Likely an upstream field rename; "
                    "check the venue's API payload against the match builder in "
                    f"{_venue_module_path(venue)}"
                ),
                suppressed_until=_suppression_for(venue, today),
            )
        )
    return findings


def _catalogue_findings(observations: list[CatalogueObservation], today: date | None) -> list[DegradationFinding]:
    """Signal C — a prefetch reported SUCCESS and returned an empty catalogue.

    Closes the hole Signal A has: it reads a venue's pool candidates, so a
    catalogue that silently empties out looks to it like a venue with nothing to
    say. Observed Kalshi series catalogues run 12,355-12,370 entries and were never
    once zero, so success-with-empty is a contradiction needing no rate. For the two
    enumerable venues this is the only alarm that can fire on an empty catalogue.
    """
    sources = sorted({obs.source for obs in observations})
    findings: list[DegradationFinding] = []
    for source in sources:
        per_source = [obs for obs in observations if obs.source == source]
        successful = [obs for obs in per_source if obs.fetch_ok]
        if not successful or any(obs.entries > 0 for obs in successful):
            continue
        findings.append(
            DegradationFinding(
                signal=SIGNAL_CATALOGUE_EMPTY,
                venue=source,
                detail={"entries": 0, "fetch_ok": True},
                questions=len(successful),
                remedy=(
                    "the prefetch reported success and returned an empty catalogue, so the local fuzzy "
                    "matcher had nothing to match against on any question. Check the prefetch's response "
                    f"parsing in {_venue_module_path(source)}"
                ),
                suppressed_until=_suppression_for(source, today),
            )
        )
    return findings


def _suppression_for(venue: str, today: date | None) -> date | None:
    """The dated resume for ``venue`` while its degradation is accepted, else ``None``.

    Reads the shared dict object rather than a copy, so a test (or a future runtime
    override) mutating ``constants.PROVIDER_DEGRADATION_SUPPRESSED_UNTIL`` is seen
    here without a re-import.
    """
    if provider_degradation_alerts_active(venue, today):
        return None
    return PROVIDER_DEGRADATION_SUPPRESSED_UNTIL.get(venue)


def provider_degradation_findings(today: date | None = None) -> list[DegradationFinding]:
    """Evaluate both rules over this run's observations.

    ``today`` is threaded through to the per-venue suppression check so tests can
    exercise both sides of a resume date forever instead of depending on the wall
    clock; production passes ``None`` and the clock is read at CALL time.
    """
    findings = _field_contract_findings(_RUN.venue_observations, today)
    findings.extend(_catalogue_findings(_RUN.catalogue_observations, today))
    return findings


def provider_degradation_count(today: date | None = None) -> int:
    """Count of ALERTABLE findings — suppressed ones are logged but not counted."""
    return sum(1 for finding in provider_degradation_findings(today) if finding.is_alertable)


def log_provider_degradation_summary(today: date | None = None) -> list[DegradationFinding]:
    """Emit the per-run marker + one WARN per finding; return the findings.

    The marker fires at INFO even at zero — but a zero is only a positive
    statement of health when the line says what was measured. 96% of the
    archived triple-era records came from runs that forecast nothing and so
    recorded zero observations, and those vacuous zeros were byte-identical to a
    genuine clean read over ~400 pool rows (2026-08-24 residual round). The
    ``venues_observed=`` / ``catalogues_observed=`` / ``pool_rows=`` denominators,
    read off ``recorded_observations()``, are what let a reader tell
    ``findings=0 venues_observed=0`` ("not measured") from
    ``findings=0 venues_observed=4 pool_rows=404`` ("measured, healthy"). A
    fully-suppressed degradation still prints its arithmetic and its resume date,
    following ``cli.py``'s precedent of emitting the breakdown on BOTH exit
    paths: the green run is exactly the one that would otherwise leave no record.
    """
    findings = provider_degradation_findings(today)
    alertable = [finding for finding in findings if finding.is_alertable]
    suppressed = [finding for finding in findings if not finding.is_alertable]
    venue_observations, catalogue_observations = recorded_observations()
    detail_json = json.dumps([finding.as_marker_detail() for finding in findings], separators=(",", ":"))
    suppression_note = ""
    if suppressed:
        rendered = "; ".join(
            f"{finding.venue}:{finding.signal} suppressed until {finding.suppressed_until}"  # type: ignore[union-attr]
            for finding in suppressed
        )
        suppression_note = f" ({rendered}); run stays green on those."
    logger.info(
        "PROVIDER_DEGRADATION: run=%s findings=%d alertable=%d suppressed=%d "
        "venues_observed=%d catalogues_observed=%d pool_rows=%d detail=%s%s",
        os.environ.get("GITHUB_RUN_ID", "local"),
        len(findings),
        len(alertable),
        len(suppressed),
        len(venue_observations),
        len(catalogue_observations),
        sum(obs.candidates_pre_filter for obs in venue_observations),
        detail_json,
        suppression_note,
    )
    for finding in findings:
        status = "alertable" if finding.is_alertable else f"suppressed until {finding.suppressed_until}"
        logger.warning("PROVIDER DEGRADATION (%s): %s", status, finding.human_line())
    return findings
