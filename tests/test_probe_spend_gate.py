"""The spend gates on the committed probe scripts are the only thing that makes them safe to keep.

Every credit spend in this repo goes through the operator (AGENTS.md, "Cost discipline"), so a
script anyone may run must be structurally incapable of spending. Two probes have a gate worth
pinning, one class each below; the section-strip bench pins its own in tests/test_section_strip_bench.py.

`gemini_verify.py` is the one script whose whole purpose is to spend: three live calls on the
operator's personal Google AI Studio key, which is why it sits in the PAID list. Between a bare
`uv run python scripts/probes/gemini_verify.py` and those three billed calls there is exactly one
guard, the `--i-accept-spend` refusal, and it was asserted nowhere. So this module pins both halves:
the refusal must exit before the client is even constructed, and the accepted path must make three
calls and no more — a fourth call added later would spend more than the flag's own cost estimate,
and the operator's "go" was given against that estimate.

`fetch_diagnostic.py` is on the FREE list, but its column D runs the production escalation ladder,
whose one paid rung (the Gemini `url_context` read) is gated only on an env flag and a key that a
laptop mirroring prod supplies. `main` forces that flag off before any probe runs; the last class
here pins that the paid reader is never invoked even with the flag and a key set and the rung's
403 trigger population deliberately forced, so the free property is structural rather than a matter
of the environment.

Nothing here touches the network. The gemini client is replaced with a fake whose
`generate_content` returns real `google.genai` response objects, so `tests/conftest.py`'s autouse
`_block_network_egress` fixture has nothing to block and the shapes the probe reads
(`model_version`, `usage_metadata`, `grounding_metadata`, `url_context_metadata`) come from the
SDK's own models rather than from a hand-rolled stub that could drift from them. The fetch-diagnostic
class drives the real provider ladder against a fake refused session, so its 403 is produced by the
fetch layer rather than by the egress guard, whose `ssrf_blocked` status is not a url_context trigger
and would make the assertion pass vacuously.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import urlparse

import pytest
from google.genai import types as genai_types

from metaculus_bot.constants import RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV
from metaculus_bot.research import page_digest
from metaculus_bot.research.fetch_ladder import guard, rungs
from metaculus_bot.research.http_fetch import reset_host_semaphores
from metaculus_bot.research.impersonated_fetch import reset_impersonation_memo
from metaculus_bot.research.resolution_fetch_result import FetchRoute, FetchStatus, RungAttempt, RungSkipReason
from metaculus_bot.research.robots_policy import reset_robots_cache
from scripts.probes import fetch_diagnostic, gemini_verify
from tests.resolution_source_fakes import (
    _URL,
    FakeResponse,
    FakeSession,
    _impersonated,
    arm_paid_rung,
    paid_reader,
    refused_page_with_robots,
)


def _probe_response() -> genai_types.GenerateContentResponse:
    """One response carrying every field the probe's three print paths read.

    Deliberately serves BOTH call shapes: `run_grounded_call` reads `grounding_metadata` and
    `run_url_context_call` reads `url_context_metadata`, and one response answering both keeps the
    fake client a pure call counter rather than a second implementation of the probe's own routing.
    """
    return genai_types.GenerateContentResponse(
        model_version="probe-model",
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=11,
            candidates_token_count=7,
            thoughts_token_count=3,
            tool_use_prompt_token_count=5,
            total_token_count=26,
        ),
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part(text="probe answer")]),
                grounding_metadata=genai_types.GroundingMetadata(web_search_queries=["unemployment rate"]),
                url_context_metadata=genai_types.UrlContextMetadata(
                    url_metadata=[
                        genai_types.UrlMetadata(
                            retrieved_url="https://example.test/a",
                            url_retrieval_status=genai_types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                        )
                    ]
                ),
            )
        ],
    )


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: object, config: object) -> genai_types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _probe_response()


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


class TestGeminiVerifyRefusesWithoutTheFlag:
    """A bare invocation must cost nothing, and must not even build a client."""

    def test_it_exits_two_and_never_builds_a_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[str] = []
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: built.append("built"))
        monkeypatch.setattr("sys.argv", ["gemini_verify.py"])

        with pytest.raises(SystemExit) as exc:
            gemini_verify.main()

        # Exit 2 is argparse's usage-error code, and that is what a missing required flag is.
        assert exc.value.code == 2
        # Refusing BEFORE the client reads GOOGLE_API_KEY keeps the gate independent of the environment.
        assert built == [], "the refusal path built a client, so the gate sits after the spend decision"

    def test_the_cost_estimate_prints_before_the_refusal(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The operator's "go" is given against a stated price, so the refusal has to say what the
        run would cost rather than only that it declined."""
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: pytest.fail("client built"))
        monkeypatch.setattr("sys.argv", ["gemini_verify.py"])

        with pytest.raises(SystemExit):
            gemini_verify.main()

        out = capsys.readouterr().out
        assert "Estimated cost of this run" in out
        assert "--i-accept-spend" in out


class TestGeminiVerifySpendsExactlyThreeCalls:
    """With the flag, the probe makes the three calls its cost estimate priced and no others."""

    def test_three_calls_on_the_flagged_path(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = _FakeClient()
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: client)
        monkeypatch.setattr("sys.argv", ["gemini_verify.py", "--i-accept-spend"])

        gemini_verify.main()

        assert len(client.models.calls) == 3, (
            f"the probe made {len(client.models.calls)} calls; its cost estimate and its place in "
            "AGENTS.md's paid list both say three, and the operator's approval is given against "
            "that number"
        )
        # Calls 2 and 3 are the matched robots pair; only the target host differs, which is what makes it a control.
        urls = [c["contents"] for c in client.models.calls[1:]]
        assert gemini_verify.ROBOTS_ALLOWED_URL in str(urls[0])
        assert gemini_verify.ROBOTS_DISALLOWED_URL in str(urls[1])
        # Both retrieved on the fake, so the verdict is the both-retrieved branch.
        assert "Google-Extended hypothesis REFUTED" in capsys.readouterr().out

    def test_the_model_under_test_is_the_one_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(gemini_verify, "build_probe_client", lambda: client)
        monkeypatch.setattr("sys.argv", ["gemini_verify.py", "--i-accept-spend", "--model", "candidate-9"])

        gemini_verify.main()

        assert {c["model"] for c in client.models.calls} == {"candidate-9"}


class TestFetchDiagnosticForcesThePaidRungOff:
    """Column D runs the production ladder, whose last rung is the paid Gemini ``url_context``
    read. ``main`` forces the flag off before probing, so even a laptop with the flag and key set
    cannot spend. The trigger population is forced deliberately: the fetch layer returns a 403, the
    ``blocked`` status the paid rung fires on, rather than the egress guard's ``ssrf_blocked``,
    which the rung does not trigger on and which would make the assertion pass vacuously.
    """

    @pytest.fixture
    def forced_403_ladder(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
        """Arm the paid rung, force the 403 population, and neutralise the intervening free rungs.

        The reader spy is armed (flag on, key set, budget granted) so a call would be recorded; the
        cited page answers 403 via ``refused_page_with_robots`` so the ladder reaches the paid gate;
        and the impersonation, browser and Wayback rungs are stubbed to decline without a network
        hop so the only rung whose gate is under test is ``url_context``. Yields the reader's
        recorded calls.

        The three resets run on the way out as well as on the way in, matching
        ``tests/resolution_source/conftest.py``: the stubbed still-403 writes ``tracker.example.com``
        into the process-global refused-host memo, and a setup-only reset left that entry for every
        module collected after this one.
        """
        reset_host_semaphores()
        reset_robots_cache()
        reset_impersonation_memo()

        def _getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
            del host, port, args, kwargs
            return [(0, 0, 0, "", ("8.8.8.8", 0))]

        monkeypatch.setattr(guard.socket, "getaddrinfo", _getaddrinfo)

        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader, budget_s=30.0)

        session = refused_page_with_robots()
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        async def _still_refused(url: str, **kwargs: Any) -> Any:
            """The impersonated retry fires on the 403; decline without a network dial (``**kwargs`` absorbs
            ``document_max_bytes`` whether or not the caller passes it yet)."""
            del kwargs
            await asyncio.sleep(0)  # a real yield point, so the stub schedules like the transport
            return _impersonated(403, url=url)

        monkeypatch.setattr(rungs, "fetch_impersonated", _still_refused)

        async def _no_browser(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            await asyncio.sleep(0)  # the browser rung's declined signal, scheduled like the render

        monkeypatch.setattr(rungs, "render_page", _no_browser)
        monkeypatch.setattr(rungs, "_WAYBACK_TRIGGER_STATUSES", frozenset())
        yield calls
        reset_host_semaphores()
        reset_robots_cache()
        reset_impersonation_memo()

    async def test_the_forced_population_would_otherwise_reach_the_paid_reader(
        self, forced_403_ladder: list[dict[str, Any]]
    ) -> None:
        """The control: with the flag left on, the 403 population reaches the paid reader, so the
        guard test below is not passing vacuously on a population that never triggers the rung."""
        result = await fetch_diagnostic.probe_ladder(_URL)

        assert result.status == "success"
        assert result.route == "url_context"
        assert len(forced_403_ladder) == 1

    async def test_long_html_never_builds_the_page_digest_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Column D keeps the production fetch path while replacing its paid digest seat."""
        body = (
            "<html><body><article>"
            + ("Employment and unemployment figures for the diagnostic control. " * 200)
            + "</article></body></html>"
        ).encode()
        session = FakeSession({_URL: FakeResponse(200, body=body, content_type="text/html")})

        def _getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
            del host, port, args, kwargs
            return [(0, 0, 0, "", ("8.8.8.8", 0))]

        monkeypatch.setattr(guard.socket, "getaddrinfo", _getaddrinfo)
        monkeypatch.setattr(guard, "_get_session", lambda: session)
        monkeypatch.setattr(
            page_digest,
            "build_llm_with_openrouter_fallback",
            lambda **_kwargs: pytest.fail("the free diagnostic built the paid digest client"),
        )

        result = await fetch_diagnostic.probe_ladder(_URL)

        assert result.status == "no_resolving_content"
        assert result.n_chars == 0
        assert session.requested == [_URL]

    async def test_main_forcing_the_flag_off_keeps_the_paid_reader_unspent(
        self, forced_403_ladder: list[dict[str, Any]]
    ) -> None:
        """The guard: once ``main`` has forced the flag off, the same 403 population never spends,
        because the rung declines on its flag before it looks for its key or its robots policy."""
        fetch_diagnostic._disable_the_paid_rung()

        assert os.environ[RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV] == "false"
        result = await fetch_diagnostic.probe_ladder(_URL)

        assert result.status == "blocked"
        assert forced_403_ladder == []

    async def test_main_runs_the_force_off_before_any_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The placement: ``main`` disables the paid rung before it opens a session or probes, so
        the force-off cannot be raced by a probe that ran first."""
        order: list[str] = []
        monkeypatch.setenv(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV, "true")

        def _record_force() -> None:
            order.append("force")
            os.environ[RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV] = "false"

        monkeypatch.setattr(fetch_diagnostic, "_disable_the_paid_rung", _record_force)

        class _NoSession:
            async def __aenter__(self) -> _NoSession:
                await asyncio.sleep(0)
                order.append("session")
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                await asyncio.sleep(0)

        monkeypatch.setattr(fetch_diagnostic, "_get_session", _NoSession)

        async def _no_egress_ip(_session: Any) -> str:
            await asyncio.sleep(0)
            return "203.0.113.7"

        async def _no_probes(_session: Any, _pacer: Any) -> list[Any]:
            await asyncio.sleep(0)
            order.append("probes")
            return []

        monkeypatch.setattr(fetch_diagnostic, "read_egress_ip", _no_egress_ip)
        monkeypatch.setattr(fetch_diagnostic, "run_probes", _no_probes)

        await fetch_diagnostic.main()

        assert order[0] == "force"
        assert order.index("force") < order.index("probes")
        assert os.environ[RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV] == "false"


def _attempt(
    rung: FetchRoute, outcome: FetchStatus | None, *, skipped: RungSkipReason | Literal[""] = ""
) -> RungAttempt:
    """One closed rung attempt as the dispatcher leaves it: ``outcome`` stamped, or a skip reason."""
    return RungAttempt(
        rung=rung, from_status="blocked", url=_URL, started_at=0.0, wall_s=0.3, outcome=outcome, skipped_reason=skipped
    )


def _ladder(status: str, route: str, *attempts: RungAttempt) -> fetch_diagnostic.LadderOutcome:
    return fetch_diagnostic.LadderOutcome(
        status=status, route=route, http_status=200 if status == "success" else 403, n_chars=0, rung_attempts=attempts
    )


def _row(url: str, *, bot: int | None, impersonated: int | None, ladder: fetch_diagnostic.LadderOutcome) -> Any:
    """A probe row whose columns A and B are the synthetic statuses and whose column D is ``ladder``."""
    return fetch_diagnostic.ProbeRow(
        probe_url=fetch_diagnostic.ProbeUrl(url, "synthetic"),
        bot=fetch_diagnostic.ProbeOutcome(bot, 512, "AkamaiGHost"),
        impersonated=fetch_diagnostic.ProbeOutcome(impersonated, 512, "AkamaiGHost 0.3s"),
        wayback=fetch_diagnostic.ProbeOutcome(200, 512, "20260901000000 (3d old)"),
        ladder=ladder,
    )


class TestFetchDiagnosticVerdict:
    """The verdict paragraph is the sentence the operator's merge decision is read from, and its D
    buckets read ONE signal: the impersonate attempt that fired on the URL, never ``route`` (the
    last rung that fired) and never the ladder's final ``status`` (which mixes in every later
    rung). The corpus misfired both ways on the final status: the ``wkstp.pdf`` rescue, digested
    with an empty query, ended ``blocked`` downstream and printed as still blocked, and a decline
    the archive then served ended ``success`` and printed as recovered. No network, no provider:
    every ``LadderOutcome`` here is built from synthetic ``RungAttempt``s.
    """

    # Column B recovered every one of these (A 403, B 200); column D differs per row.
    RESCUED_HTML = _row(
        "https://www.bls.gov/wsp/",
        bot=403,
        impersonated=200,
        ladder=_ladder("success", "impersonate", _attempt("impersonate", "success")),
    )
    # The bls.gov PDF: impersonation worked, the local read digested to no_resolving_content, so the direct blocked stood.
    RESCUED_PDF_ENDING_BLOCKED = _row(
        "https://www.bls.gov/news.release/pdf/wkstp.pdf",
        bot=403,
        impersonated=200,
        ladder=_ladder(
            "blocked",
            "pdf_local",
            _attempt("impersonate", "no_resolving_content"),
            _attempt("pdf_local", "no_resolving_content"),
        ),
    )
    RESCUED_PDF_ENDING_NO_CONTENT = _row(
        "https://www.bls.gov/opub/ted/2026/report.pdf",
        bot=403,
        impersonated=200,
        ladder=_ladder(
            "no_resolving_content",
            "pdf_local",
            _attempt("impersonate", "no_resolving_content"),
            _attempt("pdf_local", "no_resolving_content"),
        ),
    )
    STILL_403 = _row(
        "https://www.fsis.usda.gov/",
        bot=403,
        impersonated=200,
        ladder=_ladder("blocked", "impersonate", _attempt("impersonate", "blocked")),
    )
    ANSWERED_503 = _row(
        "https://www.cdc.gov/cyclosporiasis/php/surveillance/index.html",
        bot=403,
        impersonated=200,
        ladder=_ladder("blocked", "impersonate", _attempt("impersonate", "error")),
    )
    # An ImpersonatePinNotHeld decline that Wayback then served: FINAL status success, the attempt still blocked.
    PIN_NOT_HELD_THEN_ARCHIVED = _row(
        "https://www.congress.gov/bill/119th-congress/house-bill/2913",
        bot=403,
        impersonated=200,
        ladder=_ladder(
            "success",
            "wayback",
            _attempt("impersonate", "blocked"),
            _attempt("wayback", "success"),
        ),
    )
    NEVER_FIRED = _row(
        "https://tracxn.com/d/companies/deepseek/x",
        bot=403,
        impersonated=200,
        ladder=_ladder("blocked", "direct", _attempt("impersonate", None, skipped="wall_budget")),
    )
    # Not in the recovered population: A got 200, so the rung had nothing to do.
    OPEN_CONTROL = _row(
        "https://en.wikipedia.org/wiki/Nuri_(rocket)",
        bot=200,
        impersonated=200,
        ladder=_ladder("success", "direct"),
    )
    # B was refused yet the shipped rung got through: columns B and D diverging is its own finding, counted apart.
    DIVERGENT = _row(
        "https://www.trueup.io/big-tech-hiring",
        bot=403,
        impersonated=403,
        ladder=_ladder("success", "impersonate", _attempt("impersonate", "success")),
    )
    HELPED = (
        RESCUED_HTML,
        RESCUED_PDF_ENDING_BLOCKED,
        RESCUED_PDF_ENDING_NO_CONTENT,
        STILL_403,
        ANSWERED_503,
        PIN_NOT_HELD_THEN_ARCHIVED,
        NEVER_FIRED,
    )
    ROWS = (*HELPED, OPEN_CONTROL, DIVERGENT)

    @pytest.mark.parametrize(
        ("row", "verdict"),
        [
            (RESCUED_HTML, "answered"),
            (RESCUED_PDF_ENDING_BLOCKED, "answered"),
            (RESCUED_PDF_ENDING_NO_CONTENT, "answered"),
            (STILL_403, "still_blocked"),
            (ANSWERED_503, "errored"),
            (PIN_NOT_HELD_THEN_ARCHIVED, "still_blocked"),
            (NEVER_FIRED, "not_attempted"),
            (OPEN_CONTROL, "not_attempted"),
            (DIVERGENT, "answered"),
        ],
        ids=lambda value: value if isinstance(value, str) else urlparse(value.probe_url.url).netloc,
    )
    def test_the_verdict_reads_the_impersonate_attempt_alone(self, row: Any, verdict: str) -> None:
        assert row.ladder.impersonate_verdict == verdict

    def test_the_four_buckets_partition_the_recovered_population(self) -> None:
        buckets = fetch_diagnostic.verdict_buckets(list(self.HELPED))

        assert buckets.answered == [
            self.RESCUED_HTML,
            self.RESCUED_PDF_ENDING_BLOCKED,
            self.RESCUED_PDF_ENDING_NO_CONTENT,
        ]
        assert buckets.still_blocked == [self.STILL_403, self.PIN_NOT_HELD_THEN_ARCHIVED]
        assert buckets.errored == [self.ANSWERED_503]
        assert buckets.not_attempted == [self.NEVER_FIRED]
        placed = [*buckets.answered, *buckets.still_blocked, *buckets.errored, *buckets.not_attempted]
        assert sorted(placed, key=lambda r: r.probe_url.url) == sorted(self.HELPED, key=lambda r: r.probe_url.url)
        assert len(placed) == len(self.HELPED), "a row landed in two buckets or in none"

    def test_the_sentence_names_each_bucket_with_its_hosts(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The final status would have printed the working PDF rescue as still blocked and the
        archived pin failure as recovered; the sentence built from the attempt says the opposite,
        and the 503 answer sits in its own clause rather than in either."""
        fetch_diagnostic.print_verdict(list(self.ROWS))

        out = capsys.readouterr().out
        assert "Of the 7 URLs column B recovered" in out
        assert "answered on 3 URLs (www.bls.gov)" in out
        assert "still blocked on 2 URLs (www.fsis.usda.gov, www.congress.gov)" in out
        assert "5xx interstitial on 1 URL (www.cdc.gov)" in out
        assert "never fired on 1 URL (tracxn.com)" in out
        assert "Outside that population the attempt answered on 1 URL (www.trueup.io)" in out
        assert "reads the impersonate ATTEMPT alone, never route= or the final status" in out

    def test_an_empty_run_still_prints_a_well_formed_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        fetch_diagnostic.print_verdict([])

        out = capsys.readouterr().out
        assert "Of the 0 URLs column B recovered" in out
        assert "answered on 0 URLs (none)" in out
