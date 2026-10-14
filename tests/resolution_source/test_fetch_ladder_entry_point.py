"""The shared ladder's entry point: what ``fetch_url`` binds onto the context, and rung 0.

``fetch_url`` is the one coroutine both cited-page fetchers call, so what it pairs is the whole
contract: the caller's :class:`LadderPolicy`, the aiohttp session and the process-wide per-host
politeness map. Each test here drives it end to end against a ``FakeSession`` rather than
patching ``_fetch_one``, because the thing worth pinning is that a knob set on the policy reaches
the rung that reads it, not that one function called another.

Lives in ``tests/resolution_source/`` for the package conftest's autouse DNS stub and its
process-wide gate resets; every hostname is an RFC-2606 reserved name with no real DNS.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from time import monotonic

from metaculus_bot.research.fetch_ladder import direct_fetch, guard
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.digest import DigestFn, bm25_digest
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import RESOLUTION_SOURCE_POLICY
from metaculus_bot.research.http_fetch import host_semaphores
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.source_evidence import SourceExcerpt
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://tracker.example.com/senate"
# Above the chrome floor on its extraction alone, so a plain fetch of it is a `success`.
_PAGE = (
    b"<!doctype html><html><head><title>Report</title></head><body><nav>Home</nav><article>"
    b"<h1>Latest reading</h1><p>The agency reported 922 hospitalizations in the week ending "
    b"September 6, 2026, and 2 deaths, both figures revised upward from the preliminary count "
    b"published three days earlier.</p>"
    b"<p>The weekly series counts laboratory-confirmed admissions reported by the 187 hospitals "
    b"in the sentinel network, excludes emergency-department visits that did not end in an "
    b"admission, and is revised once as late reports arrive. The revision window closes four "
    b"weeks after the reporting week, after which the figure is final and carries the annual "
    b"surveillance summary's own definition of a confirmed case.</p>"
    b"</article><footer>&copy; 2026</footer></body></html>"
)


def _session() -> FakeSession:
    return FakeSession({_URL: FakeResponse(200, body=_PAGE, content_type="text/html")})


class TestFetchUrlBindsThePolicy:
    async def test_the_walls_on_the_policy_are_the_walls_the_rungs_read(self):
        """The wall pair is the knob the entry point exists for, so it has to reach a rung.

        Read through the per-hop ``ClientTimeout``, which ``fetch_ladder.direct_fetch._fetch_one_hop`` clamps to the
        remaining budget: a policy whose whole wall is already spent leaves the floor, where the
        45 s preset leaves the full per-hop timeout.
        """
        session = _session()
        spent = replace(RESOLUTION_SOURCE_POLICY, total_wall_s=0.0)

        await fetch_url(_URL, policy=spent, ctx=LadderContext(session=session, host_sems={}))

        assert session.get_kwargs[0]["timeout"].total == direct_fetch.RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S

    async def test_the_default_preset_leaves_the_full_per_hop_timeout(self):
        session = _session()

        result = await fetch_url(
            _URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(session=session, host_sems={})
        )

        assert result.status == "success"
        assert session.get_kwargs[0]["timeout"].total == direct_fetch.RESOLUTION_SOURCE_HTTP_TIMEOUT


class TestFetchUrlResolvesTheSessionAndTheHostMap:
    async def test_a_context_carrying_no_session_gets_one_opened_and_closed(self, monkeypatch):
        """The gap-fill loop's plain fetch holds no session, so the entry point opens one.

        Closing it is the half that matters: a session left open per URL leaks its connector.
        """
        session = _session()
        monkeypatch.setattr(guard, "_get_session", lambda: session)

        result = await fetch_url(_URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext())

        assert result.status == "success"
        assert session.closed

    async def test_a_context_naming_no_host_map_contends_on_the_process_wide_one(self):
        """Politeness is only politeness if it is shared: a fresh dict per fetch is no gate."""
        session = _session()

        await fetch_url(_URL, policy=RESOLUTION_SOURCE_POLICY, ctx=LadderContext(session=session))

        assert "tracker.example.com" in host_semaphores()


class TestRungZero:
    async def test_a_known_api_answer_short_circuits_before_any_request(self):
        """Rung 0's whole point is costing no page fetch, and its result stands as it is."""
        session = _session()
        answered = FetchResult(
            url=_URL, status="success", text="922 hospitalizations", http_status=200, content_type="application/json"
        )

        async def _known_api(url: str) -> FetchResult:
            assert url == _URL
            # A real yield point, so the seat schedules like the registry call it stands in for.
            await asyncio.sleep(0)
            return answered

        result = await fetch_url(
            _URL,
            policy=replace(RESOLUTION_SOURCE_POLICY, known_api=_known_api),
            ctx=LadderContext(session=session, host_sems={}),
        )

        assert result is answered
        assert session.requested == []

    async def test_a_known_api_that_declines_falls_through_to_the_ladder(self):
        """A registry that does not know a host has to cost nothing but its own None."""
        session = _session()
        asked: list[str] = []

        async def _known_api(url: str) -> FetchResult | None:
            asked.append(url)
            # A real yield point, as above.
            await asyncio.sleep(0)
            return None

        result = await fetch_url(
            _URL,
            policy=replace(RESOLUTION_SOURCE_POLICY, known_api=_known_api),
            ctx=LadderContext(session=session, host_sems={}),
        )

        assert asked == [_URL]
        assert result.status == "success"
        assert session.requested == [_URL]

    async def test_a_near_exhausted_wall_declines_rung_zero_before_fallback(self):
        session = _session()
        called = False

        async def _known_api(url: str) -> FetchResult:
            nonlocal called
            called = True
            raise AssertionError(f"rung zero should have declined: {url}")

        policy = replace(RESOLUTION_SOURCE_POLICY, known_api=_known_api)
        ctx = replace(
            LadderContext(session=session, host_sems={}),
            started=monotonic() - policy.total_wall_s - 1.0,
        )

        result = await fetch_url(_URL, policy=policy, ctx=ctx)

        assert not called
        assert result.status == "success"
        assert session.requested == [_URL]

    async def test_rung_zero_timeout_is_per_url_and_does_not_cancel_a_sibling(self):
        slow_url = "https://slow.example.com/source"
        fast_url = "https://fast.example.com/source"
        session = FakeSession(
            {
                slow_url: FakeResponse(200, body=_PAGE, content_type="text/html"),
                fast_url: FakeResponse(200, body=_PAGE, content_type="text/html"),
            }
        )
        calls: list[str] = []
        answered = FetchResult(
            url=fast_url,
            status="success",
            text="known API body",
            http_status=None,
            content_type="application/json",
            route="known_api",
        )

        async def _known_api(url: str) -> FetchResult | None:
            calls.append(url)
            if url == slow_url:
                await asyncio.sleep(0.05)
                return None
            return answered

        policy = replace(RESOLUTION_SOURCE_POLICY, known_api=_known_api, rung_wall_margin_s=0.0, total_wall_s=0.01)
        started = monotonic() - policy.total_wall_s + 0.005
        slow_context = replace(LadderContext(session=session, host_sems={}), started=started)
        fast_context = replace(LadderContext(session=session, host_sems={}), started=started)

        slow_result, fast_result = await asyncio.gather(
            fetch_url(slow_url, policy=policy, ctx=slow_context),
            fetch_url(fast_url, policy=policy, ctx=fast_context),
        )

        assert calls == [slow_url, fast_url]
        assert slow_result.status == "success"
        assert fast_result is answered
        assert session.requested == [slow_url]


class TestTheDigestSeat:
    async def test_the_sibling_digest_signature_drops_into_the_seat(self):
        """The seat is shaped as ``page_digest.digest_page`` is, so the wiring step assigns it and nothing else.

        The double below mirrors that function's signature and its ``PageDigest`` result field for
        field; basedpyright checks the assignment to ``DigestFn`` over this file, so a drift in either
        fails ``make typecheck`` here rather than at the merge.
        """

        @dataclass(frozen=True)
        class _PageDigest:
            passages: list[str]
            passages_returned: int
            passages_grounded: int
            fallback_used: bool
            method: str
            outcome: str
            scope: str
            evidence_passages: tuple[SourceExcerpt, ...]

        async def _digest_page(text: str, query: str, *, budget_seconds: float) -> _PageDigest:
            del budget_seconds
            await asyncio.sleep(0)
            return _PageDigest(
                [text[:12]],
                1,
                1,
                False,
                f"llm_extractive:{query}",
                "answered",
                "full_page",
                (),
            )

        seat: DigestFn = _digest_page
        armed = replace(RESOLUTION_SOURCE_POLICY, digest=seat)

        assert armed.digest is _digest_page
        assert RESOLUTION_SOURCE_POLICY.digest is not None
        assert (await bm25_digest("no shared vocabulary here", "hospitalizations", budget_seconds=1.0)).passages == []
