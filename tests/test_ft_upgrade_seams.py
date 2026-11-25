"""Seam-pinning tests for the forecasting-tools integration points that the
0.2.54 -> 0.2.92+ upgrade is known to silently break (Workstream B of the retired design log (git history)).

All tests are green on the currently-installed 0.2.54 and are designed to go red
at ft HEAD until the corresponding shim is repointed at the moved seam.

Seam 1 (``TestPchipCdfIsWhatGetsPublished``): our ``PchipNumericDistribution``
overrides only the ``.cdf`` property, which the 0.2.54 publish path reads
directly — at ft HEAD that internal read becomes ``get_cdf()``, which we don't
override, so the base-class CDF builder would silently take over and publish the
wrong distribution. Fix at HEAD: override ``get_cdf()`` on our subclasses.

Seam 2 (``TestPublicFetchRoutesThroughHardenedChokepoint``): ``fetch_hardening``
patches ``MetaculusApi._get_questions_from_api``, the 0.2.54 chokepoint for every
question-list GET — at ft HEAD the real fetch moves to ``MetaculusClient``, so
the patch could become a silent no-op. Fix at HEAD: repoint the fetch-hardening
patches at ``MetaculusClient``.

Seam 3 (``TestPublishHardeningWrapsRealPublishPath``): ``publish_hardening``
wraps the shared private helper ``MetaculusClient._post_question_prediction``
(the single POST all three public ``post_*_question_prediction`` wrappers
delegate to) and ``post_question_comment`` to inject a socket ``timeout`` (and
retry) on the blocking ``requests.post`` inside each, sitting *beneath* the
upstream ``@retry_with_exponential_backoff`` via ``__wrapped__`` so ours is the
single retry layer. The 0.2.92 report ``publish_report_to_metaculus`` methods
publish through ``MetaculusClient()`` instance methods, so driving the real
publish path lets us assert our timeout landed on every POST. Two ways a future
ft version silently breaks this: (a) it reroutes publishing away from those
methods (positive test regresses to the upstream 30s default and goes red);
(b) it renames the private helper ``_post_question_prediction`` — caught by the
dedicated ``test_post_question_prediction_helper_seam_exists`` below, which pins
that the shared helper still exists and carries ``__wrapped__``, and by the
fail-fast raise in ``apply_publish_hardening``.

Seam 4 (``TestQuestionBoundsPatchTargetExistsAndBehaves``): ``question_patches``
monkeypatches ``BoundedQuestionMixin._get_bounds_from_api_json``. On 0.2.92
upstream float-casts range_max/range_min itself, so the patch is narrowed to
coercing only ``zero_point`` — which upstream still returns raw, violating its own
``float | None`` return annotation for an integer JSON zero_point. This pins the
patched tuple contract and that the captured original float-casts the bounds but
leaves zero_point raw — the exact split that makes the narrowing (rather than a
full drop) correct, and not a silent behavior drift.

Seam 5 (``TestHeartbeatWrapsRunABatch``): ``install_benchmarker_heartbeat`` wraps
``Benchmarker._run_a_batch`` (0.2.54, in ``cp_benchmarking``) to emit progress
logs during long backtests. If ft HEAD renames the batch method or moves the
Benchmarker module, the wrap becomes a silent no-op and backtests lose their
heartbeat. This pins that the wrap actually replaces the method and that a stub
batch run emits the ``[HB]`` log line.

Seam 6 (``TestApiPreflightBaseUrlSeam``): ``api_preflight`` derives the URL it vets
from ``MetaculusClient().base_url``. This one has already bitten: the module was
originally written against ``MetaculusApi.API_BASE_URL``, which 0.2.92 removed, so
importing it raised ``AttributeError`` — a guard that fails at import is a guard that
isn't running. The pin makes a future rename fail here, loudly and in CI, rather than at
prod startup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests
from forecasting_tools import (
    Benchmarker,
    BinaryQuestion,
    MetaculusApi,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
)
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericReport, Percentile
from forecasting_tools.data_models.questions import BoundedQuestionMixin
from forecasting_tools.helpers import metaculus_client as ft_client
from forecasting_tools.helpers.metaculus_client import ApiFilter

from metaculus_bot import api_preflight, fetch_hardening, publish_hardening
from metaculus_bot.benchmark.heartbeat import install_benchmarker_heartbeat
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

# In-range 13-percentile forecast on closed [0, 100] bounds. PCHIP (monotone
# cubic + uniform mixture) and the base builder (piecewise-linear) both produce
# a valid 201-point CDF here, but with materially different interior probabilities
# — that divergence is what makes the "publish reads our CDF, not the base one"
# assertion meaningful.
_DECLARED_VALUES = [5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96]


class TestPchipCdfIsWhatGetsPublished:
    """Seam 1: the CDF submitted to Metaculus must be our PCHIP output, not the base builder's."""

    @pytest.fixture
    def question(self) -> NumericQuestion:
        # cdf_size=201 is load-bearing: publish_report_to_metaculus rebuilds a
        # plain base NumericDistribution (dropping our subclass) when cdf_size is None.
        return NumericQuestion(
            question_text="What will the value be?",
            id_of_question=4242,
            id_of_post=4242,
            page_url="https://www.metaculus.com/questions/4242/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            lower_bound=0.0,
            upper_bound=100.0,
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
            unit_of_measure="units",
            cdf_size=201,
        )

    @pytest.fixture
    def pchip_distribution(self, question: NumericQuestion) -> NumericDistribution:
        declared = [
            Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, _DECLARED_VALUES, strict=True)
        ]
        sanitized, zero_point = sanitize_percentiles(declared, question)
        return build_numeric_distribution(sanitized, question, zero_point)

    def test_publish_submits_our_pchip_cdf_not_the_base_builder(
        self, question: NumericQuestion, pchip_distribution: NumericDistribution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert type(pchip_distribution).__name__ == "PchipNumericDistribution"
        assert pchip_distribution.cdf_size == 201, "cdf_size must be set or publish drops the subclass"

        our_cdf = [p.percentile for p in pchip_distribution.cdf]
        base_cdf = [
            p.percentile
            for p in NumericDistribution.from_question(pchip_distribution.declared_percentiles, question).cdf
        ]
        # The submitted == our_cdf assertion below only proves the seam if the base
        # builder would have produced something different. Confirm that first.
        max_divergence = float(np.max(np.abs(np.array(our_cdf) - np.array(base_cdf))))
        assert our_cdf != base_cdf, f"PCHIP and base CDFs identical ({max_divergence=}); test can't detect a swap"

        post_numeric = MagicMock()
        post_comment = MagicMock()
        monkeypatch.setattr(MetaculusClient, "post_numeric_question_prediction", post_numeric)
        monkeypatch.setattr(MetaculusClient, "post_question_comment", post_comment)

        report = NumericReport(question=question, prediction=pchip_distribution, explanation="# Seam test")
        # Pydantic must not coerce our subclass away to a plain NumericDistribution,
        # or the .cdf override would be lost before publish ever runs.
        assert report.prediction is pchip_distribution

        asyncio.run(report.publish_report_to_metaculus())

        post_numeric.assert_called_once()
        submitted_question_id, submitted_cdf = post_numeric.call_args.args
        assert submitted_question_id == question.id_of_question
        assert list(submitted_cdf) == our_cdf, "publish must submit our PCHIP CDF, not the base builder's"


def _reset_fetch_hardening_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the patched MetaculusClient methods, sentinel, and global requests.get.

    ``apply_fetch_hardening`` mutates the MetaculusClient class in place and is
    idempotent via a sentinel. Snapshot the current descriptors through
    monkeypatch (auto-restored on teardown) and clear the sentinel so each test
    exercises a fresh install. Mirrors the helper in test_fetch_hardening.py;
    re-implemented here to avoid importing across test modules.
    """

    for name in fetch_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusClient, name, MetaculusClient.__dict__[name])

    monkeypatch.setattr(ft_client.requests, "get", ft_client.requests.get)

    if hasattr(MetaculusClient, fetch_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusClient, fetch_hardening._SENTINEL, False)
        delattr(MetaculusClient, fetch_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusClient, fetch_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusClient, fetch_hardening._SENTINEL)


class TestPublicFetchRoutesThroughHardenedChokepoint:
    """Seam 2: the public question-list fetch must route through the hardened chokepoint.

    fetch_hardening wraps ``MetaculusApi._get_questions_from_api`` with bounded
    retry. This drives the framework's public entry points (the ones the bot's
    ``forecast_on_tournament`` actually calls) with the HTTP transport failing
    once then succeeding, and asserts the retry fired — i.e. the public path
    still funnels through the chokepoint we patched. If a future ft version
    reroutes the public fetch away from ``_get_questions_from_api``, the first
    403 propagates and the fetch raises (transport called once, not twice).
    """

    @pytest.fixture(autouse=True)
    def _fast_and_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Zero backoff, no random pre-request sleep, a token for _get_auth_headers,
        # two retries, and a clean hardening install for this test.
        monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_BACKOFF_BASE", 0.0)
        monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_BACKOFF_JITTER", 0.0)
        monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_BACKOFF_BASE", 0.0)
        monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_BACKOFF_JITTER", 0.0)
        monkeypatch.setattr("metaculus_bot.constants.FETCH_GET_RETRIES", 2)
        monkeypatch.setattr("metaculus_bot.fetch_hardening.FETCH_GET_RETRIES", 2)
        monkeypatch.setattr("forecasting_tools.helpers.metaculus_client.time.sleep", lambda *_: None)
        monkeypatch.setenv("METACULUS_TOKEN", "test-token")
        _reset_fetch_hardening_state(monkeypatch)

    def _install_403_then_ok_transport(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Stage forecasting-tools' ``requests.get``: 403 on the first call, empty page after.

        Installed BEFORE apply_fetch_hardening so the global GET-timeout wrapper
        closes over this fake, mirroring the real install over ft's requests.get.
        Returns a mutable call counter.
        """

        n_calls = {"n": 0}

        def fake_get(*args, **kwargs):
            n_calls["n"] += 1
            response = MagicMock()
            if n_calls["n"] == 1:
                # The observed 2026-05-19 incident: a CDN/WAF-style 403.
                response.status_code = 403
                response.url = "https://www.metaculus.com/api/posts/?dummy"
                response.reason = "Forbidden"
                response.text = "API only available to authenticated users"
                response.json.return_value = None
                response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden", response=response)
            else:
                response.status_code = 200
                response.content = b'{"results": []}'
                response.raise_for_status.return_value = None
            return response

        monkeypatch.setattr(ft_client.requests, "get", fake_get)
        return n_calls

    def test_get_all_open_questions_from_tournament_retries_through_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bot's actual fetch (forecast_on_tournament -> this) must absorb a transient 403.

        Driven on a MetaculusClient instance — the object the bot constructs — so the
        class-level chokepoint patch is exercised through the real public entry point.
        """
        n_calls = self._install_403_then_ok_transport(monkeypatch)
        fetch_hardening.apply_fetch_hardening()

        result = MetaculusClient().get_all_open_questions_from_tournament("ft-seam-test")

        assert result == []  # fetch succeeded — the retry absorbed the 403
        assert n_calls["n"] == 2  # transport called twice: 403 then success, through the chokepoint

    def test_get_questions_matching_filter_retries_through_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other documented public entry point routes through the same chokepoint."""
        n_calls = self._install_403_then_ok_transport(monkeypatch)
        fetch_hardening.apply_fetch_hardening()

        api_filter = ApiFilter(allowed_tournaments=["ft-seam-test"], allowed_statuses=["open"])
        result = asyncio.run(MetaculusClient().get_questions_matching_filter(api_filter))

        assert result == []
        assert n_calls["n"] == 2


def _reset_publish_hardening_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the patched MetaculusClient.post_* methods and clear the sentinel.

    ``apply_publish_hardening`` mutates the MetaculusClient class in place and is
    idempotent via a sentinel. Snapshot the current descriptors through
    monkeypatch (auto-restored on teardown) and clear the sentinel so each test
    exercises a fresh install. ``monkeypatch.delattr(..., raising=False)`` is
    used for the sentinel because it is teardown-safe whether or not the test
    subsequently re-applies hardening (the negative-control test deliberately
    does not), unlike a manual setattr+delattr dance.
    """

    for name in publish_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusClient, name, MetaculusClient.__dict__[name])

    monkeypatch.delattr(MetaculusClient, publish_hardening._SENTINEL, raising=False)


class TestPublishHardeningWrapsRealPublishPath:
    """Seam 3: publish hardening's injected socket timeout must reach the real HTTP boundary.

    ``publish_hardening`` wraps the shared private helper
    ``MetaculusClient._post_question_prediction`` (which all three public
    ``post_*_question_prediction`` wrappers delegate to) and
    ``post_question_comment`` to force ``timeout=PUBLISH_POST_TIMEOUT`` on the
    underlying ``requests.post``. The 0.2.92 report publish methods call the public
    wrappers on the client instance, which delegate into the hardened helper, so
    driving ``publish_report_to_metaculus`` for each question type with the HTTP
    boundary mocked lets us assert our timeout landed on every captured POST
    (prediction + comment).

    Negative control (``test_..._without_hardening_no_timeout_injected``) drives
    the same publish path WITHOUT ``apply_publish_hardening`` and asserts the POST
    carries the *upstream* default timeout (``MetaculusClient.timeout``, 30s) and
    NOT our value — that non-vacuity proof is what makes the positive assertion
    meaningful. (0.2.92's MetaculusClient always passes its own ``timeout``, so
    "no timeout at all" is no longer the unhardened baseline; the wrapper's job is
    to *override* it with the tighter publish ceiling.) If a future ft version
    reroutes publishing away from those methods, wrapping them becomes a silent
    no-op: the positive test regresses to the upstream default (30s, not our value)
    and goes red. A rename of the private helper is caught separately by
    ``test_post_question_prediction_helper_seam_exists``.
    """

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A token for _get_auth_headers and a clean hardening install per test.
        # No-op the client's inter-request sleep (~4s/POST) so the suite stays fast;
        # it's unrelated to the timeout-injection invariant these tests pin.
        monkeypatch.setattr("forecasting_tools.helpers.metaculus_client.time.sleep", lambda *_: None)
        monkeypatch.setenv("METACULUS_TOKEN", "test-token")
        _reset_publish_hardening_state(monkeypatch)

    def _install_capturing_transport(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Patch metaculus_client's ``requests.post`` to capture kwargs and 200-OK.

        Returns the mutable list of captured kwarg dicts (one per POST).
        """

        captured: list[dict[str, Any]] = []

        def fake_post(*_args: Any, **kwargs: Any) -> Any:
            captured.append(dict(kwargs))
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.json.return_value = {}
            return response

        monkeypatch.setattr(ft_client.requests, "post", fake_post)
        return captured

    @pytest.fixture
    def binary_report(self) -> BinaryReport:
        question = BinaryQuestion(
            question_text="Will it?",
            id_of_question=101,
            id_of_post=101,
            page_url="https://www.metaculus.com/questions/101/",
            background_info="",
            resolution_criteria="",
            fine_print="",
        )
        return BinaryReport(question=question, prediction=0.5, explanation="# Seam test")

    @pytest.fixture
    def numeric_report(self) -> NumericReport:
        question = NumericQuestion(
            question_text="How many?",
            id_of_question=102,
            id_of_post=102,
            page_url="https://www.metaculus.com/questions/102/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            lower_bound=0.0,
            upper_bound=100.0,
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
            unit_of_measure="units",
            cdf_size=201,
        )
        declared = [
            Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, _DECLARED_VALUES, strict=True)
        ]
        distribution = NumericDistribution.from_question(declared, question)
        return NumericReport(question=question, prediction=distribution, explanation="# Seam test")

    @pytest.fixture
    def multiple_choice_report(self) -> MultipleChoiceReport:
        question = MultipleChoiceQuestion(
            question_text="Which?",
            id_of_question=103,
            id_of_post=103,
            page_url="https://www.metaculus.com/questions/103/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            options=["A", "B", "C"],
        )
        options = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.5),
                PredictedOption(option_name="B", probability=0.3),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        return MultipleChoiceReport(question=question, prediction=options, explanation="# Seam test")

    @pytest.mark.parametrize("report_fixture", ["binary_report", "numeric_report", "multiple_choice_report"])
    def test_publish_injects_socket_timeout_on_every_post(
        self, report_fixture: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After apply_publish_hardening, every POST on the real publish path carries the injected timeout."""
        report = request.getfixturevalue(report_fixture)
        captured = self._install_capturing_transport(monkeypatch)
        publish_hardening.apply_publish_hardening()

        asyncio.run(report.publish_report_to_metaculus())

        # Each report type does a prediction POST + a comment POST — both go
        # through a wrapped instance method, so both must carry our timeout.
        assert len(captured) == 2, f"expected prediction + comment POST, got {len(captured)}"
        for kwargs in captured:
            assert kwargs.get("timeout") == publish_hardening.PUBLISH_POST_TIMEOUT, (
                f"publish hardening must inject timeout on every POST; got {kwargs}"
            )

    @pytest.mark.parametrize("report_fixture", ["binary_report", "numeric_report", "multiple_choice_report"])
    def test_publish_without_hardening_carries_upstream_default_not_ours(
        self, report_fixture: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative control: without apply_publish_hardening, the POST carries the upstream default, not ours.

        Proves the positive test isn't vacuous — our ``PUBLISH_POST_TIMEOUT`` only
        appears on the POST because the wrapper forced it. 0.2.92's MetaculusClient
        always passes its own ``timeout=self.timeout`` (30s), so the unhardened
        baseline is that upstream default rather than "no timeout at all".
        """
        report = request.getfixturevalue(report_fixture)
        captured = self._install_capturing_transport(monkeypatch)
        # Deliberately do NOT call apply_publish_hardening().

        asyncio.run(report.publish_report_to_metaculus())

        assert len(captured) == 2
        for kwargs in captured:
            timeout = kwargs.get("timeout")
            assert timeout is not None, f"upstream MetaculusClient always sets its own timeout; got {kwargs}"
            assert timeout != publish_hardening.PUBLISH_POST_TIMEOUT, (
                f"unhardened publish must carry the upstream default, not our injected value; got {kwargs}"
            )

    def test_post_question_prediction_helper_seam_exists(self) -> None:
        """The shared private helper we patch must exist and carry ``__wrapped__`` (decorated upstream).

        publish_hardening patches ``MetaculusClient._post_question_prediction`` (the single POST
        all three public ``post_*_question_prediction`` wrappers delegate to) — NOT the public
        wrappers — and unwraps its ``@retry_with_exponential_backoff`` via ``__wrapped__`` so ours
        is the single retry layer. Two ft-drift failure modes this pins:

        - If ft renames/moves the helper, ``apply_publish_hardening`` would silently leave
          predictions unhardened. It now raises (fail-fast), and this seam-pin goes red first.
        - If ft drops the upstream decorator, ``__wrapped__`` disappears; this goes red, flagging
          that the unwrap became a no-op and the single-retry-layer reasoning needs re-checking.

        The ``_isolated`` fixture resets publish-hardening state but never applies it (hardening
        runs lazily from cli.py, never at import), so the descriptor read here is the pristine
        upstream one, not our wrapper.
        """
        assert "_post_question_prediction" in publish_hardening._PATCHED_METHODS, (
            "the fix must patch the shared private helper, not the public prediction wrappers"
        )
        helper = MetaculusClient.__dict__["_post_question_prediction"]
        assert callable(helper), "MetaculusClient must define _post_question_prediction directly on the class"
        assert hasattr(helper, "__wrapped__"), (
            "_post_question_prediction must be @retry_with_exponential_backoff-decorated upstream "
            "(carry __wrapped__) — the unwrap that makes our retry the single layer depends on it"
        )
        # The three public prediction wrappers still exist and stay OUT of the patch table:
        # they delegate into the hardened helper, keeping only their synchronous input validation.
        for public in (
            "post_binary_question_prediction",
            "post_numeric_question_prediction",
            "post_multiple_choice_question_prediction",
        ):
            assert public in MetaculusClient.__dict__, f"public wrapper {public} vanished from ft"
            assert public not in publish_hardening._PATCHED_METHODS, (
                f"{public} must stay unpatched (it delegates into the hardened helper)"
            )


class TestQuestionBoundsPatchTargetExistsAndBehaves:
    """Seam 4: our zero_point coercion patch must stay installed and honor the tuple contract.

    ``question_patches.apply_question_patches`` is applied at
    ``metaculus_bot`` import (``metaculus_bot/__init__.py``), so by the time this
    module imports, ``BoundedQuestionMixin._get_bounds_from_api_json`` is already
    the patched classmethod. This pins two things: (1) the patch returns the exact
    5-tuple contract with an int zero_point coerced to float, and (2) the
    closure-captured original (the 0.2.92 body) float-casts range_max/range_min
    itself but leaves zero_point raw — which is why the patch is narrowed to
    zero_point-only coercion rather than dropped, confirming the narrowing tracks a
    real upstream split and not a silent behavior drift.
    """

    @staticmethod
    def _scaling_json(
        *,
        range_max: int | float,
        range_min: int | float,
        zero_point: int | float | None = None,
        open_upper_bound: bool = False,
        open_lower_bound: bool = True,
    ) -> dict[str, Any]:
        return {
            "question": {
                "open_upper_bound": open_upper_bound,
                "open_lower_bound": open_lower_bound,
                "scaling": {"range_max": range_max, "range_min": range_min, "zero_point": zero_point},
            }
        }

    def test_patch_is_installed_as_classmethod(self) -> None:
        descriptor = BoundedQuestionMixin.__dict__["_get_bounds_from_api_json"]
        assert isinstance(descriptor, classmethod), "our patch reattaches as a classmethod"
        assert descriptor.__func__.__name__ == "_patched", (
            "the installed function is our patch, not the upstream original"
        )

    def test_int_bounds_return_exact_patched_tuple_contract(self) -> None:
        result = BoundedQuestionMixin._get_bounds_from_api_json(
            self._scaling_json(range_max=200, range_min=0, zero_point=100)
        )
        # Exact tuple: (open_upper, open_lower, upper, lower, zero_point), ints coerced to float.
        assert result == (False, True, 200.0, 0.0, 100.0)
        open_upper, open_lower, upper, lower, zero_point = result
        assert isinstance(open_upper, bool)
        assert isinstance(open_lower, bool)
        assert isinstance(upper, float)
        assert isinstance(lower, float)
        assert isinstance(zero_point, float)

    def test_none_zero_point_preserved(self) -> None:
        result = BoundedQuestionMixin._get_bounds_from_api_json(
            self._scaling_json(range_max=200, range_min=0, zero_point=None)
        )
        assert result == (False, True, 200.0, 0.0, None)

    def test_end_to_end_numeric_question_builds_from_int_scaling(self) -> None:
        """The whole point of the patch: NumericQuestion.from_metaculus_api_json survives int scaling."""
        api_json = {
            "id": 4242,
            "nr_forecasters": 100,
            "forecasts_count": 250,
            "published_at": "2026-01-01T00:00:00Z",
            "projects": {"default_project": {"id": 32813}, "tournament": [{"slug": "ft-seam-test"}]},
            "question": {
                "id": 4242,
                "title": "How many?",
                "status": "open",
                "description": "bg",
                "fine_print": "fp",
                "resolution_criteria": "rc",
                "unit": "units",
                "type": "numeric",
                # Integer scaling — the exact shape that trips the upstream isinstance assert.
                "scaling": {"range_max": 200, "range_min": 0, "zero_point": None, "inbound_outcome_count": 200},
                "open_upper_bound": False,
                "open_lower_bound": True,
                "include_bots_in_aggregates": False,
                "question_weight": 1.0,
                "scheduled_close_time": "2027-01-01T00:00:00Z",
                "scheduled_resolve_time": "2027-01-02T00:00:00Z",
                "open_time": "2026-01-01T00:00:00Z",
            },
        }
        question = NumericQuestion.from_metaculus_api_json(api_json)
        assert question.lower_bound == 0.0
        assert isinstance(question.lower_bound, float)
        assert question.upper_bound == 200.0
        assert isinstance(question.upper_bound, float)
        assert question.open_lower_bound is True
        assert question.open_upper_bound is False

    def test_captured_original_floatcasts_bounds_but_leaves_zero_point_raw(self) -> None:
        """0.2.92 contract of the closure-captured upstream original.

        On 0.2.92 the captured original float-casts range_max/range_min itself, so
        our old bounds coercion is redundant and was dropped. It still returns
        zero_point raw, so an integer JSON zero_point comes back as an int —
        violating the method's own ``float | None`` annotation. That split is
        exactly why the patch is narrowed to zero_point-only coercion, not retired.
        (This supersedes the 0.2.54 control that pinned the original *rejecting*
        ints; upstream no longer does.)
        """
        descriptor = BoundedQuestionMixin.__dict__["_get_bounds_from_api_json"]
        free = dict(zip(descriptor.__func__.__code__.co_freevars, descriptor.__func__.__closure__ or (), strict=False))
        # Annotate Any: the closure-cell extraction is untyped, and unpacking the
        # call result below would otherwise trip basedpyright's "Never not iterable".
        original: Any = free["_original_func"].cell_contents

        _, _, upper, lower, zero_point = original(
            BoundedQuestionMixin, self._scaling_json(range_max=200, range_min=0, zero_point=100)
        )
        assert (upper, lower) == (200.0, 0.0)
        assert isinstance(upper, float)  # upstream float-casts the bounds
        assert isinstance(lower, float)
        assert zero_point == 100
        assert isinstance(zero_point, int)  # ...but leaves zero_point a raw int


class TestHeartbeatWrapsRunABatch:
    """Seam 5: the heartbeat must actually replace Benchmarker._run_a_batch and fire.

    ``install_benchmarker_heartbeat`` wraps ``Benchmarker._run_a_batch`` (0.2.54,
    ``cp_benchmarking.benchmarker``) so long backtests emit ``[HB]`` progress
    logs. This pins that the wrap swaps the method for a different callable (with
    the ``_has_heartbeat`` sentinel) and that a stubbed batch run both delegates
    to the original and logs the heartbeat. If ft HEAD renames the batch method
    or relocates Benchmarker, the wrap silently no-ops and this goes red.
    """

    @pytest.fixture
    def restore_run_a_batch(self) -> Any:
        original = Benchmarker.__dict__["_run_a_batch"]
        try:
            yield
        finally:
            Benchmarker._run_a_batch = original

    def test_install_wraps_and_heartbeat_fires(
        self, restore_run_a_batch: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list[Any] = []

        async def stub_run_a_batch(self: Any, batch: Any) -> None:
            calls.append(batch)

        Benchmarker._run_a_batch = stub_run_a_batch  # type: ignore[assignment]

        # interval_seconds=0 so the heartbeat loop ticks at least once before the
        # stubbed batch (which returns immediately) completes.
        install_benchmarker_heartbeat(interval_seconds=0, progress_state={})

        wrapped = Benchmarker._run_a_batch
        assert wrapped is not stub_run_a_batch, "heartbeat must replace the batch method"
        assert getattr(wrapped, "_has_heartbeat", False) is True, "the wrapper must carry the _has_heartbeat sentinel"

        batch = MagicMock()
        batch.benchmark.name = "seam-batch"
        batch.questions = [object(), object()]

        with caplog.at_level(logging.INFO, logger="metaculus_bot.benchmark.heartbeat"):
            asyncio.run(Benchmarker._run_a_batch(MagicMock(), batch))

        assert len(calls) == 1, "the wrapper must delegate to the original _run_a_batch"
        heartbeat_logs = [r.getMessage() for r in caplog.records if "[HB]" in r.getMessage()]
        assert heartbeat_logs, "a heartbeat log line must fire during the batch run"
        assert "seam-batch" in heartbeat_logs[0]

    def test_install_is_idempotent(self, restore_run_a_batch: None) -> None:
        """A second install must not double-wrap (guards against runaway nesting)."""

        async def stub_run_a_batch(self: Any, batch: Any) -> None:
            return None

        Benchmarker._run_a_batch = stub_run_a_batch  # type: ignore[assignment]

        install_benchmarker_heartbeat(interval_seconds=0, progress_state={})
        after_first = Benchmarker._run_a_batch
        install_benchmarker_heartbeat(interval_seconds=0, progress_state={})
        after_second = Benchmarker._run_a_batch

        assert after_first is after_second, "second install must be a no-op (already wrapped)"


class TestApiPreflightBaseUrlSeam:
    """Seam 6: the DNS-hijack preflight must keep resolving the API root it vets.

    The preflight makes one unauthenticated request and aborts unless the host behaves
    like the real Metaculus API, so the bot never sends METACULUS_TOKEN to a parked or
    hijacked host (the 2026-07-21 incident, where scheduled runs kept firing at a GoDaddy
    parking host with the token attached). Its whole value depends on vetting the SAME
    host the authenticated fetches will use.

    This seam has already broken once: the module read ``MetaculusApi.API_BASE_URL``,
    which 0.2.92 does not define, so importing it raised ``AttributeError`` at startup.
    Pin the attribute it reads now, so the next rename fails in CI.
    """

    def test_metaculus_client_still_exposes_base_url(self) -> None:
        client = MetaculusClient()
        assert isinstance(client.base_url, str)
        assert client.base_url.startswith("http")

    def test_preflight_url_targets_the_clients_api_root(self) -> None:
        # Same root the real question fetch hits, plus the posts-list path whose
        # unauthenticated response IS the identity fingerprint.
        assert api_preflight.preflight_url().startswith(MetaculusClient().base_url)
        assert "/posts/" in api_preflight.preflight_url()

    def test_preflight_url_follows_the_base_url_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # MetaculusClient reads METACULUS_API_BASE_URL, so an override must move the
        # preflight target too — otherwise we would vet the default host and then send
        # the token somewhere else entirely. No module reload: the URL is resolved per call,
        # which is what lets an override reaching the process AFTER import (the .env.local
        # case — nothing loads that file at import time) still move the vetted host.
        monkeypatch.setenv("METACULUS_API_BASE_URL", "https://staging.example.invalid/api")
        assert api_preflight.preflight_url().startswith("https://staging.example.invalid/api")

    def test_deprecated_shim_attribute_is_really_gone(self) -> None:
        # Documents WHY the module was repointed, so nobody "restores" the old read.
        assert not hasattr(MetaculusApi, "API_BASE_URL"), (
            "MetaculusApi.API_BASE_URL is back; api_preflight deliberately reads "
            "MetaculusClient().base_url instead — see its _api_base_url docstring"
        )
