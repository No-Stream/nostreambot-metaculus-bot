"""One-shot identity check for a question platform's API host, run before we send the token.

On 2026-07-21 metaculus.com's DNS was repointed at a GoDaddy parking host (WHOIS
updated that day, GoDaddy nameservers serving parking IPs, a fresh GoDaddy DV
cert for www.metaculus.com issued the same afternoon). The bot's scheduled
GitHub Actions runs kept firing and forecasting-tools attached
``Authorization: Token $METACULUS_TOKEN`` to the question-fetch GET
(``forecasting_tools/helpers/metaculus_api.py`` line 409:
``requests.get(url, params=params, **cls._get_auth_headers())``), leaking the
token to an unknown host before dying on an opaque 404. The real Metaculus
origin was still alive behind Cloudflare — this was an upstream domain incident,
not a code bug.

``verify_api_identity`` makes an unauthenticated request (no token, no headers)
to the posts list under the given API base URL and confirms the host behaves
like the real API before any authenticated call runs. A transport failure or
transient/server-error response gets one retry after five seconds, using the
same full timeout. Other responses are evaluated immediately for identity.
``verify_metaculus_api_identity`` is the Metaculus run modes' wrapper over it;
the Mantic run mode calls ``verify_api_identity`` with ``MANTIC_API_BASE_URL``
directly, so a Mantic run never depends on Metaculus DNS health. Two jobs:

1. Never send the token to a host we haven't sanity-checked — the preflight
   itself carries no credentials.
2. Fail fast with a diagnostic that names the vetted host and the likely cause
   (DNS parking/hijack), instead of a bare ``HTTPError`` traceback from deep
   inside forecasting-tools.

The request goes through an isolated ``requests.Session`` with
``trust_env=False``. That is load-bearing, not hygiene: with the default
``trust_env=True``, ``Session.prepare_request`` calls ``get_netrc_auth(url)``
and a matching ``machine www.metaculus.com`` — or a generic ``default`` — entry
in ``~/.netrc`` / ``$NETRC`` would attach ``Authorization: Basic ...`` to this
"unauthenticated" probe (reproduced live on requests 2.34.2), defeating the
whole point on exactly the hijack path this guard exists to cover. Disabling
``trust_env`` also skips proxy-env and ``REQUESTS_CA_BUNDLE`` pickup for this
one call; that is an accepted trade — GHA runners have direct egress, and the
no-credentials-to-an-unknown-host invariant wins over honoring ambient network
config for a single identity probe.

The retry is safe because both attempts use the same isolated session without
credentials; retrying cannot send a token or ambient netrc credentials to an
unvetted host. A transient response still cannot establish identity, so a
second transient/server-error response fails shut as unreachable. Retries with
the token attached belong to ``fetch_hardening``, which runs only after identity
is established.

Signatures observed live:

- Metaculus, unauthenticated GET ``/api/posts/?limit=1`` -> 403, ``text/plain``
  body "Permission Error: The API is only available to authenticated users."
  (2026-07-21).
- Metaculus, authenticated -> 200 JSON dict with a ``"results"`` key (2026-07-21).
- Mantic (``competitions.mantic.com/api``), unauthenticated GET ``/posts/?limit=1``
  -> 200 ``application/json`` dict with a ``"results"`` key: its posts list is
  public and there is no Cloudflare in front (2026-09-08).
- Parked/hijacked host, same URL -> 404 empty body; ``/api2/...`` paths return
  200 with an HTML lander redirect (2026-07-21).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

import requests
from forecasting_tools.helpers.metaculus_client import MetaculusClient

logger = logging.getLogger(__name__)


def _api_base_url() -> str:
    """The API root the bot's own Metaculus fetches will use.

    Read from ``MetaculusClient`` rather than the deprecated ``MetaculusApi`` shim, which
    on ft 0.2.92 no longer carries ``API_BASE_URL`` at all — this module was written
    against an older ft and raised ``AttributeError`` on import until repointed, and a
    guard that fails at import is a guard that isn't running. Constructing a client is
    cheap and side-effect-free apart from a missing-token warning, and it resolves
    ``METACULUS_API_BASE_URL`` exactly the way every real fetch does, so the host we vet
    is the host we will actually send the token to even under an env override.
    ``tests/test_ft_upgrade_seams.py`` pins the attribute so a future ft bump fails in CI
    instead of at prod startup.
    """
    return MetaculusClient().base_url


def _preflight_url(base_url: str) -> str:
    """The identity probe's URL under ``base_url``: the posts list, whose unauthenticated
    behavior IS the fingerprint (it is the exact endpoint the question fetch uses)."""
    return f"{base_url}/posts/?limit=1"


def preflight_url() -> str:
    """The URL the Metaculus identity probe hits.

    Resolved at CALL time rather than bound as a module constant, because the base URL comes
    out of the environment and this module is imported BEFORE the bot loads its ``.env``
    files. A constant here read whatever ``METACULUS_API_BASE_URL`` was set at import, while
    every real fetch resolves it after ``load_environment()`` — measured divergence: with the
    override in ``.env.local`` (which ``load_environment`` applies with ``override=True``, and
    which nothing loads at import), the constant vetted ``www.metaculus.com`` and the token
    then went to the override host. That is precisely the "credentials to an unvetted host"
    failure this module exists to prevent, so the two reads have to happen at the same time.
    """
    return _preflight_url(_api_base_url())


# A real question-platform API gates unauthenticated access behind these statuses —
# Metaculus's fingerprint when we send no token.
_AUTH_GATED_STATUSES = frozenset({401, 403})

# Transient edge conditions the real Metaculus front door emits under load
# (see fetch_hardening._RETRYABLE_STATUSES and the 2026-05-19 CDN-403 incident).
_TRANSIENT_STATUSES = frozenset({408, 429, 502, 503, 504})

# One retry gives short edge incidents time to clear without creating a long startup loop.
_RETRY_DELAY_SECONDS = 5.0

# Cap on the body echoed into an ApiIdentityError, shared with the Mantic preflight so both gates document one cap.
BODY_PREVIEW_CHARS = 200


def _hijack_hint(host: str) -> str:
    """Tail of every hijack-flavored failure message. Names the likely cause and,
    critically, tells the operator NOT to retry with credentials."""
    return (
        f"looks like DNS parking/hijack or an imposter answering {host}; "
        f"do NOT retry with credentials; check `dig {host}` and the platform's status channels"
    )


class ApiIdentityError(RuntimeError):
    """Raised when the host answering a question platform's API base URL doesn't behave like the real API.

    Also the failure class of the Mantic tournament preflight (``mantic.preflight_mantic_tournaments``)
    when the token cannot be confirmed to hold forecast permission on the configured tournament: the
    same fail-shut-before-any-spend gate, one exception for the operator to grep.
    """


class ApiUnreachableError(ApiIdentityError):
    """Raised when a preflight attempt cannot reach a response that verifies API identity."""

    http_status: int | None

    def __init__(self, message: str, *, http_status: int | None) -> None:
        super().__init__(message)
        self.http_status = http_status


def _parse_json_object(body: str) -> dict[str, Any] | None:
    """Parse ``body`` as JSON, returning the object as a dict or None if it isn't a JSON object."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def verify_api_identity(base_url: str, *, timeout: float = 20.0) -> None:
    """Confirm ``base_url`` is answered by the real question-platform API before any authed call.

    Sends an unauthenticated GET (no token, no headers) to ``{base_url}/posts/?limit=1``
    through an isolated ``trust_env=False`` session (so no netrc/env credential is
    attached). Retries once after :data:`_RETRY_DELAY_SECONDS` on a request transport
    failure or a transient/5xx response, using the same timeout. Passes silently on a
    real API's fingerprint — an auth-gated 401/403 (Metaculus without a token) or a
    200 JSON object carrying ``results`` (Metaculus with a token; Mantic, whose posts
    list is public); raises ``ApiIdentityError`` for identity mismatches and
    ``ApiUnreachableError`` if the retry also fails transiently.
    """
    url = _preflight_url(base_url)
    host = urlparse(base_url).hostname or base_url
    preflight = f"API identity preflight for {host}"
    with requests.Session() as session:
        session.trust_env = False  # do not let ~/.netrc or proxy env inject credentials
        try:
            response = session.get(url, timeout=timeout, allow_redirects=False)
        except requests.RequestException:
            time.sleep(_RETRY_DELAY_SECONDS)
            response = _retry_preflight_request(session, url, timeout, host)
        else:
            if _is_unreachable_status(response.status_code):
                time.sleep(_RETRY_DELAY_SECONDS)
                response = _retry_preflight_request(session, url, timeout, host, first_status=response.status_code)

        status = response.status_code
        if _is_unreachable_status(status):
            body_preview = response.text[:BODY_PREVIEW_CHARS]
            if status in _TRANSIENT_STATUSES:
                raise ApiUnreachableError(
                    f"{preflight} got {status=} from {url!r} "
                    f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); transient edge throttle/server "
                    "condition — not necessarily an identity problem; a later retry of the whole run is "
                    "appropriate; do NOT retry with credentials now.",
                    http_status=status,
                )
            raise ApiUnreachableError(
                f"{preflight} got {status=} from {url!r} "
                f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); genuine or transient server error "
                f"from {host}; a later retry of the whole run may be appropriate; do NOT retry with credentials.",
                http_status=status,
            )

        _validate_api_identity_response(response, host, preflight, url)


def _retry_preflight_request(
    session: requests.Session, url: str, timeout: float, host: str, *, first_status: int | None = None
) -> requests.Response:
    """Retry once with the same unauthenticated session and full request timeout."""
    preflight = f"API identity preflight for {host}"
    try:
        return session.get(url, timeout=timeout, allow_redirects=False)
    except requests.RequestException as retry_error:
        first_attempt = (
            "failed without a response on both attempts"
            if first_status is None
            else f"retry after {first_status=} failed without a response"
        )
        raise ApiUnreachableError(
            f"{preflight} request for {url!r} {first_attempt} "
            f"({type(retry_error).__name__}: {retry_error}); API identity remains unverified; "
            f"a later retry of the whole run is appropriate; check `dig {host}` and platform status channels; "
            "do NOT retry with credentials.",
            http_status=None,
        ) from retry_error


def _is_unreachable_status(status: int) -> bool:
    return status in _TRANSIENT_STATUSES or 500 <= status < 600


def _validate_api_identity_response(response: requests.Response, host: str, preflight: str, url: str) -> None:
    """Accept the known API fingerprints and reject other non-transient responses."""
    status = response.status_code
    body_preview = response.text[:BODY_PREVIEW_CHARS]

    if status in _AUTH_GATED_STATUSES:
        logger.info(f"API identity preflight passed for {host} ({status=} auth-gated)")
        return

    if status == 200:
        parsed = _parse_json_object(response.text)
        if parsed is not None and "results" in parsed:
            logger.info(f"API identity preflight passed for {host} ({status=} JSON results payload)")
            return
        raise ApiIdentityError(
            f"{preflight} got {status=} from {url!r} but the body is not the expected JSON results "
            f"payload (first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); {_hijack_hint(host)}."
        )

    # Any other status — 404, a 3xx redirect (allow_redirects=False keeps it a
    # status, not a followed hop), an unexpected 2xx, or a stray 4xx.
    raise ApiIdentityError(
        f"{preflight} got unexpected {status=} from {url!r} "
        f"(first {BODY_PREVIEW_CHARS} chars: {body_preview!r}); {_hijack_hint(host)}."
    )


def verify_metaculus_api_identity(timeout: float = 20.0) -> None:
    """The Metaculus run modes' preflight: :func:`verify_api_identity` against the API root
    the bot's own Metaculus fetches use (see :func:`_api_base_url`)."""
    verify_api_identity(_api_base_url(), timeout=timeout)
