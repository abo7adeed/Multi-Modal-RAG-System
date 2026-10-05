"""
HTTP middleware: request identity, authentication, rate limiting and
access logging.

Each concern is a separate middleware so they compose predictably and
can be tested one at a time. Order matters:

  RequestContext  - assigns the ID every later layer logs
  APIKeyAuth      - rejects unauthenticated callers before any work
  RateLimit       - only counts requests that passed auth
"""
import logging
import secrets
import threading
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

#: Header carrying the request ID in and out. Echoed so a client can
#: quote it in a bug report and an operator can find the exact log
#: lines for that request.
REQUEST_ID_HEADER = "X-Request-ID"

#: Header a client may set to supply its own correlation ID.
REQUEST_ID_INBOUND_HEADER = "X-Request-ID"

#: Compared with a constant-time comparison, so a wrong key cannot be
#: recovered by timing the response.
_API_KEY_HEADER = "X-API-Key"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Assign every request an ID and echo it back.

    Without one, a log line cannot be tied to a user's report: several
    requests interleave in the log and there is nothing to join on.
    """

    async def dispatch(self, request: Request, call_next):
        # Honour an upstream ID when a proxy already assigned one, so
        # the whole chain shares it; otherwise mint a fresh one.
        request_id = (
            request.headers.get(REQUEST_ID_INBOUND_HEADER)
            or secrets.token_hex(8)
        )
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Require a shared secret when one is configured.

    Disabled (no-op) while `api_key` is empty, so local development
    needs no configuration and tests are unaffected.
    """

    async def dispatch(self, request: Request, call_next):
        expected = settings.api_key

        if not expected:
            return await call_next(request)

        provided = request.headers.get(_API_KEY_HEADER, "")

        if not secrets.compare_digest(provided, expected):
            logger.warning(
                "auth_failed path=%s request_id=%s",
                request.url.path,
                getattr(request.state, "request_id", None),
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key.",
                        "type": "AuthenticationError",
                    }
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)


class SlidingWindowRateLimiter:
    """
    Per-key request counter over a sliding window.

    A fixed window would let a caller send twice the limit across a
    boundary; keeping the actual timestamps avoids that. The store is
    an in-process deque per key, which is enough for a single worker
    and honest about its limit - see the note in the middleware.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """
        Record a hit for `key`.

        Returns (allowed, retry_after_seconds). retry_after is 0 when
        the request is allowed.
        """
        now = now if now is not None else time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = max(
                    1, int(hits[0] + self.window_seconds - now) + 1
                )
                return False, retry_after

            hits.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Throttle requests per client IP.

    Runs after authentication so an unauthenticated flood is rejected
    by the cheap check instead of consuming limiter capacity.

    The counter is per process: with several workers behind a load
    balancer each keeps its own window, so the effective limit is
    `limit x workers`. A shared store (Redis) is the fix if that
    matters; for a single-worker deployment it is exact.
    """

    def __init__(self, app, limiter: SlidingWindowRateLimiter | None = None):
        super().__init__(app)
        self.limiter = limiter or SlidingWindowRateLimiter(
            limit=settings.api_rate_limit_requests,
            window_seconds=settings.api_rate_limit_window_seconds,
        )

    async def dispatch(self, request: Request, call_next):
        # The proxy address is the client as far as this app is
        # concerned; the connection peer is the proxy itself.
        client = request.client
        key = client.host if client else "unknown"

        allowed, retry_after = self.limiter.check(key)

        if not allowed:
            logger.warning(
                "rate_limited path=%s client=%s request_id=%s",
                request.url.path,
                key,
                getattr(request.state, "request_id", None),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": (
                            "Too many requests. Please slow down."
                        ),
                        "type": "RateLimitError",
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.limiter.limit),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.limit)
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    One log line per request, with its ID, status and duration.

    Without the duration there is no way to tell a slow model from a
    slow index; without the ID the line cannot be matched to a report.
    """

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)

        duration_ms = (time.perf_counter() - started) * 1000

        # Health checks run every few seconds; logging them buries
        # everything else.
        if request.url.path.endswith("/health"):
            return response

        logger.info(
            "request method=%s path=%s status=%d duration_ms=%.1f "
            "request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            getattr(request.state, "request_id", None),
        )

        return response