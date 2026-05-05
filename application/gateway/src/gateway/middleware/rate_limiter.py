"""Rate-limit middleware — slowapi-based, keyed by session_id or IP.

Wired in `gateway.api.app.build_app` only when
`config.rate_limit_enabled` is True. Tests typically run with it
disabled to avoid false positives from rapid sequential requests.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from gateway.api.auth import SESSION_COOKIE_NAME


def _key_func(request: Request) -> str:
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid:
        return f"session:{sid}"
    return f"ip:{get_remote_address(request)}"


def build_limiter(per_min: int) -> Limiter:
    """Build a Limiter with default `<per_min>/minute` budget."""
    return Limiter(key_func=_key_func, default_limits=[f"{per_min}/minute"])


__all__ = ["RateLimitExceeded", "build_limiter"]
