"""FastAPI app factory for the gateway.

`build_app(...)` lets tests inject fakes; the production entrypoint
calls `build_app()` with defaults loaded from env.

Middleware stack (outermost → innermost):
    CORS → RateLimit → Routes
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from gateway.api import auth as auth_router
from gateway.api import health as health_router
from gateway.api import visits as visits_router
from gateway.clients.cloud import CloudClient
from gateway.clients.on_premise import OnPremiseClient
from gateway.config import GatewayConfig
from gateway.middleware.rate_limiter import build_limiter
from gateway.orchestrator import VisitOrchestrator
from gateway.storage import SessionStore, VisitStore, build_session_store, build_visit_store


def build_app(
    *,
    config: GatewayConfig | None = None,
    session_store: SessionStore | None = None,
    visit_store: VisitStore | None = None,
    orchestrator: VisitOrchestrator | None = None,
    on_premise_client: OnPremiseClient | None = None,
    cloud_client: CloudClient | None = None,
) -> FastAPI:
    cfg = config or GatewayConfig.from_env()
    sessions = session_store or build_session_store(Path(cfg.output_dir) / "sessions")
    visits = visit_store or build_visit_store(Path(cfg.output_dir) / "visits")

    app = FastAPI(title="MedVoicePL Gateway", version="0.1.0")

    # Dependency providers.
    app.dependency_overrides[auth_router.get_config] = lambda: cfg
    app.dependency_overrides[auth_router.get_session_store] = lambda: sessions

    # State stash (for state-based dep providers in visits/health routers).
    app.state.config = cfg
    app.state.session_store = sessions
    app.state.visit_store = visits
    app.state.orchestrator = orchestrator
    app.state.on_premise_client = on_premise_client
    app.state.cloud_client = cloud_client

    # Routers.
    app.include_router(auth_router.router)
    app.include_router(visits_router.router)
    app.include_router(health_router.router)

    # Rate limit (innermost; only when enabled).
    if cfg.rate_limit_enabled:
        limiter = build_limiter(cfg.rate_limit_per_min)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

    # CORS (outermost).
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        )

    return app


@contextmanager
def temp_app(config: GatewayConfig) -> Iterator[FastAPI]:
    app = build_app(config=config)
    try:
        yield app
    finally:
        app.dependency_overrides.clear()
