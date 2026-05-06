"""Production app bootstrap: env → real httpx clients + audit + orchestrator.

Test code uses `gateway.api.app.build_app(...)` with explicit fakes.
The `uvicorn` entrypoint and the Typer CLI both call
`build_production_app()`, which reads `.env`, constructs
`_HttpxOnPremiseClient` / `_HttpxCloudClient`, wires the audit logger
and orchestrator, then hands the result to `build_app`.

Demo mode (`GATEWAY_DEMO_MODE=true`) swaps the real clients for fakes
preloaded with one of the existing `application/on_premise/output/anonymized/`
fixtures. This allows running the full gateway HTTP flow (login →
visit → audio → finalize) end-to-end on any laptop without loading
MLX Whisper, Bielik, Vertex Gemini or Qdrant. Useful for thesis demo
videos and the Step 9 smoke test.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI

from gateway.api.app import build_app
from gateway.audit import GatewayAuditLogger
from gateway.clients.cloud import CloudClient, _FakeCloudClient, _HttpxCloudClient
from gateway.clients.on_premise import (
    OnPremiseClient,
    _FakeOnPremiseClient,
    _HttpxOnPremiseClient,
)
from gateway.config import GatewayConfig
from gateway.orchestrator import VisitOrchestrator
from gateway.storage import build_session_store, build_visit_store

log = logging.getLogger(__name__)

# Path to the canonical demo fixture (Graves-Basedow scenario; produces
# revealed diagnoses via the eponym filter and a non-trivial drug
# suggestion list).
_DEMO_SCENARIO = "endo-gra"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _demo_clients(cfg: GatewayConfig) -> tuple[OnPremiseClient, CloudClient]:
    """Build fake clients from existing on-prem + cloud output fixtures."""
    from shared.schemas import AgentSuggestion, AnonymizedSOAPNote

    soap_path = (
        _REPO_ROOT
        / "application"
        / "on_premise"
        / "output"
        / "anonymized"
        / _DEMO_SCENARIO
        / "anonymized_soap.json"
    )
    sugg_path = (
        _REPO_ROOT
        / "application"
        / "cloud"
        / "output"
        / "agent_suggestions"
        / _DEMO_SCENARIO
        / "suggestion.json"
    )
    if not soap_path.is_file() or not sugg_path.is_file():
        raise RuntimeError(
            f"demo fixtures not found ({soap_path}, {sugg_path}) — "
            "run Phase 2c + 3a outputs first or disable GATEWAY_DEMO_MODE"
        )
    soap = AnonymizedSOAPNote.model_validate_json(soap_path.read_text("utf-8"))
    sugg = AgentSuggestion.model_validate_json(sugg_path.read_text("utf-8"))
    # Demo on-prem fakes: any transcript_id resolves to the same canned SOAP
    # so the smoke flow doesn't need to coordinate IDs.
    on_prem = _DemoOnPrem(soap=soap)
    cloud = _FakeCloudClient(response=sugg)
    return on_prem, cloud


class _DemoOnPrem(_FakeOnPremiseClient):
    """Fake that returns the same `AnonymizedSOAPNote` for any transcript_id."""

    def __init__(self, *, soap: Any) -> None:
        super().__init__(responses={}, healthy=True)
        self._soap = soap

    async def anonymize_visit(self, **kwargs: Any) -> Any:
        # Track the call so smoke tests can introspect.
        self.calls.append(kwargs)
        return self._soap


def _is_truthy(raw: str | None) -> bool:
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


def build_production_app() -> FastAPI:
    """Read `.env`, instantiate clients (real or demo), build the gateway app."""
    # `override=True`: the project's `.env` is authoritative — stale shell
    # state (e.g. an old `GATEWAY_DOCTOR_PASSWORD_HASH` lingering in the
    # parent process) must not silently shadow the committed config.
    load_dotenv(override=True)
    cfg = GatewayConfig.from_env()
    demo_mode = _is_truthy(os.environ.get("GATEWAY_DEMO_MODE"))

    if cfg.host not in {"127.0.0.1", "localhost"}:
        log.warning(
            "GATEWAY_HOST=%s is not localhost — production deployment must "
            "front the gateway with TLS-terminating reverse proxy",
            cfg.host,
        )
    sessions = build_session_store(Path(cfg.output_dir) / "sessions")
    visits = build_visit_store(Path(cfg.output_dir) / "visits")

    on_premise_client: OnPremiseClient
    cloud_client: CloudClient
    if demo_mode:
        log.warning("GATEWAY_DEMO_MODE=true — using fake on-prem + cloud clients")
        on_premise_client, cloud_client = _demo_clients(cfg)
    else:
        on_premise_client = _HttpxOnPremiseClient(cfg.on_premise_api_url)
        cloud_client = _HttpxCloudClient(cfg.cloud_api_url)

    audit = GatewayAuditLogger(
        audit_dir=Path(cfg.audit_dir),
        retention_days=cfg.audit_retention_days,
    )

    orchestrator = VisitOrchestrator(
        on_premise=on_premise_client,
        cloud=cloud_client,
        visit_store=visits,
        audit=audit,
        on_premise_audit_dir=Path(cfg.on_premise_audit_dir),
    )

    app = build_app(
        config=cfg,
        session_store=sessions,
        visit_store=visits,
        orchestrator=orchestrator,
        on_premise_client=on_premise_client,
        cloud_client=cloud_client,
    )
    app.state.audit = audit
    app.state.demo_mode = demo_mode
    return app
