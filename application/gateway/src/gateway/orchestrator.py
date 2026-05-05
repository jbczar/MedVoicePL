"""Visit orchestrator: chains on-prem anonymization → cloud agent.

Synchronous from the request handler's perspective: `finalize` blocks
until the full pipeline produces an `AgentSuggestion`. State machine
transitions are persisted at every boundary so a poll on
`GET /v1/visits/{vid}` reflects current step.

Idempotency: re-finalizing a `completed` visit returns the cached
suggestion from disk (`<visit_dir>/suggestion.json`) without calling
downstream. `force=True` bypasses cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from gateway.clients.cloud import CloudClient, CloudError
from gateway.clients.on_premise import OnPremiseClient, OnPremiseError
from gateway.models import Visit
from gateway.storage import VisitStore, _atomic_write_json
from shared.schemas import AgentSuggestion, VisitStatus

log = logging.getLogger(__name__)


class OrchestratorError(RuntimeError):
    """Raised when the orchestrator could not finalize a visit."""


class GatewayAuditPort(Protocol):
    """Slim port the orchestrator writes to. Real impl lands in Step 8."""

    def log_visit_complete(self, visit: Visit, suggestion: AgentSuggestion) -> None: ...
    def log_failure(self, visit: Visit, error_type: str, message: str) -> None: ...


class _NullAudit:
    def log_visit_complete(self, visit: Visit, suggestion: AgentSuggestion) -> None:
        return None

    def log_failure(self, visit: Visit, error_type: str, message: str) -> None:
        return None


@dataclass
class VisitOrchestrator:
    on_premise: OnPremiseClient
    cloud: CloudClient
    visit_store: VisitStore
    audit: GatewayAuditPort = _NullAudit()  # noqa: RUF009 — intentional default

    async def finalize(self, visit_id: UUID, *, force: bool = False) -> AgentSuggestion:
        visit = self.visit_store.get(visit_id)
        if visit is None:
            raise OrchestratorError(f"visit not found: {visit_id}")

        # Idempotency: completed visits return cached suggestion.
        suggestion_path = self.visit_store.visit_dir(visit_id) / "suggestion.json"
        if (
            not force
            and visit.status == VisitStatus.completed
            and suggestion_path.is_file()
        ):
            try:
                return AgentSuggestion.model_validate_json(suggestion_path.read_text("utf-8"))
            except ValueError:
                log.warning("stale suggestion cache for %s — recomputing", visit_id)

        if visit.status not in {VisitStatus.audio_uploaded, VisitStatus.completed, VisitStatus.failed}:
            raise OrchestratorError(
                f"visit {visit_id} not ready for finalize (status={visit.status.value})"
            )
        if visit.specialization is None or visit.visit_date is None:
            raise OrchestratorError(
                f"visit {visit_id} missing specialization or visit_date"
            )

        # Step 1 — on-premise anonymization.
        visit.status = VisitStatus.anonymizing
        visit.error = None
        self.visit_store.save(visit)
        try:
            transcript_id = str(visit.visit_id)[:8]
            visit.transcript_id = transcript_id
            soap = await self.on_premise.anonymize_visit(
                transcript_id=transcript_id,
                session_id=visit.session_id,
                specialization=visit.specialization,
                visit_date=visit.visit_date,
                force=force,
            )
        except OnPremiseError as exc:
            self._fail(visit, "on_premise_error", str(exc))
            raise OrchestratorError(f"on-prem failed: {exc}") from exc

        # Persist the SOAP payload (audit copy) before pinging cloud.
        soap_path = self.visit_store.visit_dir(visit_id) / "soap.json"
        _atomic_write_json(soap_path, soap.model_dump(mode="json"))

        # Step 2 — cloud agent suggestion.
        visit.status = VisitStatus.suggesting
        self.visit_store.save(visit)
        try:
            suggestion = await self.cloud.suggest(soap, force=force)
        except CloudError as exc:
            self._fail(visit, "cloud_error", str(exc))
            raise OrchestratorError(f"cloud failed: {exc}") from exc

        # Persist suggestion + complete visit.
        _atomic_write_json(suggestion_path, suggestion.model_dump(mode="json"))
        visit.status = VisitStatus.completed
        visit.completed_at = datetime.now(UTC)
        self.visit_store.save(visit)
        try:
            self.audit.log_visit_complete(visit, suggestion)
        except Exception:
            log.exception("audit log_visit_complete failed for %s", visit_id)
        return suggestion

    def _fail(self, visit: Visit, error_type: str, message: str) -> None:
        visit.status = VisitStatus.failed
        visit.error = f"{error_type}: {message}"
        self.visit_store.save(visit)
        try:
            self.audit.log_failure(visit, error_type, message)
        except Exception:
            log.exception("audit log_failure failed for %s", visit.visit_id)


__all__ = ["GatewayAuditPort", "OrchestratorError", "VisitOrchestrator"]
