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
from pathlib import Path
from typing import Protocol
from uuid import UUID

from gateway.clients.cloud import CloudClient, CloudError
from gateway.clients.on_premise import OnPremiseClient, OnPremiseError
from gateway.models import Visit
from gateway.reverse_map import TokenMapNotFoundError, load_token_map, reveal_for_doctor
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
class FinalizeResult:
    """Outcome of a successful `finalize`.

    `suggestion` is the raw output from the cloud agent (still
    placeholder-tokenized). `revealed_suggestion` is the doctor-facing
    view with PII restored; equals `suggestion` if no token map was
    available. `unmapped_token_count` flags placeholders we couldn't
    resolve (operational hint, not a hard failure).
    """

    suggestion: AgentSuggestion
    revealed_suggestion: AgentSuggestion
    unmapped_token_count: int


@dataclass
class VisitOrchestrator:
    on_premise: OnPremiseClient
    cloud: CloudClient
    visit_store: VisitStore
    audit: GatewayAuditPort = _NullAudit()  # noqa: RUF009 — intentional default
    on_premise_audit_dir: Path | None = None

    async def finalize(self, visit_id: UUID, *, force: bool = False) -> FinalizeResult:
        visit = self.visit_store.get(visit_id)
        if visit is None:
            raise OrchestratorError(f"visit not found: {visit_id}")

        # Idempotency: completed visits return cached suggestion.
        visit_dir = self.visit_store.visit_dir(visit_id)
        suggestion_path = visit_dir / "suggestion.json"
        revealed_path = visit_dir / "suggestion_revealed.json"
        if (
            not force
            and visit.status == VisitStatus.completed
            and suggestion_path.is_file()
        ):
            try:
                cached = AgentSuggestion.model_validate_json(suggestion_path.read_text("utf-8"))
                cached_revealed = cached
                if revealed_path.is_file():
                    try:
                        cached_revealed = AgentSuggestion.model_validate_json(
                            revealed_path.read_text("utf-8")
                        )
                    except ValueError:
                        log.warning("stale revealed cache for %s — using raw", visit_id)
                return FinalizeResult(
                    suggestion=cached,
                    revealed_suggestion=cached_revealed,
                    unmapped_token_count=0,
                )
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

        # Step 1a — on-premise STT (Phase 2a via /v1/transcripts).
        # If `visit.transcript_id` is already set we reuse it (idempotent
        # replay or a test fixture pre-populated the id). Otherwise we
        # upload the audio file and derive the transcript id from on-prem.
        visit.error = None
        if visit.transcript_id is None:
            if visit.audio_path is None or not visit.audio_path.is_file():
                raise OrchestratorError(
                    f"visit {visit_id} has no audio uploaded"
                )
            visit.status = VisitStatus.transcribing
            self.visit_store.save(visit)
            try:
                transcript_id = await self.on_premise.transcribe_audio(visit.audio_path)
            except OnPremiseError as exc:
                self._fail(visit, "on_premise_error", f"STT failed: {exc}")
                raise OrchestratorError(f"on-prem STT failed: {exc}") from exc
            visit.transcript_id = transcript_id
            self.visit_store.save(visit)
        else:
            transcript_id = visit.transcript_id

        # Step 1b — on-premise SOAP extraction + anonymization.
        visit.status = VisitStatus.anonymizing
        self.visit_store.save(visit)
        try:
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

        # Persist raw suggestion + reveal for doctor's view.
        _atomic_write_json(suggestion_path, suggestion.model_dump(mode="json"))
        revealed, unmapped = self._reveal(suggestion, visit=visit)
        _atomic_write_json(revealed_path, revealed.model_dump(mode="json"))

        visit.status = VisitStatus.completed
        visit.completed_at = datetime.now(UTC)
        self.visit_store.save(visit)
        try:
            self.audit.log_visit_complete(visit, suggestion)
        except Exception:
            log.exception("audit log_visit_complete failed for %s", visit_id)
        return FinalizeResult(
            suggestion=suggestion,
            revealed_suggestion=revealed,
            unmapped_token_count=unmapped,
        )

    def _reveal(
        self, suggestion: AgentSuggestion, *, visit: Visit
    ) -> tuple[AgentSuggestion, int]:
        """Load the on-prem token map for `visit` and reveal the suggestion.

        Fail-soft: if the token map is unavailable (audit dir not
        configured, file missing, malformed) the raw suggestion is
        returned and a warning is logged. Doctor sees placeholders;
        nothing crashes.
        """
        if self.on_premise_audit_dir is None:
            return suggestion, 0
        try:
            token_map = load_token_map(
                audit_root=self.on_premise_audit_dir,
                session_id=visit.session_id,
                visit_date=visit.visit_date,
            )
        except TokenMapNotFoundError as exc:
            log.warning("token map unavailable for %s: %s", visit.visit_id, exc)
            return suggestion, 0
        return reveal_for_doctor(suggestion, token_map=token_map)

    def _fail(self, visit: Visit, error_type: str, message: str) -> None:
        visit.status = VisitStatus.failed
        visit.error = f"{error_type}: {message}"
        self.visit_store.save(visit)
        try:
            self.audit.log_failure(visit, error_type, message)
        except Exception:
            log.exception("audit log_failure failed for %s", visit.visit_id)


__all__ = [
    "FinalizeResult",
    "GatewayAuditPort",
    "OrchestratorError",
    "VisitOrchestrator",
]
