"""Visit CRUD + audio upload endpoints.

State machine:
    POST /v1/visits                  draft
    POST /v1/visits/{vid}/audio      audio_uploaded   (mutates only when status=draft)
    POST /v1/visits/{vid}/finalize   anonymizing → suggesting → completed | failed
                                     (Step 5 wires the orchestrator)
    GET  /v1/visits/{vid}            VisitSummary  (poll-friendly)

Cross-session isolation: every visit-scoped route checks
`visit.session_id == request.state.session.session_id`. A doctor never
reads someone else's visit even if they guess the visit_id.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict

from gateway.api.auth import require_session
from gateway.models import Session, Visit
from gateway.orchestrator import OrchestratorError, VisitOrchestrator
from gateway.storage import VisitStore
from shared.schemas import AgentSuggestion, Specialization, VisitStatus, VisitSummary

log = logging.getLogger(__name__)

# 50 MB cap. Multipart streams chunked to disk; this is enforced both by
# Content-Length (cheap pre-check) and by counting bytes during stream.
MAX_AUDIO_BYTES = 50 * 1024 * 1024
ACCEPTED_AUDIO_MIMES = {"audio/wav", "audio/x-wav", "audio/wave"}


class CreateVisitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specialization: Specialization | None = None
    visit_date: date | None = None


class CreateVisitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visit_id: UUID
    status: VisitStatus


class AudioUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visit_id: UUID
    status: VisitStatus
    audio_sha256: str


def _get_visit_store(request: Request) -> VisitStore:
    return request.app.state.visit_store


def _get_orchestrator(request: Request) -> VisitOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail="orchestrator_not_configured",
        )
    return orch


class FinalizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visit: VisitSummary
    suggestion: AgentSuggestion


router = APIRouter(prefix="/v1/visits", tags=["visits"])


@router.post("", response_model=CreateVisitResponse, status_code=status.HTTP_201_CREATED)
def create_visit(
    body: CreateVisitRequest,
    session: Annotated[Session, Depends(require_session)],
    visits: Annotated[VisitStore, Depends(_get_visit_store)],
) -> CreateVisitResponse:
    visit = Visit(
        visit_id=uuid4(),
        session_id=session.session_id,
        status=VisitStatus.draft,
        specialization=body.specialization,
        visit_date=body.visit_date,
    )
    visits.save(visit)
    return CreateVisitResponse(visit_id=visit.visit_id, status=visit.status)


@router.get("/{visit_id}", response_model=VisitSummary)
def get_visit(
    visit_id: UUID,
    session: Annotated[Session, Depends(require_session)],
    visits: Annotated[VisitStore, Depends(_get_visit_store)],
) -> VisitSummary:
    visit = _load_owned_visit(visit_id, visits=visits, session=session)
    return visit.to_summary()


@router.post("/{visit_id}/audio", response_model=AudioUploadResponse)
async def upload_audio(
    visit_id: UUID,
    session: Annotated[Session, Depends(require_session)],
    visits: Annotated[VisitStore, Depends(_get_visit_store)],
    file: Annotated[UploadFile, File(description="WAV file, max 50 MB")],
) -> AudioUploadResponse:
    visit = _load_owned_visit(visit_id, visits=visits, session=session)

    # Idempotency: if already uploaded with matching SHA, return cached.
    # Re-upload overwrites only if status == draft.
    if visit.status == VisitStatus.audio_uploaded:
        return AudioUploadResponse(
            visit_id=visit.visit_id,
            status=visit.status,
            audio_sha256=visit.audio_sha256 or "",
        )
    if visit.status != VisitStatus.draft:
        raise HTTPException(
            status_code=409,
            detail=f"cannot_upload_audio_in_status:{visit.status.value}",
        )
    if file.content_type and file.content_type not in ACCEPTED_AUDIO_MIMES:
        raise HTTPException(status_code=415, detail="unsupported_media_type")

    target = visits.visit_dir(visit.visit_id) / "audio.wav"
    sha = hashlib.sha256()
    written = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_AUDIO_BYTES:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="audio_too_large")
                sha.update(chunk)
                out.write(chunk)
    finally:
        await file.close()

    _set_owner_only(target)
    visit.audio_path = target
    visit.audio_sha256 = sha.hexdigest()
    visit.status = VisitStatus.audio_uploaded
    visits.save(visit)
    return AudioUploadResponse(
        visit_id=visit.visit_id,
        status=visit.status,
        audio_sha256=visit.audio_sha256,
    )


@router.post("/{visit_id}/finalize", response_model=FinalizeResponse)
async def finalize_visit(
    visit_id: UUID,
    session: Annotated[Session, Depends(require_session)],
    visits: Annotated[VisitStore, Depends(_get_visit_store)],
    orchestrator: Annotated[VisitOrchestrator, Depends(_get_orchestrator)],
) -> FinalizeResponse:
    visit = _load_owned_visit(visit_id, visits=visits, session=session)
    if visit.status not in {
        VisitStatus.audio_uploaded,
        VisitStatus.completed,
        VisitStatus.failed,
    }:
        raise HTTPException(
            status_code=409,
            detail=f"cannot_finalize_in_status:{visit.status.value}",
        )
    try:
        suggestion = await orchestrator.finalize(visit_id)
    except OrchestratorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    refreshed = visits.get(visit_id)
    assert refreshed is not None
    return FinalizeResponse(visit=refreshed.to_summary(), suggestion=suggestion)


def _load_owned_visit(
    visit_id: UUID,
    *,
    visits: VisitStore,
    session: Session,
) -> Visit:
    visit = visits.get(visit_id)
    if visit is None:
        raise HTTPException(status_code=404, detail="visit_not_found")
    if visit.session_id != session.session_id:
        # Same response as not_found — don't reveal existence.
        raise HTTPException(status_code=404, detail="visit_not_found")
    return visit


def _set_owner_only(path: Path) -> None:
    import os
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        log.warning("chmod 0600 failed for %s: %s", path, exc)
