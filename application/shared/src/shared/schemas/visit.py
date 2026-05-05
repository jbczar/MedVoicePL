"""Cross-tier visit lifecycle types: VisitStatus enum + VisitSummary.

`VisitSummary` is the trust-boundary-safe response shape served to the
frontend (Phase 3c). It carries NO PII — no patient names, no audio
hashes, no transcript IDs that could re-identify. The doctor-facing UI
gets the full revealed `AgentSuggestion` separately, after the gateway
de-anonymizes it via the on-prem token map.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from shared.schemas.anonymized import Specialization


class VisitStatus(StrEnum):
    """Visit state machine states.

    Linear progression for the happy path: draft → audio_uploaded →
    anonymizing → suggesting → completed. `failed` is a terminal sink
    from any intermediate state. `transcribing` / `structuring` exist
    for future async observability but are not currently observable
    from the gateway side (Phase 2c collapses them inside the on-prem
    endpoint).
    """

    draft = "draft"
    audio_uploaded = "audio_uploaded"
    transcribing = "transcribing"
    structuring = "structuring"
    anonymizing = "anonymizing"
    suggesting = "suggesting"
    completed = "completed"
    failed = "failed"


class VisitSummary(BaseModel):
    """Trust-boundary-safe view of a visit. No PII fields.

    Gateway returns this on `GET /v1/visits/{vid}` and embeds it in
    finalize responses. The frontend uses it for status polling and
    the visit list. Anything PII-bearing (revealed agent suggestion,
    token map) is served via separate, session-scoped endpoints.
    """

    model_config = ConfigDict(extra="forbid")

    visit_id: UUID
    status: VisitStatus
    created_at: datetime
    completed_at: datetime | None = None
    specialization: Specialization | None = None
    visit_date: date | None = None
    duration_ms: int | None = None
    error: str | None = None
