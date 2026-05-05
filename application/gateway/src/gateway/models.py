"""In-process dataclasses for gateway sessions and visits.

These are NOT cross-tier types — they live exclusively inside the
gateway process. The cross-tier-safe response shape is
`shared.schemas.visit.VisitSummary`. Persisting these to disk is
handled by `gateway.storage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

from shared.schemas import Specialization, VisitStatus, VisitSummary


@dataclass(frozen=True)
class DoctorIdentity:
    """The single demo doctor for the thesis. Multi-tenant deferred."""

    username: str
    full_name: str = "Demo Lekarz"


@dataclass
class Session:
    """Gateway-minted session bound to a doctor identity.

    `session_id` is the cookie value AND the gateway-side identifier
    handed downstream to the on-prem anonymizer (which expects a
    gateway-minted UUID per Phase 2c contract).
    """

    session_id: UUID
    doctor: DoctorIdentity
    created_at: datetime
    expires_at: datetime
    last_seen: datetime

    def is_expired(self, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        return now >= self.expires_at


@dataclass
class Visit:
    """A single doctor-patient visit, owned by exactly one session.

    The state machine progresses through `VisitStatus` values; cross-
    session reads are forbidden at the API layer (gateway checks
    `visit.session_id == request.state.session.session_id`).
    """

    visit_id: UUID
    session_id: UUID
    status: VisitStatus
    specialization: Specialization | None = None
    visit_date: date | None = None
    audio_path: Path | None = None
    audio_sha256: str | None = None
    transcript_id: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def to_summary(self) -> VisitSummary:
        duration_ms: int | None = None
        if self.completed_at is not None:
            delta = self.completed_at - self.created_at
            duration_ms = int(delta.total_seconds() * 1000)
        return VisitSummary(
            visit_id=self.visit_id,
            status=self.status,
            created_at=self.created_at,
            completed_at=self.completed_at,
            specialization=self.specialization,
            visit_date=self.visit_date,
            duration_ms=duration_ms,
            error=self.error,
        )
