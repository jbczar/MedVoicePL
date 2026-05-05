"""Tests for in-process gateway dataclasses."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from gateway.models import DoctorIdentity, Session, Visit
from shared.schemas import Specialization, VisitStatus


def _make_session(*, expires_in: timedelta = timedelta(hours=8)) -> Session:
    now = datetime.now(UTC)
    return Session(
        session_id=uuid4(),
        doctor=DoctorIdentity(username="demo_doctor"),
        created_at=now,
        expires_at=now + expires_in,
        last_seen=now,
    )


def test_session_is_expired_after_ttl() -> None:
    session = _make_session(expires_in=timedelta(seconds=-1))
    assert session.is_expired() is True


def test_session_is_active_within_ttl() -> None:
    session = _make_session()
    assert session.is_expired() is False


def test_doctor_identity_default_full_name() -> None:
    doc = DoctorIdentity(username="demo_doctor")
    assert doc.full_name == "Demo Lekarz"


def test_visit_to_summary_strips_internal_fields() -> None:
    visit = Visit(
        visit_id=uuid4(),
        session_id=uuid4(),
        status=VisitStatus.audio_uploaded,
        specialization=Specialization.kardiologia,
        visit_date=date(2026, 5, 5),
        audio_path=Path("/tmp/audio.wav"),
        audio_sha256="abc" * 21 + "a",
        transcript_id="abc12345",
    )
    summary = visit.to_summary()
    dumped = summary.model_dump()
    # No PII / internal fields leak.
    assert "audio_path" not in dumped
    assert "audio_sha256" not in dumped
    assert "transcript_id" not in dumped
    assert "session_id" not in dumped
    assert dumped["status"] == "audio_uploaded"


def test_visit_summary_duration_ms_computed_when_completed() -> None:
    created = datetime(2026, 5, 5, 10, 0, 0, tzinfo=UTC)
    completed = datetime(2026, 5, 5, 10, 0, 12, 500_000, tzinfo=UTC)
    visit = Visit(
        visit_id=uuid4(),
        session_id=uuid4(),
        status=VisitStatus.completed,
        created_at=created,
        completed_at=completed,
    )
    assert visit.to_summary().duration_ms == 12500


def test_visit_summary_duration_none_when_not_completed() -> None:
    visit = Visit(
        visit_id=uuid4(),
        session_id=uuid4(),
        status=VisitStatus.draft,
    )
    assert visit.to_summary().duration_ms is None
