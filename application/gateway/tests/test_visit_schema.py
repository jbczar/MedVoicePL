"""Tests for cross-tier-safe VisitSummary + VisitStatus."""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from shared.schemas import Specialization, VisitStatus, VisitSummary


def test_visit_status_values() -> None:
    expected = {
        "draft",
        "audio_uploaded",
        "transcribing",
        "structuring",
        "anonymizing",
        "suggesting",
        "completed",
        "failed",
    }
    assert {s.value for s in VisitStatus} == expected


def test_visit_summary_round_trip() -> None:
    summary = VisitSummary(
        visit_id=uuid4(),
        status=VisitStatus.completed,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        specialization=Specialization.kardiologia,
        visit_date=date(2026, 5, 5),
        duration_ms=12345,
    )
    dumped = summary.model_dump_json()
    rebuilt = VisitSummary.model_validate_json(dumped)
    assert rebuilt.status == VisitStatus.completed
    assert rebuilt.duration_ms == 12345


def test_visit_summary_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        VisitSummary(
            visit_id=uuid4(),
            status=VisitStatus.draft,
            created_at=datetime.now(UTC),
            patient_name="leak",  # type: ignore[call-arg]
        )


def test_visit_summary_minimal_fields() -> None:
    summary = VisitSummary(
        visit_id=uuid4(),
        status=VisitStatus.draft,
        created_at=datetime.now(UTC),
    )
    assert summary.completed_at is None
    assert summary.specialization is None
    assert summary.duration_ms is None
    assert summary.error is None
