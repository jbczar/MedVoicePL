"""Schema tests for `shared.schemas.anonymized`.

Pure Pydantic round-trip + validation rules. No model downloads,
no on-premise imports.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AnonymizedAssessment,
    AnonymizedMetadata,
    AnonymizedObjective,
    AnonymizedPlan,
    AnonymizedSOAPNote,
    AnonymizedSubjective,
    Specialization,
)


def _valid_metadata(**overrides: object) -> AnonymizedMetadata:
    base = {
        "anonymizer_version": "2c.0",
        "n_entities_per_type": {"PERSON": 2, "PHONE_NUMBER": 1},
        "visit_date": date(2026, 4, 26),
        "specialization": Specialization.kardiologia,
        "session_id": uuid4(),
    }
    base.update(overrides)  # type: ignore[arg-type]
    return AnonymizedMetadata(**base)  # type: ignore[arg-type]


def _valid_note(**overrides: object) -> AnonymizedSOAPNote:
    base = {
        "subjective": AnonymizedSubjective(
            chief_complaint="<PERSON_1> reports headaches",
            history_of_present_illness="Started two days ago.",
        ),
        "objective": AnonymizedObjective(),
        "assessment": AnonymizedAssessment(primary_diagnosis="napięciowy ból głowy"),
        "plan": AnonymizedPlan(),
        "metadata": _valid_metadata(),
    }
    base.update(overrides)  # type: ignore[arg-type]
    return AnonymizedSOAPNote(**base)  # type: ignore[arg-type]


def test_round_trip_json() -> None:
    note = _valid_note()
    payload = note.model_dump_json()
    clone = AnonymizedSOAPNote.model_validate_json(payload)
    assert clone.subjective.chief_complaint == "<PERSON_1> reports headaches"
    assert clone.metadata.specialization == Specialization.kardiologia


def test_extra_forbid_rejects_unknown_fields() -> None:
    """Surplus keys at any level fail validation — `extra="forbid"`."""
    with pytest.raises(ValidationError):
        AnonymizedSOAPNote.model_validate(
            {
                **_valid_note().model_dump(mode="json"),
                "secret_pii_field": "boom",
            }
        )


def test_visit_date_rejects_datetime() -> None:
    """`visit_date` is a `date`, not `datetime` — strips appointment timing.

    A pure datetime input must NOT silently pass through (which would
    leak ms-precision timing across the trust boundary).
    """
    with pytest.raises(ValidationError):
        _valid_metadata(visit_date=datetime(2026, 4, 26, 14, 35, 0))


def test_visit_date_accepts_iso_date_string() -> None:
    """ISO date strings parse cleanly — `2026-04-26` works."""
    md = _valid_metadata(visit_date="2026-04-26")  # type: ignore[arg-type]
    assert md.visit_date == date(2026, 4, 26)


def test_session_id_rejects_non_uuid() -> None:
    """Random strings must not slip through as `session_id`."""
    with pytest.raises(ValidationError):
        _valid_metadata(session_id="not-a-uuid")


def test_session_id_accepts_uuid_string() -> None:
    """Standard UUID string round-trips into a `UUID` object."""
    sid = "12345678-1234-5678-1234-567812345678"
    md = _valid_metadata(session_id=sid)  # type: ignore[arg-type]
    assert md.session_id == UUID(sid)


def test_metadata_strips_forbidden_fields() -> None:
    """The metadata block must reject the PII-leaking fields we dropped:
    `source_transcript_sha256`, `slm_backend`, `generated_at`, etc.
    """
    sha = "a" * 64
    base = {
        "anonymizer_version": "2c.0",
        "n_entities_per_type": {},
        "visit_date": "2026-04-26",
        "specialization": "kardiologia",
        "session_id": str(uuid4()),
    }
    # Each forbidden field must individually trip extra=forbid.
    for forbidden in (
        "source_transcript_sha256",
        "source_audio_sha256",
        "slm_backend",
        "slm_model",
        "slm_device",
        "generated_at",
        "scenario_id",
    ):
        payload = {**base, forbidden: sha}
        with pytest.raises(ValidationError):
            AnonymizedMetadata.model_validate(payload)


def test_anonymized_soap_note_is_importable_from_shared_schemas() -> None:
    """Positive contract — gateway/cloud must be able to import the type."""
    from shared.schemas import AnonymizedSOAPNote as Imported

    assert Imported is AnonymizedSOAPNote


def test_provenance_defaults_empty() -> None:
    """Each section's `provenance: list[int]` defaults to `[]`."""
    s = AnonymizedSubjective(chief_complaint="x", history_of_present_illness="y")
    assert s.provenance == []


def test_icd10_pattern_inherited() -> None:
    """The ICD-10 regex from the on-prem schema is preserved on the
    anonymized side (clinical content unchanged across boundary).
    """
    AnonymizedAssessment(primary_diagnosis="dx", icd10_code="I10.0")
    with pytest.raises(ValidationError):
        AnonymizedAssessment(
            primary_diagnosis="dx",
            icd10_code="I10.0 (nadciśnienie)",
        )
