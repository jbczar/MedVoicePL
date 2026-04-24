"""Schema tests for `on_premise.structuring.soap_schema`.

No model downloads — pure Pydantic round-trip + validation rules.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from on_premise.structuring.soap_schema import (
    LabMention,
    SOAPAssessment,
    SOAPNote,
    SOAPNoteMetadata,
    SOAPObjective,
    SOAPPlan,
    SOAPSubjective,
    Vital,
)

_SHA = "a" * 64


def _valid_note(**overrides) -> SOAPNote:  # type: ignore[no-untyped-def]
    metadata = SOAPNoteMetadata(
        source_transcript_sha256=_SHA,
        source_audio_sha256=_SHA,
        scenario_id="test-01",
        slm_backend="mlx-lm",
        slm_model="Bielik-11B-v3.0-Instruct-MLX-4bit",
        slm_device="mps",
        slm_structured_mode="grammar",
        generated_at=datetime.now().astimezone(),
        generation_duration_s=1.5,
    )
    base = {
        "subjective": SOAPSubjective(
            chief_complaint="ból głowy",
            history_of_present_illness="od dwóch dni.",
        ),
        "objective": SOAPObjective(),
        "assessment": SOAPAssessment(primary_diagnosis="napięciowy ból głowy"),
        "plan": SOAPPlan(),
        "metadata": metadata,
    }
    base.update(overrides)
    return SOAPNote(**base)


def test_round_trip_json() -> None:
    note = _valid_note()
    payload = note.model_dump_json()
    clone = SOAPNote.model_validate_json(payload)
    assert clone.subjective.chief_complaint == "ból głowy"
    assert clone.metadata.slm_backend == "mlx-lm"


def test_extra_forbid_rejects_unknown_fields() -> None:
    """Surplus keys from a hallucinated section must fail validation."""
    with pytest.raises(ValidationError):
        SOAPNote.model_validate(
            {
                **_valid_note().model_dump(mode="json"),
                "unexpected_key": "boom",
            }
        )


def test_icd10_accepts_canonical_forms() -> None:
    for code in ("I10", "I10.0", "I10.00", "A00.1"):
        SOAPAssessment(primary_diagnosis="dx", icd10_code=code)


def test_icd10_rejects_polish_gloss() -> None:
    with pytest.raises(ValidationError):
        SOAPAssessment(
            primary_diagnosis="dx",
            icd10_code="I10.0 (nadciśnienie tętnicze)",
        )


def test_icd10_rejects_reversed_format() -> None:
    with pytest.raises(ValidationError):
        SOAPAssessment(primary_diagnosis="dx", icd10_code="Nadciśnienie I10")


def test_icd10_allows_none() -> None:
    asm = SOAPAssessment(primary_diagnosis="dx", icd10_code=None)
    assert asm.icd10_code is None


def test_vital_whitelist_accepts_canonical_names() -> None:
    for name in ("bp", "hr", "rr", "temp", "spo2", "weight", "height", "bmi"):
        Vital(name=name, value="1", unit=None)  # type: ignore[arg-type]


def test_vital_rejects_non_whitelisted_name() -> None:
    """`cholesterol` must NOT be a vital — it goes to labs_mentioned."""
    with pytest.raises(ValidationError):
        Vital.model_validate({"name": "cholesterol", "value": "200", "unit": "mg/dL"})


def test_labmention_accepts_free_form_name() -> None:
    # Labs are intentionally not a whitelist.
    LabMention(name="TSH", value="3.1", unit="mIU/L")
    LabMention(name="echo serca", value=None, unit=None)


def test_provenance_defaults_empty() -> None:
    s = SOAPSubjective(chief_complaint="x", history_of_present_illness="y")
    assert s.provenance == []


def test_metadata_rejects_malformed_sha() -> None:
    with pytest.raises(ValidationError):
        SOAPNoteMetadata(
            source_transcript_sha256="short",
            source_audio_sha256=_SHA,
            scenario_id=None,
            slm_backend="mlx-lm",
            slm_model="x",
            slm_device="mps",
            slm_structured_mode="grammar",
            generated_at=datetime.now().astimezone(),
            generation_duration_s=0.0,
        )
