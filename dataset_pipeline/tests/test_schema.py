"""Unit tests for ScenarioModel schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataset_pipeline.scenario_generator.schema import (
    DialogueTurn,
    ScenarioModel,
    SpeakerRole,
    Specialization,
)


def _valid_turns() -> list[DialogueTurn]:
    roles = [SpeakerRole.doctor, SpeakerRole.patient] * 3
    return [DialogueTurn(speaker=r, text=f"kwestia {i}") for i, r in enumerate(roles)]


def _valid_kwargs() -> dict[str, object]:
    return dict(
        specialization=Specialization.kardiologia,
        patient_age=55,
        patient_sex="M",
        patient_full_name="Jan Kowalski",
        patient_pesel="62050512345",
        chief_complaint="Ból w klatce",
        expected_diagnosis="Dławica",
        icd10_code="I20.9",
        turns=_valid_turns(),
        model="gemini-test",
    )


def test_valid_scenario_roundtrip() -> None:
    scenario = ScenarioModel(**_valid_kwargs())  # type: ignore[arg-type]
    payload = scenario.model_dump_json()
    restored = ScenarioModel.model_validate_json(payload)
    assert restored.specialization is Specialization.kardiologia
    assert restored.patient_sex == "M"
    assert len(restored.turns) == 6


def test_first_turn_must_be_doctor() -> None:
    kwargs = _valid_kwargs()
    kwargs["turns"] = [
        DialogueTurn(speaker=SpeakerRole.patient, text="a"),
        DialogueTurn(speaker=SpeakerRole.doctor, text="b"),
        DialogueTurn(speaker=SpeakerRole.patient, text="c"),
        DialogueTurn(speaker=SpeakerRole.doctor, text="d"),
        DialogueTurn(speaker=SpeakerRole.patient, text="e"),
        DialogueTurn(speaker=SpeakerRole.doctor, text="f"),
    ]
    with pytest.raises(ValidationError, match="first turn"):
        ScenarioModel(**kwargs)  # type: ignore[arg-type]


def test_consecutive_same_speaker_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["turns"] = [
        DialogueTurn(speaker=SpeakerRole.doctor, text="a"),
        DialogueTurn(speaker=SpeakerRole.doctor, text="b"),  # bad
        DialogueTurn(speaker=SpeakerRole.patient, text="c"),
        DialogueTurn(speaker=SpeakerRole.doctor, text="d"),
        DialogueTurn(speaker=SpeakerRole.patient, text="e"),
        DialogueTurn(speaker=SpeakerRole.doctor, text="f"),
    ]
    with pytest.raises(ValidationError, match="consecutive"):
        ScenarioModel(**kwargs)  # type: ignore[arg-type]


def test_too_few_turns_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["turns"] = _valid_turns()[:4]
    with pytest.raises(ValidationError):
        ScenarioModel(**kwargs)  # type: ignore[arg-type]


def test_age_out_of_range_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["patient_age"] = 15
    with pytest.raises(ValidationError):
        ScenarioModel(**kwargs)  # type: ignore[arg-type]


def test_pesel_pattern_enforced() -> None:
    kwargs = _valid_kwargs()
    kwargs["patient_pesel"] = "12345"
    with pytest.raises(ValidationError):
        ScenarioModel(**kwargs)  # type: ignore[arg-type]


def test_empty_turn_text_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        DialogueTurn(speaker=SpeakerRole.doctor, text="")
