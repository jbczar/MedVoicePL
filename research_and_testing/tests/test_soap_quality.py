"""Tests for `evaluation.slm_eval.soap_quality`."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from dataset_pipeline.scenario_generator.schema import (
    DialogueTurn,
    ScenarioModel,
    SpeakerRole,
    Specialization,
)
from evaluation.slm_eval.soap_quality import (
    SchemaAttempt,
    chief_complaint_recall,
    diagnosis_match,
    field_coverage,
    icd10_match,
    plan_nonempty,
    schema_compliance_rate,
)
from on_premise.structuring.soap_schema import (
    SOAPAssessment,
    SOAPNote,
    SOAPNoteMetadata,
    SOAPObjective,
    SOAPPlan,
    SOAPSubjective,
    Vital,
)

_SHA = "a" * 64


def _scenario(
    *,
    cc: str = "ból głowy",
    dx: str = "napięciowy ból głowy",
    icd: str | None = "R51",
) -> ScenarioModel:
    return ScenarioModel(
        id=str(uuid4()),
        specialization=Specialization.medycyna_rodzinna,
        patient_age=40,
        patient_sex="M",
        patient_full_name="Jan Kowalski",
        patient_pesel="85050512345",
        chief_complaint=cc,
        expected_diagnosis=dx,
        icd10_code=icd,
        turns=[
            DialogueTurn(speaker=SpeakerRole.doctor, text="Dzień dobry."),
            DialogueTurn(speaker=SpeakerRole.patient, text="Boli mnie głowa."),
            DialogueTurn(speaker=SpeakerRole.doctor, text="Od kiedy?"),
            DialogueTurn(speaker=SpeakerRole.patient, text="Od rana."),
            DialogueTurn(speaker=SpeakerRole.doctor, text="Proszę wziąć paracetamol."),
            DialogueTurn(speaker=SpeakerRole.patient, text="Dziękuję."),
        ],
        model="gemini-2.5-pro",
    )


def _soap(
    *,
    cc: str = "ból głowy",
    dx: str = "napięciowy ból głowy",
    icd: str | None = "R51",
    plan_med: list[str] | None = None,
    with_vitals: bool = False,
) -> SOAPNote:
    vitals = [Vital(name="bp", value="120/80", unit="mmHg")] if with_vitals else []
    return SOAPNote(
        subjective=SOAPSubjective(
            chief_complaint=cc,
            history_of_present_illness="od rana.",
        ),
        objective=SOAPObjective(vitals=vitals),
        assessment=SOAPAssessment(primary_diagnosis=dx, icd10_code=icd),
        plan=SOAPPlan(medications_prescribed=plan_med or []),
        metadata=SOAPNoteMetadata(
            source_transcript_sha256=_SHA,
            source_audio_sha256=_SHA,
            scenario_id="t",
            slm_backend="mlx-lm",
            slm_model="stub",
            slm_device="cpu",
            slm_structured_mode="grammar",
            generated_at=datetime.now().astimezone(),
            generation_duration_s=0.0,
        ),
    )


def test_schema_compliance_rate() -> None:
    attempts = [
        SchemaAttempt("a", True),
        SchemaAttempt("b", False),
        SchemaAttempt("c", True),
        SchemaAttempt("d", True),
    ]
    assert schema_compliance_rate(attempts) == 0.75
    assert schema_compliance_rate([]) == 0.0


def test_cc_recall_exact_match_is_1() -> None:
    soap = _soap(cc="ból głowy")
    scenario = _scenario(cc="ból głowy")
    assert chief_complaint_recall(soap, scenario) == 1.0


def test_cc_recall_casefold_and_punct_invariant() -> None:
    # Same tokens, different casing + diacritics + punctuation.
    soap = _soap(cc="BÓL GŁOWY.")
    scenario = _scenario(cc="ból głowy")
    assert chief_complaint_recall(soap, scenario) == 1.0


def test_cc_recall_partial_overlap() -> None:
    soap = _soap(cc="silny ból głowy od rana")
    scenario = _scenario(cc="ból głowy")
    r = chief_complaint_recall(soap, scenario)
    assert 0.0 < r < 1.0


def test_cc_recall_disjoint_is_0() -> None:
    soap = _soap(cc="duszność spoczynkowa")
    scenario = _scenario(cc="ból brzucha")
    assert chief_complaint_recall(soap, scenario) == 0.0


def test_diagnosis_match_substring_either_direction() -> None:
    # Predicted is superset of expected.
    assert diagnosis_match(
        _soap(dx="nadciśnienie tętnicze pierwotne"),
        _scenario(dx="nadciśnienie tętnicze"),
    )
    # Expected is superset of predicted.
    assert diagnosis_match(
        _soap(dx="nadciśnienie"),
        _scenario(dx="nadciśnienie tętnicze"),
    )
    # No overlap.
    assert not diagnosis_match(
        _soap(dx="cukrzyca"),
        _scenario(dx="nadciśnienie tętnicze"),
    )


def test_icd10_match_exact() -> None:
    assert icd10_match(_soap(icd="I10"), _scenario(icd="I10"))
    assert not icd10_match(_soap(icd="I10.0"), _scenario(icd="I10"))
    assert not icd10_match(_soap(icd=None), _scenario(icd="I10"))
    assert not icd10_match(_soap(icd="I10"), _scenario(icd=None))


def test_plan_nonempty_detects_any_populated_field() -> None:
    assert not plan_nonempty(_soap())  # all empty
    assert plan_nonempty(_soap(plan_med=["Paracetamol 500 mg"]))


def test_field_coverage_returns_bools_for_all_sections() -> None:
    cov = field_coverage(_soap(with_vitals=True, plan_med=["Paracetamol"]))
    assert cov["subjective.chief_complaint"] is True
    assert cov["objective.vitals"] is True
    assert cov["plan.medications_prescribed"] is True
    assert cov["objective.labs_mentioned"] is False
    assert cov["red_flags"] is False
    # Every key is a boolean.
    assert all(isinstance(v, bool) for v in cov.values())
