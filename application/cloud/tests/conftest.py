"""Shared fixtures for cloud tests.

Builds canned `AnonymizedSOAPNote` instances without disk IO so unit
tests can exercise prompts/orchestrator/api without anonymized samples.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID, uuid4

import pytest

from shared.schemas import (
    AnonymizedAssessment,
    AnonymizedMetadata,
    AnonymizedObjective,
    AnonymizedPlan,
    AnonymizedSOAPNote,
    AnonymizedSubjective,
    Specialization,
)


@pytest.fixture
def make_anonymized_soap():
    """Factory returning an `AnonymizedSOAPNote` with override-able fields.

    Default scenario mirrors `endo-gra` (Graves' disease): tachycardia +
    weight loss → propranolol candidate. Suitable for happy-path tests.
    """

    def _make(**overrides: Any) -> AnonymizedSOAPNote:
        defaults: dict[str, Any] = {
            "subjective": AnonymizedSubjective(
                chief_complaint="kołatanie serca, chudnięcie, nerwowość",
                history_of_present_illness=(
                    "Od kilku miesięcy tachykardia, utrata masy ciała, drżenie rąk."
                ),
                medical_history=[],
                medications_current=[],
                allergies=[],
            ),
            "objective": AnonymizedObjective(
                physical_exam=["tachykardia, drżenie rąk, wytrzeszcz oczu"],
                vitals=[],
                labs_mentioned=[],
            ),
            "assessment": AnonymizedAssessment(
                primary_diagnosis="nadczynność tarczycy (choroba Gravesa-Basedowa)",
                icd10_code="E05.0",
                differential=["zaburzenia lękowe"],
                clinical_reasoning="Typowe objawy nadczynności tarczycy.",
            ),
            "plan": AnonymizedPlan(
                medications_prescribed=["Propranolol"],
                diagnostic_tests=["TSH, fT3, fT4, anty-TRAB"],
                referrals=["endokrynologia"],
                follow_up="kontrola za tydzień",
            ),
            "red_flags": [],
            "metadata": AnonymizedMetadata(
                anonymizer_version="2c.0",
                n_entities_per_type={},
                visit_date=date(2026, 4, 26),
                specialization=Specialization.endokrynologia,
                session_id=UUID(int=0),
            ),
        }
        defaults.update(overrides)
        return AnonymizedSOAPNote(**defaults)

    return _make


@pytest.fixture
def session_id() -> UUID:
    return uuid4()
