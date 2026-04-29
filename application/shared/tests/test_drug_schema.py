"""Schema tests for `shared.schemas.drug` and `shared.schemas.agent`.

Pure Pydantic round-trip + validation rules. No model downloads,
no on-premise / cloud imports.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AgentMetadata,
    AgentSuggestion,
    DiagnosticTest,
    DrugSuggestion,
    Referral,
    ReimbursementLevel,
)

# ---------------------------------------------------------------------------
# DrugSuggestion
# ---------------------------------------------------------------------------


def _valid_drug(**overrides: object) -> DrugSuggestion:
    base = {
        "name": "Propranolol",
        "atc_code": "C07AA05",
        "reimbursement": ReimbursementLevel.standard,
        "patient_copay_pln": Decimal("4.50"),
        "indication_restrictions": [],
        "provenance": ["assessment.primary_diagnosis"],
        "rationale": "Beta-blocker for symptomatic relief in Graves'.",
    }
    base.update(overrides)  # type: ignore[arg-type]
    return DrugSuggestion(**base)  # type: ignore[arg-type]


class TestATCCodePattern:
    """Canonical ATC: 1 letter + 2 digits + 2 letters + 2 digits."""

    @pytest.mark.parametrize(
        "code",
        [
            "A02BC01",  # Omeprazol
            "C07AB07",  # Bisoprolol
            "B01AF02",  # Apixaban
            "R03AC04",  # Salbutamol
        ],
    )
    def test_valid_codes_accepted(self, code: str) -> None:
        _valid_drug(atc_code=code)

    @pytest.mark.parametrize(
        "code",
        [
            "I10",          # ICD-10 shape, not ATC
            "C07AA",        # too short
            "C07AA055",     # too long
            "c07aa05",      # lowercase
            "C07AA0X",      # non-digit at end
            "707AA01",      # leading digit
        ],
    )
    def test_malformed_codes_rejected(self, code: str) -> None:
        with pytest.raises(ValidationError):
            _valid_drug(atc_code=code)


class TestProvenanceRequired:
    """DD-5 — provenance must be non-empty (every suggestion is sourced)."""

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_drug(provenance=[])

    def test_single_path_accepted(self) -> None:
        _valid_drug(provenance=["assessment.primary_diagnosis"])

    def test_multiple_paths_accepted(self) -> None:
        _valid_drug(
            provenance=[
                "assessment.primary_diagnosis",
                "subjective.chief_complaint",
                "subjective.history_of_present_illness",
            ],
        )


class TestExtraForbid:
    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DrugSuggestion.model_validate(
                {
                    **_valid_drug().model_dump(mode="json"),
                    "secret_field": "nope",
                }
            )


class TestPatientCopay:
    def test_decimal_round_trip(self) -> None:
        """Money is `Decimal`, not float — JSON round-trip preserves exact value."""
        drug = _valid_drug(patient_copay_pln=Decimal("3.20"))
        clone = DrugSuggestion.model_validate_json(drug.model_dump_json())
        assert clone.patient_copay_pln == Decimal("3.20")

    def test_none_accepted(self) -> None:
        drug = _valid_drug(patient_copay_pln=None)
        assert drug.patient_copay_pln is None


class TestReimbursementLevel:
    @pytest.mark.parametrize("level", list(ReimbursementLevel))
    def test_all_tiers_round_trip(self, level: ReimbursementLevel) -> None:
        drug = _valid_drug(reimbursement=level)
        clone = DrugSuggestion.model_validate_json(drug.model_dump_json())
        assert clone.reimbursement == level

    def test_unknown_tier_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _valid_drug(reimbursement="80%")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AgentSuggestion (top-level)
# ---------------------------------------------------------------------------


def _valid_metadata(**overrides: object) -> AgentMetadata:
    base = {
        "agent_version": "3a.0",
        "llm_model": "gemini-2.0-pro",
        "nfz_snapshot_date": date(2026, 4, 1),
        "chpl_corpus_size": 12,
        "duration_s": 3.45,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return AgentMetadata(**base)  # type: ignore[arg-type]


def _valid_agent_suggestion(**overrides: object) -> AgentSuggestion:
    base = {
        "drugs": [_valid_drug()],
        "referrals": [Referral(specialty="endokrynologia")],
        "tests": [DiagnosticTest(name="USG tarczycy")],
        "notes": [],
        "metadata": _valid_metadata(),
    }
    base.update(overrides)  # type: ignore[arg-type]
    return AgentSuggestion(**base)  # type: ignore[arg-type]


def test_round_trip_json() -> None:
    suggestion = _valid_agent_suggestion()
    clone = AgentSuggestion.model_validate_json(suggestion.model_dump_json())
    assert clone.drugs[0].name == "Propranolol"
    assert clone.referrals[0].specialty == "endokrynologia"
    assert clone.metadata.agent_version == "3a.0"


def test_empty_drugs_list_is_valid() -> None:
    """Half the corpus is diagnostic-only — empty drugs is a normal output."""
    suggestion = _valid_agent_suggestion(
        drugs=[],
        notes=["Diagnoza wymaga potwierdzenia — nie zaleca się farmakoterapii"],
    )
    assert len(suggestion.drugs) == 0
    assert "wymaga potwierdzenia" in suggestion.notes[0]


def test_extra_forbid_at_top_level() -> None:
    with pytest.raises(ValidationError):
        AgentSuggestion.model_validate(
            {
                **_valid_agent_suggestion().model_dump(mode="json"),
                "patient_full_name": "Jan Kowalski",  # PII would be a contract breach
            }
        )


def test_metadata_visit_date_is_date_not_datetime() -> None:
    """`nfz_snapshot_date` is `date`, not `datetime` — same posture as
    `AnonymizedMetadata.visit_date` (no time-of-day side channel)."""
    from datetime import datetime

    with pytest.raises(ValidationError):
        _valid_metadata(nfz_snapshot_date=datetime(2026, 4, 1, 14, 35))


def test_referral_minimal_construction() -> None:
    """`Referral.rationale` defaults to empty string."""
    r = Referral(specialty="kardiologia")
    assert r.rationale == ""


def test_diagnostic_test_minimal_construction() -> None:
    t = DiagnosticTest(name="EMG")
    assert t.urgency is None
    assert t.rationale == ""


def test_referral_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        Referral.model_validate({"specialty": "kardiologia", "patient_id": "X"})


def test_diagnostic_test_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        DiagnosticTest.model_validate({"name": "EMG", "patient_id": "X"})


# ---------------------------------------------------------------------------
# Re-export contract — gateway/cloud need these importable from shared.schemas
# ---------------------------------------------------------------------------


def test_all_phase_3a_types_importable_from_shared() -> None:
    """Positive contract — gateway must `from shared.schemas import X`."""
    import shared.schemas as ss

    for name in (
        "DrugSuggestion",
        "ReimbursementLevel",
        "AgentSuggestion",
        "AgentMetadata",
        "Referral",
        "DiagnosticTest",
    ):
        assert hasattr(ss, name), f"shared.schemas does not expose {name}"
