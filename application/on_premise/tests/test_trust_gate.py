"""Tests for `on_premise.trust_gate`.

Three-check coverage:

* Type rejection: an un-anonymized SOAP raises `kind="type"`.
* Regex catch: anonymized payload with PESEL-shape / phone-shape /
  NIP-shape / email-shape in any string leaf raises `kind="regex"`.
* Charset catch: stray quoted plaintext name (e.g. `"Jan Kowalski"`)
  raises `kind="charset"` because Polish letters are allowed but
  unmatched quotes / curly chars / arbitrary punctuation aren't.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

import pytest

from on_premise.structuring.soap_schema import (
    SOAPAssessment,
    SOAPNote,
    SOAPNoteMetadata,
    SOAPObjective,
    SOAPPlan,
    SOAPSubjective,
)
from on_premise.trust_gate import (
    TrustBoundaryViolation,
    validate_outbound,
    violations_to_dicts,
)
from shared.schemas import (
    AnonymizedAssessment,
    AnonymizedMetadata,
    AnonymizedObjective,
    AnonymizedPlan,
    AnonymizedSOAPNote,
    AnonymizedSubjective,
    Specialization,
)


def _valid_anonymized_note(**field_overrides: object) -> AnonymizedSOAPNote:
    """Construct a minimal valid AnonymizedSOAPNote with placeholder
    tokens — passes all three checks by default.
    """
    base = {
        "subjective": AnonymizedSubjective(
            chief_complaint="Pacjent <PERSON_1> zgłasza ból głowy.",
            history_of_present_illness=("Bóle od dwóch dni, kontakt: <PHONE_NUMBER_1>."),
        ),
        "objective": AnonymizedObjective(),
        "assessment": AnonymizedAssessment(primary_diagnosis="napięciowy ból głowy"),
        "plan": AnonymizedPlan(),
        "metadata": AnonymizedMetadata(
            anonymizer_version="2c.0",
            n_entities_per_type={"PERSON": 1, "PHONE_NUMBER": 1},
            visit_date=date(2026, 4, 26),
            specialization=Specialization.medycyna_rodzinna,
            session_id=uuid4(),
        ),
    }
    base.update(field_overrides)  # type: ignore[arg-type]
    return AnonymizedSOAPNote(**base)  # type: ignore[arg-type]


def _un_anonymized_note() -> SOAPNote:
    """A real (PII-bearing) SOAPNote for the type-rejection test."""
    sha = "a" * 64
    return SOAPNote(
        subjective=SOAPSubjective(
            chief_complaint="ból głowy",
            history_of_present_illness="dwa dni",
        ),
        objective=SOAPObjective(),
        assessment=SOAPAssessment(primary_diagnosis="dx"),
        plan=SOAPPlan(),
        metadata=SOAPNoteMetadata(
            source_transcript_sha256=sha,
            source_audio_sha256=sha,
            scenario_id=None,
            slm_backend="mlx-lm",
            slm_model="x",
            slm_device="mps",
            slm_structured_mode="grammar",
            generated_at=datetime.now().astimezone(),
            generation_duration_s=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_anonymized_payload_passes() -> None:
    """A clean anonymized payload returns None (no violations)."""
    note = _valid_anonymized_note()
    assert validate_outbound(note) is None


def test_polish_diacritics_pass() -> None:
    """Polish letters (ę, ą, ł, ć, etc.) are in the allowed charset."""
    note = _valid_anonymized_note(
        subjective=AnonymizedSubjective(
            chief_complaint="Bóle głowy, szczękościsk i mdłości.",
            history_of_present_illness="Występują od dwóch dni.",
        ),
    )
    assert validate_outbound(note) is None


def test_medical_typography_pass() -> None:
    """Bielik SLM emits real Polish medical text — % ° ± — – „" α β …

    Regression for bug discovered 2026-05-11: trust_gate charset
    whitelist was too narrow, rejected legitimate medical typography
    like "Nasycenie tlenem 96%", "37,5°C", "BP 120 ± 10 mmHg",
    "β-bloker", em-dash, en-dash, Polish typographic quotes.
    """
    note = _valid_anonymized_note(
        assessment=AnonymizedAssessment(
            primary_diagnosis="Nadczynność tarczycy",
            clinical_reasoning=(
                "Pacjent z saturacją 96%, temperaturą 37,5°C i BP 120 ± 10 mmHg. "
                "Wskazany β-bloker — np. Bisoprolol 5 mg/dobę. "
                "Czas leczenia: 3–6 miesięcy. "
                "Limit ≤80 bpm tętna. "
                'Według wytycznych „PTE 2024", konsultacja endokrynologa…'
            ),
        ),
    )
    assert validate_outbound(note) is None


def test_empty_strings_skipped() -> None:
    """Empty string leaves don't trip the gate."""
    note = _valid_anonymized_note(
        assessment=AnonymizedAssessment(
            primary_diagnosis="dx",
            clinical_reasoning="",  # explicit empty
        ),
    )
    assert validate_outbound(note) is None


def test_multiple_placeholders_in_one_field() -> None:
    """Multiple `<TOKEN_n>` in a single string all pass charset check."""
    note = _valid_anonymized_note(
        subjective=AnonymizedSubjective(
            chief_complaint="<PERSON_1> zgłosił, <PERSON_2> potwierdził.",
            history_of_present_illness="Kontakt <PHONE_NUMBER_1>.",
        ),
    )
    assert validate_outbound(note) is None


# ---------------------------------------------------------------------------
# 1. Type contract
# ---------------------------------------------------------------------------


def test_unanonymized_soap_rejected_by_type_check() -> None:
    """Passing a `SOAPNote` (not anonymized) trips the type check."""
    note = _un_anonymized_note()
    with pytest.raises(TrustBoundaryViolation) as exc:
        validate_outbound(note)
    violations = exc.value.violations
    assert len(violations) == 1
    assert violations[0].kind == "type"
    assert "SOAPNote" in violations[0].detail


def test_string_input_rejected_by_type_check() -> None:
    """Even a string sneak-attack trips the type check."""
    with pytest.raises(TrustBoundaryViolation) as exc:
        validate_outbound("just a string")
    assert exc.value.violations[0].kind == "type"


def test_dict_input_rejected_by_type_check() -> None:
    """A dict-shaped-like-payload still trips — must be the actual type."""
    with pytest.raises(TrustBoundaryViolation):
        validate_outbound({"subjective": {}, "metadata": {}})


# ---------------------------------------------------------------------------
# 2. Regex pre-filter
# ---------------------------------------------------------------------------


class TestRegexCatch:
    """Each PII-shape regex independently trips the gate."""

    def test_pesel_in_chief_complaint(self) -> None:
        note = _valid_anonymized_note(
            subjective=AnonymizedSubjective(
                chief_complaint="Pacjent z PESEL 75081512345 zgłosił.",
                history_of_present_illness="ok",
            ),
        )
        with pytest.raises(TrustBoundaryViolation) as exc:
            validate_outbound(note)
        kinds = {v.detail for v in exc.value.violations if v.kind == "regex"}
        assert "pesel" in kinds

    def test_phone_in_clinical_reasoning(self) -> None:
        note = _valid_anonymized_note(
            assessment=AnonymizedAssessment(
                primary_diagnosis="dx",
                clinical_reasoning="Numer +48 501 234 567",
            ),
        )
        with pytest.raises(TrustBoundaryViolation) as exc:
            validate_outbound(note)
        kinds = {v.detail for v in exc.value.violations if v.kind == "regex"}
        assert "phone" in kinds

    def test_nip_in_red_flags(self) -> None:
        note = _valid_anonymized_note()
        note.red_flags = ["NIP 5260250995 hospitalizacja"]
        with pytest.raises(TrustBoundaryViolation) as exc:
            validate_outbound(note)
        kinds = {v.detail for v in exc.value.violations if v.kind == "regex"}
        assert "nip" in kinds

    def test_email_in_follow_up(self) -> None:
        note = _valid_anonymized_note(
            plan=AnonymizedPlan(follow_up="Email: doctor@clinic.pl"),
        )
        with pytest.raises(TrustBoundaryViolation) as exc:
            validate_outbound(note)
        kinds = {v.detail for v in exc.value.violations if v.kind == "regex"}
        assert "email" in kinds

    def test_violation_path_points_to_offending_field(self) -> None:
        note = _valid_anonymized_note(
            subjective=AnonymizedSubjective(
                chief_complaint="ok",
                history_of_present_illness="PESEL: 75081512345",
            ),
        )
        with pytest.raises(TrustBoundaryViolation) as exc:
            validate_outbound(note)
        # Path includes the offending field for debuggability.
        paths = [v.location for v in exc.value.violations]
        assert any("history_of_present_illness" in p for p in paths)


# ---------------------------------------------------------------------------
# 3. Charset invariant — allow-list of characters
# ---------------------------------------------------------------------------


class TestCharsetCatch:
    def test_curly_quotes_accepted(self) -> None:
        """Smart quotes „..." are legitimate Polish typography (Pi/Pf
        Unicode categories) and are ACCEPTED by the v3 charset policy.

        Updated 2026-05-11 after switching from whitelist regex to
        Unicode-category blocklist. Bielik SLM normally emits Polish
        curly quotes — rejecting them caused live-mode failures.
        """
        note = _valid_anonymized_note(
            subjective=AnonymizedSubjective(
                chief_complaint='Pacjent powiedział „boli mnie głowa”.',
                history_of_present_illness="ok",
            ),
        )
        assert validate_outbound(note) is None

    def test_emoji_rejected(self) -> None:
        """Emoji aren't anonymizer output — disallowed character."""
        note = _valid_anonymized_note(
            subjective=AnonymizedSubjective(
                chief_complaint="Ból głowy 😀",
                history_of_present_illness="ok",
            ),
        )
        with pytest.raises(TrustBoundaryViolation) as exc:
            validate_outbound(note)
        assert any(v.kind == "charset" for v in exc.value.violations)


# ---------------------------------------------------------------------------
# violations_to_dicts adapter for audit logger
# ---------------------------------------------------------------------------


def test_violations_to_dicts_shape() -> None:
    """Adapter produces the dict shape `audit_logger` accepts."""
    note = _un_anonymized_note()
    with pytest.raises(TrustBoundaryViolation) as exc:
        validate_outbound(note)
    dicts = violations_to_dicts(exc.value.violations)
    assert all(set(d.keys()) == {"location", "kind", "detail"} for d in dicts)
    assert all(isinstance(v, str) for d in dicts for v in d.values())
