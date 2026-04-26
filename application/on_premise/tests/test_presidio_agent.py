"""Tests for `on_premise.anonymization.presidio_agent`.

Unit tests use `_FakeRegistry` (canned `RecognizerResult` lists) to
avoid loading spaCy `pl_core_news_lg` — these are CI-friendly and run
in milliseconds. The tests focus on:

  - tree walk over the full SOAPNoteSubmission shape
  - per-session counter assignment (same original → same token)
  - splice/redaction at correct offsets
  - `Replacement` records + token_map for the audit logger
  - rebuild into a valid `AnonymizedSOAPNote`

Real-spaCy integration tests live in `test_pl_recognizers.py`'s
factory tests + Step 8's E2E corpus run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4

from presidio_analyzer import RecognizerResult

from on_premise.anonymization.presidio_agent import (
    AnonymizationResult,
    PresidioAnonymizer,
    _resolve_overlaps,
)
from on_premise.structuring.soap_schema import (
    SOAPAssessment,
    SOAPNoteSubmission,
    SOAPObjective,
    SOAPPlan,
    SOAPSubjective,
)

# ---------------------------------------------------------------------------
# _FakeRegistry — canned analyzer results per text.
# ---------------------------------------------------------------------------


@dataclass
class _FakeRegistry:
    """Returns a fixed list of `RecognizerResult` for matching texts.

    The map key is an exact-match text fragment; if the analyzed text
    starts with that fragment, we return the corresponding spans.
    Simple and adequate for unit-test coverage.
    """

    canned: dict[str, list[RecognizerResult]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def analyze(self, text: str, language: str = "pl") -> list[RecognizerResult]:
        self.calls.append(text)
        for trigger, results in self.canned.items():
            if trigger in text:
                # Adjust offsets if trigger is offset within text — keep it
                # simple by anchoring fixtures to whole-string matches.
                offset = text.index(trigger)
                return [
                    RecognizerResult(
                        entity_type=r.entity_type,
                        start=r.start + offset,
                        end=r.end + offset,
                        score=r.score,
                    )
                    for r in results
                ]
        return []


# ---------------------------------------------------------------------------
# _resolve_overlaps — pure function, easy to verify.
# ---------------------------------------------------------------------------


class TestResolveOverlaps:
    def test_no_overlaps_returns_unchanged(self) -> None:
        results = [
            RecognizerResult(entity_type="A", start=0, end=5, score=0.9),
            RecognizerResult(entity_type="B", start=10, end=15, score=0.9),
        ]
        out = _resolve_overlaps(results)
        assert len(out) == 2

    def test_contained_span_dropped(self) -> None:
        results = [
            RecognizerResult(entity_type="A", start=0, end=20, score=0.9),
            RecognizerResult(entity_type="B", start=5, end=10, score=0.85),  # inside A
        ]
        out = _resolve_overlaps(results)
        assert len(out) == 1
        assert out[0].entity_type == "A"

    def test_higher_score_wins_at_same_start(self) -> None:
        # When two spans tie on start, the higher-score one sorts first
        # and keeps the second from being included (since same range).
        results = [
            RecognizerResult(entity_type="A", start=0, end=10, score=0.5),
            RecognizerResult(entity_type="B", start=0, end=10, score=0.9),
        ]
        out = _resolve_overlaps(results)
        assert len(out) == 1
        assert out[0].entity_type == "B"

    def test_partial_overlap_keeps_both(self) -> None:
        # Spans that overlap but neither contains the other → both kept.
        # (This means downstream splicing needs to be careful — but we
        # don't expect this in real Presidio output.)
        results = [
            RecognizerResult(entity_type="A", start=0, end=5, score=0.9),
            RecognizerResult(entity_type="B", start=3, end=8, score=0.9),
        ]
        out = _resolve_overlaps(results)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# PresidioAnonymizer — full pipeline with _FakeRegistry.
# ---------------------------------------------------------------------------


def _valid_submission(**overrides: object) -> SOAPNoteSubmission:
    base = {
        "subjective": SOAPSubjective(
            chief_complaint="ból głowy",
            history_of_present_illness="Pacjent Krzysztof Nowakowski zgłosił ból od dwóch dni.",
        ),
        "objective": SOAPObjective(),
        "assessment": SOAPAssessment(primary_diagnosis="napięciowy ból głowy"),
        "plan": SOAPPlan(follow_up="Telefon kontaktowy 501-234-567."),
    }
    base.update(overrides)  # type: ignore[arg-type]
    return SOAPNoteSubmission(**base)  # type: ignore[arg-type]


class TestAnonymizeSoap:
    def setup_method(self) -> None:
        # Canned hits for the two real PII strings in our valid submission:
        # - "Krzysztof Nowakowski" at offset 8-28 inside the HPI text.
        # - "501-234-567" at offset 19-30 inside the follow-up text.
        # Trigger text is anchored to the leading substring; the
        # _FakeRegistry adjusts offsets relative to the input.
        self.registry = _FakeRegistry(
            canned={
                "Pacjent Krzysztof Nowakowski": [
                    RecognizerResult(entity_type="PERSON", start=8, end=28, score=1.0),
                ],
                "Telefon kontaktowy 501-234-567": [
                    RecognizerResult(entity_type="PHONE_NUMBER", start=19, end=30, score=0.9),
                ],
            }
        )
        self.anonymizer = PresidioAnonymizer(registry=self.registry)

    def _run(self, submission: SOAPNoteSubmission | None = None) -> AnonymizationResult:
        return self.anonymizer.anonymize_soap(
            submission or _valid_submission(),
            session_id=uuid4(),
            visit_date=date(2026, 4, 26),
            specialization="medycyna_rodzinna",
            anonymizer_version="2c.0",
        )

    def test_returns_valid_anonymized_note(self) -> None:
        result = self._run()
        # Pydantic validation passed — the redacted dict matches
        # AnonymizedSOAPNote shape.
        assert result.note.metadata.anonymizer_version == "2c.0"
        assert result.note.metadata.specialization == "medycyna_rodzinna"
        assert result.note.metadata.visit_date == date(2026, 4, 26)

    def test_redacts_person_with_token(self) -> None:
        result = self._run()
        hpi = result.note.subjective.history_of_present_illness
        # Original PII gone, replacement present.
        assert "Krzysztof Nowakowski" not in hpi
        assert "<PERSON_1>" in hpi
        # Surrounding text preserved — we don't damage the rest of the field.
        assert "ból od dwóch dni" in hpi

    def test_redacts_phone_with_token(self) -> None:
        result = self._run()
        follow_up = result.note.plan.follow_up
        assert follow_up is not None
        assert "501-234-567" not in follow_up
        assert "<PHONE_NUMBER_1>" in follow_up

    def test_replacements_recorded_for_audit(self) -> None:
        result = self._run()
        assert len(result.replacements) == 2
        types = {r.entity_type for r in result.replacements}
        assert types == {"PERSON", "PHONE_NUMBER"}
        # Each Replacement carries the SHA, not the plaintext.
        for r in result.replacements:
            assert re.match(r"^[0-9a-f]{64}$", r.original_sha256)
            assert r.token.startswith("<") and r.token.endswith(">")

    def test_token_map_is_populated(self) -> None:
        result = self._run()
        assert result.token_map["<PERSON_1>"] == "Krzysztof Nowakowski"
        assert result.token_map["<PHONE_NUMBER_1>"] == "501-234-567"

    def test_n_entities_per_type_aggregates(self) -> None:
        result = self._run()
        assert result.n_entities_per_type == {"PERSON": 1, "PHONE_NUMBER": 1}
        assert result.note.metadata.n_entities_per_type == {"PERSON": 1, "PHONE_NUMBER": 1}

    def test_same_original_reuses_same_token(self) -> None:
        """Per-session idempotency — the same name in two fields gets
        the same `<PERSON_1>` token, not `<PERSON_2>`.
        """
        # Both HPI and one of the follow-up texts mention the same name.
        registry = _FakeRegistry(
            canned={
                "Pacjent Krzysztof Nowakowski": [
                    RecognizerResult(entity_type="PERSON", start=8, end=28, score=1.0),
                ],
                "Pan Krzysztof Nowakowski znowu mówi": [
                    RecognizerResult(entity_type="PERSON", start=4, end=24, score=1.0),
                ],
            }
        )
        sub = _valid_submission(
            plan=SOAPPlan(follow_up="Pan Krzysztof Nowakowski znowu mówi o bólu."),
        )
        anonymizer = PresidioAnonymizer(registry=registry)
        result = anonymizer.anonymize_soap(
            sub,
            session_id=uuid4(),
            visit_date=date(2026, 4, 26),
            specialization="medycyna_rodzinna",
            anonymizer_version="2c.0",
        )
        # Both occurrences map to the same token.
        assert result.note.subjective.history_of_present_illness.count("<PERSON_1>") == 1
        assert (result.note.plan.follow_up or "").count("<PERSON_1>") == 1
        # Counter didn't reach 2 — only one unique original.
        assert result.n_entities_per_type == {"PERSON": 2}  # 2 occurrences, 1 unique
        assert "<PERSON_2>" not in result.note.subjective.history_of_present_illness
        assert "<PERSON_2>" not in (result.note.plan.follow_up or "")

    def test_empty_strings_are_skipped(self) -> None:
        """Empty / whitespace-only strings don't trigger an analyze call."""
        sub = _valid_submission(
            assessment=SOAPAssessment(primary_diagnosis="dx", clinical_reasoning=""),
        )
        result = self._run(sub)
        assert result.note.assessment.clinical_reasoning == ""
        # Confirm no analyze call was made for the empty string.
        # (analyze is called for the non-empty fields, but never with empty input.)
        assert "" not in self.registry.calls

    def test_walk_visits_list_items(self) -> None:
        """Tree walk recurses into list-of-strings fields."""
        registry = _FakeRegistry(
            canned={
                "Krzysztof Nowakowski jest alergikiem": [
                    RecognizerResult(entity_type="PERSON", start=0, end=20, score=1.0),
                ],
            }
        )
        sub = _valid_submission(
            subjective=SOAPSubjective(
                chief_complaint="x",
                history_of_present_illness="y",
                allergies=["Krzysztof Nowakowski jest alergikiem na orzechy"],
            ),
        )
        anonymizer = PresidioAnonymizer(registry=registry)
        result = anonymizer.anonymize_soap(
            sub,
            session_id=uuid4(),
            visit_date=date(2026, 4, 26),
            specialization="medycyna_rodzinna",
            anonymizer_version="2c.0",
        )
        assert "Krzysztof Nowakowski" not in result.note.subjective.allergies[0]
        assert "<PERSON_1>" in result.note.subjective.allergies[0]

    def test_red_flags_are_anonymized(self) -> None:
        """Red flags can contain names; anonymizer must walk in."""
        registry = _FakeRegistry(
            canned={
                "alergia u pana Nowakowskiego": [
                    RecognizerResult(entity_type="PERSON", start=14, end=28, score=1.0),
                ],
            }
        )
        sub = _valid_submission()
        sub.red_flags = ["alergia u pana Nowakowskiego, krytyczna"]
        anonymizer = PresidioAnonymizer(registry=registry)
        result = anonymizer.anonymize_soap(
            sub,
            session_id=uuid4(),
            visit_date=date(2026, 4, 26),
            specialization="medycyna_rodzinna",
            anonymizer_version="2c.0",
        )
        assert "Nowakowskiego" not in result.note.red_flags[0]
        assert "<PERSON_1>" in result.note.red_flags[0]

    def test_provenance_lists_pass_through_unchanged(self) -> None:
        """`provenance: list[int]` is not a string field — passes through."""
        sub = _valid_submission()
        sub.subjective.provenance = [1, 2, 3]
        result = self._run(sub)
        assert result.note.subjective.provenance == [1, 2, 3]


# ---------------------------------------------------------------------------
# Token grammar conformance — the grammar trust_gate enforces (Step 5)
# requires `<TYPE_NUMBER>` form. Confirm assigner only emits that.
# ---------------------------------------------------------------------------


def test_token_grammar_matches_trust_gate_expectation() -> None:
    """Every assigned token matches `<[A-Z_]+_\\d+>`."""
    registry = _FakeRegistry(
        canned={
            "Pacjent Krzysztof Nowakowski": [
                RecognizerResult(entity_type="PERSON", start=8, end=28, score=1.0),
            ],
        }
    )
    sub = _valid_submission()
    anonymizer = PresidioAnonymizer(registry=registry)
    result = anonymizer.anonymize_soap(
        sub,
        session_id=uuid4(),
        visit_date=date(2026, 4, 26),
        specialization="medycyna_rodzinna",
        anonymizer_version="2c.0",
    )
    grammar = re.compile(r"^<[A-Z_]+_\d+>$")
    for token in result.token_map:
        assert grammar.match(token), f"token violates grammar: {token!r}"
