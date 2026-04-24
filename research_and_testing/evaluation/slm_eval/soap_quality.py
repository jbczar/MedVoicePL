"""SOAP note quality metrics — pure Python, dep-free.

Same posture as `stt_eval/wer_calculator.py` and `der_calculator.py`:
no torch, no LLMs, no similarity models. We measure what we can count,
and defer clinician-rubric SOAP grading to Phase 4 (on real audio).

Metrics:

* `schema_compliance_rate(attempts)` — fraction of extractions that
  validated against `SOAPNote` on the first try (before any correction
  retry). Driven by whether `lm-format-enforcer` is pulling its weight
  vs. raw decoding.
* `field_coverage(soap)` — which S/O/A/P buckets are populated.
  Deliberately boolean: "did the model write anything here" is a
  signal; whether the content is correct is a separate question.
* `chief_complaint_recall(soap, scenario)` — normalized Levenshtein
  similarity between the SOAP's CC and the scenario's ground-truth CC.
  Reuses the WER normalizer so punctuation + diacritics don't count.
* `diagnosis_match(soap, scenario)` — substring OR normalized exact
  match against `ScenarioModel.expected_diagnosis`. Substring is
  permissive on purpose: "nadciśnienie tętnicze pierwotne" should
  match "nadciśnienie tętnicze" without us writing a synonym table.
* `icd10_match(soap, scenario)` — exact-string match, case-sensitive.
  ICD-10 codes are uppercase letters, so casefold would be noise.
* `plan_nonempty(soap)` — at least one of
  `{medications_prescribed, diagnostic_tests, referrals, follow_up}`
  is populated. Sanity check without pretending to grade plan content.

No scoring of plan / assessment content — `ScenarioModel` carries only
CC + expected_diagnosis + icd10_code as ground truth, and synthesizing
reference SOAP via Gemini would couple eval to Gemini conventions.
Defer to Phase 4 rubric on real audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Reuse the WER normalizer so "Nadciśnienie tętnicze." matches
# "nadciśnienie tętnicze" without us rewriting punctuation-stripping.
from evaluation.stt_eval.wer_calculator import _levenshtein, normalize

if TYPE_CHECKING:
    from dataset_pipeline.scenario_generator.schema import ScenarioModel
    from on_premise.structuring.soap_schema import SOAPNote


@dataclass(frozen=True)
class SchemaAttempt:
    """One SOAP-extraction attempt: did it validate?"""

    scenario_id: str
    validated_first_try: bool


def schema_compliance_rate(attempts: list[SchemaAttempt]) -> float:
    """Fraction of attempts that validated on the first try."""
    if not attempts:
        return 0.0
    return sum(1 for a in attempts if a.validated_first_try) / len(attempts)


def field_coverage(soap: SOAPNote) -> dict[str, bool]:
    """Per-section populated-or-not map. Booleans so we can aggregate easily."""
    s = soap.subjective
    o = soap.objective
    a = soap.assessment
    p = soap.plan
    return {
        "subjective.chief_complaint": bool(s.chief_complaint.strip()),
        "subjective.history_of_present_illness": bool(
            s.history_of_present_illness.strip()
        ),
        "subjective.review_of_systems": bool(s.review_of_systems),
        "subjective.medical_history": bool(s.medical_history),
        "subjective.medications_current": bool(s.medications_current),
        "subjective.allergies": bool(s.allergies),
        "objective.physical_exam": bool(o.physical_exam),
        "objective.vitals": bool(o.vitals),
        "objective.labs_mentioned": bool(o.labs_mentioned),
        "assessment.primary_diagnosis": bool(a.primary_diagnosis.strip()),
        "assessment.icd10_code": bool(a.icd10_code),
        "assessment.differential": bool(a.differential),
        "assessment.clinical_reasoning": bool(a.clinical_reasoning.strip()),
        "plan.medications_prescribed": bool(p.medications_prescribed),
        "plan.diagnostic_tests": bool(p.diagnostic_tests),
        "plan.referrals": bool(p.referrals),
        "plan.follow_up": bool(p.follow_up),
        "plan.patient_education": bool(p.patient_education),
        "red_flags": bool(soap.red_flags),
    }


def _normalized_similarity(a: str, b: str) -> float:
    """1 - normalized Levenshtein distance on casefold+punct-stripped tokens."""
    ra = normalize(a).split()
    rb = normalize(b).split()
    if not ra and not rb:
        return 1.0
    if not ra or not rb:
        return 0.0
    dist = _levenshtein(ra, rb)
    return float(1.0 - dist / max(len(ra), len(rb)))


def chief_complaint_recall(soap: SOAPNote, scenario: ScenarioModel) -> float:
    """Normalized token-level similarity between SOAP CC and scenario CC.

    Returns a float in [0, 1]. 1.0 means exact match after normalization;
    0.0 means completely disjoint. Both fields are short (one-liner), so
    token-level Levenshtein behaves well.
    """
    return _normalized_similarity(soap.subjective.chief_complaint, scenario.chief_complaint)


def diagnosis_match(soap: SOAPNote, scenario: ScenarioModel) -> bool:
    """Substring OR normalized exact match against scenario.expected_diagnosis."""
    predicted = normalize(soap.assessment.primary_diagnosis)
    expected = normalize(scenario.expected_diagnosis)
    if not predicted or not expected:
        return False
    # Either direction: the model might say "nadciśnienie tętnicze
    # pierwotne" (expected "nadciśnienie tętnicze") OR say
    # "nadciśnienie" (expected "nadciśnienie tętnicze"). Both count.
    return predicted in expected or expected in predicted


def icd10_match(soap: SOAPNote, scenario: ScenarioModel) -> bool:
    """Exact string match on ICD-10 code (case-sensitive)."""
    predicted = (soap.assessment.icd10_code or "").strip()
    expected = (scenario.icd10_code or "").strip()
    if not predicted or not expected:
        return False
    return predicted == expected


def plan_nonempty(soap: SOAPNote) -> bool:
    """Sanity check: did the SLM write *something* in the Plan section?"""
    p = soap.plan
    return bool(
        p.medications_prescribed
        or p.diagnostic_tests
        or p.referrals
        or (p.follow_up and p.follow_up.strip())
        or p.patient_education
    )
