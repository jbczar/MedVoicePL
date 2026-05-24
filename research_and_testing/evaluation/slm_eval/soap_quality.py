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
* `diagnosis_match(soap, scenario)` — substring fast-path against
  `ScenarioModel.expected_diagnosis`, falling back to Polish-tolerant
  stem-set containment when the substring check fails. Polish
  inflection (`przewlekła` → `przewlekłej`, `choroba` → `choroby`)
  defeats raw substring on real model output even when the model
  emitted the right diagnosis. The prefix-stemming fallback (first
  5 chars of each content token) closes the gap without needing a
  Polish lemmatizer dependency.
* `icd10_match(soap, scenario)` — equality OR boundary-aware prefix
  match. The model is not penalized for under-specifying (`J44`
  matches scenario `J44.9`) or over-specifying (`I10.0` matches
  scenario `I10`) — both are clinically the same family. Match only
  at code-segment boundaries: `J44` matches `J44.9` but not `J445`.
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


def _content_stems(normalized: str) -> set[str]:
    """Polish-tolerant stems for set-containment matching.

    Takes a `normalize()`-d string and returns the set of first-5-char
    prefixes of every token of length ≥3. Drops short connectives
    (`od`, `u`, `i`, `do`) so they can't accidentally match. Five
    chars is enough to discriminate distinct medical terms while
    collapsing Polish inflectional suffixes:

        "przewlekła"  -> "przew"
        "przewlekłej" -> "przew"   # same stem
        "choroba"     -> "choro"
        "choroby"     -> "choro"   # same stem
        "POChP"       -> "pochp"   # acronym preserved
    """
    return {tok[:5] for tok in normalized.split() if len(tok) >= 3}


def diagnosis_match(soap: SOAPNote, scenario: ScenarioModel) -> bool:
    """Substring fast-path then Polish-tolerant stem-set containment.

    Two layers:

    1. **Substring on normalized text**: handles "nadciśnienie tętnicze
       pierwotne" ⊇ "nadciśnienie tętnicze" and the reverse. Cheap and
       sufficient when the model uses the same word forms as the
       scenario.

    2. **Stem-set containment**: when substring fails, compare sets of
       5-char content-token prefixes. "Przewlekła obturacyjna choroba
       płuc (POChP)" and "podejrzenie przewlekłej obturacyjnej choroby
       płuc (POChP) u wieloletniego palacza" have identical stems for
       the diagnostic content tokens; either side's stems being a
       subset of the other's counts as a match. This is what catches
       Polish inflection without pulling in a lemmatizer dep.
    """
    predicted = normalize(soap.assessment.primary_diagnosis)
    expected = normalize(scenario.expected_diagnosis)
    if not predicted or not expected:
        return False
    if predicted in expected or expected in predicted:
        return True
    pred_stems = _content_stems(predicted)
    exp_stems = _content_stems(expected)
    if not pred_stems or not exp_stems:
        return False
    return exp_stems.issubset(pred_stems) or pred_stems.issubset(exp_stems)


def diagnosis_match_semantic(
    soap: SOAPNote,
    scenario: ScenarioModel,
    *,
    embedder: object,
    threshold: float = 0.85,
) -> tuple[bool, float]:
    """Semantic similarity between scenario.expected_diagnosis and
    soap.assessment.primary_diagnosis via mmlw-roberta-large embeddings.

    Returns (match_bool, cosine_similarity_float).

    Why this complements `diagnosis_match`:

    - `diagnosis_match` is lexical (substring + 5-char stems). Misses
      true paraphrases like "POChP zaostrzenie" ↔ "Zaostrzenie
      przewlekłej obturacyjnej choroby płuc". These get stem-set hits
      sometimes but not always.

    - `diagnosis_match_semantic` is dense semantic — uses the same
      `sdadas/mmlw-roberta-large` (1024-dim Polish embedder) we already
      load for the ChPL RAG. Polish-tuned, 0 USD cost, ~10 ms per pair.

    Threshold 0.85 is a clinically-defensible cutoff: paraphrases with
    same medical content score ≥0.90, distinct conditions (e.g.
    "POChP" vs "astma") score 0.60-0.75. Tuned empirically; consider
    sweep in writeup if needed.
    """
    predicted = (soap.assessment.primary_diagnosis or "").strip()
    expected = (scenario.expected_diagnosis or "").strip()
    if not predicted or not expected:
        return False, 0.0
    # Embedder must expose `embed_query(str) -> list[float]`.
    v1 = embedder.embed_query(expected)
    v2 = embedder.embed_query(predicted)
    # Cosine sim (vectors already L2-normalized by mmlw default).
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    return dot >= threshold, float(dot)


def icd10_match(soap: SOAPNote, scenario: ScenarioModel) -> bool:
    """Equality OR boundary-aware prefix match.

    `J44` matches scenario `J44.9` (model under-specified — clinically
    the same family). `I10.0` matches scenario `I10` (model
    over-specified). `J44` does NOT match `J45` and does NOT match
    `J445` — prefix only at the `.` boundary, never on raw character
    overlap. Codes are case-sensitive (uppercase by ICD-10 convention),
    so we don't casefold.
    """
    predicted = (soap.assessment.icd10_code or "").strip()
    expected = (scenario.icd10_code or "").strip()
    if not predicted or not expected:
        return False
    if predicted == expected:
        return True
    longer, shorter = (
        (predicted, expected) if len(predicted) > len(expected) else (expected, predicted)
    )
    return longer.startswith(shorter + ".")


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
