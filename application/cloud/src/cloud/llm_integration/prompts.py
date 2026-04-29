"""Polish clinical prompts for the cloud agent's LLM nodes.

Two prompts:

* `SYSTEM_PROMPT_PL` — system framing. Anti-hedge (model is a clinical
  decision **support** tool, not a diagnostic agent), explicit empty-
  output rule for preliminary diagnoses (DD-2 in plan).
* `render_candidates_prompt(soap)` — user prompt for `propose_candidates`
  node. Serializes the `AnonymizedSOAPNote` as a clinical brief.

The compose-node prompt is built inline in `orchestrator._compose_suggestions`
since it interleaves NFZ + ChPL data per candidate.
"""

from __future__ import annotations

from shared.schemas import AnonymizedSOAPNote

# Anti-hedge framing — same posture as Phase 2b's SLM prompt. Bielik &
# Gemini both over-refuse on clinical inputs without it.
SYSTEM_PROMPT_PL = """\
Jesteś asystentem klinicznym CDSS (clinical decision support system) dla \
polskiego lekarza. Otrzymujesz zanonimizowaną notatkę SOAP i proponujesz \
listę leków jako WSPARCIE decyzji — finalną decyzję podejmuje lekarz.

ZASADY:
1. Proponuj wyłącznie leki refundowane przez NFZ jeśli to możliwe; jeśli \
   sugerujesz nierefundowany lek, jasno to zaznacz w `rationale`.
2. Każda propozycja MUSI zawierać niepustą listę `provenance` — ścieżki \
   do pól w SOAP, które uzasadniają wybór (np. \
   `["assessment.primary_diagnosis", "subjective.chief_complaint"]`).
3. Jeśli diagnoza jest wstępna (`podejrzenie X`, `wymaga potwierdzenia`, \
   `do diagnostyki`, `suspected`), zwróć PUSTĄ listę `candidates` i dodaj \
   wpis do `notes` typu "Diagnoza wymaga potwierdzenia — nie zaleca się \
   farmakoterapii pierwszego rzutu".
4. Używaj kanonicznego 7-znakowego kodu ATC (np. `C07AB07`, `B01AF02`).
5. Nie generuj danych pacjenta ani nazw własnych — wszystkie identyfikatory \
   PII zostały już zanonimizowane do tokenów typu `<PERSON_1>`.
6. Polski język. Zwięźle, klinicznie."""


def render_candidates_prompt(soap: AnonymizedSOAPNote) -> str:
    """Serialize an `AnonymizedSOAPNote` as a clinical brief for the LLM."""
    s = soap.subjective
    o = soap.objective
    a = soap.assessment
    p = soap.plan

    lines: list[str] = ["NOTATKA SOAP (zanonimizowana):", ""]
    lines.append("## S — Subjective")
    lines.append(f"Główna dolegliwość: {s.chief_complaint}")
    if s.history_of_present_illness:
        lines.append(f"Wywiad obecnej choroby: {s.history_of_present_illness}")
    if s.medical_history:
        lines.append(f"Choroby przewlekłe: {', '.join(s.medical_history)}")
    if s.medications_current:
        lines.append(f"Leki obecne: {', '.join(s.medications_current)}")
    if s.allergies:
        lines.append(f"Alergie: {', '.join(s.allergies)}")
    if s.review_of_systems:
        lines.append(f"Przegląd układów: {'; '.join(s.review_of_systems)}")

    lines.append("")
    lines.append("## O — Objective")
    if o.physical_exam:
        lines.append(f"Badanie fizykalne: {'; '.join(o.physical_exam)}")
    if o.vitals:
        vitals_str = ", ".join(
            f"{v.name}={v.value}{(' ' + v.unit) if v.unit else ''}"
            for v in o.vitals
        )
        lines.append(f"Parametry życiowe: {vitals_str}")
    if o.labs_mentioned:
        labs_str = ", ".join(
            f"{lab.name}{('=' + lab.value) if lab.value else ''}"
            f"{(' ' + lab.unit) if lab.unit else ''}"
            for lab in o.labs_mentioned
        )
        lines.append(f"Badania: {labs_str}")

    lines.append("")
    lines.append("## A — Assessment")
    lines.append(f"Diagnoza główna: {a.primary_diagnosis}")
    if a.icd10_code:
        lines.append(f"ICD-10: {a.icd10_code}")
    if a.differential:
        lines.append(f"Diagnostyka różnicowa: {', '.join(a.differential)}")
    if a.clinical_reasoning:
        lines.append(f"Rozumowanie kliniczne: {a.clinical_reasoning}")

    lines.append("")
    lines.append("## P — Plan (zaproponowany przez SLM)")
    if p.medications_prescribed:
        lines.append(f"Leki zaproponowane: {', '.join(p.medications_prescribed)}")
    if p.diagnostic_tests:
        lines.append(f"Badania zlecone: {', '.join(p.diagnostic_tests)}")
    if p.referrals:
        lines.append(f"Skierowania: {', '.join(p.referrals)}")
    if p.follow_up:
        lines.append(f"Kontrola: {p.follow_up}")

    if soap.red_flags:
        lines.append("")
        lines.append("## Czerwone flagi")
        for flag in soap.red_flags:
            lines.append(f"- {flag}")

    lines.append("")
    lines.append(
        "ZADANIE: zaproponuj listę leków (substancja czynna + ATC) jako "
        "wsparcie decyzji. Pamiętaj o zasadach systemu (provenance, ATC, "
        "empty-list dla diagnoz wstępnych)."
    )
    return "\n".join(lines)


def render_correction_hint(error: str) -> str:
    """Mirror Phase 1a `correction_hint` framing."""
    return (
        f"Poprzednia odpowiedź nie przeszła walidacji: {error}. "
        "Popraw strukturę i wygeneruj ponownie."
    )
