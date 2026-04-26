"""Polish prompt templates for the SOAP-structuring SLM.

Design posture:

* Polish throughout. Bielik-11B (v3 default, v2.3 tested) was
  instruction-tuned on Polish conversations; mixing English in the
  system prompt noticeably degrades compliance on clinical terminology.
* Anti-hedging framing: "strukturyzujesz istniejący wywiad, nie
  diagnozujesz". Both v2.3 (SFT+DPO) and v3.0 (DPO-Positive + GRPO)
  over-refuse on clinical prompts ("Jako model językowy nie mogę
  postawić diagnozy...") unless the task is explicitly reframed as
  reformatting existing dialogue.
* Provenance convention: transcript rendered as `[N] doctor: ...` so
  the model can cite segment indices in `provenance` lists. Matches
  the schema defaults.
* `render_correction_hint` mirrors Phase 1a's
  `_summarize_validation_error` — top 5 errors as
  `loc: msg` pairs, followed by "Popraw i zwróć pełny JSON".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import ValidationError

    from on_premise.stt.schema import Transcript


SYSTEM_PROMPT_PL = """\
Jesteś asystentem lekarza. Strukturyzujesz istniejący wywiad medyczny w \
format SOAP (Subjective / Objective / Assessment / Plan). Nie stawiasz \
nowych diagnoz ani nie doradzasz leczenia — Twoim zadaniem jest \
wyłącznie reformatowanie tego, co padło w transkrypcie, do struktury \
JSON zgodnej ze schematem.

Zasady:
1. Zwróć WYŁĄCZNIE pojedynczy obiekt JSON zgodny ze schematem SOAPNote. \
Bez markdown, bez komentarzy, bez tekstu przed/po JSON-ie.
2. Cytuj frazy pacjenta dosłownie, gdy zawierają dane identyfikujące \
(imię, nazwisko, PESEL, telefon, adres) — anonimizacja jest kolejnym \
etapem i potrzebuje oryginalnego tekstu.
3. Każde pole listowe może być pustą listą `[]` — nie wymyślaj treści, \
której nie ma w transkrypcie.
4. Kody ICD-10: format `A00` lub `A00.0` lub `A00.00` (litera + 2 cyfry \
+ opcjonalnie kropka i 1–2 cyfry). Bez polskich dopisków w nawiasie. \
Gdy niepewny — zwróć `null`.
5. Witalne (`vitals`): dozwolone nazwy to wyłącznie `bp`, `hr`, `rr`, \
`temp`, `spo2`, `weight`, `height`, `bmi`. Cokolwiek innego (np. \
cholesterol, glukoza) wrzuć do `objective.labs_mentioned`.
6. Pola `provenance` / `red_flags_provenance` to listy indeksów \
segmentów transkryptu (liczby w nawiasach kwadratowych `[N]`) z których \
pochodzi dana sekcja. Nieobowiązkowe — gdy niepewny zwróć `[]`.
7. `red_flags`: listuj objawy alarmowe (np. "ból zamostkowy z \
promieniowaniem", "duszność spoczynkowa", "utrata masy ciała > 10%"), \
które padły w transkrypcie. Nie dodawaj interpretacji klinicznej — \
tylko cytaty/parafrazy z dialogu.

Specyfika sekcji:
- `subjective.chief_complaint`: jedno zdanie — główna dolegliwość \
w słowach pacjenta.
- `subjective.history_of_present_illness`: narracja (1–4 zdania) — \
ewolucja dolegliwości.
- `objective.physical_exam`: obserwacje lekarza (osłuchiwanie, palpacja, \
oglądanie). Pomiary liczbowe idą do `vitals`.
- `assessment.primary_diagnosis`: rozpoznanie wiodące słownie \
(np. "nadciśnienie tętnicze pierwotne").
- `assessment.clinical_reasoning`: krótkie uzasadnienie (1–3 zdania) \
dlaczego to rozpoznanie.
- `plan.medications_prescribed`: leki + dawkowanie jednym stringiem \
(np. "Amlodypina 5 mg 1x dziennie").
"""


# One compact cardiology example — shortest Phase 1a scenario, truncated.
# Included only when the context window allows (~3000 tokens headroom).
FEW_SHOT_EXAMPLE = """\
Przykład (cardiology — skrócony):

Transkrypt:
[0] doctor: Dzień dobry, co Panią sprowadza?
[1] patient: Od tygodnia kołacze mi serce, głównie wieczorami.
[2] doctor: Jak często?
[3] patient: Może dwa–trzy razy dziennie, po parę minut.
[4] doctor: Ciśnienie Pani mierzyła?
[5] patient: Tak, wczoraj 150 na 95.
[6] doctor: Osłuchuję serce — rytm miarowy, bez szmerów. Tętno 88. \
Zrobimy EKG i Holter 24-godzinny. Proszę przyjmować Bisoprolol 2,5 mg \
rano i kontrola za dwa tygodnie.

JSON:
{
  "subjective": {
    "chief_complaint": "kołatanie serca od tygodnia",
    "history_of_present_illness": "Od tygodnia odczuwa kołatania serca, \
najczęściej wieczorami, 2–3 razy dziennie, trwające po kilka minut.",
    "review_of_systems": [],
    "medical_history": [],
    "medications_current": [],
    "allergies": [],
    "provenance": [1, 2, 3]
  },
  "objective": {
    "physical_exam": ["osłuchowo: rytm miarowy, bez szmerów"],
    "vitals": [
      {"name": "bp", "value": "150/95", "unit": "mmHg"},
      {"name": "hr", "value": "88", "unit": "bpm"}
    ],
    "labs_mentioned": [],
    "provenance": [5, 6]
  },
  "assessment": {
    "primary_diagnosis": "podejrzenie nadciśnienia tętniczego z napadowym \
kołataniem serca",
    "icd10_code": "I10",
    "differential": ["zaburzenia rytmu serca"],
    "clinical_reasoning": "Podwyższone ciśnienie w pomiarze domowym \
i kołatania przy miarowym rytmie w badaniu — wymaga potwierdzenia EKG \
i Holterem.",
    "provenance": [5, 6]
  },
  "plan": {
    "medications_prescribed": ["Bisoprolol 2,5 mg rano"],
    "diagnostic_tests": ["EKG", "Holter 24-godzinny"],
    "referrals": [],
    "follow_up": "kontrola za 2 tygodnie",
    "patient_education": [],
    "provenance": [6]
  },
  "red_flags": [],
  "red_flags_provenance": []
}
"""


def render_user_prompt(
    transcript: Transcript,
    *,
    include_few_shot: bool = True,
) -> str:
    """Serialize the transcript into the user turn for the SLM.

    Format: one line per segment, `[N] speaker: text`. `N` is the
    segment index — the SLM cites it in `provenance` lists. Empty
    transcripts are valid inputs; they just produce empty sections
    (which the schema allows).

    `include_few_shot=False` reserved for A/B eval or when the
    transcript is long enough that few-shot would blow past the 4K
    native context window.
    """
    segment_lines: list[str] = []
    for idx, seg in enumerate(transcript.segments):
        text = (seg.text or "").strip()
        if not text:
            continue
        segment_lines.append(f"[{idx}] {seg.speaker}: {text}")
    transcript_block = "\n".join(segment_lines) if segment_lines else "(transkrypt pusty)"

    parts: list[str] = []
    if include_few_shot:
        parts.append(FEW_SHOT_EXAMPLE)
    parts.append("Transkrypt wizyty do ustrukturyzowania:")
    parts.append(transcript_block)
    parts.append(
        "Zwróć pojedynczy obiekt JSON zgodny ze schematem SOAPNote (bez markdown, bez komentarzy):"
    )
    return "\n\n".join(parts)


def render_correction_hint(error: ValidationError) -> str:
    """Turn a Pydantic ValidationError into a Polish hint for the retry prompt.

    Mirrors `dataset_pipeline.scenario_generator.generator._summarize_validation_error`:
    take the first 5 error items, format as `path: message`, wrap in
    a "popraw i zwróć pełny JSON" instruction.
    """
    parts: list[str] = []
    for err in error.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "")
        parts.append(f"{loc}: {msg}" if loc else msg)
    joined = "; ".join(p for p in parts if p) or "walidacja nie powiodła się"
    return (
        "Poprzednia odpowiedź nie przeszła walidacji schematu: "
        f"{joined}. Popraw błędy i zwróć PEŁNY obiekt JSON zgodny ze "
        "schematem SOAPNote (bez markdown, bez komentarzy)."
    )
