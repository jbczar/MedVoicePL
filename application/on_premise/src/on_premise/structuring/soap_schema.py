"""Typed contract for un-anonymized SOAP notes produced by the local SLM.

**TRUST BOUNDARY NOTICE**
This module is importable ONLY from within `application/on_premise/`. The
`SOAPNote` hierarchy describes *un-anonymized* clinical content — it can
quote PESEL numbers, Polish phone numbers, surnames, addresses verbatim
from the source `Transcript`. Do NOT re-export any symbol from this module
via `application/shared/schemas/`. Phase 2c owns that slot and will fill
`shared/schemas/soap.py` with `AnonymizedSOAPNote` — a *different* type,
produced by running this output through Presidio + trust_gate.

Schema design notes:

* `extra="forbid"` on every model — the SLM must produce exactly these
  fields; surplus keys from hallucinated sections raise ValidationError
  and trigger a correction-hint retry in `pipeline.transcript_to_soap`.
* Vitals are a whitelist (`VITAL_NAMES`). Bielik loves inventing
  `"cholesterol": "wysoki"` as a vital sign; forcing the whitelist pushes
  those into `SOAPObjective.labs_mentioned` where they belong.
* `icd10_code` has a regex — `"I10"`, `"I10.0"`, `"I10.00"` accepted;
  `"I10.0 (nadciśnienie tętnicze)"` rejected. Bielik routinely appends
  a Polish gloss which the regex catches before it corrupts downstream
  NFZ lookups.
* Every section carries an optional `provenance: list[int]` — segment
  indices from the source `Transcript.segments` the SLM drew from. Phase
  2c's audit logger uses this to answer "which turn was red_flag X
  extracted from". Defaults to `[]` (opt-in — the few-shot shows the
  model how to populate it, but a valid SOAPNote without provenance
  is still accepted).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Whitelist of recognized vital-sign names. Anything else goes into
# `SOAPObjective.labs_mentioned` (a free-form bag) instead. Adding a new
# vital requires a code change + schema redeploy on purpose — the value
# of this field is structure, and widening it silently defeats that.
VITAL_NAMES = Literal[
    "bp",  # blood pressure, e.g. "140/90" mmHg
    "hr",  # heart rate, e.g. "72" bpm
    "rr",  # respiratory rate, e.g. "16" /min
    "temp",  # temperature, e.g. "36.8" °C
    "spo2",  # pulse oximetry, e.g. "98" %
    "weight",  # kg
    "height",  # cm
    "bmi",  # kg/m^2
]


class Vital(BaseModel):
    """One vital-sign reading from the dialogue."""

    model_config = ConfigDict(extra="forbid")

    name: VITAL_NAMES
    value: str  # "140/90", "72", "36.8" — string to preserve original spelling
    unit: str | None = None  # "mmHg", "bpm", "°C" — omitted when obvious from `name`


class LabMention(BaseModel):
    """A lab / imaging result or request surfaced in the dialogue.

    Free-form `name` (not a whitelist) because Polish lab terminology is
    too wide to enumerate: `"TSH"`, `"glukoza na czczo"`, `"echo serca"`,
    `"RTG klatki piersiowej"`. Downstream we only need the mention, not
    the ontology.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | None = None  # "180", "podwyższone", None when only requested
    unit: str | None = None  # "mg/dL", None for qualitative


class SOAPSubjective(BaseModel):
    """S — Subjective. What the patient reports."""

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str  # one-sentence lede
    history_of_present_illness: str  # narrative paragraph
    review_of_systems: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    medications_current: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    provenance: list[int] = Field(default_factory=list)


class SOAPObjective(BaseModel):
    """O — Objective. What the doctor observes / measures."""

    model_config = ConfigDict(extra="forbid")

    physical_exam: list[str] = Field(default_factory=list)
    vitals: list[Vital] = Field(default_factory=list)
    labs_mentioned: list[LabMention] = Field(default_factory=list)
    provenance: list[int] = Field(default_factory=list)


class SOAPAssessment(BaseModel):
    """A — Assessment. Working diagnosis + reasoning."""

    model_config = ConfigDict(extra="forbid")

    primary_diagnosis: str
    # ICD-10: letter + 2 digits, optional `.dd`. "I10", "I10.0", "I10.00" pass;
    # "I10.0 (nadciśnienie tętnicze)" and "Nadciśnienie I10" fail — Bielik
    # loves to append a Polish gloss that would corrupt NFZ lookups.
    icd10_code: str | None = Field(
        default=None,
        pattern=r"^[A-Z]\d{2}(\.\d{1,2})?$",
    )
    differential: list[str] = Field(default_factory=list)
    clinical_reasoning: str = ""
    provenance: list[int] = Field(default_factory=list)


class SOAPPlan(BaseModel):
    """P — Plan. What happens next."""

    model_config = ConfigDict(extra="forbid")

    medications_prescribed: list[str] = Field(default_factory=list)
    diagnostic_tests: list[str] = Field(default_factory=list)
    referrals: list[str] = Field(default_factory=list)
    follow_up: str | None = None
    patient_education: list[str] = Field(default_factory=list)
    provenance: list[int] = Field(default_factory=list)


class SOAPNoteMetadata(BaseModel):
    """Provenance of this SOAP note — who produced it, from what, when."""

    model_config = ConfigDict(extra="forbid")

    # sha256(transcript.model_dump_json(sort_keys=True)) — anchors this
    # note to the exact transcript bytes. Computed in 2b pipeline so
    # Phase 2a's `Transcript` stays frozen (no backport of a new field).
    source_transcript_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # Copied from `Transcript.audio_sha256` so downstream consumers can
    # trace SOAP -> transcript -> audio without reloading the transcript.
    source_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scenario_id: str | None = None
    slm_backend: Literal["mlx-lm", "llama-cpp"]
    slm_model: str
    slm_device: str
    slm_structured_mode: Literal["grammar", "raw"]
    generated_at: datetime
    generation_duration_s: float = Field(ge=0.0)


class SOAPNoteSubmission(BaseModel):
    """SOAP note as the SLM submits it — clinical content only, no metadata.

    This is the schema we hand to `lm-format-enforcer` (grammar mode) and the
    type we validate the model's raw JSON against. Splitting it from the full
    `SOAPNote` matters because:

    * The model can't produce `metadata` truthfully (it doesn't know the
      transcript SHA, generation timestamp, or backend it's running on).
      Asking it to invent those values wastes tokens in grammar mode and
      fails validation outright in raw mode.
    * The pipeline owns SHA anchoring + timing, then composes the full
      `SOAPNote` by attaching a `SOAPNoteMetadata` it built itself.
    * Phase 2c's Presidio + trust gate operates on the clinical content,
      not the plumbing — `SOAPNoteSubmission` is the natural input type
      for anonymization.
    """

    model_config = ConfigDict(extra="forbid")

    subjective: SOAPSubjective
    objective: SOAPObjective
    assessment: SOAPAssessment
    plan: SOAPPlan
    # Clinical red flags the SLM surfaced (e.g. "ból zamostkowy +
    # duszność" → prompt to rule out ACS). Separate from S/O/A/P buckets
    # so Phase 2c's audit can flag these specifically on the doctor UI.
    red_flags: list[str] = Field(default_factory=list)
    red_flags_provenance: list[int] = Field(default_factory=list)


class SOAPNote(SOAPNoteSubmission):
    """Full SOAP note for a single clinical visit — submission + metadata.

    Inherits all clinical fields from `SOAPNoteSubmission` and adds the
    pipeline-built metadata block. `extra="forbid"` is inherited via
    `model_config`.
    """

    metadata: SOAPNoteMetadata
