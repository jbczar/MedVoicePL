"""Trust-boundary contract: `AnonymizedSOAPNote` for cross-tier transport.

This module is the **only** anchor of the on-premise → cloud trust boundary
on the `shared/` side. The on-premise anonymizer (Phase 2c,
`application/on_premise/src/on_premise/anonymization/presidio_agent.py`)
produces `AnonymizedSOAPNote` instances; everything from `cloud/` and
`gateway/` consumes them.

Design constraints (cross-referenced to plan DD-1, DD-11):

* Mirrors the **field shape** of `on_premise.structuring.soap_schema.SOAPNoteSubmission`
  (clinical content only) but is declared as a standalone `BaseModel`.
  Subclassing would require importing `SOAPNoteSubmission` into `shared/`,
  which `test_trust_boundary_imports.py`'s AST-walk rejects. The
  boilerplate is the price of the boundary — and it leaves room for the
  anonymized type to diverge in the future (e.g. redact
  `clinical_reasoning` further) without rippling into the un-anonymized
  type.
* `extra="forbid"` everywhere — the anonymizer must produce exactly
  these fields; surplus keys are a contract violation, not an extension
  point.
* `provenance: list[int]` is preserved (small side channel — it leaks
  dialogue structure to the cloud, but explainability requires it for
  Phase 3 doctor-side audit views).
* `red_flags` is anonymized like other clinical fields. The SLM might
  emit `"alergia u pana Nowakowskiego"`; the anonymizer walks into
  `red_flags` exactly as it walks into other strings.
* `AnonymizedMetadata` is deliberately tight: no SHA hashes (audio /
  transcript SHAs are direct re-identifiers via the on-prem audit log),
  no backend fingerprint (leaks per-clinic throughput), no millisecond
  timestamp (leaks appointment timing). Only `visit_date` (date, not
  datetime), `specialization`, gateway-minted `session_id`, plus
  `anonymizer_version` + `n_entities_per_type` for ops/audit.
* `Specialization` is duplicated here from
  `dataset_pipeline.scenario_generator.schema` on purpose — `shared/`
  must not depend on `dataset_pipeline/` (a synthetic-data tool, not a
  runtime dep). Phase 3 may consolidate by moving the canonical
  definition into `shared/schemas/visit.py`.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Specialization(StrEnum):
    """Medical specialization. Duplicated from
    `dataset_pipeline.scenario_generator.schema.Specialization` to keep
    `shared/` self-contained.

    Keep this list in sync with:
    - `dataset_pipeline/src/dataset_pipeline/scenario_generator/schema.py`
    - `application/frontend/src/lib/types.ts` (Specialization union)
    """

    kardiologia = "kardiologia"
    pulmonologia = "pulmonologia"
    medycyna_rodzinna = "medycyna_rodzinna"
    endokrynologia = "endokrynologia"
    neurologia = "neurologia"
    # Phase 4 additions — match dataset_pipeline enum.
    gastrologia = "gastrologia"
    ginekologia = "ginekologia"
    pediatria = "pediatria"


# ---------------------------------------------------------------------------
# Section types — mirror SOAPNoteSubmission shape (declared, NOT subclassed).
# ---------------------------------------------------------------------------


class AnonymizedSubjective(BaseModel):
    """S — Subjective. PII-redacted: free-text fields contain placeholder
    tokens (`<PERSON_n>`, `<ADDRESS_n>`, `<PHONE_n>`, ...) where PII once was.
    """

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str
    history_of_present_illness: str
    review_of_systems: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    medications_current: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    provenance: list[int] = Field(default_factory=list)


class AnonymizedVital(BaseModel):
    """One vital-sign reading — clinical content, no PII risk by shape."""

    model_config = ConfigDict(extra="forbid")

    name: str  # mirrors VITAL_NAMES whitelist on the on-prem side
    value: str
    unit: str | None = None


class AnonymizedLabMention(BaseModel):
    """Lab/imaging mention — free-form name allowed; no PII expected."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | None = None
    unit: str | None = None


class AnonymizedObjective(BaseModel):
    """O — Objective. Vitals + labs + physical exam observations."""

    model_config = ConfigDict(extra="forbid")

    physical_exam: list[str] = Field(default_factory=list)
    vitals: list[AnonymizedVital] = Field(default_factory=list)
    labs_mentioned: list[AnonymizedLabMention] = Field(default_factory=list)
    provenance: list[int] = Field(default_factory=list)


class AnonymizedAssessment(BaseModel):
    """A — Assessment. Diagnosis + reasoning. ICD-10 + diagnoses are
    clinical content, not PII; preserved verbatim from the source SOAP.
    """

    model_config = ConfigDict(extra="forbid")

    primary_diagnosis: str
    icd10_code: str | None = Field(
        default=None,
        pattern=r"^[A-Z]\d{2}(\.\d{1,2})?$",
    )
    differential: list[str] = Field(default_factory=list)
    clinical_reasoning: str = ""
    provenance: list[int] = Field(default_factory=list)


class AnonymizedPlan(BaseModel):
    """P — Plan. Medications, referrals, follow-up."""

    model_config = ConfigDict(extra="forbid")

    medications_prescribed: list[str] = Field(default_factory=list)
    diagnostic_tests: list[str] = Field(default_factory=list)
    referrals: list[str] = Field(default_factory=list)
    follow_up: str | None = None
    patient_education: list[str] = Field(default_factory=list)
    provenance: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metadata — deliberately tight. See module docstring for what's stripped.
# ---------------------------------------------------------------------------


class AnonymizedMetadata(BaseModel):
    """Provenance metadata that's safe to cross the trust boundary.

    What's NOT here, and why:

    * `source_transcript_sha256` / `source_audio_sha256` — direct
      re-identifiers via the on-prem audit log. Pair an audio SHA with
      the audit and you get the patient. Stays on-prem.
    * `slm_backend` / `slm_model` / `slm_device` — fingerprint joining
      payloads to one workstation; over many visits leaks per-clinic
      throughput. Belongs in gateway telemetry, not per-payload metadata.
    * `generated_at: datetime` — millisecond precision + clinic working
      hours leaks appointment timing. Replaced by `visit_date` (date
      only).
    * `scenario_id` — synthetic-data only; never present in production.
    """

    model_config = ConfigDict(extra="forbid")

    anonymizer_version: str
    """Bumped on every breaking change to anonymizer behavior. Cached
    anonymized outputs include this version; bumping invalidates caches.
    """

    n_entities_per_type: dict[str, int]
    """Counts per entity type that were redacted (e.g.
    `{"PERSON": 4, "PHONE_NUMBER": 2}`). Audit + observability signal.
    Below the re-identification threshold for any realistic threat model.
    """

    visit_date: date
    """Date of the visit, NOT a datetime — strips appointment timing
    that would leak per-clinic schedule patterns over many visits.
    """

    specialization: Specialization
    """Which clinical specialty handled the visit. Useful for
    cloud-side routing of RAG / NFZ lookups.
    """

    session_id: UUID
    """**Gateway-minted** UUID. Phase 3 gateway will pass this to the
    on-prem API; the on-prem side never generates session IDs itself.
    Avoids two parallel ID spaces.
    """


# ---------------------------------------------------------------------------
# Top-level anonymized SOAP note — cross-boundary contract.
# ---------------------------------------------------------------------------


class AnonymizedSOAPNote(BaseModel):
    """RODO/GDPR-clean SOAP note. The only payload type allowed to cross
    the on-premise → cloud trust boundary.

    `trust_gate.validate_outbound` (Phase 2c) accepts only this type;
    passing a `SOAPNote` instead raises `TrustBoundaryViolation`.
    """

    model_config = ConfigDict(extra="forbid")

    subjective: AnonymizedSubjective
    objective: AnonymizedObjective
    assessment: AnonymizedAssessment
    plan: AnonymizedPlan
    red_flags: list[str] = Field(default_factory=list)
    red_flags_provenance: list[int] = Field(default_factory=list)
    metadata: AnonymizedMetadata
