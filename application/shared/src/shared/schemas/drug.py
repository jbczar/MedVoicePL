"""Cross-tier drug-suggestion contract — Phase 3a output type.

`DrugSuggestion` is the unit emitted by the cloud agent (Phase 3a) for
each candidate drug. Wrapped inside `AgentSuggestion` (see `agent.py`),
it crosses from cloud → gateway → frontend.

Design constraints (cross-referenced to plan DD-4, DD-5):

* `extra="forbid"` everywhere — surplus keys are a contract violation.
* `provenance: list[str]` is **non-empty** by Pydantic constraint.
  Every suggestion must point back to the SOAP fields that drove it
  (e.g. `["assessment.primary_diagnosis", "subjective.chief_complaint"]`).
  This is the doctor's audit trail; empty provenance means the agent
  produced an unsourced recommendation, which is a bug.
* `atc_code` is regex-validated against the canonical 7-character ATC
  format (`A02BC01`, `C07AB07`, `B01AF02`). The cloud agent's NFZ
  matching logic depends on this.
* `reimbursement` is one of the canonical NFZ tiers; `none` means the
  drug is in our suggestion but NOT on the refundacja list — the
  patient pays full price. The agent flags this in `notes`.
* Polish złoty values use `Decimal` (not float) — money never floats.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReimbursementLevel(StrEnum):
    """NFZ refundacja tier. Source: official NFZ wykaz leków refundowanych.

    Naming mirrors gov.pl's tier strings so the NFZ ingestion script can
    map XLSX cell text directly to enum values via
    `reimbursement.parse_reimbursement_string()` on the cloud side.
    """

    full = "100%"           # bezpłatne — refunded in full
    standard = "70%"        # standard tier; patient pays 30%
    half = "50%"            # patient pays 50%
    flat = "ryczałt"        # flat copay (e.g. 3.20 PLN regardless of price)
    none = "nierefundowane"  # not on the NFZ list; patient pays full price


# Canonical 7-char ATC code: 1 letter + 2 digits + 2 letters + 2 digits.
# E.g. `A02BC01` (Omeprazol), `C07AB07` (Bisoprolol), `B01AF02` (Apixaban).
_ATC_PATTERN = r"^[A-Z]\d{2}[A-Z]{2}\d{2}$"


class DrugSuggestion(BaseModel):
    """One drug recommended by the cloud agent.

    Carries enough context for the doctor's UI to:
    1. Display the suggestion (name + reimbursement tier + copay).
    2. Justify it (rationale + provenance pointing into the SOAP).
    3. Surface contraindications (indication_restrictions list).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Drug name as the agent suggested it. Usually the active substance
    in Polish (e.g. ``"propranolol"``) or a common trade name
    (e.g. ``"Eliquis"``).
    """

    atc_code: str = Field(pattern=_ATC_PATTERN)
    """Canonical 7-character ATC code. Validated against the regex —
    malformed codes are a contract violation.
    """

    reimbursement: ReimbursementLevel
    """NFZ refundacja tier. ``none`` if the drug isn't in the formulary
    (the agent still suggests it, but flags the cost concern in
    `notes` on the parent `AgentSuggestion`).
    """

    patient_copay_pln: Decimal | None = None
    """Patient out-of-pocket cost in Polish złoty. ``None`` if not
    available from the NFZ index. ``Decimal`` (not float) — currency
    arithmetic must be exact.
    """

    indication_restrictions: list[str] = Field(default_factory=list)
    """Conditions under which the refundacja applies (e.g.
    ``["w połączeniu z metforminą"]`` for type-2-diabetes specific
    refunds). Free-form Polish.
    """

    provenance: list[str] = Field(min_length=1)
    """JSON-path-ish locations in the source `AnonymizedSOAPNote` that
    triggered this suggestion. **Must be non-empty** (DD-5) — empty
    provenance means the agent produced an unsourced recommendation.
    """

    rationale: str
    """Short clinical rationale. Polish, 1-2 sentences. The doctor's
    UI shows this on the suggestion card.
    """
