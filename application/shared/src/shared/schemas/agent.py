"""Cross-tier cloud-agent output contract — Phase 3a top-level result.

`AgentSuggestion` is what the cloud agent emits per visit. It bundles
drug suggestions (the headline output), referrals + diagnostic tests
(deterministic carry-over from the SOAP `plan` section), free-text
`notes` for things the agent surfaces but can't structure (e.g.
"diagnosis pending confirmation"), and a tight metadata block.

Trust-boundary posture mirrors `AnonymizedSOAPNote`:

* `extra="forbid"` everywhere.
* No PII fields — by construction, the agent only sees `AnonymizedSOAPNote`
  on its way in, so it can't leak PII on the way out.
* Metadata is deliberately tight — same posture as `AnonymizedMetadata`
  in `anonymized.py`. We surface enough for ops/audit (model, version,
  corpus snapshot, latency) but no fields that fingerprint the patient
  or the clinic.

Cross-references:

* DD-4 (plan): `AgentSuggestion` is a new top-level cross-boundary type.
* DD-9: this is the response body of `POST /v1/agent/suggest`.
* DD-10: trust-boundary lint asserts these are importable from
  `shared.schemas`; positive contract test.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from shared.schemas.drug import DrugSuggestion


class Referral(BaseModel):
    """One specialist referral. Deterministic carry-over from
    `AnonymizedSOAPNote.plan.referrals`, normalized for downstream UI.
    """

    model_config = ConfigDict(extra="forbid")

    specialty: str
    """Polish specialty name (e.g. ``"endokrynologia"``,
    ``"kardiologia"``).
    """

    rationale: str = ""
    """Optional 1-line clinical reason for the referral. Empty string
    when the SOAP didn't justify it explicitly.
    """


class DiagnosticTest(BaseModel):
    """One diagnostic test recommendation. Carries over from
    `AnonymizedSOAPNote.plan.diagnostic_tests` with optional NFZ refund
    info (some tests are also refundable).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Test name in Polish (e.g. ``"USG tarczycy"``,
    ``"Holter EKG (24h)"``, ``"TSH, fT3, fT4"``).
    """

    rationale: str = ""
    """Optional 1-line clinical justification."""

    urgency: str | None = None
    """Optional urgency hint (``"pilne"``, ``"plan"``, etc.). ``None``
    when not specified.
    """


class AgentMetadata(BaseModel):
    """Provenance + ops fields for the agent's output.

    Tight set — see DD-9. No clinic / patient fingerprints.
    """

    model_config = ConfigDict(extra="forbid")

    agent_version: str
    """Bumped on every breaking change to agent behavior. Cached
    suggestions in the API include this; bumping invalidates the cache.
    """

    llm_model: str
    """E.g. ``"gemini-2.0-pro"``. For audit / reproducibility."""

    nfz_snapshot_date: date
    """Date of the NFZ refundacja list snapshot used. Bimonthly cadence;
    bumping the snapshot is a deliberate operational action.
    """

    chpl_corpus_size: int
    """Number of ChPL excerpts indexed at agent build time. Sanity
    check that the RAG corpus didn't shrink unexpectedly.
    """

    duration_s: float
    """Wall-clock seconds for the agent run, including LLM + RAG +
    NFZ lookups. Doesn't include client-side network round-trip.
    """


class AgentSuggestion(BaseModel):
    """Top-level cloud-agent output. The response body of
    ``POST /v1/agent/suggest``.

    Empty `drugs` is a valid output — half the test corpus is
    diagnostic-only, where the agent should NOT prescribe (DD-2 empty
    output rule). In that case `notes` carries the explanation.
    """

    model_config = ConfigDict(extra="forbid")

    drugs: list[DrugSuggestion] = Field(default_factory=list)
    """Drugs the agent recommends. May be empty when diagnosis is
    preliminary; `notes` explains why.
    """

    referrals: list[Referral] = Field(default_factory=list)
    """Specialist referrals. Carried over deterministically from the
    SOAP `plan.referrals`; not LLM-generated.
    """

    tests: list[DiagnosticTest] = Field(default_factory=list)
    """Diagnostic tests. Same posture as `referrals`."""

    notes: list[str] = Field(default_factory=list)
    """Free-text caveats the agent wants surfaced to the doctor.
    Examples: ``"Diagnoza wymaga potwierdzenia — nie zaleca się
    farmakoterapii"``, ``"Apixaban nie znaleziony w wykazie NFZ"``.
    """

    metadata: AgentMetadata
