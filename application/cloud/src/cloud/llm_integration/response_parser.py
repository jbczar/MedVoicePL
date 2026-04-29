"""Pydantic schemas for the cloud agent's intermediate LLM outputs.

`DrugCandidate` is the **internal** structure produced by the agent's
`propose_candidates` node — Vertex Gemini emits it as JSON via
`response_schema`. The cloud agent then enriches each candidate with
NFZ refundacja data and ChPL retrieval before composing the final
cross-boundary `DrugSuggestion` (see `shared/schemas/drug.py`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Same ATC pattern as `shared.schemas.drug._ATC_PATTERN`; duplicated to
# avoid importing a private name. Both must stay in sync.
_ATC_PATTERN = r"^[A-Z]\d{2}[A-Z]{2}\d{2}$"


class DrugCandidate(BaseModel):
    """One drug proposal from the LLM, before NFZ + ChPL enrichment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    atc_code: str = Field(pattern=_ATC_PATTERN)
    rationale: str
    provenance: list[str] = Field(min_length=1)


class DrugCandidateList(BaseModel):
    """Top-level wrapper for `response_schema` — Vertex Gemini handles
    object roots more reliably than list roots.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: list[DrugCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
