"""shared.schemas — cross-tier Pydantic contracts.

Exposes the trust-boundary-safe types that flow between on-premise and
cloud. **Never** re-exports anything from `application/on_premise/` —
the AST-walk lint in
`application/on_premise/tests/test_trust_boundary_imports.py` enforces
this at the module-graph level.
"""

from __future__ import annotations

from shared.schemas.agent import (
    AgentMetadata,
    AgentSuggestion,
    DiagnosticTest,
    Referral,
)
from shared.schemas.anonymized import (
    AnonymizedAssessment,
    AnonymizedLabMention,
    AnonymizedMetadata,
    AnonymizedObjective,
    AnonymizedPlan,
    AnonymizedSOAPNote,
    AnonymizedSubjective,
    AnonymizedVital,
    Specialization,
)
from shared.schemas.drug import DrugSuggestion, ReimbursementLevel
from shared.schemas.visit import VisitStatus, VisitSummary

__all__ = [
    "AgentMetadata",
    "AgentSuggestion",
    "AnonymizedAssessment",
    "AnonymizedLabMention",
    "AnonymizedMetadata",
    "AnonymizedObjective",
    "AnonymizedPlan",
    "AnonymizedSOAPNote",
    "AnonymizedSubjective",
    "AnonymizedVital",
    "DiagnosticTest",
    "DrugSuggestion",
    "Referral",
    "ReimbursementLevel",
    "Specialization",
    "VisitStatus",
    "VisitSummary",
]
