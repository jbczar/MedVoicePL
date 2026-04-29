"""LangGraph orchestrator for the cloud agent — 3 explicit nodes.

Per DD-2 in the plan, we use a `StateGraph` with three deterministic
nodes instead of `create_react_agent`:

1. `propose_candidates` — Vertex Gemini reads the SOAP, emits a list
   of `DrugCandidate` records.
2. `enrich_from_nfz` — deterministic NFZ lookup per candidate. Attaches
   `reimbursement` + `patient_copay_pln` + `indication_restrictions`.
3. `compose_suggestions` — pulls ChPL excerpts per drug for context,
   formats final `AgentSuggestion` with provenance + metadata.

Tests use `_FakeLLM` + `_FakeNFZIndex` + `_FakeRetriever` to drive
the graph without touching Vertex/Qdrant.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cloud.llm_integration.llm_client import LLMClient, PermanentLLMError
from cloud.llm_integration.prompts import (
    SYSTEM_PROMPT_PL,
    render_candidates_prompt,
    render_correction_hint,
)
from cloud.llm_integration.response_parser import DrugCandidate, DrugCandidateList
from cloud.nfz.drug_matcher import MatchResult, match_candidate
from cloud.nfz.formulary_search import NFZEntry, NFZIndex
from cloud.rag.chain import ChPLChain
from cloud.rag.retriever import ChPLExcerpt
from shared.schemas import (
    AgentMetadata,
    AgentSuggestion,
    AnonymizedSOAPNote,
    DiagnosticTest,
    DrugSuggestion,
    Referral,
    ReimbursementLevel,
)

log = logging.getLogger(__name__)


# Polish + English markers for "preliminary diagnosis" — DD-2 empty-output rule.
_PRELIMINARY_RE = re.compile(
    r"\b(podejrzenie|suspected|do\s+diagnostyki|wymaga\s+potwierdzenia|do\s+wykluczenia)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class EnrichedCandidate:
    """A `DrugCandidate` with NFZ data attached."""

    candidate: DrugCandidate
    nfz_entry: NFZEntry | None
    match_strategy: str | None  # "atc"|"substance"|"trade_name"|"stem"|None


class AgentState(TypedDict, total=False):
    soap: AnonymizedSOAPNote
    candidates: list[DrugCandidate]
    enriched: list[EnrichedCandidate]
    notes: list[str]
    suggestion: AgentSuggestion
    started_at: float
    # Deps injected via state — LangGraph nodes read them per-call.
    llm: LLMClient
    nfz: NFZIndex
    chain: ChPLChain
    agent_version: str


# ---------------------------------------------------------------------------
# Node 1 — propose candidates (or short-circuit on preliminary diagnosis).
# ---------------------------------------------------------------------------


def _propose_candidates_node(state: AgentState) -> dict[str, Any]:
    soap: AnonymizedSOAPNote = state["soap"]
    notes: list[str] = list(state.get("notes", []))

    if _is_preliminary(soap):
        log.info("preliminary diagnosis detected — skipping LLM, returning empty drugs")
        notes.append(
            "Diagnoza wstępna — wymaga potwierdzenia przed farmakoterapią."
        )
        return {"candidates": [], "notes": notes}

    llm: LLMClient = state["llm"]
    prompt = render_candidates_prompt(soap)

    try:
        result = llm.generate(
            prompt,
            response_schema=DrugCandidateList,
            system_instruction=SYSTEM_PROMPT_PL,
        )
    except PermanentLLMError as exc:
        log.warning("LLM permanent error — empty candidates: %s", exc)
        notes.append(f"LLM nie wygenerował propozycji: {exc}")
        return {"candidates": [], "notes": notes}
    except Exception as exc:
        log.exception("LLM unexpected error")
        notes.append(f"LLM unexpected error: {exc}")
        return {"candidates": [], "notes": notes}

    if not isinstance(result, DrugCandidateList):
        # Try one correction-hint round if the schema didn't bind.
        hint = render_correction_hint("expected DrugCandidateList JSON shape")
        try:
            result = llm.generate(
                prompt + "\n\n" + hint,
                response_schema=DrugCandidateList,
                system_instruction=SYSTEM_PROMPT_PL,
            )
        except Exception as exc:
            notes.append(f"LLM correction retry failed: {exc}")
            return {"candidates": [], "notes": notes}

    notes.extend(result.notes)
    return {"candidates": list(result.candidates), "notes": notes}


def _is_preliminary(soap: AnonymizedSOAPNote) -> bool:
    text_blob = " ".join(
        [
            soap.assessment.primary_diagnosis or "",
            soap.assessment.clinical_reasoning or "",
        ]
    )
    return bool(_PRELIMINARY_RE.search(text_blob))


# ---------------------------------------------------------------------------
# Node 2 — enrich from NFZ.
# ---------------------------------------------------------------------------


def _enrich_from_nfz_node(state: AgentState) -> dict[str, Any]:
    nfz: NFZIndex = state["nfz"]
    candidates: list[DrugCandidate] = state.get("candidates", [])
    notes: list[str] = list(state.get("notes", []))

    enriched: list[EnrichedCandidate] = []
    for cand in candidates:
        match: MatchResult | None = match_candidate(
            name=cand.name, atc_code=cand.atc_code, index=nfz
        )
        if match is None:
            notes.append(
                f"{cand.name} ({cand.atc_code}): brak w wykazie NFZ — pełnopłatne."
            )
            enriched.append(EnrichedCandidate(candidate=cand, nfz_entry=None, match_strategy=None))
            continue
        enriched.append(
            EnrichedCandidate(
                candidate=cand, nfz_entry=match.entry, match_strategy=match.strategy
            )
        )
    return {"enriched": enriched, "notes": notes}


# ---------------------------------------------------------------------------
# Node 3 — compose final suggestions.
# ---------------------------------------------------------------------------


def _compose_suggestions_node(state: AgentState) -> dict[str, Any]:
    soap: AnonymizedSOAPNote = state["soap"]
    enriched: list[EnrichedCandidate] = state.get("enriched", [])
    notes: list[str] = list(state.get("notes", []))
    chain: ChPLChain = state["chain"]
    started_at: float = state.get("started_at", time.time())
    agent_version: str = state["agent_version"]
    llm: LLMClient = state["llm"]

    drug_suggestions: list[DrugSuggestion] = []
    for enr in enriched:
        cand = enr.candidate
        # ATC from NFZ entry wins (canonical) when found, else from candidate.
        atc_code = enr.nfz_entry.atc_code if enr.nfz_entry else cand.atc_code

        # Pull ChPL excerpts to enrich rationale with regulatory context.
        excerpts: list[ChPLExcerpt] = chain.retrieve_for_drug(
            atc_code=atc_code, drug_name=cand.name, k=1
        )
        rationale = cand.rationale
        if excerpts:
            rationale = _augment_rationale(rationale, excerpts[0])

        drug_suggestions.append(
            DrugSuggestion(
                name=cand.name,
                atc_code=atc_code,
                reimbursement=(
                    enr.nfz_entry.reimbursement
                    if enr.nfz_entry
                    else ReimbursementLevel.none
                ),
                patient_copay_pln=(
                    enr.nfz_entry.patient_copay_pln if enr.nfz_entry else None
                ),
                indication_restrictions=(
                    list(enr.nfz_entry.indication_restrictions) if enr.nfz_entry else []
                ),
                provenance=list(cand.provenance),
                rationale=rationale,
            )
        )

    referrals = [
        Referral(specialty=spec.strip())
        for spec in soap.plan.referrals
        if spec.strip()
    ]
    tests = [
        DiagnosticTest(name=name.strip())
        for name in soap.plan.diagnostic_tests
        if name.strip()
    ]

    nfz_snapshot = _nfz_snapshot_date(state)
    metadata = AgentMetadata(
        agent_version=agent_version,
        llm_model=getattr(llm, "name", "unknown"),
        nfz_snapshot_date=nfz_snapshot,
        chpl_corpus_size=len(chain.retriever),
        duration_s=round(time.time() - started_at, 3),
    )
    suggestion = AgentSuggestion(
        drugs=drug_suggestions,
        referrals=referrals,
        tests=tests,
        notes=notes,
        metadata=metadata,
    )
    return {"suggestion": suggestion}


def _nfz_snapshot_date(state: AgentState) -> date:
    nfz = state["nfz"]
    snapshot = getattr(nfz, "snapshot_date", None)
    if isinstance(snapshot, date):
        return snapshot
    return date.today()


def _augment_rationale(base: str, excerpt: ChPLExcerpt) -> str:
    """Append a brief ChPL signal to the LLM's rationale."""
    head = excerpt.text.splitlines()[0].strip() if excerpt.text else ""
    if not head:
        return base
    if len(head) > 180:
        head = head[:177] + "..."
    return f"{base} (ChPL: {head})"


# ---------------------------------------------------------------------------
# Graph build + invoke.
# ---------------------------------------------------------------------------


@dataclass
class CloudAgent:
    """Compiled LangGraph agent + its deps. Constructed once at startup
    (FastAPI lifespan) and reused per request.
    """

    llm: LLMClient
    nfz: NFZIndex
    chain: ChPLChain
    agent_version: str = "3a.0"
    _graph: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._graph = build_agent_graph()

    def run(self, soap: AnonymizedSOAPNote) -> AgentSuggestion:
        state: AgentState = {
            "soap": soap,
            "candidates": [],
            "enriched": [],
            "notes": [],
            "started_at": time.time(),
            "llm": self.llm,
            "nfz": self.nfz,
            "chain": self.chain,
            "agent_version": self.agent_version,
        }
        result = self._graph.invoke(state)
        suggestion = result.get("suggestion")
        if suggestion is None:
            raise RuntimeError("agent graph did not produce a suggestion")
        return suggestion


def build_agent_graph() -> Any:
    """Compile the 3-node `StateGraph`."""
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("propose_candidates", _propose_candidates_node)
    graph.add_node("enrich_from_nfz", _enrich_from_nfz_node)
    graph.add_node("compose_suggestions", _compose_suggestions_node)
    graph.add_edge(START, "propose_candidates")
    graph.add_edge("propose_candidates", "enrich_from_nfz")
    graph.add_edge("enrich_from_nfz", "compose_suggestions")
    graph.add_edge("compose_suggestions", END)
    return graph.compile()
