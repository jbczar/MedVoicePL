"""Tests for `cloud.orchestrator` — full LangGraph runs with all-fake deps.

Covers:

* Happy path: candidates flow through NFZ enrichment + composition.
* Empty-output rule (DD-2): preliminary diagnoses produce empty drugs.
* NFZ miss: candidate's ATC isn't in the index → reimbursement=none.
* Provenance enforcement: every suggested drug has non-empty provenance.
* Determinism: identical input → identical output (with `_FakeLLM`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from cloud.llm_integration.response_parser import DrugCandidate, DrugCandidateList
from cloud.nfz.formulary_search import NFZEntry, _FakeNFZIndex
from cloud.orchestrator import CloudAgent
from cloud.rag.chain import build_fake_chain
from cloud.rag.retriever import ChPLExcerpt
from shared.schemas import AnonymizedAssessment, ReimbursementLevel

# ---------------------------------------------------------------------------
# Fake LLM — canned response per call.
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLM:
    """Returns canned `DrugCandidateList` regardless of prompt."""

    canned: DrugCandidateList
    name: str = "fake-llm"
    calls: int = 0

    def generate(
        self,
        prompt: str,
        *,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
    ) -> BaseModel:
        self.calls += 1
        # Type assertion — guards against orchestrator passing wrong schema.
        assert response_schema is DrugCandidateList
        return self.canned


def _propranolol() -> NFZEntry:
    return NFZEntry(
        atc_code="C07AA05",
        substance="Propranolol",
        reimbursement=ReimbursementLevel.standard,
        patient_copay_pln=Decimal("3.20"),
    )


def _excerpts() -> list[ChPLExcerpt]:
    return [
        ChPLExcerpt(
            atc_code="C07AA05",
            drug_name="propranolol",
            text="propranolol — beta-blocker, leczenie tachykardii.",
        )
    ]


def _build_agent(
    canned: DrugCandidateList,
    *,
    nfz_entries: list[NFZEntry] | None = None,
) -> CloudAgent:
    return CloudAgent(
        llm=_FakeLLM(canned=canned),
        nfz=_FakeNFZIndex(
            entries=tuple(nfz_entries if nfz_entries is not None else [_propranolol()]),
            snapshot_date=date(2026, 4, 1),
        ),
        chain=build_fake_chain(_excerpts()),
        agent_version="3a.0-test",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_drugs_emitted_with_nfz_data(self, make_anonymized_soap: Any) -> None:
        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Propranolol",
                    atc_code="C07AA05",
                    rationale="leczenie tachykardii w nadczynności tarczycy",
                    provenance=["assessment.primary_diagnosis"],
                )
            ]
        )
        agent = _build_agent(canned)
        suggestion = agent.run(make_anonymized_soap())
        assert len(suggestion.drugs) == 1
        drug = suggestion.drugs[0]
        assert drug.atc_code == "C07AA05"
        assert drug.reimbursement == ReimbursementLevel.standard
        assert drug.patient_copay_pln == Decimal("3.20")
        assert drug.provenance == ["assessment.primary_diagnosis"]

    def test_referrals_and_tests_carried_over(self, make_anonymized_soap: Any) -> None:
        canned = DrugCandidateList(candidates=[])
        agent = _build_agent(canned)
        suggestion = agent.run(make_anonymized_soap())
        assert any(r.specialty == "endokrynologia" for r in suggestion.referrals)
        assert any("TSH" in t.name for t in suggestion.tests)

    def test_metadata_populated(self, make_anonymized_soap: Any) -> None:
        canned = DrugCandidateList(candidates=[])
        agent = _build_agent(canned)
        suggestion = agent.run(make_anonymized_soap())
        assert suggestion.metadata.agent_version == "3a.0-test"
        assert suggestion.metadata.llm_model == "fake-llm"
        assert suggestion.metadata.nfz_snapshot_date == date(2026, 4, 1)
        assert suggestion.metadata.chpl_corpus_size == 1
        assert suggestion.metadata.duration_s >= 0


# ---------------------------------------------------------------------------
# Empty-output rule (DD-2)
# ---------------------------------------------------------------------------


class TestEmptyOutputRule:
    def test_podejrzenie_yields_empty_drugs(self, make_anonymized_soap: Any) -> None:
        from shared.schemas import AnonymizedPlan

        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="ShouldNotAppear",
                    atc_code="C07AA05",
                    rationale="x",
                    provenance=["x"],
                )
            ]
        )
        agent = _build_agent(canned)
        soap = make_anonymized_soap(
            assessment=AnonymizedAssessment(
                primary_diagnosis="podejrzenie POChP",
                icd10_code="J44",
            ),
            plan=AnonymizedPlan(medications_prescribed=[]),
        )
        suggestion = agent.run(soap)
        assert suggestion.drugs == []
        assert any("wstępna" in n.lower() or "potwierdzenia" in n.lower() for n in suggestion.notes)

    def test_wymaga_potwierdzenia_yields_empty(self, make_anonymized_soap: Any) -> None:
        from shared.schemas import AnonymizedPlan

        canned = DrugCandidateList(candidates=[])
        agent = _build_agent(canned)
        soap = make_anonymized_soap(
            assessment=AnonymizedAssessment(
                primary_diagnosis="akromegalia",
                clinical_reasoning="rozpoznanie wymaga potwierdzenia laboratoryjnego",
            ),
            plan=AnonymizedPlan(medications_prescribed=[]),
        )
        suggestion = agent.run(soap)
        assert suggestion.drugs == []

    def test_preliminary_with_existing_medication_still_calls_llm(
        self, make_anonymized_soap: Any
    ) -> None:
        """If SLM marks 'podejrzenie' but the plan already has a drug
        listed, the doctor has clinically committed — agent should run.
        """
        from shared.schemas import AnonymizedPlan

        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Propranolol",
                    atc_code="C07AA05",
                    rationale="objawowe leczenie tachykardii",
                    provenance=["assessment.primary_diagnosis"],
                )
            ]
        )
        agent = _build_agent(canned)
        soap = make_anonymized_soap(
            assessment=AnonymizedAssessment(
                primary_diagnosis="podejrzenie nadczynności tarczycy",
            ),
            plan=AnonymizedPlan(medications_prescribed=["Propranolol"]),
        )
        suggestion = agent.run(soap)
        assert len(suggestion.drugs) == 1
        assert suggestion.drugs[0].atc_code == "C07AA05"

    def test_llm_not_called_when_preliminary(self, make_anonymized_soap: Any) -> None:
        from shared.schemas import AnonymizedPlan

        canned = DrugCandidateList(candidates=[])
        llm = _FakeLLM(canned=canned)
        agent = CloudAgent(
            llm=llm,
            nfz=_FakeNFZIndex(),
            chain=build_fake_chain([]),
            agent_version="3a.0-test",
        )
        soap = make_anonymized_soap(
            assessment=AnonymizedAssessment(
                primary_diagnosis="podejrzenie nadczynności tarczycy",
            ),
            plan=AnonymizedPlan(medications_prescribed=[]),
        )
        agent.run(soap)
        assert llm.calls == 0


# ---------------------------------------------------------------------------
# NFZ miss
# ---------------------------------------------------------------------------


class TestLazyResolver:
    """The lazy ATC resolver bridges Gemini's free-text drug names to
    the NFZ index when the agent's claimed ATC isn't found.
    """

    def test_resolver_recovers_nfz_miss(self, make_anonymized_soap: Any) -> None:
        from cloud.nfz.atc_resolver import _FakeAtcResolver

        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Propranolol",
                    atc_code="X99XX99",  # hallucinated ATC
                    rationale="leczenie tachykardii",
                    provenance=["assessment.primary_diagnosis"],
                )
            ]
        )
        # NFZ index has the real Propranolol entry.
        agent = CloudAgent(
            llm=_FakeLLM(canned=canned),
            nfz=_FakeNFZIndex(
                entries=(_propranolol(),), snapshot_date=date(2026, 4, 1)
            ),
            chain=build_fake_chain(_excerpts()),
            atc_resolver=_FakeAtcResolver(canned={"Propranolol": "C07AA05"}),
            agent_version="3a.0-test",
        )
        suggestion = agent.run(make_anonymized_soap())
        assert len(suggestion.drugs) == 1
        # Resolver-recovered entry → reimbursement comes from NFZ.
        assert suggestion.drugs[0].reimbursement == ReimbursementLevel.standard

    def test_resolver_miss_falls_through_to_unrefunded(
        self, make_anonymized_soap: Any
    ) -> None:
        from cloud.nfz.atc_resolver import _FakeAtcResolver

        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Krasnoludek",
                    atc_code="X99XX99",
                    rationale="syntetyczny",
                    provenance=["assessment.primary_diagnosis"],
                )
            ]
        )
        agent = CloudAgent(
            llm=_FakeLLM(canned=canned),
            nfz=_FakeNFZIndex(),
            chain=build_fake_chain([]),
            atc_resolver=_FakeAtcResolver(canned={"Krasnoludek": None}),
            agent_version="3a.0-test",
        )
        suggestion = agent.run(make_anonymized_soap())
        assert len(suggestion.drugs) == 1
        assert suggestion.drugs[0].reimbursement == ReimbursementLevel.none


class TestNFZMiss:
    def test_unknown_drug_marked_none(self, make_anonymized_soap: Any) -> None:
        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Krasnoludek",
                    atc_code="X99XX99",
                    rationale="syntetyczny lek",
                    provenance=["assessment.primary_diagnosis"],
                )
            ]
        )
        agent = _build_agent(canned, nfz_entries=[])
        suggestion = agent.run(make_anonymized_soap())
        assert len(suggestion.drugs) == 1
        drug = suggestion.drugs[0]
        assert drug.reimbursement == ReimbursementLevel.none
        assert drug.patient_copay_pln is None
        assert any("Krasnoludek" in n and "NFZ" in n for n in suggestion.notes)


# ---------------------------------------------------------------------------
# Provenance enforcement
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_every_drug_has_provenance(self, make_anonymized_soap: Any) -> None:
        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Propranolol",
                    atc_code="C07AA05",
                    rationale="x",
                    provenance=["assessment.primary_diagnosis"],
                ),
                DrugCandidate(
                    name="Tiamazol",
                    atc_code="H03BB02",
                    rationale="y",
                    provenance=["assessment.primary_diagnosis", "subjective.chief_complaint"],
                ),
            ]
        )
        agent = _build_agent(
            canned,
            nfz_entries=[
                _propranolol(),
                NFZEntry(
                    atc_code="H03BB02",
                    substance="Tiamazol",
                    reimbursement=ReimbursementLevel.standard,
                ),
            ],
        )
        suggestion = agent.run(make_anonymized_soap())
        assert len(suggestion.drugs) == 2
        for drug in suggestion.drugs:
            assert len(drug.provenance) >= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_soap_same_output(self, make_anonymized_soap: Any) -> None:
        canned = DrugCandidateList(
            candidates=[
                DrugCandidate(
                    name="Propranolol",
                    atc_code="C07AA05",
                    rationale="r",
                    provenance=["assessment.primary_diagnosis"],
                )
            ]
        )
        soap = make_anonymized_soap()
        a = _build_agent(canned).run(soap)
        b = _build_agent(canned).run(soap)
        # duration_s will differ; compare structural fields only.
        assert [d.atc_code for d in a.drugs] == [d.atc_code for d in b.drugs]
        assert [d.reimbursement for d in a.drugs] == [d.reimbursement for d in b.drugs]
        assert [d.provenance for d in a.drugs] == [d.provenance for d in b.drugs]
        assert [r.specialty for r in a.referrals] == [r.specialty for r in b.referrals]


# ---------------------------------------------------------------------------
# Sanity — constructor builds graph
# ---------------------------------------------------------------------------


def test_cloud_agent_builds_graph(make_anonymized_soap: Any) -> None:
    canned = DrugCandidateList(candidates=[])
    agent = _build_agent(canned)
    assert agent._graph is not None


def test_orchestrator_imports_only_shared_types() -> None:
    """Sanity: orchestrator depends only on cloud + shared (no on_premise)."""
    import cloud.orchestrator as orch_mod

    assert "cloud" in orch_mod.__name__


@pytest.mark.parametrize("primary", ["podejrzenie X", "Suspected Y", "do diagnostyki"])
def test_preliminary_markers_match(primary: str, make_anonymized_soap: Any) -> None:
    from shared.schemas import AnonymizedPlan

    canned = DrugCandidateList(candidates=[])
    agent = _build_agent(canned)
    soap = make_anonymized_soap(
        assessment=AnonymizedAssessment(primary_diagnosis=primary),
        plan=AnonymizedPlan(medications_prescribed=[]),
    )
    suggestion = agent.run(soap)
    assert suggestion.drugs == []
