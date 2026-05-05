"""Tests for VisitOrchestrator (Step 5)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from gateway.clients.cloud import CloudError, _FakeCloudClient
from gateway.clients.on_premise import _FakeOnPremiseClient
from gateway.models import Visit
from gateway.orchestrator import OrchestratorError, VisitOrchestrator
from gateway.storage import build_visit_store
from shared.schemas import (
    AgentSuggestion,
    AnonymizedSOAPNote,
    Specialization,
    VisitStatus,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_soap() -> AnonymizedSOAPNote:
    return AnonymizedSOAPNote.model_validate_json(
        (FIXTURES / "anonymized_soap_endo_gra.json").read_text()
    )


def _load_suggestion() -> AgentSuggestion:
    return AgentSuggestion.model_validate_json(
        (FIXTURES / "agent_suggestion_endo_gra.json").read_text()
    )


def _make_visit(visit_store, status: VisitStatus = VisitStatus.audio_uploaded) -> Visit:
    visit = Visit(
        visit_id=uuid4(),
        session_id=uuid4(),
        status=status,
        specialization=Specialization.endokrynologia,
        visit_date=date(2026, 5, 5),
    )
    visit_store.save(visit)
    return visit


async def test_orchestrator_happy_path(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store)

    transcript_id = str(visit.visit_id)[:8]
    on_prem = _FakeOnPremiseClient(responses={transcript_id: _load_soap()})
    cloud = _FakeCloudClient(response=_load_suggestion())
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)

    suggestion = await orch.finalize(visit.visit_id)
    assert isinstance(suggestion, AgentSuggestion)

    refreshed = store.get(visit.visit_id)
    assert refreshed is not None
    assert refreshed.status == VisitStatus.completed
    assert refreshed.completed_at is not None
    assert refreshed.transcript_id == transcript_id
    # Audit copies persisted.
    assert (store.visit_dir(visit.visit_id) / "soap.json").is_file()
    assert (store.visit_dir(visit.visit_id) / "suggestion.json").is_file()


async def test_orchestrator_on_premise_failure_marks_visit_failed(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store)
    transcript_id = str(visit.visit_id)[:8]
    on_prem = _FakeOnPremiseClient(raise_on={transcript_id})
    cloud = _FakeCloudClient(response=_load_suggestion())
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)

    with pytest.raises(OrchestratorError, match="on-prem failed"):
        await orch.finalize(visit.visit_id)
    refreshed = store.get(visit.visit_id)
    assert refreshed is not None
    assert refreshed.status == VisitStatus.failed
    assert refreshed.error and "on_premise_error" in refreshed.error
    # Cloud was never called.
    assert cloud.calls == []


async def test_orchestrator_cloud_failure_marks_visit_failed(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store)
    transcript_id = str(visit.visit_id)[:8]
    on_prem = _FakeOnPremiseClient(responses={transcript_id: _load_soap()})
    cloud = _FakeCloudClient(raise_exc=CloudError("upstream timeout"))
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)

    with pytest.raises(OrchestratorError, match="cloud failed"):
        await orch.finalize(visit.visit_id)
    refreshed = store.get(visit.visit_id)
    assert refreshed is not None
    assert refreshed.status == VisitStatus.failed


async def test_orchestrator_idempotent_replay(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store)
    transcript_id = str(visit.visit_id)[:8]
    on_prem = _FakeOnPremiseClient(responses={transcript_id: _load_soap()})
    cloud = _FakeCloudClient(response=_load_suggestion())
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)

    first = await orch.finalize(visit.visit_id)
    second = await orch.finalize(visit.visit_id)
    assert first == second
    # On-prem called only once — second is cached.
    assert len(on_prem.calls) == 1
    assert len(cloud.calls) == 1


async def test_orchestrator_force_replays_downstream(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store)
    transcript_id = str(visit.visit_id)[:8]
    on_prem = _FakeOnPremiseClient(responses={transcript_id: _load_soap()})
    cloud = _FakeCloudClient(response=_load_suggestion())
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)

    await orch.finalize(visit.visit_id)
    await orch.finalize(visit.visit_id, force=True)
    assert len(on_prem.calls) == 2
    assert len(cloud.calls) == 2


async def test_orchestrator_rejects_visit_in_draft(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store, status=VisitStatus.draft)
    on_prem = _FakeOnPremiseClient(responses={})
    cloud = _FakeCloudClient(response=_load_suggestion())
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)

    with pytest.raises(OrchestratorError, match="not ready"):
        await orch.finalize(visit.visit_id)


async def test_orchestrator_unknown_visit_raises(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    on_prem = _FakeOnPremiseClient()
    cloud = _FakeCloudClient()
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)
    with pytest.raises(OrchestratorError, match="visit not found"):
        await orch.finalize(uuid4())


async def test_orchestrator_persists_intermediate_state(tmp_path: Path) -> None:
    """Status is anonymizing/suggesting visible mid-pipeline (via async observability)."""
    store = build_visit_store(tmp_path / "visits")
    visit = _make_visit(store)
    transcript_id = str(visit.visit_id)[:8]
    soap = _load_soap()

    captured_status: list[str] = []

    class _ObservingFakeOn(_FakeOnPremiseClient):
        async def anonymize_visit(self, **kwargs):  # type: ignore[no-untyped-def]
            current = store.get(visit.visit_id)
            if current is not None:
                captured_status.append(current.status.value)
            return await super().anonymize_visit(**kwargs)

    on_prem = _ObservingFakeOn(responses={transcript_id: soap})
    cloud = _FakeCloudClient(response=_load_suggestion())
    orch = VisitOrchestrator(on_premise=on_prem, cloud=cloud, visit_store=store)
    await orch.finalize(visit.visit_id)
    assert "anonymizing" in captured_status
