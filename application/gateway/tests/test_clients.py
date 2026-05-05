"""Tests for OnPremise/Cloud HTTP clients (Step 4)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from gateway.clients.cloud import CloudError, _FakeCloudClient, _HttpxCloudClient
from gateway.clients.on_premise import (
    OnPremiseError,
    _FakeOnPremiseClient,
    _HttpxOnPremiseClient,
)
from shared.schemas import AgentSuggestion, AnonymizedSOAPNote, Specialization

FIXTURES = Path(__file__).parent / "fixtures"


def _load_soap() -> AnonymizedSOAPNote:
    return AnonymizedSOAPNote.model_validate_json(
        (FIXTURES / "anonymized_soap_endo_gra.json").read_text()
    )


def _load_suggestion() -> AgentSuggestion:
    return AgentSuggestion.model_validate_json(
        (FIXTURES / "agent_suggestion_endo_gra.json").read_text()
    )


# ---------------------------------------------------------------------------
# Fake clients (no httpx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_on_premise_returns_canned_response() -> None:
    soap = _load_soap()
    client = _FakeOnPremiseClient(responses={"abc12345": soap})
    out = await client.anonymize_visit(
        transcript_id="abc12345",
        session_id=uuid4(),
        specialization=Specialization.kardiologia,
        visit_date=date(2026, 5, 5),
    )
    assert out == soap
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_fake_on_premise_raises_when_unset() -> None:
    client = _FakeOnPremiseClient()
    with pytest.raises(OnPremiseError):
        await client.anonymize_visit(
            transcript_id="xxx",
            session_id=uuid4(),
            specialization=Specialization.kardiologia,
            visit_date=date(2026, 5, 5),
        )


@pytest.mark.asyncio
async def test_fake_on_premise_raise_on_set() -> None:
    client = _FakeOnPremiseClient(raise_on={"boom"})
    with pytest.raises(OnPremiseError):
        await client.anonymize_visit(
            transcript_id="boom",
            session_id=uuid4(),
            specialization=Specialization.kardiologia,
            visit_date=date(2026, 5, 5),
        )


@pytest.mark.asyncio
async def test_fake_cloud_returns_canned() -> None:
    suggestion = _load_suggestion()
    client = _FakeCloudClient(response=suggestion)
    out = await client.suggest(_load_soap())
    assert out == suggestion


@pytest.mark.asyncio
async def test_fake_cloud_raises_when_no_response() -> None:
    client = _FakeCloudClient()
    with pytest.raises(CloudError):
        await client.suggest(_load_soap())


# ---------------------------------------------------------------------------
# Real clients via httpx MockTransport
# ---------------------------------------------------------------------------


def _on_prem_handler(soap_json: str):
    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/v1/visits/")
        assert request.url.path.endswith("/anonymized-soap")
        params = dict(request.url.params)
        assert "session_id" in params
        assert "specialization" in params
        assert "visit_date" in params
        return httpx.Response(200, content=soap_json)

    return _handler


@pytest.mark.asyncio
async def test_httpx_on_premise_happy_path() -> None:
    soap = _load_soap()
    soap_json = soap.model_dump_json()
    transport = httpx.MockTransport(_on_prem_handler(soap_json))
    inner = httpx.AsyncClient(base_url="http://test", transport=transport)
    client = _HttpxOnPremiseClient("http://test", client=inner)
    try:
        out = await client.anonymize_visit(
            transcript_id="abc12345",
            session_id=uuid4(),
            specialization=Specialization.endokrynologia,
            visit_date=date(2026, 5, 5),
        )
        assert out == soap
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_on_premise_404_raises_immediately() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "transcript not found"})

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxOnPremiseClient("http://test", client=inner)
    try:
        with pytest.raises(OnPremiseError, match="transcript not found"):
            await client.anonymize_visit(
                transcript_id="xxx",
                session_id=uuid4(),
                specialization=Specialization.kardiologia,
                visit_date=date(2026, 5, 5),
            )
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_on_premise_5xx_retries_and_succeeds() -> None:
    soap_json = _load_soap().model_dump_json()
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] < 3:
            return httpx.Response(503, text="service degraded")
        return httpx.Response(200, content=soap_json)

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxOnPremiseClient("http://test", client=inner, retries=4)
    try:
        out = await client.anonymize_visit(
            transcript_id="abc12345",
            session_id=uuid4(),
            specialization=Specialization.kardiologia,
            visit_date=date(2026, 5, 5),
        )
        assert state["calls"] == 3
        assert out is not None
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_on_premise_5xx_exhausted_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxOnPremiseClient("http://test", client=inner, retries=2)
    try:
        with pytest.raises(OnPremiseError, match="unreachable after"):
            await client.anonymize_visit(
                transcript_id="abc12345",
                session_id=uuid4(),
                specialization=Specialization.kardiologia,
                visit_date=date(2026, 5, 5),
            )
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_on_premise_unparsable_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxOnPremiseClient("http://test", client=inner)
    try:
        with pytest.raises(OnPremiseError, match="unparsable"):
            await client.anonymize_visit(
                transcript_id="abc12345",
                session_id=uuid4(),
                specialization=Specialization.kardiologia,
                visit_date=date(2026, 5, 5),
            )
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_on_premise_health_ok_and_fail() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(ok))
    client = _HttpxOnPremiseClient("http://test", client=inner)
    try:
        assert await client.health() is True
    finally:
        await inner.aclose()

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(fail))
    client = _HttpxOnPremiseClient("http://test", client=inner)
    try:
        assert await client.health() is False
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_cloud_happy_path() -> None:
    suggestion = _load_suggestion()
    suggestion_json = suggestion.model_dump_json()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/agent/suggest"
        body = json.loads(request.content.decode("utf-8"))
        # Should be a valid AnonymizedSOAPNote payload.
        assert "subjective" in body
        return httpx.Response(200, content=suggestion_json)

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxCloudClient("http://test", client=inner)
    try:
        out = await client.suggest(_load_soap())
        assert out == suggestion
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_cloud_5xx_raises_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxCloudClient("http://test", client=inner, retries=2)
    try:
        with pytest.raises(CloudError, match="unreachable"):
            await client.suggest(_load_soap())
    finally:
        await inner.aclose()


@pytest.mark.asyncio
async def test_httpx_cloud_4xx_raises_immediately() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(400, text="bad request")

    inner = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
    client = _HttpxCloudClient("http://test", client=inner, retries=4)
    try:
        with pytest.raises(CloudError, match="rejected"):
            await client.suggest(_load_soap())
        assert state["calls"] == 1  # not retried
    finally:
        await inner.aclose()
