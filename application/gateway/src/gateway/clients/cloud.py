"""HTTP client for the cloud FastAPI service (Phase 3a, port 8002).

Talks to:
    POST /v1/agent/suggest   (body: AnonymizedSOAPNote, response: AgentSuggestion)
    GET  /v1/health
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import httpx

from shared.schemas import AgentSuggestion, AnonymizedSOAPNote

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RETRIES = 3


class CloudError(RuntimeError):
    """Raised when the cloud agent is unreachable, returns 5xx, or
    yields an unparsable `AgentSuggestion`.
    """


class CloudClient(Protocol):
    async def suggest(
        self, payload: AnonymizedSOAPNote, *, force: bool = False
    ) -> AgentSuggestion: ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...


class _HttpxCloudClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self._retries = retries

    async def suggest(
        self, payload: AnonymizedSOAPNote, *, force: bool = False
    ) -> AgentSuggestion:
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                resp = await self._client.post(
                    "/v1/agent/suggest",
                    params={"force": "true" if force else "false"},
                    content=payload.model_dump_json().encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if 500 <= resp.status_code < 600:
                last_exc = CloudError(f"cloud returned {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise CloudError(
                    f"cloud rejected request ({resp.status_code}): {resp.text[:200]}"
                )
            try:
                return AgentSuggestion.model_validate_json(resp.content)
            except ValueError as exc:
                raise CloudError(
                    f"cloud returned unparsable AgentSuggestion: {exc}"
                ) from exc
        raise CloudError(f"cloud unreachable after {self._retries} attempts: {last_exc}")

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/v1/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            log.warning("cloud health probe failed: %s", exc)
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class _FakeCloudClient:
    def __init__(
        self,
        response: AgentSuggestion | None = None,
        *,
        healthy: bool = True,
        raise_exc: Exception | None = None,
    ) -> None:
        self.response = response
        self.healthy = healthy
        self.raise_exc = raise_exc
        self.calls: list[AnonymizedSOAPNote] = []

    async def suggest(
        self, payload: AnonymizedSOAPNote, *, force: bool = False
    ) -> AgentSuggestion:
        self.calls.append(payload)
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.response is None:
            raise CloudError("no canned response set")
        return self.response

    async def health(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        return None


def build_cloud_client(base_url: str) -> CloudClient:
    return _HttpxCloudClient(base_url)
