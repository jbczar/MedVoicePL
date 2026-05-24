"""HTTP client for the on-premise FastAPI service (Phase 2c, port 8001).

Talks to:
    POST /v1/visits/{transcript_id}/anonymized-soap?session_id&specialization&visit_date
    GET  /v1/health

Retry policy: 3 attempts with exponential backoff on 5xx / network
errors. 4xx are not retried (validation failures aren't transient).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import httpx

from shared.schemas import AnonymizedSOAPNote, Specialization

log = logging.getLogger(__name__)

# SOAP extraction with Bielik can take 60-180 s per visit + STT 30-60 s
# + analyzer warm. Bumped from the 60 s default so live MLX backends
# don't trip read timeout on cold starts.
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_RETRIES = 3


class OnPremiseError(RuntimeError):
    """Raised when the on-prem service is unreachable, returns 5xx, or
    yields a body the gateway cannot parse as `AnonymizedSOAPNote`.
    """


class OnPremiseClient(Protocol):
    async def transcribe_audio(self, audio_path: Path) -> str: ...

    async def get_transcript(self, transcript_id: str) -> dict[str, Any]: ...

    async def anonymize_visit(
        self,
        *,
        transcript_id: str,
        session_id: UUID,
        specialization: Specialization,
        visit_date: date,
        force: bool = False,
    ) -> AnonymizedSOAPNote: ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...


class _HttpxOnPremiseClient:
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

    async def transcribe_audio(self, audio_path: Path) -> str:
        """POST audio to `/v1/transcripts`, return on-prem `transcript_id`.

        Network/5xx → retry 3x with exp backoff (mirrors anonymize_visit).
        STT is expensive (30-60 s warm, ~1x realtime); the timeout is the
        constructor's `timeout_s` (default 300 s).
        """
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                with audio_path.open("rb") as fh:
                    resp = await self._client.post(
                        "/v1/transcripts",
                        files={"file": (audio_path.name, fh, "audio/wav")},
                    )
            except httpx.HTTPError as exc:
                last_exc = exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if 500 <= resp.status_code < 600:
                last_exc = OnPremiseError(
                    f"on-prem /v1/transcripts {resp.status_code}: {resp.text[:200]}"
                )
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise OnPremiseError(
                    f"on-prem rejected transcript ({resp.status_code}): {resp.text[:200]}"
                )
            try:
                body = json.loads(resp.content)
                return str(body["transcript_id"])
            except (ValueError, KeyError) as exc:
                raise OnPremiseError(
                    f"on-prem /v1/transcripts unparsable response: {exc}"
                ) from exc
        raise OnPremiseError(
            f"on-prem /v1/transcripts unreachable after {self._retries} attempts: {last_exc}"
        )

    async def get_transcript(self, transcript_id: str) -> dict[str, Any]:
        """GET cached transcript JSON for the doctor's view.

        Read-only — on-prem persisted it during `transcribe_audio`. The
        gateway proxies this for the frontend's TranscriptViewer.
        """
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                resp = await self._client.get(f"/v1/transcripts/{transcript_id}")
            except httpx.HTTPError as exc:
                last_exc = exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if resp.status_code == 404:
                raise OnPremiseError(f"transcript {transcript_id} not found on-prem")
            if 500 <= resp.status_code < 600:
                last_exc = OnPremiseError(
                    f"on-prem /v1/transcripts/{{id}} {resp.status_code}: {resp.text[:200]}"
                )
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise OnPremiseError(
                    f"on-prem rejected GET transcript ({resp.status_code}): {resp.text[:200]}"
                )
            try:
                return json.loads(resp.content)
            except ValueError as exc:
                raise OnPremiseError(
                    f"on-prem /v1/transcripts/{{id}} unparsable: {exc}"
                ) from exc
        raise OnPremiseError(
            f"on-prem GET transcript unreachable after {self._retries} attempts: {last_exc}"
        )

    async def anonymize_visit(
        self,
        *,
        transcript_id: str,
        session_id: UUID,
        specialization: Specialization,
        visit_date: date,
        force: bool = False,
    ) -> AnonymizedSOAPNote:
        params = {
            "session_id": str(session_id),
            "specialization": specialization.value,
            "visit_date": visit_date.isoformat(),
            "force": "true" if force else "false",
        }
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                resp = await self._client.post(
                    f"/v1/visits/{transcript_id}/anonymized-soap",
                    params=params,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if 500 <= resp.status_code < 600:
                last_exc = OnPremiseError(
                    f"on-prem returned {resp.status_code}: {resp.text[:200]}"
                )
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            if resp.status_code == 404:
                raise OnPremiseError(f"transcript not found: {transcript_id}")
            if resp.status_code >= 400:
                raise OnPremiseError(
                    f"on-prem rejected request ({resp.status_code}): {resp.text[:200]}"
                )
            try:
                return AnonymizedSOAPNote.model_validate_json(resp.content)
            except ValueError as exc:
                raise OnPremiseError(
                    f"on-prem returned unparsable AnonymizedSOAPNote: {exc}"
                ) from exc
        raise OnPremiseError(
            f"on-prem unreachable after {self._retries} attempts: {last_exc}"
        )

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/v1/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            log.warning("on-prem health probe failed: %s", exc)
            return False

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class _FakeOnPremiseClient:
    """Test double — return canned `AnonymizedSOAPNote` per transcript_id."""

    def __init__(
        self,
        responses: dict[str, AnonymizedSOAPNote] | None = None,
        *,
        healthy: bool = True,
        raise_on: set[str] | None = None,
        canned_transcript_id: str | None = None,
        canned_transcripts: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.healthy = healthy
        self.raise_on = raise_on or set()
        self.canned_transcript_id = canned_transcript_id
        self.canned_transcripts = canned_transcripts or {}
        self.calls: list[dict] = []
        self.transcribe_calls: list[Path] = []
        self.get_transcript_calls: list[str] = []

    async def transcribe_audio(self, audio_path: Path) -> str:
        self.transcribe_calls.append(audio_path)
        if self.canned_transcript_id is not None:
            return self.canned_transcript_id
        # Deterministic from audio bytes (mirrors real on-prem id derivation).
        import hashlib

        return hashlib.sha256(audio_path.read_bytes()).hexdigest()[:8]

    async def get_transcript(self, transcript_id: str) -> dict[str, Any]:
        self.get_transcript_calls.append(transcript_id)
        if transcript_id in self.canned_transcripts:
            return self.canned_transcripts[transcript_id]
        # Minimal canned shape — matches `Transcript` schema enough for UI.
        return {
            "scenario_id": transcript_id,
            "audio_path": f"<fake>/{transcript_id}.wav",
            "audio_sha256": "0" * 64,
            "duration_s": 0.0,
            "segments": [],
            "metadata": {"engine": "fake", "model": "fake"},
        }

    async def anonymize_visit(
        self,
        *,
        transcript_id: str,
        session_id: UUID,
        specialization: Specialization,
        visit_date: date,
        force: bool = False,
    ) -> AnonymizedSOAPNote:
        self.calls.append(
            {
                "transcript_id": transcript_id,
                "session_id": session_id,
                "specialization": specialization,
                "visit_date": visit_date,
                "force": force,
            }
        )
        if transcript_id in self.raise_on:
            raise OnPremiseError(f"forced failure for {transcript_id}")
        if transcript_id not in self.responses:
            raise OnPremiseError(f"no canned response for {transcript_id}")
        return self.responses[transcript_id]

    async def health(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        return None


def build_on_premise_client(base_url: str) -> OnPremiseClient:
    return _HttpxOnPremiseClient(base_url)
