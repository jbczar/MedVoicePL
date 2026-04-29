"""LLM client for the cloud agent — Vertex AI Gemini with retry.

Mirrors `dataset_pipeline.scenario_generator.llm_client.GeminiClient`
exactly: same Protocol, same tenacity retry decorator, same
`TransientLLMError`/`PermanentLLMError` split. Both modules use Vertex
AI in `vertexai=True` mode against `GOOGLE_CLOUD_PROJECT` /
`GOOGLE_CLOUD_LOCATION`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class TransientLLMError(RuntimeError):
    """Raised for 5xx / 429 — tenacity retries up to 4x."""


class PermanentLLMError(RuntimeError):
    """Raised for 4xx / auth / schema violations — fail fast."""


class LLMClient(Protocol):
    """Minimum surface the agent needs from any LLM."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
    ) -> BaseModel: ...


@dataclass
class VertexSettings:
    model_name: str
    temperature: float
    max_output_tokens: int
    project: str
    location: str

    @classmethod
    def from_env(cls, *, model_override: str | None = None) -> VertexSettings:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set; copy .env.example to .env "
                "and fill it in (Vertex AI requires a GCP project)"
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        model_name = (
            model_override
            or os.environ.get("VERTEX_AGENT_MODEL")
            or "gemini-2.0-pro"
        )
        return cls(
            model_name=model_name,
            temperature=float(os.environ.get("VERTEX_AGENT_TEMPERATURE", "0.2")),
            max_output_tokens=int(os.environ.get("VERTEX_AGENT_MAX_TOKENS", "4096")),
            project=project,
            location=location,
        )


@dataclass
class VertexClient:
    """Vertex AI Gemini client with structured-output via `response_schema`."""

    settings: VertexSettings

    def __post_init__(self) -> None:
        from google import genai

        self._client = genai.Client(
            vertexai=True,
            project=self.settings.project,
            location=self.settings.location,
        )

    @property
    def name(self) -> str:
        return self.settings.model_name

    def generate(
        self,
        prompt: str,
        *,
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
    ) -> BaseModel:
        return self._call_with_retry(
            prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(TransientLLMError),
    )
    def _call_with_retry(
        self,
        prompt: str,
        *,
        response_schema: type[BaseModel],
        system_instruction: str | None,
    ) -> BaseModel:
        from google.genai import errors as genai_errors
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        try:
            response = self._client.models.generate_content(
                model=self.settings.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if isinstance(status, int) and (status == 429 or 500 <= status < 600):
                raise TransientLLMError(str(exc)) from exc
            raise PermanentLLMError(str(exc)) from exc

        parsed: Any = getattr(response, "parsed", None)
        if isinstance(parsed, response_schema):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise PermanentLLMError("empty response from Vertex Gemini")
        try:
            return response_schema.model_validate_json(text)
        except Exception as exc:
            raise PermanentLLMError(f"schema-violating response: {exc}") from exc


def build_llm_client(*, model_override: str | None = None) -> LLMClient:
    """Factory — returns the configured production `LLMClient`."""
    settings = VertexSettings.from_env(model_override=model_override)
    return VertexClient(settings=settings)
