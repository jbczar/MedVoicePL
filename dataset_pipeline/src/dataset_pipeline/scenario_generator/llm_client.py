"""LLM API client for generating medical visit scenarios.

Primary implementation targets Google Gemini via the `google-genai` SDK in
API-key mode (AI Studio). A thin `LLMClient` protocol keeps OpenAI /
Anthropic drop-ins possible without changes to the orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .prompts import SYSTEM_PROMPT_PL, render_few_shot_block, render_user_prompt
from .schema import MAX_TURNS, MIN_TURNS, ScenarioModel, Specialization


class LLMClient(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate_scenario(
        self,
        specialization: Specialization,
        *,
        correction_hint: str | None = None,
    ) -> ScenarioModel: ...


@dataclass
class GeminiSettings:
    model_name: str
    temperature: float
    max_output_tokens: int
    api_key: str

    @classmethod
    def from_env_and_config(cls, llm_config: dict[str, Any]) -> GeminiSettings:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set; copy .env.example to .env and fill it in"
            )
        model_name = os.environ.get("DATASET_LLM_MODEL") or str(llm_config["model"])
        if model_name.startswith("$"):
            raise RuntimeError(
                f"DATASET_LLM_MODEL env var not resolved (got {model_name!r}); "
                f"set it in .env"
            )
        return cls(
            model_name=model_name,
            temperature=float(llm_config.get("temperature", 0.8)),
            max_output_tokens=int(llm_config.get("max_tokens", 4096)),
            api_key=api_key,
        )


class TransientLLMError(RuntimeError):
    """Raised for 5xx / rate-limit responses so tenacity retries."""


class PermanentLLMError(RuntimeError):
    """Raised for 4xx / auth / schema violations that retry won't fix."""


class GeminiClient:
    def __init__(self, settings: GeminiSettings) -> None:
        from google import genai

        self._settings = settings
        self._client = genai.Client(api_key=settings.api_key)

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    def generate_scenario(
        self,
        specialization: Specialization,
        *,
        correction_hint: str | None = None,
    ) -> ScenarioModel:
        user_prompt = render_user_prompt(
            specialization,
            min_turns=MIN_TURNS,
            max_turns=MAX_TURNS,
        )
        few_shot = render_few_shot_block(specialization)
        parts = [
            SYSTEM_PROMPT_PL,
            "Przykład referencyjny (ta sama specjalizacja, inny przypadek):",
            few_shot,
            user_prompt,
        ]
        if correction_hint:
            parts.append(
                "Poprzednia próba nie przeszła walidacji: "
                f"{correction_hint}. Popraw i wygeneruj ponownie."
            )
        full_prompt = "\n\n".join(parts)
        return self._call_with_retry(full_prompt)

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(TransientLLMError),
    )
    def _call_with_retry(self, prompt: str) -> ScenarioModel:
        from google.genai import errors as genai_errors
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self._settings.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._settings.temperature,
                    max_output_tokens=self._settings.max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=ScenarioModel,
                ),
            )
        except genai_errors.APIError as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if isinstance(status, int) and (status == 429 or 500 <= status < 600):
                raise TransientLLMError(str(exc)) from exc
            raise PermanentLLMError(str(exc)) from exc

        parsed: Any = getattr(response, "parsed", None)
        if isinstance(parsed, ScenarioModel):
            model = parsed
        else:
            text = getattr(response, "text", None)
            if not text:
                raise PermanentLLMError("empty response from Gemini")
            model = ScenarioModel.model_validate_json(text)

        model.model = self._settings.model_name
        return model


def build_default_client(llm_config: dict[str, Any]) -> LLMClient:
    provider = os.environ.get("DATASET_LLM_PROVIDER", "google").lower()
    if provider != "google":
        raise NotImplementedError(
            f"DATASET_LLM_PROVIDER={provider!r} not implemented yet; "
            f"only 'google' is supported in Phase 1a"
        )
    settings = GeminiSettings.from_env_and_config(llm_config)
    return GeminiClient(settings)
