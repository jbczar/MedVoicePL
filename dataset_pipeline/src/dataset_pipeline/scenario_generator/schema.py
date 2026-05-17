"""Pydantic models for medical visit scenario structure.

The `ScenarioModel` here is the stable contract consumed by Phase 1b (TTS),
`application/on_premise/stt` evaluation, and `research_and_testing/test_cases`.
Fields are deliberately minimal; downstream modules (SOAP notes, drug
suggestions) live in `application/shared/src/schemas/`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Specialization(StrEnum):
    kardiologia = "kardiologia"
    pulmonologia = "pulmonologia"
    medycyna_rodzinna = "medycyna_rodzinna"
    endokrynologia = "endokrynologia"
    neurologia = "neurologia"
    # Phase 4 — added for benchmark expansion (40 scenarios target)
    gastrologia = "gastrologia"
    ginekologia = "ginekologia"
    pediatria = "pediatria"


class SpeakerRole(StrEnum):
    doctor = "doctor"
    patient = "patient"


# Phase 4: bumped MAX_TURNS 20 → 50 (5-min dialogs target ~30-40 turns).
# MIN_AGE lowered 18 → 0 for pediatric scenarios (0-17 patient_age).
MIN_TURNS = 6
MAX_TURNS = 50
MIN_AGE = 0
MAX_AGE = 95

# Phase 4: target dialog duration in seconds. Used by scenario generator
# prompt to instruct LLM on conversation length (~30-40 turns ≈ 5 min).
DEFAULT_TARGET_DURATION_S = 300
MIN_TARGET_DURATION_S = 60
MAX_TARGET_DURATION_S = 600


class DialogueTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: SpeakerRole
    text: str = Field(min_length=1)


class ScenarioModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    specialization: Specialization
    patient_age: int = Field(ge=MIN_AGE, le=MAX_AGE)
    patient_sex: Literal["M", "F"]
    patient_full_name: str = Field(min_length=3)
    patient_pesel: str = Field(pattern=r"^\d{11}$")
    chief_complaint: str = Field(min_length=3)
    expected_diagnosis: str = Field(min_length=3)
    icd10_code: str | None = Field(default=None)
    turns: list[DialogueTurn] = Field(min_length=MIN_TURNS, max_length=MAX_TURNS)
    target_duration_seconds: int = Field(
        default=DEFAULT_TARGET_DURATION_S,
        ge=MIN_TARGET_DURATION_S,
        le=MAX_TARGET_DURATION_S,
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    model: str
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_turn_structure(self) -> ScenarioModel:
        if self.turns[0].speaker is not SpeakerRole.doctor:
            raise ValueError("first turn must be spoken by the doctor")
        for i in range(1, len(self.turns)):
            if self.turns[i].speaker is self.turns[i - 1].speaker:
                raise ValueError(
                    f"consecutive turns by same speaker at index {i} "
                    f"({self.turns[i].speaker}); dialogue must alternate"
                )
        return self
