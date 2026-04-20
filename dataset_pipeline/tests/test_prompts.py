"""Tests that every few-shot example parses as a valid ScenarioModel."""

from __future__ import annotations

import pytest
from src.scenario_generator.prompts import (
    FEW_SHOT_EXAMPLES,
    render_few_shot_block,
    render_user_prompt,
)
from src.scenario_generator.schema import MAX_TURNS, MIN_TURNS, Specialization


@pytest.mark.parametrize("spec", list(Specialization))
def test_few_shot_example_valid(spec: Specialization) -> None:
    example = FEW_SHOT_EXAMPLES[spec]
    assert example.specialization is spec
    assert MIN_TURNS <= len(example.turns) <= MAX_TURNS


@pytest.mark.parametrize("spec", list(Specialization))
def test_few_shot_block_is_json_string(spec: Specialization) -> None:
    block = render_few_shot_block(spec)
    assert block.startswith("{")
    assert spec.value in block


def test_render_user_prompt_substitutes() -> None:
    prompt = render_user_prompt(Specialization.kardiologia, min_turns=6, max_turns=20)
    assert "kardiologia" in prompt
    assert "6" in prompt and "20" in prompt
