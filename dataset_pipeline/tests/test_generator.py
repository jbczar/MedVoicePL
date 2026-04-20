"""Tests for the batch scenario generator using a fake LLM client."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError
from src.scenario_generator.generator import ScenarioGenerator
from src.scenario_generator.prompts import FEW_SHOT_EXAMPLES
from src.scenario_generator.schema import ScenarioModel, Specialization


class _FakeClient:
    model_name = "fake-model"

    def __init__(self, behaviors: dict[Specialization, list[str]]) -> None:
        # behaviors[spec] is a stack of "ok" | "validation" | "raise" actions
        self._behaviors = {k: list(v) for k, v in behaviors.items()}
        self.calls: list[tuple[Specialization, str | None]] = []

    def generate_scenario(
        self,
        specialization: Specialization,
        *,
        correction_hint: str | None = None,
    ) -> ScenarioModel:
        self.calls.append((specialization, correction_hint))
        action = self._behaviors[specialization].pop(0)
        if action == "raise":
            raise RuntimeError("simulated permanent error")
        if action == "validation":
            raise ValidationError.from_exception_data("ScenarioModel", [])
        example = FEW_SHOT_EXAMPLES[specialization]
        # give each synthesized scenario a fresh UUID so parallel outputs
        # don't collide on the id-prefix-based filename
        return example.model_copy(
            update={"model": self.model_name, "id": str(uuid4())}
        )


def test_single_specialization_success(tmp_path: Path) -> None:
    client = _FakeClient({Specialization.kardiologia: ["ok", "ok"]})
    generator = ScenarioGenerator(
        client=client, output_dir=tmp_path, max_concurrency=1
    )
    report = generator.run([Specialization.kardiologia], per_specialization=2)
    assert report.success_count == 2
    assert not report.failures
    assert len(list(tmp_path.glob("kardiologia_*.json"))) == 2


def test_validation_retry_then_success(tmp_path: Path) -> None:
    client = _FakeClient(
        {Specialization.pulmonologia: ["validation", "ok"]}
    )
    generator = ScenarioGenerator(
        client=client, output_dir=tmp_path, max_concurrency=1
    )
    report = generator.run([Specialization.pulmonologia], per_specialization=1)
    assert report.success_count == 1
    assert not report.failures
    # second call must carry a correction hint
    assert client.calls[0][1] is None
    assert client.calls[1][1] is not None


def test_validation_exhausted_records_failure(tmp_path: Path) -> None:
    client = _FakeClient(
        {Specialization.neurologia: ["validation", "validation", "validation"]}
    )
    generator = ScenarioGenerator(
        client=client,
        output_dir=tmp_path,
        max_validation_retries=2,
        max_concurrency=1,
    )
    report = generator.run([Specialization.neurologia], per_specialization=1)
    assert report.success_count == 0
    assert len(report.failures) == 1
    failures_path = tmp_path / "failures.json"
    assert failures_path.exists()
    payload = json.loads(failures_path.read_text(encoding="utf-8"))
    assert payload[0]["specialization"] == "neurologia"


def test_output_files_are_valid_scenarios(tmp_path: Path) -> None:
    client = _FakeClient({Specialization.endokrynologia: ["ok"]})
    generator = ScenarioGenerator(
        client=client, output_dir=tmp_path, max_concurrency=1
    )
    generator.run([Specialization.endokrynologia], per_specialization=1)
    written = list(tmp_path.glob("endokrynologia_*.json"))
    assert len(written) == 1
    ScenarioModel.model_validate_json(written[0].read_text(encoding="utf-8"))
