"""Tests for the validate_dataset CLI script."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.validate_dataset import app
from src.scenario_generator.prompts import FEW_SHOT_EXAMPLES
from src.scenario_generator.schema import Specialization
from typer.testing import CliRunner

runner = CliRunner()


def _write_valid(tmp_path: Path) -> None:
    example = FEW_SHOT_EXAMPLES[Specialization.kardiologia]
    (tmp_path / "kardio.json").write_text(
        example.model_dump_json(indent=2), encoding="utf-8"
    )


def test_validate_passes_for_good_files(tmp_path: Path) -> None:
    _write_valid(tmp_path)
    result = runner.invoke(app, ["--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Invalid: 0" in result.output


def test_validate_fails_for_bad_file(tmp_path: Path) -> None:
    _write_valid(tmp_path)
    (tmp_path / "broken.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "broken.json" in result.output


def test_validate_empty_dir_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--dir", str(tmp_path)])
    assert result.exit_code == 1


@pytest.fixture(autouse=True)
def _isolate_sys_path() -> None:
    # conftest already puts dataset_pipeline/ on sys.path; nothing else needed.
    return None
