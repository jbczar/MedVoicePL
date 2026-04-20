"""Batch scenario generation orchestrator with JSON validation."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .llm_client import PermanentLLMError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .llm_client import LLMClient
    from .schema import ScenarioModel, Specialization

log = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_scenarios_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    expanded = _expand_env(raw)
    if not isinstance(expanded, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return expanded


@dataclass
class GenerationTask:
    specialization: Specialization
    index: int  # 1-based per specialization


@dataclass
class GenerationFailure:
    specialization: str
    index: int
    attempts: int
    error: str


@dataclass
class GenerationReport:
    written: list[Path] = field(default_factory=list)
    failures: list[GenerationFailure] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.written)


class ScenarioGenerator:
    def __init__(
        self,
        client: LLMClient,
        output_dir: Path,
        *,
        max_validation_retries: int = 2,
        max_concurrency: int | None = None,
    ) -> None:
        self._client = client
        self._output_dir = output_dir
        self._max_validation_retries = max_validation_retries
        self._max_concurrency = max_concurrency or int(
            os.environ.get("DATASET_LLM_MAX_CONCURRENCY", "4")
        )

    def run(
        self,
        specializations: Iterable[Specialization],
        per_specialization: int,
    ) -> GenerationReport:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            GenerationTask(spec, i + 1)
            for spec in specializations
            for i in range(per_specialization)
        ]
        report = GenerationReport()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            bar = progress.add_task("Generating scenarios", total=len(tasks))
            with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
                futures = {pool.submit(self._run_one, t): t for t in tasks}
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        path = fut.result()
                        report.written.append(path)
                    except Exception as exc:
                        log.exception(
                            "task failed: %s #%d", task.specialization.value, task.index
                        )
                        report.failures.append(
                            GenerationFailure(
                                specialization=task.specialization.value,
                                index=task.index,
                                attempts=self._max_validation_retries + 1,
                                error=str(exc),
                            )
                        )
                    progress.advance(bar)

        self._write_failures(report)
        return report

    def _run_one(self, task: GenerationTask) -> Path:
        correction: str | None = None
        last_error: Exception | None = None
        for attempt in range(self._max_validation_retries + 1):
            try:
                scenario = self._client.generate_scenario(
                    task.specialization,
                    correction_hint=correction,
                )
            except PermanentLLMError:
                raise
            except ValidationError as exc:
                last_error = exc
                correction = _summarize_validation_error(exc)
                log.warning(
                    "validation failed (attempt %d) for %s #%d: %s",
                    attempt + 1,
                    task.specialization.value,
                    task.index,
                    correction,
                )
                continue
            return self._persist(scenario)
        raise RuntimeError(
            f"validation failed after {self._max_validation_retries + 1} attempts"
        ) from last_error

    def _persist(self, scenario: ScenarioModel) -> Path:
        filename = f"{scenario.specialization.value}_{scenario.id[:8]}.json"
        path = self._output_dir / filename
        path.write_text(
            scenario.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    def _write_failures(self, report: GenerationReport) -> None:
        if not report.failures:
            return
        path = self._output_dir / "failures.json"
        payload = [f.__dict__ for f in report.failures]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', '')}")
    return "; ".join(parts)
