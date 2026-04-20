"""End-to-end pipeline: scenario -> TTS -> post-processing -> output.

Phase 1a implementation covers scenarios only. Phase 1b will chain TTS
synthesis and audio post-processing onto the same entry point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .scenario_generator.generator import ScenarioGenerator, load_scenarios_config
from .scenario_generator.llm_client import build_default_client
from .scenario_generator.schema import Specialization

if TYPE_CHECKING:
    from pathlib import Path

    from .scenario_generator.generator import GenerationReport

log = logging.getLogger(__name__)


def run_scenarios(
    config_path: Path,
    output_dir: Path,
    per_specialization: int | None = None,
) -> GenerationReport:
    """Generate scenarios end-to-end and persist JSON files to output_dir."""
    config = load_scenarios_config(config_path)
    specializations = [Specialization(name) for name in config["specializations"]]
    count = per_specialization or int(config.get("scenarios_per_specialization", 1))
    client = build_default_client(config["llm"])
    generator = ScenarioGenerator(client=client, output_dir=output_dir)
    return generator.run(specializations, per_specialization=count)
