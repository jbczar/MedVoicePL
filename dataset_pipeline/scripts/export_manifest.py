"""Generate dataset manifest (paths, metadata, hashes).

Produces `output/manifest.json` with two top-level arrays:

  - `scenarios`: one entry per JSON file in `output/scenarios/` (Phase 1a).
  - `audio`: one entry per `output/audio/<id>/bundle.json` sidecar (Phase 1b).

The manifest is the integrity contract Phase 2+ reads. When audio hasn't been
synthesized yet, `audio` is an empty list.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from dataset_pipeline.audio_schema import ScenarioAudioBundle
from dataset_pipeline.scenario_generator.schema import ScenarioModel

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_SCENARIOS_DIR = Path("dataset_pipeline/output/scenarios")
DEFAULT_AUDIO_DIR = Path("dataset_pipeline/output/audio")
DEFAULT_MANIFEST = Path("dataset_pipeline/output/manifest.json")


def _collect_scenarios(scenarios_dir: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    files = sorted(
        p for p in scenarios_dir.glob("*.json") if p.name != "failures.json"
    )
    for path in files:
        raw = path.read_bytes()
        scenario = ScenarioModel.model_validate_json(raw)
        entries.append(
            {
                "path": str(path.relative_to(scenarios_dir.parent.parent)),
                "id": scenario.id,
                "specialization": scenario.specialization.value,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "turn_count": len(scenario.turns),
                "generated_at": scenario.generated_at.isoformat(),
                "model": scenario.model,
            }
        )
    return entries


def _collect_audio_bundles(audio_dir: Path) -> list[dict[str, object]]:
    if not audio_dir.exists():
        return []
    bundles: list[dict[str, object]] = []
    for sidecar in sorted(audio_dir.glob("*/bundle.json")):
        bundle = ScenarioAudioBundle.model_validate_json(
            sidecar.read_text(encoding="utf-8")
        )
        bundles.append(json.loads(bundle.model_dump_json()))
    return bundles


@app.command()
def export(
    scenarios_dir: Path = typer.Option(
        DEFAULT_SCENARIOS_DIR, "--scenarios-dir", "-d", exists=True
    ),
    audio_dir: Path = typer.Option(DEFAULT_AUDIO_DIR, "--audio-dir"),
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST, "--manifest", "-m"),
) -> None:
    scenarios = _collect_scenarios(scenarios_dir)
    audio = _collect_audio_bundles(audio_dir)

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scenarios": {
            "count": len(scenarios),
            "entries": scenarios,
        },
        "audio": {
            "count": len(audio),
            "entries": audio,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(
        f"[green]Wrote manifest:[/green] {manifest_path} "
        f"(scenarios={len(scenarios)}, audio={len(audio)})"
    )


if __name__ == "__main__":
    app()
