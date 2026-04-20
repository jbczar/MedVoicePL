"""Generate dataset manifest (paths, metadata, hashes).

Produces `output/manifest.json` with an entry per scenario file containing
SHA-256, specialization, turn count, model, and timestamps. The manifest
is the integrity contract Phase 1b (TTS) will read.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from dataset_pipeline.scenario_generator.schema import ScenarioModel

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_INPUT = Path("dataset_pipeline/output/scenarios")
DEFAULT_MANIFEST = Path("dataset_pipeline/output/manifest.json")


@app.command()
def export(
    scenarios_dir: Path = typer.Option(DEFAULT_INPUT, "--dir", "-d", exists=True),
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST, "--manifest", "-m"),
) -> None:
    entries: list[dict[str, object]] = []
    files = sorted(p for p in scenarios_dir.glob("*.json") if p.name != "failures.json")
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

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "count": len(entries),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"[green]Wrote manifest:[/green] {manifest_path} ({len(entries)} entries)")


if __name__ == "__main__":
    app()
