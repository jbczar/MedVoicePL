"""Validate integrity of the generated dataset.

Re-parses every JSON file under the scenarios output dir through
`ScenarioModel`. Non-zero exit on any validation failure (CI-friendly).
"""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from dataset_pipeline.scenario_generator.schema import ScenarioModel

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_DIR = Path("dataset_pipeline/output/scenarios")


@app.command()
def validate(
    scenarios_dir: Path = typer.Option(DEFAULT_DIR, "--dir", "-d", exists=True),
) -> None:
    files = sorted(p for p in scenarios_dir.glob("*.json") if p.name != "failures.json")
    if not files:
        console.print(f"[yellow]No scenario files found in {scenarios_dir}[/yellow]")
        raise typer.Exit(code=1)

    failures: list[tuple[Path, str]] = []
    for path in files:
        try:
            ScenarioModel.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            failures.append((path, _format_error(exc)))

    console.rule("[bold]Validation summary")
    console.print(f"Checked: {len(files)}")
    console.print(f"[green]Valid:[/green]   {len(files) - len(failures)}")
    console.print(f"[red]Invalid:[/red] {len(failures)}")
    for path, err in failures:
        console.print(f"  - {path.name}: {err}")
    if failures:
        raise typer.Exit(code=1)


def _format_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg', '')}"
        for e in exc.errors()[:3]
    )


if __name__ == "__main__":
    app()
