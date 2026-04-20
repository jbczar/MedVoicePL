"""Smoke-test the Vertex AI connection end-to-end.

Verifies, in order:
  1. Env vars are set (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, DATASET_LLM_MODEL)
  2. `google-genai` can build a Vertex client (ADC works)
  3. A plain text completion round-trips
  4. Structured output with `response_schema` round-trips (what the real
     pipeline relies on)

Run:
    uv run python dataset_pipeline/scripts/check_vertex.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

console = Console()


class _Ping(BaseModel):
    greeting_pl: str
    model_reports_ok: bool


def _require(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or default
    if not value:
        console.print(f"[red]FAIL[/red] env var {name} is not set")
        sys.exit(1)
    return value


def main() -> int:
    load_dotenv()

    console.rule("[bold]Vertex AI connection check")

    project = _require("GOOGLE_CLOUD_PROJECT")
    location = _require("GOOGLE_CLOUD_LOCATION", "us-central1")
    model = _require("DATASET_LLM_MODEL", "gemini-2.5-pro")
    console.print(f"project    = {project}")
    console.print(f"location   = {location}")
    console.print(f"model      = {model}")

    # Step 1: build client
    try:
        from google import genai
    except ImportError:
        console.print("[red]FAIL[/red] google-genai not installed; run `uv sync --all-packages`")
        return 1

    console.print("\n[cyan]1.[/cyan] Building Vertex client...")
    try:
        client = genai.Client(vertexai=True, project=project, location=location)
    except Exception as exc:
        console.print(f"[red]FAIL[/red] client init: {exc}")
        _hint_auth()
        return 1
    console.print("   [green]OK[/green]")

    # Step 2: plain text round-trip
    console.print("\n[cyan]2.[/cyan] Plain text call...")
    try:
        resp = client.models.generate_content(
            model=model,
            contents="Powiedz krótko po polsku: 'Witaj'.",
        )
        text = (resp.text or "").strip()
    except Exception as exc:
        console.print(f"[red]FAIL[/red] text call: {exc}")
        _hint_common_errors(str(exc))
        return 1
    if not text:
        console.print("[red]FAIL[/red] empty response")
        return 1
    console.print(f"   [green]OK[/green] model replied: {text!r}")

    # Step 3: structured output (matches real pipeline)
    console.print("\n[cyan]3.[/cyan] Structured output with response_schema...")
    try:
        from google.genai import types

        resp = client.models.generate_content(
            model=model,
            contents=(
                "Zwróć JSON: greeting_pl = 'Witaj', model_reports_ok = true."
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_Ping,
            ),
        )
    except Exception as exc:
        console.print(f"[red]FAIL[/red] structured call: {exc}")
        _hint_common_errors(str(exc))
        return 1

    parsed = getattr(resp, "parsed", None)
    if not isinstance(parsed, _Ping):
        try:
            parsed = _Ping.model_validate_json(resp.text or "{}")
        except Exception as exc:
            console.print(f"[red]FAIL[/red] could not parse structured output: {exc}")
            console.print(f"raw text: {resp.text!r}")
            return 1
    console.print(f"   [green]OK[/green] parsed: {parsed!r}")

    console.rule("[bold green]All checks passed")
    console.print(
        "[dim]You can now run: "
        "uv run python dataset_pipeline/scripts/run_generation.py --per-specialization 1[/dim]"
    )
    return 0


def _hint_auth() -> None:
    console.print(
        "[yellow]Hint:[/yellow] did you run `gcloud auth application-default login`?"
    )


def _hint_common_errors(msg: str) -> None:
    lower = msg.lower()
    if "permission_denied" in lower or "has not been used" in lower:
        console.print(
            "[yellow]Hint:[/yellow] enable the API: "
            "`gcloud services enable aiplatform.googleapis.com`"
        )
    elif "billing" in lower:
        console.print(
            "[yellow]Hint:[/yellow] link a billing account in the GCP console."
        )
    elif "not found" in lower and "model" in lower:
        console.print(
            "[yellow]Hint:[/yellow] model not available in this region; "
            "try GOOGLE_CLOUD_LOCATION=us-central1 or europe-west4."
        )
    elif "could not automatically determine credentials" in lower:
        _hint_auth()
    elif "resource_exhausted" in lower or "429" in lower:
        console.print(
            "[yellow]Hint:[/yellow] quota exceeded; wait a moment or request a quota increase."
        )


if __name__ == "__main__":
    sys.exit(main())
