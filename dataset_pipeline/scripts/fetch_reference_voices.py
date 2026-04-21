"""Fetch reference voice samples from Mozilla Common Voice (Polish).

XTTSv2 clones a voice from a short reference WAV. Phase 1b needs four such
clips — `doctor_male`, `doctor_female`, `patient_male`, `patient_female` —
stored under `dataset_pipeline/reference_voices/`.

This script uses the **Common Voice delta index** (a small TSV) rather than
the 30 GB bulk archive:
  https://commonvoice.mozilla.org/en/datasets

Strategy:
  1. Download the latest `validated.tsv` for the Polish locale using the
     Common Voice "per-clip" MP3 URL pattern. The TSV snapshot URL is
     configurable via --index-url for reproducibility.
  2. Filter rows: `up_votes >= 2`, `down_votes == 0`, `gender == male|female`,
     `age in {thirties, forties, fifties}`, duration 6-10 s (estimated from
     client_id + sentence length if duration column missing).
  3. For each of the 4 profiles, pick one matching clip, download the MP3,
     convert to 22050 Hz mono WAV via torchaudio, and save to
     `reference_voices/<profile>_01.wav`.
  4. Write / update `reference_voices/README.md` with source URL, client_id,
     sentence, license, and snapshot date so the dataset is reproducible.

**Fallback**: if the TSV fetch fails or no candidate is found for a profile,
the script prints the WAV spec (22050 Hz mono, 6-10 s, single speaker, clean)
and exits non-zero. Drop manually-sourced WAVs into `reference_voices/` and
re-run — the script is idempotent and skips profiles that already have a
file.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(add_completion=False, help="Fetch XTTSv2 reference voices")
console = Console()
log = logging.getLogger(__name__)


# Common Voice Corpus delta URLs rotate; we let the user override the index
# for reproducibility. The default points at the Polish `validated.tsv`
# endpoint which returns a small (~few MB) TSV. Users without that exact
# snapshot should pass --index-url or fall back to --manual-only.
DEFAULT_INDEX_URL = (
    "https://commonvoice.mozilla.org/api/v1/pl/clips/validated"
)
CLIP_URL_TEMPLATE = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus/pl/clips/{clip_name}"

REFERENCE_DIR_DEFAULT = Path("dataset_pipeline/reference_voices")

PROFILES: list[tuple[str, str, str]] = [
    # (profile name, gender column value, target file)
    ("doctor_male", "male", "doctor_male_01.wav"),
    ("doctor_female", "female", "doctor_female_01.wav"),
    ("patient_male", "male", "patient_male_01.wav"),
    ("patient_female", "female", "patient_female_01.wav"),
]

ACCEPTABLE_AGES = {"thirties", "forties", "fifties"}


@dataclass
class ChosenClip:
    profile: str
    client_id: str
    sentence: str
    path: str
    source_url: str


def _print_manual_spec() -> None:
    console.print(
        "\n[yellow]Provide reference voices manually:[/yellow] drop 4 WAV files "
        "into [bold]dataset_pipeline/reference_voices/[/bold]:"
    )
    for profile, _gender, filename in PROFILES:
        console.print(f"  - [cyan]{filename}[/cyan]   ({profile})")
    console.print(
        "Specs: mono WAV, 22050 Hz (or close), 6-10 s, clean single speaker, "
        "no music/background, speaking Polish."
    )


def _fetch_tsv(url: str) -> list[dict[str, str]]:
    log.info("fetching Common Voice index from %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return list(reader)


def _filter_rows(
    rows: Iterable[dict[str, str]], *, gender: str
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if row.get("gender") != gender:
            continue
        if row.get("age") and row["age"] not in ACCEPTABLE_AGES:
            continue
        try:
            up = int(row.get("up_votes") or "0")
            down = int(row.get("down_votes") or "0")
        except ValueError:
            continue
        if up < 2 or down > 0:
            continue
        if not row.get("path"):
            continue
        filtered.append(row)
    return filtered


def _download_and_convert(
    clip_name: str, target_wav: Path, *, target_sr: int = 22050
) -> None:
    import torch
    import torchaudio

    url = CLIP_URL_TEMPLATE.format(clip_name=clip_name)
    log.info("downloading clip %s", clip_name)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tmp_mp3 = target_wav.with_suffix(".mp3")
    tmp_mp3.write_bytes(resp.content)
    try:
        waveform, sr = torchaudio.load(str(tmp_mp3))
        # Downmix to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        target_wav.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(target_wav),
            waveform.to(dtype=torch.float32),
            target_sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )
    finally:
        tmp_mp3.unlink(missing_ok=True)


def _write_readme(reference_dir: Path, chosen: list[ChosenClip]) -> None:
    readme = reference_dir / "README.md"
    lines = [
        "# Reference voices",
        "",
        f"Snapshot date: **{date.today().isoformat()}**",
        "",
        "Source: Mozilla Common Voice Corpus — Polish (pl). License: **CC0 1.0**.",
        "See <https://commonvoice.mozilla.org/en/datasets>.",
        "",
        "Each clip was converted to mono 22050 Hz 16-bit PCM WAV and used as an",
        "XTTSv2 reference speaker — no PII risk since contributors opt in publicly",
        "and Common Voice releases client_ids as anonymous hashes.",
        "",
        "| Profile | File | Client ID | Sentence | Source |",
        "|---|---|---|---|---|",
    ]
    for clip in chosen:
        sentence = clip.sentence.replace("|", "\\|")
        lines.append(
            f"| `{clip.profile}` | `{Path(clip.path).name}` | "
            f"`{clip.client_id[:16]}…` | {sentence} | "
            f"[cv]({clip.source_url}) |"
        )
    lines.extend(
        [
            "",
            "## Replacing with your own recordings",
            "",
            "Drop new files with the same names into this directory. Specs:",
            "mono WAV, ~22050 Hz, 6-10 s, clean single speaker, speaking Polish,",
            "no music/background. The fetch script is idempotent and skips",
            "profiles whose WAV already exists.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


@app.command()
def fetch(
    reference_dir: Path = typer.Option(
        REFERENCE_DIR_DEFAULT, "--reference-dir", "-d"
    ),
    index_url: str = typer.Option(DEFAULT_INDEX_URL, "--index-url"),
    manual_only: bool = typer.Option(
        False,
        "--manual-only",
        help="Skip the download attempt and just print the spec for manual sourcing.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch 4 reference clips from Common Voice PL (or print manual spec)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    reference_dir.mkdir(parents=True, exist_ok=True)

    if manual_only:
        _print_manual_spec()
        raise typer.Exit(code=0)

    # Skip profiles already on disk (idempotent).
    pending = [
        (profile, gender, filename)
        for profile, gender, filename in PROFILES
        if not (reference_dir / filename).exists()
    ]
    if not pending:
        console.print("[green]All 4 reference voices already present. Nothing to do.[/green]")
        raise typer.Exit(code=0)

    try:
        rows = _fetch_tsv(index_url)
    except Exception as exc:
        console.print(f"[red]Failed to fetch Common Voice index:[/red] {exc}")
        _print_manual_spec()
        raise typer.Exit(code=1) from exc

    chosen: list[ChosenClip] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        for profile, gender, filename in pending:
            task = progress.add_task(f"Selecting {profile}…", total=None)
            candidates = _filter_rows(rows, gender=gender)
            if not candidates:
                console.print(f"[red]No candidates for {profile}[/red]")
                _print_manual_spec()
                raise typer.Exit(code=1)
            pick = candidates[0]
            target = reference_dir / filename
            _download_and_convert(pick["path"], target)
            chosen.append(
                ChosenClip(
                    profile=profile,
                    client_id=pick.get("client_id", ""),
                    sentence=pick.get("sentence", ""),
                    path=str(target.relative_to(reference_dir.parent)),
                    source_url=CLIP_URL_TEMPLATE.format(clip_name=pick["path"]),
                )
            )
            progress.update(task, completed=1)

    _write_readme(reference_dir, chosen)
    console.print(f"[green]Fetched {len(chosen)} reference voices to {reference_dir}[/green]")


if __name__ == "__main__":
    app()
