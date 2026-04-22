"""Fetch reference voice samples from Mozilla Common Voice (Polish).

XTTSv2 clones a voice from a short reference WAV. Phase 1b needs four such
clips — `doctor_male`, `doctor_female`, `patient_male`, `patient_female` —
stored under `dataset_pipeline/reference_voices/`.

This script downloads the **Common Voice Scripted Speech 25.0 — Polish**
dataset from the Mozilla Data Collective API, streams the tar.gz archive to
extract `validated.tsv` and selected speaker clips, then converts each clip
to 22050 Hz mono 16-bit PCM WAV.

Download flow:
  1. POST to Mozilla Data Collective `/download` endpoint with
     `MEDVOICEPL_VOICES_API_KEY` to get a presigned download URL.
  2. Stream the tar.gz archive, first extracting `validated.tsv` to memory.
  3. Parse `validated.tsv`, filter by gender/age/votes, pick one clip per
     profile (doctor_male, doctor_female, patient_male, patient_female).
  4. Stream the archive a second time (or use cached download) to extract
     only the 4 chosen MP3 clips.
  5. Convert each MP3 → 22050 Hz mono WAV using torchaudio.
  6. Write `reference_voices/README.md` with provenance metadata.

**Env var**: `MEDVOICEPL_VOICES_API_KEY` — API key for mozilladatacollective.com

**Fallback**: if the download fails or no candidate is found for a profile,
the script prints the WAV spec (22050 Hz mono, 10-20 s, single speaker, clean)
and exits non-zero. Drop manually-sourced WAVs into `reference_voices/` and
re-run — the script is idempotent and skips profiles that already have a
file.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

app = typer.Typer(add_completion=False, help="Fetch XTTSv2 reference voices")
console = Console()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mozilla Data Collective settings
# ---------------------------------------------------------------------------
MDC_DATASET_ID = "cmn27nz69015hmm0720txf781"
MDC_DOWNLOAD_ENDPOINT = (
    f"https://mozilladatacollective.com/api/datasets/{MDC_DATASET_ID}/download"
)

REFERENCE_DIR_DEFAULT = Path("dataset_pipeline/reference_voices")
ARCHIVE_CACHE_DEFAULT = Path("~/.cache/medvoicepl/cv-corpus-25.0-pl.tar.gz")

PROFILES: list[tuple[str, str, str]] = [
    # (profile name, gender column value, target file)
    ("doctor_male", "male_masculine", "doctor_male_01.wav"),
    ("doctor_female", "female_feminine", "doctor_female_01.wav"),
    ("patient_male", "male_masculine", "patient_male_01.wav"),
    ("patient_female", "female_feminine", "patient_female_01.wav"),
]

ACCEPTABLE_AGES = {"thirties", "forties", "fourties", "fifties"}

# Sentence-length window used as a proxy for clip duration. Common Voice
# `validated.tsv` doesn't carry durations (those live in a separate
# `clip_durations.tsv`), but Polish read speech averages ~13 characters per
# second, so the character count of the sentence is a reliable proxy.
#
# XTTSv2 produces noticeably more stable prosody when conditioned on longer
# reference clips — ~10-15 s beats the old 6-10 s window, especially for
# long Polish medical turns. 130-260 chars maps to roughly 10-20 s.
MIN_SENTENCE_CHARS = 130
MAX_SENTENCE_CHARS = 260

# We want clips from *different* speakers for each profile, so we track
# already-used client_ids.
_used_client_ids: set[str] = set()


@dataclass
class ChosenClip:
    profile: str
    client_id: str
    sentence: str
    path: str
    source_url: str


# ---------------------------------------------------------------------------
# Manual-fallback helper
# ---------------------------------------------------------------------------

def _print_manual_spec() -> None:
    console.print(
        "\n[yellow]Provide reference voices manually:[/yellow] drop 4 WAV files "
        "into [bold]dataset_pipeline/reference_voices/[/bold]:"
    )
    for profile, _gender, filename in PROFILES:
        console.print(f"  - [cyan]{filename}[/cyan]   ({profile})")
    console.print(
        "Specs: mono WAV, 22050 Hz (or close), 10-20 s, clean single speaker, "
        "no music/background, speaking Polish."
    )


# ---------------------------------------------------------------------------
# Mozilla Data Collective download helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Read MEDVOICEPL_VOICES_API_KEY from env (loads .env automatically)."""
    load_dotenv()
    key = os.getenv("MEDVOICEPL_VOICES_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MEDVOICEPL_VOICES_API_KEY is not set. "
            "Add it to .env or export it in your shell."
        )
    return key


def _get_presigned_url(api_key: str) -> str:
    """POST to MDC to obtain a presigned download URL for the dataset."""
    log.info("requesting presigned download URL from Mozilla Data Collective")
    resp = requests.post(
        MDC_DOWNLOAD_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    url = data.get("downloadUrl") or data.get("download_url") or data.get("url")
    if not url:
        raise RuntimeError(
            f"MDC response did not contain a download URL. Keys: {list(data.keys())}"
        )
    return url


def _download_archive(download_url: str, dest: Path) -> None:
    """Stream-download the tar.gz archive with a progress bar."""
    log.info("downloading archive to %s", dest)
    resp = requests.get(download_url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0)) or None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading Common Voice PL…", total=total)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB
                f.write(chunk)
                progress.advance(task, len(chunk))


# ---------------------------------------------------------------------------
# Archive extraction helpers
# ---------------------------------------------------------------------------

def _extract_validated_tsv(archive_path: Path) -> list[dict[str, str]]:
    """Open the tar.gz and extract validated.tsv into memory as parsed rows."""
    log.info("extracting validated.tsv from archive")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith("validated.tsv"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                text = f.read().decode("utf-8")
                reader = csv.DictReader(io.StringIO(text), delimiter="\t")
                rows = list(reader)
                log.info("parsed %d rows from validated.tsv", len(rows))
                return rows
    raise RuntimeError("validated.tsv not found in archive")


def _extract_clips(
    archive_path: Path, clip_names: set[str], dest_dir: Path
) -> dict[str, Path]:
    """Extract only the listed clip filenames from the archive."""
    log.info("extracting %d selected clips from archive", len(clip_names))
    found: dict[str, Path] = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            basename = Path(member.name).name
            if basename in clip_names:
                dest_file = dest_dir / basename
                src = tar.extractfile(member)
                if src is None:
                    continue
                dest_file.write_bytes(src.read())
                found[basename] = dest_file
                log.info("  extracted %s", basename)
                if len(found) == len(clip_names):
                    break  # got them all, stop early
    return found


# ---------------------------------------------------------------------------
# Filtering / selection
# ---------------------------------------------------------------------------

def _filter_rows(
    rows: Iterable[dict[str, str]],
    *,
    gender: str,
    min_chars: int = MIN_SENTENCE_CHARS,
    max_chars: int = MAX_SENTENCE_CHARS,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        row_gender = (row.get("gender") or "").strip().lower()
        if row_gender != gender:
            continue
        if row.get("age") and row["age"] not in ACCEPTABLE_AGES:
            continue
        if not row.get("path"):
            continue
        # Duration proxy: sentence length in characters. Clips outside the
        # target window are rejected so we end up with ~10-20 s of reference
        # audio per profile (better XTTSv2 prosody stability than 6-10 s).
        sentence = (row.get("sentence") or "").strip()
        if not (min_chars <= len(sentence) <= max_chars):
            continue
        # Prefer clips from speakers we haven't used yet.
        cid = row.get("client_id", "")
        if cid in _used_client_ids:
            continue
        filtered.append(row)
    # Sort: longer sentences first (within the window), then by net community
    # votes. Length wins because XTTSv2 reference quality scales with duration
    # more than with marginal vote differences among already-validated clips.
    def _sort_key(r: dict[str, str]) -> tuple[int, int]:
        sentence_len = len((r.get("sentence") or "").strip())
        try:
            votes = int(r.get("up_votes") or "0") - int(r.get("down_votes") or "0")
        except ValueError:
            votes = 0
        return (sentence_len, votes)
    filtered.sort(key=_sort_key, reverse=True)
    return filtered


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def _convert_mp3_to_wav(
    mp3_path: Path, wav_path: Path, *, target_sr: int = 22050
) -> None:
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(mp3_path))
    # Downmix to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(wav_path),
        waveform.to(dtype=torch.float32),
        target_sr,
        encoding="PCM_S",
        bits_per_sample=16,
    )


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------

def _write_readme(reference_dir: Path, chosen: list[ChosenClip]) -> None:
    readme = reference_dir / "README.md"
    lines = [
        "# Reference voices",
        "",
        f"Snapshot date: **{date.today().isoformat()}**",
        "",
        "Source: Mozilla Common Voice Corpus 25.0 — Polish (pl).",
        "Downloaded from Mozilla Data Collective. License: **CC0 1.0**.",
        "See <https://mozilladatacollective.com> and <https://commonvoice.mozilla.org/en/datasets>.",
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
            "mono WAV, ~22050 Hz, 10-20 s, clean single speaker, speaking Polish,",
            "no music/background. The fetch script is idempotent and skips",
            "profiles whose WAV already exists.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@app.command()
def fetch(
    reference_dir: Path = typer.Option(
        REFERENCE_DIR_DEFAULT, "--reference-dir", "-d"
    ),
    manual_only: bool = typer.Option(
        False,
        "--manual-only",
        help="Skip the download attempt and just print the spec for manual sourcing.",
    ),
    archive_path: Path = typer.Option(
        ARCHIVE_CACHE_DEFAULT,
        "--archive-path",
        help="Path to cache the downloaded Common Voice tar.gz (reused across runs).",
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
        console.print(
            "[green]All 4 reference voices already present. Nothing to do.[/green]"
        )
        raise typer.Exit(code=0)

    # ------------------------------------------------------------------
    # Step 1: Obtain or locate the archive (cached across runs)
    # ------------------------------------------------------------------
    tar_path = archive_path.expanduser().resolve()
    if tar_path.exists():
        console.print(
            f"[dim]Using cached archive:[/dim] {tar_path} "
            f"({tar_path.stat().st_size / 1e9:.1f} GB)"
        )
    else:
        try:
            api_key = _get_api_key()
            download_url = _get_presigned_url(api_key)
        except Exception as exc:
            console.print(f"[red]Failed to get download URL:[/red] {exc}")
            _print_manual_spec()
            raise typer.Exit(code=1) from exc

        tar_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            _download_archive(download_url, tar_path)
        except Exception as exc:
            # Remove partial download on failure.
            tar_path.unlink(missing_ok=True)
            console.print(f"[red]Download failed:[/red] {exc}")
            _print_manual_spec()
            raise typer.Exit(code=1) from exc

        console.print(
            f"[dim]Archive cached at:[/dim] {tar_path} — "
            "subsequent runs will skip the download."
        )

    # ------------------------------------------------------------------
    # Step 2: Parse validated.tsv from the archive
    # ------------------------------------------------------------------
    try:
        rows = _extract_validated_tsv(tar_path)
    except Exception as exc:
        console.print(f"[red]Failed to read validated.tsv:[/red] {exc}")
        _print_manual_spec()
        raise typer.Exit(code=1) from exc

    # ------------------------------------------------------------------
    # Step 3: Select one clip per pending profile
    # ------------------------------------------------------------------
    selections: list[tuple[str, str, str, dict[str, str]]] = []
    clip_names_needed: set[str] = set()

    for profile, gender, filename in pending:
        plain_gender = gender.split("_")[0]
        # Try progressively looser filters: full gender + target window, then
        # plain gender + target window, then both with a relaxed lower bound.
        # We never drop the upper bound — overly long refs (>25 s) stress the
        # speaker encoder for little gain.
        candidates: list[dict[str, str]] = []
        for g in (gender, plain_gender):
            for lo in (MIN_SENTENCE_CHARS, 90, 60):
                candidates = _filter_rows(
                    rows, gender=g, min_chars=lo, max_chars=MAX_SENTENCE_CHARS
                )
                if candidates:
                    if lo < MIN_SENTENCE_CHARS:
                        log.info(
                            "relaxed min sentence length to %d chars for %s",
                            lo,
                            profile,
                        )
                    break
            if candidates:
                break
        if not candidates:
            console.print(
                f"[red]No candidates for {profile} (gender={gender})[/red]"
            )
            _print_manual_spec()
            raise typer.Exit(code=1)
        pick = candidates[0]
        _used_client_ids.add(pick.get("client_id", ""))
        clip_name = pick["path"]
        clip_names_needed.add(clip_name)
        selections.append((profile, filename, clip_name, pick))
        log.info(
            "selected clip %s for %s (client_id=%s)",
            clip_name,
            profile,
            pick.get("client_id", "")[:16],
        )

    # ------------------------------------------------------------------
    # Step 4: Extract only the needed MP3 clips from the archive
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="cv_clips_") as clips_tmp:
        clips_dir = Path(clips_tmp)
        try:
            extracted = _extract_clips(tar_path, clip_names_needed, clips_dir)
        except Exception as exc:
            console.print(f"[red]Failed to extract clips:[/red] {exc}")
            _print_manual_spec()
            raise typer.Exit(code=1) from exc

        # Check all clips were found
        missing = clip_names_needed - set(extracted.keys())
        if missing:
            console.print(
                f"[red]Could not find clips in archive:[/red] {missing}"
            )
            _print_manual_spec()
            raise typer.Exit(code=1)

        # ---------------------------------------------------------------
        # Step 5: Convert MP3 → WAV
        # ---------------------------------------------------------------
        chosen: list[ChosenClip] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            for profile, filename, clip_name, row in selections:
                task = progress.add_task(f"Converting {profile}…", total=None)
                mp3_path = extracted[clip_name]
                wav_target = reference_dir / filename
                _convert_mp3_to_wav(mp3_path, wav_target)
                chosen.append(
                    ChosenClip(
                        profile=profile,
                        client_id=row.get("client_id", ""),
                        sentence=row.get("sentence", ""),
                        path=str(
                            wav_target.relative_to(reference_dir.parent)
                        ),
                        source_url=(
                            f"https://mozilladatacollective.com/datasets/"
                            f"{MDC_DATASET_ID}"
                        ),
                    )
                )
                progress.update(task, completed=1)

    # ------------------------------------------------------------------
    # Step 6: Write README
    # ------------------------------------------------------------------
    _write_readme(reference_dir, chosen)

    console.print(
        f"[green]Fetched {len(chosen)} reference voices to "
        f"{reference_dir}[/green]"
    )


if __name__ == "__main__":
    app()
