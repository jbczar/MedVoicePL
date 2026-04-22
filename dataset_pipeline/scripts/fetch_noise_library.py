"""Fetch clinical ambience noise clips from Freesound.org (Phase 1b.1).

`noise_injector.py` mixes these clips into the clean dialogue mix at a
seeded SNR. We need a small, CC0/CC-BY-licensed library that plausibly
matches what a Polish clinic sounds like:

  - keyboard clicks / typing
  - HVAC hum / ventilation
  - paper rustle / form shuffle
  - distant chatter / waiting-room murmur
  - footsteps on hard floor
  - door open/close

Each clip is downloaded as the OGG preview (low-quality preview is fine —
these are background at 10-25 dB SNR, nobody will hear the compression),
converted to 22050 Hz mono 16-bit PCM WAV, trimmed to 15 s max, and saved
under `dataset_pipeline/noise_library/<category>_<id>.wav`.

**Env var**: `FREESOUND_API_KEY` — get one at https://freesound.org/apiv2/apply/

The script is idempotent: it skips categories that already have the
target number of clips. Re-run to top-up after adding categories.

Fallback: if the API fails, drop any CC0 WAV/OGG into `noise_library/`
and the injector will pick it up. It just expects `.wav` files — use
`ffmpeg -i in.ogg -ar 22050 -ac 1 out.wav` to convert.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import requests
import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(add_completion=False, help="Freesound clinical-ambience fetcher")
console = Console()
log = logging.getLogger(__name__)

FREESOUND_API_BASE = "https://freesound.org/apiv2"
# CC0 (1) and Attribution (0) are the two Freesound "filter" license slugs
# that are safe to redistribute in an academic dataset. We exclude
# NonCommercial and Sampling+ variants.
ALLOWED_LICENSE_FILTER = 'license:("Creative Commons 0" OR "Attribution")'

DEFAULT_LIBRARY_DIR = Path("dataset_pipeline/noise_library")


@dataclass(frozen=True)
class Category:
    slug: str  # used in output filename
    query: str  # Freesound text search
    count: int  # target clips for this category
    min_duration: float = 4.0
    max_duration: float = 30.0


# Curated categories — small total (~20 clips) is deliberate. More variety
# adds little once the SNR bucket range covers 10-25 dB, and the dataset
# stays small enough to commit. Tune counts with care: the injector picks
# uniformly across the whole library, so categories with more clips are
# effectively over-weighted.
CATEGORIES: tuple[Category, ...] = (
    Category("keyboard", "keyboard typing office", count=3),
    Category("hvac", "hvac hum ventilation", count=3),
    Category("paper", "paper rustle shuffle", count=3),
    Category("chatter", "distant chatter background voices office", count=3),
    Category("footsteps", "footsteps hard floor tile", count=2),
    Category("door", "door open close office", count=2),
    Category("clinic", "hospital clinic ambience waiting room", count=4),
)


def _search(query: str, api_key: str, page_size: int = 15) -> list[dict]:
    """Search Freesound for clips matching `query` under allowed licenses."""
    params = {
        "query": query,
        "filter": f"duration:[4.0 TO 30.0] AND {ALLOWED_LICENSE_FILTER}",
        "fields": "id,name,duration,license,previews,username,url",
        "sort": "rating_desc",
        "page_size": page_size,
        "token": api_key,
    }
    r = requests.get(f"{FREESOUND_API_BASE}/search/text/", params=params, timeout=30)
    r.raise_for_status()
    return list(r.json().get("results") or [])


def _download_preview(url: str, dest: Path) -> None:
    """Stream the OGG preview to `dest`. Freesound previews are public —
    no auth needed on the preview URL itself, only the search endpoint.
    """
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                fh.write(chunk)


def _convert_to_wav(src: Path, dst: Path, target_sr: int = 22050, max_s: float = 15.0) -> None:
    """OGG to 22050 Hz mono 16-bit WAV via torchaudio. Trims to `max_s`."""
    import soundfile as sf
    import torchaudio

    waveform, sr = torchaudio.load(str(src))
    # Mono mix-down.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample.
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Trim.
    max_samples = int(max_s * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    arr = waveform.squeeze(0).cpu().numpy().astype("float32")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), arr, target_sr, subtype="PCM_16")


def _existing_count(library_dir: Path, category: Category) -> int:
    return sum(1 for p in library_dir.glob(f"{category.slug}_*.wav"))


@app.command()
def fetch(
    library_dir: Path = typer.Option(DEFAULT_LIBRARY_DIR, "--library-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download CC0/CC-BY ambience clips from Freesound and convert to WAV."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    api_key = os.environ.get("FREESOUND_API_KEY")
    if not api_key:
        console.print(
            "[red]FREESOUND_API_KEY not set.[/red] Get a key at "
            "https://freesound.org/apiv2/apply/ and add it to .env, or drop "
            "CC0 WAVs manually into the library dir.",
        )
        raise typer.Exit(code=1)

    library_dir.mkdir(parents=True, exist_ok=True)
    attribution_lines: list[str] = []

    import tempfile

    for category in CATEGORIES:
        have = _existing_count(library_dir, category)
        if have >= category.count:
            log.info("skipping %s (already %d clips)", category.slug, have)
            continue
        needed = category.count - have
        log.info("fetching %d clips for %s", needed, category.slug)

        try:
            results = _search(category.query, api_key)
        except requests.HTTPError as exc:
            log.error("search failed for %s: %s", category.slug, exc)
            continue

        fetched = 0
        for hit in results:
            if fetched >= needed:
                break
            duration = float(hit.get("duration") or 0.0)
            if not (category.min_duration <= duration <= category.max_duration):
                continue
            previews = hit.get("previews") or {}
            preview_url = (
                previews.get("preview-hq-ogg")
                or previews.get("preview-lq-ogg")
            )
            if not preview_url:
                continue

            clip_id = hit["id"]
            dst_wav = library_dir / f"{category.slug}_{clip_id}.wav"
            if dst_wav.exists():
                fetched += 1
                continue

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _download_preview(preview_url, tmp_path)
                _convert_to_wav(tmp_path, dst_wav)
            except Exception as exc:
                log.warning("failed %s/%s: %s", category.slug, clip_id, exc)
                if dst_wav.exists():
                    dst_wav.unlink()
                continue
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            attribution_lines.append(
                f"- `{dst_wav.name}` — {hit.get('name')!r} by "
                f"{hit.get('username')} ({hit.get('license')}) — {hit.get('url')}"
            )
            fetched += 1
            log.info("fetched %s (%.1fs)", dst_wav.name, duration)

        if fetched < needed:
            log.warning(
                "only fetched %d/%d for %s — re-run after broadening the query",
                fetched, needed, category.slug,
            )

    # Append attribution footer (idempotent — overwrites each run).
    readme = library_dir / "README.md"
    header = (
        "# Noise library (Phase 1b.1)\n\n"
        "Clinical ambience clips from Freesound.org (CC0 / CC-BY). Mixed into\n"
        "the noisy dialogue variant at a stratified SNR by `noise_injector.py`.\n\n"
        "Clips on disk:\n\n"
    )
    existing_wavs = sorted(p.name for p in library_dir.glob("*.wav"))
    wav_lines = [f"- `{name}`" for name in existing_wavs]
    readme.write_text(
        header + "\n".join(wav_lines) + "\n\n## Attributions (this run)\n\n"
        + ("\n".join(attribution_lines) if attribution_lines else "_(no new downloads)_")
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Library:[/green] {len(existing_wavs)} clips in {library_dir}")


if __name__ == "__main__":
    app()
