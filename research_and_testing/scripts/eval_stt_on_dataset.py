"""WER / CER / DER benchmark over the Phase 1b (+1b.1) audio bundles.

Transcribes both the clean `dialogue.wav` and the noisy `dialogue_noisy.wav`
from every bundle under `dataset_pipeline/output/audio/`, then compares each
hypothesis to the ground-truth transcript (concatenated `turns[*].text` from
the scenario JSON). For the `on-premise` backend it additionally runs
diarization and scores DER against the per-turn WAV layout in the bundle.

Three backends, selected via `--backend`:

* **faster-whisper** — ctranslate2 runtime. Runs on CPU (int8) on Apple
  Silicon; CUDA / int8_float16 on NVIDIA. Portable baseline.
* **mlx** — mlx-whisper on Apple Silicon, Metal GPU. Typically 5-10x faster
  than faster-whisper CPU int8 on an M-series chip, same `large-v3` quality.
* **on-premise** — the Phase 2a pipeline (`on_premise.stt.pipeline`). Runs
  Whisper (backend chosen by `WHISPER_BACKEND` env) + pyannote diarization,
  emits `Transcript` objects with speaker-labeled segments, and scores both
  WER (vs. concatenated ground truth) and DER (vs. the per-turn WAV layout).

The bare `faster-whisper` / `mlx` backends import `on_premise.stt.whisper_engine`,
so the transcription code path is the same as Phase 2a's deploy path —
we benchmark exactly what we ship.

Output:
  research_and_testing/results/stt_eval_<timestamp>.json  — per-scenario
    clean_wer, noisy_wer, delta, der (when available), with model + backend
    + device metadata.
  stdout summary table.

Install:
    uv pip install faster-whisper        # CPU/CUDA baseline
    uv pip install mlx-whisper           # Apple Silicon GPU (M-series only)
    # on-premise additionally needs: pyannote.audio + HUGGINGFACE_TOKEN
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from evaluation.stt_eval.der_calculator import (
    DERResult,
    diarization_error_rate,
    reference_from_turn_durations,
)
from evaluation.stt_eval.wer_calculator import cer, wer
from on_premise.stt.whisper_engine import (
    FasterWhisperTranscriber,
    MLXWhisperTranscriber,
    Transcriber,
)

app = typer.Typer(add_completion=False, help="STT WER/DER eval over the synthetic dataset")
console = Console()
log = logging.getLogger(__name__)


DEFAULT_SCENARIOS_DIR = Path("dataset_pipeline/output/scenarios")
DEFAULT_AUDIO_DIR = Path("dataset_pipeline/output/audio")
DEFAULT_RESULTS_DIR = Path("research_and_testing/results")

# Mixer default — matches `dataset_pipeline/configs/audio_postprocess.yaml`.
# Used to reconstruct per-turn offsets in `dialogue.wav` for DER reference.
_DEFAULT_CROSSFADE_S = 0.200


def _load_ground_truth(scenarios_dir: Path) -> dict[str, str]:
    """Map scenario_id -> concatenated turn text (ground-truth transcript)."""
    gt: dict[str, str] = {}
    for path in scenarios_dir.glob("*.json"):
        if path.name == "failures.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = data.get("id")
        turns = data.get("turns") or []
        if not sid or not turns:
            continue
        gt[sid] = " ".join(t["text"] for t in turns)
    return gt


def _estimate_pause_s(bundle: dict) -> float:
    """Back out the per-turn pause used by the mixer from bundle metadata.

    The mixer produces ``total = sum(turns) + (n-1)*pause - (n-1)*crossfade``,
    so ``pause = (total - sum(turns)) / (n-1) + crossfade``. If we can't
    back it out (single-turn scenarios, missing fields), fall back to the
    mean of the configured 300-1500 ms range (~0.9 s).
    """
    turns = bundle.get("turns") or []
    if len(turns) < 2:
        return 0.9
    total = float(bundle.get("dialogue_duration_s") or 0.0)
    sum_turns = sum(float(t.get("duration_s") or 0.0) for t in turns)
    if total <= 0 or sum_turns <= 0:
        return 0.9
    pause = (total - sum_turns) / (len(turns) - 1) + _DEFAULT_CROSSFADE_S
    # Clamp to the configured range — protects DER from outlier bundles.
    return max(0.0, min(pause, 2.0))


def _reference_timeline(bundle: dict) -> list[tuple[float, float, str]]:
    """Build the DER reference timeline from a bundle's turn list."""
    turns = bundle.get("turns") or []
    durations = [float(t.get("duration_s") or 0.0) for t in turns]
    speakers = [str(t.get("speaker") or "") for t in turns]
    pause = _estimate_pause_s(bundle)
    # Cast via list(...) to strip the Any that mypy infers across the
    # package boundary (evaluation/ isn't in mypy's resolved path list).
    segs: list[tuple[float, float, str]] = list(
        reference_from_turn_durations(
            durations, speakers, pause_between_turns_s=pause
        )
    )
    return segs


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


def _build_bare_transcriber(
    *,
    backend: str,
    model_size: str,
    mlx_model_repo: str,
    device: str,
    compute_type: str,
) -> Transcriber:
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(
            model_size=model_size, device=device, compute_type=compute_type
        )
    if backend == "mlx":
        return MLXWhisperTranscriber(model_repo=mlx_model_repo)
    raise RuntimeError(
        f"unknown backend: {backend!r} (expected: faster-whisper | mlx | on-premise)"
    )


@app.command()
def run(
    scenarios_dir: Path = typer.Option(DEFAULT_SCENARIOS_DIR, "--scenarios-dir"),
    audio_dir: Path = typer.Option(DEFAULT_AUDIO_DIR, "--audio-dir"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    backend: str = typer.Option(
        "mlx",
        "--backend",
        help=(
            "faster-whisper (portable baseline) | mlx (Apple Silicon GPU) | "
            "on-premise (Phase 2a pipeline: Whisper + pyannote + DER)."
        ),
    ),
    model_size: str = typer.Option(
        "large-v3",
        "--model-size",
        help="faster-whisper size (tiny/base/small/medium/large-v3). Ignored for mlx.",
    ),
    mlx_model_repo: str = typer.Option(
        "mlx-community/whisper-large-v3-mlx",
        "--mlx-model-repo",
        help="HF repo id for mlx-whisper weights. Only used when --backend=mlx.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="cpu | cuda | auto. Only used by faster-whisper (mlx is always Metal).",
    ),
    compute_type: str = typer.Option(
        "int8",
        "--compute-type",
        help="int8 (CPU, Apple Silicon), int8_float16 (CUDA), float16 (GPU). faster-whisper only.",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
) -> None:
    """Transcribe clean + noisy variants, report per-scenario WER/CER/DER."""
    # Load `.env` so `HUGGINGFACE_TOKEN` and `WHISPER_*` vars are visible to
    # the on-premise pipeline's `build_diarizer` / `build_transcriber`. The
    # other two CLIs (`check_stt_models.py`, `run_transcription.py`) also
    # call `load_dotenv()` — keeping the three entrypoints symmetric means
    # a token that works for the preflight also works for the eval loop.
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ground_truth = _load_ground_truth(scenarios_dir)
    if not ground_truth:
        console.print(f"[red]No scenarios found in {scenarios_dir}[/red]")
        raise typer.Exit(code=1)

    bundles = sorted(audio_dir.glob("*/bundle.json"))
    if not bundles:
        console.print(f"[red]No audio bundles found in {audio_dir}[/red]")
        raise typer.Exit(code=1)
    if limit:
        bundles = bundles[:limit]

    # Runner is closure-based so on-premise (transcriber+diarizer) and bare
    # backends share the per-bundle loop without a bunch of branching.
    runner = _make_runner(
        backend=backend,
        model_size=model_size,
        mlx_model_repo=mlx_model_repo,
        device=device,
        compute_type=compute_type,
    )

    rows: list[dict] = []
    for bundle_path in bundles:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        sid = data["scenario_id"]
        gt = ground_truth.get(sid)
        if gt is None:
            log.warning("no ground truth for %s — skipping", sid)
            continue
        bundle_dir = bundle_path.parent

        clean_hyp, clean_der = runner(bundle_dir / "dialogue.wav", data)
        clean_wer_rate = wer(gt, clean_hyp).rate
        clean_cer_rate = cer(gt, clean_hyp).rate

        noisy_wer_rate: float | None = None
        noisy_cer_rate: float | None = None
        noisy_der: DERResult | None = None
        noisy_path = bundle_dir / "dialogue_noisy.wav"
        if noisy_path.exists():
            noisy_hyp, noisy_der = runner(noisy_path, data)
            noisy_wer_rate = wer(gt, noisy_hyp).rate
            noisy_cer_rate = cer(gt, noisy_hyp).rate

        rows.append(
            {
                "scenario_id": sid,
                "clean_wer": clean_wer_rate,
                "clean_cer": clean_cer_rate,
                "noisy_wer": noisy_wer_rate,
                "noisy_cer": noisy_cer_rate,
                "delta_wer": (
                    (noisy_wer_rate - clean_wer_rate) if noisy_wer_rate is not None else None
                ),
                "clean_der": clean_der.rate if clean_der is not None else None,
                "noisy_der": noisy_der.rate if noisy_der is not None else None,
                "noise_snr_db": data.get("noise_snr_db"),
                "reverb_room": data.get("reverb_room"),
            }
        )
        log.info(
            "%s: clean WER=%.2f%% noisy WER=%s%s",
            sid[:8],
            100 * clean_wer_rate,
            (
                f"{100 * noisy_wer_rate:.2f}%"
                if noisy_wer_rate is not None
                else "n/a"
            ),
            f" clean DER={100 * clean_der.rate:.2f}%" if clean_der is not None else "",
        )

    # ---- Summary + JSON out ------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"stt_eval_{stamp}.json"

    # Extract the model/device label from the runner's pre-built description.
    meta = runner.meta  # type: ignore[attr-defined]
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "backend": backend,
                "model": meta["model"],
                "device": meta["device"],
                "compute_type": meta.get("compute_type"),
                "diarization_model": meta.get("diarization_model"),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_table(rows, backend=backend)

    # Aggregates.
    n = len(rows)
    if n:
        mean_clean = sum(r["clean_wer"] for r in rows) / n
        noisy_wers = [r["noisy_wer"] for r in rows if r["noisy_wer"] is not None]
        mean_noisy = sum(noisy_wers) / len(noisy_wers) if noisy_wers else None
        console.print(f"[bold]Mean clean WER:[/bold] {100 * mean_clean:.2f}%")
        if mean_noisy is not None:
            console.print(f"[bold]Mean noisy WER:[/bold] {100 * mean_noisy:.2f}%")
        clean_ders = [r["clean_der"] for r in rows if r["clean_der"] is not None]
        if clean_ders:
            console.print(
                f"[bold]Mean clean DER:[/bold] {100 * sum(clean_ders) / len(clean_ders):.2f}%"
            )
        noisy_ders = [r["noisy_der"] for r in rows if r["noisy_der"] is not None]
        if noisy_ders:
            console.print(
                f"[bold]Mean noisy DER:[/bold] {100 * sum(noisy_ders) / len(noisy_ders):.2f}%"
            )
    console.print(f"[green]Wrote[/green] {out_path}")


# ---------------------------------------------------------------------------
# Runner factory — unifies on-premise and bare backends behind one call signature
# ---------------------------------------------------------------------------


class _Runner:
    """Callable (wav_path, bundle_dict) -> (hypothesis_text, der_or_none)."""

    def __init__(self, meta: dict[str, str | None]) -> None:
        self.meta = meta

    def __call__(self, wav_path: Path, bundle: dict) -> tuple[str, DERResult | None]:
        raise NotImplementedError


class _BareRunner(_Runner):
    """faster-whisper or mlx: transcription only, no diarization, no DER."""

    def __init__(self, transcriber: Transcriber, meta: dict[str, str | None]) -> None:
        super().__init__(meta)
        self._t = transcriber

    def __call__(self, wav_path: Path, bundle: dict) -> tuple[str, DERResult | None]:
        raw = self._t.transcribe(wav_path)
        return raw.text, None


class _OnPremiseRunner(_Runner):
    """Full Phase 2a pipeline: Whisper + pyannote + Transcript + DER."""

    def __init__(self, meta: dict[str, str | None]) -> None:
        from on_premise.stt.diarization import build_diarizer
        from on_premise.stt.whisper_engine import build_transcriber

        # Load config + env; the factories understand templating.
        self._transcriber: Transcriber = build_transcriber({}, os.environ)
        self._diarizer = build_diarizer(config=None, env=os.environ)
        meta["model"] = self._transcriber.name
        meta["device"] = self._transcriber.device
        meta["compute_type"] = getattr(self._transcriber, "compute_type", None)
        meta["diarization_model"] = self._diarizer.name
        super().__init__(meta)

    def __call__(self, wav_path: Path, bundle: dict) -> tuple[str, DERResult | None]:
        from on_premise.stt.pipeline import transcribe_and_diarize

        transcript = transcribe_and_diarize(
            wav_path,
            transcriber=self._transcriber,
            diarizer=self._diarizer,
            scenario_id=bundle.get("scenario_id"),
        )
        ref = _reference_timeline(bundle)
        hyp = [(seg.start_s, seg.end_s, seg.speaker) for seg in transcript.segments]
        der = diarization_error_rate(ref, hyp)
        return transcript.text, der


def _make_runner(
    *,
    backend: str,
    model_size: str,
    mlx_model_repo: str,
    device: str,
    compute_type: str,
) -> _Runner:
    if backend == "on-premise":
        return _OnPremiseRunner(meta={})
    try:
        transcriber = _build_bare_transcriber(
            backend=backend,
            model_size=model_size,
            mlx_model_repo=mlx_model_repo,
            device=device,
            compute_type=compute_type,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    meta: dict[str, str | None] = {
        "model": transcriber.name,
        "device": transcriber.device,
        "compute_type": compute_type if backend == "faster-whisper" else None,
        "diarization_model": None,
    }
    return _BareRunner(transcriber, meta)


def _print_table(rows: list[dict], *, backend: str) -> None:
    with_der = backend == "on-premise"
    title = "STT WER / CER" + (" / DER" if with_der else "") + " (clean vs noisy)"
    table = Table(title=title)
    table.add_column("scenario")
    table.add_column("clean WER", justify="right")
    table.add_column("noisy WER", justify="right")
    table.add_column("delta", justify="right")
    if with_der:
        table.add_column("clean DER", justify="right")
        table.add_column("noisy DER", justify="right")
    table.add_column("snr", justify="right")
    for row in rows:
        cells = [
            row["scenario_id"][:8],
            f"{100 * row['clean_wer']:.1f}%",
            f"{100 * row['noisy_wer']:.1f}%" if row["noisy_wer"] is not None else "-",
            f"{100 * row['delta_wer']:+.1f}pp" if row["delta_wer"] is not None else "-",
        ]
        if with_der:
            cells.append(
                f"{100 * row['clean_der']:.1f}%" if row["clean_der"] is not None else "-"
            )
            cells.append(
                f"{100 * row['noisy_der']:.1f}%" if row["noisy_der"] is not None else "-"
            )
        cells.append(
            f"{row['noise_snr_db']:.1f}" if row["noise_snr_db"] is not None else "-"
        )
        table.add_row(*cells)
    console.print(table)


if __name__ == "__main__":
    app()
