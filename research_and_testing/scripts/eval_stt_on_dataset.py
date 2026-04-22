"""WER / CER benchmark over the Phase 1b (+1b.1) audio bundles.

Transcribes both the clean `dialogue.wav` and the noisy `dialogue_noisy.wav`
from every bundle under `dataset_pipeline/output/audio/`, then compares each
hypothesis to the ground-truth transcript (concatenated `turns[*].text` from
the scenario JSON).

Two backends, selected via `--backend`:

* **faster-whisper** (default) — ctranslate2 runtime. Runs on CPU (int8) on
  Apple Silicon; CUDA / int8_float16 on NVIDIA. Portable baseline.
* **mlx** — mlx-whisper on Apple Silicon, Metal GPU. Typically 5-10x faster
  than faster-whisper CPU int8 on an M-series chip, same `large-v3` quality.

Output:
  research_and_testing/results/stt_eval_<timestamp>.json  — per-scenario
    clean_wer, noisy_wer, delta, with model + backend + device metadata.
  stdout summary table.

This is the feedback loop for tuning `audio_postprocess.yaml`: if noisy
WER stays close to clean WER, the noise config is too mild; if it
explodes, it's too harsh for whatever we expect the deployed STT to
handle.

Install:
    uv pip install faster-whisper        # CPU/CUDA baseline
    uv pip install mlx-whisper           # Apple Silicon GPU (M-series only)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

import typer
from rich.console import Console
from rich.table import Table

from evaluation.stt_eval.wer_calculator import cer, wer

app = typer.Typer(add_completion=False, help="STT WER eval over the synthetic dataset")
console = Console()
log = logging.getLogger(__name__)


DEFAULT_SCENARIOS_DIR = Path("dataset_pipeline/output/scenarios")
DEFAULT_AUDIO_DIR = Path("dataset_pipeline/output/audio")
DEFAULT_RESULTS_DIR = Path("research_and_testing/results")


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


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class Transcriber(Protocol):
    """Minimal contract the eval loop needs from any STT backend."""

    name: str
    device: str

    def transcribe(self, wav_path: Path) -> str: ...


class _FasterWhisperTranscriber:
    """faster-whisper (ctranslate2) backend. CPU int8 by default on Apple."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: uv pip install faster-whisper"
            ) from exc

        resolved = device if device != "auto" else ("cuda" if _cuda_available() else "cpu")
        log.info("loading faster-whisper %s on %s (%s)", model_size, resolved, compute_type)
        self._model = WhisperModel(model_size, device=resolved, compute_type=compute_type)
        self.name = f"faster-whisper:{model_size}"
        self.device = resolved
        self.compute_type = compute_type

    def transcribe(self, wav_path: Path) -> str:
        segments, _info = self._model.transcribe(
            str(wav_path),
            language="pl",
            vad_filter=True,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments)


class _MLXWhisperTranscriber:
    """mlx-whisper backend. Metal GPU on Apple Silicon."""

    def __init__(self, model_repo: str) -> None:
        try:
            import mlx_whisper  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed. Run: uv pip install mlx-whisper"
            ) from exc

        self._model_repo = model_repo
        self.name = f"mlx-whisper:{model_repo}"
        self.device = "mps"  # Metal; kept as "mps" for parity with the torch label
        log.info("using mlx-whisper %s on Metal GPU", model_repo)

    def transcribe(self, wav_path: Path) -> str:
        import mlx_whisper  # type: ignore[import-not-found]

        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=self._model_repo,
            language="pl",
        )
        # mlx_whisper returns {"text": str, "segments": [...], "language": str}
        text = result.get("text", "")
        if isinstance(text, str):
            return text.strip()
        # Fallback: concat segments.
        segs = result.get("segments") or []
        return " ".join((s.get("text") or "").strip() for s in segs)


@app.command()
def run(
    scenarios_dir: Path = typer.Option(DEFAULT_SCENARIOS_DIR, "--scenarios-dir"),
    audio_dir: Path = typer.Option(DEFAULT_AUDIO_DIR, "--audio-dir"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    backend: str = typer.Option(
        "faster-whisper",
        "--backend",
        help="faster-whisper (CPU/CUDA baseline) | mlx (Apple Silicon GPU).",
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
    """Transcribe clean + noisy variants, report per-scenario WER/CER."""
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

    try:
        transcriber: Transcriber = _build_transcriber(
            backend=backend,
            model_size=model_size,
            mlx_model_repo=mlx_model_repo,
            device=device,
            compute_type=compute_type,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    rows: list[dict] = []
    for bundle_path in bundles:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        sid = data["scenario_id"]
        gt = ground_truth.get(sid)
        if gt is None:
            log.warning("no ground truth for %s — skipping", sid)
            continue
        bundle_dir = bundle_path.parent

        clean_hyp = transcriber.transcribe(bundle_dir / "dialogue.wav")
        clean_wer = wer(gt, clean_hyp).rate
        clean_cer = cer(gt, clean_hyp).rate

        noisy_wer = None
        noisy_cer = None
        noisy_path = bundle_dir / "dialogue_noisy.wav"
        if noisy_path.exists():
            noisy_hyp = transcriber.transcribe(noisy_path)
            noisy_wer = wer(gt, noisy_hyp).rate
            noisy_cer = cer(gt, noisy_hyp).rate

        rows.append(
            {
                "scenario_id": sid,
                "clean_wer": clean_wer,
                "clean_cer": clean_cer,
                "noisy_wer": noisy_wer,
                "noisy_cer": noisy_cer,
                "delta_wer": (noisy_wer - clean_wer) if noisy_wer is not None else None,
                "noise_snr_db": data.get("noise_snr_db"),
                "reverb_room": data.get("reverb_room"),
            }
        )
        log.info(
            "%s: clean WER=%.2f%% noisy WER=%s",
            sid[:8],
            100 * clean_wer,
            f"{100 * noisy_wer:.2f}%" if noisy_wer is not None else "n/a",
        )

    # ---- Summary + JSON out ------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"stt_eval_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "backend": backend,
                "model": transcriber.name,
                "device": transcriber.device,
                "compute_type": compute_type if backend == "faster-whisper" else None,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table = Table(title="STT WER / CER (clean vs noisy)")
    table.add_column("scenario")
    table.add_column("clean WER", justify="right")
    table.add_column("noisy WER", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("snr", justify="right")
    for row in rows:
        table.add_row(
            row["scenario_id"][:8],
            f"{100 * row['clean_wer']:.1f}%",
            f"{100 * row['noisy_wer']:.1f}%" if row["noisy_wer"] is not None else "—",
            f"{100 * row['delta_wer']:+.1f}pp" if row["delta_wer"] is not None else "—",
            f"{row['noise_snr_db']:.1f}" if row["noise_snr_db"] is not None else "—",
        )
    console.print(table)

    # Aggregate.
    n = len(rows)
    if n:
        mean_clean = sum(r["clean_wer"] for r in rows) / n
        noisy = [r["noisy_wer"] for r in rows if r["noisy_wer"] is not None]
        mean_noisy = sum(noisy) / len(noisy) if noisy else None
        console.print(f"[bold]Mean clean WER:[/bold] {100 * mean_clean:.2f}%")
        if mean_noisy is not None:
            console.print(f"[bold]Mean noisy WER:[/bold] {100 * mean_noisy:.2f}%")
    console.print(f"[green]Wrote[/green] {out_path}")


def _build_transcriber(
    *,
    backend: str,
    model_size: str,
    mlx_model_repo: str,
    device: str,
    compute_type: str,
) -> Transcriber:
    if backend == "faster-whisper":
        return _FasterWhisperTranscriber(
            model_size=model_size, device=device, compute_type=compute_type
        )
    if backend == "mlx":
        return _MLXWhisperTranscriber(model_repo=mlx_model_repo)
    raise RuntimeError(f"unknown backend: {backend!r} (expected: faster-whisper | mlx)")


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    app()
