"""End-to-end pipeline: scenario -> TTS -> post-processing -> output.

Phase 1a: `run_scenarios` produces JSON dialogue files.
Phase 1b: `run_audio` consumes those JSONs and produces per-turn + mixed WAVs.

Concurrency note: `run_audio` is intentionally **serial** over scenarios.
XTTSv2 holds a single model on one device (MPS/CUDA/CPU); running turns in
parallel would thrash the accelerator rather than speed anything up. Scenario-
level parallelism would require multiple model instances, which on Apple
Silicon would OOM MPS. Keep it serial; revisit only if we move to batch
inference on a proper GPU.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .scenario_generator.generator import ScenarioGenerator, load_scenarios_config
from .scenario_generator.llm_client import build_default_client
from .scenario_generator.schema import ScenarioModel, Specialization

if TYPE_CHECKING:
    from pathlib import Path

    from .audio_schema import ScenarioAudioBundle
    from .scenario_generator.generator import GenerationReport
    from .tts_engine.xtts_synthesizer import Synthesizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1a — scenario generation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 1b — TTS + dialogue mixing
# ---------------------------------------------------------------------------


@dataclass
class AudioReport:
    bundles: list[ScenarioAudioBundle] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.bundles)


def run_audio(
    scenarios_dir: Path,
    audio_out_dir: Path,
    tts_config_path: Path,
    postprocess_config_path: Path,
    *,
    synthesizer: Synthesizer | None = None,
    scenario_filter: list[str] | None = None,
    limit: int | None = None,
) -> AudioReport:
    """Synthesize audio for every scenario JSON under `scenarios_dir`.

    Writes, for each scenario:
      audio_out_dir/<id[:8]>/turn_NN_<speaker>.wav (at profile sample rate)
      audio_out_dir/<id[:8]>/dialogue.wav            (resampled to 16 kHz)
      audio_out_dir/<id[:8]>/bundle.json             (sidecar for manifest)

    Parameters:
      scenarios_dir: directory containing Phase 1a scenario JSONs.
      audio_out_dir: root for per-scenario audio bundles.
      tts_config_path: path to `configs/tts_profiles.yaml`.
      postprocess_config_path: path to `configs/audio_postprocess.yaml`.
      synthesizer: optional override (tests pass a `_FakeSynthesizer`); if
                   None, a real `XTTSSynthesizer` is constructed.
      scenario_filter: if provided, only scenarios whose id starts with one
                       of these prefixes are synthesized.
      limit: stop after this many successful bundles.
    """
    # Deferred heavy imports — keep this module importable without torch.
    import yaml

    from .audio_schema import AudioArtifact, ScenarioAudioBundle
    from .postprocessing.dialog_mixer import load_mixer_config, mix_turns
    from .postprocessing.noise_injector import inject as inject_noise
    from .tts_engine.audio_utils import (
        duration_seconds,
        normalize_peak,
        resample,
        save_wav,
    )
    from .tts_engine.voice_manager import load_profiles, profile_for
    from .tts_engine.xtts_synthesizer import XTTSSynthesizer

    catalog = load_profiles(tts_config_path)
    mixer_config = load_mixer_config(postprocess_config_path)

    # Output-stage config (final sample rate, normalize flag) comes from the
    # same postprocess yaml.
    post_raw = yaml.safe_load(postprocess_config_path.read_text(encoding="utf-8"))
    out_cfg = post_raw.get("output") or {}
    out_sr = int(out_cfg.get("sample_rate", 16000))
    out_normalize = bool(out_cfg.get("normalize", True))

    synth: Synthesizer = synthesizer if synthesizer is not None else XTTSSynthesizer()

    scenario_paths = sorted(
        p for p in scenarios_dir.glob("*.json") if p.name != "failures.json"
    )
    if scenario_filter:
        scenario_paths = [
            p
            for p in scenario_paths
            if any(prefix in p.name or prefix in p.stem for prefix in scenario_filter)
        ]

    report = AudioReport()
    for path in scenario_paths:
        if limit is not None and report.success_count >= limit:
            break
        try:
            scenario = ScenarioModel.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            log.warning("skipping %s: %s", path.name, exc)
            report.failures.append((path.name, str(exc)))
            continue

        try:
            bundle = _synthesize_scenario(
                scenario,
                audio_out_dir=audio_out_dir,
                synth=synth,
                catalog=catalog,
                mixer_config=mixer_config,
                out_sr=out_sr,
                out_normalize=out_normalize,
                inject_noise=inject_noise,
                _save_wav=save_wav,
                _resample=resample,
                _normalize_peak=normalize_peak,
                _duration_seconds=duration_seconds,
                _mix_turns=mix_turns,
                _audio_artifact_cls=AudioArtifact,
                _scenario_audio_bundle_cls=ScenarioAudioBundle,
                _profile_for=profile_for,
            )
        except Exception as exc:
            log.exception("synthesis failed for %s", scenario.id)
            report.failures.append((scenario.id, str(exc)))
            continue
        report.bundles.append(bundle)
        log.info(
            "synthesized %s (%d turns, %.1fs)",
            scenario.id[:8],
            len(bundle.turns),
            bundle.dialogue_duration_s,
        )

    return report


# The helper takes heavy deps via keyword args so unit tests can patch with
# cheap fakes, and so this module stays parse-able in tests that never load
# torch / TTS.
def _synthesize_scenario(
    scenario: ScenarioModel,
    *,
    audio_out_dir: Path,
    synth: Synthesizer,
    catalog,
    mixer_config,
    out_sr: int,
    out_normalize: bool,
    inject_noise,
    _save_wav,
    _resample,
    _normalize_peak,
    _duration_seconds,
    _mix_turns,
    _audio_artifact_cls,
    _scenario_audio_bundle_cls,
    _profile_for,
) -> ScenarioAudioBundle:
    bundle_dir = audio_out_dir / scenario.id[:8]
    bundle_dir.mkdir(parents=True, exist_ok=True)

    per_turn_audio = []
    artifacts = []
    for idx, turn in enumerate(scenario.turns):
        profile = _profile_for(
            catalog,
            speaker=turn.speaker,
            patient_sex=scenario.patient_sex,
            scenario_id=scenario.id,
        )
        waveform = synth.synthesize(turn.text, profile)
        per_turn_audio.append(waveform)

        wav_path = bundle_dir / f"turn_{idx:02d}_{turn.speaker.value}.wav"
        _save_wav(wav_path, waveform, synth.sample_rate)
        artifacts.append(
            _audio_artifact_cls(
                scenario_id=scenario.id,
                turn_index=idx,
                speaker=turn.speaker.value,
                path=f"audio/{scenario.id[:8]}/{wav_path.name}",
                sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                duration_s=_duration_seconds(waveform, synth.sample_rate),
                sample_rate=synth.sample_rate,
                voice_profile=profile.name,
            )
        )

    # Mix → optional noise (no-op in Phase 1b) → resample to output sample
    # rate → normalize. Seed the mixer with the scenario id so re-runs are
    # byte-identical and the manifest SHA stays stable.
    seed = int.from_bytes(
        hashlib.sha256(scenario.id.encode("utf-8")).digest()[:8], "big", signed=False
    )
    mixed = _mix_turns(
        per_turn_audio,
        sample_rate=synth.sample_rate,
        config=mixer_config,
        seed=seed,
    )
    mixed = inject_noise(mixed, synth.sample_rate)
    if out_sr != synth.sample_rate:
        mixed = _resample(mixed, synth.sample_rate, out_sr)
    if out_normalize:
        mixed = _normalize_peak(mixed)

    dialogue_path = bundle_dir / "dialogue.wav"
    _save_wav(dialogue_path, mixed, out_sr)

    bundle = _scenario_audio_bundle_cls(
        scenario_id=scenario.id,
        dialogue_path=f"audio/{scenario.id[:8]}/dialogue.wav",
        dialogue_sha256=hashlib.sha256(dialogue_path.read_bytes()).hexdigest(),
        dialogue_duration_s=_duration_seconds(mixed, out_sr),
        dialogue_sample_rate=out_sr,
        turns=artifacts,
        synthesized_at=datetime.now().astimezone(),
        tts_model=synth.model_name,
        tts_device=synth.device,
    )
    # Sidecar so export_manifest can rebuild bundles without keeping state
    # between the synthesis and manifest steps.
    (bundle_dir / "bundle.json").write_text(
        bundle.model_dump_json(indent=2), encoding="utf-8"
    )
    return bundle  # type: ignore[no-any-return]


def load_bundle_sidecars(audio_dir: Path) -> list[ScenarioAudioBundle]:
    """Read every `<bundle>/bundle.json` under `audio_dir`."""
    from .audio_schema import ScenarioAudioBundle

    bundles: list[ScenarioAudioBundle] = []
    for sidecar in sorted(audio_dir.glob("*/bundle.json")):
        bundles.append(
            ScenarioAudioBundle.model_validate_json(
                sidecar.read_text(encoding="utf-8")
            )
        )
    return bundles
