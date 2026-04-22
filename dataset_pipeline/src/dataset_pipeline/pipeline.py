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
    clean_only: bool = False,
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
      clean_only: if True, skip Phase 1b.1 stages (augment/reverb/noise) and
                  only emit `dialogue.wav`. Useful for TTS-quality diffing
                  without waiting on noise library / pedalboard.
    """
    # Deferred heavy imports — keep this module importable without torch.
    import yaml

    from .audio_schema import AudioArtifact, ScenarioAudioBundle
    from .postprocessing.augmentation import (
        load_augment_config,
        load_reverb_config,
        pitch_shift,
        reverb,
        tempo,
    )
    from .postprocessing.dialog_mixer import load_mixer_config, mix_turns
    from .postprocessing.noise_injector import (
        inject as inject_noise,
    )
    from .postprocessing.noise_injector import (
        load_noise_config,
    )
    from .postprocessing.noise_injector import (
        pick_params as pick_noise_params,
    )
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
    augment_config = load_augment_config(postprocess_config_path)
    reverb_config = load_reverb_config(postprocess_config_path)
    noise_config = load_noise_config(postprocess_config_path)
    if clean_only:
        # Disable 1b.1 stages without mutating the loaded configs elsewhere.
        from dataclasses import replace

        augment_config = replace(augment_config, enabled=False)
        reverb_config = replace(reverb_config, enabled=False)
        noise_config = replace(noise_config, enabled=False)

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
                augment_config=augment_config,
                reverb_config=reverb_config,
                noise_config=noise_config,
                out_sr=out_sr,
                out_normalize=out_normalize,
                inject_noise=inject_noise,
                _pick_noise_params=pick_noise_params,
                _pitch_shift=pitch_shift,
                _tempo=tempo,
                _reverb=reverb,
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
    augment_config,
    reverb_config,
    noise_config,
    out_sr: int,
    out_normalize: bool,
    inject_noise,
    _pick_noise_params,
    _pitch_shift,
    _tempo,
    _reverb,
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

    # Per-turn RNG seed so XTTSv2 sampling is reproducible. Uses a mix of
    # the scenario id and turn index so each turn has its own deterministic
    # starting point (seeding once at the scenario level would make every
    # turn draw from the same state and risk correlated outputs).
    scenario_seed_bytes = hashlib.sha256(scenario.id.encode("utf-8")).digest()
    scenario_seed_hi = int.from_bytes(scenario_seed_bytes[:4], "big", signed=False)

    # Per-turn augmentation RNG is decoupled from the TTS-sampling seed so
    # we can change the pitch/tempo ranges without re-rolling XTTS outputs.
    # Using an XOR offset keeps it deterministic per (scenario, turn).
    import random as _random

    per_turn_audio_clean = []  # raw TTS — used for `dialogue.wav`
    per_turn_audio_noisy = []  # post-augment — used for `dialogue_noisy.wav`
    artifacts = []
    for idx, turn in enumerate(scenario.turns):
        profile = _profile_for(
            catalog,
            speaker=turn.speaker,
            patient_sex=scenario.patient_sex,
            scenario_id=scenario.id,
        )
        # 32-bit seed — torch.manual_seed accepts up to 2**64-1 but most
        # downstream ops are uint32-friendly.
        turn_seed = (scenario_seed_hi ^ (idx * 0x9E3779B1)) & 0xFFFFFFFF
        waveform = synth.synthesize(turn.text, profile, seed=turn_seed)
        per_turn_audio_clean.append(waveform)

        # Per-turn pitch + tempo for the noisy variant only. We keep the
        # per-turn WAVs on disk clean (no augment) so they remain usable as
        # ground truth for STT / diarization eval.
        aug_waveform = waveform
        if augment_config.enabled:
            aug_rng = _random.Random(turn_seed ^ 0xA5A5A5A5)
            semitones = aug_rng.uniform(
                augment_config.pitch_semitones_min,
                augment_config.pitch_semitones_max,
            )
            factor = aug_rng.uniform(
                augment_config.tempo_factor_min,
                augment_config.tempo_factor_max,
            )
            aug_waveform = _pitch_shift(aug_waveform, synth.sample_rate, semitones)
            aug_waveform = _tempo(aug_waveform, synth.sample_rate, factor)
        per_turn_audio_noisy.append(aug_waveform)

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

    # ---- Clean mix (ground truth) -------------------------------------
    # Seed the mixer with the scenario id so re-runs are byte-identical and
    # the manifest SHA stays stable.
    mix_seed = int.from_bytes(
        hashlib.sha256(scenario.id.encode("utf-8")).digest()[:8], "big", signed=False
    )
    mixed_clean = _mix_turns(
        per_turn_audio_clean,
        sample_rate=synth.sample_rate,
        config=mixer_config,
        seed=mix_seed,
    )
    if out_sr != synth.sample_rate:
        mixed_clean = _resample(mixed_clean, synth.sample_rate, out_sr)
    if out_normalize:
        mixed_clean = _normalize_peak(mixed_clean)

    dialogue_path = bundle_dir / "dialogue.wav"
    _save_wav(dialogue_path, mixed_clean, out_sr)

    # ---- Noisy variant (Phase 1b.1) -----------------------------------
    # Mix pre-augmented turns (so tempo/pitch vary per speaker), apply
    # room reverb once across the full mix (tail crosses turn boundaries
    # naturally), then mix background noise at a seeded SNR. Uses its own
    # seed derived from scenario.id + "noise" so noise-param changes can't
    # perturb the clean ground truth (different RNG sequence entirely).
    noisy_path_rel: str | None = None
    noisy_sha: str | None = None
    noisy_duration: float | None = None
    noise_snr_db_meta: float | None = None
    noise_clip_meta: str | None = None
    reverb_room_meta: str | None = None

    if augment_config.enabled or reverb_config.enabled or noise_config.enabled:
        noise_seed = int.from_bytes(
            hashlib.sha256((scenario.id + "|noise").encode("utf-8")).digest()[:8],
            "big",
            signed=False,
        )
        fx_rng = _random.Random(noise_seed ^ 0x5EED5EED)

        mixed_noisy = _mix_turns(
            per_turn_audio_noisy,
            sample_rate=synth.sample_rate,
            config=mixer_config,
            seed=mix_seed,
        )

        if reverb_config.enabled and reverb_config.room_sizes:
            rooms = list(reverb_config.room_sizes)
            weights = [r.weight for r in rooms]
            room = fx_rng.choices(rooms, weights=weights, k=1)[0]
            mixed_noisy = _reverb(
                mixed_noisy,
                synth.sample_rate,
                room.size,
                damping=reverb_config.damping,
                wet_level=reverb_config.wet_level,
                dry_level=reverb_config.dry_level,
            )
            reverb_room_meta = room.name

        # Resample BEFORE noise so the noise clip matches the output rate —
        # saves a resample on every noise sample and keeps the SNR measured
        # at the output rate (what the STT model actually sees).
        if out_sr != synth.sample_rate:
            mixed_noisy = _resample(mixed_noisy, synth.sample_rate, out_sr)

        if noise_config.enabled:
            mixed_noisy = inject_noise(
                mixed_noisy,
                out_sr,
                noise_config,
                seed=noise_seed,
            )
            noise_clip_meta, noise_snr_db_meta = _pick_noise_params(
                noise_config, seed=noise_seed
            )

        if out_normalize:
            mixed_noisy = _normalize_peak(mixed_noisy)

        noisy_wav_path = bundle_dir / "dialogue_noisy.wav"
        _save_wav(noisy_wav_path, mixed_noisy, out_sr)
        noisy_path_rel = f"audio/{scenario.id[:8]}/dialogue_noisy.wav"
        noisy_sha = hashlib.sha256(noisy_wav_path.read_bytes()).hexdigest()
        noisy_duration = _duration_seconds(mixed_noisy, out_sr)

    bundle = _scenario_audio_bundle_cls(
        scenario_id=scenario.id,
        dialogue_path=f"audio/{scenario.id[:8]}/dialogue.wav",
        dialogue_sha256=hashlib.sha256(dialogue_path.read_bytes()).hexdigest(),
        dialogue_duration_s=_duration_seconds(mixed_clean, out_sr),
        dialogue_sample_rate=out_sr,
        turns=artifacts,
        synthesized_at=datetime.now().astimezone(),
        tts_model=synth.model_name,
        tts_device=synth.device,
        dialogue_noisy_path=noisy_path_rel,
        dialogue_noisy_sha256=noisy_sha,
        dialogue_noisy_duration_s=noisy_duration,
        noise_snr_db=noise_snr_db_meta,
        noise_clip=noise_clip_meta,
        reverb_room=reverb_room_meta,
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
