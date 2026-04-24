# CLAUDE.md — Kontekst projektu MedVoicePL

## Przegląd projektu

**MedVoicePL** — Multimodalny, wieloagentowy system hybrydowy do wspomagania decyzji klinicznych (CDSS).
Projekt magisterski. System automatyzuje tworzenie notatek medycznych z nagrań wizyt lekarskich, anonimizuje dane pacjentów (RODO) i sugeruje leki z refundacją NFZ. Ostateczna decyzja kliniczna zawsze należy do lekarza.

## Architektura monorepo

Repozytorium jest podzielone na trzy niezależne filary:

### 1. `dataset_pipeline/` — Generowanie syntetycznego datasetu
- Generacja scenariuszy wizyt lekarskich (JSON) przez API LLM
- Synteza mowy polskiej przez lokalny model XTTSv2 (Coqui TTS)
- Postprodukcja audio: szumy tła, crossfade dialogów, augmentacje
- Pipeline: scenariusz → synteza per-wypowiedź → miksowanie → augmentacja

### 2. `application/` — System CDSS
Podział na cztery moduły:
- **`on_premise/`** — Strefa zaufania wewnętrzna. Surowe audio NIGDY nie opuszcza tego modułu.
  - STT: MLX Whisper (primary, Apple Silicon) / faster-whisper (portable fallback)
  - Diaryzacja: pyannote.audio 3.1 (2 speakers: doctor + patient)
  - `Transcript` contract (un-anonimizowany) żyje w `on_premise.stt.schema` i nie wycieka do `shared/`
  - SLM: lokalny model do strukturyzacji SOAP (Phase 2b)
  - Anonimizacja: Microsoft Presidio + custom PL recognizery (Phase 2c)
  - `trust_gate.py` wymusza programistycznie granicę zaufania (Phase 2c)
- **`cloud/`** — Operuje wyłącznie na danych zanonimizowanych.
  - LLM: analiza kliniczna przez chmurowe modele
  - RAG: wyszukiwanie semantyczne w bazie wiedzy
  - NFZ: przeszukiwanie bazy refundacji, dopasowanie leków
- **`gateway/`** — API Gateway, orkiestrator międzymodułowy (on-premise → cloud)
- **`frontend/`** — Pulpit lekarza (Next.js). Nagrywanie, notatka SOAP, sugestie do zatwierdzenia.
- **`shared/`** — Współdzielone schematy Pydantic (SOAP, DrugSuggestion, AnonymizedPayload)

### 3. `research_and_testing/` — Badania i ewaluacja
- Złote przypadki testowe (ground truth)
- Ewaluacja: STT (WER/CER), SLM (jakość SOAP), RAG (trafność sugestii), anonimizacja (wycieki PII)
- Benchmarki wydajnościowe, analiza oszczędności czasu lekarzy
- Notatniki Jupyter do wizualizacji wyników

## Stos technologiczny

| Warstwa | Technologia |
|---|---|
| Orkiestracja agentów | LangGraph |
| Backend API | FastAPI + uvicorn |
| Frontend | Next.js (TypeScript) |
| STT | MLX Whisper (Apple Silicon) / faster-whisper (portable) |
| Diaryzacja | pyannote.audio 3.1 |
| SLM (lokalne) | Transformers (quantized) |
| Anonimizacja | Microsoft Presidio |
| LLM (chmura) | OpenAI / Anthropic / Google (`google-genai` używane też przez dataset_pipeline) |
| RAG / Vector DB | ChromaDB lub Qdrant |
| TTS (dataset) | XTTSv2 (Coqui TTS) |
| Walidacja danych | Pydantic v2 |
| Konteneryzacja | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Linting | ruff + mypy |
| Logowanie | structlog |

## Konwencje kodu

- **Język kodu**: Python 3.11+, TypeScript (frontend)
- **Język komentarzy/docstringów**: angielski
- **Język dokumentacji**: polski (praca magisterska), angielski (README techniczne)
- **Schematy danych**: Pydantic v2 — wszystkie kontrakty między modułami
- **Konfiguracja**: pliki YAML w `configs/` per moduł
- **Zmienne środowiskowe**: `.env` (nigdy commitowany), `.env.example` jako wzorzec
- **Testy**: pytest, umieszczane w `tests/` per moduł

## Zasady bezpieczeństwa (Trust Boundary)

1. Surowe audio nie opuszcza `on_premise/`
2. Niezanonimizowana transkrypcja nie opuszcza `on_premise/`
3. `trust_gate.py` waliduje każdy payload wychodzący z on-premise
4. Kontrakt granicy: `shared/schemas/anonymized.py` (`AnonymizedPayload`)
5. Każda operacja anonimizacji jest logowana (`audit_logger.py`)
6. Moduł `cloud/` nigdy nie otrzymuje danych PII
7. **Dataset syntetyczny** — żadne dane PII w `dataset_pipeline/output/` nie są prawdziwe; generowane PESEL-e, imiona, adresy są fikcyjne i służą wyłącznie jako cele treningowe dla modułu anonimizacji.

## Status implementacji

| Faza | Zakres | Status |
|---|---|---|
| Scaffolding | Struktura monorepo, configi, stuby | ✅ ukończone (2026-04-14) |
| Phase 1a | `dataset_pipeline`: generacja scenariuszy (JSON) przez Gemini | ✅ ukończone (2026-04-20) — 8 scenariuszy w `output/scenarios/` (1-2 per specjalizacja × 5) |
| Phase 1b | `dataset_pipeline`: XTTSv2 + postprocessing audio (clean) | ✅ ukończone (2026-04-22) — smoke test OK (13 turns, 185.5 s), determinizm SHA zweryfikowany |
| Phase 1b.1 | `dataset_pipeline`: noise injection + augmentacje (tempo/pitch/reverb) + WER eval loop | ✅ ukończone (2026-04-22) — 11 klipów Freesound, clean/noisy renderuje się bez błędów, pierwszy WER 9.1%→11.2% przy 15 dB SNR |
| Phase 2a | `application/on_premise`: STT (MLX/faster-whisper) + diaryzacja (pyannote) + `Transcript` contract + DER eval | ✅ ukończone (2026-04-24) — 8 scenariuszy, mean clean WER 3.67% / noisy 4.68%, mean clean DER 32.96% / noisy 43.36% |
| Phase 2b | `application/on_premise`: SLM engine + SOAP extraction + prompts | 🔄 w toku (2026-04-24) — kod + testy gotowe (85 zielonych, lint + mypy czyste), czeka na pierwszy smoke Bielik-11B |
| Phase 2c | `application/on_premise`: Presidio + PL recognizers + `trust_gate` + audit logger + FastAPI | ⏳ kolejne |
| Phase 3 | `application/cloud` + `gateway` + `frontend` | ⏳ kolejne |
| Phase 4 | `research_and_testing`: ewaluacja E2E | ⏳ kolejne |

### Phase 1a — decyzje projektowe

- **LLM provider**: Google Gemini przez **Vertex AI** (konsumuje kredyty $300 z GCP). SDK: `google-genai` w trybie `vertexai=True`. Uwierzytelnianie przez Application Default Credentials (ADC): `gcloud auth application-default login` **lub** `GOOGLE_APPLICATION_CREDENTIALS` wskazujący na service-account JSON.
- **Wymagane env**: `GOOGLE_CLOUD_PROJECT` (ID projektu GCP), `GOOGLE_CLOUD_LOCATION` (np. `us-central1`, domyślnie `us-central1`). Klucz `GOOGLE_API_KEY` (AI Studio) zostaje w `.env` na później dla modułu `cloud/`, ale **nie jest używany przez dataset pipeline**.
- **Setup dla nowego środowiska**:
  1. `gcloud auth application-default login`
  2. Włącz Vertex AI API: `gcloud services enable aiplatform.googleapis.com`
  3. Ustaw `GOOGLE_CLOUD_PROJECT` w `.env`
- **Skala MVP**: 2 scenariusze × 5 specjalizacji = 10 scenariuszy na pierwszy przebieg.
- **Kontrakt danych**: `dataset_pipeline/src/dataset_pipeline/scenario_generator/schema.py` (`ScenarioModel`) — stabilny interfejs dla Phase 1b (TTS) i `research_and_testing`.
- **Enforcement JSON**: Gemini `response_schema=ScenarioModel` (structured output) zamiast parsowania tekstu.
- **Współbieżność**: `ThreadPoolExecutor`, domyślnie 4 wątki (Vertex AI ma wyższe limity niż AI Studio, można zwiększyć). Konfigurowalne przez `DATASET_LLM_MAX_CONCURRENCY`.
- **Retry strategy**: tenacity (exp backoff, 4 próby) dla błędów przejściowych 5xx/429; osobny mechanizm retry z "correction hint" przy błędach walidacji Pydantic (2 próby).
- **Few-shot examples**: pięć ręcznie napisanych scenariuszy (po jednym na specjalizację) w `prompts.py` — nie zastępować outputem modelu, żeby uniknąć drift.

### Phase 1a — kluczowe pliki

| Plik | Rola |
|---|---|
| `dataset_pipeline/src/dataset_pipeline/scenario_generator/schema.py` | `ScenarioModel`, `DialogueTurn`, `Specialization`, `SpeakerRole` — kontrakt Pydantic |
| `dataset_pipeline/src/dataset_pipeline/scenario_generator/prompts.py` | `SYSTEM_PROMPT_PL`, `FEW_SHOT_EXAMPLES`, `render_user_prompt` |
| `dataset_pipeline/src/dataset_pipeline/scenario_generator/llm_client.py` | `GeminiClient`, `LLMClient` Protocol, `TransientLLMError`/`PermanentLLMError` |
| `dataset_pipeline/src/dataset_pipeline/scenario_generator/generator.py` | `ScenarioGenerator` (batch + retry + failures.json), `load_scenarios_config` |
| `dataset_pipeline/src/dataset_pipeline/pipeline.py` | `run_scenarios()` — entrypoint do chainowania z TTS w Phase 1b |
| `dataset_pipeline/scripts/run_generation.py` | Typer CLI: generacja |
| `dataset_pipeline/scripts/validate_dataset.py` | Typer CLI: ponowna walidacja plików JSON |
| `dataset_pipeline/scripts/export_manifest.py` | Typer CLI: `manifest.json` z SHA-256 |
| `dataset_pipeline/tests/` | 25 testów (schema, prompts, generator z fake clientem, validator CLI) |

### Phase 1a — uruchomienie

```bash
# 1. Uwierzytelnienie Vertex AI (raz per maszyna)
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com --project=<YOUR_PROJECT>

# 2. Konfiguracja env
cp .env.example .env   # uzupełnić GOOGLE_CLOUD_PROJECT

# 3. Sync zależności + generacja
uv sync --all-packages
uv run python dataset_pipeline/scripts/run_generation.py --per-specialization 2
uv run python dataset_pipeline/scripts/validate_dataset.py
uv run python dataset_pipeline/scripts/export_manifest.py
```

Output: `dataset_pipeline/output/scenarios/*.json` + `output/manifest.json`. Katalog `output/` jest gitignorowany.

### Phase 1b — decyzje projektowe

- **Próbki referencyjne głosów**: **Common Voice Scripted Speech 25.0 — Polish** pobierany z **Mozilla Data Collective** (Bearer auth via `MEDVOICEPL_VOICES_API_KEY`, dataset id `cmn27nz69015hmm0720txf781`). Oryginalny publiczny endpoint `voice.mozilla.org/api/v1` wymaga sesji, więc MDC jest jedyną stabilną ścieżką. Cztery profile: `doctor_male`, `doctor_female`, `patient_male`, `patient_female`. Skrypt `scripts/fetch_reference_voices.py` jest idempotentny; filtruje długość zdania (130–260 znaków ≈ 10–20 s — dłuższe refy stabilizują prozodię XTTSv2 bardziej niż 6–10 s) z ladderem rozluźnień (90, potem 60 znaków) gdy płeć/wiek ogranicza pulę. Fallback manualny: drop WAV (mono, 22050 Hz, 10–20 s) do `reference_voices/` i re-run.
- **XTTSv2 stabilność sampli**: `temperature=0.65` (poniżej domyślnego 0.75), `sound_norm_refs=True` (auto-normalizacja refu przed conditioningiem — kluczowe przy cichych/szeptanych klipach Common Voice), `split_sentences=True` (zdanie-po-zdaniu → prozodia nie degraduje się na długich polskich turach). Wszystkie trzy eksponowane jako constructor params na `XTTSSynthesizer`.
- **Backend TTS**: XTTSv2 przez community fork `coqui-tts` (oryginalny Coqui został zarchiwizowany w 2024). Model ~2 GB cache'owany pod `TTS_MODEL_CACHE` (domyślnie `~/.cache/medvoicepl/xtts`, poza repo).
- **Sprzęt deweloperskie**: MacBook Pro M5, 24 GB unified memory, 10-core GPU. Liczby obserwowane na tym sprzęcie są referencyjne w całym projekcie — jak jakiś krok trwa znacząco dłużej u innego dev'a, to indykacja że coś nie działa. Autodetekcja przez `TTS_DEVICE=auto` → **MPS** (fallback CPU, CUDA jeśli dostępne). XTTSv2 na MPS: RTF ~0.4–0.5 → 3-min dialog renderuje się w ~2:30. Pełen 8-scenariuszowy render ≈ 25 min. 24 GB RAM nigdy nie OOM-uje przy 2 GB XTTS + 1 GB faster-whisper jednocześnie.
- **Współbieżność**: **serial** over scenarios. XTTSv2 trzyma pojedynczy model na jednym urządzeniu; paralelizm would thrash MPS. Revisit dopiero przy batch inference na prawdziwym GPU.
- **Noise**: odłożone do Phase 1b.1 — `noise_injector.py` jest stubem, `augmentation.py` ma tylko `normalize()`. Pozwala zwalidować jakość TTS niezależnie od tuningu szumu.
- **Output audio**: oba pliki na scenariusz — per-turn WAVs (`turn_NN_<speaker>.wav` przy 22050 Hz) + miks `dialogue.wav` (16 kHz, po crossfade + normalize). Per-turn to ground truth dla STT/diaryzacji; miks to input dla faster-whisper.
- **Determinizm**: seed miksera = `sha256(scenario.id)[:8]` + per-turn RNG seeding XTTSv2 (`torch.manual_seed` + `np.random.seed` + `random.seed`, wszystkie trzy potrzebne — coqui-tts sampluje z trzech różnych RNG). Seed per-turn = `sha256(scenario.id)[:4] XOR (turn_index * 0x9E3779B1)` żeby każda tura miała własny deterministyczny start. Zweryfikowane end-to-end na MPS: dwa kolejne reruny tego samego scenariusza produkują byte-identyczny `dialogue.wav`. Caveat: determinizm trzyma się w obrębie jednego hardware'u i wersji torcha; MPS→CPU zmieni bity mimo tych samych seedów (limitacja PyTorch).
- **Mapowanie głosu lekarza**: `sha256(scenario_id)[0] % 2` → stabilny per scenariusz, różni się między scenariuszami. Płeć pacjenta bierze się wprost z `ScenarioModel.patient_sex`.
- **Kontrakt audio**: `dataset_pipeline/src/dataset_pipeline/audio_schema.py` (`AudioArtifact`, `ScenarioAudioBundle`) — oddzielny od `ScenarioModel`, żeby tekstowy kontrakt dialogu nie zmieniał się razem z metadanymi audio. Manifest Phase 1b zawiera oba bloki: `scenarios` (z 1a) i `audio` (nowy).
- **Testy bez modelu**: `Synthesizer` Protocol + `_FakeSynthesizer` w testach — żaden test nie ściąga 2 GB XTTSv2. Torch sam w sobie jest zainstalowanym dep-em (używany do resample w testach).

### Phase 1b — kluczowe pliki

| Plik | Rola |
|---|---|
| `dataset_pipeline/pyproject.toml` | Dodane TTS deps: `coqui-tts>=0.27`, `torch`, `torchaudio`, `soundfile`, `numpy`, `requests`, `tqdm` |
| `.env.example` | Dodane `TTS_DEVICE`, `TTS_MODEL_CACHE`, `TTS_ACCEPT_COQUI_LICENSE`, `MEDVOICEPL_VOICES_API_KEY` (MDC) |
| `dataset_pipeline/src/dataset_pipeline/tts_engine/voice_manager.py` | `VoiceProfile`, `VoiceCatalog`, `load_profiles`, `profile_for` (mapowanie scenariusz→głos) |
| `dataset_pipeline/src/dataset_pipeline/tts_engine/xtts_synthesizer.py` | `XTTSSynthesizer` (lazy model load, auto-device), `Synthesizer` Protocol, czyszczenie tekstu PL prozodia, stability knobs (`temperature`/`sound_norm_refs`/`split_sentences`), `seed=` → `torch`+`numpy`+`random` |
| `dataset_pipeline/src/dataset_pipeline/tts_engine/audio_utils.py` | `save_wav`/`load_wav` (soundfile), `resample` (torchaudio), `normalize_peak`/`normalize_rms`, `duration_seconds` |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/dialog_mixer.py` | `mix_turns` (equal-power crossfade + seeded randomized pause), `MixerConfig`, `load_mixer_config` |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/noise_injector.py` | **Stub** — zwraca audio bez zmian; implementacja w Phase 1b.1 |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/augmentation.py` | `normalize()` (peak-normalize); tempo/pitch/reverb → `NotImplementedError` do Phase 1b.1 |
| `dataset_pipeline/src/dataset_pipeline/audio_schema.py` | `AudioArtifact`, `ScenarioAudioBundle` (Pydantic), `bundle_from_dir` (re-hashing z dysku) |
| `dataset_pipeline/src/dataset_pipeline/pipeline.py` | `run_audio()`, `AudioReport`, `_synthesize_scenario` (internal wiring z DI dla testów), `load_bundle_sidecars` |
| `dataset_pipeline/scripts/run_synthesis.py` | Typer CLI: `--scenarios-dir`, `--audio-dir`, `--scenario-id` (repeatable), `--limit`, `--verbose` |
| `dataset_pipeline/scripts/fetch_reference_voices.py` | Typer CLI: pobiera Common Voice Scripted Speech 25.0 — PL przez Mozilla Data Collective API (Bearer auth), filtruje po długości zdania (130–260 znaków, relaxation ladder 90/60), konwertuje do 22050 Hz mono WAV, pisze `reference_voices/README.md` |
| `dataset_pipeline/scripts/export_manifest.py` | Rozszerzone: manifest ma dwa bloki (`scenarios`, `audio`), każdy z `count` + `entries` |
| `dataset_pipeline/tests/fixtures/tiny_scenario.json` | Fixture: 6-turn kardiologia (bezpieczne — syntetyczne od początku) |
| `dataset_pipeline/tests/test_{audio_utils,voice_manager,dialog_mixer,synthesis_pipeline}.py` | 32 testy Phase 1b (razem z 1a: 57 zielonych) |

### Phase 1b — uruchomienie

```bash
# 1. Sync + TTS consent (runtime opt-in dla coqui-tts MPL-2.0)
uv sync --all-packages
# W .env: TTS_ACCEPT_COQUI_LICENSE=1, TTS_DEVICE=auto, MEDVOICEPL_VOICES_API_KEY=<z mozilladatacollective.com>

# 2. Próbki referencyjne (jednorazowo, idempotentne) — ~500 MB tar.gz cache w ~/.cache/medvoicepl/
uv run python dataset_pipeline/scripts/fetch_reference_voices.py
# lub ręcznie: drop 4 WAV do dataset_pipeline/reference_voices/ (mono, 22050 Hz, 10–20 s)

# 3. Smoke test — 1 scenariusz (~4–6 min na MPS, ~15–20 min na CPU)
uv run python dataset_pipeline/scripts/run_synthesis.py --limit 1

# 4. Manifest
uv run python dataset_pipeline/scripts/export_manifest.py
```

Output: `dataset_pipeline/output/audio/<id[:8]>/turn_NN_<speaker>.wav` × N + `dialogue.wav` (16 kHz) + `bundle.json`. `manifest.json` zawiera bloki `scenarios` + `audio` z SHA-256.

### Phase 1b.1 — decyzje projektowe

- **Cel**: dodać „realistic clinic” degradację (augmentacje per-turn + reverb + noise) obok czystego `dialogue.wav`, który zostaje nietknięty jako ground truth. Dwa pliki na scenariusz: `dialogue.wav` (clean, 16 kHz) + `dialogue_noisy.wav` (noisy, 16 kHz). Per-turn WAVs zostają czyste (do ewaluacji diaryzacji / segment-level STT).
- **Biblioteka szumu**: **Freesound.org** (CC0 / CC-BY) zamiast MUSAN. Powody: (1) MUSAN ma głównie muzykę/mowę, mało „biurowej ciszy”; (2) Freesound daje `license:("Creative Commons 0" OR "Attribution")` w filtrze, więc fetcher z definicji nie ściągnie NC/SA. Kuratowany zestaw 7 kategorii (~20 klipów): keyboard, HVAC, paper rustle, distant chatter, footsteps, door, hospital ambience. Fetcher ściąga OGG preview (niska jakość — i tak miksujemy na 10–25 dB SNR, nikt nie usłyszy kompresji), konwertuje do 22050 Hz mono WAV, trimuje do 15 s.
- **SNR stratified buckets**: 20–25 dB (40%, clean clinic), 15–20 dB (40%, typowy), 10–15 dB (20%, stress test). `snr_db` samplowane uniform w obrębie wylosowanego bucketa. Seed osobny od mixer seeda (`sha256(scenario.id + "|noise")[:8]`) żeby tuning noise config nigdy nie perturbował czystego `dialogue.wav`.
- **FX stack**: **pedalboard** (Spotify, MIT) zamiast `librosa`. Powód: librosa ciągnie `numba` + `llvmlite` — stos wolny w instalacji i kruchy na Apple Silicon. Pedalboard = natywne binarki JUCE, `pip install` w sekundy. `time_stretch` (tempo bez zmiany pitcha), `PitchShift` (pitch bez zmiany tempa), `Reverb` (room size 0/1) — wszystkie są pure funkcjami (bez RNG w środku), randomizacja parametrów na poziomie pipeline'u.
- **Per-turn augmentation**: pitch ±1 semitone, tempo 0.95–1.05. Aplikowane PRZED miksowaniem, więc naturalna wariancja przechodzi przez crossfade. Seed per-turn `turn_seed XOR 0xA5A5A5A5` — rozdzielone od TTS seed, tzn. zmiana zakresów augment nie wymaga re-renderu TTS.
- **Dialog-level reverb**: aplikowany RAZ na zmiksowanym dialogu (ogon reverb przechodzi przez granice tur naturalnie). Pokój sampled z (`small=0.15`, `medium=0.35`, `large=0.55`) z wagami 0.4/0.4/0.2.
- **Kolejność efektów w variancie noisy**: mix → reverb → resample do 16 kHz → noise injection → normalize. Resample przed noisem: (1) oszczędza resample każdego klipa szumu; (2) SNR mierzony na sample rate STT-a, czyli tak jak „widzi” to faster-whisper.
- **Kontrakt manifestu**: `ScenarioAudioBundle` ma 6 nowych opcjonalnych pól (`dialogue_noisy_path/sha256/duration_s`, `noise_snr_db`, `noise_clip`, `reverb_room`). Null na bundle'ach z Phase 1b i na `--clean-only` runach.
- **WER eval loop**: `research_and_testing/scripts/eval_stt_on_dataset.py` + `evaluation/stt_eval/wer_calculator.py`. Dwa backendy za `Transcriber` Protocol: `--backend faster-whisper` (default, CPU/CUDA baseline, portable) i `--backend mlx` (mlx-whisper, Metal GPU na Apple Silicon). Pure-Python Levenshtein (bez `jiwer` — moduł evaluation zostaje dep-free). Output: tabela w konsoli + `results/stt_eval_<timestamp>.json` z polem `backend`. To jest feedback loop do tuningu noise/augment: za łagodne → clean≈noisy; za ostre → noisy eksploduje.
- **STT backend na Apple Silicon — wybór**: preferowany **`--backend mlx`** (MLX Whisper, `mlx-community/whisper-large-v3-mlx`) — wykorzystuje 10-rdzeniowy GPU M5 przez Metal, ~5–10× szybciej niż faster-whisper CPU int8, ta sama jakość `large-v3`. Install: `uv pip install mlx-whisper`. faster-whisper zostaje jako portable baseline (CUDA / inne maszyny / CI).
- **faster-whisper na Apple Silicon — gotchas** (tylko jeśli nie używasz MLX): (1) `compute_type=float16` CRASHUJE na CPU (`ctranslate2` nie ma efektywnego fp16 path'a poza CUDA/ROCm). Używamy `int8` jako default; `int8_float32` dla marginalnie lepszej jakości przy tej samej prędkości; `float32` jako accuracy ceiling (~2× wolniejsze). (2) `device=cpu` jest jedynym realistycznym wyborem — `ctranslate2` nie wspiera MPS backendu, a CoreML path jest eksperymentalny. (3) Obserwowana prędkość: large-v3 int8 na M5 CPU = ~0.6× realtime → 3-min dialog transkrybuje się w ~5 min. Klin+noisy (pełen eval per-scenariusz) ≈ 10 min. MLX redukuje to do ~1–2 min.
- **Pierwszy pomiar WER (2026-04-22, 1 scenariusz)**: clean 9.12%, noisy 11.23% przy SNR 15.2 dB. Delta +2.1pp = zdrowa kalibracja (za łagodne gdyby <0.5pp, za ostre gdyby >10pp). 9% clean WER to trafione: głównie zjadanie liczbowych ciągów (PESEL, telefon, adres) i terminów medycznych (IGF-1, insulinopodobny). To baseline referencyjny dla Phase 2 STT — każdy on-premise model testujemy dokładnie tym samym eval scriptem na tym samym zbiorze `dialogue.wav`/`dialogue_noisy.wav`.

### Phase 1b.1 — kluczowe pliki

| Plik | Rola |
|---|---|
| `dataset_pipeline/pyproject.toml` | Dodane `pedalboard>=0.9` |
| `.env.example` | Dodany `FREESOUND_API_KEY` |
| `dataset_pipeline/configs/audio_postprocess.yaml` | Rozszerzone o bloki `augment`, `reverb`, `noise` (1b ma tylko `crossfade` + `output`) |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/noise_injector.py` | **Pełna implementacja** — `NoiseConfig`, `SnrBucket`, `load_noise_config`, stratified bucket sampling, RMS-based gain, `pick_params` (dla manifestu, ta sama sekwencja RNG co `inject`) |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/augmentation.py` | **Pełna implementacja** — `AugmentConfig`, `ReverbConfig`, `tempo`/`pitch_shift`/`reverb` przez pedalboard, `normalize` zachowane z 1b |
| `dataset_pipeline/src/dataset_pipeline/audio_schema.py` | 6 nowych opcjonalnych pól na `ScenarioAudioBundle` |
| `dataset_pipeline/src/dataset_pipeline/pipeline.py` | `run_audio()` z `clean_only` flagą; `_synthesize_scenario` emituje oba varianty + populuje nowe pola bundle'a |
| `dataset_pipeline/scripts/run_synthesis.py` | Nowa flaga `--clean-only` |
| `dataset_pipeline/scripts/fetch_noise_library.py` | **NOWY** — Freesound fetcher, 7 kategorii, filtr licencji CC0/CC-BY w search query, idempotentny |
| `dataset_pipeline/tests/test_noise_injector.py` | **NOWY** — 8 testów (enabled/disabled, empty lib, SNR accuracy, determinism, RNG parity) |
| `dataset_pipeline/tests/test_augmentation.py` | **NOWY** — 11 testów (identity, length preservation, spectrum shift, config loaders) |
| `dataset_pipeline/tests/test_synthesis_pipeline.py` | +2 testy: end-to-end noisy variant, `clean_only=True` skip |
| `research_and_testing/evaluation/stt_eval/wer_calculator.py` | **Pełna implementacja** — pure-Python WER/CER, Unicode-punctuation-aware normalizer (obsługuje `„…"`, `—`, itd.) |
| `research_and_testing/scripts/eval_stt_on_dataset.py` | **NOWY** — faster-whisper large-v3 PL na clean vs noisy, tabela WER/ΔWER + JSON artifact |
| `research_and_testing/tests/test_wer_calculator.py` | **NOWY** — 7 testów normalizera + WER/CER edge cases |

### Phase 1b.1 — uruchomienie

```bash
# 1. Sync (dociąga pedalboard)
uv sync --all-packages

# 2. Pobranie biblioteki szumu (jednorazowo, idempotentne)
#    W .env: FREESOUND_API_KEY=<z https://freesound.org/apiv2/apply/>
uv run python dataset_pipeline/scripts/fetch_noise_library.py
# lub ręcznie: drop CC0 WAVs do dataset_pipeline/noise_library/

# 3. Synteza — generuje dialogue.wav + dialogue_noisy.wav
uv run python dataset_pipeline/scripts/run_synthesis.py --limit 1
# --clean-only: pomiń noisy (useful przy diffingu samego TTS)

# 4. Eval WER (wymaga faster-whisper: `uv pip install faster-whisper`)
uv run python research_and_testing/scripts/eval_stt_on_dataset.py --limit 1
```

### Phase 2a — decyzje projektowe

- **Podział Phase 2 na 2a/2b/2c**: STT+diaryzacja (2a) to fundament, którego konsumują SLM (2b) i anonimizacja+API (2c). Rozłożenie pozwala (1) osiągnąć stabilny `Transcript` kontrakt ZANIM jakikolwiek kod z 2b/2c zacznie go importować, (2) trzymać każdą fazę w przedziale 2-3 dni pracy + weryfikowalnego smoke'a. W 2a SLM/Presidio/trust_gate zostają stubami.
- **STT backend — dual**: `MLX Whisper` primary na Apple Silicon (~5-10x szybciej niż ctranslate2 CPU int8, już sprawdzone w Phase 1b.1 eval), `faster-whisper` fallback dla CUDA / portable / CI. Za `Transcriber` Protocol identycznego kształtu w obu ścieżkach; Protocol żyje w `on_premise.stt.whisper_engine` i jest importowany przez `research_and_testing/scripts/eval_stt_on_dataset.py` — benchmarkujemy dokładnie ten sam kod, który deployujemy.
- **Diaryzacja**: `pyannote.audio 3.1` (`pyannote/speaker-diarization-3.1`). Gated model na HF — wymaga `HUGGINGFACE_TOKEN` env + jednorazowego license accept na stronie modelu. `num_speakers=2` hardcoded (doctor + patient, bez triage nurse'a / meetings w naszym domenie). Token-check before pyannote import, żeby error komunikat, który user musi zobaczyć, wyświetlał się pierwszy (i żeby testy jednostkowe nie wymagały pyannote.audio).
- **Speaker role mapping — „first-segment wins"**: pyannote zwraca opaque `SPEAKER_00`/`SPEAKER_01`. Heurystyka: **speaker, którego najwcześniejszy segment zaczyna się najwcześniej → `doctor`**; drugi → `patient`. Klinicznie poprawne dla naszego domeny (wizyta otwiera się powitaniem lekarza). `role_override={"SPEAKER_00": "patient"}` wystawiony na poziomie funkcji `transcribe_and_diarize()` jako escape hatch. **Dwa bardziej wyrafinowane warianty odpadły w iteracjach 2026-04-23/24**: (a) "dominant speaker w pierwszych 30 s" wywrócił role na 6/8 scenariuszach — pacjent typowo mówi dłużej niż lekarz w pierwszych 30 s (lekarz zadaje krótkie pytania otwierające, pacjent odpowiada rozlegle), DER hit 97-100%. (b) "pierwszy ≥2 s segment wygrywa" wywrócił role gdy pyannote zaczynał kaskadą sub-sekundowych fragmentów, a pierwszy ≥2 s segment lądował środku rozmowy na arbitralnie sklastrowanym labelu. Oba regresy były artefaktem pyannote clustering collapse na TTS — wespeaker embedding nie separuje sklonowanych voice'ów XTTSv2 niezawodnie, więc każda heurystyka próbująca identyfikować speaker'a z *późniejszego* segmentu sampluje z niestabilnego labelowania. Najwcześniejszy segment to jedyny punkt gdzie pyannote konsystentnie wkłada doctora w ten cluster, z którym zaczął. Wszystkie trzy warianty udokumentowane w docstringu `_build_role_map` jako regression guard.
- **Merge strategy Whisper+pyannote**: word-level Whisper timestamps (`word_timestamps=True`) bucketowane do **nearest pyannote segment by time distance** — jeśli midpoint słowa jest w segmencie, distance=0; jeśli w gap pomiędzy, distance to gap do najbliższej krawędzi. Strict "midpoint-inside" gubił ~15-25% słów których midpoint wpadał w silent gap (pyannote ma ciaśniejszy VAD niż Whisper), co eksplodowało WER o +16pp na neuro-ct. Nowy scheme nie gubi żadnego słowa Whispera, a każde słowo atachuje się do dokładnie jednego segmentu, więc DER i speaker-attribution nie zmieniają się. Gdy pyannote mówi "speaker" a żadne słowo nie ląduje jako nearest → segment droppowany (pyannote czasem łapie wdechy/kaszel jako mowę).
- **Anti-hallucination flag — `condition_on_previous_text=False`**: wymagane w obu backendach Whisper (MLX + faster-whisper). Whisper-large-v3 na studio-quality TTS audio wpada w repetition loops — emituje "Dziękuję za obejrzenie" albo powtarza zdanie przez minuty — z prev-segment decoder promptem włączonym. Neuro-ct clean WER był 72.7% z powodu tego, noisy tego samego scenariusza 6.7% bo szum przerywał pętlę. Wyłączenie promptu usuwa sampling bias bez zmiany wag ani per-segment quality. Symmetryczne w obu backendach żeby switch `WHISPER_BACKEND` na deploy time nie perturbował WER baselines.
- **Transcript contract**: `on_premise.stt.schema.Transcript` (Pydantic v2, `extra="forbid"`, SHA anchor audio). Żyje **w obrębie** `on_premise` i **nie jest re-exportowany** z `shared/schemas/`. `tests/test_trust_boundary_imports.py` to preview-guard (`pkgutil.iter_modules` po `shared.schemas`, fail jeśli wycieknie `Transcript`/`TranscriptSegment`/`TranscriptionMetadata`/`WordTiming`) — prawdziwy runtime gate dopisujemy w 2c.
- **Ewaluacja DER**: pure-Python `der_calculator.py` (dep-free, tak samo jak `wer_calculator.py`), reference-timeline rekonstruowana z `ScenarioAudioBundle.turns` (pause między turami backed-out z `dialogue_duration_s - sum(turn_durations) + (n-1)*crossfade`). Eval script rozszerzony o `--backend on-premise` → WER (via `transcript.text`) + DER w jednym przebiegu.
- **Trust posture w 2a**: brak HTTP/API endpointu. `run_transcription.py` CLI zapisuje do `application/on_premise/output/transcripts/<sid[:8]>/transcript.json` — artefakt *lokalny*, gitignorowany. Gateway integration czeka do Phase 2c gdy istnieje `AnonymizedPayload` do gate'owania.
- **Testy bez downloadów**: `Transcriber`+`Diarizer` Protocol + Fake pattern z Phase 1b. Żaden test nie pobiera wag Whisper/pyannote; `build_transcriber` sprawdzany przez dispatch (assert klasa, nie wywołanie `.transcribe`).

### Phase 2a — kluczowe pliki

| Plik | Rola |
|---|---|
| `application/on_premise/pyproject.toml` | Dodane: `faster-whisper>=1.0`, `mlx-whisper>=0.4`, `pyannote.audio>=3.1`, `torch>=2.2`, `torchaudio>=2.2`, `numpy`, `soundfile`, `pydantic>=2.6`, `pyyaml`, `typer`, `rich`. Extra `[apple] = ["mlx-whisper>=0.4"]` dla Linux deploys. |
| `.env.example` | Dodane: `WHISPER_BACKEND` (mlx), `WHISPER_MLX_REPO`, `HUGGINGFACE_TOKEN`, `PYANNOTE_DEVICE`. Zmieniony default `WHISPER_COMPUTE_TYPE=int8` (fp16 crashuje na CPU). |
| `application/on_premise/src/on_premise/stt/schema.py` | **NEW** — `WordTiming`, `TranscriptSegment`, `TranscriptionMetadata`, `Transcript` (Pydantic v2, `extra="forbid"`, SHA anchor). Docstring explicit trust-boundary notice. |
| `application/on_premise/src/on_premise/stt/whisper_engine.py` | **NEW** — `Transcriber` Protocol, `RawWord`+`RawTranscription` dataclasses, `MLXWhisperTranscriber` i `FasterWhisperTranscriber` (oba zwracają word timings), `build_transcriber(config, env)` factory, `load_whisper_config` (ENV template expansion). |
| `application/on_premise/src/on_premise/stt/diarization.py` | **NEW** — `Diarizer` Protocol, `RawDiarizationSegment`, `PyannoteDiarizer` (token-check przed pyannote import, `num_speakers=2`, `_resolve_device`), `build_diarizer(config, env)`. |
| `application/on_premise/src/on_premise/stt/vad.py` | **NEW** — 15-LOC `detect_silence_segments` (RMS-threshold numpy, mono only) dla CLI `--trim-silence` flag. |
| `application/on_premise/src/on_premise/stt/pipeline.py` | **NEW** — `transcribe_and_diarize(wav_path, *, transcriber, diarizer, scenario_id, role_override)`. Merge po midpoincie słowa, role mapping (first-speaker=doctor), drop empty segs, SHA anchor audio. |
| `application/on_premise/scripts/run_transcription.py` | **NEW** — Typer CLI: `--audio-dir`, `--transcripts-dir`, `--variant {clean,noisy,both}`, `--scenario-id`, `--backend`, `--limit`, `--verbose`, `--trim-silence`. Pisze `<sid[:8]>/transcript.json`(+`transcript_noisy.json`). |
| `application/on_premise/scripts/check_stt_models.py` | **NEW** — preflight: mlx-whisper import, faster-whisper import, HF token presence, pyannote license accepted (actually loads). Red/green table, exit non-zero on missing. |
| `application/on_premise/tests/test_{schema,whisper_engine,diarization,pipeline,vad,trust_boundary_imports}.py` | **NEW** — 25 testów (6 + 5 + 3 + 5 + 5 + 1). Zero model downloads: Protocol + Fake pattern. |
| `research_and_testing/evaluation/stt_eval/der_calculator.py` | **NEW** — pure-Python DER: `diarization_error_rate(reference, hypothesis, *, label_mapping)` via union-of-endpoints time-slicing + bisect. `reference_from_turn_durations` helper. Zero deps (no `pyannote.metrics`). |
| `research_and_testing/tests/test_der_calculator.py` | **NEW** — 8 testów (identity, pure confusion, pure miss, pure FA, partial overlap, label mapping, turn-duration reference builder, length mismatch). |
| `research_and_testing/scripts/eval_stt_on_dataset.py` | Refactor: importuje `FasterWhisperTranscriber`/`MLXWhisperTranscriber` z `on_premise.stt.whisper_engine`. Nowy `--backend on-premise` → pełna pipeline + DER. `_Runner` abstrakcja unifikuje bare + on-premise przebieg. Tabela + JSON rozszerzone o DER. |
| `.gitignore` | Dodane `application/on_premise/output/` (un-anonimizowane transkrypty nie idą do gita). |

### Phase 2a — uruchomienie

```bash
# 1. Sync (dociąga pyannote.audio, mlx-whisper, faster-whisper)
uv sync --all-packages

# 2. HF token + license accept dla pyannote (jednorazowo)
#    https://huggingface.co/pyannote/speaker-diarization-3.1 -> Agree to license
#    W .env: HUGGINGFACE_TOKEN=<read-token>, WHISPER_BACKEND=mlx

# 3. Preflight — diagnozuje brakujące komponenty z linkiem do fixa
uv run python application/on_premise/scripts/check_stt_models.py

# 4. Smoke — 1 scenariusz, clean only (~25-40 s na M5: MLX ~15 s + pyannote ~10 s)
uv run python application/on_premise/scripts/run_transcription.py --limit 1 --variant clean
# Output: application/on_premise/output/transcripts/<sid[:8]>/transcript.json

# 5. Eval WER+DER przez Phase 2a pipeline (1 scenariusz, clean+noisy)
uv run python research_and_testing/scripts/eval_stt_on_dataset.py --backend on-premise --limit 1

# 6. Regression test — bare MLX backend musi dać tę samą WER co przed refactorem
uv run python research_and_testing/scripts/eval_stt_on_dataset.py --backend mlx --limit 1
```

Output CLI: `<transcripts-dir>/<sid[:8]>/transcript.json` (+ `transcript_noisy.json` przy `--variant both`). Output eval: `research_and_testing/results/stt_eval_<timestamp>.json` z blokiem `der` dla `--backend on-premise`.

### Phase 2b — decyzje projektowe

- **Podział Phase 2b vs 2c**: 2b owns Transcript → validated `SOAPNote` (lokalny SLM + grammar + retry + PII-hallucination scrub). 2c owns `SOAPNote` → `AnonymizedSOAPNote` (Presidio + trust_gate + FastAPI). `SOAPNote` żyje **w obrębie** `on_premise.structuring` i nie wycieka do `shared/` — analogicznie do `Transcript` z 2a. Un-anonimizowane SOAP zapisywane do `application/on_premise/output/soap_notes/<sid[:8]>/soap_note.json` (gitignored).
- **SLM backend — dual, mirror Phase 2a**: `mlx-lm` primary na Apple Silicon (Metal GPU, ~10× szybciej niż CPU int8), `llama-cpp-python` portable fallback (CPU/CUDA, GGUF). Oba za `StructuredExtractor` Protocol z `extract(prompt, *, schema) -> str`. `build_extractor(config, env)` dispatchuje po `SLM_BACKEND`. `llama-cpp-python` pod `[project.optional-dependencies] portable` (kompiluje się ze źródeł ~3–5 min na M5 z `CMAKE_ARGS="-DLLAMA_METAL=on"`), Linux/CI opt-in przez `uv sync --extra portable`.
- **Model: Bielik-11B-v3.0-Instruct** (SpeakLeash). Polish-native, ~6 GB 4-bit (mieści się obok Whispera w 24 GB M5). MLX repo default: `speakleash/Bielik-11B-v3.0-Instruct-MLX-4bit` — **pierwszoustrona publikacja SpeakLeash, ungated** (`gated=false` w HF API), więc preflight nie wymaga HF license accept jak przy v2.3. GGUF: `speakleash/Bielik-11B-v3.0-Instruct-GGUF`. **Dlaczego v3.0 zamiast v2.3**: (1) community MLX repack `mlx-community/Bielik-11B-v2.3-Instruct-4bit` wyparował z HF (preflight dostawał 404/401 — "Repository Not Found"); (2) v3.0 używa DPO-Positive + GRPO zamiast zwykłego DPO z v2.3 → lepsze instruction-following (kluczowe dla JSON-schema compliance); (3) v3.0 trenowany na 20M+ instrukcjach (~17B tokens) z multi-turn support. Chat template: ChatML (`<\|im_start\|>user … <\|im_end\|>`). **Context length**: **32768 tokens native** (potwierdzone przez `check_slm_models.py` 2026-04-24). 8× więcej niż v2.3 (4K) — znosi cały "tight budget" constraint z pierwotnego planu. Few-shot domyślnie włączony, brak potrzeby chunkowania transkryptów (najdłuższy bundle Phase 2a ~4 min ≈ 2-3K tokenów), retry loop ma wygodny headroom. **PLLuM-12B-instruct** (NASK, 32K native, Polish-first) nominowany jako benchmark-alternatywa przez `SLM_MODEL_ID` swap — nie default, ale schema-compliance A/B w eval.
- **Structured output — grammar od pierwszego dnia**: `lm-format-enforcer` jako logits-processor w obu backendach, schema z `SOAPNote.model_json_schema()`. Toggle przez `SLM_STRUCTURED_MODE={grammar,raw}` — eval robi A/B. Rationale: lokalne 11B PL instruct emitują markdown fences, trailing commas, pomieszane klucze; Phase 1a dostała to z `response_schema` za darmo z Vertex, tu kupujemy przez grammar. Pydantic validation + correction-hint retry jako drugie ogrodzenie dla semantic failures (pusty `chief_complaint`, zły ICD-10) których grammar nie łapie. `TransientSLMError`/`PermanentSLMError` mirrorują Phase 1a.
- **Anti-hedging**: Bielik (v2.3 SFT+DPO, v3.0 DPO-Positive + GRPO) over-refuses na klinicznych promptach ("Jako model językowy nie mogę postawić diagnozy…"). Counter w systemie: "Strukturyzujesz istniejący wywiad, nie diagnozujesz — reformatowanie, nie rozumowanie kliniczne". Test `test_system_prompt_has_anti_hedge_framing` to regression guard. Czeka na weryfikację na v3 — RLHF pipeline ma inny shaping, może hedgować mniej albo więcej.
- **SOAP schema — szczelna z hookami do 2c**:
  - `vitals: list[Vital]`, `Vital = {name: Literal["bp","hr","rr","temp","spo2","weight","height","bmi"], value: str, unit: str | None}`. Nie-whitelistowana nazwa (cholesterol, glukoza) ląduje w `objective.labs_mentioned`.
  - `icd10_code: str | None = Field(pattern=r"^[A-Z]\d{2}(\.\d{1,2})?$")` — stopuje model przed oddaniem `"I10.0 (nadciśnienie)"`.
  - `provenance: list[int]` opcjonalne na każdej sekcji (S/O/A/P) i na `red_flags` — indeksy segmentów `[N]` z których lec pole pochodzi. Phase 2c audit logger chce to dla "red flag X z segmentu N".
  - `SOAPNoteMetadata.source_transcript_sha256 = sha256(transcript.model_dump_json())` liczone w 2b pipeline — Phase 2a `Transcript` zostaje frozen.
- **PII hallucination post-check**: po walidacji SOAP-a, walk po wszystkich string leaves; każdy PESEL-shaped (`\b\d{11}\b`) / phone-shaped (`\+?\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}`) token, który NIE jest verbatim w `transcript.text` → drop + `log.warning`. Cheap guard — SLM potrafi wymyślić plausible polskie numery, wolimy zgłosić drop rate niż puścić do 2c. Prawdziwa anonimizacja (PII-in-source) to 2c.
- **Ewaluacja — pure Python, dep-free (mirror WER/DER)**:
  - `schema_compliance_rate(attempts)` — fraction validated first-try (przed retry).
  - `field_coverage(soap)` — boolean per-bucket mapa (19 pól), aggregate-friendly.
  - `chief_complaint_recall(soap, scenario)` — normalized token-Levenshtein vs `ScenarioModel.chief_complaint`. Reuses `wer_calculator.normalize` + `_levenshtein`.
  - `diagnosis_match(soap, scenario)` — bidirectional substring po normalize vs `expected_diagnosis`.
  - `icd10_match(soap, scenario)` — exact case-sensitive.
  - `plan_nonempty(soap)` — sanity: ≥1 z `{medications_prescribed, diagnostic_tests, referrals, follow_up, patient_education}` niepusty.
  - **Brak scoringu plan-content** w 2b. `ScenarioModel` ma tylko CC + expected_diagnosis + icd10_code; Gemini-judge dla reference SOAP coupuje eval do Gemini konwencji. Defer do Phase 4 (rubryka kliniczna na realnym audio).
- **Trust-boundary — AST-walk**: `test_trust_boundary_imports.py` już skanuje `shared.schemas` po zakazanych *nazwach*. 2b dodaje drugi test `test_shared_does_not_import_from_on_premise`: AST-walk po `application/shared/src/shared/**/*.py`, fail jeśli któryś `Import`/`ImportFrom` ma moduł startujący `on_premise.`. Łapie prawdziwy failure mode (ktoś filluje `shared/schemas/soap.py` z `from on_premise.structuring.soap_schema import SOAPNote`).
- **Trust posture w 2b**: brak HTTP endpointu (tak jak 2a). `run_soap_extraction.py` CLI sync batch → lokalne pliki. `shared/schemas/soap.py` zostaje one-line stubem do 2c.
- **Testy bez downloadów**: `StructuredExtractor` Protocol + `_StubExtractor` zwracający canned JSON — zero model weights. `build_extractor` testowane przez class dispatch (monkeypatch `sys.modules["mlx_lm"]`). Protocol *nie jest* `@runtime_checkable`, więc test stub-conformance używa `hasattr` duck-check zamiast `isinstance`.

### Phase 2b — kluczowe pliki

| Plik | Rola |
|---|---|
| `application/on_premise/pyproject.toml` | Dodane: `mlx-lm>=0.19`, `lm-format-enforcer>=0.10`. Extra `[portable] = ["llama-cpp-python>=0.2.90,<0.3"]` dla Linux/CI. |
| `.env.example` | Dodany blok Phase 2b: `SLM_BACKEND`, `SLM_MODEL_ID`, `SLM_MLX_REPO`, `SLM_GGUF_PATH`, `SLM_DEVICE`, `SLM_MAX_NEW_TOKENS`, `SLM_TEMPERATURE`, `SLM_STRUCTURED_MODE`. |
| `application/on_premise/configs/slm.yaml` | Full defaults z `${VAR}` expansion, mirror `whisper.yaml`. |
| `application/on_premise/src/on_premise/structuring/soap_schema.py` | **Filled** — `SOAPNote`, `SOAPSubjective`/`Objective`/`Assessment`/`Plan`, `Vital`, `LabMention`, `SOAPNoteMetadata`. Pydantic v2, `extra="forbid"`, ICD-10 regex, vitals Literal whitelist, `provenance: list[int]` na każdej sekcji. |
| `application/on_premise/src/on_premise/structuring/slm_engine.py` | **Filled** — `StructuredExtractor` Protocol, `MLXLMExtractor` (lazy load, chat_template guard, `lmformatenforcer.integrations.mlxlm.build_mlx_logits_processor`), `LlamaCppExtractor` (GGUF path + cuda-detect + `create_chat_completion`), `build_extractor(config, env)`, `load_slm_config` (dup `_expand_env_templates`), `TransientSLMError`/`PermanentSLMError`. |
| `application/on_premise/src/on_premise/structuring/prompts.py` | **Filled** — `SYSTEM_PROMPT_PL` (anti-hedge), kompaktowy `FEW_SHOT_EXAMPLE` (kardiologia), `render_user_prompt(transcript, *, include_few_shot=True)` (segmenty jako `[N] speaker: text`), `render_correction_hint(error)` mirror Phase 1a. |
| `application/on_premise/src/on_premise/structuring/pipeline.py` | **NEW** — `transcript_to_soap()`, retry loop na `ValidationError` + correction hint, `_scrub_hallucinated_pii` (walk BaseModel tree, drop PESEL/phone not in source), `_build_metadata` (SHA anchor transcript, backend parse z `extractor.name.partition(":")`). |
| `application/on_premise/src/on_premise/structuring/entity_extractor.py` | Stub z docstringiem "Phase 2c (Presidio NER)". |
| `application/on_premise/scripts/run_soap_extraction.py` | **NEW** — Typer CLI, `--variant {clean,noisy,both}`, `--scenario-id`, `--backend`, `--structured-mode`, `--limit`. Pisze `output/soap_notes/<sid[:8]>/soap_note{,_noisy}.json` (gitignored). |
| `application/on_premise/scripts/check_slm_models.py` | **NEW** — preflight: mlx-lm import, llama-cpp-python (warn-only), lm-format-enforcer, MLX repo load, chat_template check, context length probe, 1-token smoke generation. Red/green Rich table, exit non-zero. |
| `application/on_premise/tests/test_soap_schema.py` | **NEW** — round-trip, `extra="forbid"`, ICD-10 accepts `I10`/`I10.0`/`I10.00` rejects gloss, vitals whitelist, `LabMention` free-form, provenance default `[]`, metadata sha256-regex. |
| `application/on_premise/tests/test_slm_engine.py` | **Replaced stub** — `build_extractor` dispatch z `monkeypatch.setitem(sys.modules, "mlx_lm", dummy)`, Protocol shape via duck-check, `load_slm_config` env expansion. |
| `application/on_premise/tests/test_prompts.py` | **NEW** — segment numbering, few-shot toggle, empty transcript handling, skip-empty-segments, correction hint formats errors, anti-hedge framing w systemie. |
| `application/on_premise/tests/test_structuring_pipeline.py` | **NEW** — happy path (stub extractor), retry-then-succeed, retry exhausted raises `SOAPExtractionError`, PII scrub drops hallucinated, PII keeps real, deterministic `source_transcript_sha256`, extractor exception surfaces. |
| `application/on_premise/tests/test_trust_boundary_imports.py` | **Extended** — `FORBIDDEN_NAMES` rozszerzone o 8 Phase 2b symboli; nowy `test_shared_does_not_import_from_on_premise` AST-walk. |
| `research_and_testing/evaluation/slm_eval/soap_quality.py` | **Filled** — `SchemaAttempt`, `schema_compliance_rate`, `field_coverage` (dict[str, bool] na 19 pól), `chief_complaint_recall` (reuse `normalize`+`_levenshtein`), `diagnosis_match` (bi-substring), `icd10_match` (exact), `plan_nonempty`. |
| `research_and_testing/evaluation/slm_eval/benchmark_slm.py` | **Filled** — `EvalRow`, `evaluate_one(scenario, transcript, *, extractor, variant)`, `summarize(rows) -> EvalSummary` z aggregate metrics. |
| `research_and_testing/scripts/eval_slm_on_transcripts.py` | **NEW** — Typer CLI, `load_dotenv()` wewnątrz `run()` (Phase 2a lekcja), `--structured-mode`/`--backend` override, Rich table, pisze `research_and_testing/results/slm_eval_<timestamp>.json`. |
| `research_and_testing/tests/test_soap_quality.py` | **NEW** — 9 testów (normalizer edge cases, CC recall, diagnosis substring, ICD-10 exact, plan-nonempty, field_coverage kształt). |
| `pyproject.toml` (root) | `per-file-ignores` dostaje `"**/structuring/prompts.py" = ["RUF001"]` — polskie stringi używają EN DASH dla zakresów liczb (1–2, 2–3) intencjonalnie. |

### Phase 2b — uruchomienie

```bash
# 1. Sync (dociąga mlx-lm, lm-format-enforcer, interegular, sentencepiece)
uv sync --all-packages
# Opcjonalnie dla portable: uv sync --all-packages --extra portable

# 2. HF login (pierwsza load ściąga ~6 GB Bielik-4bit z HF do ~/.cache/huggingface/)
#    W .env: SLM_BACKEND=mlx-lm, SLM_STRUCTURED_MODE=grammar, HUGGINGFACE_TOKEN=<z 2a>

# 3. Preflight — diagnozuje brakujące komponenty + sprawdza chat_template
uv run python application/on_premise/scripts/check_slm_models.py

# 4. Smoke — 1 scenariusz, grammar mode (~30-60 s na M5: weights load ~20 s + gen ~15-30 s)
uv run python application/on_premise/scripts/run_soap_extraction.py --limit 1 --variant clean --structured-mode grammar
# Output: application/on_premise/output/soap_notes/<sid[:8]>/soap_note.json

# 5. A/B raw vs grammar (schema compliance delta)
uv run python application/on_premise/scripts/run_soap_extraction.py --limit 1 --structured-mode raw

# 6. Eval na 8 bundlach (WER+DER+SOAP w jednym pipeline)
uv run python research_and_testing/scripts/eval_slm_on_transcripts.py --structured-mode grammar
```

Output eval: `research_and_testing/results/slm_eval_<timestamp>.json` z aggregate (schema_compliance_rate, mean_cc_recall, diagnosis_match_rate, icd10_match_rate, plan_nonempty_rate, mean_duration_s) + per-row breakdown.

## Komendy deweloperskie

```bash
make install          # Instalacja zależności całego monorepo
make lint             # ruff + mypy
make test             # pytest per moduł
make up               # docker-compose up (cały system)
make eval             # Uruchomienie benchmarków ewaluacyjnych
```

## Historia zmian architektonicznych

| Data | Zmiana |
|---|---|
| 2026-04-14 | Zaprojektowano strukturę monorepo: trzy filary (dataset_pipeline, application, research_and_testing). Zdefiniowano trust boundary, podział on-premise/cloud, gateway jako mediator. |
| 2026-04-20 | Phase 1a rozpoczęta: scenariusze generowane przez Gemini Developer API (`google-genai`, klucz API). Zmiana domyślnego providera w `dataset_pipeline` z OpenAI → Google. TTS/postprocessing (XTTSv2, pydub, sox) odłożone do Phase 1b — czekają na próbki referencyjne głosów. |
| 2026-04-20 | Phase 1a zaimplementowana: `ScenarioModel` (Pydantic v2) jako stabilny kontrakt dialogu, `GeminiClient` z `response_schema=ScenarioModel` + tenacity retry, `ScenarioGenerator` z walidacyjnym retry + `failures.json`, Typer CLI (generate / validate / manifest). 25 testów zielonych, ruff + mypy czyste. Zmiany lint-config: `B008` zignorowane w `scripts/**`, `TC00x` w `tests/**`, `disable_error_code=["import-untyped"]` dla mypy. |
| 2026-04-20 | Workflowy GitHub Actions (`ci.yml`, `docker-build.yml`, `eval.yml`) tymczasowo wyłączone przez zmianę rozszerzenia na `.yml.disabled`. Zostaną przywrócone po ukończeniu implementacji — lint/test/build lokalnie przez `uv run ...`. |
| 2026-04-20 | **Migracja z Gemini Developer API (AI Studio) na Vertex AI** (user uzyskał kredyty $300). `GeminiClient` używa `genai.Client(vertexai=True, project=..., location=...)`. Autoryzacja: ADC (`gcloud auth application-default login`) lub `GOOGLE_APPLICATION_CREDENTIALS`. Dodane env: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`. Żadne zmiany w schema, promptach ani generatorze — tylko inicjalizacja klienta. |
| 2026-04-20 | **Fix kolizji pakietów w workspace uv.** Każdy workspace member (`dataset_pipeline`, `application/{cloud,gateway,on_premise,shared}`) miał `packages = ["src"]` w hatchlingu → wszystkie instalowały się jako top-level `src`, kolidowały i `from src.xyz` losowo trafiał w innego membera (alfabetycznie pierwszy `cloud` wygrywał). Restrukturyzacja src-layout: `<member>/src/<pkg>/...` z `packages = ["src/<pkg>"]`. Importy: `from src.pipeline` → `from dataset_pipeline.pipeline`. Usunięte sys.path hacki w scriptach i `conftest.py`. Smoke test 5/5 zielony, 25 testów OK. |
| 2026-04-21 | **Phase 1b — kod gotowy (czeka na próbki głosów).** XTTSv2 (community `coqui-tts>=0.27`) z lazy model load na MPS/CUDA/CPU autodetect. `tts_engine/{voice_manager,xtts_synthesizer,audio_utils}.py` i `postprocessing/dialog_mixer.py` zaimplementowane. Mikser z equal-power crossfade + seeded randomized pause → deterministyczny SHA `dialogue.wav`. `Synthesizer` Protocol + `_FakeSynthesizer` w testach = zero modelu przy testach (torch sam w sobie jest dep-em). Nowy `audio_schema.py` (`AudioArtifact`, `ScenarioAudioBundle`) oddzielony od `ScenarioModel`. Rozszerzone `pipeline.run_audio()`, CLI `scripts/run_synthesis.py`, fetcher `scripts/fetch_reference_voices.py` (Mozilla Common Voice PL, CC0). Noise/augmentacje (tempo/pitch/reverb) odłożone do 1b.1. 57 testów zielonych (25 z 1a + 32 nowe), ruff + mypy czyste. Lint-fixes: 5× `# type: ignore[no-any-return]` na granicy numpy/torch; N803 zamiast CamelCase kwargs w `_synthesize_scenario`. |
| 2026-04-22 | **Phase 1b.1 — kod gotowy (czeka na `noise_library/` + smoke).** Dodane: `pedalboard>=0.9`, `FREESOUND_API_KEY` env. Rozszerzone `audio_postprocess.yaml` o bloki `augment` (pitch ±1 semi, tempo 0.95–1.05, per-turn), `reverb` (small/medium/large room 0.15/0.35/0.55 z wagami 0.4/0.4/0.2, dialog-level), `noise` (stratified SNR buckets 10–25 dB, library uniform sampling). **Pełne implementacje**: `noise_injector.py` (RMS-based gain, `pick_params` z parzystą sekwencją RNG na potrzeby manifestu), `augmentation.py` (pedalboard: `time_stretch`/`PitchShift`/`Reverb`). `pipeline.py` emituje `dialogue.wav` (clean, nietknięty) + `dialogue_noisy.wav` (mix→reverb→resample→noise→normalize); osobny seed `sha256(id + "\|noise")[:8]` żeby tuning noise config nie perturbował clean SHA. `audio_schema.py` rozszerzone o 6 opcjonalnych pól (`dialogue_noisy_*`, `noise_snr_db`, `noise_clip`, `reverb_room`). Nowy flag `--clean-only` w CLI. Nowy fetcher `fetch_noise_library.py` (Freesound, filtr licencji w query = nie ściągnie NC/SA). WER eval loop: pure-Python `wer_calculator.py` (Unicode-punct aware) + `eval_stt_on_dataset.py` (faster-whisper large-v3 PL, clean vs noisy). 84 testy zielone (25 z 1a + 32 z 1b + 19 nowych z 1b.1 + 7 w WER calc + 1 ekstra). ruff + mypy czyste na `dataset_pipeline` i `research_and_testing`. Czeka na: `fetch_noise_library.py` z prawdziwym kluczem Freesound, smoke na prawdziwym XTTS, pierwszy run eval. |
| 2026-04-22 | **Phase 1b.1 ukończona — smoke + pierwszy WER.** Pobranych 11 klipów ambience z Freesound (keyboard=3, HVAC=1, paper=3, footsteps=2, door=2; chatter/clinic query returned 0 — skew akceptowalny przy 11-clip uniform pick). Fix path resolution: `noise_library/` w `audio_postprocess.yaml` zmienione na `../noise_library/` (rozwiązuje się relatywnie do dir configu, więc bez prefiksu szukało `configs/noise_library`). Smoke endo-acromegaly-01: 13 turns, clean + noisy produkowane, `bundle.json` z pełnymi metadanymi (`noise_snr_db=15.2`, `noise_clip=keyboard_*.wav`, `reverb_room=small`). **Pierwszy pomiar WER (faster-whisper large-v3 PL, CPU int8)**: clean 9.12%, noisy 11.23% → delta +2.1pp. Healthy calibration — nie za łagodnie, nie za ostro. Gotchas faster-whisper na Apple Silicon udokumentowane: fp16 crashuje na CPU, używamy `int8`. `.gitignore` rozszerzone o `*.wav/*.mp3/*.ogg/*.flac` + whitelisted README w `reference_voices/` i `noise_library/` (provenance trzymamy w git, binarki nie). |
| 2026-04-22 | **MLX Whisper jako drugi backend WER eval.** `eval_stt_on_dataset.py` refaktor: `Transcriber` Protocol + `_FasterWhisperTranscriber` / `_MLXWhisperTranscriber`, `--backend {faster-whisper,mlx}` (default `faster-whisper` dla portability). MLX wykorzystuje 10-rdzeniowy GPU M5 przez Metal (`mlx-community/whisper-large-v3-mlx`) — ~5–10× szybciej niż ctranslate2 CPU int8, ta sama jakość `large-v3`. `--backend mlx` jest odtąd preferowaną ścieżką na Apple Silicon. JSON output dostaje pole `backend`. Install: `uv pip install mlx-whisper`. |
| 2026-04-22 | **Phase 1 audit + domknięcie.** Weryfikacja przed startem Phase 2: wszystkie 8 bundli (scenariusze + audio clean/noisy) spójne z manifestem po re-exporcie (`manifest.json`: scenarios=8, audio=8). Skorygowano status Phase 1a z "19 scenariuszy" na faktyczne 8 (1-2 per specjalizacja × 5) — wcześniejsza liczba była z pierwszego planu, nie z disku. Fix ruff RUF002 w `eval_stt_on_dataset.py` (ambiguous `–`/`×` w docstringu zostawionym przez MLX refactor). Pełen eval WER na 8 scenariuszach na MLX: mean clean 9.73%, mean noisy 14.08%, delta +4.35pp — rozpiętość 3.1%–21.1% clean, 3.4%–40.9% noisy. Baseline zapisany w `results/stt_eval_20260422_140634.json`. Faza 1 zamknięta, gotowe do Phase 2. |
| 2026-04-22 | **Phase 1b ukończona — smoke test + determinizm OK.** (1) Common Voice fetcher przepisany na Mozilla Data Collective API (oryginalny publiczny endpoint zwracał 401 bez sesji) — Bearer auth via `MEDVOICEPL_VOICES_API_KEY`, dataset CV Scripted Speech 25.0 — PL. (2) Filtr długości zdania 130–260 znaków (≈ 10–20 s) zamiast 6–10 s — dłuższe refy wyraźnie stabilizują prozodię XTTSv2; relaxation ladder 90/60 znaków gdy filtr za ciasny (zwłaszcza `female_feminine`). (3) Fix szeptania/dryfu tonu: `temperature=0.65`, `sound_norm_refs=True`, `split_sentences=True` wire'owane do `tts.tts()` (z `TypeError` fallback na legacy `enable_text_splitting`). (4) Per-turn RNG seeding: `torch.manual_seed` + `np.random.seed` + `random.seed` — wszystkie trzy potrzebne, coqui-tts sampluje z trzech niezależnych RNG. Seed per-turn = `sha256(scenario.id)[:4] XOR (turn_index * 0x9E3779B1)`. Weryfikacja: dwa kolejne reruny endo-acromegaly-01 na MPS → byte-identyczny `dialogue.wav` (SHA `295633c5…`). Smoke test: 13 turns, 185.5 s audio w 2:28 wall-clock (RTF ~0.5). Manifest ma bloki `scenarios` (8) + `audio` (1) z pełnymi SHA. 57 testów zielonych (bez zmian — `_FakeSynthesizer` ignoruje `seed`). |
| 2026-04-24 | **Phase 2a ukończona — baseline WER/DER na 8 bundlach.** Pełny eval `--backend on-premise` na 8 scenariuszach: **mean clean WER 3.67% / noisy WER 4.68%** (delta +1.0pp), **mean clean DER 32.96% / noisy DER 43.36%**. Trzy interwencje wymagane po pierwszym przebiegu (14.79% clean WER, 33.12% clean DER): (1) **`load_dotenv()` w `eval_stt_on_dataset.py`** — CLI nie czytał `.env`, `HUGGINGFACE_TOKEN` był pusty i pyannote failował cichutko do błędu na pierwszym scenariuszu. Wywołane wewnątrz `def run()` (nie na module-level, żeby uniknąć E402 z ruff). (2) **Fix merge drop w `_merge_words_into_segments`**: strict midpoint-inside bucketing gubił ~15-25% słów których midpoint wpadał w silent gap między segmentami pyannote (pyannote ma ciaśniejszy VAD niż Whisper). Przepisane na nearest-segment assignment z distance=0 dla inside + distance-to-edge dla gap. Regresja WER z 9.12% na 25.09% (neuro-ct) wracała do ~4% po fixie. (3) **Anti-hallucination flag** `condition_on_previous_text=False` w obu backendach Whisper — whisper-large-v3 na studio-quality TTS wpadało w pętle repetycji ("Dziękuję za obejrzenie" przez minuty), co zawalało clean WER neuro-ct na 72.7%. Noisy wariant 6.7% bo szum przerywał pętlę. Wyłączenie prev-segment prompt usunęło bias bez dotykania wag. **Role mapping heurystyka — dwie nieudane próby i rewert**: (a) "dominant speaker in first 30 s" wywracał role na 6/8 scenariuszach clean (pacjent mówi dłużej w pierwszych 30 s niż lekarz — lekarz zadaje krótkie pytania, pacjent odpowiada rozlegle), DER 97-100%. (b) "first ≥2 s segment wins" wywracał role na scenariuszach gdzie pyannote na TTS audio zaczynał kaskadą sub-sekundowych fragmentów, pierwszy ≥2 s segment lądował środku rozmowy na losowo sklastrowanym labelu. Rewert do `min(raw_segs, key=.start_s)` — najwcześniejszy segment wygrywa. Wszystkie trzy warianty udokumentowane w docstringu `_build_role_map` jako regression guard. **Pozostałe outliery DER**: b3f0c1a9 noisy 76.8%, kardio-m noisy 78.0% — driven by pyannote clustering collapse pod szumem (wespeaker embedding nie separuje sklonowanych voice'ów XTTSv2 niezawodnie, same physical speaker relabeluje się SPEAKER_00↔SPEAKER_01 w obrębie dialogu). Nie adresowalne na poziomie pipeline'u — artefakt TTS voice similarity, nie błąd kodu; rozmawiamy ponownie przy Phase 4 na realnym klinicznym audio. `num_speakers=2` już jest hard-pinned (silniejsze niż min/max w pyannote 3.1). WhisperX rozważone i odrzucone — dociąga wav2vec jako nowy dep i łamie Protocol symmetry z `FasterWhisperTranscriber`. HuggingFace Hub free tier: model downloads + HEAD revalidation = zero kosztów. Baseline final w `results/stt_eval_20260424_112153.json`. Phase 2a zamknięta, gotowe do Phase 2b (SLM + SOAP extraction). |
| 2026-04-24 | **Phase 2b — switch modelu Bielik v2.3 → v3.0.** `check_slm_models.py` na `mlx-community/Bielik-11B-v2.3-Instruct-4bit` dostał 404/401 "Repository Not Found" — community MLX repack wyparował z HF między momentem pisania planu a pierwszym preflight runem. Kuratowany HF search dla "Bielik-11B-v3" pokazał 8 repo MLX, w tym **pierwszoustronny `speakleash/Bielik-11B-v3.0-Instruct-MLX-4bit` (gated=false)**. Switched do v3 jako default: (1) first-party SpeakLeash quant = stabilna proweniencja; (2) ungated = preflight nie wymaga HF license accept (v3 full-precision jest gated, ale MLX quant nie); (3) v3 używa DPO-Positive + GRPO + Dr. GRPO zamiast SFT+DPO z v2.3 → lepszy instruction-following, co bezpośrednio pomaga w JSON-schema compliance; (4) v3 trenowany na 20M+ instrukcjach (~17B tokens) z multi-turn support. Chat template: ChatML. Zmiany: `.env.example` (SLM_MODEL_ID + SLM_MLX_REPO), `slm_engine.py` (fallback repo), `check_slm_models.py` (fallback repo + `load()` 3-tuple compat), `download_models.sh` (snapshot hint), fixtures test (`slm_model` string). Dokumentacja + prompts.py zaktualizowane żeby anti-hedge guard obejmował obie wersje (RLHF shaping różne, może hedgować inaczej — do weryfikacji na smoke). Pozostało niezweryfikowane: v3 native context length (v2.3 był 4K; jeśli v3 <3000 tokens, disable few-shot na rendererze). 85 testów nadal zielonych, lint + mypy czyste. **Preflight green (2026-04-24)**: `check_slm_models.py` zjechał na 6.29 GB download w 3:29 min (59.8 MB/s), wszystkie hard prerequisites satisfied. Confirmed: **native context 32768 tokens** (v2.3 był 4K — 8× więcej headroom), chat_template present, 1-token smoke OK. Jedyny WARN: `llama-cpp-python` not installed (oczekiwane, jest w `[portable]` extra). |
| 2026-04-24 | **Phase 2b — kod + testy gotowe (czeka na pierwszy smoke Bielik-11B).** Nowy pakiet `application/on_premise/src/on_premise/structuring/` z 5 modułami: `soap_schema.py` (SOAPNote hierarchy, Pydantic v2, `extra="forbid"`, ICD-10 regex `^[A-Z]\d{2}(\.\d{1,2})?$`, vitals Literal whitelist 8 nazw, `provenance: list[int]` na każdej sekcji + red_flags), `slm_engine.py` (`StructuredExtractor` Protocol + `MLXLMExtractor` z chat_template guard + `LlamaCppExtractor` z GGUF path check, `build_extractor(config, env)` factory, `load_slm_config` dup `_expand_env_templates` z 2a, `TransientSLMError`/`PermanentSLMError` mirror Phase 1a), `prompts.py` (anti-hedge `SYSTEM_PROMPT_PL`, kompaktowy kardio `FEW_SHOT_EXAMPLE`, `render_user_prompt` serializuje segmenty jako `[N] speaker: text` dla provenance linking, `render_correction_hint` mirror Phase 1a), `pipeline.py` (`transcript_to_soap()` — extract → validate → retry na ValidationError z correction hint → `_scrub_hallucinated_pii` walk tree na PESEL/phone NOT in transcript → `_build_metadata` z SHA anchor). **Grammar-constrained decoding** przez `lm-format-enforcer` w obu backendach (MLX i llama-cpp) — `SLM_STRUCTURED_MODE={grammar,raw}` toggle dla A/B w evalu. **Model**: `speakleash/Bielik-11B-v3.0-Instruct-MLX-4bit` default (~6 GB 4-bit, Polish-native, pierwszoustrona publikacja SpeakLeash, ungated — switched from v2.3 bo `mlx-community/Bielik-11B-v2.3-Instruct-4bit` wyparował z HF, a v3 ma DPO-Positive + GRPO dla lepszego instruction-following). Dwa CLI: `run_soap_extraction.py` (Typer, `--variant {clean,noisy,both}`, pisze `output/soap_notes/<sid[:8]>/soap_note{,_noisy}.json` gitignored) + `check_slm_models.py` preflight (mlx-lm/llama-cpp/lm-format-enforcer import, MLX repo load, chat_template present, context probe, 1-token smoke). Eval dep-free (mirror WER/DER posture): `soap_quality.py` fills `schema_compliance_rate`, `field_coverage` (19-field bool map), `chief_complaint_recall` (reuse `wer_calculator.normalize` + `_levenshtein`), `diagnosis_match` (bi-substring), `icd10_match` (exact), `plan_nonempty`. `benchmark_slm.py` + `eval_slm_on_transcripts.py` (Typer, `load_dotenv()` wewnątrz `run()` — Phase 2a lekcja, pisze `results/slm_eval_<timestamp>.json`). **Trust boundary**: `test_trust_boundary_imports.py` rozszerzony o 8 Phase 2b symboli; nowy `test_shared_does_not_import_from_on_premise` — AST-walk `application/shared/src/shared/**/*.py`, fail jeśli `Import`/`ImportFrom` zaczyna się `on_premise.`. Łapie prawdziwy failure mode (ktoś fillsuje `shared/schemas/soap.py` z `from on_premise.structuring.soap_schema import SOAPNote`). **85 testów zielonych** (61 on-premise + 24 R&T), lint + mypy czyste. `per-file-ignores` w root `pyproject.toml` dostaje `"**/structuring/prompts.py" = ["RUF001"]` — polskie stringi używają EN DASH dla zakresów liczb (1–2, 2–3) intencjonalnie. `.env.example` dostaje blok Phase 2b (SLM_BACKEND/MODEL_ID/MLX_REPO/GGUF_PATH/DEVICE/MAX_NEW_TOKENS/TEMPERATURE/STRUCTURED_MODE). `mlx-lm>=0.19` + `lm-format-enforcer>=0.10` w main deps, `llama-cpp-python>=0.2.90,<0.3` pod `[portable]` extra (kompiluje się ze źródeł ~3-5 min z `CMAKE_ARGS="-DLLAMA_METAL=on"`, opt-in przez `uv sync --extra portable`). **Protocol gotcha**: `StructuredExtractor` nie jest `@runtime_checkable`, więc test stub-conformance używa `hasattr` duck-check zamiast `isinstance` (isinstance raziłby TypeError). **mlx-lm API gotcha**: `load()` zwraca `(model, tokenizer)` w 0.19–0.20 i `(model, tokenizer, config)` w ≥0.21 — indexujemy `[0], [1]` żeby tolerować obie wersje. Czeka na: `check_slm_models.py` green (v3 MLX repo ungated, więc bez HF license accept), smoke na 1 scenariuszu z `--structured-mode grammar`, pełen eval na 8 bundlach (schema compliance ≥90%, CC recall ≥0.6, diag_match ≥50%, icd10 ≥40%, plan_nonempty ≥80%), A/B grammar vs raw dla writeupu. |
| 2026-04-23 | **Phase 2a — kod gotowy (czeka na HF token pyannote + pierwszy smoke).** Nowy pakiet `application/on_premise/src/on_premise/stt/` z 5 modułami: `schema.py` (`Transcript`/`TranscriptSegment`/`TranscriptionMetadata`/`WordTiming` — Pydantic v2, `extra="forbid"`, SHA anchor; docstring explicit trust-boundary notice), `whisper_engine.py` (`Transcriber` Protocol + `RawWord`/`RawTranscription`, `MLXWhisperTranscriber` i `FasterWhisperTranscriber` z word timestamps, `build_transcriber(config, env)` factory), `diarization.py` (`Diarizer` Protocol + `RawDiarizationSegment`, `PyannoteDiarizer` z token-check PRZED pyannote import — żeby error message surfaceował pierwszy i testy nie potrzebowały `pyannote.audio`, `num_speakers=2` hardcoded), `vad.py` (15-LOC RMS-threshold `detect_silence_segments` dla `--trim-silence` CLI flag), `pipeline.py` (`transcribe_and_diarize()` — merge po midpoincie słowa, role mapping first-speaker=doctor, drop empty segs). Dwa CLI: `run_transcription.py` (Typer, `--variant {clean,noisy,both}`, pisze do `application/on_premise/output/transcripts/<sid[:8]>/transcript.json`, gitignorowane) + `check_stt_models.py` preflight (mlx/faster-whisper/pyannote, red/green table, exit non-zero). Nowy `research_and_testing/evaluation/stt_eval/der_calculator.py` — pure-Python DER (union-of-endpoints time-slicing + bisect, no `pyannote.metrics` dep) + `reference_from_turn_durations` helper który odtwarza reference timeline z `ScenarioAudioBundle.turns`. `eval_stt_on_dataset.py` refactor: importuje `FasterWhisperTranscriber`/`MLXWhisperTranscriber` z `on_premise.stt.whisper_engine` (eval i deploy używają TEGO SAMEGO kodu), nowy `--backend on-premise` → pełna pipeline + WER + DER w jednym przebiegu. **Testy**: 25 nowych w `application/on_premise/tests/` (schema, whisper_engine, diarization, pipeline, vad, trust_boundary_imports) + 8 w `research_and_testing/tests/test_der_calculator.py`. `test_trust_boundary_imports.py` to preview-guard (`pkgutil.iter_modules` po `shared.schemas`, fail jeśli wycieknie któryś z 4 zakazanych symboli) — runtime gate dopisujemy w 2c. 126 testów zielonych (57 z 1b + 42 z 1b.1 + 27 nowych z 2a — zero model downloads, Protocol+Fake pattern z 1b). ruff + mypy czyste. `.env.example` dostaje `WHISPER_BACKEND`/`WHISPER_MLX_REPO`/`HUGGINGFACE_TOKEN`/`PYANNOTE_DEVICE` (i zmiana `WHISPER_COMPUTE_TYPE` default float16→int8, fp16 crashuje na CPU). `.gitignore` rozszerzone o `application/on_premise/output/` — un-anonimizowane transkrypty nie idą do gita. Czeka na: HF license accept + token, `check_stt_models.py` green, smoke na 1 scenariuszu, pełen eval z DER na 8 bundlach. |
