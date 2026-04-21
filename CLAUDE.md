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
  - STT: faster-whisper (transkrypcja + diaryzacja)
  - SLM: lokalny model do strukturyzacji SOAP
  - Anonimizacja: Microsoft Presidio + custom PL recognizery
  - `trust_gate.py` wymusza programistycznie granicę zaufania
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
| STT | faster-whisper |
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
| Phase 1a | `dataset_pipeline`: generacja scenariuszy (JSON) przez Gemini | ✅ ukończone (2026-04-20) — 19 scenariuszy w `output/scenarios/` |
| Phase 1b | `dataset_pipeline`: XTTSv2 + postprocessing audio (clean) | 🔄 kod + testy gotowe (2026-04-21) — czeka na próbki głosów + smoke test |
| Phase 1b.1 | `dataset_pipeline`: noise injection + augmentacje (tempo/pitch/reverb) | ⏳ kolejne (odłożone do ukończenia 1b) |
| Phase 2 | `application/on_premise`: STT + SLM + anonimizacja | ⏳ kolejne |
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

- **Próbki referencyjne głosów**: publiczny korpus Mozilla Common Voice PL (licencja CC0 — łatwa redystrybucja atrybucji). Cztery profile: `doctor_male`, `doctor_female`, `patient_male`, `patient_female`. Skrypt `scripts/fetch_reference_voices.py` jest idempotentny; jeśli filtr Common Voice nie znajdzie kandydata, drukuje specyfikację (mono WAV, 22050 Hz, 6–10 s, czysty pojedynczy mówca) i można podrzucić własne nagrania do `reference_voices/`.
- **Backend TTS**: XTTSv2 przez community fork `coqui-tts` (oryginalny Coqui został zarchiwizowany w 2024). Model ~2 GB cache'owany pod `TTS_MODEL_CACHE` (domyślnie `~/.cache/medvoicepl/xtts`, poza repo).
- **Sprzęt**: Apple Silicon z MPS (CPU fallback, CUDA jeśli dostępne). Autodetekcja przez `TTS_DEVICE=auto`. Przepustowość na MPS ~3–5× realtime → ~4–6 min per scenariusz.
- **Współbieżność**: **serial** over scenarios. XTTSv2 trzyma pojedynczy model na jednym urządzeniu; paralelizm would thrash MPS. Revisit dopiero przy batch inference na prawdziwym GPU.
- **Noise**: odłożone do Phase 1b.1 — `noise_injector.py` jest stubem, `augmentation.py` ma tylko `normalize()`. Pozwala zwalidować jakość TTS niezależnie od tuningu szumu.
- **Output audio**: oba pliki na scenariusz — per-turn WAVs (`turn_NN_<speaker>.wav` przy 22050 Hz) + miks `dialogue.wav` (16 kHz, po crossfade + normalize). Per-turn to ground truth dla STT/diaryzacji; miks to input dla faster-whisper.
- **Determinizm**: seed miksera = `sha256(scenario.id)[:8]`. Ten sam scenariusz → byte-identyczne `dialogue.wav` → stabilny SHA w manifeście.
- **Mapowanie głosu lekarza**: `sha256(scenario_id)[0] % 2` → stabilny per scenariusz, różni się między scenariuszami. Płeć pacjenta bierze się wprost z `ScenarioModel.patient_sex`.
- **Kontrakt audio**: `dataset_pipeline/src/dataset_pipeline/audio_schema.py` (`AudioArtifact`, `ScenarioAudioBundle`) — oddzielny od `ScenarioModel`, żeby tekstowy kontrakt dialogu nie zmieniał się razem z metadanymi audio. Manifest Phase 1b zawiera oba bloki: `scenarios` (z 1a) i `audio` (nowy).
- **Testy bez modelu**: `Synthesizer` Protocol + `_FakeSynthesizer` w testach — żaden test nie ściąga 2 GB XTTSv2. Torch sam w sobie jest zainstalowanym dep-em (używany do resample w testach).

### Phase 1b — kluczowe pliki

| Plik | Rola |
|---|---|
| `dataset_pipeline/pyproject.toml` | Dodane TTS deps: `coqui-tts>=0.27`, `torch`, `torchaudio`, `soundfile`, `numpy`, `requests`, `tqdm` |
| `.env.example` | Dodane `TTS_DEVICE`, `TTS_MODEL_CACHE`, `TTS_ACCEPT_COQUI_LICENSE` |
| `dataset_pipeline/src/dataset_pipeline/tts_engine/voice_manager.py` | `VoiceProfile`, `VoiceCatalog`, `load_profiles`, `profile_for` (mapowanie scenariusz→głos) |
| `dataset_pipeline/src/dataset_pipeline/tts_engine/xtts_synthesizer.py` | `XTTSSynthesizer` (lazy model load, auto-device), `Synthesizer` Protocol, czyszczenie tekstu pod PL prozodia |
| `dataset_pipeline/src/dataset_pipeline/tts_engine/audio_utils.py` | `save_wav`/`load_wav` (soundfile), `resample` (torchaudio), `normalize_peak`/`normalize_rms`, `duration_seconds` |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/dialog_mixer.py` | `mix_turns` (equal-power crossfade + seeded randomized pause), `MixerConfig`, `load_mixer_config` |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/noise_injector.py` | **Stub** — zwraca audio bez zmian; implementacja w Phase 1b.1 |
| `dataset_pipeline/src/dataset_pipeline/postprocessing/augmentation.py` | `normalize()` (peak-normalize); tempo/pitch/reverb → `NotImplementedError` do Phase 1b.1 |
| `dataset_pipeline/src/dataset_pipeline/audio_schema.py` | `AudioArtifact`, `ScenarioAudioBundle` (Pydantic), `bundle_from_dir` (re-hashing z dysku) |
| `dataset_pipeline/src/dataset_pipeline/pipeline.py` | `run_audio()`, `AudioReport`, `_synthesize_scenario` (internal wiring z DI dla testów), `load_bundle_sidecars` |
| `dataset_pipeline/scripts/run_synthesis.py` | Typer CLI: `--scenarios-dir`, `--audio-dir`, `--scenario-id` (repeatable), `--limit`, `--verbose` |
| `dataset_pipeline/scripts/fetch_reference_voices.py` | Typer CLI: pobiera Common Voice PL → konwertuje do 22050 Hz mono WAV, pisze `reference_voices/README.md` |
| `dataset_pipeline/scripts/export_manifest.py` | Rozszerzone: manifest ma dwa bloki (`scenarios`, `audio`), każdy z `count` + `entries` |
| `dataset_pipeline/tests/fixtures/tiny_scenario.json` | Fixture: 6-turn kardiologia (bezpieczne — syntetyczne od początku) |
| `dataset_pipeline/tests/test_{audio_utils,voice_manager,dialog_mixer,synthesis_pipeline}.py` | 32 testy Phase 1b (razem z 1a: 57 zielonych) |

### Phase 1b — uruchomienie

```bash
# 1. Sync + TTS consent (runtime opt-in dla coqui-tts MPL-2.0)
uv sync --all-packages
# W .env: TTS_ACCEPT_COQUI_LICENSE=1, TTS_DEVICE=auto

# 2. Próbki referencyjne (jednorazowo, idempotentne)
uv run python dataset_pipeline/scripts/fetch_reference_voices.py
# lub ręcznie: drop 4 WAV do dataset_pipeline/reference_voices/ (mono, 22050 Hz, 6–10 s)

# 3. Smoke test — 1 scenariusz (~4–6 min na MPS, ~15–20 min na CPU)
uv run python dataset_pipeline/scripts/run_synthesis.py --limit 1

# 4. Manifest
uv run python dataset_pipeline/scripts/export_manifest.py
```

Output: `dataset_pipeline/output/audio/<id[:8]>/turn_NN_<speaker>.wav` × N + `dialogue.wav` (16 kHz) + `bundle.json`. `manifest.json` zawiera bloki `scenarios` + `audio` z SHA-256.

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
