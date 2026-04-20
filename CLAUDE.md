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
| Phase 1a | `dataset_pipeline`: generacja scenariuszy (JSON) przez Gemini | ✅ kod gotowy (2026-04-20) — czeka na smoke test z `GOOGLE_API_KEY` |
| Phase 1b | `dataset_pipeline`: XTTSv2 + postprocessing audio | ⏸ odłożone (wymaga próbek referencyjnych głosów) |
| Phase 2 | `application/on_premise`: STT + SLM + anonimizacja | ⏳ kolejne |
| Phase 3 | `application/cloud` + `gateway` + `frontend` | ⏳ kolejne |
| Phase 4 | `research_and_testing`: ewaluacja E2E | ⏳ kolejne |

### Phase 1a — decyzje projektowe

- **LLM provider**: Google Gemini Developer API (AI Studio), nie Vertex AI. Uwierzytelnianie przez `GOOGLE_API_KEY`. SDK: `google-genai`.
- **Uwaga bilingowa**: klucz z AI Studio **nie** konsumuje kredytów $300 Vertex AI — to osobne produkty. Vertex AI wymaga ADC/service account i `google-cloud-aiplatform`. Migracja możliwa później (`google-genai` wspiera oba tryby).
- **Skala MVP**: 2 scenariusze × 5 specjalizacji = 10 scenariuszy na pierwszy przebieg.
- **Kontrakt danych**: `dataset_pipeline/src/scenario_generator/schema.py` (`ScenarioModel`) — stabilny interfejs dla Phase 1b (TTS) i `research_and_testing`.
- **Enforcement JSON**: Gemini `response_schema=ScenarioModel` (structured output) zamiast parsowania tekstu.
- **Współbieżność**: `ThreadPoolExecutor`, domyślnie 4 wątki (ograniczenie RPM darmowego tieru Gemini). Konfigurowalne przez `DATASET_LLM_MAX_CONCURRENCY`.
- **Retry strategy**: tenacity (exp backoff, 4 próby) dla błędów przejściowych 5xx/429; osobny mechanizm retry z "correction hint" przy błędach walidacji Pydantic (2 próby).
- **Few-shot examples**: pięć ręcznie napisanych scenariuszy (po jednym na specjalizację) w `prompts.py` — nie zastępować outputem modelu, żeby uniknąć drift.

### Phase 1a — kluczowe pliki

| Plik | Rola |
|---|---|
| `dataset_pipeline/src/scenario_generator/schema.py` | `ScenarioModel`, `DialogueTurn`, `Specialization`, `SpeakerRole` — kontrakt Pydantic |
| `dataset_pipeline/src/scenario_generator/prompts.py` | `SYSTEM_PROMPT_PL`, `FEW_SHOT_EXAMPLES`, `render_user_prompt` |
| `dataset_pipeline/src/scenario_generator/llm_client.py` | `GeminiClient`, `LLMClient` Protocol, `TransientLLMError`/`PermanentLLMError` |
| `dataset_pipeline/src/scenario_generator/generator.py` | `ScenarioGenerator` (batch + retry + failures.json), `load_scenarios_config` |
| `dataset_pipeline/src/pipeline.py` | `run_scenarios()` — entrypoint do chainowania z TTS w Phase 1b |
| `dataset_pipeline/scripts/run_generation.py` | Typer CLI: generacja |
| `dataset_pipeline/scripts/validate_dataset.py` | Typer CLI: ponowna walidacja plików JSON |
| `dataset_pipeline/scripts/export_manifest.py` | Typer CLI: `manifest.json` z SHA-256 |
| `dataset_pipeline/tests/` | 25 testów (schema, prompts, generator z fake clientem, validator CLI) |

### Phase 1a — uruchomienie

```bash
cp .env.example .env   # uzupełnić GOOGLE_API_KEY=AI...
uv sync --all-packages
uv run python dataset_pipeline/scripts/run_generation.py --per-specialization 2
uv run python dataset_pipeline/scripts/validate_dataset.py
uv run python dataset_pipeline/scripts/export_manifest.py
```

Output: `dataset_pipeline/output/scenarios/*.json` + `output/manifest.json`. Katalog `output/` jest gitignorowany.

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
