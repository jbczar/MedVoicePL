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
| LLM (chmura) | OpenAI / Anthropic / Google |
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
