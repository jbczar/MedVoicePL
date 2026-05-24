# MedVoicePL

System wspomagania decyzji klinicznych (CDSS) dla polskich lekarzy — hybryda on-prem + chmura. Projekt magisterski (PJATK).

Lekarz nagrywa wizytę, system transkrybuje mowę (MLX Whisper), strukturyzuje rozmowę w notatkę SOAP (Bielik-11B), anonimizuje dane pacjenta zgodnie z RODO (Presidio), i sugeruje leki z refundacją NFZ przez chmurowy agent LangGraph (Vertex Gemini + Qdrant RAG). **Surowe audio i niezanonimizowane dane nigdy nie opuszczają strefy on-premise** — chmura widzi tylko anonimowy SOAP.

## Status faz

| Faza | Zakres |
|---|---|
| Phase 1 | Generacja syntetycznego datasetu (scenariusze Vertex Gemini + XTTSv2 + noise injection) |
| Phase 2 | On-prem: STT + diaryzacja + SLM SOAP + Presidio anonimizacja + trust gate + FastAPI |
| Phase 3 | Cloud agent (LangGraph + RAG + NFZ) + Gateway + Next.js frontend |
| Phase 4 | E2E benchmark na 40 scenariuszach × 8 specjalizacji × 5-min dialogi |

### Wyniki Phase 4 (40 scenariuszy)

| Metryka | Wartość |
|---|---|
| STT clean WER | **4.92%** |
| SOAP schema compliance | **100%** |
| Diagnosis match (semantic, mmlw cosine ≥0.85) | **97.5%** |
| Trust gate pass rate | **100%** |
| Agent drug recall | **79%** |
| NFZ refund accuracy | **80%** |
| Provenance non-empty | **100%** |

Pełny raport: [docs/phase4_report.md](docs/phase4_report.md).

## Struktura monorepo

| Katalog | Opis |
|---|---|
| `dataset_pipeline/` | Generacja syntetycznych nagrań wizyt (scenariusze Vertex Gemini + XTTSv2 + postprodukcja audio + noise injection) |
| `application/on_premise/` | Przetwarzanie lokalne: MLX Whisper STT, pyannote diaryzacja, Bielik SLM strukturyzacja SOAP, Presidio anonimizacja, trust gate |
| `application/cloud/` | Integracja chmurowa: Vertex Gemini agent, Qdrant RAG z 621 ChPL excerptami, NFZ formulary z 619 ATC entries |
| `application/gateway/` | API Gateway (FastAPI :8080), orkiestrator wizyt, bcrypt auth, reverse token map, audit GDPR Art. 30 |
| `application/frontend/` | Pulpit lekarza (Next.js 14 + TypeScript + Tailwind, audio recorder, finalize progress, drug suggestions) |
| `application/shared/` | Współdzielone schematy Pydantic (cross-boundary contracts) |
| `research_and_testing/` | Ewaluacja: STT/SLM/agent benchmarks, ground truth, notebook wizualizacji |
| `docs/` | Dokumentacja: trust boundary, raport Phase 4, diagramy mermaid |

## Wymagania

- Python 3.13 (spaCy 3.8 nie ma kół dla 3.14)
- [uv](https://docs.astral.sh/uv/) (package manager)
- Node.js 20+ (frontend)
- **Apple Silicon (M-series)** dla MLX Whisper + mlx-lm Bielik (testowane na M5/24 GB) — **lub** Linux z CUDA z portable backendem (`faster-whisper` + `llama-cpp-python` przez `uv sync --extra portable`)
- Vertex AI access (Google Cloud) dla cloud agenta i generacji datasetu

## Szybki start

```bash
# 1. Sklonuj repo + uzupełnij sekrety
git clone https://github.com/<owner>/MedVoicePL.git && cd MedVoicePL
cp .env.example .env  # uzupełnij HUGGINGFACE_TOKEN, GOOGLE_CLOUD_PROJECT, GATEWAY_DOCTOR_PASSWORD_HASH

# 2. Vertex auth (raz per maszyna)
gcloud auth application-default login

# 3. Zainstaluj zależności (cały workspace) + frontend
uv sync --all-packages --extra anonymization --python 3.13
make frontend-install

# 4. Demo mode (gateway + frontend, bez modeli i Vertex)
make demo-up
# → otwórz http://localhost:3000 (demo_doctor / demo_password)
```

Dla pełnego live mode (z modelami i Vertex): zobacz [GUIDEBOOK.md](GUIDEBOOK.md).

## Komendy deweloperskie

```bash
make help              # Lista wszystkich komend
make lint              # Linting (ruff)
make format            # Formatowanie kodu
make test              # Testy jednostkowe (~628 testów Python + 10 frontend)
make demo-up           # Demo: gateway + frontend (bez modeli)
make live-up           # Live: on-prem + cloud (z modelami + Vertex)
make eval              # Benchmarki ewaluacyjne
```

## Dokumentacja

- [GUIDEBOOK.md](GUIDEBOOK.md) — przewodnik operacyjny (architektura, demo/live, codzienne workflowy, troubleshooting)
- [docs/phase4_report.md](docs/phase4_report.md) — raport końcowy z benchmarków
- [docs/trust_boundary.md](docs/trust_boundary.md) — kontrakt granicy zaufania on-prem ↔ cloud

## Licencja

MIT
