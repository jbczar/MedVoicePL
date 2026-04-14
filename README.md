# MedVoicePL

Multimodalny, wieloagentowy system hybrydowy do wspomagania decyzji klinicznych (CDSS).

Automatyzuje tworzenie notatek medycznych z nagrań wizyt lekarskich w języku polskim. Wykorzystuje lokalny STT i SLM do strukturyzacji, anonimizuje dane pacjentów (RODO) za pomocą Microsoft Presidio, a następnie sugeruje leki z refundacją NFZ poprzez chmurowy moduł RAG. Ostateczna decyzja kliniczna zawsze należy do lekarza.

## Struktura monorepo

| Katalog | Opis |
|---|---|
| `dataset_pipeline/` | Generowanie syntetycznych nagrań wizyt (scenariusze LLM + XTTSv2 + postprodukcja audio) |
| `application/on_premise/` | Przetwarzanie lokalne: STT, strukturyzacja SOAP, anonimizacja (trust boundary) |
| `application/cloud/` | Integracja chmurowa: LLM, RAG, baza refundacji NFZ |
| `application/gateway/` | API Gateway, orkiestrator międzymodułowy |
| `application/frontend/` | Pulpit lekarza (Next.js) |
| `application/shared/` | Współdzielone schematy Pydantic |
| `research_and_testing/` | Ewaluacja: benchmarki STT/SLM/RAG/anonimizacji, analiza wydajności |

## Wymagania

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker & Docker Compose
- Node.js 20+ (frontend)
- GPU z CUDA lub Apple Silicon (MPS) — zalecane dla STT/SLM

## Szybki start

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/<owner>/MedVoicePL.git
cd MedVoicePL

# 2. Skopiuj i uzupełnij zmienne środowiskowe
cp .env.example .env

# 3. Zainstaluj zależności Python (cały workspace)
make install

# 4. Zainstaluj zależności frontend
make install-frontend

# 5. Pobierz modele ML
make download-models

# 6. Uruchom system (Docker Compose)
make up
```

## Komendy deweloperskie

```bash
make help              # Lista wszystkich komend
make lint              # Linting (ruff)
make format            # Formatowanie kodu
make typecheck         # Type checking (mypy)
make test              # Testy jednostkowe
make up / make down    # Start / stop Docker Compose
make generate-dataset  # Generacja syntetycznego datasetu
make eval              # Benchmarki ewaluacyjne
```

## Licencja

MIT
