# Deployment

## Prerequisites

- Docker & Docker Compose v2+
- NVIDIA GPU with CUDA drivers (for production STT/SLM) or Apple Silicon (MPS)
- API keys for at least one cloud LLM provider

## Quick Start (Docker Compose)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys and model preferences

# 2. Build and start all services
docker compose up --build -d

# 3. Verify services
docker compose ps
docker compose logs -f gateway

# 4. Access the system
# Frontend: http://localhost:3000
# Gateway API: http://localhost:8080
# API docs: http://localhost:8080/docs
```

## Service Ports

| Service | Port | Description |
|---|---|---|
| Frontend | 3000 | Doctor's dashboard |
| Gateway | 8080 | API Gateway |
| VectorDB | 8000 | ChromaDB |

## GPU Configuration

### NVIDIA (Linux)
The `docker-compose.yml` includes GPU reservation. Ensure the NVIDIA Container Toolkit is installed.

### Apple Silicon (macOS)
Run on-premise services natively (outside Docker) with `WHISPER_DEVICE=mps` and `SLM_DEVICE=mps`.
