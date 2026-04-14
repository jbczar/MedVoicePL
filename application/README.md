# Application — MedVoicePL CDSS

The main Clinical Decision Support System. Architecture is split across four modules:

- **`on_premise/`** — Local processing (Trust Boundary: internal). Raw audio never leaves.
- **`cloud/`** — Cloud integration (Trust Boundary: external). Anonymized data only.
- **`gateway/`** — API Gateway bridging on-premise and cloud.
- **`frontend/`** — Doctor's dashboard (Next.js).
- **`shared/`** — Shared Pydantic schemas and utilities.
