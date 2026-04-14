# Trust Boundary

## Overview

MedVoicePL enforces a strict trust boundary between on-premise and cloud processing to comply with GDPR (RODO) requirements.

## Invariants

1. **Raw audio** never leaves the on-premise module.
2. **Unredacted transcripts** never leave the on-premise module.
3. **PII (Personally Identifiable Information)** is stripped before any data crosses the boundary.
4. Every boundary crossing is validated by `trust_gate.py`.
5. Every anonymization operation is audit-logged for compliance.

## Boundary Enforcement

### Network Level
- Docker Compose defines an `internal` network with `internal: true` (no internet access).
- On-premise containers are connected exclusively to this network.
- Only the gateway can communicate across both `internal` and `cloud` networks.

### Application Level
- `trust_gate.py` validates every outbound payload against `trust_boundary.yaml`.
- The `AnonymizedPayload` Pydantic model (in `shared/schemas/anonymized.py`) serves as the formal contract.
- Regex patterns in `trust_boundary.yaml` scan for potential PII leaks (PESEL, NIP, phone numbers).

### Audit Level
- `audit_logger.py` records every anonymization operation with timestamps and entity types.
- Logs are stored locally and never transmitted externally.

## Allowed Outbound Fields

Only these fields may cross the trust boundary (on-premise → cloud):

| Field | Description |
|---|---|
| `anonymized_transcript` | Transcript with all PII replaced by tags |
| `soap_note` | Structured SOAP note (anonymized) |
| `extracted_symptoms` | List of extracted symptoms |
| `extracted_drugs` | List of mentioned drugs |
| `visit_metadata.specialization` | Medical specialization |
| `visit_metadata.timestamp` | Visit timestamp |
| `visit_metadata.session_id` | Anonymous session identifier |
