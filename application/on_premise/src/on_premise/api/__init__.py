"""FastAPI surface for the on-premise pipeline.

Phase 2c: a single endpoint that wraps `transcript → SOAP → anonymize
→ trust_gate` and returns a cross-boundary-safe `AnonymizedSOAPNote`.
Localhost-only by default; the gateway (Phase 3) is the only intended
external caller.
"""
