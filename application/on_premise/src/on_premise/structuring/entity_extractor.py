"""Medical entity extraction: symptoms, drugs, dosages (NER).

**Deferred to Phase 2c** — this is Presidio-adjacent NER territory.
The trust gate (2c) wires Microsoft Presidio for PII anonymization;
clinical-entity extraction for structured downstream lookups (e.g.
matching medications to NFZ refund codes) belongs to the same pass so
we only walk the SOAP text once.

Phase 2b uses the SLM's own extraction (see `slm_engine.py` + the
`SOAPNote` schema) — no separate NER step is required to fill the
SOAP buckets. This module stays a stub until 2c.
"""
