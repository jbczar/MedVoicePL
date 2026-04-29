# ChPL Excerpts — Curated Polish Summary Drug References

This directory holds **hand-curated** Polish-language summary excerpts of
ChPL (Charakterystyka Produktu Leczniczego) entries for the ~12 drugs
that the Phase 3a 8-scenario test corpus needs. Each file is keyed by
ATC code in its filename and frontmatter.

## Why curated, not full ingestion

URPL (Urząd Rejestracji Produktów Leczniczych) does not offer a bulk
download or public API. Per-drug PDFs require an institutional
agreement with URPL. Phase 3a uses **public-summary excerpts only** —
sourced from public ChPL documents and published Polish clinical
guidelines (PTK, PTL public summaries).

A future-URPL plan (Phase 4) is documented in the project root
`CLAUDE.md` once an institutional path is secured.

## File format

Each `.md` file has YAML frontmatter:

```yaml
---
atc_code: C07AA05
drug_name: Propranolol
source: ChPL public summary; PTK guidelines
accessed_at: 2026-04
licensing: public-summary
---
```

Body is plain Polish — the agent's RAG retriever returns the body as
context for the LLM's `compose_suggestions` node.

## Disclaimer

These summaries are **not** a substitute for the official ChPL. They
support the cloud agent's RAG step; the doctor remains the authority on
prescribing decisions.
