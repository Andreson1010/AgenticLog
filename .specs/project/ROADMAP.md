# Roadmap

## Done

### Foundation
- [x] LangGraph 6-node FSM (decision → retrieve/generate/web → rank)
- [x] ChromaDB RAG pipeline with security validation (path traversal, forbidden keys, size limits)
- [x] Cosine similarity ranking across 5 candidate responses
- [x] Streamlit UI with confidence bar + retrieved docs expander
- [x] pytest suite for rag.py + agent nodes (all mocked)
- [x] Module docstrings in Portuguese
- [x] tlc-spec-driven brownfield mapping + project init

### Reliability
- [x] Lazy LLM initialization — PR #8
- [x] Retry logic on LLM calls (exponential backoff, 3 attempts) — PR #9
- [x] Health check before first workflow invocation — PR #10

### Observability
- [x] Replace print() with logging module — PR #11
- [x] Structured log output (LOG_LEVEL, LOG_FORMAT env vars) — PR #12
- [x] Query history audit logging (SQLite, GET /history) — PR #27

### Configuration
- [x] Routing keywords in config.py constants — PR #15
- [x] LLM provider portability (LLM_MODEL env var, Protocol) — PR #32

### Quality
- [x] Integration tests + Streamlit UI tests + GitHub Actions CI — PR #14
- [x] Lint tooling (ruff, mypy, bandit, pre-commit, CI lint job) — PR #24

### Features
- [x] Document ingestion via Streamlit UI — JSON — PR #17
- [x] Windows reserved name validation — PR #19
- [x] PDF upload + ingestion via PyMuPDF — PR #21
- [x] Incremental ChromaDB ingestion without full rebuild (JSON) — PR #22
- [x] FastAPI REST API — POST /query, GET /health — PR #23
- [x] Streamlit → HTTP client (app.py calls API via httpx) — PR #25
- [x] Multi-collection ChromaDB support — PR #26
- [x] Portuguese multilingual embeddings (paraphrase-multilingual-mpnet-base-v2) — PR #28
- [x] Chunking estrutura-aware (jq_schema por chave, sentence separators, PDF 1 Doc/página) — PR #31
- [x] LLM provider portability (LLM_MODEL env, OpenAI Protocol) — PR #32

### Ingestão Incremental
- [x] REC-01 — Unificar metadados de chunks (`source`, `file_hash`, `chunk_index`, `page`, `doc_type`) — PR #38
- [x] REC-02 — Implementar `adicionar_pdf_incrementalmente()` — PR #39
- [x] REC-03 — Upsert pattern (deleção de chunks antigos antes de re-ingestão; upsert atômico) — PRs #42, #43
- [x] REC-04 — CLI incremental por padrão (`python -m agenticlog.rag` sem flags; `--rebuild` para reconstrução)

---

## Backlog

_Nenhum item pendente na trilha de Ingestão Incremental (REC-01 a REC-04 entregues)._

---

## Feature Specs

| Feature | Spec | PR | Status |
|---|---|---|---|
| Portuguese docstrings | .specs/features/portuguese-docstrings/spec.md | — | Done |
| Retry LLM calls | .specs/features/retry-llm-backoff/spec.md | — | Done |
| Health check LMStudio | .specs/features/health-check-lmstudio/spec.md | — | Done |
| Logging module | .specs/features/logging-module/spec.md | — | Done |
| Structured log config | .specs/features/structured-log-config/spec.md | #12 | Done |
| Document ingestion UI | .specs/features/document-ingestion-ui/spec.md | #17 | Done |
| PDF upload ingestion | .specs/features/pdf-upload-ingestion/spec.md | #21 | Done |
| Incremental ChromaDB ingestion | .specs/features/incremental-chroma-ingestion/spec.md | #22 | Done |
| FastAPI REST API | .specs/features/fastapi-rest-api/spec.md | #23 | Done |
| Chunking estrutura-aware | .specs/features/chunking-estrutura-aware/spec.md | #31 | Done |
| LLM provider portability | .specs/features/llm-provider-portability/spec.md | #32 | Done |
| Unificar metadados de chunks | .specs/features/unificar-metadados-chunks/spec.md | #38 | Done |
| Ingestão incremental PDF | .specs/features/adicionar-pdf-incrementalmente/spec.md | #39 | Done |
