# Roadmap

## Done

- [x] LangGraph 6-node FSM (decision → retrieve/generate/web → rank)
- [x] ChromaDB RAG pipeline with security validation (path traversal, forbidden keys, size limits)
- [x] Cosine similarity ranking across 5 candidate responses
- [x] Streamlit UI with confidence bar + retrieved docs expander
- [x] pytest suite for rag.py + agent nodes (all mocked)
- [x] Module docstrings in Portuguese
- [x] tlc-spec-driven brownfield mapping + project init

## Backlog

### Reliability
- [x] Lazy LLM initialization (fix SPOF at module import)
- [x] Retry logic on LLM calls (httpx.ConnectError handler)
- [x] Health check before first workflow invocation (PR #10)

### Observability
- [x] Replace print() with logging module across rag.py + agent.py (PR #11)
- [x] Structured log output with log level config in config.py (PR #12)

### Configuration
- [x] Move routing keyword lists to config.py constants (PR #15)
- [ ] ~~Load LLM credentials from .env~~ — descartado (LMStudio local, hardcoded aceitável)

### Quality
- [x] Integration test for agent_workflow.invoke() (PR #14)
- [x] Streamlit UI test (streamlit.testing) (PR #14)
- [x] GitHub Actions CI with coverage gate (PR #14)

### Features
- [x] Document ingestion via Streamlit UI — JSON (PR #17)
- [x] Windows reserved name validation (PR #19)
- [x] PDF upload + ingestion via PyMuPDF (PR #21)
- [x] Incremental ChromaDB ingestion without full rebuild (PR #22)
- [x] REST API via FastAPI — POST /query (PR #23)

### Features (backlog)
- [ ] Migração app.py para cliente HTTP da API (REST API client story)
- [ ] Multi-document collection support in ChromaDB
- [ ] Query history and audit logging

## Feature Specs

| Feature | Spec | Status |
|---|---|---|
| Portuguese docstrings | .specs/features/portuguese-docstrings/spec.md | Done |
| Retry LLM calls | .specs/features/retry-llm-backoff/spec.md | Done |
| Health check LMStudio | .specs/features/health-check-lmstudio/spec.md | Done |
| Logging module | .specs/features/logging-module/spec.md | Done |
| Structured log config | .specs/features/structured-log-config/spec.md | Done |
| Document ingestion UI | .specs/features/document-ingestion-ui/spec.md | Done |
| PDF upload ingestion | .specs/features/pdf-upload-ingestion/spec.md | Done |
| Incremental ChromaDB ingestion | .specs/features/incremental-chroma-ingestion/spec.md | Done |
| FastAPI REST API | .specs/features/fastapi-rest-api/spec.md | Done |
