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
- [ ] Health check before first workflow invocation

### Observability
- [ ] Replace print() with logging module across rag.py + agent.py
- [ ] Structured log output with log level config in config.py

### Configuration
- [ ] Load LLM credentials from .env (remove hardcoded values in config.py)
- [ ] Move routing keyword lists to config.py constants

### Quality
- [ ] Integration test for agent_workflow.invoke() (HTTP-level LMStudio mock)
- [ ] Streamlit UI test (streamlit.testing)
- [ ] GitHub Actions CI with coverage gate (--cov-fail-under=80)

### Features (future)
- [ ] Document ingestion via Streamlit UI
- [ ] REST API via FastAPI (already installed)
- [ ] Multi-document collection support in ChromaDB
- [ ] Query history and audit logging

## Feature Specs

| Feature | Spec | Status |
|---|---|---|
| Portuguese docstrings | .specs/features/portuguese-docstrings/spec.md | Done |
| Retry LLM calls | .specs/features/retry-llm-backoff/spec.md | Done |
