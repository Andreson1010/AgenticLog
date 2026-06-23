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
- [x] REC-04 — CLI incremental por padrão (`python -m agenticlog.rag` sem flags; `--rebuild` para reconstrução) — PR #44

### Chunking & Retrieval
- [x] Semantic chunking (`SemanticChunker`, breakpoint percentile) — ADR-013 — PR #41
- [x] Roteamento retrieve-first (`gerar` vira fallback; decisão só `usar_web` vs `retrieve`) — PR #45
- [x] Rebuild sem duplicação (`_resetar_colecao` antes de gravar; `--rebuild` deixa de anexar) — PR #45
- [x] UI compacta (parágrafos, fontes só com nome do arquivo, Enter via `on_change`, XSS corrigido) — PR #45

### Auditoria RAG — Correções P0 (relatório `docs/rag-audit-2026-06-23.md`)
- [x] Purga de segmentos órfãos no rebuild (reset por coleção, wipe condicional) — ADR-014 — PR #51
- [x] Guardrail fail-loud no rebuild + WARNING em retrieval vazio — ADR-015 — PR #51
- [x] Distância cosine + normalização de embeddings unificada — ADR-016 — PR #51
- [x] Geração única de candidata (`NUM_CANDIDATE_RESPONSES=1`, temp 0) — ADR-017 — PR #51

---

## Backlog

Recomendações priorizadas da auditoria RAG (`docs/rag-audit-2026-06-23.md`) ainda não
realizadas. Prioridade pela rubrica do relatório (impacto ÷ esforço); evidência ancorada nas
métricas da Fase 2 (baseline sintético n=6) quando disponível.

### P1 — mensurar e ampliar recall
- [ ] **Golden set de avaliação + `rag_eval` no CI** — criar `evals/rag_golden.json` (perguntas
  + ground-truth) e plugar `scripts/rag_eval.py` no CI. Destrava Context Recall e Answer
  Correctness (hoje "requer golden set") e cria rede de regressão. *Evidência:* métricas só
  mensuráveis em baseline sintético; sem trava de regressão para a coleção-vazia voltar.
- [ ] **Aumentar top-k (3→5–8) + guardrails de tamanho de chunk** — `RETRIEVAL_K_TOTAL` maior e
  min/max + overlap no `SemanticChunker`. *Evidência:* Hit Rate 0.83 (< 0.9); chunks sem teto
  de tamanho.

### P2 — qualidade de retrieval (estrutural)
- [ ] **Re-ranking cross-encoder no top-N** — re-ordenar os candidatos do retriever antes da
  geração. *Evidência:* 1 caso real com Context Precision 0.0 (chunks errados no topo);
  priorizar se MRR cair < 0.7 no golden set.
- [ ] **Hybrid search (BM25 + denso)** — combinar busca lexical e semântica. *Evidência:*
  logística usa códigos/termos exatos que a busca densa pura erra.

### P3 — refinamentos
- [ ] **Filtro por metadado na consulta (doc_type/page)** — usar os metadados já gravados
  (REC-01) como filtro no retriever; hoje são peso morto no query-time.
- [ ] **Query transformation (rewriting / multi-query / HyDE)** — hoje a query vai literal ao
  retriever.
- [ ] **Parsing JSON estruturado + cleaning de boilerplate PDF** — preservar aninhamento em vez
  de `tostring`; remover cabeçalhos/rodapés repetidos.

### Dívida técnica (achado menor da auditoria)
- [ ] **Contrato de `AgentState.ranked_response`** — tipado `str` mas `rank_respostas` atribui
  dict `{"answer": ...}`. Normalizar para `str` (ou ajustar o tipo + consumidores).

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
| Semantic chunking | .specs/features/semantic-chunking/spec.md | #41 | Done |
| Roteamento retrieve-first + UI compacta | — | #45 | Done |
