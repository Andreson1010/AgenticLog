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

### Auditoria RAG — P1 (mensurar recall)
- [x] Golden set de avaliação + `rag_eval` no CI (`evals/rag_golden.json`, job `rag-eval`, gate Hit≥0.7/MRR≥0.6, métricas embedding-only) — PR #53

---

## Backlog

Recomendações priorizadas da auditoria RAG (`docs/rag-audit-2026-06-23.md`) ainda não
realizadas. Prioridade pela rubrica do relatório (impacto ÷ esforço); evidência ancorada nas
métricas da Fase 2 (baseline sintético n=6) quando disponível.

### P1 — mensurar e ampliar recall
- [x] ~~**Golden set de avaliação + `rag_eval` no CI**~~ — entregue na PR #53 (ver Done acima).
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
  (REC-01) como filtro no retriever; hoje são peso morto no query-time. *Evidência:*
  `_get_retriever` (`agent.py:197`) chama `as_retriever(search_type="similarity",
  search_kwargs={"k": k})` sem cláusula `filter`/`where` — os metadados (`source`, `page`,
  `doc_type`, `file_hash`, `chunk_index`) só servem para dedup/exibição de fonte, nunca filtram
  a busca (ex.: "só PDFs", "página > 10").
- [ ] **Query transformation (rewriting / multi-query / HyDE)** — hoje a query vai literal ao
  retriever. *Evidência:* `passo_decisao_agente` (`agent.py:291`) só faz roteamento por keyword
  (`usar_web` vs `retrieve`), não reescreve/expande a pergunta; `retrieve_info` (`agent.py:340`)
  embeda `state.query` como veio. Sem rewriting, expansão, multi-query ou HyDE.
- [ ] **Parsing JSON estruturado + cleaning de boilerplate PDF** — preservar aninhamento em vez
  de `tostring`; remover cabeçalhos/rodapés repetidos.

### Observabilidade (dívida — gap dos 12 estágios do pipeline)
Hoje há **observabilidade básica**: logging estruturado em JSON (`_JsonFormatter`,
`config.py:166`), audit log de toda query (`HistoryStore`, `api.py:161/326`) e logs de
estágio (`agent.py:358` — "Retrieval retornou N documentos"). **Faltam** os sinais que
permitem diagnosticar *por que* uma resposta saiu ruim:
- [ ] **Tracing por estágio (span)** — instrumentar cada nó do grafo (decision → retrieve →
  generate → rank) com spans para ver latência e dados de entrada/saída por estágio.
  Candidatos: LangSmith ou OpenTelemetry/Phoenix.
- [ ] **Métricas de latência e custo** — tempo por estágio, nº de tokens, nº de chunks
  recuperados por query; expor em logs/endpoint para acompanhar regressão de performance.
- [ ] **Persistir scores de retrieval no audit log** — gravar no `HistoryStore` os chunks
  recuperados e seus scores de cosseno (hoje só a resposta final é auditada), permitindo
  depuração offline de retrieval ruim sem reproduzir a query.

### Dívida técnica (achado menor da auditoria)
- [ ] **Contrato de `AgentState.ranked_response`** — tipado `str` mas `rank_respostas` atribui
  dict `{"answer": ...}`. Normalizar para `str` (ou ajustar o tipo + consumidores).

### Manutenibilidade (feature-factory)

> **DECISÃO 2026-06-25 ([ADR-018](../../docs/adr/ADR-018-redesign-arquitetura-offline-online.md)):**
> o split por coesão abaixo foi **substituído** por um **redesign por fronteira offline/online**.
> Árvore-alvo e faseamento em [`docs/arquitetura-alvo-rag.md`](../../docs/arquitetura-alvo-rag.md).
> Motivos: o split barato só reorganizava (não reflete os estágios RAG nem o corte offline/online)
> e a migração de ~638 patches não trazia ganho arquitetural. O item abaixo fica como histórico.

- [ ] **Redesign por fronteira offline/online** — `ingestion/` (security/extraction/cleaning/
  chunking/embeddings/metadata/store/orchestrator/cli) + `retrieval/` (retriever/generation/
  graph/state) + `observability/`/`serving/`/`shared/`. **Não é refactor puro** (decompõe as
  funções verticais de ingestão; reprojeta atomicidade do upsert; testes reescritos com injeção
  de dependência). Ver ADR-018 e `docs/arquitetura-alvo-rag.md`. *Pronto para feature-factory.*

- [ ] ~~**Split de `src/agenticlog/rag.py` (1013 linhas) → pacote `rag/`** (refactor puro por
  coesão)~~ — **superado pelo redesign acima** (ADR-018). Mantido como histórico:

  **Módulos propostos** (cada um < 400 linhas):
  | Módulo | Funções | Responsabilidade |
  |--------|---------|------------------|
  | `errors.py` | `RAGSecurityError` | Exceção compartilhada |
  | `embeddings.py` | `_get_rag_embedding_model` + singleton | Modelo de embedding (1/processo) |
  | `security.py` | `_valida_path_documentos`, `_valida_json_sem_chaves_proibidas`, `_valida_arquivos_json`, `_sanitizar_nome_arquivo`, `_sanitizar_nome_colecao`, `sanitizar_nome_colecao` | Validação/sanitização |
  | `metadata.py` | `_computar_hash_conteudo`, `_hash_arquivo`, `_enriquecer_metadados_chunks` | Hash + metadados (REC-01) |
  | `pdf.py` | `extrair_texto_pdf` | Extração de texto PDF (PyMuPDF) |
  | `uploads.py` | `salvar_documento_enviado`, `salvar_pdf_enviado` | Persistir upload do operador |
  | `incremental.py` | `_backup_arquivo`, `_reverter_disco`, `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente` | Ingestão incremental + upsert atômico |
  | `build.py` | `_outras_colecoes_existem`, `_resetar_colecao`, `cria_vectordb`, `reconstruir_vectordb` | Rebuild completo |
  | `cli.py` | `_configurar_logging_cli`, `_executar_main`, entrypoint | CLI `python -m agenticlog.rag` |
  | `__init__.py` | re-export da API pública | Compat: `from agenticlog.rag import X` segue válido |

  **Dedup incluído:** `adicionar_documento_incrementalmente` e `adicionar_pdf_incrementalmente` são
  ~90% idênticas — extrair `_upsert_atomico(...)` compartilhado (cai `incremental.py` de ~360 → ~200).

  **⚠️ Risco crítico (testes):** ~8 testes usam `@patch("agenticlog.rag._resetar_colecao")` e outros
  patches em `agenticlog.rag.X`. Após o split, o alvo do patch muda para o submódulo onde a função é
  **usada** (ex.: `agenticlog.rag.build._resetar_colecao`) — o re-export do `__init__` cobre *imports*
  mas **não** referências internas/patches. Migrar todos os `@patch` em `tests/test_rag.py`. Mesma
  atenção em `agent.py`, `app.py` e `scripts/rag_eval.py` (imports cobertos; patches/refs internos não).

  *Critérios de aceite:* suíte verde sem mudança de comportamento; cada módulo < 400 linhas;
  `ruff`/`mypy`/`bandit` (escopo `src/`) limpos; `python -m agenticlog.rag [--rebuild]` inalterado;
  API pública importável de `agenticlog.rag` preservada.

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
