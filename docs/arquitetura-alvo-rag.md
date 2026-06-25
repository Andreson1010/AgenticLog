# Arquitetura-alvo — redesign RAG por fronteira offline/online

**Status:** Alvo aprovado (decisão de 2026-06-25) — ver [ADR-018](adr/ADR-018-redesign-arquitetura-offline-online.md)
**Escopo:** redesign da camada de ingestão/recuperação; **não** é o split barato por coesão.
**Relacionado:** `docs/rag-audit-2026-06-23.md` (auditoria dos 12 estágios), `.specs/project/ROADMAP.md`.

---

## 1. Por que

Estado atual (2026-06-25):

- `src/agenticlog/rag.py` tem **1013 linhas** e mistura concerns: segurança, extração PDF,
  embeddings, metadados, chunking, escrita no Chroma, upsert incremental e CLI num arquivo só
  (viola o teto de 800 linhas do coding-style).
- `src/agenticlog/agent.py` concentra retrieval + geração + ranqueamento + grafo LangGraph.
- Os **estágios do pipeline RAG não são unidades isoláveis/testáveis** individualmente; cada
  função de ingestão (`cria_vectordb`, `adicionar_documento_incrementalmente`,
  `adicionar_pdf_incrementalmente`) roda a sequência inteira (load → clean → chunk → embed →
  store) "verticalmente".

A auditoria (`docs/rag-audit-2026-06-23.md`) e a pesquisa de boas práticas (ver §6) apontam o
mesmo norte: organizar o código pela **fronteira offline/online** e expor os estágios como
módulos coesos e trocáveis.

---

## 2. Princípio-dorsal: separar OFFLINE de ONLINE

A fronteira mais importante de um sistema RAG **não** é "uma pasta por estágio" — é o corte
entre **construir o índice** (batch, antes da query) e **consumir o índice** (por request).
A indústria converge nisso: Azure ("RAG data pipeline flow" vs "RAG application flow") e o
*LLM Engineer's Handbook* (padrão **FTI — Feature/Training/Inference**).

| OFFLINE — data/feature pipeline | ONLINE — inference pipeline |
|---|---|
| roda em batch, antes da query | roda por request |
| load → parse → clean → chunk → enrich → embed → **persist** | embed(query) → retrieve → re-rank → augment → generate |
| constrói a "memória" (vector DB) | lê a "memória" e responde |
| hoje: `rag.py` | hoje: `agent.py` |

Segunda regra: **componentes trocáveis** (retriever, embedder, vector store, parser, generator)
atrás de interfaces, para poder trocar embedding model / vector DB sem reescrever o resto
(casa com o Repository Pattern do coding-style). Terceiro: **cross-cutting** (config, eval,
observabilidade) fica fora dos dois fluxos.

---

## 3. Árvore-alvo

> Right-sizing: pasta-por-micro-estágio (ex.: `/cleaning`, `/chunking` isolados em diretórios)
> é over-engineering para um projeto deste tamanho. Os estágios viram **módulos** dentro de dois
> pacotes (`ingestion/`, `retrieval/`), não diretórios próprios.

```
src/agenticlog/
├── config.py                 # cross-cutting (single source of truth — permanece)
│
├── ingestion/                # OFFLINE — escrita do índice
│   ├── __init__.py           # API pública do pacote
│   ├── security.py           # validação/sanitização (path traversal, chaves proibidas, nomes)
│   ├── extraction.py         # extração: PDF (PyMuPDF) + JSON (JSONLoader/jq_schema)
│   ├── cleaning.py           # descarte de page_content vazio (ADR-011) e boilerplate
│   ├── chunking.py           # SemanticChunker (ADR-013)
│   ├── embeddings.py         # modelo de embedding (singleton; compartilhado c/ retrieval)
│   ├── metadata.py           # enriquecimento de metadados + hash (REC-01)
│   ├── store.py              # escrita no Chroma, _resetar_colecao, upsert atômico
│   ├── orchestrator.py       # encadeia os estágios (cria_vectordb / ingerir_incrementalmente)
│   └── cli.py                # entrypoint `python -m agenticlog.ingestion`
│
├── retrieval/                # ONLINE — leitura do índice
│   ├── __init__.py
│   ├── retriever.py          # _get_retriever, fan-out multi-coleção, (futuro) filtro/metadata
│   ├── generation.py         # geração + ranqueamento por cosseno
│   ├── graph.py              # LangGraph FSM (decision → retrieve → generate → rank)
│   └── state.py              # AgentState (Pydantic)
│
├── observability/            # cross-cutting
│   ├── __init__.py
│   ├── logging.py            # _JsonFormatter / configuração de logging
│   └── history.py            # audit log (SQLite)
│
├── serving/                  # camada de entrega
│   ├── __init__.py
│   ├── api.py                # FastAPI
│   └── health.py             # health check LMStudio
│
└── shared/                   # contratos compartilhados
    ├── __init__.py
    └── errors.py             # RAGSecurityError e exceções de domínio
```

`embeddings.py` mora em `ingestion/` mas é **importado por `retrieval/`** — a query e os
documentos precisam usar o MESMO espaço vetorial (ver "Silent-degradation risk" no CLAUDE.md).
Alternativa: promover `embeddings.py` para `shared/` se o acoplamento incomodar.

---

## 4. Mapa estágio → módulo

| Estágio (12 estágios da auditoria) | Módulo-alvo | Origem hoje |
|---|---|---|
| Data ingestion | `ingestion/orchestrator.py` + `cli.py` | `rag.py: ingerir_incrementalmente, cria_vectordb` |
| Parsing & extraction | `ingestion/extraction.py` | `rag.py: extrair_texto_pdf, JSONLoader` |
| Cleaning | `ingestion/cleaning.py` | `rag.py` (filtros ADR-011, inline hoje) |
| Chunking | `ingestion/chunking.py` | `rag.py: SemanticChunker` |
| Embedding | `ingestion/embeddings.py` | `rag.py: _get_rag_embedding_model` |
| Metadata | `ingestion/metadata.py` | `rag.py: _enriquecer_metadados_chunks, _hash_*` |
| Vector DB / store | `ingestion/store.py` | `rag.py: Chroma write, _resetar_colecao, upsert` |
| Retrieval | `retrieval/retriever.py` | `agent.py: _get_retriever` |
| Re-ranking (gap — P2) | `retrieval/retriever.py` (slot futuro) | — |
| Query transformation (gap — backlog) | `retrieval/retriever.py` (slot futuro) | — |
| Context construction + Generation | `retrieval/generation.py` + `graph.py` | `agent.py: retrieve_info, gera_*, rank_*` |
| Observability | `observability/` | `config.py: _JsonFormatter`, `history.py` |
| Evaluation | `scripts/rag_eval*.py` (permanece) | já externo a `--cov=agenticlog` |

Os dois gaps de retrieval (re-ranking, query transformation) ganham **slot natural** em
`retrieval/retriever.py` — o redesign os torna baratos de plugar depois.

---

## 5. Implicações da escolha "redesign" (custo honesto)

Chegar nessa árvore por estágio exige **decompor as funções verticais** de ingestão
(`cria_vectordb`, `adicionar_*`) em passos `extract() → clean() → chunk() → embed() → store()`
reconectados por `orchestrator.py`. Consequências:

- **Muda comportamento de implementação** (não é "mover código"): deixa de ser refactor puro.
- **Atomicidade do upsert** (PRs #42/#43: backup → move → chunk → add → rollback-em-falha)
  precisa ser **reprojetada** para atravessar a fronteira de módulos sem perder a garantia
  transacional. É o ponto mais delicado do redesign.
- **Testes serão reescritos.** Os ~638 `@patch("agenticlog.rag.*")` da suíte atual deixam de
  valer; o redesign é a oportunidade de trocar mock-por-atributo-de-módulo por
  **injeção de dependência / fixtures** (testes robustos a futuras mudanças de estrutura).
- **Perda do oráculo durante a transição:** mexer em código E testes ao mesmo tempo remove a
  rede de segurança. **Mitigação obrigatória:** manter uma camada de **testes de caracterização
  de ponta-a-ponta** (entrada → vector DB → resposta) que NÃO dependem de patches internos,
  como oráculo estável enquanto os módulos são extraídos. O gate `rag_eval` (golden set, PR #53)
  já serve de rede comportamental do lado de retrieval.

---

## 6. Fontes (boas práticas pesquisadas)

- [Azure Architecture Center — Design and develop a RAG solution](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide) — separação data pipeline flow vs application flow.
- [LLM Engineer's Handbook — FTI (Feature/Training/Inference) pipelines](https://www.oreilly.com/library/view/llm-engineers-handbook/9781836200079/Text/Preface.xhtml) — formalização do corte offline/online.
- [Unstructured — Best Practices for RAG Systems in Production](https://unstructured.io/insights/rag-systems-best-practices-unstructured-data-pipeline) — módulos logáveis/testáveis/replayáveis; connectors/parsers.
- [Humanloop — 8 RAG Architectures](https://humanloop.com/blog/rag-architectures) — componentes trocáveis.
- [Towards Data Science — Six Lessons Learned Building RAG Systems](https://towardsdatascience.com/six-lessons-learned-building-rag-systems-in-production/) — auditabilidade/traceabilidade de chunks recuperados.

---

## 7. Faseamento sugerido (a detalhar em spec/feature-factory)

1. **Rede de caracterização** — escrever testes E2E sem patches internos (oráculo estável).
2. **`shared/` + `observability/`** — extrações de baixo risco, sem decompor fluxo.
3. **`ingestion/` (estágios)** — extrair `security/extraction/cleaning/chunking/embeddings/metadata/store`,
   depois reprojetar `orchestrator.py` recompondo a atomicidade do upsert.
4. **`retrieval/`** — separar `retriever/generation/graph/state`.
5. **`serving/`** — mover `api.py`/`health.py`.
6. **Reescrita de testes** com injeção de dependência, removendo os `@patch("agenticlog.rag.*")`.

> Este documento é o **norte arquitetural**. A execução (spec, ordem fina, ADRs por decisão de
> design) será conduzida pelo pipeline feature-factory quando a feature for priorizada.
