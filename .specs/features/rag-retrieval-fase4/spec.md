# ADR-018 Fase 4 — extracao do pacote `retrieval/` de `agent.py` — Technical Spec

**Path:** `.specs/features/rag-retrieval-fase4/spec.md`
**TLC scope:** complex
**Based on story:** ADR-018 Fase 4 — extrair `state.py`, `retriever.py`, `generation.py`, `graph.py` de `agent.py` (474 ln), deixando `agent.py` com singletons + shims, comportamento byte-identico, e os dois arquivos-oraculo (`tests/test_rag_caracterizacao.py`, `tests/ingestion/test_shims_identidade.py`) com ZERO diff.
**Status:** Awaiting human approval

---

## Problem Statement

`src/agenticlog/agent.py` concentra 474 linhas que misturam concerns de 5 camadas distintas: estado (`AgentState`), recuperacao vetorial (`_get_retriever`, `_get_vector_db`, `_listar_colecoes`), geracao/ranqueamento (`_invoke_chain`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas`, `_llm_retry`, prompts), grafo LangGraph (6 nos, edges, compilacao) e inicializacao (`inicializar_recursos`, `invalidar_vector_db`). Isso viola a regra de arquivo pequeno (max 400 ln) e impede que os modulos de recuperacao sejam testados independentemente uns dos outros — qualquer mudanca no grafo ou na geracao exige recompilar todo `agent.py`. A **Fase 4** da ADR-018 extrai as 4 camadas para `agenticlog.retrieval/{state,retriever,generation,graph}.py`, mantendo `agent.py` como fachada com singletons fisicos e shims de re-export, preservando o monkeypatch do oraculo de caracterizacao (`test_rag_caracterizacao.py`) e o contrato de identidade da Fase 3a (`test_shims_identidade.py`) — ambos com ZERO diff.

## Goals

- [ ] Criar `src/agenticlog/retrieval/state.py` com `AgentState` (Pydantic), re-exportado como shim `is`-identico em `agent.py`.
- [ ] Criar `src/agenticlog/retrieval/generation.py` com `LLMClient` Protocol, `_llm_retry`, `prompt_rag_retrieve`, `prompt_gerar`, `_prompt_web`, `_get_llm`, `_invoke_chain`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas` — cada um re-exportado como shim identidade em `agent.py`.
- [ ] Criar `src/agenticlog/retrieval/retriever.py` com `_get_embedding_model` (via `criar_embedding_model` de `ingestion/embeddings.py` + singleton `_embedding_model` de `agent` via lazy import), `_get_vector_db` + `_listar_colecoes` (wrappers de call-time para `DIR_VECTORDB`), `_get_retriever`, `invalidar_vector_db` — re-exportados como shims em `agent.py` (exceto `_get_vector_db` e `_listar_colecoes` que sao wrappers que resolvem `DIR_VECTORDB` no momento da chamada).
- [ ] Criar `src/agenticlog/retrieval/graph.py` com `passo_decisao_agente`, `usar_ferramenta_web`, `retrieve_info`, `agent_workflow` (compilado), `inicializar_recursos` — `search` global permanece fisicamente em `agent.py`; `usar_ferramenta_web` acessa `search` via lazy import de `agenticlog.agent` no corpo da funcao.
- [ ] Comportamento byte-identico: fluxo, mensagens, ordem de excecoes, `AgentState` field-set, node-set do grafo e FSM routing preservados.
- [ ] `agent.py` reduzido de 474 ln para aproximadamente shims + singletons + wrappers (~180 ln).
- [ ] Dois arquivos-oraculo com `git diff` VAZIO: `tests/test_rag_caracterizacao.py` e `tests/ingestion/test_shims_identidade.py`.
- [ ] Suites 677+ testes mantida; cobertura >= 80%.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Fase 5 (`serving/`) | Fase separada no roadmap ([arquitetura-alvo-rag.md] §7 item 5). |
| Fase 6 (remocao de shims + reescrita de testes) | Fase separada; shims tem marcador `# Re-export shim (ADR-018 Fase 4) — remover na Fase 6`. |
| Re-ranking / query-transform | Slots futuros em `retrieval/retriever.py` (arquitetura-alvo §4). |
| Metadata filtering | Nao implementado; slot futuro. |
| Mudanca do mecanismo de retry | `_llm_retry` moves verbatim com `_invoke_chain` para `generation.py`. |
| Migracao de `_rag_embedding_model` | Permanece em `rag.py` (invariante Fase 3a). |
| Alterar `config.py`, `app.py`, `api.py` ou `__init__.py` | Apenas manter imports — `__init__.py` continua importando `AgentState` e `agent_workflow` de `agenticlog.agent`. |

---

## User Stories

### P1: Pacote `retrieval/` extraido com comportamento identico e zero-diff oracle  (RETR-01..15)

**User Story**: Como mantenedor do AgenticLog, quero as quatro camadas de `agent.py` (estado, recuperacao, geracao, grafo) em modulos coesos e testaveis sob `agenticlog.retrieval`, com shims identity-preserving em `agent.py`, singletons fisicos mantidos, e zero diff nos arquivos-oraculo, para reduzir `agent.py` para ~180 ln, alinhar com a arquitetura-alvo offline/online (ADR-018 §3/§4/§7), e permitir testes independentes dos modulos de retrieval.

**Why P1**: E o objetivo unico e indivisivel da fase; sem ele nao ha entrega da Fase 4.

**Acceptance Criteria**:

1. WHEN os 4 modulos sao criados THEN `state.py`, `retriever.py`, `generation.py`, `graph.py` SHALL ser importaveis sob `agenticlog.retrieval.*` sem `ImportError`, e `import agenticlog.retrieval` SHALL funcionar sem importar `agent` em top-level. *(RETR-01)*

2. WHEN `AgentState` e comparado entre `agenticlog.agent` e `agenticlog.retrieval.state` THEN `agent.AgentState is retrieval.state.AgentState` SHALL ser verdadeiro (identidade `is`), E o field-set SHALL ser identico ao atual: `query`, `next_step`, `retrieved_info`, `possible_responses`, `similarity_scores`, `ranked_response`, `confidence_score` — sem adicao, remocao ou reordenacao. *(RETR-02)*

3. WHEN `_invoke_chain`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas` sao chamados via shim de `agent.py` THEN o comportamento (fluxo, mensagens, ordem de excecoes, retorno) SHALL ser byte-identico ao atual. *(RETR-03)*

4. WHEN `agent_workflow` e compilado em `graph.py` e re-exportado como shim THEN `agent.agent_workflow is retrieval.graph.agent_workflow` SHALL ser verdadeiro (identidade `is`), E o node-set SHALL ser o mesmo 6-nos (`decision`, `retrieve`, `generate_multiple`, `evaluate_similarity`, `rank_responses`, `usar_web`), edges e conditional routing identicos. *(RETR-04)*

5. WHEN `inicializar_recursos()` e chamado via shim THEN SHALL inicializar `_llm`/`_embedding_model`/`_vector_dbs` (singletons que ficam fisicamente em `agent.py`). *(RETR-05)*

6. WHEN `pytest -m integration tests/test_rag_caracterizacao.py` roda no CI Linux THEN SHALL passar com `git diff` VAZIO no arquivo. *(RETR-06)*

7. WHEN `pytest tests/ingestion/test_shims_identidade.py` roda THEN SHALL passar com `git diff` VAZIO no arquivo. *(RETR-07)*

8. WHEN `agent.py` e carregado THEN os singletons `_llm`/`_embedding_model`/`_vector_dbs` SHALL continuar fisicamente em `agent.py` (variaveis globais de modulo), nao movidos para `retrieval/`. *(RETR-08)*

9. WHEN `retrieval/retriever.py` constroi o modelo de embedding THEN SHALL reusar `criar_embedding_model` de `agenticlog.ingestion.embeddings` (mesmo espaco vetorial), envolvendo-o com o singleton `_embedding_model` de `agent.py` via lazy import dentro do corpo de `_get_embedding_model`. *(RETR-09)*

10. WHEN `_llm_retry` e extraido THEN SHALL mover com `_invoke_chain` para `generation.py`, com a mesma politica de retry (`LLM_MAX_RETRY_ATTEMPTS`, `LLM_RETRY_WAIT_INITIAL_SECONDS`, `LLM_RETRY_WAIT_MAX_SECONDS`) e `before_sleep_log`. *(RETR-10)*

11. WHEN `_get_vector_db` e `_listar_colecoes` sao chamados via shim/wrapper THEN `DIR_VECTORDB` SHALL ser resolvido NO MOMENTO DA CHAMADA (wrapper pattern, ADR-019 D4), nao capturado no import — o wrapper de `agent.py` le `agent.DIR_VECTORDB` e passa como argumento a cada chamada. *(RETR-11)*

12. WHEN testes Nao-oraculo com `@patch("agenticlog.agent.<simbolo_movido>")` sao executados THEN os patch-targets SHALL ser migrados para o novo namespace `retrieval.<mod>.<simbolo>` (para simbolos que efetivamente se moveram) ou permanecer em `agent.` (para shims que delegam). *(RETR-12)*

13. WHEN `import agenticlog.retrieval.state`, `import agenticlog.retrieval.generation`, `import agenticlog.retrieval.retriever`, `import agenticlog.retrieval.graph` rodam num interpretador frio THEN SHALL sair com codigo 0 (sem ciclo), verificado por novo teste de aceitacao. *(RETR-13)*

14. WHEN `app.py`, `api.py`, `__init__.py` ou qualquer teste importa de `agenticlog.agent` THEN `from agenticlog.agent import <simbolo>` SHALL continuar resolvendo sem `ImportError` (fachada preservada). *(RETR-14)*

15. WHEN a suite completa de testes roda (`pytest --cov=agenticlog --cov-report=term-missing -v`) THEN SHALL passar com 677+ testes verdes (ou o numero atual) e cobertura >= 80%. *(RETR-15)*

**Independent Test**: `pytest -m integration tests/test_rag_caracterizacao.py -v` verde; `pytest tests/ingestion/test_shims_identidade.py -v` verde; `git diff --stat` VAZIO nos dois; `pytest --cov=agenticlog -v` completo verde no CI Linux com `agent.py ~180 ln`.

---

## Edge Cases

- WHEN `_get_retriever` encontra colecao vazia (0 documentos) THEN SHALL fazer fallback: `next_step = "gerar"`, log WARNING "Retrieval retornou 0 documentos para a query; caindo para geracao direta" e retornar `retrieved_info=[]`.
- WHEN `usar_ferramenta_web` falha (DuckDuckGo indisponivel) THEN SHALL retornar `ranked_response="Busca indisponivel no momento."`, `confidence_score=0.0`.
- WHEN `_llm_retry` exaure todas as tentativas THEN SHALL propagar a ULTIMA excecao (comportamento `reraise=True`).
- WHEN `_get_retriever` faz fan-out multi-colecao THEN SHALL deduplicar por MD5 do `page_content` e limitar ao total `RETRIEVAL_K_TOTAL`.
- WHEN `monkeypatch.setattr("agenticlog.agent._embedding_model", stub)` esta ativo THEN `retrieval.retriever._get_embedding_model` SHALL ler o stub via lazy `from agenticlog.agent import _embedding_model` (import no corpo).
- WHEN `monkeypatch.setattr("agenticlog.agent.search", fake)` esta ativo THEN `retrieval.graph.usar_ferramenta_web` SHALL ver o fake (busca via lazy `from agenticlog.agent import search` no corpo).
- WHEN `monkeypatch.setattr("agenticlog.agent.DIR_VECTORDB", vdb)` esta ativo THEN `_get_vector_db` e `_listar_colecoes` (wrappers em `agent.py`) SHALL ler `agent.DIR_VECTORDB` no momento da chamada e passar como argumento — a funcao movida usa o parametro.
- WHEN `patch.object(agent_mod, "_listar_colecoes", ...)` e usado (test_agent.py) THEN o `agent_mod` importado como `import agenticlog.agent as agent_mod` SHALL ainda ser o namespace correto com `_listar_colecoes` como wrapper.
- WHEN os prompts (`prompt_rag_retrieve`, `prompt_gerar`, `_prompt_web`) sao movidos para `generation.py` e shim em `agent.py` THEN `agent.X is generation.X` SHALL ser verdadeiro (identidade `is`) — testes que comparam instancias de prompt nao quebram.

---

## Requirement Traceability

| Requirement ID | Story | AC | Phase | Status |
|----------------|-------|----|-------|--------|
| RETR-01 (4 modulos importaveis) | P1 | AC1 | Design | Pending |
| RETR-02 (AgentState field-set identico) | P1 | AC2 | Design | Pending |
| RETR-03 (comportamento byte-identico) | P1 | AC3 | Design | Pending |
| RETR-04 (agent_workflow node-set identico) | P1 | AC4 | Design | Pending |
| RETR-05 (inicializar_recursos -> singletons agent.py) | P1 | AC5 | Design | Pending |
| RETR-06 (oraculo test_rag_caracterizacao.py zero-diff) | P1 | AC6 | Design | Pending |
| RETR-07 (test_shims_identidade.py zero-diff) | P1 | AC7 | Design | Pending |
| RETR-08 (singletons _llm/_embedding_model/_vector_dbs em agent.py) | P1 | AC8 | Design | Pending |
| RETR-09 (retriever reusa criar_embedding_model de ingestion) | P1 | AC9 | Tasks | Pending |
| RETR-10 (_llm_retry move com _invoke_chain) | P1 | AC10 | Tasks | Pending |
| RETR-11 (DIR_VECTORDB resolvido no call time) | P1 | AC11 | Design | Pending |
| RETR-12 (patch-targets migrados) | P1 | AC12 | Tasks | Pending |
| RETR-13 (sem circular import) | P1 | AC13 | Tasks | Pending |
| RETR-14 (agent.py fachada preservada) | P1 | AC14 | Design | Pending |
| RETR-15 (suíte 677+ testes mantida) | P1 | AC15 | Tasks | Pending |

**ID format:** `RETR-[NUMBER]` (RETR = RETrieval).

**Coverage:** 15 IDs totais; todos mapeados a tasks (ver `tasks.md`).

---

## Data Model Changes

Nenhuma. Sem alteracao em `AgentState` (field-set identico), schema de metadados de chunk, colecao Chroma, espaco vetorial ou config. Os prompts movidos sao objetos identicos (`is`). O `search` global permanece instancia de `DuckDuckGoSearchAPIWrapper`.

---

## Process / Background Flow

### Happy path — rota retrieve
`app.py` chama `agent_workflow.invoke(AgentState(query=...))` via shim `agent.agent_workflow` -> `graph.agent_workflow` -> `graph.passo_decisao_agente` (keyword routing) -> se "retrieve": `graph.retrieve_info` -> chama `retriever._get_retriever(query)` -> este chama `retriever._listar_colecoes()` (wrapper em `agent._listar_colecoes` que resolve `DIR_VECTORDB` no call time) + `retriever._get_vector_db(name)` (wrapper em `agent._get_vector_db` que resolve `DIR_VECTORDB` no call time) -> retorna documentos -> `graph_workflow` prossegue para `generation.gera_multiplas_respostas` -> `generation.avalia_similaridade` (chama `retriever._get_embedding_model` que le `_embedding_model` singleton via lazy import de `agent`) -> `generation.rank_respostas` -> retorna `ranked_response` + `confidence_score`.

### Happy path — rota web
`passo_decisao_agente` -> `next_step="usar_web"` -> `graph.usar_ferramenta_web` -> `from agenticlog.agent import search` (lazy, no corpo) -> chama `search.run(state.query)` -> invoca `_prompt_web | _get_llm() | StrOutputParser` -> retorna resposta com `confidence_score=0.0`.

### Failure path — retry exhaustion
`_invoke_chain` marcado com `@_llm_retry` -> tenta `chain.invoke(inputs)` ate `LLM_MAX_RETRY_ATTEMPTS` tentativas com backoff exponencial -> se todas falham, propaga a ultima excecao (httpx.ConnectError, TimeoutException, RemoteProtocolError, APIConnectionError).

### Failure path — empty retrieval
`_get_retriever` retorna lista vazia -> `graph.retrieve_info` loga WARNING e seta `next_step="gerar"` -> `gera_multiplas_respostas` usa `prompt_gerar` sem contexto - resposta sem grounding.

### Initialization path
`api.py` chama `inicializar_recursos()` via shim de `agent` -> `graph.inicializar_recursos()` -> `retriever._get_embedding_model()` (cria/cacheia `_embedding_model` via lazy import de `agent`) -> `retriever._get_vector_db(DEFAULT_COLLECTION_NAME)` (wrapper em `agent` que cria/cacheia `_vector_dbs[DEFAULT]`) -> `generation._get_llm()` (cria/cacheia `_llm` via lazy import de `agent`).

---

## API Changes

No API changes. `agent.AgentState`, `agent.agent_workflow`, `agent.inicializar_recursos`, `agent.invalidar_vector_db`, `agent._listar_colecoes` continuam resolvendo (shims ou wrappers em `agent.py`). `__init__.py` continua importando `AgentState` e `agent_workflow` de `agenticlog.agent`.

---

## Frontend Changes

No frontend changes. `app.py` importa `_listar_colecoes` de `agenticlog.agent` – preservado via shim de `retriever` ou wrapper.

---

## Tests Required

**Novos (RETR-13):**
- **Acicidade** (novo teste de aceitacao, ex. `tests/acceptance/test_rag_retrieval_fase4.py`): `subprocess` importando `agenticlog.retrieval.state`, `agenticlog.retrieval.generation`, `agenticlog.retrieval.retriever`, `agenticlog.retrieval.graph` em interpretador frio -> exit 0, sem "circular"/"partially initialized".
- **Identidade/delegacao** (mesmo arquivo novo): `agent.AgentState is retrieval.state.AgentState`; `agent.LLMClient is retrieval.generation.LLMClient`; `agent.agent_workflow is retrieval.graph.agent_workflow`; verificacao de delecao dos wrappers `agent._listar_colecoes` e `agent._get_vector_db` (nao sao `is`-identicos porque sao funcoes distintas que ligam seams).
- **Seam binding de search**: monkeypatch `agent.search` + chamar `usar_ferramenta_web` -> a funcao movida ve o fake (coberto pelo oraculo + novo teste seam-specific).
- **Seam binding de DIR_VECTORDB**: monkeypatch `agent.DIR_VECTORDB` + chamar `_listar_colecoes` via wrapper -> o wrapper le o valor patchado e passa para retriever._listar_colecoes.
- **Seam binding de _embedding_model**: monkeypatch `agent._embedding_model` + chamar `_get_embedding_model` via shim -> a funcao movida ve o stub (lazy import de `agent`).

**Edge cases:** empty collection fallback; web route erro; retry exhaustion; multi-collection dedup.

**Testes que QUEBRAM e migram ALVO de patch (RETR-12 — so namespace, nao comportamento):**

Inventario de patch-target (~32 sites):
- `tests/test_agentic_rag.py`: `agent._invoke_chain` -> `retrieval.generation._invoke_chain`; `agent.search` -> permanece em `agent.search` (shim); `agent._get_retriever` -> `retrieval.retriever._get_retriever`; `agent.StrOutputParser` -> `retrieval.generation.StrOutputParser`; `agent._get_llm` -> `retrieval.generation._get_llm`; `agent.prompt_rag_retrieve` -> `retrieval.generation.prompt_rag_retrieve`; `agent._get_embedding_model` -> `retrieval.retriever._get_embedding_model`; `agent.ChatOpenAI` -> permanece em `agent.ChatOpenAI` (import no topo de generation.py); `agent.Chroma` -> `retrieval.retriever.Chroma`.
- `tests/test_agent.py`: `agent.HuggingFaceEmbeddings` -> `retrieval.retriever.HuggingFaceEmbeddings`; `patch.object(agent_mod, "_listar_colecoes", ...)` -> permanece em `agent._listar_colecoes` (wrapper); `patch.object(agent_mod, "_get_vector_db", ...)` -> permanece em `agent._get_vector_db` (wrapper).
- `tests/acceptance/test_agent_workflow_integration.py`: `agent._get_embedding_model` -> `retrieval.retriever._get_embedding_model`; `agent._get_retriever` -> `retrieval.retriever._get_retriever`; `agent._invoke_chain` -> `retrieval.generation._invoke_chain`; `agent.search` -> permanece `agent.search`.
- `tests/acceptance/test_retry_logic.py`: `agent._get_llm` -> `retrieval.generation._get_llm`; `agent.prompt_gerar` -> `retrieval.generation.prompt_gerar`; `agent.search` -> permanece `agent.search`; `agent._prompt_web` -> `retrieval.generation._prompt_web`; `agent._invoke_chain` -> `retrieval.generation._invoke_chain`.
- `tests/acceptance/test_multi_collection_chromadb.py`: `agent._listar_colecoes` -> permanece `agent._listar_colecoes` (wrapper); `agent._get_vector_db` -> permanece `agent._get_vector_db` (wrapper); `agent.invalidar_vector_db` -> permanece `agent.invalidar_vector_db` (shim); `agent.Chroma` -> `retrieval.retriever.Chroma`; `agent._get_embedding_model` -> `retrieval.retriever._get_embedding_model`.
- `tests/acceptance/test_portuguese_embedding_model.py`: `agent.HuggingFaceEmbeddings` -> `retrieval.retriever.HuggingFaceEmbeddings`; `agent._get_embedding_model` -> `retrieval.retriever._get_embedding_model`.
- `tests/test_rag.py` / `tests/test_rag_integration.py`: `agent.invalidar_vector_db` -> `retrieval.retriever.invalidar_vector_db` (shim).
- `tests/acceptance/test_health_check.py`: `agent.agent_workflow` -> permanece `agent.agent_workflow` (shim).
- `tests/acceptance/test_llm_provider_portability.py`: `agent.ChatOpenAI` -> permanece `agent.ChatOpenAI`.
- `tests/acceptance/test_semantic_chunking.py` / `tests/acceptance/test_unificar_metadados_chunks.py`: `agent.invalidar_vector_db` -> `retrieval.retriever.invalidar_vector_db`.

**Ficam intocados (verde SEM edicao):** `tests/test_rag_caracterizacao.py` (oraculo — HARD); `tests/ingestion/test_shims_identidade.py` (oraculo de identidade 3a — HARD); `tests/acceptance/test_adicionar_pdf_incrementalmente.py` (nao patcha simbolos de agent movidos nesta fase); `tests/acceptance/test_rag_ingestion_fase3b.py`; `tests/ingestion/test_store.py`.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/retrieval/__init__.py` | Novo | Init package + re-exports. |
| `src/agenticlog/retrieval/state.py` | Novo | `AgentState` (Pydantic) — corpo verbatim de `agent.py:275-289`. |
| `src/agenticlog/retrieval/generation.py` | Novo | `LLMClient`, `_llm_retry`, prompts, `_get_llm`, `_invoke_chain`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas`. Acessa singletons `_llm`/`_embedding_model` via lazy imports de `agent.py` no corpo das funcoes. |
| `src/agenticlog/retrieval/retriever.py` | Novo | `_get_embedding_model` (via `criar_embedding_model` + singleton `_embedding_model` lazy de `agent`), `_get_vector_db`, `_listar_colecoes`, `_get_retriever`, `invalidar_vector_db`. |
| `src/agenticlog/retrieval/graph.py` | Novo | `passo_decisao_agente`, `usar_ferramenta_web`, `retrieve_info`, `agent_workflow` (compilado), `inicializar_recursos`. |
| `src/agenticlog/agent.py` | Modificado (~180 ln) | Remover corpos movidos; manter docstring, imports reduzidos, warnings/torch, config imports, logger, singletons `_lll`/`_embedding_model`/`_vector_dbs`, `search` global (fisicamente), e bloco de shims identity + 2 wrappers. |
| `tests/acceptance/test_rag_retrieval_fase4.py` | Novo | Acicidade + identidade/delegacao + seam binding. |
| `tests/acceptance/test_rag_retrieval_fase4_patch_targets.py` | Novo (opcional) | Verificacao de patch-target coverage. |
| `tests/test_agentic_rag.py` | Modificado | Migracao de patch-target (~10 sites). |
| `tests/test_agent.py` | Modificado | Migracao de patch-target (~3 sites internos; `patch.object` permanece para wrappers). |
| `tests/acceptance/test_agent_workflow_integration.py` | Modificado | Migracao de patch-target (~12 sites). |
| `tests/acceptance/test_retry_logic.py` | Modificado | Migracao de patch-target (~8 sites). |
| `tests/acceptance/test_multi_collection_chromadb.py` | Modificado | Migracao de patch-target (~10 sites). |
| `tests/acceptance/test_portuguese_embedding_model.py` | Modificado | Migracao de patch-target (~5 sites). |
| `tests/test_rag.py` | Modificado | `invalidar_vector_db` -> `retrieval.retriever.invalidar_vector_db` (~4 sites). |
| `tests/test_rag_integration.py` | Modificado | `invalidar_vector_db` + `_listar_colecoes` + `_get_vector_db` (~5 sites). |
| `tests/acceptance/test_semantic_chunking.py` | Modificado | `invalidar_vector_db` -> `retrieval.retriever.invalidar_vector_db` (~4 sites). |
| `tests/acceptance/test_unificar_metadados_chunks.py` | Modificado | `invalidar_vector_db` -> `retrieval.retriever.invalidar_vector_db` (~2 sites). |
| `tests/acceptance/test_health_check.py` | Modificado | `agent.agent_workflow` -> permanece (shim). |
| `tests/test_rag_caracterizacao.py` | **NAO tocar** | Oraculo de caracterizacao (HARD constraint — zero diff). |
| `tests/ingestion/test_shims_identidade.py` | **NAO tocar** | Oraculo de identidade da Fase 3a (HARD constraint — zero diff). |

---

## Risks

- **[ALTO] Seam de `search` capturado em tempo de import:** se `graph.py` fizer `from agenticlog.agent import search` em nivel de modulo (capturado no import de `graph`), e o oraculo fizer `monkeypatch.setattr("agenticlog.agent.search", fake)` APOS o import, `graph.usar_ferramenta_web` veria o `search` ORIGINAL (nao o fake). **Mitigacao (DN-1):** `usar_ferramenta_web` faz `from agenticlog.agent import search` DENTRO do corpo da funcao (lazy import, resolvido a cada chamada). Detalhado em `design.md §5`.
- **[ALTO] Singletons `_llm`/`_embedding_model`/`_vector_dbs` com trapped-name:** se `generation.py` ou `retriever.py` lerem `_llm`/`_embedding_model` como nome de nivel de modulo (capturado no import), veriam `None` (valor inicial pre-inicializacao) em vez do singleton inicializado — e o oraculo patch `agent._embedding_model = emb` nao fluiria. **Mitigacao (DN-2):** todos os modulos de `retrieval/` acessam singletons via `from agenticlog.agent import _llm, _embedding_model, _vector_dbs` DENTRO do corpo da funcao (lazy import, resolvido a cada chamada). Detalhado em `design.md §4.1`.
- **[ALTO] `patch.object(agent_mod, ...)` quebra se o alvo nao estiver no namespace de `agent`:** `test_agent.py` usa `patch.object(agent_mod, "_listar_colecoes", ...)` e `patch.object(agent_mod, "_get_vector_db", ...)` — isso funciona porque ambos sao WRAPPERS que ficam fisicamente em `agent.py` (nao identity shims). Simbolos que viram shims (ex.: `_get_retriever`) precisam ser patchados como `"agenticlog.retrieval.retriever._get_retriever"` ou via `patch.object(retrieval_mod, ...)`. **Mitigacao:** tabela de migracao em `design.md §6` especifica exatamente o destino de cada alvo.
- **[ALTO] DIR_VECTORDB trapped-name (ADR-019 D4 repeat):** `_listar_colecoes` em `retriever.py` nao pode ler `DIR_VECTORDB` como nome de modulo capturado no import — o oraculo patcha `agent.DIR_VECTORDB`. **Mitigacao (RETR-11):** `_listar_colecoes` e `_get_vector_db` sao WRAPPERS em `agent.py` (nao identity shims) que resolvem `agent.DIR_VECTORDB` no momento da chamada e passam como argumento para `retriever._listar_colecoes(vectordb_dir=...)` / `retriever._get_vector_db(name, vectordb_dir=...)`.
- **[MEDIO] Import circular:** `retrieval/generation.py` nao pode importar `agent` em top-level (criaria ciclo: `agent -> generation -> agent`). `retrieval/retriever.py` idem. **Mitigacao (RETR-13):** lazy imports (DENTRO de funcoes, nunca no topo do modulo). Teste de acicidade fresh-interpreter.
- **[MEDIO] Perda de identidade `is` nos prompts:** testes que comparam instancias de `PromptTemplate` com `is` quebram se o shim nao for `is`-identico. **Mitigacao:** shim identity `from agenticlog.retrieval.generation import prompt_rag_retrieve` em nivel de modulo de `agent.py` -> `agent.x is generation.x`.
- **[MEDIO] `_get_retriever` chama `_listar_colecoes` e `_get_vector_db` internamente:** se `retriever._get_retriever` chamasse `retriever._listar_colecoes` diretamente (sem passar pelo wrapper de `agent`), a resolucao de `DIR_VECTORDB` usaria fallback `config.*` em vez do valor monkeypatchado. **Mitigacao (DN-3):** `retriever._get_retriever` acessa `_listar_colecoes` e `_get_vector_db` via lazy imports de `agenticlog.agent` no corpo (as versoes wrapper que resolvem `DIR_VECTORDB` no call time). Detalhado em `design.md §4.4`.
- **[MEDIO] `inicializar_recursos` chama 3 getters que acessam singletons:** se os getters estiverem em modulos diferentes (`retriever._get_embedding_model`, `retriever._get_vector_db`, `generation._get_llm`), a inicializacao via `graph.inicializar_recursos` precisa coordenar a ordem: embeddings -> vector_db -> llm (mesma ordem de `agent.py:221-223`). **Mitigacao:** a ordem e mantida dentro de `graph.inicializar_recursos`.
- **[MEDIO] `test_rag_caracterizacao.py` patcha `agent.DIR_VECTORDB` APENAS para o pipeline de ingestao:** o oraculo de retrieval (se existir) pode patchar `_embedding_model`, `_llm`, `_vector_dbs` diretamente — mas nao `DIR_VECTORDB` para retrieval. Confirmar que `_listar_colecoes` wrapper em `agent` resolve corretamente. Coberto por `teste_seam_binding_listar_colecoes`.
- **[Baixo] `test_agent.py` `patch.object` com `agent_mod` importado como `import agenticlog.agent as agent_mod`:** o namespace `agenticlog.agent` continua tendo `_listar_colecoes` e `_get_vector_db` como funcoes definidas em `agent.py` (wrappers) — `patch.object` funciona. Simbolos movidos que viram shims identity precisam ser patchados com `"agenticlog.retrieval.mod.name"` — mas `test_agent.py` so patchou funcoes que ficam como wrapper, entao nao ha quebra nesse arquivo especifico (confirmar no inventario).
- **[Ambiental] hnswlib SAC no Windows:** oraculo/testes Chroma sao *skipped* local (Windows/Smart App Control); o CI Linux e o gate autoritativo.
- **CLAUDE.md conflicts:** nenhum. Alinha com "small files" (`agent.py ~180`, modulos 200-400), type hints, docstrings PT, imutabilidade, commits Conventional em PT.

---

## Open Questions

None. As decisoes de design (DN-1 seam de search, DN-2 lazy singleton access, DN-3 resolucao de `_get_retriever` chamando wrappers via lazy import, tabela shim-vs-wrapper, localizacao fisica de `search`) foram resolvidas na analise e estao baked em `design.md`. Ver tambem `docs/adr/ADR-020` (draft) para a documentacao consolidada das decisoes desta fase.

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde no CI Linux; cobertura >= 80%.
- [ ] `pytest -m integration tests/test_rag_caracterizacao.py -v` verde; `git diff --stat tests/test_rag_caracterizacao.py` VAZIO.
- [ ] `git diff --stat tests/ingestion/test_shims_identidade.py` VAZIO; arquivo verde.
- [ ] `agent.py ~180 ln`; `state.py < 50 ln`; `generation.py` 200-400 ln; `retriever.py` 200-400 ln; `graph.py` 200-400 ln.
- [ ] `agenticlog.agent.X is agenticlog.retrieval.state.X` para `AgentState`; `agent.X is retrieval.generation.X` para `LLMClient`, prompts, funcoes de geracao; `agent.X is retrieval.retriever.X` para `_get_retriever`, `invalidar_vector_db`; `agent.X is retrieval.graph.X` para `agent_workflow`, `inicializar_recursos`, `passo_decisao_agente`, `retrieve_info`, `usar_ferramenta_web`.
- [ ] `import agenticlog.retrieval` e imports de cada submodulo saem 0 em interpretador frio.
- [ ] `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings em PT (Entrada/Saida/Lanca).
- [ ] ADR-020 draft mencionado em `design.md` (escrita do ADR e decisao separada do humano).
