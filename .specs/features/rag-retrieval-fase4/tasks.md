# ADR-018 Fase 4 — extracao do pacote `retrieval/` de `agent.py` — Tasks

**Path:** `.specs/features/rag-retrieval-fase4/tasks.md`
**Spec:** `.specs/features/rag-retrieval-fase4/spec.md` · **Design:** `.specs/features/rag-retrieval-fase4/design.md`
**TLC scope:** complex
**Status:** Awaiting human approval

Tasks para o backend-builder da feature-factory. Ordem projetada para o oraculo + suite completa verdes ao final. TLC Execute NAO e usado. Gate autoritativo: **CI Linux** (`pytest --cov=agenticlog -v`). Local Windows pula testes Chroma/hnswlib (Smart App Control) — nao confundir com regressao. Padroes: `ruff`/`black`/`isort` limpos, type hints, docstrings PT (Entrada/Saida/Lanca), imutabilidade, funcoes <50 ln, commits Conventional em PT (`refactor:`), branch-first, code-review antes do merge.

**Regra de ouro (todas as tasks):** NUNCA editar `tests/test_rag_caracterizacao.py` nem `tests/ingestion/test_shims_identidade.py`. Preservar comportamento, mensagens, ordem de excecoes verbatim. Migracao de teste = so ALVO de patch (namespace), nunca a assercao de comportamento.

**Marcadores:** todo shim de re-export em `agent.py` deve ter comentario:
```python
# Re-export shim (ADR-018 Fase 4) — remover na Fase 6
```

Ordem macro: **state.py -> generation.py -> retriever.py -> graph.py -> agent.py shims/wrappers -> testes novos (acicidade + identidade/delegacao) -> migracao de patch-target -> suite completa.**

---

## Execution Plan

### Fase A — `state.py` + `__init__.py` (sequencial)
```
T1 -> T2
```

### Fase B — `generation.py` (sequencial, depende de state)
```
T2 -> T3
```

### Fase C — `retriever.py` (sequencial, depende de config + ingest/embeddings)
```
T3 -> T4
```

### Fase D — `graph.py` (sequencial, depende de state + generation + retriever)
```
T4 -> T5
```

### Fase E — Religar `agent.py` (sequencial, depende de todos os modulos)
```
T5 -> T6
```

### Fase F — Testes novos (paralelo apos religar)
```
         -> T7 [P]
T6 ------ -> T8 [P]
         -> T9 [P]
```

### Fase G — Migracao de patch-target dos testes NAO-oraculo (paralelo por-arquivo)
```
         -> T10 [P]
T6 ------ -> T11 [P]
         -> T12 [P]
         -> T13 [P]
         -> T14 [P]
         -> T15 [P]
         -> T16 [P]
```

### Fase H — Gate final (sequencial)
```
T7..T16 -> T17
```

---

## Task Breakdown

### T1: Criar `retrieval/__init__.py` e `state.py` com AgentState + shim identity · [RETR-01, RETR-02, RETR-14]

**What:** Criar `src/agenticlog/retrieval/__init__.py` com `from agenticlog.retrieval.state import AgentState` e `__all__`. Criar `src/agenticlog/retrieval/state.py` com `AgentState` (Pydantic BaseModel) — field-set verbatim de `agent.py:275-289`: `query`, `next_step`, `retrieved_info`, `possible_responses`, `similarity_scores`, `ranked_response`, `confidence_score`. Nao importar `agent` em nenhum nivel. Adicionar ao `agent.py` o shim identity: `from agenticlog.retrieval.state import AgentState # Re-export shim (ADR-018 Fase 4) — remover na Fase 6`.
**Where:** `src/agenticlog/retrieval/__init__.py` (novo), `src/agenticlog/retrieval/state.py` (novo), `src/agenticlog/agent.py` (modificado)
**Depends on:** None
**Reuses:** `agent.py:275-289` (corpo verbatim); `__init__.py` padrao do pacote `ingestion/` como referencia de estrutura.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `state.py` importa so `pydantic` e stdlib; NAO importa `agent`.
- [ ] `from agenticlog.retrieval.state import AgentState` funciona sem erro.
- [ ] `agent.AgentState is retrieval.state.AgentState` (identity `is`).
- [ ] Field-set de `AgentState` identico ao original: 7 fields, mesmos defaults, mesma ordem.
- [ ] `python -c "import agenticlog.retrieval"` sai 0.

**Tests:** (validado em T7/T8) · **Gate:** quick

---

### T2: Atualizar `retrieval/__init__.py` com re-exports aditivos · [RETR-01]

**What:** Apos cada modulo criado (T3-T5), atualizar `retrieval/__init__.py` com os re-exports publicos mais importantes. Iniciar com `AgentState` de T1; expandir em T3/T4/T5.
**Where:** `src/agenticlog/retrieval/__init__.py`
**Depends on:** T1 (inicio), expandido em T3,T4,T5
**Reuses:** mesmo padrao de `ingestion/__init__.py`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `from agenticlog.retrieval import AgentState, LLMClient, prompt_rag_retrieve, _get_retriever, agent_workflow` resolve.
- [ ] `python -c "import agenticlog.retrieval"` sai 0.

**Tests:** (validado em T7/T8) · **Gate:** quick

---

### T3: Criar `generation.py` — LLMClient, _llm_retry, prompts, geracao, ranqueamento · [RETR-01, RETR-03, RETR-10]

**What:** Criar `src/agenticlog/retrieval/generation.py` com corpos verbatim de `agent.py`:
- `LLMClient` Protocol (linhas 89-102)
- `_llm_retry` decorator (68-81) — mesma politica: `LLM_MAX_RETRY_ATTEMPTS`, `wait_exponential`, `retry_if_exception_type(httpx.ConnectError, TimeoutException, RemoteProtocolError, APIConnectionError)`, `reraise=True`, `before_sleep_log`
- `prompt_rag_retrieve`, `prompt_gerar`, `_prompt_web` (237-269)
- `_get_llm()` (105-120) — acessa `_llm` singleton via `import agenticlog.agent as _agent_mod` lazy no corpo; se `_agent_mod._llm is None`, cria `ChatOpenAI(...)` e faz `_agent_mod._llm = ...`; retorna `_agent_mod._llm`
- `_invoke_chain(chain, inputs) -> str` (307-314) — marcado com `@_llm_retry`, corpo verbatim
- `gera_multiplas_respostas(state) -> AgentState` (362-397) — usa `_invoke_chain`, prompts, `_get_llm` (todos locais)
- `avalia_similaridade(state) -> AgentState` (400-435) — usa `_get_embedding_model` via lazy import de `agent._get_embedding_model` no corpo (ver design §4.2 DN-2a — o wrapper de agent.py)
- `rank_respostas(state) -> AgentState` (438-454)

NAO importar `agent` em nivel de modulo. Todos os accessos a singletons de `agent` sao lazy dentro de funcoes.
NAO importar `retrieval.retriever` ou `retrieval.graph` — `avalia_similaridade` acessa `_get_embedding_model` via `agent` lazy.
**Where:** `src/agenticlog/retrieval/generation.py` (novo)
**Depends on:** T1 (AgentState — usado nos parametros/retornos dos nos)
**Reuses:** corpos de `agent.py` linhas 68-81, 89-102, 105-120, 237-269, 307-314, 362-454; `config` constants; `sklearn.metrics.pairwise.cosine_similarity`; `tenacity.*`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `from agenticlog.retrieval.generation import _invoke_chain, gera_multiplas_respostas, avalia_similaridade, rank_respostas, _get_llm, LLMClient, prompt_rag_retrieve` resolve.
- [ ] `_get_llm` retorna o singleton de `agent._llm` (cria se None, seta via `_agent_mod._llm = ChatOpenAI(...)`).
- [ ] `_llm_retry` aplica a mesma politica de retry (verificar constantes).
- [ ] Prompt objects sao identicos (`str(prompt.template)` igual ao original).
- [ ] `python -c "import agenticlog.retrieval.generation"` sai 0 (sem ciclo).
- [ ] Adicionar shims identity em `agent.py` para todos os simbolos movidos (ver §6 da design.md).

**Tests:** (validado em T7/T8 mais suite completa) · **Gate:** quick

---

### T4: Criar `retriever.py` — _get_embedding_model wrapper, _get_vector_db, _listar_colecoes, _get_retriever, invalidar_vector_db · [RETR-01, RETR-09, RETR-11]

**What:** Criar `src/agenticlog/retrieval/retriever.py` com:

1. `_build_embedding_model() -> HuggingFaceEmbeddings` — factory pura que chama `criar_embedding_model()` de `agenticlog.ingestion.embeddings`. SEM singleton. Usada pelo wrapper de `agent.py`.
2. `_listar_colecoes(*, vectordb_dir: Path | None = None) -> list[str]` — corpo de `agent.py:158-180` com `DIR_VECTORDB` -> `vectordb_dir` resolvido no corpo (`vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir`).
3. `_get_vector_db(collection_name: str, *, vectordb_dir: Path | None = None) -> Chroma` — corpo de `agent.py:142-155` com `DIR_VECTORDB` -> `vectordb_dir` resolvido no corpo. Usa `_get_embedding_model()` via lazy import de `agent._get_embedding_model` (o wrapper — design §4.2).
4. `_get_retriever(query: str) -> list[Document]` — corpo de `agent.py:183-209` verbatim, com `_listar_colecoes` e `_get_vector_db` acessados via `from agenticlog.agent import _listar_colecoes, _get_vector_db` lazy no corpo (design §4.4 DN-3).
5. `invalidar_vector_db() -> None` — corpo de `agent.py:226-233` com `_vector_dbs.clear()` via `import agenticlog.agent as _agent_mod; _agent_mod._vector_dbs.clear()` lazy no corpo.

Em `agent.py`, adicionar:
- Shim identity: `_build_embedding_model`, `_get_retriever`, `invalidar_vector_db`
- Wrapper `_get_embedding_model()` (design §4.2 DN-2a): le/seta `_embedding_model` global; delegando construcao a `_retr._build_embedding_model()`
- Wrapper `_listar_colecoes()` (design §4.3): le `agent.DIR_VECTORDB` e chama `_retr._listar_colecoes(vectordb_dir=DIR_VECTORDB)`
- Wrapper `_get_vector_db(collection_name)` (design §4.3): le `agent.DIR_VECTORDB` e chama `_retr._get_vector_db(collection_name, vectordb_dir=DIR_VECTORDB)`

**Where:** `src/agenticlog/retrieval/retriever.py` (novo), `src/agenticlog/agent.py` (modificado)
**Depends on:** T3 (acessa `_get_embedding_model` wrapper de agent, que e adicionado nesta task mas usa funcao de T3 — a ordem de adicao dos shims em agent.py e coordenada em T6)
**Reuses:** `agent.py:123-139` (adaptado), `agent.py:142-155` (adaptado), `agent.py:158-180` (adaptado), `agent.py:183-209` (verbatim com lazy imports), `agent.py:226-233` (adaptado), `ingestion/embeddings.criar_embedding_model`, `config.*`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `_build_embedding_model` chama `criar_embedding_model()` e retorna o modelo (sem cache).
- [ ] `_listar_colecoes(vectordb_dir=...)` usa o parametro (nao captura `DIR_VECTORDB` no import).
- [ ] `_get_vector_db(name, vectordb_dir=...)` usa o parametro; acessa `_get_embedding_model` lazy de `agent`.
- [ ] `_get_retriever` chama `_listar_colecoes` e `_get_vector_db` via lazy import de `agent` (wrappers).
- [ ] `invalidar_vector_db` faz `_agent_mod._vector_dbs.clear()`.
- [ ] `python -c "import agenticlog.retrieval.retriever"` sai 0 (sem ciclo).
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T7/T8) · **Gate:** quick

---

### T5: Criar `graph.py` — search binding, nos do grafo, FSM, inicializar_recursos · [RETR-01, RETR-04, RETR-05, RETR-14]

**What:** Criar `src/agenticlog/retrieval/graph.py` com:

1. `passo_decisao_agente(state: AgentState) -> AgentState` — corpo verbatim de `agent.py:291-304`.
2. `usar_ferramenta_web(state: AgentState) -> AgentState` — corpo de `agent.py:317-337` adaptado: `search` acessado via `from agenticlog.agent import search` lazy no corpo (design §5 DN-1); `_invoke_chain`, `_prompt_web`, `_get_llm` de `generation` (import top-level — estao no mesmo pacote, sem ciclo).
3. `retrieve_info(state: AgentState) -> AgentState` — corpo verbatim de `agent.py:340-359`; chama `_get_retriever` de `retriever` (import top-level).
4. `inicializar_recursos() -> None` — corpo de `agent.py:212-223` adaptado: acessa `_get_embedding_model`, `_get_vector_db`, `_get_llm` via lazy imports de `agent` (design §3.4).
5. `agent_workflow` — compilado de `agent.py:457-474`: `StateGraph(AgentState)`, 6 nos, conditional edges, compile. Usa as funcoes-no locais (importadas de `generation` e `retriever`).

```python
# estrutura do workflow
workflow = StateGraph(AgentState)
workflow.add_node("decision", passo_decisao_agente)
workflow.add_node("retrieve", retrieve_info)
workflow.add_node("generate_multiple", gera_multiplas_respostas)    # de generation
workflow.add_node("evaluate_similarity", avalia_similaridade)       # de generation
workflow.add_node("rank_responses", rank_respostas)                 # de generation
workflow.add_node("usar_web", usar_ferramenta_web)
workflow.set_entry_point("decision")
workflow.add_conditional_edges("decision", lambda s: {"retrieve": "retrieve", "usar_web": "usar_web"}[s.next_step])
workflow.add_edge("retrieve", "generate_multiple")
workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")
agent_workflow = workflow.compile()  # variavel de modulo
```

**Where:** `src/agenticlog/retrieval/graph.py` (novo)
**Depends on:** T4 (retriever), T3 (generation), T1 (state)
**Reuses:** `agent.py:291-304`, `317-337`, `340-359`, `212-223`, `457-474`; `langgraph.StateGraph`; `StrOutputParser`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `from agenticlog.retrieval.graph import agent_workflow` resolve e `agent_workflow` e um `CompiledStateGraph`.
- [ ] `usar_ferramenta_web` acessa `search` via lazy `from agenticlog.agent import search` (ver design §5).
- [ ] `inicializar_recursos` acessa getters lazy de `agent` (ver design §3.4).
- [ ] FSM tem 6 nos, mesmos edges, mesma funcao de routing condicional.
- [ ] `python -c "import agenticlog.retrieval.graph"` sai 0 (sem ciclo).

**Tests:** (validado em T7/T8 + suite completa) · **Gate:** quick

---

### T6: Religar `agent.py` — shims identity + 3 wrappers + manutencao de singletons + search · [RETR-08, RETR-11, RETR-14]

**What:** Reestruturar `src/agenticlog/agent.py` como fachada:

1. **Manter:** docstring, imports uteis (reduzidos — remover `numpy`, `sklearn`, `tenacity`, `ChatOpenAI`, `Chroma`, `HuggingFaceEmbeddings`, `StrOutputParser`, `cosine_similarity`, `APIConnectionError`, `httpx`), `warnings`, `torch`, `os.environ`, config imports (manter `DIR_VECTORDB`, `DEFAULT_COLLECTION_NAME`, `CHROMA_COLLECTION_METADATA`, `LLM_*`, `NUM_CANDIDATE_RESPONSES`, `RETRIEVAL_K_*`, `ROUTING_KEYWORDS_WEB`), `logger`, singletons `_llm = None`, `_vector_dbs = {}`, `_embedding_model = None`, `search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)`.

2. **Adicionar** `import agenticlog.retrieval.generation as _gen`, `import agenticlog.retrieval.retriever as _retr`, `import agenticlog.retrieval.graph as _graph` (apos os globais/config, com `# noqa: E402`).

3. **Adicionar 3 wrappers:**
   ```python
   def _get_embedding_model() -> HuggingFaceEmbeddings:
       global _embedding_model
       if _embedding_model is None:
           _embedding_model = _retr._build_embedding_model()
       return _embedding_model
   
   def _listar_colecoes() -> list[str]:
       return _retr._listar_colecoes(vectordb_dir=DIR_VECTORDB)
   
   def _get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME) -> Chroma:
       return _retr._get_vector_db(collection_name, vectordb_dir=DIR_VECTORDB)
   ```

4. **Adicionar bloco de shims identity:**
   ```python
   # ── Re-export shims de state (ADR-018 Fase 4) — remover na Fase 6 ──────
   from agenticlog.retrieval.state import AgentState  # noqa: E402,F401
   # ── Re-export shims de generation (ADR-018 Fase 4) — remover na Fase 6 ─
   from agenticlog.retrieval.generation import (      # noqa: E402,F401
       LLMClient, _llm_retry, prompt_rag_retrieve, prompt_gerar, _prompt_web,
       _get_llm, _invoke_chain, gera_multiplas_respostas, avalia_similaridade, rank_respostas,
   )
   # ── Re-export shims de retriever (ADR-018 Fase 4) — remover na Fase 6 ──
   from agenticlog.retrieval.retriever import (        # noqa: E402,F401
       _build_embedding_model, _get_retriever, invalidar_vector_db,
   )
   # ── Re-export shims de graph (ADR-018 Fase 4) — remover na Fase 6 ──────
   from agenticlog.retrieval.graph import (             # noqa: E402,F401
       passo_decisao_agente, usar_ferramenta_web, retrieve_info,
       inicializar_recursos, agent_workflow,
   )
   ```

5. **Remover** corpos de funcoes/classes que foram movidos para `retrieval/`.

**Where:** `src/agenticlog/agent.py`
**Depends on:** T5 (graph), T4 (retriever), T3 (generation), T1 (state)
**Reuses:** padrao de shim de Fase 3b (`rag.py` shim block).

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `agent.py` tem ~180 ln (verificar com `wc -l`).
- [ ] `from agenticlog.agent import <todos os simbolos da tabela de identidade>` resolve sem `ImportError`.
- [ ] `agenticlog.agent.X is agenticlog.retrieval.state.AgentState` para `AgentState` (e similar para cada shim).
- [ ] `agent._listar_colecoes` e `agent._get_vector_db` sao WRAPPERS (funcoes definidas em `agent.py`, nao shims).
- [ ] `agent.search` e a definicao fisica (nao shim).
- [ ] `python -c "import agenticlog.agent"` sai 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T7/T8/T9) · **Gate:** quick

---

### T7: Novo teste de acicidade `retrieval/*` · [P] · [RETR-13]

**What:** Criar `tests/acceptance/test_rag_retrieval_fase4.py` com teste de acicidade fresh-interpreter (`subprocess`): `import agenticlog.retrieval.state`, `import agenticlog.retrieval.generation`, `import agenticlog.retrieval.retriever`, `import agenticlog.retrieval.graph` -> returncode 0, sem "circular"/"partially initialized". Incluir `import agenticlog.agent` apos os 4 imports.
**Where:** `tests/acceptance/test_rag_retrieval_fase4.py` (novo)
**Depends on:** T6
**Reuses:** `test_rag_shared_observability.py::test_ac06`, `test_shims_identidade.py::TestIngestionAcyclic`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_rag_retrieval_fase4.py -v` verde.
- [ ] Todos os 4 modulos de `retrieval/` + `agent` importaveis em interpretador frio sem ciclo.

**Tests:** integration (subprocess) · **Gate:** full

---

### T8: Novo teste de identidade shim + seam binding · [P] · [RETR-02, RETR-04, RETR-11, RETR-14]

**What:** No mesmo arquivo novo (`test_rag_retrieval_fase4.py`):
1. Afirmar `agent.AgentState is retrieval.state.AgentState`
2. Afirmar `agent.LLMClient is retrieval.generation.LLMClient`
3. Afirmar `agent.agent_workflow is retrieval.graph.agent_workflow`
4. Afirmar `agent.prompt_rag_retrieve is retrieval.generation.prompt_rag_retrieve`
5. Afirmar `agent._get_retriever is retrieval.retriever._get_retriever`
6. Afirmar `agent.invalidar_vector_db is retrieval.retriever.invalidar_vector_db`
7. Afirmar que `agent._listar_colecoes` NAO e `retrieval.retriever._listar_colecoes` (wrapper, nao identity)
8. Afirmar que `agent._get_vector_db` NAO e `retrieval.retriever._get_vector_db` (wrapper, nao identity)
9. Afirmar que `agent._get_embedding_model` NAO e `retrieval.retriever._build_embedding_model` (wrapper, nao identity)
10. Teste de seam binding de `DIR_VECTORDB`: monkeypatch `agent.DIR_VECTORDB` com path fake; chamar `agent._listar_colecoes()` (wrapper); afirmar que `chromadb.PersistentClient` (patchado em `retrieval.retriever`) recebeu o path fake.
11. Teste de seam binding de `search`: monkeypatch `agent.search` com fake; chamar `agent.usar_ferramenta_web(AgentState(query="teste"))`; afirmar que o fake foi chamado e a resposta contem "Busca indisponivel" (ou o retorno do fake — conforme o estagio do oraculo).

**Where:** `tests/acceptance/test_rag_retrieval_fase4.py`
**Depends on:** T6
**Reuses:** fixtures de monkeypatch de `test_rag_caracterizacao.py` como modelo.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Identidade `is` afirmada para todos os ~20 shims.
- [ ] Wrappers `_listar_colecoes`, `_get_vector_db`, `_get_embedding_model` identificados como nao-`is`.
- [ ] Seam binding de `DIR_VECTORDB` e `search` verificados.
- [ ] `pytest tests/acceptance/test_rag_retrieval_fase4.py -v` verde.

**Tests:** unit/integration · **Gate:** full

---

### T9: Verificar oraculos zero-diff · [P] · [RETR-06, RETR-07]

**What:** Rodar os dois oraculos e confirmar `git diff` VAZIO. Gate de verificacao, nao de edicao.
**Where:** `tests/test_rag_caracterizacao.py`, `tests/ingestion/test_shims_identidade.py` (so leitura/execucao)
**Depends on:** T6
**Reuses:** —

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest -m integration tests/test_rag_caracterizacao.py -v` verde (CI Linux).
- [ ] `pytest tests/ingestion/test_shims_identidade.py -v` verde.
- [ ] `git diff --stat tests/test_rag_caracterizacao.py tests/ingestion/test_shims_identidade.py` VAZIO.

**Tests:** integration · **Gate:** full

---

### T10: Migrar patch-target de `tests/test_agentic_rag.py` · [P] · [RETR-12]

**What:** Migrar ALVOS de patch (namespace) para os simbolos que se moveram fisicamente:

| Patch atual (agent.) | Novo alvo | Motivo |
|---------------------|-----------|--------|
| `agent._invoke_chain` | `retrieval.generation._invoke_chain` | Moveu para generation.py |
| `agent.search` | **permanece** `agent.search` | search fica em agent.py (definition) |
| `agent._get_retriever` | `retrieval.retriever._get_retriever` | Moveu para retriever.py (shim identity) |
| `agent.StrOutputParser` | `retrieval.generation.StrOutputParser` | Moveu para generation.py |
| `agent._get_llm` | `retrieval.generation._get_llm` | Moveu para generation.py (shim identity) |
| `agent.prompt_rag_retrieve` | `retrieval.generation.prompt_rag_retrieve` | Moveu para generation.py (shim identity) |
| `agent._get_embedding_model` | **permanece** `agent._get_embedding_model` | Wrapper em agent.py |
| `agent.ChatOpenAI` | **permanece** `agent.ChatOpenAI` | Import em generation.py ainda referenciado via agent |
| `agent.Chroma` | `retrieval.retriever.Chroma` | Moveu para retriever.py |

**NOTA:** Symbols que permanecem como wrapper em `agent.py` (`_get_embedding_model`) ou definicao fisica (`search`) NAO migram. Symbols que viram shim identity (`_get_retriever`, `_invoke_chain`) MIGRAM para `retrieval.<mod>.<name>`.
**Where:** `tests/test_agentic_rag.py`
**Depends on:** T6
**Reuses:** inventario de design.md §6, spec.md §Tests Required.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/test_agentic_rag.py -v` verde; assercoes de comportamento inalteradas.
- [ ] Contagem de testes preservada.

**Tests:** unit · **Gate:** quick

---

### T11: Migrar patch-target de `tests/acceptance/test_agent_workflow_integration.py` · [P] · [RETR-12]

**What:** Migrar ALVOS de patch:

| Patch atual (agent.) | Novo alvo |
|---------------------|-----------|
| `agent._get_embedding_model` | **permanece** `agent._get_embedding_model` (wrapper em agent.py) |
| `agent._get_retriever` | `retrieval.retriever._get_retriever` |
| `agent._invoke_chain` | `retrieval.generation._invoke_chain` |
| `agent.search` | **permanece** `agent.search` |

**Where:** `tests/acceptance/test_agent_workflow_integration.py`
**Depends on:** T6

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_agent_workflow_integration.py -v` verde.
- [ ] Comportamento verificado inalterado; contagem preservada.

**Tests:** unit · **Gate:** quick

---

### T12: Migrar patch-target de `tests/acceptance/test_retry_logic.py` · [P] · [RETR-12]

**What:** Migrar ALVOS de patch:

| Patch atual (agent.) | Novo alvo |
|---------------------|-----------|
| `agent._get_llm` | `retrieval.generation._get_llm` |
| `agent.prompt_gerar` | `retrieval.generation.prompt_gerar` |
| `agent.search` | **permanece** `agent.search` |
| `agent._prompt_web` | `retrieval.generation._prompt_web` |
| `agent._invoke_chain` | `retrieval.generation._invoke_chain` |
| `agent.ChatOpenAI` | **permanece** `agent.ChatOpenAI` |

**Where:** `tests/acceptance/test_retry_logic.py`
**Depends on:** T6

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_retry_logic.py -v` verde.
- [ ] Comportamento verificado inalterado.

**Tests:** unit · **Gate:** quick

---

### T13: Migrar patch-target de `tests/acceptance/test_multi_collection_chromadb.py` · [P] · [RETR-12]

**What:** Migrar ALVOS de patch:

| Patch atual (agent.) | Novo alvo |
|---------------------|-----------|
| `agent._listar_colecoes` | **permanece** `agent._listar_colecoes` (wrapper em agent.py) |
| `agent._get_vector_db` | **permanece** `agent._get_vector_db` (wrapper em agent.py) |
| `agent.invalidar_vector_db` | `retrieval.retriever.invalidar_vector_db` |
| `agent.Chroma` | `retrieval.retriever.Chroma` |
| `agent._get_embedding_model` | **permanece** `agent._get_embedding_model` (wrapper em agent.py) |

**Where:** `tests/acceptance/test_multi_collection_chromadb.py`
**Depends on:** T6

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_multi_collection_chromadb.py -v` verde.
- [ ] Comportamento inalterado.

**Tests:** unit/integration · **Gate:** full

---

### T14: Migrar patch-target de `tests/acceptance/test_portuguese_embedding_model.py` · [P] · [RETR-12]

**What:** Migrar ALVOS de patch:

| Patch atual (agent.) | Novo alvo |
|---------------------|-----------|
| `agent.HuggingFaceEmbeddings` | `retrieval.retriever.HuggingFaceEmbeddings` |
| `agent._get_embedding_model` | **permanece** `agent._get_embedding_model` (wrapper) |

**Where:** `tests/acceptance/test_portuguese_embedding_model.py`
**Depends on:** T6

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_portuguese_embedding_model.py -v` verde.
- [ ] Testes de inspecao de fonte NAO alterados.

**Tests:** unit/integration · **Gate:** full

---

### T15: Migrar patch-target de `tests/test_rag.py` e `tests/test_rag_integration.py` · [P] · [RETR-12]

**What:** Migrar ALVOS:

| Patch atual (agent.) | Novo alvo | Arquivo |
|---------------------|-----------|---------|
| `agent.invalidar_vector_db` | `retrieval.retriever.invalidar_vector_db` | `test_rag.py` (~4 sites) |
| `agent.invalidar_vector_db` | `retrieval.retriever.invalidar_vector_db` | `test_rag_integration.py` (~5 sites) |
| `agent._listar_colecoes` | **permanece** `agent._listar_colecoes` | `test_rag_integration.py` |
| `agent._get_vector_db` | **permanece** `agent._get_vector_db` | `test_rag_integration.py` |

**Where:** `tests/test_rag.py`, `tests/test_rag_integration.py`
**Depends on:** T6

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/test_rag.py -v` verde.
- [ ] `pytest tests/test_rag_integration.py -v` verde.
- [ ] Comportamento inalterado.

**Tests:** unit/integration · **Gate:** quick/full

---

### T16: Migrar patch-target de acceptance tests residuais · [P] · [RETR-12]

**What:** Migrar ALVOS de `tests/acceptance/test_semantic_chunking.py` e `tests/acceptance/test_unificar_metadados_chunks.py`:

| Patch atual (agent.) | Novo alvo |
|---------------------|-----------|
| `agent.invalidar_vector_db` | `retrieval.retriever.invalidar_vector_db` |

Tambem verificar `tests/acceptance/test_health_check.py`: `agent.agent_workflow` **permanece** (shim identity em agent.py — patch-target continua `agenticlog.agent.agent_workflow`).
**Where:** `tests/acceptance/test_semantic_chunking.py`, `tests/acceptance/test_unificar_metadados_chunks.py`, `tests/acceptance/test_health_check.py` (verificacao)
**Depends on:** T6

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_semantic_chunking.py -v` verde.
- [ ] `pytest tests/acceptance/test_unificar_metadados_chunks.py -v` verde.
- [ ] `tests/acceptance/test_health_check.py` inalterado (patch-target permanece valido).

**Tests:** unit/integration · **Gate:** full

---

### T17: Gate final — suite completa + lint + contagem de linhas + verificacao de seam · [RETR-15, todos]

**What:** Rodar a suite completa, lint e as verificacoes de tamanho/diff.
**Where:** todo o repo
**Depends on:** T7, T8, T9, T10, T11, T12, T13, T14, T15, T16
**Reuses:** —

**Tools:** MCP: NONE · Skill: `code-review` (antes do merge)

**Done when:**
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde (CI Linux); cobertura >= 80%.
- [ ] `git diff --stat tests/test_rag_caracterizacao.py tests/ingestion/test_shims_identidade.py` VAZIO.
- [ ] `agent.py ~180 ln`; `state.py < 50 ln`; `generation.py` 200-400 ln; `retriever.py` 200-400 ln; `graph.py` 120-200 ln.
- [ ] `ruff check .`, `black --check .`, `isort --check .` limpos.
- [ ] Varredura de seam: `rg "@patch\(\"agenticlog\.agent\.(.*)\"" tests/` — cada ocorrencia remanescente DEVE exercitar um simbolo que ficou em `agent.py` (wrapper, singleton, search, shim identity). Qualquer teste que patche simbolo MOVIDO para `retrieval/` deve migrar.
- [ ] Varredura de `patch.object(agent_mod, ...)` — verificar que simbolos patchados sao wrappers em `agent.py`.
- [ ] `code-review` executado antes do merge.

**Tests:** integration (full suite) · **Gate:** build

**Commit:** `refactor(rag): extrai pacote retrieval/ de agent.py (ADR-018 Fase 4)`

---

## Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram | Status |
|------|-------------------|---------|--------|
| T1 | None | (raiz Fase A) | Pendente |
| T2 | T1 | T1->T2 | Pendente |
| T3 | T2 | T2->T3 | Pendente |
| T4 | T3 | T3->T4 | Pendente |
| T5 | T4 | T4->T5 | Pendente |
| T6 | T5 | T5->T6 | Pendente |
| T7 [P] | T6 | T6->T7 | Pendente |
| T8 [P] | T6 | T6->T8 | Pendente |
| T9 [P] | T6 | T6->T9 | Pendente |
| T10 [P] | T6 | T6->T10 | Pendente |
| T11 [P] | T6 | T6->T11 | Pendente |
| T12 [P] | T6 | T6->T12 | Pendente |
| T13 [P] | T6 | T6->T13 | Pendente |
| T14 [P] | T6 | T6->T14 | Pendente |
| T15 [P] | T6 | T6->T15 | Pendente |
| T16 [P] | T6 | T6->T16 | Pendente |
| T17 | T7..T16 | ->T17 | Pendente |

Tarefas `[P]` (T7-T16) nao dependem entre si — todas dependem so de T6. T7-T9 tocam arquivos de teste novos/leitura; T10-T16 tocam arquivos distintos (testes de diferentes modulos) — paralelizaveis.

## Test Co-location Validation

| Task | Layer criado/modificado | Matriz exige | Task diz | Status |
|------|------------------------|--------------|----------|--------|
| T1/T2 | retrieval/state | unit | validado por T7/T8 | @validado-adiante |
| T3 | retrieval/generation | unit | validado por T7/T8 + T12 | @validado-adiante |
| T4 | retrieval/retriever | unit | validado por T7/T8 + T10/T13/T14/T15 | @validado-adiante |
| T5 | retrieval/graph | unit | validado por T7/T8 + T11 | @validado-adiante |
| T6 | agent (fachada) | unit | validado por T8 (identidade) | @validado-adiante |
| T7/T8 | teste novo | integration/unit | integration/unit | @ok |
| T10..T16 | testes migrados | unit/integration | unit/integration | @ok |

Nota: T3/T4/T5 criam codigo de retrieval cujos unit tests JA existem (em `test_agentic_rag.py` etc.) e sao migrados por-namespace em T10-T16 — nao e deferral (os testes existem e passam ao final da mesma fase de trabalho); a co-locacao e satisfeita pela migracao dirigida + novos testes de seam em T8.

---

## Notas de teste / gate (de `.specs/codebase/TESTING.md` e CLAUDE.md)
- Sempre mockar LLM; sempre testar retrieval vazio.
- Nomes de teste com prefixo `teste_N_` (dominio) / `test_` (erro/seguranca).
- Gate autoritativo: CI Linux `pytest --cov=agenticlog --cov-report=term-missing -v`.
- Commits Conventional em PT (`refactor:`); branch ja criada (`feature/rag-retrieval-fase4`); code-review antes do merge.
- `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings PT (Entrada/Saida/Lanca); imutabilidade; funcoes <50 ln.
