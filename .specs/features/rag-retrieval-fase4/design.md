# ADR-018 Fase 4 — extracao do pacote `retrieval/` de `agent.py` — Design

**Path:** `.specs/features/rag-retrieval-fase4/design.md`
**Spec:** `.specs/features/rag-retrieval-fase4/spec.md`
**TLC scope:** complex
**Status:** Awaiting human approval

---

## 1. Architecture Overview

A Fase 4 extrai as 4 camadas de `agenticlog.agent` (474 ln) para o pacote `agenticlog.retrieval/`, completando o bloco ONLINE da arquitetura-alvo (ADR-018 §3/§4/§7):

- **`retrieval/state.py`** — `AgentState` (Pydantic), field-set imutavel, 0 dependencias alem de `pydantic`.
- **`retrieval/generation.py`** — geracao + ranqueamento: `LLMClient` Protocol, `_llm_retry`, 3 prompts, `_get_llm`, `_invoke_chain`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas`.
- **`retrieval/retriever.py`** — recuperacao vetorial: `_get_embedding_model` (via `criar_embedding_model` + singleton lazy), `_get_vector_db`, `_listar_colecoes`, `_get_retriever` (fan-out + dedup), `invalidar_vector_db`.
- **`retrieval/graph.py`** — orquestracao LangGraph: `passo_decisao_agente`, `usar_ferramenta_web`, `retrieve_info`, `inicializar_recursos`, `agent_workflow` (compilado).

`agent.py` vira a **fachada de compatibilidade**: mantem singletons fisicos (`_llm`, `_embedding_model`, `_vector_dbs`), global `search`, imports reduzidos, e um bloco de shims identity-preserving + 2 wrappers que resolvem `DIR_VECTORDB` no momento da chamada.

```
config.py (constantes: fonte unica) ─────────────────────────┐
ingestion/embeddings.py (criar_embedding_model) ─────────────┤
                                                              ▼
        retrieval/state.py        (NOVO — importa so pydantic)
        retrieval/generation.py   (NOVO — importa config, stdlib, langchain*;
                                    lazy import de agent p/ singletons)
        retrieval/retriever.py    (NOVO — importa config, stdlib, langchain*,
                                    ingestion/embeddings; lazy import de agent)
        retrieval/graph.py        (NOVO — importa state, generation, retriever,
                                    langgraph; lazy import de agent p/ search)
                                                              ▲
        agenticlog.agent ─────────────────────────────────────┘
         (fachada: singletons + search + shims identity + 2 wrappers)
```

Arestas de import (todas unidirecionais, DAG): `config -> shared -> ingestion -> retrieval -> agent -> app/api/tests`. `retrieval/generation.py`, `retrieval/retriever.py` e `retrieval/graph.py` NUNCA importam `agent` em nivel de modulo — so em lazy imports DENTRO de corpos de funcao, evitando ciclo. `retrieval/state.py` nao importa `agent` em nenhum nivel.

---

## 2. Code Reuse Analysis

### Componentes existentes a alavancar

| Component | Location | How to Use |
|-----------|----------|-----------|
| Padrao shim `is`-identity | `rag.py:76-99`, `shared/__init__.py`, `observability/__init__.py`, Fase 3b shims | Aplicar o MESMO bloco de re-export para cada symbol de `retrieval/*` em `agent.py`. |
| `criar_embedding_model` | `agenticlog.ingestion.embeddings` | `retrieval/retriever.py` importa e usa como factory — mesmo espaco vetorial (§3). |
| Padrao wrapper call-time (ADR-019 D4) | `rag.py` wrappers de orquestrador (Fase 3b) | Replicar para `_get_vector_db` e `_listar_colecoes` — resolver `agent.DIR_VECTORDB` no momento da chamada. |
| Padrao lazy import p/ singletons | Fase 3a `rag.py` shims de embedding | `retrieval/*` acessa `_llm`, `_embedding_model`, `_vector_dbs` via `from agenticlog.agent import <simbolo>` no corpo da funcao. |
| Teste de acicidade fresh-interpreter | `tests/acceptance/test_rag_shared_observability.py::test_ac06`; `tests/ingestion/test_shims_identidade.py::TestIngestionAcyclic` | Estender o padrao `subprocess` para `retrieval/*`. |
| Fixture do oraculo | `tests/test_rag_caracterizacao.py` (patches `agent.DIR_VECTORDB`, `_embedding_model`, `_llm`, `_vector_dbs`, `search`) | NAO modificar; o design garante que todas as 5 variaveis patchadas fluem para as funcoes movidas. |

### Integracao

| System | Integration Method |
|--------|--------------------|
| `app.py` | Importa `_listar_colecoes` de `agenticlog.agent` (wrapper preservado). |
| `api.py` | Importa `inicializar_recursos` de `agenticlog.agent` (shim para `graph.inicializar_recursos`). |
| `__init__.py` | Importa `AgentState` e `agent_workflow` de `agenticlog.agent` (shims para `state.AgentState` e `graph.agent_workflow`). |
| `ingestion/orchestrator.py` | Importa `invalidar_vector_db` de `agenticlog.agent` (shim) — lazy import. |

---

## 3. Componentes e interfaces

### 3.1 `retrieval/state.py`

Camada de estado. Importa: `pydantic.BaseModel`. **Nao** importa `agent`, `config`, ou qualquer outro modulo do projeto.

- `AgentState(BaseModel)` — corpo VERBATIM de `agent.py:275-289`. Field-set:
  ```python
  query: str
  next_step: str = ""
  retrieved_info: list = []
  possible_responses: list = []
  similarity_scores: list = []
  ranked_response: str = ""
  confidence_score: float = 0.0
  ```

**Tamanho estimado:** ~20 ln (excluindo docstring). Abaixo do alvo de 200; aceitavel — e intencionalmente minimalista.

### 3.2 `retrieval/generation.py`

Camada de geracao + ranqueamento. Importa: `logging`, `hashlib`, `numpy`, `langchain_core.*`, `langchain_openai.ChatOpenAI`, `langchain_huggingface.HuggingFaceEmbeddings`, `sklearn.*`, `tenacity`, `httpx`, `openai.APIConnectionError`, `pydantic.BaseModel` (para `LLMClient?`). De `config`: `LLM_*` constants, `NUM_CANDIDATE_RESPONSES`. **Nao** importa `agent`, `retrieval.retriever`, `retrieval.graph`, `retrieval.state` em nivel de modulo.

Contem verbatim de `agent.py`:
- `LLMClient` Protocol — linhas 89-102
- `_llm_retry` — linhas 68-81 (decorator `@retry` com `tenacity`)
- `_get_llm()` — linhas 105-120 (cria/cacheia singleton `_llm` via lazy import: `from agenticlog.agent import _llm` dentro do corpo, ver §4.1)
- `prompt_rag_retrieve`, `prompt_gerar`, `_prompt_web` — linhas 237-269 (objetos identicos)
- `_invoke_chain` — linhas 307-314 (marcado com `@_llm_retry`)
- `gera_multiplas_respostas` — linhas 362-397 (usa `_invoke_chain`, `prompt_rag_retrieve`/`prompt_gerar`, `_get_llm()`)
- `avalia_similaridade` — linhas 400-435 (usa `_get_embedding_model` via lazy import: `from agenticlog.agent import _get_embedding_model` no corpo — ver §4.1)
- `rank_respostas` — linhas 438-454

`gera_multiplas_respostas` e `avalia_similaridade` importam `_get_embedding_model` e funcoes de `generation` via caminho local (ja estao no mesmo modulo) — nenhum ciclo.

`from sklearn.metrics.pairwise import cosine_similarity` permanece em `generation.py` (nao compartilhado).

**Tamanho estimado:** ~200-250 ln.

### 3.3 `retrieval/retriever.py`

Camada de recuperacao vetorial. Importa: `hashlib`, `logging`, `torch`, `Chroma`, `Document`, `HuggingFaceEmbeddings`, `chromadb` (lazy). De `config`: `DIR_VECTORDB`, `DEFAULT_COLLECTION_NAME`, `CHROMA_COLLECTION_METADATA`, `EMBEDDING_MODEL`, `RETRIEVAL_K_*`. De `ingestion/embeddings`: `criar_embedding_model`. **Nao** importa `agent`, `retrieval.generation`, `retrieval.graph`, `retrieval.state` em nivel de modulo.

Contem (corpos adaptados de `agent.py`):
- `_get_embedding_model() -> HuggingFaceEmbeddings` — linha 123-139 adaptado: usa `criar_embedding_model()` como factory (RETR-09), mantem o singleton `_embedding_model` via lazy `from agenticlog.agent import _embedding_model` (global) e `from agenticlog.agent import _embedding_model as _set_emb` (atribuicao para setter — ver §4.1 DN-2 design detail). Estrutura:
  ```python
  def _get_embedding_model() -> HuggingFaceEmbeddings:
      from agenticlog.agent import _embedding_model  # lazy — le o singleton no momento da chamada
      if _embedding_model is None:
          # constroi usando criar_embedding_model() e seta via lazy import
          from agenticlog.agent import _embedding_model as _emb_global
          _emb_global = criar_embedding_model()  # not Python syntax — precisa de workaround
      return _embedding_model
  ```
  **Nota de implementacao:** Python nao permite reatribuir um nome importado (`from X import Y; Y = Z` nao modifica `X.Y`). A implementacao precisa de um workaround:
  1. `import agenticlog.agent as _agent_mod; ...; _agent_mod._embedding_model = criar_embedding_model()` (import modulo inteiro lazy), OU
  2. `_get_embedding_model` permanece como WRAPPER em `agent.py` que le e seta o singleton diretamente, e `retrieval/retriever.py` exporta a funcao sem o singleton (apenas `criar_embedding_model()`).
  
  **Decisao (DN-2a):** `_get_embedding_model` permanece como **wrapper** em `agent.py` (le/seta `_embedding_model` global diretamente) e a funcao movida em `retriever.py` e renomeada para `_build_embedding_model()` — contendo apenas o factory `criar_embedding_model()`. O shim em `agent.py` e:
  ```python
  def _get_embedding_model():
      global _embedding_model
      if _embedding_model is None:
          _embedding_model = _retr._build_embedding_model()
      return _embedding_model
  ```
  Isso elimina o problema de reatribuicao de nome importado e mantem o singleton SETAVEL via `agent._embedding_model = stub`. Oraculo preservado (patches `agent._embedding_model` diretamente, bypassa o getter).

- `_get_vector_db(collection_name: str, *, vectordb_dir: Path | None = None) -> Chroma` — corpo de `agent.py:142-155`, com `DIR_VECTORDB` substituido por `vectordb_dir` resolvido no corpo (`vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir`). Usa `_get_embedding_model()` (via shim de `agent` — lazy import dentro do corpo). **Nao e shim identity:** e funcao pura parametrizada.
- `_listar_colecoes(*, vectordb_dir: Path | None = None) -> list[str]` — corpo de `agent.py:158-180`, com `DIR_VECTORDB` -> `vectordb_dir` resolvido no corpo. **Nao e shim identity:** e funcao pura parametrizada.
- `_get_retriever(query: str) -> list[Document]` — corpo de `agent.py:183-209`, com `_listar_colecoes` e `_get_vector_db` acessados via lazy `from agenticlog.agent import _listar_colecoes, _get_vector_db` dentro do corpo (para que os WRAPPERS de `agent` — que resolvem `DIR_VECTORDB` — sejam chamados a cada invocacao, ver §4.4 DN-3).
- `invalidar_vector_db() -> None` — corpo de `agent.py:226-233` simplificado: `_vector_dbs.clear()`, acessado via lazy `from agenticlog.agent import _vector_dbs` no corpo. Se a reatribuicao for necessaria (`_vector_dbs = {}`), usa `import agenticlog.agent as _agent_mod; _agent_mod._vector_dbs.clear()`.

**Tamanho estimado:** ~200-250 ln.

### 3.4 `retrieval/graph.py`

Camada de orquestracao LangGraph. Importa: `logging`, `StrOutputParser`, `PromptTemplate`, `StateGraph`. De `config`: `ROUTING_KEYWORDS_WEB`, `DEFAULT_COLLECTION_NAME`. De `retrieval`: `state.AgentState`, `generation.*`, `retriever.*` (todas as funcoes de no). **Nao** importa `agent` em nivel de modulo.

Contem (corpos de `agent.py`):
- `search` global — **NAO** move para `graph.py`. Permanece fisicamente em `agent.py` (ver §5 DN-1). `graph.py` acessa via `from agenticlog.agent import search` dentro do corpo de `usar_ferramenta_web`.
- `passo_decisao_agente(state: AgentState) -> AgentState` — corpo de `agent.py:291-304` verbatim.
- `usar_ferramenta_web(state: AgentState) -> AgentState` — corpo de `agent.py:317-337` adaptado: `search` acessado via lazy import (ver §5 DN-1). `_invoke_chain` e `_prompt_web` de `generation` (disponiveis localmente, sem ciclo).
- `retrieve_info(state: AgentState) -> AgentState` — corpo de `agent.py:340-359` verbatim (chama `_get_retriever` de `retriever`, disponivel localmente, sem ciclo).
- `inicializar_recursos() -> None` — corpo de `agent.py:212-223`:
  ```python
  def inicializar_recursos():
      from agenticlog.agent import _get_embedding_model as _get_emb
      from agenticlog.agent import _get_llm as _get_llm_fn
      from agenticlog.agent import _get_vector_db as _get_vdb
      _get_emb()
      _get_vdb(DEFAULT_COLLECTION_NAME)
      _get_llm_fn()
  ```
  Acessa os 3 getters via lazy import de `agent` para garantir que os singletons sejam criados no namespace de `agent`.
- `agent_workflow` — compilado: corpo de `agent.py:457-474` verbatim, com `StateGraph(AgentState)` e 6 nos + edges. A definicao do grafo usa as funcoes-no locais de `graph.py` (que sao shims identity re-exportados por `agent.py`).

**Tamanho estimado:** ~120-150 ln.

### 3.5 `agent.py` (fachada) apos a fase

Mantem: docstring, imports reduzidos (sem `numpy`, `sklearn`, `tenacity`, `ChatOpenAI` — estes vao para `generation`; sem `Chroma`, `HuggingFaceEmbeddings` — estes vao para `retriever`), `warnings`, `torch`, `os.environ["TOKENIZERS_PARALLELISM"]`, config imports, `logger`, singletons (`_llm = None`, `_vector_dbs = {}`, `_embedding_model = None`), global `search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)`.

Adiciona:
- **2 wrappers** (call-time DIR_VECTORDB resolution — ver §4.3):
  - `_listar_colecoes()` wrapper que le `agent.DIR_VECTORDB` e chama `retriever._listar_colecoes(vectordb_dir=DIR_VECTORDB)`
  - `_get_vector_db(collection_name)` wrapper que le `agent.DIR_VECTORDB` e chama `retriever._get_vector_db(collection_name, vectordb_dir=DIR_VECTORDB)`
- **1 wrapper de singleton getter** (`_get_embedding_model` wrapper — ver §4.2 DN-2a):
  - Mantem o padrao atual de ler/setar `_embedding_model` global de `agent.py` diretamente, delegando a construcao a `retriever._build_embedding_model()`
- **Bloco de shims identity** (cada um com marcador `# Re-export shim (ADR-018 Fase 4) — remover na Fase 6`):
  - De `state`: `AgentState`
  - De `generation`: `LLMClient`, `_llm_retry`, `prompt_rag_retrieve`, `prompt_gerar`, `_prompt_web`, `_get_llm`, `_invoke_chain`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas`
  - De `retriever`: `_get_retriever`, `invalidar_vector_db`, `_build_embedding_model` (factory sem singleton)
  - De `graph`: `passo_decisao_agente`, `usar_ferramenta_web`, `retrieve_info`, `inicializar_recursos`, `agent_workflow`

**Estimativa de linhas:** `agent.py` atual ~474 ln - (estado ~20 + geracao ~250 + retriever ~220 + graph ~150) + (shims ~30 + 3 wrappers ~40 + search global + singletons ~15 + docstring+imports ~40) = **~160-190 ln** (bem dentro do alvo ~180).

---

## 4. Mecanica de seam (a parte delicada) — RETR-08, RETR-11

### 4.1 Singletons acessiveis via lazy import DENTRO do corpo (DN-2)

Todos os modulos de `retrieval/` que precisam acessar `_llm`, `_embedding_model`, `_vector_dbs` — os 3 singletons que ficam fisicamente em `agent.py` — usam `import agenticlog.agent as _agent_mod` ou `from agenticlog.agent import <simbolo>` DENTRO do corpo da funcao, NUNCA em nivel de modulo.

```python
# retrieval/generation.py — correto
def _get_llm():
    from agenticlog.agent import _llm  # lazy import, resolvido a cada chamada
    # se _llm for None, precisa setar via import agenticlog.agent as _agent_mod
    import agenticlog.agent as _agent_mod
    if _agent_mod._llm is None:
        _agent_mod._llm = ChatOpenAI(...)
    return _agent_mod._llm
```

Isso garante:
1. O modulo `agent.py` e importado APENAS quando a funcao e chamada pela primeira vez (nao durante o import de `retrieval/`).
2. Cada chamada ve o valor ATUAL de `agent._llm` (que pode ter sido setado por `inicializar_recursos` ou monkeypatchado pelo oraculo).
3. **Nao ha ciclo de import** porque `retrieval/*` nunca importa `agent` em top-level — a aresta `agent -> retrieval.generation` existe em `agent.py` (shim `from agenticlog.retrieval.generation import ...`), mas `retrieval.generation` so importa `agent` quando a funcao e invocada (em tempo de execucao, nao de import).

### 4.2 Wrapper de `_get_embedding_model` em agent.py (DN-2a)

Conforme analisado em §3.3, `_get_embedding_model` precisa ler E SETAR o singleton `_embedding_model` de `agent.py`. Como `from agenticlog.agent import _embedding_model` cria uma copia local do nome (reatribuir nao modifica `agent._embedding_model`), a solucao e manter `_get_embedding_model` como **wrapper** em `agent.py`:

```python
# agent.py — wrapper
import agenticlog.retrieval.retriever as _retr

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _retr._build_embedding_model()
    return _embedding_model
```

E em `retrieval/retriever.py`:
```python
def _build_embedding_model() -> HuggingFaceEmbeddings:
    """Factory do modelo de embedding — sem singleton. Usado por agent._get_embedding_model()."""
    return criar_embedding_model()
```

O wrapper de `agent.py` e o UNICO lugar que le/atribui `_embedding_model` global. `avalia_similaridade` em `generation.py` acessa o modelo via lazy import de `agent._get_embedding_model` (a funcao wrapper). O oraculo patcha `agent._embedding_model` diretamente — o wrapper ve o stub porque faz `if _embedding_model is None` e o stub nao e None.

### 4.3 Wrappers de `_listar_colecoes` e `_get_vector_db` (RETR-11)

Mesmo padrao ADR-019 D4 (Fase 3b): wrappers em `agent.py` que resolvem `DIR_VECTORDB` no momento da chamada e passam como argumento.

```python
# agent.py — wrappers
def _listar_colecoes():
    return _retr._listar_colecoes(vectordb_dir=DIR_VECTORDB)

def _get_vector_db(collection_name: str):
    return _retr._get_vector_db(collection_name, vectordb_dir=DIR_VECTORDB)
```

As funcoes em `retriever.py` sao parametrizadas:
```python
# retrieval/retriever.py
def _listar_colecoes(*, vectordb_dir: Path | None = None) -> list[str]:
    actual_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
    # ... corpo usando actual_dir ...

def _get_vector_db(collection_name: str, *, vectordb_dir: Path | None = None) -> Chroma:
    actual_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
    # ... corpo usando actual_dir ...
```

### 4.4 Resolucao de `_get_retriever` chamando wrappers (DN-3)

O oraculo patcha `agent.DIR_VECTORDB`. `_get_retriever` (que e shim identity, move para `retriever.py`) chama `_listar_colecoes` e `_get_vector_db` internamente. Se chamasse `retriever._listar_colecoes()` diretamente (sem `vectordb_dir`), o fallback leria `config.DIR_VECTORDB` — nao o monkeypatch.

**Solucao (DN-3):** `_get_retriever` em `retriever.py` acessa `_listar_colecoes` e `_get_vector_db` via lazy import de `agenticlog.agent` (as versoes WRAPPER que resolvem `DIR_VECTORDB`):

```python
# retrieval/retriever.py
def _get_retriever(query: str) -> list[Document]:
    from agenticlog.agent import _listar_colecoes  # lazy — o wrapper de agent
    from agenticlog.agent import _get_vector_db    # lazy — o wrapper de agent
    
    collection_names = _listar_colecoes()  # chama o wrapper -> le agent.DIR_VECTORDB
    # ... resto verbatim ...
```

Isso garante que cada chamada a `_get_retriever` resolve `DIR_VECTORDB` atraves do wrapper de `agent`, que le `agent.DIR_VECTORDB` no momento da chamada — exatamente o que o oraculo patcha.

---

## 5. Search binding resolution (DN-1) — a sutileza critica

### O problema

O oraculo faz:
```python
monkeypatch.setattr("agenticlog.agent.search", fake)
```

Se `search` e movido para `graph.py` e agent.py tem shim `from agenticlog.retrieval.graph import search`:
1. `monkeypatch.setattr("agenticlog.agent.search", fake)` substitui `agent.search` pelo fake.
2. Mas `graph.usar_ferramenta_web` referencia `search` como **nome de nivel de modulo** de `graph.py` — capturado no momento do import de `graph.py`, quando `agent.search` ainda era o original.
3. `graph.usar_ferramenta_web` veria o `search` ORIGINAL, nao o fake.

Este e o mesmo problema de `DIR_VECTORDB` trapped-name (ADR-019 D4).

### Decisao: search permanece FISICAMENTE em agent.py, graph.py acessa via lazy import (DN-1)

**DECISAO (DN-1):** `search` permanece como variavel global fisica em `agent.py`:
```python
# agent.py
search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)
```

`graph.py` acessa `search` via lazy import DENTRO do corpo de `usar_ferramenta_web`:
```python
# retrieval/graph.py
def usar_ferramenta_web(state: AgentState) -> AgentState:
    from agenticlog.agent import search  # lazy import, resolvido a cada chamada
    # ... usa search ...
```

**Nao ha identity shim para `search` em `agent.py`** — `search` nunca sai de `agent.py`. `agent.search` e a definicao real (nao um shim). O oraculo patcha `agent.search` -> funciona.

**Alternativa rejeitada — (a) search em graph.py + lazy import de agent.py:** `search` estaria em `graph.py`, `agent.py` teria `from agenticlog.retrieval.graph import search`, e `usar_ferramenta_web` faria `from agenticlog.agent import search` lazy. Isso funciona, mas adiciona um nivel de indirecao desnecessario (search vive em graph.py mas ninguem o importa de la diretamente — sempre via agent.py). A escolha de manter search em agent.py e mais direta e segue o principio de minimo movimento para simbolos que sao patchados pelo oraculo.

**Justificativa:** `search` e o unico global de `agent.py` que e uma instancia concreta (nao um singleton lazy) e o unico que o oraculo patcha com `monkeypatch.setattr` direto em `agent`. Manter `search` fisicamente em `agent.py` — o mesmo padrao de `_rag_embedding_model` em `rag.py` (Fase 3a) — e a opcao mais segura e com menor churn. `graph.py` importa `search` lazy, exatamente como `retriever.py` importa `_listar_colecoes` lazy de `agent` (DN-3).

---

## 6. Contrato de identidade vs. delegacao — tabela por-simbolo

| Simbolo | Em `agent.py` | Destino em `retrieval/` | Contrato | Resolucao de seam |
|---------|-------------|------------------------|----------|-------------------|
| `AgentState` | shim `from .state import AgentState` | `state.py` | **`is`-identico** | N/A (dados puros) |
| `LLMClient` | shim `from .generation import LLMClient` | `generation.py` | **`is`-identico** | N/A (Protocol) |
| `_llm_retry` | shim `from .generation import _llm_retry` | `generation.py` | **`is`-identico** | Aplica-se a `_invoke_chain` |
| `prompt_rag_retrieve` | shim `from .generation import prompt_rag_retrieve` | `generation.py` | **`is`-identico** | N/A (objeto constante) |
| `prompt_gerar` | shim `from .generation import prompt_gerar` | `generation.py` | **`is`-identico** | N/A |
| `_prompt_web` | shim `from .generation import _prompt_web` | `generation.py` | **`is`-identico** | N/A |
| `_get_llm` | shim `from .generation import _get_llm` | `generation.py` | **`is`-identico** | Acessa `_llm` via lazy import de `agent` (§4.1) |
| `_invoke_chain` | shim `from .generation import _invoke_chain` | `generation.py` | **`is`-identico** | Usa `_get_llm` (local) |
| `gera_multiplas_respostas` | shim `from .generation import gera_multiplas_respostas` | `generation.py` | **`is`-identico** | Usa `_invoke_chain`, `_get_llm` (locais) |
| `avalia_similaridade` | shim `from .generation import avalia_similaridade` | `generation.py` | **`is`-identico** | Usa `_get_embedding_model` via lazy import de `agent._get_embedding_model` (§4.2) |
| `rank_respostas` | shim `from .generation import rank_respostas` | `generation.py` | **`is`-identico** | N/A (dados puros) |
| `_build_embedding_model` | shim `from .retriever import _build_embedding_model` | `retriever.py` | **`is`-identico** | N/A (factory pura) |
| `_get_embedding_model` | **WRAPPER** (ver §4.2 DN-2a) | `agent.py` (wrapper) + `retriever._build_embedding_model()` | **Nao `is`-identico** | Le/seta `_embedding_model` global diretamente; delega construcao a `_build_embedding_model` |
| `_get_vector_db` | **WRAPPER** (ver §4.3) | `agent.py` (wrapper) + `retriever._get_vector_db` parametrizada | **Nao `is`-identico** | Le `agent.DIR_VECTORDB` e passa como `vectordb_dir` |
| `_listar_colecoes` | **WRAPPER** (ver §4.3) | `agent.py` (wrapper) + `retriever._listar_colecoes` parametrizada | **Nao `is`-identico** | Le `agent.DIR_VECTORDB` e passa como `vectordb_dir` |
| `_get_retriever` | shim `from .retriever import _get_retriever` | `retriever.py` | **`is`-identico** | Acessa `_listar_colecoes`/`_get_vector_db` via lazy import dos wrappers de `agent` (§4.4 DN-3) |
| `invalidar_vector_db` | shim `from .retriever import invalidar_vector_db` | `retriever.py` | **`is`-identico** | Acessa `_vector_dbs` via lazy import de `agent` (§4.1) |
| `passo_decisao_agente` | shim `from .graph import passo_decisao_agente` | `graph.py` | **`is`-identico** | N/A (dados puros) |
| `usar_ferramenta_web` | shim `from .graph import usar_ferramenta_web` | `graph.py` | **`is`-identico** | Acessa `search` via lazy import de `agent` (§5 DN-1) |
| `retrieve_info` | shim `from .graph import retrieve_info` | `graph.py` | **`is`-identico** | Usa `_get_retriever` (local, shim de retriever) |
| `inicializar_recursos` | shim `from .graph import inicializar_recursos` | `graph.py` | **`is`-identico** | Acessa getters lazy de `agent` (§3.4) |
| `agent_workflow` | shim `from .graph import agent_workflow` | `graph.py` | **`is`-identico** | Compilado em graph.py com nos locais |
| `search` | **DEFINICAO FISICA** (global `agent.py`) | **NAO move** | N/A (var global) | Permanece em agent.py; graph.py le lazy (§5) |
| `_llm` | **DEFINICAO FISICA** (global `agent.py`) | **NAO move** | N/A (var global) | Permanece em agent.py; generation.py le lazy |
| `_embedding_model` | **DEFINICAO FISICA** (global `agent.py`) | **NAO move** | N/A (var global) | Permanece em agent.py; wrapper de `_get_embedding_model` le/seta |
| `_vector_dbs` | **DEFINICAO FISICA** (global `agent.py`) | **NAO move** | N/A (var global) | Permanece em agent.py; retriever/invalidation le lazy |

### Resumo da estrategia

| Tipo | Quantidade | Exemplos |
|------|-----------|----------|
| Shim `is`-identico | ~20 simbolos | `AgentState`, `LLMClient`, `_invoke_chain`, `gera_*`, `avalia_*`, `rank_*`, `passo_*`, `_get_retriever`, `invalidar_vector_db`, `inicializar_recursos`, `retrieve_info`, `usar_ferramenta_web`, `agent_workflow`, prompts, `_llm_retry`, `_get_llm`, `_build_embedding_model` |
| Wrapper (seam binding) | 3 simbolos | `_get_embedding_model` (singleton), `_get_vector_db` (DIR_VECTORDB), `_listar_colecoes` (DIR_VECTORDB) |
| Definicao fisica (nao move) | 4 simbolos | `_llm`, `_embedding_model`, `_vector_dbs`, `search` |

---

## 7. Error Handling Strategy

| Scenario | Handling | Impact |
|----------|----------|--------|
| LLM connection failure (todas as tentativas) | `_llm_retry` exaure retries, `reraise=True` propaga a ultima excecao | Caller (nodo do grafo) nao trata — excecao propaga para `agent_workflow.invoke()` -> `app.py` exibe erro |
| DuckDuckGo indisponivel | `usar_ferramenta_web` captura `Exception`, loga WARNING, retorna `ranked_response="Busca indisponivel no momento."`, `confidence=0.0` | Fallback gracioso sem quebra |
| ChromaDB query fail (qualquer colecao) | `_get_retriever` propaga excecao (fail-fast) | Nao engole erros de banco vetorial |
| Retrieval vazio | `retrieve_info` loga WARNING, seta `next_step="gerar"` | Degradacao para geracao sem contexto |
| `invalidar_vector_db` `ImportError` | `logger.warning` | Nao-fatal (shim de import opcional) |

---

## 8. Tech Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Localizacao de singletons (Open Q1) | `_llm`, `_embedding_model`, `_vector_dbs` ficam fisicamente em `agent.py` | Preserva monkeypatch do oraculo (`agent._embedding_model`, `agent._llm`, `agent._vector_dbs`); seguiu o precedente Fase 3a (`_rag_embedding_model` em `rag.py`). |
| Localizacao de `search` (DN-1) | `search` fica fisicamente em `agent.py`; `graph.py` acessa via lazy import | Unico global concreto patchado pelo oraculo (`agent.search`); minimo movimento para simbolos patchados. |
| Shim vs. wrapper para `_get_embedding_model` (DN-2a) | **Wrapper** em `agent.py` que le/seta `_embedding_model` global | Python nao permite reatribuir nome importado; wrapper mantem o singleton setavel via `agent._embedding_model = stub`. |
| `_get_retriever` chama wrappers via lazy import (DN-3) | Lazy import de `agent._listar_colecoes` e `agent._get_vector_db` dentro do corpo | Garante resolucao de `DIR_VECTORDB` no call time; mesmo padrao ADR-019 D4. |
| `_llm_retry` move com `_invoke_chain` | Para `generation.py` | `_llm_retry` e um decorator que envolve `_invoke_chain`; movendo juntos preservam a gramatica de marcacao. |
| `criar_embedding_model` reuse (Open Q4) | `retriever._build_embedding_model` chama `criar_embedding_model()` de `ingestion/embeddings` | Mesmo espaco vetorial (§3 arquitetura-alvo); sem duplicacao de config. |
| Prompt templates identity `is` | Shim `from agenticlog.retrieval.generation import prompt_rag_retrieve` | `agent.prompt_rag_retrieve is retrieval.generation.prompt_rag_retrieve` — testes de identidade nao quebram. |

---

## 9. Argumento de acicidade — RETR-13

Grafo de import (todas as arestas direcionais, DAG):

```
config (folha)
  -> ingestion/embeddings (folha — so importa config)
  -> retrieval/state (folha — so importa pydantic, stdlib)
  -> retrieval/generation (so importa config, stdlib, langchain*, sklearn*, pydantic;
                           lazy import de agent DENTRO de funcoes — nao em tempo de import)
  -> retrieval/retriever (so importa config, stdlib, langchain*, ingestion/embeddings;
                           lazy import de agent DENTRO de funcoes)
  -> retrieval/graph (so importa config, stdlib, langgraph*, retrieval.{state,generation,retriever};
                       lazy import de agent DENTRO de funcoes)
  -> agent (importa config, stdlib, langchain*, retrieval.* shims)
  -> app/api/__init__ (importam de agent)
```

Nenhum modulo de `retrieval/` importa `agent` em nivel de modulo (top-level). Todos os acessos a singletons de `agent` sao lazy imports DENTRO de corpos de funcao — que so ocorrem em tempo de execucao, nao em tempo de import. Logo `import agenticlog.retrieval.state`, `import agenticlog.retrieval.generation`, `import agenticlog.retrieval.retriever`, `import agenticlog.retrieval.graph` em interpretador frio nao formam ciclo.

`import agenticlog.agent` importa `retrieval.*` em top-level (shims), que importam `config` e stdlib (folhas) — mas `retrieval.*` nunca importam `agent` em top-level. Logo o grafo e estritamente DAG.

Verificacao por `subprocess` (padrao `test_rag_shared_observability.py::test_ac06`), adicionada no NOVO `tests/acceptance/test_rag_retrieval_fase4.py`, asserindo exit 0 sem "circular"/"partially initialized" em stderr.

---

## 10. Mitigacoes dos itens de risco (spec §Risks / CONCERNS.md)

| Risco | Mitigacao de design |
|-------|---------------------|
| Seam de `search` capturado em tempo de import | `search` fica em `agent.py`; `graph.py` acessa via lazy import no corpo de `usar_ferramenta_web` — mesma tecnica ADR-019 D4 para `DIR_VECTORDB`. |
| Singletons com trapped-name | `retrieval/*` acessa `_llm`/`_embedding_model`/`_vector_dbs` via `import agenticlog.agent as _agent_mod` lazy no corpo. `_get_embedding_model` wrapper em `agent` resolve o problema de reatribuicao. |
| `patch.object(agent_mod, ...)` quebra | `_listar_colecoes` e `_get_vector_db` permanecem como wrappers em `agent.py` (funcoes definidas em `agent.py`, nao shims). `patch.object` funciona. Simbolos que viram shims (ex.: `_get_retriever`) tem alvo migrado para `"agenticlog.retrieval.retriever._get_retriever"`. |
| `DIR_VECTORDB` trapped-name | Wrappers em `agent.py` que resolvem `agent.DIR_VECTORDB` no call time (§4.3). `_get_retriever` acessa os wrappers via lazy import (§4.4). |
| Import circular | Todos os acessos a `agent` de `retrieval/*` sao lazy imports DENTRO de funcoes — nunca top-level. Grafo e DAG (§9). |
| Perda de identidade `is` nos prompts | Shims identity `from retrieval.generation import prompt_rag_retrieve` — `is` preservado. |
| `inicializar_recursos` coordenacao | Ordem de inicializacao (embeddings -> vdb -> llm) mantida dentro de `graph.inicializar_recursos`. |
| hnswlib SAC Windows | CI Linux e gate autoritativo. |
| Missing error handling / logging (CONCERNS) | Preservado verbatim; os mesmos `logger.warning`/`error` de `agent.py` movem com as funcoes para `retrieval/*`. |

---

## 11. ADR draft consideration

Esta fase envolve decisoes de design significativas que merecem documentacao permanente:

1. **Singleton preservation** (`_llm`, `_embedding_model`, `_vector_dbs` fisicamente em `agent.py`)
2. **Lazy import pattern** para acesso a singletons de `retrieval/*` (DN-2, DN-2a)
3. **Search binding decision** (DN-1 — search fica em agent.py com lazy import em graph.py)
4. **Wrapper pattern** para `DIR_VECTORDB` (DN-3, continuacao ADR-019 D4)
5. **Tabela shim-vs-wrapper** (§6)

Estas decisoes podem ser consolidadas em um **ADR-020** documentando a extracao do pacote `retrieval/`. O ADR NAO deve ser escrito nesta fase (e uma decisao separada do humano), mas o draft de conteudo esta delineado neste `design.md`. O marcador nos shims deve incluir `# Re-export shim (ADR-018 Fase 4) — remover na Fase 6`, estabelecendo a divida tecnica explicita.

---

## 12. Orientacoes para migracao de patch-target (RETR-12)

Cada teste com `@patch("agenticlog.agent.X")` onde `X` e um simbolo que:
1. **Moveu fisicamente para `retrieval/<mod>.py` E virou shim identity:** o patch-target muda para `"agenticlog.retrieval.<mod>.X"`.
2. **Permanece como wrapper em `agent.py`:** o patch-target **nao muda** — `"agenticlog.agent.X"` ainda e a definicao real.
3. **E singleton fisico (`_llm`, `_embedding_model`, `_vector_dbs`, `search`):** o patch-target **nao muda** — o simbolo nunca saiu de `agent.py`.

Regra pratica: se `agent.X is retrieval.mod.X` (identity shim), o patch migra. Se `agent.X` e uma funcao distinta (wrapper), o patch fica. Se o simbolo nao esta em `retrieval/` (singleton, search), o patch fica.

O inventario completo (~32 sites) esta em `spec.md §Tests Required`. A migracao e puramente mecanica (substituicao de namespace) — nenhuma assercao de comportamento e alterada.

---

## 13. Verificacao pos-extracao

| Verificacao | Comando / Metodo | Gate |
|------------|------------------|------|
| Acicidade | `subprocess import agenticlog.retrieval.*` -> exit 0 | full |
| Identidade `is` | `agent.X is retrieval.mod.X` para todos os shims | unit |
| Identity preserve (search) | monkeypatch `agent.search` + chamar `usar_ferramenta_web` -> ve o fake | integration |
| Identity preserve (singletons) | monkeypatch `agent._embedding_model` + chamar `_get_embedding_model` (wrapper) -> ve o stub | integration |
| Identity preserve (DIR_VECTORDB) | monkeypatch `agent.DIR_VECTORDB` + chamar `_listar_colecoes` (wrapper) -> ve o path patchado | integration |
| Line count | `wc -l agent.py` < 200 | quick |
| Oraculo zero-diff | `git diff tests/test_rag_caracterizacao.py tests/ingestion/test_shims_identidade.py` vazio | full |
| Suite completa | `pytest --cov=agenticlog -v` verde | build |
| Lint | `ruff`, `black --check`, `isort --check` limpos | build |
