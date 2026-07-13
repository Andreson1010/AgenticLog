# ADR-018 Fase 5 — extracao do pacote `serving/` de `api.py` + `health.py` — Design

**Path:** `.specs/features/rag-serving-fase5/design.md`
**Spec:** `.specs/features/rag-serving-fase5/spec.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## 1. Architecture Overview

A Fase 5 extrai `api.py` (397 ln) e `health.py` (131 ln) para o pacote `agenticlog.serving/`, completando o bloco SERVING da arquitetura-alvo (ADR-018 SS7 item 5):

```
src/agenticlog/
  serving/
    __init__.py    — re-exports + __all__
    api.py         — FastAPI app, endpoints, helpers, exception handlers (verbatim de api.py)
    health.py      — health check LMStudio (verbatim de health.py)
```

Relacao com outros pacotes:

```
config (folha)
  -> agent.py / retrieval/ (singletons, workflow)
  -> serving/api.py  (importa agent, health, history, config, rag)
  -> serving/health.py (importa config)
  -> api.py (shim -> serving/api.py)
  -> health.py (shim -> serving/health.py)
  -> __init__.py (importa de agenticlog.api e agenticlog.health — via shims)
  -> main_api.py / app.py (importa de agenticlog.api — via shim)
```

`serving/api.py` importa de:
- `agenticlog.agent` (inicializar_recursos, AgentState, agent_workflow)
- `agenticlog.config` (DEFAULT_COLLECTION_NAME, DIR_VECTORDB, etc.)
- `agenticlog.health` (check_lmstudio_health, LMStudioUnavailableError, ModeloNaoCarregadoError)
- `agenticlog.history` (HistoryStore)
- `agenticlog.rag` (_get_rag_embedding_model)

`serving/health.py` importa de:
- `agenticlog.config` (LLM_API_BASE, LLM_HEALTH_CHECK_TIMEOUT_SECONDS, LLM_MODEL)

Nenhum destes modulos importa `serving/` — o grafo e estritamente DAG. `api.py` (shim) importa `serving.api` mas `serving.api` nunca importa `api.py` — aciclico.

---

## 2. Code Reuse Analysis

### Componentes existentes a alavancar

| Component | Location | How to Use |
|-----------|----------|-----------|
| Padrao shim `is`-identity | `rag.py:55-87`, `agent.py:80-110`, Fases 3a/3b/4 | Aplicar o MESMO bloco de re-export para cada symbol de `serving/` em `api.py` e `health.py`. |
| Padrao `import httpx` + `import logging` no shim | `rag.py` shim block (mantem `import fitz` para `@patch("agenticlog.rag.fitz")` resolver) | `health.py` shim mantem `import httpx` + `import logging` no topo para `@patch("agenticlog.health.httpx.Client")` e `@patch("agenticlog.health.logger")` resolverem. |
| Teste de acicidade fresh-interpreter | `tests/ingestion/test_shims_identidade.py::TestIngestionAcyclic` | Estender o padrao `subprocess` para `agenticlog.serving`. |
| Teste de round-trip identidade | `tests/ingestion/test_shims_identidade.py` | Novo `tests/acceptance/test_rag_serving_fase5.py` com mesmo padrao. |
| Logger string explicita | Fase 4 `agent.py` mantem `logging.getLogger(__name__)` mas serving/ usa string fixa | Pattern novo para esta fase: `logging.getLogger("agenticlog.api")` em vez de `__name__`. |

### Integracao

| System | Integration Method |
|--------|--------------------|
| `main_api.py` | `uvicorn.run("agenticlog.api:app")` — resolve via shim `agenticlog/api.py` (SERV-05). |
| `app.py` | Importa `_listar_colecoes` de `agenticlog.agent`, `adicionar_documento_incrementalmente` de `agenticlog.rag`, e a string `agenticlog.api:app` na mensagem de erro. Nao importa `serving/` diretamente. |
| `__init__.py` | Importa `AgentState`, `agent_workflow` de `agenticlog.agent`; `check_lmstudio_health`, `LMStudioUnavailableError`, `ModeloNaoCarregadoError` de `agenticlog.health` (shim). Permanece inalterado. |
| Historico (tests) | `test_api.py`, `test_health.py` e acceptance tests importam de `agenticlog.api` / `agenticlog.health` (shims preservados). |

---

## 3. Componentes — `serving/api.py`

Arquivo novo com conteudo verbatim de `src/agenticlog/api.py` com 2 adaptacoes:

1. **Logger name:** `logger = logging.getLogger("agenticlog.api")` (string explicita, nao `__name__`).
2. **Nenhuma outra alteracao de logica.**

Tamanho estimado: ~397 ln (igual ao original).

### Simbolos publicos (21 total):

| Simbolo | Tipo | Importado como nome local em api.py? | Patch class |
|---------|------|--------------------------------------|-------------|
| `app` | FastAPI instance | N/A (definido em api.py) | (c) — shim re-exporta |
| `consultar` | async function | N/A (definido em api.py) | (c) — shim re-exporta |
| `listar_historico` | async function | N/A (definido em api.py) | (c) — shim re-exporta |
| `QueryRequest` | Pydantic model | N/A (definido em api.py) | (c) — shim re-exporta |
| `QueryResponse` | Pydantic model | N/A (definido em api.py) | (c) — shim re-exporta |
| `HistoryEntry` | Pydantic model | N/A (definido em api.py) | (c) — shim re-exporta |
| `DocumentInfo` | Pydantic model | N/A (definido em api.py) | (c) — shim re-exporta |
| `lifespan` | async context manager | N/A (definido em api.py) | (c) — shim re-exporta |
| `_verificar_vectordb` | function | N/A (definido em api.py) | (c) — shim re-exporta |
| `_serializar_documentos` | function | N/A (definido em api.py) | (c) — shim re-exporta |
| `_normalizar_estado` | function | N/A (definido em api.py) | (c) — shim re-exporta |
| `_resposta_segura` | function | N/A (definido em api.py) | (c) — shim re-exporta |
| `_construir_registro` | function | N/A (definido em api.py) | (c) — shim re-exporta |
| `handler_lmstudio` | exception handler | N/A (definido em api.py) | (c) — shim re-exporta |
| `handler_connect_error` | exception handler | N/A (definido em api.py) | (c) — shim re-exporta |
| `handler_generico` | exception handler | N/A (definido em api.py) | (c) — shim re-exporta |
| `MSG_LMSTUDIO_INDISPONIVEL` | string constant | N/A (definido em api.py) | (c) — shim re-exporta |
| `MSG_VECTORDB_AUSENTE` | string constant | N/A (definido em api.py) | (c) — shim re-exporta |
| `_ERROS_MODO_SEGURO` | tuple | N/A (definido em api.py) | (c) — shim re-exporta |

### Imports que viram nomes LOCAIS em serving/api.py:

Estes sao importados de outros modulos (ex.: `from agenticlog.agent import inicializar_recursos`). Quando `serving/api.py` faz `from agenticlog.agent import inicializar_recursos`, o nome `inicializar_recursos` e uma referencia local que NAO passa pelo namespace `agenticlog.api`.

| Simbolo importado | Origem | Em serving/api.py | Patch-target ATUAL | Nova classe | Acao |
|-------------------|--------|-------------------|---------------------|-------------|------|
| `inicializar_recursos` | `agenticlog.agent` | nome local | `agenticlog.api.inicializar_recursos` | (a) | **Migrar para** `agenticlog.serving.api.inicializar_recursos` |
| `AgentState` | `agenticlog.agent` | nome local | — (nao patchado) | N/A | N/A |
| `agent_workflow` | `agenticlog.agent` | nome local | `agenticlog.api.agent_workflow.invoke` | (a) | **Migrar para** `agenticlog.serving.api.agent_workflow.invoke` |
| `check_lmstudio_health` | `agenticlog.health` | nome local | `agenticlog.api.check_lmstudio_health` | (a) | **Migrar para** `agenticlog.serving.api.check_lmstudio_health` |
| `LMStudioUnavailableError` | `agenticlog.health` | nome local | — (nao patchado diretamente) | N/A | N/A |
| `ModeloNaoCarregadoError` | `agenticlog.health` | nome local | — (nao patchado diretamente) | N/A | N/A |
| `HistoryStore` | `agenticlog.history` | nome local | `agenticlog.api.HistoryStore` | (a) | **Migrar para** `agenticlog.serving.api.HistoryStore` |
| `_get_rag_embedding_model` | `agenticlog.rag` | nome local | `agenticlog.api._get_rag_embedding_model` | (a) | **Migrar para** `agenticlog.serving.api._get_rag_embedding_model` |
| `DIR_VECTORDB` | `agenticlog.config` | nome local via `from config import DIR_VECTORDB` | `agenticlog.api.DIR_VECTORDB` | (a) | **Migrar para** `agenticlog.serving.api.DIR_VECTORDB` |
| `Chroma` | `langchain_chroma` | nome local | `agenticlog.api.Chroma` | (a) | **Migrar para** `agenticlog.serving.api.Chroma` |
| `asyncio` | stdlib | nome local (modulo) | `agenticlog.api.asyncio.to_thread` | (a) | **Migrar para** `agenticlog.serving.api.asyncio.to_thread` |
| `httpx` | third-party | nome local (modulo) | `agenticlog.api.httpx.*` (exception handlers) | (a) | **Migrar para** `agenticlog.serving.api.httpx.*` |

### Simbolos que sao so atributos do shim (classe b):

Nenhum simbolo de api.py ou health.py se enquadra em classe (b) "atributo do shim" — todos os simbolos definidos em api.py sao definicoes reais que sao re-exportadas pelo shim via `from agenticlog.serving.api import ...`. O shim e a ponte de re-export, e o teste que referencia `agenticlog.api.app` ve o mesmo objeto via shim. Nao ha atributos que existam SOMENTE no shim.

### Simbolos de terceiros (classe c):

| Simbolo | Importado em health.py como | Patch-target ATUAL | Acao |
|---------|---------------------------|---------------------|------|
| `httpx.Client` | `import httpx` (modulo topo) | `agenticlog.health.httpx.Client` | **Manter** — shim `agenticlog/health.py` mantem `import httpx` no topo |
| `logger` | `logger = logging.getLogger("agenticlog.health")` | `agenticlog.health.logger` (via `patch.object(health_module, "logger")`) | **Manter** — shim mantem `import logging` + `logger = ...` no topo |

---

## 4. Componentes — `serving/health.py`

Arquivo novo com conteudo verbatim de `src/agenticlog/health.py` com 1 adaptacao:

1. **Logger name:** `logger = logging.getLogger("agenticlog.health")` (string explicita, nao `__name__`).

Tamanho estimado: ~131 ln (igual ao original).

### Simbolos publicos (8 total):

| Simbolo | Tipo |
|---------|------|
| `check_lmstudio_health` | function |
| `reset_health_check_sentinel` | function |
| `LMStudioUnavailableError` | exception class |
| `ModeloNaoCarregadoError` | exception class |
| `_health_checked` | module-level bool |
| `_extrair_ids_modelos` | function |
| `MAX_MODELOS_LOG` | module-level int |
| `logger` | module-level Logger (preservado no shim) |

---

## 5. Inventario por-simbolo de patch-target migration — CRITICO

### Metodologia

Para cada `@patch("agenticlog.api.X")` em testes, verificar se X e:

- **(a) Nome local em serving/api.py:** X foi importado de outro modulo (ex.: `from agenticlog.agent import inicializar_recursos`). `serving/api.py` ja tem sua propria referencia a `inicializar_recursos` — o namespace `agenticlog.api` (shim) nao e consultado em tempo de execucao. **Patch DEVE migrar** para `agenticlog.serving.api.X`.

- **(b) Atributo do shim:** X e um simbolo que so existe no namespace do shim `agenticlog.api` e nao e importado como nome local em serving/api.py. **Patch PERMANECE** em `agenticlog.api.X`.

- **(c) Terceiro ou definicao local:** X e um modulo/simbolo de terceiros (ex.: `httpx.Client`) ou e definido no proprio api.py (ex.: `app`, `_normalizar_estado`). **Patch mantem** `agenticlog.<mod>.X` se o shim preservar o namespace, OU migra para `serving.<mod>.X` se o patch precisar interceptar a definicao.

### Tabela completa

| Patch-target ATUAL | Simbolo | Onde e definido | Em serving/api.py | Classe | Acao | Testes afetados |
|--------------------|---------|-----------------|-------------------|-------|------|-----------------|
| `agenticlog.api.inicializar_recursos` | `inicializar_recursos` | `agenticlog.agent` | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.inicializar_recursos` | test_api.py (24x), test_api_query_endpoint.py (13x), test_modo_seguro.py (5x), test_history_endpoint.py (2x), test_query_history_audit_logging.py (2x) |
| `agenticlog.api._verificar_vectordb` | `_verificar_vectordb` | **serving/api.py** | definido localmente | (c) | **Migrar** `-> agenticlog.serving.api._verificar_vectordb` | test_api.py (24x), test_modo_seguro.py (5x) |
| `agenticlog.api.check_lmstudio_health` | `check_lmstudio_health` | `agenticlog.health` (shim) | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.check_lmstudio_health` | test_api.py (24x), test_api_query_endpoint.py (13x) |
| `agenticlog.api.agent_workflow.invoke` | `agent_workflow` | `agenticlog.agent` | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.agent_workflow.invoke` | test_api.py (24x), test_modo_seguro.py (5x), test_history_endpoint.py (2x), test_query_history_audit_logging.py (2x) |
| `agenticlog.api.HistoryStore` | `HistoryStore` | `agenticlog.history` | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.HistoryStore` | test_api.py (24x), test_modo_seguro.py (5x) |
| `agenticlog.api.DIR_VECTORDB` | `DIR_VECTORDB` | `agenticlog.config` | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.DIR_VECTORDB` | test_api.py (2x: test_15, test_16) |
| `agenticlog.api.Chroma` | `Chroma` | `langchain_chroma` | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.Chroma` | test_api.py (1x: test_16) |
| `agenticlog.api._get_rag_embedding_model` | `_get_rag_embedding_model` | `agenticlog.rag` | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api._get_rag_embedding_model` | test_api.py (1x: test_16) |
| `agenticlog.api.asyncio.to_thread` | `asyncio` | stdlib | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.asyncio.to_thread` | test_api.py (1x: test_13) |
| `agenticlog.api.httpx.*` (exception handlers) | `httpx` | third-party | nome local (import) | (a) | **Migrar** `-> agenticlog.serving.api.httpx.*` | test_modo_seguro.py (referencia indireta) |
| `agenticlog.health.httpx.Client` | `httpx` | shim health.py | N/A (health shim import) | (c) | **Manter** — shim mantem `import httpx` | test_health.py (14x), test_health_check.py (5x) |
| `agenticlog.health.logger` | `logger` | shim health.py | N/A (health shim define) | (c) | **Manter** — shim mantem `import logging` + `logger = ...` | test_health.py (via `patch.object`), test_health_check.py (3x) |
| `agenticlog.agent.agent_workflow` | `agent_workflow` | agent.py | N/A (test_health_check.py referencia agent direto) | N/A | **Manter** | test_health_check.py |

### Nota sobre o padrao de fixtures compartilhadas

`test_api.py`, `test_api_query_endpoint.py` e `test_modo_seguro_modelo_indisponivel.py` tem funcoes helper (`_client_vectordb_pronto`, `_client`, `_client_ctx`) que encapsulam o conjunto de patches. A migracao de patch-target requer atualizar essas funcoes helper (1 local por arquivo), nao cada teste individualmente. Em `test_api.py`:
- `_client_vectordb_pronto` (linhas 63-75): 5 patches
- `_client_vectordb_ausente` (linhas 78-88): 3 patches

Em `test_modo_seguro_modelo_indisponivel.py`:
- `_client_ctx` (linhas 66-101): 5 patches

Em `test_api_query_endpoint.py`:
- `_client` (linhas 53-67): 5 patches

### Testes com `assert_called_once()` — sensiveis

| Teste | Assert | Patch-target | Risco se nao migrar |
|-------|--------|--------------|---------------------|
| `test_ac_api_10_singletons_initialized_at_startup` | `mock_init.call_count == 1` | `agenticlog.api.inicializar_recursos` | Mock nunca chamado (serving/api.py tem ref local) -> `call_count == 0` -> **teste FALHA** |
| `test_13_workflow_executa_em_thread` | `mock_invoke.call_count == 1` | `agenticlog.api.agent_workflow.invoke` | Mock nunca chamado -> `call_count == 0` -> **teste FALHA** |
| `teste_1_happy_path_retorna_sem_excecao` | `mock_client_cls.assert_called_once_with(...)` | `agenticlog.health.httpx.Client` | **Nao migra** — shim health mantem `import httpx` |

---

## 6. Decisao sobre Open Question Q2 — String `agenticlog.api:app` em entry-points

### Contexto

`main_api.py:12`: `uvicorn.run("agenticlog.api:app", ...)`

`app.py:27`: `MSG_CONNECT_ERROR = "Nao foi possivel conectar ao servidor FastAPI. Inicie com: uvicorn agenticlog.api:app"`

### Opcao A — Manter via shim (recomendada)

Manter ambas as strings como estao. `agenticlog.api:app` resolve porque `agenticlog/api.py` (shim) importa `app` de `serving/api.py` e o re-exporta no namespace `agenticlog.api`.

**Pros:**
- Consistente com Fases 3a/3b/4 (entry points apontam para shims).
- Zero diff em `main_api.py` e `app.py` — arquivos que nao sao alvo desta fase.
- O padrao "shim como ponto de entrada" e documentado e esperado ate a Fase 6.

**Contras:**
- Cria dependencia vitalicia do shim como entry point (landmine documentado para Fase 6).
- Se alguem remover o shim em Fase 6 sem atualizar a string, o servidor nao sobe.

**Veredito:** Manter via shim. E consistente com o padrao estabelecido e minimiza diff. A Fase 6 explicitamente documentara a necessidade de atualizar entry points.

### Opcao B — Atualizar para `agenticlog.serving.api:app`

Atualizar `main_api.py:12` e `app.py:27` para `agenticlog.serving.api:app`.

**Pros:**
- Elimina a landmine para Fase 6.
- Entry point aponta diretamente para o modulo de definicao.

**Contras:**
- Diff extra em arquivos que nao sao alvo da fase (viola "refatoracao pura, zero mudanca de comportamento").
- Inconsistente com Fases 3a/3b/4 que mantiveram shims como entry points.

---

## 7. Decisao sobre Open Question Q4 — Import style em serving/api.py

### Contexto

Em `api.py` atual, imports de outros modulos do projeto usam `from X import Y`:

```python
from agenticlog.agent import AgentState, agent_workflow, inicializar_recursos
from agenticlog.health import LMStudioUnavailableError, ModeloNaoCarregadoError, check_lmstudio_health
from agenticlog.history import HistoryStore
from agenticlog.rag import _get_rag_embedding_model
```

### Opcao A — `from X import Y` (nome local, recomendada)

Manter o estilo `from X import Y` em `serving/api.py`.

**Pros:**
- Preserva codigo verbatim (minimo diff entre api.py e serving/api.py).
- Nenhuma alteracao nas ~70 linhas que usam esses simbolos (ex.: `inicializar_recursos()`, `check_lmstudio_health()`, `AgentState(query=...)`).
- Oráculo `test_rag_caracterizacao.py` nao ve diferenca.

**Contras:**
- Patch-target MIGRA para `agenticlog.serving.api.X` (aproximadamente 50 patches).
- Mas a migracao e mecanica e ocorre UMA vez nesta fase.

### Opcao B — `import agenticlog.agent` (dot-qualified, patch-safe)

Usar `import agenticlog.agent as agent_mod` e `agent_mod.inicializar_recursos()`.

**Pros:**
- Patch-target poderia permanecer `agenticlog.agent.inicializar_recursos` (mas NAO `agenticlog.api.inicializar_recursos` — entao ainda requer migracao).
- Estilo Fase 3b (usado em `ingestion/orchestrator.py` e `ingestion/store.py`).

**Contras:**
- Muda todas as ~70+ call sites em api.py (diff massivo).
- `app.py`, `main_api.py` e outros consumidores importam `agenticlog.api:app` — nao se beneficiam.
- A migracao de patch ainda e necessaria (de `agenticlog.api.X` para `agenticlog.serving.api.X` — nao para `agenticlog.agent.X`).
- Viola "refatoracao pura, zero mudanca de comportamento" — muda assinaturas e estrutura de codigo.

**Veredito:** Manter `from X import Y` (Opcao A). A migracao de patch e inevitavel de qualquer forma (arquivo muda de `api.py` para `serving/api.py`), e manter o estilo original minimiza diff e risco.

---

## 8. Estrategia de logger name

**DECISAO:** `serving/api.py` e `serving/health.py` usam strings explicitas:

```python
# serving/api.py
logger = logging.getLogger("agenticlog.api")

# serving/health.py
logger = logging.getLogger("agenticlog.health")
```

**Racional:**
- Configuracoes de logging existentes (filtros, handlers, _JsonFormatter) referenciam `"agenticlog.api"` e `"agenticlog.health"` por nome.
- `__name__` em `serving/api.py` seria `"agenticlog.serving.api"` — quebraria a configuracao de logging.
- Testes que patcheam `"agenticlog.health.logger"` (ou usam `patch.object(health_module, "logger")`) esperam que o nome `logger` esteja no namespace de `agenticlog.health`.

Implementacao do shim:
```python
# agenticlog/health.py — shim
import httpx
import logging

from agenticlog.serving.health import (  # noqa: E402,F401  # Re-export shim (ADR-018 Fase 5) — remover na Fase 6
    check_lmstudio_health,
    reset_health_check_sentinel,
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    _health_checked,
    _extrair_ids_modelos,
    MAX_MODELOS_LOG,
)

logger = logging.getLogger("agenticlog.health")
```

A importacao `from agenticlog.serving.health import ...` no topo do shim traz todas as definicoes, incluindo a `logger` de serving/health.py — mas ela tem o nome `"agenticlog.health"`. A linha extra `logger = logging.getLogger("agenticlog.health")` no shim garante que o namespace `agenticlog.health.logger` tenha o logger correto para `patch.object(health_module, "logger")`.

---

## 9. Estrategia do shim — agenticlog/api.py e agenticlog/health.py

### api.py (shim)

```python
"""Camada FastAPI do AgenticLog — shim de compatibilidade (ADR-018 Fase 5).

Re-exporta todos os simbolos de agenticlog.serving.api.
Remove na Fase 6.
"""

from agenticlog.serving.api import (  # noqa: F401  # Re-export shim (ADR-018 Fase 5) — remover na Fase 6
    # Modelos Pydantic
    QueryRequest,
    QueryResponse,
    HistoryEntry,
    DocumentInfo,
    # Constantes
    MSG_LMSTUDIO_INDISPONIVEL,
    MSG_VECTORDB_AUSENTE,
    _ERROS_MODO_SEGURO,
    # Helpers
    _verificar_vectordb,
    _serializar_documentos,
    _normalizar_estado,
    _resposta_segura,
    _construir_registro,
    # FastAPI
    lifespan,
    app,
    # Endpoints
    consultar,
    listar_historico,
    # Exception handlers
    handler_lmstudio,
    handler_connect_error,
    handler_generico,
)
```

### health.py (shim)

```python
# AgenticLog - Health check do LMStudio — shim de compatibilidade (ADR-018 Fase 5)
"""Re-exporta todos os simbolos de agenticlog.serving.health.
Remove na Fase 6.
"""

import httpx
import logging

from agenticlog.serving.health import (  # noqa: E402,F401  # Re-export shim (ADR-018 Fase 5) — remover na Fase 6
    check_lmstudio_health,
    reset_health_check_sentinel,
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    _health_checked,
    _extrair_ids_modelos,
    MAX_MODELOS_LOG,
)

logger = logging.getLogger("agenticlog.health")
```

### serving/__init__.py

```python
"""Pacote `serving`: camada de entrega FastAPI + health check (ADR-018 Fase 5).

Extraido de agenticlog.api e agenticlog.health preservando identidade de simbolo
via shims em api.py e health.py.
"""

from agenticlog.serving.api import (
    DocumentInfo,
    HistoryEntry,
    QueryRequest,
    QueryResponse,
    app,
    consultar,
    handler_connect_error,
    handler_generico,
    handler_lmstudio,
    lifespan,
    listar_historico,
    _construir_registro,
    _ERROS_MODO_SEGURO,
    _normalizar_estado,
    _resposta_segura,
    _serializar_documentos,
    _verificar_vectordb,
)
from agenticlog.serving.health import (
    LMStudioUnavailableError,
    MAX_MODELOS_LOG,
    ModeloNaoCarregadoError,
    _extrair_ids_modelos,
    _health_checked,
    check_lmstudio_health,
    reset_health_check_sentinel,
)

__all__ = [
    # api
    "app",
    "consultar",
    "listar_historico",
    "QueryRequest",
    "QueryResponse",
    "HistoryEntry",
    "DocumentInfo",
    "lifespan",
    "_verificar_vectordb",
    "_serializar_documentos",
    "_normalizar_estado",
    "_resposta_segura",
    "_construir_registro",
    "handler_lmstudio",
    "handler_connect_error",
    "handler_generico",
    "MSG_LMSTUDIO_INDISPONIVEL",
    "MSG_VECTORDB_AUSENTE",
    "_ERROS_MODO_SEGURO",
    # health
    "check_lmstudio_health",
    "reset_health_check_sentinel",
    "LMStudioUnavailableError",
    "ModeloNaoCarregadoError",
    "_health_checked",
    "_extrair_ids_modelos",
    "MAX_MODELOS_LOG",
]
```

---

## 10. pyproject.toml — mypy override

Linha 72 atual:

```toml
module = ["agenticlog.rag", "agenticlog.api"]
```

Atualizar para:

```toml
module = ["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]
```

---

## 11. Argumento de acicidade — SERV-09

Grafo de import:

```
config (folha)
  -> agent / retrieval/ (folhas — config + stdlib)
  -> health.py shim (importa serving.health -> config, stdlib)
  -> history (shim -> observability.history)
  -> rag (shim -> ingestion/*)
  -> serving/health.py (config, stdlib)
  -> serving/api.py (agent, health, history, rag, config, stdlib, fastapi, pydantic, httpx)
  -> api.py shim (serving.api)
  -> __init__.py (agent, health shim)
  -> main_api.py (api shim)
  -> app.py (agent, config, rag)
```

Nenhum modulo de `serving/` importa `api.py`, `health.py` (shims), ou `__init__.py`. `serving/api.py` importa de `agent`, `health` (shim), `history`, `rag`, `config` — todos folhas ou mais abaixo na DAG. `agent` nunca importa `serving/`. Logo `import agenticlog.serving` em interpretador frio nao forma ciclo.

---

## 12. Mitigacoes dos itens de risco (spec SSRisks)

| Risco | Mitigacao de design |
|-------|---------------------|
| Patch-target migration (CRITICO — ~50 patches) | Inventario completo em SS5. Cada patch classificado (a/b/c). Testes com `assert_called_once` identificados. Funcoes helper compartilhadas mapeadas. |
| Logger name `__name__` vs string explicita | String explicita `"agenticlog.api"` e `"agenticlog.health"` (SS8). `__name__` NAO usado. |
| Import circular | Grafo DAG confirmado (SS11). `serving/*` nunca importa shims. |
| `import agenticlog.health as health_module` com `patch.object` | Shim health.py mantem `import logging` + `logger = logging.getLogger("agenticlog.health")` no topo. `patch.object(health_module, "logger")` funciona. |
| `from agenticlog.api import app` | Shim re-exporta `app`. `app is serving.api.app`. `TestClient(app)` funciona. |
| `from agenticlog.api import MSG_VECTORDB_AUSENTE` | Shim re-exporta `MSG_VECTORDB_AUSENTE` e `MSG_LMSTUDIO_INDISPONIVEL`. |
| Oráculo zero-diff | Nenhum arquivo oraculo (test_rag_caracterizacao.py, test_shims_identidade.py) e modificado. |
| hnswlib SAC Windows | CI Linux e gate autoritativo. |

---

## 13. Verificacao pos-extracao

| Verificacao | Comando / Metodo | Gate |
|------------|------------------|------|
| Acicidade | `subprocess import agenticlog.serving` -> exit 0 | full |
| Identidade `is` (api) | `agenticlog.api.X is agenticlog.serving.api.X` para ~21 simbolos | unit |
| Identidade `is` (health) | `agenticlog.health.X is agenticlog.serving.health.X` para ~8 simbolos | unit |
| Logger names | `logging.getLogger("agenticlog.api").name == "agenticlog.api"` | unit |
| Patch compatibility | `@patch("agenticlog.health.httpx.Client")` funciona via shim | integration |
| Oraculo zero-diff | `git diff tests/test_rag_caracterizacao.py` vazio | full |
| Suite completa | `pytest --cov=agenticlog -v` verde | build |
| Lint | `ruff`, `black --check`, `isort --check` limpos | build |
| Mypy | `mypy src/agenticlog/serving/` sem erros (override ativo) | build |
