# ADR-018 Fase 6 — Remocao de shims de compatibilidade + reescrita de testes — Design

**Path:** `.specs/features/rag-fase6-shims-remocao/design.md`
**Spec:** `.specs/features/rag-fase6-shims-remocao/spec.md`
**TLC scope:** complex
**Status:** Awaiting human approval

---

## 1. Architecture Overview — Estado pos-remocao

A Fase 6 deleta 6 shims e realoca os ultimos singletons/globals/CLI/side-effects que ainda residiam nos modulos fachada. Apos a remocao, a arvore-alvo (ADR-018 SS3, `docs/arquitetura-alvo-rag.md`) fica COMPLETA e PURA:

```
src/agenticlog/
  config.py                 # cross-cutting (apenas constantes; linha 166 removida)
  ingestion/                # OFFLINE
    __init__.py
    security.py             # validacao/sanitizacao
    extraction.py           # PDF (import fitz) + JSON
    cleaning.py             # descarte de docs vazios
    chunking.py             # SemanticChunker
    embeddings.py           # _rag_embedding_model singleton + _get_rag_embedding_model + criar_embedding_model
    metadata.py             # hash + metadados
    store.py                # escrita Chroma, upsert atomico
    orchestrator.py         # encadeia estagios; lazy import de invalidar_vector_db (LM1)
    cli.py                  # entrypoint `python -m agenticlog.ingestion` (+ _executar_main, _configurar_logging_cli)
  retrieval/                # ONLINE
    __init__.py             # side-effects startup (torch, TOKENIZERS_PARALLELISM)
    retriever.py            # _vector_dbs, _embedding_model, _get_embedding_model, _listar_colecoes,
                            # _get_vector_db, invalidar_vector_db (limpa dict LOCAL)
    generation.py           # _llm singleton + _get_llm (ja tem)
    graph.py                # search (DuckDuckGoSearchAPIWrapper), LangGraph FSM
    state.py                # AgentState (Pydantic)
  observability/
    __init__.py
    logging.py              # _JsonFormatter
    history.py              # HistoryStore (SQLite)
  serving/
    __init__.py             # re-exports com __getattr__ lazy (Q2)
    api.py                  # FastAPI (importa de observability.history, ingestion.embeddings)
    health.py               # health check LMStudio
  shared/
    __init__.py
    errors.py               # RAGSecurityError
```

**NENHUM modulo shim remanescente.** `rg "# Re-export shim" src/` retorna 0.

### Grafo de import pos-delecao

```
config (folha)
  -> ingestion/* (exceto cli.py -> ingestion.orchestrator -> retrieval.retriever)
  -> retrieval/* (exceto retriever -> config; generation -> config; graph -> config, generation, retriever)
  -> observability/* (logging -> config; history -> stdlib)
  -> serving/* (api -> retrieval.graph, serving.health, observability.history, config, ingestion.embeddings;
                health -> config)
  -> shared/* (errors -> stdlib)
  -> __init__.py (retrieval.graph, serving.health, retrieval.state)
```

`retrieval/retriever.py` NAO importa `agent` (agora deletado). `invalidar_vector_db` limpa `_vector_dbs` local. O grafo e estritamente DAG.

---

## 2. Q1 — DECISAO DO HUMANO: Deletar rag.py e agent.py INTEIRAMENTE (travada)

Confirmada. Nao manter fachadas finas. Fiel a arvore-alvo SS3 de `docs/arquitetura-alvo-rag.md` que nao lista rag.py/agent.py.

### 2.1 Realocacao de rag.py: simbolo-por-simbolo

| Simbolo/Componente | Tipo | Modulo-destino | Notas |
|-------------------|------|----------------|-------|
| `_get_rag_embedding_model` | singleton + getter + cache `_rag_embedding_model` | `ingestion/embeddings.py` | Unificar com `criar_embedding_model`. Ja existe `criar_embedding_model` em `ingestion/embeddings.py`; `_get_rag_embedding_model` vira definicao real la com cache global. O notebook `embeddings.py` atual eh apenas factory; adicionar `_rag_embedding_model = None` + `def _get_rag_embedding_model()`. |
| `_executar_main` | CLI entrypoint | `ingestion/cli.py` | Copiar verbatim de rag.py:175-209. |
| `_configurar_logging_cli` | helper CLI | `ingestion/cli.py` | Copiar verbatim de rag.py:160-173. |
| `import fitz` | import de terceiro | `ingestion/extraction.py` | Ja existe `import fitz` em `extraction.py` (verificar). Se sim, manter. Se nao, adicionar. |
| `HuggingFaceEmbeddings` import | import de terceiro | `ingestion/embeddings.py` | Ja importado. |
| `RAGSecurityError` no CLI (rag.py:199) | except clause | `ingestion/cli.py` | Import operacional `from agenticlog.shared.errors import RAGSecurityError` (ja existe). |
| Wrappers `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `ingerir_incrementalmente`, `cria_vectordb`, `reconstruir_vectordb` | wrappers finos | **DELETADOS** | Consumers (app.py, scripts) importam DIRETAMENTE de `ingestion.orchestrator`. Wrappers resolviam seams no call time — consumers agora chamam orquestradores com argumentos diretos (docs_dir, vectordb_dir, embedding_model) que ja usam via config. Se consumer precisar de seam binding, faz inline com `from agenticlog.config import DIR_DOCUMENTS, DIR_VECTORDB`. |
| `vectordb` global | module-level var | **DELETADO** | Nao usado por nenhum consumer apos Fase 3b. |
| `logger` | module-level Logger | **DELETADO** | Nao necessario — cli.py tem seu proprio logger. |
| Re-export shims (Fase 3a/3b) | ~20 linhas de import | **DELETADOS** | Shims de ingestion/*, shared/*. |

### 2.2 Realocacao de agent.py: simbolo-por-simbolo

| Simbolo/Componente | Tipo | Modulo-destino | Notas |
|-------------------|------|----------------|-------|
| `_vector_dbs` | dict cache (singleton) | `retrieval/retriever.py` | Variavel de modulo `_vector_dbs: dict[str, Chroma] = {}`. Ja existe `_vector_dbs` em retriever.py? Verificar — se nao, criar. |
| `_embedding_model` | singleton | `retrieval/retriever.py` | Variavel de modulo `_embedding_model = None`. |
| `_get_embedding_model` | getter + cache | `retrieval/retriever.py` | Copiar logica: `global _embedding_model; if _embedding_model is None: _embedding_model = _build_embedding_model(); return _embedding_model`. |
| `_listar_colecoes` | funcao | `retrieval/retriever.py` | Copiar logica (delega a `_retr._listar_colecoes(vectordb_dir=DIR_VECTORDB)` — agora e a definicao real, nao wrapper). Parametrizar com `vectordb_dir: Path = DIR_VECTORDB`. |
| `_get_vector_db` | funcao | `retrieval/retriever.py` | Copiar logica. Parametrizar com `vectordb_dir: Path = DIR_VECTORDB`. |
| `_llm` | singleton | `retrieval/generation.py` | Ja existe `_llm` em generation.py (verificar). Se existir, manter. |
| `_get_llm` | getter + cache | `retrieval/generation.py` | Ja existe (verificar). |
| `search` | DuckDuckGoSearchAPIWrapper singleton | `retrieval/graph.py` | `search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)`. `usar_ferramenta_web` ja usa `search` como global — mover a definicao para graph.py onde a funcao que consome esta. |
| `import torch; torch.classes.__path__ = []; os.environ["TOKENIZERS_PARALLELISM"] = "false"` | side-effect startup | `retrieval/__init__.py` | Primeira carga do pacote `retrieval/` executa estes side-effects. Colocar em `retrieval/__init__.py` antes de outros imports. |
| `warnings.filterwarnings("ignore")` | side-effect | `retrieval/__init__.py` | Colocar junto com os side-effects de startup. |
| `LLMClient`, `_llm_retry`, `_invoke_chain`, `_prompt_web`, `prompt_gerar`, `prompt_rag_retrieve`, `avalia_similaridade`, `gera_multiplas_respostas`, `rank_respostas` | shims | JA EM `retrieval/generation.py` — nao realocar, so deletar shim. |
| `agent_workflow`, `inicializar_recursos`, `passo_decisao_agente`, `retrieve_info`, `usar_ferramenta_web` | shims | JA EM `retrieval/graph.py` — nao realocar. |
| `AgentState` | shim | JA EM `retrieval/state.py` — nao realocar. |
| `_build_embedding_model`, `_get_retriever`, `invalidar_vector_db` | shims | JA EM `retrieval/retriever.py` — nao realocar. Mas `invalidar_vector_db` PRECISA mudar implementacao (ver LM1). |

### 2.3 invalidar_vector_db — implementacao pos-remocao (SHIMS-10, LM1)

**Estado atual** (`retrieval/retriever.py:152-160`):
```python
def invalidar_vector_db() -> None:
    import agenticlog.agent as _agent_mod
    _agent_mod._vector_dbs.clear()
```

**Estado pos-remocao:**
```python
_vector_dbs: dict[str, Chroma] = {}  # movido de agent.py

def invalidar_vector_db() -> None:
    _vector_dbs.clear()
```

Simples: `_vector_dbs` agora e definido em `retrieval/retriever.py`, entao `invalidar_vector_db` limpa o dict local. O ciclo retriever->agent some. **LM1 resolvida.**

### 2.4 search — realocacao para graph.py

`search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)` e usado exclusivamente por `usar_ferramenta_web()` em `retrieval/graph.py`. A funcao acessa `search` como global. Mover a definicao para o topo de `graph.py`:

```python
# retrieval/graph.py — topo, apos imports
search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)
```

Nenhum consumer importa `search` diretamente — so e usado internamente por `usar_ferramenta_web`.

### 2.5 Side-effects de startup — realocacao para retrieval/__init__.py

```python
# retrieval/__init__.py
import warnings
warnings.filterwarnings("ignore")

import torch
torch.classes.__path__ = []
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

Nota: `os.environ[...]` e seguro em multiplos processos (seta no processo atual). Nao ha race.

---

## 3. Q2 — Decisao: manter lazy `__getattr__` em serving/__init__.py com HistoryStore adicionado

### Contexto

Atualmente `serving/__init__.py` usa `__getattr__` lazy (PEP 562) em vez de imports de topo para evitar import circular com `agenticlog.health` (shim). O shim `health.py` importa de `serving.health`, que dispara `serving/__init__.py`, que por sua vez importaria de `health` (shim) se usasse imports de topo — formando um ciclo.

**Apos a remocao do shim `health.py`, o ciclo some** — `serving/health.py` importa so `config` e stdlib, nunca `health` (shim). Portanto `serving/__init__.py` poderia usar imports diretos.

### Opcao A — Manter lazy + adicionar HistoryStore (RECOMENDADA)

Manter `__getattr__` e `_LAZY_MAP` como estao. Adicionar `HistoryStore` ao `_LAZY_MAP` e `__all__`.

**Pros:**
- Zero risco de regressao: o lazy `__getattr__` ja funciona e e testado (T6 Fase 5).
- O ciclo com health.py some, mas o lazy ainda protege contra futuros ciclos.
- Minimo diff em `serving/__init__.py` (so adicionar 1 entrada + mover import de `from importlib import import_module` para topo).
- Preserva acicidade fresh-interpreter ja testada.

**Contras:**
- `__getattr__` lazy e indirecao desnecessaria apos a remocao do shim.
- `HistoryStore` adicionado manualmente ao `_LAZY_MAP`.

### Opcao B — Simplificar para imports diretos

Substituir `__getattr__` + `_LAZY_MAP` por `from agenticlog.serving.api import ...` e `from agenticlog.serving.health import ...`.

**Pros:**
- Mais simples, sem indirecao.
- Typing estatico (IDEs resolvem simbolos).

**Contras:**
- Risco de import circular se `serving/health.py` algum dia importar `serving` (hoje nao importa).
- Diff maior: substituir ~50 linhas de lazy logic por imports.
- Teste `teste_5_serving_app_e_o_mesmo` em `test_rag_serving_fase5.py` precisaria ser atualizado (nao testa mais `serving.app is serving.api.app` porque isso seria import direto).

**Veredito:** Opcao A — manter lazy + adicionar HistoryStore. Risco minimo, diff minimo. O custo de manutencao de `_LAZY_MAP` e baixo (~1 entrada). Se no futuro o projeto quiser migrar para imports diretos, pode fazer em fase separada.

### Implementacao

Adicionar no `_LAZY_MAP`:
```python
# serving/__init__.py
_LAZY_MAP: dict[str, tuple[str, str]] = {
    # ... entradas existentes ...
    # observability
    "HistoryStore": ("agenticlog.observability.history", "HistoryStore"),
}
```

Adicionar em `__all__`:
```python
__all__ = [
    # ... entradas existentes ...
    "HistoryStore",
]
```

---

## 4. Q3 — invalidar_vector_db: ciclo quebra limpo

**Confirmado: SIM.** Apos a realocacao de `_vector_dbs` para `retrieval/retriever.py` e a mudanca de `invalidar_vector_db` para limpar o dict local:

1. `retrieval/retriever.py` NAO importa `agent` (agora deletado).
2. `orchestrator.py:196` lazy importa `from agenticlog.retrieval.retriever import invalidar_vector_db` — NAO de `agent`.
3. O ciclo `retriever -> agent -> retriever` some completamente.
4. `orchestrator` -> `retriever` e uma aresta direta e aciclica.

O unico lugar que chamava `invalidar_vector_db` de `agent` era o proprio shim em `agent.py`, que e deletado.

---

## 5. Q4 — Wrappers agent.py migram para retrieval/retriever.py como definicoes reais

### Estado atual

Em `agent.py` (shim/fachada):
```python
def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Wrapper: le/seta _embedding_model global, delega a _retr._build_embedding_model()."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _retr._build_embedding_model()
    return _embedding_model

def _listar_colecoes() -> list[str]:
    return _retr._listar_colecoes(vectordb_dir=DIR_VECTORDB)

def _get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME) -> Chroma:
    return _retr._get_vector_db(collection_name, vectordb_dir=DIR_VECTORDB)
```

### Estado pos-remocao

Em `retrieval/retriever.py` (definicoes reais):

```python
# Variaveis globais locais
_vector_dbs: dict[str, Chroma] = {}
_embedding_model = None

def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna o singleton do modelo de embeddings."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _build_embedding_model()
    return _embedding_model

def _listar_colecoes(vectordb_dir: Path = DIR_VECTORDB) -> list[str]:
    """Lista colecoes existentes no ChromaDB."""
    # Copiar logica de _retr._listar_colecoes (que era o target do wrapper)
    ...

def _get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME,
                    vectordb_dir: Path = DIR_VECTORDB) -> Chroma:
    """Retorna singleton de ChromaDB para uma colecao."""
    # Copiar logica de _retr._get_vector_db
    ...
```

Estas funcoes JA EXISTEM em `retrieval/retriever.py` como `_listar_colecoes` e `_get_vector_db` (parametrizadas). Os wrappers foram criados em agent.py para injetar `DIR_VECTORDB` no call time. Agora os consumers que usavam os wrappers passam a importar diretamente de `retriever` e chamam `_listar_colecoes(vectordb_dir=...)` ou usam o default de config.

Os ~8 patches `agenticlog.agent._get_embedding_model` migram para `agenticlog.retrieval.retriever._get_embedding_model`.

---

## 6. Mapa de migracao de consumers

| Consumer (arquivo:linha) | Import ATUAL | Import DESTINO |
|------------------------|-------------|----------------|
| `app.py:9` | `from agenticlog.agent import _listar_colecoes` | `from agenticlog.retrieval.retriever import _listar_colecoes` |
| `app.py:16-21` | `from agenticlog.rag import RAGSecurityError, adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente, sanitizar_nome_colecao` | `from agenticlog.shared.errors import RAGSecurityError; from agenticlog.ingestion.orchestrator import adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente; from agenticlog.ingestion.security import sanitizar_nome_colecao` |
| `app.py:27` | `MSG_CONNECT_ERROR = "... uvicorn agenticlog.api:app"` | `MSG_CONNECT_ERROR = "... uvicorn agenticlog.serving.api:app"` (LM2) |
| `main_api.py:12` | `"agenticlog.api:app"` | `"agenticlog.serving.api:app"` (LM2) |
| `src/agenticlog/__init__.py:17` | `from agenticlog.agent import AgentState, agent_workflow` | `from agenticlog.retrieval.graph import agent_workflow; from agenticlog.retrieval.state import AgentState` |
| `src/agenticlog/__init__.py:18` | `from agenticlog.health import ...` | `from agenticlog.serving.health import ...` |
| `src/agenticlog/config.py:166` | `from agenticlog.observability.logging import _JsonFormatter` | **REMOVER A LINHA.** Todo consumer que precisar de `_JsonFormatter` importa de `agenticlog.observability.logging` diretamente. |
| `src/agenticlog/serving/api.py:23` | `from agenticlog.agent import AgentState, agent_workflow, inicializar_recursos` | `from agenticlog.retrieval.graph import AgentState, agent_workflow, inicializar_recursos` |
| `src/agenticlog/serving/api.py:31-36` | `from agenticlog.health import ...` | `from agenticlog.serving.health import ...` (ja importa de serving.health interno — OK se relativo ou absoluto. `from agenticlog.serving.health import ...` e o caminho canonico.) |
| `src/agenticlog/serving/api.py:36` | `from agenticlog.history import HistoryStore` | `from agenticlog.observability.history import HistoryStore` (LM4) |
| `src/agenticlog/serving/api.py:37` | `from agenticlog.rag import _get_rag_embedding_model` | `from agenticlog.ingestion.embeddings import _get_rag_embedding_model` (LM4) |
| `scripts/rag_eval.py:84-89` | `from agenticlog.agent import AgentState, _get_retriever, agent_workflow; from agenticlog.rag import _get_rag_embedding_model` | `from agenticlog.retrieval.state import AgentState; from agenticlog.retrieval.retriever import _get_retriever; from agenticlog.retrieval.graph import agent_workflow; from agenticlog.ingestion.embeddings import _get_rag_embedding_model` |
| `scripts/pdf_to_json.py:17` | `from agenticlog.rag import RAGSecurityError, extrair_texto_pdf` | `from agenticlog.shared.errors import RAGSecurityError; from agenticlog.ingestion.extraction import extrair_texto_pdf` |

---

## 7. Mapa de migracao de patches de teste (por arquivo)

### `tests/test_rag.py` (~15 patches `agenticlog.rag.X`)
| Patch ATUAL | Patch DESTINO | Ocorrencias |
|-------------|--------------|-------------|
| `agenticlog.rag.adicionar_documento_incrementalmente` | `agenticlog.ingestion.orchestrator.adicionar_documento_incrementalmente` | varias |
| `agenticlog.rag.adicionar_pdf_incrementalmente` | `agenticlog.ingestion.orchestrator.adicionar_pdf_incrementalmente` | varias |
| `agenticlog.rag._get_rag_embedding_model` | `agenticlog.ingestion.embeddings._get_rag_embedding_model` | varias |
| `agenticlog.rag.ingerir_incrementalmente` | `agenticlog.ingestion.orchestrator.ingerir_incrementalmente` | algumas |
| `agenticlog.rag.cria_vectordb` | `agenticlog.ingestion.orchestrator.cria_vectordb` | algumas |
| `agenticlog.rag._resetar_colecao` | `agenticlog.ingestion.store._resetar_colecao` | ~8 (ja mapeado Fase 3b) |
| `agenticlog.rag._valida_path_documentos` | `agenticlog.ingestion.security._valida_path_documentos` | algumas |
| `agenticlog.rag.extrair_texto_pdf` | `agenticlog.ingestion.extraction.extrair_texto_pdf` | algumas |
| `agenticlog.rag.RAGSecurityError` | `agenticlog.shared.errors.RAGSecurityError` | algumas |
| `agenticlog.rag.invalidar_vector_db` | `agenticlog.retrieval.retriever.invalidar_vector_db` | ~2 (linhas 830, 869, 920, etc. — ver LM1) |

### `tests/test_agentic_rag.py` (~10 patches `agenticlog.agent.X`)
| Patch ATUAL | Patch DESTINO | Ocorrencias |
|-------------|--------------|-------------|
| `agenticlog.agent.retriever` | `agenticlog.retrieval.retriever._get_retriever` | varias |
| `agenticlog.agent.agent_workflow` | `agenticlog.retrieval.graph.agent_workflow` | varias |
| `agenticlog.agent.gera_multiplas_respostas` | `agenticlog.retrieval.generation.gera_multiplas_respostas` | varias |
| `agenticlog.agent.avalia_similaridade` | `agenticlog.retrieval.generation.avalia_similaridade` | algumas |
| `agenticlog.agent.rank_respostas` | `agenticlog.retrieval.generation.rank_respostas` | algumas |
| `agenticlog.agent._get_llm` | `agenticlog.retrieval.generation._get_llm` | algumas |
| `agenticlog.agent._llm` | `agenticlog.retrieval.generation._llm` | algumas |
| `agenticlog.agent.search` | `agenticlog.retrieval.graph.search` | algumas |
| `agenticlog.agent._get_embedding_model` | `agenticlog.retrieval.retriever._get_embedding_model` | algumas |
| `agenticlog.agent._embedding_model` | `agenticlog.retrieval.retriever._embedding_model` | algumas |
| `agenticlog.agent.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` | algumas |
| `agenticlog.agent.invalidar_vector_db` | `agenticlog.retrieval.retriever.invalidar_vector_db` | ~8 (test_rag.py) |

### `tests/test_rag_caracterizacao.py` (8 monkeypatches)
| monkeypatch ATUAL | monkeypatch DESTINO |
|-------------------|---------------------|
| `agenticlog.rag.DIR_DOCUMENTS` | `agenticlog.config.DIR_DOCUMENTS` |
| `agenticlog.rag.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` |
| `agenticlog.agent.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` (se ainda existir — mesmo alvo) |
| `agenticlog.rag._rag_embedding_model` | `agenticlog.ingestion.embeddings._rag_embedding_model` |
| `agenticlog.agent._embedding_model` | `agenticlog.retrieval.retriever._embedding_model` |
| `agenticlog.agent._llm` | `agenticlog.retrieval.generation._llm` |
| `agenticlog.agent._vector_dbs` | `agenticlog.retrieval.retriever._vector_dbs` |
| `agenticlog.agent.search` | `agenticlog.retrieval.graph.search` |

### `tests/adicionar_pdf_incrementalmente.py` (~3 patches)
| Patch ATUAL | Patch DESTINO |
|-------------|--------------|
| `agenticlog.rag.invalidar_vector_db` | `agenticlog.retrieval.retriever.invalidar_vector_db` |
| `agenticlog.rag.adicionar_pdf_incrementalmente` | `agenticlog.ingestion.orchestrator.adicionar_pdf_incrementalmente` |
| `agenticlog.rag._get_rag_embedding_model` | `agenticlog.ingestion.embeddings._get_rag_embedding_model` |

### `tests/acceptance/test_adicionar_pdf_incrementalmente.py` (~2 patches)
| Patch ATUAL | Patch DESTINO |
|-------------|--------------|
| `agenticlog.rag.invalidar_vector_db` (ln 99, create=True) | `agenticlog.retrieval.retriever.invalidar_vector_db` (create=True) |
| `agenticlog.rag.invalidar_vector_db` (ln 413, 466) | `agenticlog.retrieval.retriever.invalidar_vector_db` |

---

## 8. Testes de identidade a REMOVER

| Arquivo | Classe/Teste | Motivo |
|---------|-------------|--------|
| `tests/ingestion/test_shims_identidade.py` | `TestShimsIdentidade` (2 testes) | rag.py deletado — nao ha mais shim para comparar. |
| `tests/acceptance/test_rag_serving_fase5.py` | `TestServingShimsIdentidade` (7 testes) | api.py e health.py deletados — nao ha mais shim para comparar. |
| `tests/acceptance/test_rag_retrieval_fase4.py` | `TestShimIdentity` (1 parametrizado) | agent.py deletado — nao ha mais shim para comparar. |
| `tests/acceptance/test_rag_retrieval_fase4.py` | `test_ac3_identidade_is_shim_de_store` (se existir) | Idem. |
| `tests/ingestion/test_shims_identidade.py` | `teste_1_identidade_de_cada_simbolo_movido`, `teste_2_caminho_de_import_antigo_resolve` | Ja incluido acima em TestShimsIdentidade. |

### Testes de acicidade a MANTER

| Arquivo | Classe | Acao |
|---------|--------|------|
| `tests/ingestion/test_shims_identidade.py` | `TestIngestionAcyclic` | MANTER (so testa `agenticlog.ingestion`, nao depende de shims). |
| `tests/acceptance/test_rag_serving_fase5.py` | `TestServingAcyclic` | MANTER (so testa `agenticlog.serving`, nao depende de shims). |
| `tests/acceptance/test_rag_retrieval_fase4.py` | `TestAcyclicImport` | ATUALIZAR: remover `"agenticlog.agent"` da lista `MODULES`. |

### Testes de acicidade a ATUALIZAR

`tests/acceptance/test_rag_retrieval_fase4.py::TestAcyclicImport.MODULES`:
```python
# ATUAL
MODULES = [
    "agenticlog.retrieval.state",
    "agenticlog.retrieval.generation",
    "agenticlog.retrieval.retriever",
    "agenticlog.retrieval.graph",
    "agenticlog.agent",  # <-- REMOVER
]

# NOVO
MODULES = [
    "agenticlog.retrieval.state",
    "agenticlog.retrieval.generation",
    "agenticlog.retrieval.retriever",
    "agenticlog.retrieval.graph",
]
```

E `test_ac02_import_todos_frio` precisa remover `import agenticlog.agent` da string:
```python
# ATUAL
code = (
    "import agenticlog.retrieval.state; "
    "import agenticlog.retrieval.generation; "
    "import agenticlog.retrieval.retriever; "
    "import agenticlog.retrieval.graph; "
    "import agenticlog.agent; "  # <-- REMOVER
    "print('OK')"
)
```

---

## 9. Teste de acicidade do pacote raiz

Apos a remocao, `import agenticlog` deve funcionar em interpretador frio. Adicionar teste:
- `python -c "import agenticlog"` exit 0, sem "circular" ou "partially initialized".

Isto garante que `__init__.py` nao criou ciclo apos os imports atualizados.

---

## 10. pyproject.toml — mypy override

Linha atual:
```toml
module = ["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]
```

Nova linha:
```toml
module = ["agenticlog.serving.api"]
```

`agenticlog.rag` e `agenticlog.api` removidos (shims deletados).

---

## 11. CLI entrypoint — pyproject.toml / setup

Atualmente `python -m agenticlog.rag` funciona via `__init__.py` + `if __name__ == "__main__"`. Com rag.py deletado, o entrypoint muda para `python -m agenticlog.ingestion`.

Verificar `pyproject.toml` se ha `[project.scripts]` ou `[project.entry-points]` que referencie `agenticlog.rag:`. Se houver, atualizar para `agenticlog.ingestion.cli:_executar_main` ou similar. Se nao houver, `python -m agenticlog.ingestion` ja funciona porque `ingestion/__init__.py` importa `cli.py` (ou `cli.py` tem `if __name__ == "__main__"`).

---

## 12. Mapa de verificacao de import circulares pos-remocao

```
retrieval/__init__.py (side-effects)
  -> retrieval/state.py (AgentState)  [config, stdlib]
  -> retrieval/retriever.py (_vector_dbs, _get_embedding_model, invalidar_vector_db)
     -> config (DIR_VECTORDB, DEFAULT_COLLECTION_NAME, EMBEDDING_MODEL)
     -> langchain_chroma
  -> retrieval/generation.py (_llm, _get_llm, prompts, rank)
     -> config, langchain_openai, stdlib
  -> retrieval/graph.py (search, agent_workflow, nos)
     -> retrieval.state, retrieval.generation, retrieval.retriever, config
     -> duckduckgo_search

ingestion/cli.py (_executar_main, _configurar_logging_cli)
  -> ingestion/orchestrator (cria_vectordb, ingerir_incrementalmente)
     -> ingestion/store, ingestion/extraction, ingestion/cleaning, ingestion/chunking,
        ingestion/metadata, ingestion/embeddings, config, shared.errors
     -> retrieval/retriever (lazy import: invalidar_vector_db)
  -> shared/errors (RAGSecurityError)

serving/api.py
  -> retrieval/graph (AgentState, agent_workflow, inicializar_recursos)
  -> serving/health (check_lmstudio_health)
  -> observability/history (HistoryStore)
  -> config (DIR_VECTORDB, HISTORY_FILE, etc.)
  -> ingestion/embeddings (_get_rag_embedding_model)
  -> fastapi, pydantic, httpx, stdlib

serving/__init__.py (lazy __getattr__)
  -> serving/api (lazy) NUNCA importa serving diretamente
  -> serving/health (lazy) NUNCA importa serving diretamente
  -> observability/history (lazy) NUNCA importa serving

NENHUM ciclo. O grafo e DAG. `retrieval/` nunca importa `serving/` ou `ingestion/`.
`ingestion/` nunca importa `serving/`. `serving/` e folha que importa de todos.
```

---

## 13. Mitigacoes dos itens de risco (spec RSrisks)

| Risco | Mitigacao de design |
|-------|---------------------|
| Ordem de operacao (CRITICO) | Tasks organizadas em 5 fases: (A) realocar singletons, (B) migrar consumers+scripts, (C) migrar patches+oraculo, (D) remover identity tests+acicidade, (E) deletar shims. Fase E e a ULTIMA. |
| invalidar_vector_db ciclo | `_vector_dbs` movido para `retrieval/retriever.py`. `invalidar_vector_db` limpa dict local. NAO importa agent.py. |
| Oracle de caracterizacao quebra | Monkeypatches REESCRITOS (mudam namespace) mas LOGICA preservada. DIFF nao e zero, mas `pytest` verde e gate. |
| ~50 patches de teste | Inventario completo por arquivo e por-simbolo (SS7 deste design). Tasks sequenciais por arquivo de teste. |
| `__init__.py` exporta de shims | Migrado para `retrieval.graph`, `serving.health`, `retrieval.state`. Testado por `import agenticlog` em interpretador frio. |
| scripts/rag_eval.py imports condicionais | Migrar imports dentro do bloco try/except. |
| `_get_rag_embedding_model` em serving/api.py | Importar de `ingestion.embeddings` (LM4). |
| mypy override | Remover `agenticlog.rag`, `agenticlog.api`. Manter `agenticlog.serving.api`. |
| CLI entrypoint | `python -m agenticlog.ingestion` funciona. `python -m agenticlog.rag` quebra (esperado). |
| hnswlib SAC Windows | CI Linux e gate autoritativo. |
| CLAUDE.md desatualizado | Atualizar comandos: `python -m agenticlog.ingestion` em vez de `python -m agenticlog.rag`. Remover secoes sobre rag.py e agent.py. |

---

## 14. Verificacao pos-remocao

| Verificacao | Comando / Metodo | Gate |
|------------|------------------|------|
| Nenhum shim remanescente | `rg "# Re-export shim \(ADR-018" src/` retorna 0 matches | full |
| Acicidade pacote raiz | `python -c "import agenticlog"` exit 0, sem "circular" | unit |
| Acicidade ingestion | `python -c "import agenticlog.ingestion"` exit 0 | unit |
| Acicidade retrieval | `python -c "import agenticlog.retrieval"` exit 0 | unit |
| Acicidade serving | `python -c "import agenticlog.serving"` exit 0 | unit |
| CLI entrypoint | `python -m agenticlog.ingestion --help` exit 0 | unit |
| Suite completa | `pytest --cov=agenticlog -v` verde no CI Linux | build |
| Lint | `ruff check --diff .` limpo | build |
| Mypy | `mypy src/agenticlog/` limpo | build |
| Nenhum import de shim | `rg "from agenticlog\.(rag\|agent\|api\|health\|history) import" src/` 0 matches | full |
| Consumidores atualizados | Verificar app.py, main_api.py, __init__.py, serving/api.py, scripts/ | full |
