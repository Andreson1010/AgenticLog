# ADR-018 Fase 6 — Remocao de shims de compatibilidade + reescrita de testes — Technical Spec

**Path:** `.specs/features/rag-fase6-shims-remocao/spec.md`
**TLC scope:** complex
**Based on story:** Remover 6 shims de compatibilidade (rag.py, agent.py, api.py, health.py, history.py, config.py:166), migrar ~9 consumers para modulos canonicos, migrar ~50 patches de teste, remover 5 testes de identidade, atualizar acicidade, resolver 4 landmines, oraculos verdes.
**Status:** Awaiting human approval

---

## Problem Statement

As fases 2 a 5 do ADR-018 extrairam modulos canonicos (ingestion/, retrieval/, observability/, serving/, shared/) preservando shims de compatibilidade identity-preserving nos modulos originais (rag.py, agent.py, api.py, health.py, history.py, config.py:166). Estes shims foram marcados com `# Re-export shim (ADR-018 Fase N) -- remover na Fase 6`. A Fase 6 deleta todos os 6 shims, migra cada consumer para o modulo canonico, migra todos os patches de teste para os novos namespaces, remove os testes de identidade shim que perderam o objeto, e corrige 4 landmines documentadas. E a ultima fase do redesign ADR-018.

## Goals

- [ ] Deletar 6 shims: `rag.py`, `agent.py`, `api.py`, `health.py`, `history.py`, e a linha 166 de `config.py`.
- [ ] Todos os ~9 consumers (app.py, main_api.py, __init__.py, serving/api.py, scripts/) importam de modulos canonicos.
- [ ] ~50 patches de teste migrados para namespaces canonicos; 5 testes de identidade removidos; 2 acicidade MODULES atualizados.
- [ ] 4 landmines resolvidos (orchestrator.py lazy import, main_api.py/app.py strings, RAGSecurityError no CLI, HistoryStore bypass).
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde no CI Linux; cobertura >= 80%; ruff/mypy limpos.
- [ ] Oraculo de caracterizacao (`tests/test_rag_caracterizacao.py`) com monkeypatches reescritos mas comportamento inalterado.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Alteracao de comportamento de ingestion/, retrieval/, observability/, serving/, shared/ | Refatoracao pura de remocao de shim — nenhuma logica de negocio muda. |
| Criacao de novos modulos ou funcoes | So remocao de shims e realocacao de singletons/globals/CLI/side-effects. |
| Reescrita de oraculo de caracterizacao | Monkeypatches REESCRITOS (mudam namespace), mas comportamento preservado. |
| Mudanca de embedding/chunking/config | Nao mexe em config.py (exceto linha 166), nem em parametros de chunking ou embedding. |
| CI nao autoritativo local (SAC/hnswlib) | Windows local pode pular ~15 testes Chroma; CI Linux e gate autoritativo. |

---

## User Stories

### P1: Remover 6 shims de compatibilidade do ADR-018 ⭐ MVP

**User Story**: Como mantenedor do AgenticLog, quero deletar os 6 shims de compatibilidade (rag.py, agent.py, api.py, health.py, history.py, config.py:166) que re-exportam simbolos dos modulos canonicos, migrar todos os consumers e patches para os modulos canonicos, e remover os testes de identidade shim, para completar o redesign ADR-018 e eliminar a divida tecnica dos shins.

**Why P1**: E o objetivo unico da fase. Sem a remocao, os shims permanecem como divida tecnica e risco de manutencao.

**Acceptance Criteria**:

1. WHEN `src/agenticlog/rag.py` e deletado THEN `import agenticlog.rag` SHALL falhar com `ModuleNotFoundError`, e `from agenticlog.rag import X` SHALL falhar para QUALQUER X. *(SHIMS-01)*

2. WHEN `src/agenticlog/agent.py` e deletado THEN `import agenticlog.agent` SHALL falhar com `ModuleNotFoundError`. *(SHIMS-02)*

3. WHEN `src/agenticlog/api.py` e deletado THEN `import agenticlog.api` SHALL falhar com `ModuleNotFoundError`. *(SHIMS-03)*

4. WHEN `src/agenticlog/health.py` e deletado THEN `from agenticlog.health import check_lmstudio_health` SHALL falhar com `ModuleNotFoundError`. *(SHIMS-04)*

5. WHEN `src/agenticlog/history.py` e deletado THEN `from agenticlog.history import HistoryStore` SHALL falhar com `ModuleNotFoundError`. *(SHIMS-05)*

6. WHEN `src/agenticlog/config.py` linha 166 (`from agenticlog.observability.logging import _JsonFormatter`) e removida THEN `from agenticlog.config import _JsonFormatter` SHALL falhar com `ImportError`, e `from agenticlog.observability.logging import _JsonFormatter` SHALL continuar funcionando. *(SHIMS-06)*

7. WHEN todos os shims sao deletados THEN os seguintes consumers SHALL importar de modulos canonicos sem `ModuleNotFoundError`:
   - `app.py`: `from agenticlog.retrieval.retriever import _listar_colecoes` e `from agenticlog.ingestion.orchestrator import adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente` e `from agenticlog.ingestion.security import sanitizar_nome_colecao` e `from agenticlog.shared.errors import RAGSecurityError`
   - `main_api.py`: string `"agenticlog.serving.api:app"`
   - `app.py:27`: string `"uvicorn agenticlog.serving.api:app"`
   - `src/agenticlog/__init__.py`: imports de `agenticlog.retrieval.graph`, `agenticlog.serving.health`, `agenticlog.retrieval.state`
   - `serving/api.py`: `from agenticlog.observability.history import HistoryStore` e `from agenticlog.ingestion.embeddings import _get_rag_embedding_model`
   - `scripts/rag_eval.py`: `from agenticlog.retrieval.graph import agent_workflow, inicializar_recursos` (ou equivalente, ver design)
   - `scripts/pdf_to_json.py`: `from agenticlog.ingestion.extraction import extrair_texto_pdf` e `from agenticlog.shared.errors import RAGSecurityError`
   *(SHIMS-07)*

8. WHEN os singletons/globals/CLI/side-effects de `rag.py` sao realocados THEN:
   - `_get_rag_embedding_model` (singleton + cache) SHALL estar em `ingestion/embeddings.py` (unificado com `criar_embedding_model`)
   - `_executar_main` + `_configurar_logging_cli` SHALL estar em `ingestion/cli.py`
   - `import fitz` SHALL estar em `ingestion/extraction.py`
   *(SHIMS-08)*

9. WHEN os singletons/globals/CLI/side-effects/startup de `agent.py` sao realocados THEN:
   - `_vector_dbs` (dict cache) SHALL estar em `retrieval/retriever.py`
   - `search` (DuckDuckGoSearchAPIWrapper singleton) SHALL estar em `retrieval/graph.py` (ou retriever.py, ver design)
   - `import torch; torch.classes.__path__ = []; os.environ["TOKENIZERS_PARALLELISM"] = "false"` SHALL estar em `retrieval/__init__.py` (ou modulo de primeira carga)
   - `_llm` (singleton) + `_get_llm` SHALL estar em `retrieval/generation.py`
   - `_embedding_model` (singleton) + `_get_embedding_model` + `_build_embedding_model` SHALL estar em `retrieval/retriever.py`
   - `_listar_colecoes` e `_get_vector_db` SHALL estar em `retrieval/retriever.py`
   *(SHIMS-09)*

10. WHEN `invalidar_vector_db` e migrado de `agent.py` para `retrieval/retriever.py` THEN `invalidar_vector_db` SHALL limpar `_vector_dbs` LOCAL em `retrieval/retriever.py` (NAO importar agent.py). E `orchestrator.py:196-199` SHALL lazy-import `from agenticlog.retrieval.retriever import invalidar_vector_db` (NAO de agent.py). *(SHIMS-10)*

11. WHEN a linha 166 de `config.py` e removida THEN `from agenticlog.observability.logging import _JsonFormatter` SHALL ser usado em todos os locais que precisam de `_JsonFormatter`. *(SHIMS-11)*

12. WHEN `serving/__init__.py` e atualizado (Q2 resolvida) THEN:
   - `HistoryStore` SHALL estar em `_LAZY_MAP` e `__all__` se `__getattr__` lazy for mantido
   - OU `__getattr__` lazy e removido e imports sao diretos (ver design para decisao)
   *(SHIMS-12)*

13. WHEN todos os ~50 patches de teste sao migrados THEN `pytest --cov=agenticlog --cov-report=term-missing -v` SHALL passar verde. *(SHIMS-13)*

14. WHEN os 5 testes de identidade sao removidos THEN:
   - `tests/ingestion/test_shims_identidade.py::TestShimsIdentidade` SHALL ser removido
   - `tests/acceptance/test_rag_serving_fase5.py::TestServingShimsIdentidade` SHALL ser removido
   - `tests/acceptance/test_rag_retrieval_fase4.py::TestShimIdentity` SHALL ser removido
   - `tests/acceptance/test_rag_retrieval_fase4.py::test_ac3_identidade_is_shim_de_store` (se existir) SHALL ser removido
   - `tests/ingestion/test_shims_identidade.py` SHALL ter `teste_1_identidade_de_cada_simbolo_movido` e `teste_2_caminho_de_import_antigo_resolve` removidos
   *(SHIMS-14)*

15. WHEN as listas MODULES de acicidade sao atualizadas THEN `"agenticlog.agent"` e `"agenticlog.rag"` SHALL ser removidos das listas, e o pacote `"agenticlog"` como um todo SHALL ser importavel em interpretador frio. *(SHIMS-15)*

16. WHEN o oraculo `tests/test_rag_caracterizacao.py` tem seus monkeypatches reescritos THEN:
   - `monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", ...)` SHALL virar `monkeypatch.setattr("agenticlog.config.DIR_DOCUMENTS", ...)`
   - `monkeypatch.setattr("agenticlog.rag.DIR_VECTORDB", ...)` SHALL virar `monkeypatch.setattr("agenticlog.config.DIR_VECTORDB", ...)`
   - `monkeypatch.setattr("agenticlog.agent.DIR_VECTORDB", ...)` SHALL virar `monkeypatch.setattr("agenticlog.config.DIR_VECTORDB", ...)`
   - `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", ...)` SHALL virar `monkeypatch.setattr("agenticlog.ingestion.embeddings._rag_embedding_model", ...)`
   - `monkeypatch.setattr("agenticlog.agent._embedding_model", ...)` SHALL virar `monkeypatch.setattr("agenticlog.retrieval.retriever._embedding_model", ...)`
   - `monkeypatch.setattr("agenticlog.agent._llm", ...)` SHALL virar `monkeypatch.setattr("agenticlog.retrieval.generation._llm", ...)`
   - `monkeypatch.setattr("agenticlog.agent._vector_dbs", ...)` SHALL virar `monkeypatch.setattr("agenticlog.retrieval.retriever._vector_dbs", ...)`
   - `monkeypatch.setattr("agenticlog.agent.search", ...)` SHALL virar `monkeypatch.setattr("agenticlog.retrieval.graph.search", ...)` (ou onde search ficar, ver design)
   - A LOGICA do oraculo (asserts, fixtures) SHALL permanecer INALTERADA.
   *(SHIMS-16)*

17. WHEN as 4 landmines sao resolvidas (SHIMS-10 ja cobre LM1):
   - **LM1 (orchestrator.py lazy import):** SHIMS-10 cobre.
   - **LM2 (main_api.py + app.py strings):** `"agenticlog.api:app"` SHALL virar `"agenticlog.serving.api:app"`.
   - **LM3 (RAGSecurityError no CLI):** `rag.py` deletado, mas o import de `RAGSecurityError` em `ingestion/cli.py` SHALL usar `from agenticlog.shared.errors import RAGSecurityError`.
   - **LM4 (HistoryStore bypass):** `serving/api.py` SHALL importar `HistoryStore` de `agenticlog.observability.history` (nao de `agenticlog.history`).
   *(SHIMS-17)*

18. WHEN a suite completa roda THEN:
   - `pytest --cov=agenticlog --cov-report=term-missing -v` SHALL passar no CI Linux com 100% dos testes (ajustado para remocoes).
   - `ruff check --diff .` SHALL ser limpo.
   - `mypy src/agenticlog/` SHALL ser limpo.
   - `python -m agenticlog.ingestion` SHALL funcionar como entrypoint (substitui `python -m agenticlog.rag`).
   *(SHIMS-18)*

**Independent Test**: `pytest --cov=agenticlog -v` verde no CI Linux; `ruff check .` limpo; `mypy src/agenticlog/` limpo; `python -c "import agenticlog"` exit 0 sem shims.

---

## Edge Cases

- WHEN `python -m agenticlog.rag` e executado APOS a remocao THEN SHALL falhar com `ModuleNotFoundError` — o entrypoint foi movido para `python -m agenticlog.ingestion`.
- WHEN um teste usa `@patch("agenticlog.rag.X")` e o shim rag.py foi deletado THEN o patch SHALL falhar com `ModuleNotFoundError` — todos os patches DEVEM ser migrados para `agenticlog.ingestion.*` ou `agenticlog.shared.*`.
- WHEN um teste usa `@patch("agenticlog.agent.X")` e o shim agent.py foi deletado THEN o patch SHALL falhar — todos os patches DEVEM ser migrados para `agenticlog.retrieval.*`.
- WHEN um teste usa `@patch("agenticlog.api.X")` e o shim api.py foi deletado THEN o patch SHALL falhar — todos os patches DEVEM ser migrados para `agenticlog.serving.api.*`.
- WHEN um teste usa `@patch("agenticlog.health.httpx.Client")` e o shim health.py foi deletado THEN o patch SHALL falhar — o import `import httpx` nao existe mais.
- WHEN um teste usa `@patch("agenticlog.history.HistoryStore")` e o shim history.py foi deletado THEN o patch SHALL falhar — migrar para `agenticlog.observability.history.HistoryStore`.
- WHEN `from agenticlog.config import _JsonFormatter` e usado e a linha 166 foi removida THEN o import SHALL falhar com `ImportError` — migrar para `from agenticlog.observability.logging import _JsonFormatter`.
- WHEN `invalidar_vector_db` e chamado de `orchestrator.py` e o lazy import foi atualizado para `agenticlog.retrieval.retriever` THEN SHALL limpar `_vector_dbs` em `retrieval.retriever` — o ciclo retriever->agent some.
- WHEN o oraculo de caracterizacao roda com monkeypatches reescritos THEN os asserts SHALL ser identicos — a logica nao muda, so o namespace do patch.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| SHIMS-01 | P1 (delete rag.py) | Design | Pending |
| SHIMS-02 | P1 (delete agent.py) | Design | Pending |
| SHIMS-03 | P1 (delete api.py) | Design | Pending |
| SHIMS-04 | P1 (delete health.py) | Design | Pending |
| SHIMS-05 | P1 (delete history.py) | Design | Pending |
| SHIMS-06 | P1 (delete config.py:166) | Design | Pending |
| SHIMS-07 | P1 (migrate consumers) | Design | Pending |
| SHIMS-08 | P1 (realocate rag.py singletons/CLI) | Design | Pending |
| SHIMS-09 | P1 (realocate agent.py singletons/globals) | Design | Pending |
| SHIMS-10 | P1 (invalidar_vector_db local) | Design | Pending |
| SHIMS-11 | P1 (config.py:166 removal) | Design | Pending |
| SHIMS-12 | P1 (serving/__init__.py Q2) | Design | Pending |
| SHIMS-13 | P1 (migrate test patches) | Design | Pending |
| SHIMS-14 | P1 (remove identity tests) | Design | Pending |
| SHIMS-15 | P1 (update acicidade MODULES) | Design | Pending |
| SHIMS-16 | P1 (rewrite oracle monkeypatches) | Design | Pending |
| SHIMS-17 | P1 (resolve 4 landmines) | Design | Pending |
| SHIMS-18 | P1 (suite completa verde) | Design | Pending |

**ID format:** `SHIMS-[NUMBER]` (SHIMS = SHIM removal).

**Coverage:** 18 IDs totais; todos mapeados a tasks (ver `tasks.md`).

---

## Data Model Changes

Nenhuma. Nenhum schema Pydantic, modelo de banco, ou estrutura de dados e alterado. `AgentState`, `QueryRequest`, `QueryResponse`, `HistoryStore` permanecem nos mesmos modulos canonicos.

---

## Process / Background Flow

### Happy path — CLI ingestion (substitui `python -m agenticlog.rag`)
`python -m agenticlog.ingestion` -> `ingestion/cli.py:_executar_main` -> `_configurar_logging_cli` -> `orchestrator.ingerir_incrementalmente()` ou `orchestrator.cria_vectordb()`. Com `RAGSecurityError` capturado via `from agenticlog.shared.errors import RAGSecurityError`.

### Happy path — app.py (Streamlit)
`app.py` -> `from agenticlog.retrieval.retriever import _listar_colecoes` -> `from agenticlog.ingestion.orchestrator import adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente` -> `from agenticlog.ingestion.security import sanitizar_nome_colecao` -> `from agenticlog.shared.errors import RAGSecurityError`.

### Happy path — FastAPI startup
`main_api.py` -> `uvicorn.run("agenticlog.serving.api:app")` -> `serving/api.py` (importa de `retrieval.graph`, `serving.health`, `observability.history`, `config`, `ingestion.embeddings`).

### Happy path — serving/api.py imports
`serving/api.py:23` -> `from agenticlog.retrieval.graph import AgentState, agent_workflow, inicializar_recursos`
`serving/api.py:36` -> `from agenticlog.observability.history import HistoryStore`
`serving/api.py:37` -> `from agenticlog.ingestion.embeddings import _get_rag_embedding_model`

### Failure path — CLI com RAGSecurityError
`cli.py` captura `RAGSecurityError` de `from agenticlog.shared.errors import RAGSecurityError` -> `logger.error(...)` -> `raise SystemExit(1)`.

### Failure path — shim deletado
Qualquer `import agenticlog.rag` remanescente -> `ModuleNotFoundError`. Todos os consumers sao migrados antes da delecao.

---

## API Changes

Nenhuma. O FastAPI `app`, endpoints (`POST /query`, `GET /health`, `GET /history`), e modelos Pydantic permanecem identicos. A string `agenticlog.api:app` muda para `agenticlog.serving.api:app` em `main_api.py` e `app.py`.

---

## Frontend Changes

`app.py` (Streamlit) imports mudam de `agenticlog.agent` e `agenticlog.rag` para modulos canonicos. Nenhuma alteracao de UI ou comportamento.

---

## Tests Required

**Remocoes (5 classes, 1 subtest):**
- `tests/ingestion/test_shims_identidade.py::TestShimsIdentidade` (2 testes)
- `tests/acceptance/test_rag_serving_fase5.py::TestServingShimsIdentidade` (7 testes)
- `tests/acceptance/test_rag_retrieval_fase4.py::TestShimIdentity` (1 teste parametrizado)
- `tests/acceptance/test_rag_retrieval_fase4.py::TestWrapperDelegation` (manter — wrappers sao reais, nao shims)
- `tests/acceptance/test_rag_serving_fase5.py::TestServingAcyclic` (manter — acicidade serving ainda relevante)
- `tests/ingestion/test_shims_identidade.py::TestIngestionAcyclic` (manter — acicidade ingestion ainda relevante)

**Atualizacoes de acicidade MODULES:**
- `tests/acceptance/test_rag_retrieval_fase4.py::TestAcyclicImport.MODULES` remover `"agenticlog.agent"` e adicionar `"agenticlog.retrieval.graph"` (ja lista), `"agenticlog.retrieval.generation"` (ja lista), etc. Remover `"agenticlog.agent"`.
- Manter `TestIngestionAcyclic` (so testa `agenticlog.ingestion`, nao menciona rag/agent).

**Migrations de patch-target (~50 patches):**
- `tests/test_rag.py`: ~15 patches `agenticlog.rag.X` -> `agenticlog.ingestion.*` ou `agenticlog.shared.*`
- `tests/test_agentic_rag.py`: ~10 patches `agenticlog.agent.X` -> `agenticlog.retrieval.*`
- `tests/ingestion/test_rag_caracterizacao.py`: 8 monkeypatches reescritos (SHIMS-16)
- `tests/acceptance/` varios: patches `agenticlog.rag.*`, `agenticlog.agent.*` para canonicos
- `tests/adicionar_pdf_incrementalmente`: 3 patches `agenticlog.rag.*` / `agenticlog.agent.*`

**NO driver de integracao/e2e.** A suite existente cobre unit + integration com mocks.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/rag.py` | **Deletado** | Shim de compatibilidade Fase 3a/3b removido. |
| `src/agenticlog/agent.py` | **Deletado** | Shim de compatibilidade Fase 4 removido. |
| `src/agenticlog/api.py` | **Deletado** | Shim de compatibilidade Fase 5 removido. |
| `src/agenticlog/health.py` | **Deletado** | Shim de compatibilidade Fase 5 removido. |
| `src/agenticlog/history.py` | **Deletado** | Shim de compatibilidade Fase 2 removido. |
| `src/agenticlog/config.py` | Modificado (ln 166) | Remover `from agenticlog.observability.logging import _JsonFormatter` shim. |
| `src/agenticlog/__init__.py` | Modificado | Migrar imports de `agenticlog.agent` + `agenticlog.health` para `agenticlog.retrieval.graph` + `agenticlog.serving.health`. |
| `src/agenticlog/serving/__init__.py` | Modificado | Atualizar _LAZY_MAP/__all__ (Q2). Remover ou manter lazy conforme decisao. |
| `src/agenticlog/serving/api.py` | Modificado | Migrar `from agenticlog.rag import _get_rag_embedding_model` -> `from agenticlog.ingestion.embeddings import _get_rag_embedding_model` (LM4); migrar `from agenticlog.history import HistoryStore` -> `from agenticlog.observability.history import HistoryStore` (LM4). |
| `src/agenticlog/ingestion/embeddings.py` | Modificado | Receber `_get_rag_embedding_model` singleton + cache de rag.py (SHIMS-08). |
| `src/agenticlog/ingestion/cli.py` | Modificado | Receber `_executar_main` + `_configurar_logging_cli` de rag.py + import operacional `RAGSecurityError` (SHIMS-08). |
| `src/agenticlog/ingestion/extraction.py` | Modificado | Receber `import fitz` (SHIMS-08). |
| `src/agenticlog/ingestion/orchestrator.py` | Modificado (ln 196) | LM1: lazy import `from agenticlog.retrieval.retriever import invalidar_vector_db`. |
| `src/agenticlog/retrieval/retriever.py` | Modificado | Receber `_vector_dbs`, `_embedding_model`, `_get_embedding_model`, `_listar_colecoes`, `_get_vector_db` de agent.py; `invalidar_vector_db` limpa dict LOCAL. |
| `src/agenticlog/retrieval/generation.py` | Modificado | Receber `_llm` singleton + `_get_llm` de agent.py. |
| `src/agenticlog/retrieval/graph.py` | Modificado | Receber `search` (DuckDuckGoSearchAPIWrapper) de agent.py. |
| `src/agenticlog/retrieval/__init__.py` | Modificado | Receber side-effects de startup (`import torch; torch.classes.__path__ = []; os.environ[...]`). |
| `app.py` | Modificado | Migrar imports de `agenticlog.agent` + `agenticlog.rag` para canonicos. |
| `main_api.py` | Modificado (ln 12) | LM2: `"agenticlog.api:app"` -> `"agenticlog.serving.api:app"`. |
| `scripts/rag_eval.py` | Modificado | Migrar imports de `agenticlog.agent` + `agenticlog.rag`. |
| `scripts/pdf_to_json.py` | Modificado | Migrar imports de `agenticlog.rag`. |
| `tests/test_rag.py` | Modificado | ~15 patches migrados. |
| `tests/test_agentic_rag.py` | Modificado | ~10 patches migrados. |
| `tests/test_rag_caracterizacao.py` | Modificado | 8 monkeypatches reescritos (SHIMS-16). |
| `tests/ingestion/test_shims_identidade.py` | Modificado | Remover `TestShimsIdentidade`, manter `TestIngestionAcyclic`. |
| `tests/acceptance/test_rag_serving_fase5.py` | Modificado | Remover `TestServingShimsIdentidade`, manter `TestServingAcyclic`. |
| `tests/acceptance/test_rag_retrieval_fase4.py` | Modificado | Remover `TestShimIdentity`; remover `"agenticlog.agent"` de `TestAcyclicImport.MODULES`. |
| `tests/adicionar_pdf_incrementalmente.py` | Modificado | Patches migrados. |
| `tests/ingestion/test_extraction.py` | Modificado | Patches migrados se referenciam `agenticlog.rag`. |
| `tests/ingestion/test_security.py` | Modificado | Patches migrados se referenciam `agenticlog.rag`. |
| `tests/ingestion/test_metadata.py` | Modificado | Patches migrados se referenciam `agenticlog.rag`. |
| `tests/acceptance/test_adicionar_pdf_incrementalmente.py` | Modificado | Patches migrados. |

---

## Risks

- **[CRITICO] Ordem de operacao:** Deletar shims ANTES de realocar singletons/globals/CLI/side-effects e migrar consumers causa `ModuleNotFoundError` em cascata. Tasks DEVEM seguir ordem: (1) realocar singletons, (2) migrar consumers, (3) migrar patches, (4) reescrever oracle, (5) remover identity tests, (6) DELETAR shims por ultimo.

- **[CRITICO] invalidar_vector_db ciclo:** `retriever.py:invalidar_vector_db` atualmente faz lazy import de `agenticlog.agent`. Apos agent.py deletado, a funcao precisa limpar `_vector_dbs` LOCAL em `retrieval.retriever`. Confirmar que nenhum outro modulo importa `invalidar_vector_db` de agent.py.

- **[ALTO] Oracle de caracterizacao quebra:** `tests/test_rag_caracterizacao.py` DIFF nao pode ser zero (monkeypatches mudam). Mas o COMPORTAMENTO (asserts) nao pode mudar. A diferenca de diff e inevitavel e documentada. O community gate e `pytest --cov=agenticlog -v` verde, nao diff zero.

- **[ALTO] ~50 patches de teste espalhados:** Patches existem em ~10 arquivos de teste. Inventario completo por-simbolo necessario. Qualquer patch nao migrado causa `ModuleNotFoundError` no import do shim deletado.

- **[MEDIO] `__init__.py` exporta de agenticlog.agent e agenticlog.health:** `src/agenticlog/__init__.py:17-21` importa `AgentState, agent_workflow` de `agenticlog.agent` e `check_lmstudio_health, LMStudioUnavailableError, ModeloNaoCarregadoError` de `agenticlog.health`. Migrar para `agenticlog.retrieval.graph` e `agenticlog.serving.health`. Verificar que `from agenticlog import AgentState, agent_workflow, check_lmstudio_health` continua funcionando.

- **[MEDIO] scripts/rag_eval.py imports condicionais:** Dentro de `try/except`, importa de `agenticlog.agent` e `agenticlog.rag`. Migrar para canonicos.

- **[MEDIO] `_get_rag_embedding_model` em serving/api.py + scripts/rag_eval.py:** Atualmente importa de `agenticlog.rag`. Com `ingestion/embeddings.py` unificado, ambos importam de la.

- **[MEDIO] mypy override em pyproject.toml:** `module = ["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]` — remover `"agenticlog.rag"` e `"agenticlog.api"` (shims deletados). Manter `"agenticlog.serving.api"`.

- **[MEDIO] CLI entrypoint:** `python -m agenticlog.rag` quebra. `pyproject.toml` (ou setup) precisa mapear `python -m agenticlog.ingestion` como novo entrypoint ou adicionar script console.

- **[Baixo] hnswlib SAC no Windows:** ~15 testes Chroma skipped localmente; CI Linux e gate autoritativo.

- **CLAUDE.md conflicts:** CLAUDE.md referencia `python -m agenticlog.rag` como comando de build. Atualizar para `python -m agenticlog.ingestion`. Remover secoes sobre rag.py e agent.py. Atualizar arquitetura.

---

## Open Questions

Q1 (JA DECIDIDA pelo humano): DELETAR rag.py e agent.py INTEIRAMENTE. Nao manter fachadas finas. Consequencias no design.md.
Q2: serving/__init__.py — manter lazy `__getattr__` (e adicionar HistoryStore ao `_LAZY_MAP`/`__all__`) OU simplificar para imports diretos?
Q3: Com invalidar_vector_db em retriever.py (local _vector_dbs), o ciclo retriever->agent some? Sim — confirmado no design.
Q4: Wrappers agent.py (_get_embedding_model, _listar_colecoes, _get_vector_db) migram para retrieval/retriever.py (nao sao mais wrappers, sao as definicoes reais).

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde no CI Linux; cobertura >= 80%.
- [ ] `ruff check --diff .` limpo; `mypy src/agenticlog/` limpo.
- [ ] `python -m agenticlog.ingestion --help` funciona (substitui `python -m agenticlog.rag`).
- [ ] `import agenticlog` em interpretador frio exit 0, sem "circular"/"partially initialized".
- [ ] Nenhum shim remanescente: `rg "# Re-export shim \(ADR-018" src/` retorna 0 matches.
- [ ] Nenhum import de `agenticlog.rag`, `agenticlog.agent`, `agenticlog.api`, `agenticlog.health`, `agenticlog.history` em `src/` ou `scripts/`.
- [ ] `main_api.py` inicia com `uvicorn agenticlog.serving.api:app`.
- [ ] `TestShimsIdentidade`, `TestServingShimsIdentidade`, `TestShimIdentity` removidos da suite.
