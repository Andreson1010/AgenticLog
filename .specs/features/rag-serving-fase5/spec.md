# ADR-018 Fase 5 — extracao do pacote `serving/` de `api.py` + `health.py` — Technical Spec

**Path:** `.specs/features/rag-serving-fase5/spec.md`
**TLC scope:** large
**Based on story:** Extrair `src/agenticlog/api.py` (397 ln) + `src/agenticlog/health.py` (131 ln) em `src/agenticlog/serving/` (api.py + health.py + __init__.py), refatoracao pura com shims identity-preserving, sem alteracao de comportamento.
**Status:** Awaiting human approval

---

## Problem Statement

`src/agenticlog/api.py` (397 linhas) e `src/agenticlog/health.py` (131 linhas) concentram a camada de entrega (FastAPI + health check LMStudio) em modulos soltos do pacote raiz `agenticlog`. A arquitetura-alvo (ADR-018 SS7) preve `serving/` como pacote coeso para a camada de entrega, separando-a dos modulos core (retrieval, ingestion, observability). Esta fase extrai ambos os modulos para `agenticlog.serving/` criando shims identity-preserving em `api.py` e `health.py`, com zero alteracao de comportamento. E refatoracao pura — a Fase 6 removera os shims.

## Goals

- [ ] Criar `src/agenticlog/serving/__init__.py`, `src/agenticlog/serving/api.py`, `src/agenticlog/serving/health.py` com o codigo verbatim de `api.py` e `health.py`, preservando identidade `is` de todos os simbolos publicos.
- [ ] `api.py` e `health.py` originais reduzem-se a shims identity-preserving com marcador `# Re-export shim (ADR-018 Fase 5) -- remover na Fase 6`.
- [ ] Logger names preservados: `"agenticlog.api"` e `"agenticlog.health"` explicitos (nao `__name__`).
- [ ] Oráculo `tests/test_rag_caracterizacao.py` com diff VAZIO vs origin/main.
- [ ] `main_api.py:12` string `agenticlog.api:app` continua resolvendo (via shim OU string atualizada — ver design Q2).
- [ ] `app.py:27` string `agenticlog.api:app` na mensagem de erro `MSG_CONNECT_ERROR` continua resolvendo.
- [ ] pyproject.toml mypy override (linha 72) inclui `agenticlog.serving.api`.
- [ ] `serving/health.py` mantem `import httpx` + `import logging` no topo (para `@patch` resolver).
- [ ] `serving/__init__.py` re-exporta API publica + `__all__`.
- [ ] Novo teste de identidade (tipo `tests/ingestion/test_shims_identidade.py`): `agenticlog.api.X is agenticlog.serving.api.X` + acicidade subprocess.
- [ ] `import agenticlog.serving` em interpretador frio exit 0, sem circular.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Fase 6 (remocao de shims + reescrita de testes) | Fase separada no roadmap; shims tem marcador `# Re-export shim (ADR-018 Fase 5) -- remover na Fase 6`. |
| Remocao do shim `__init__.py` `from agenticlog.health import ...` | Fase 6; `__init__.py` continua importando de `agenticlog.health` (que agora e shim). |
| Alteracao de comportamento de `api.py` ou `health.py` | Refatoracao pura — nenhuma linha de logica e alterada. |
| Migracao de `main_api.py` para `serving/` | `main_api.py` e entrypoint, nao parte de serving/. |
| Migracao de `app.py` (Streamlit) para `serving/` | app.py e frontend Streamlit, nao parte de serving/. |

---

## User Stories

### P1: Pacote `serving/` extraido com comportamento identico e zero-diff oracle (SERV-01..12)

**User Story**: Como mantenedor do AgenticLog, quero `api.py` e `health.py` extraidos para `agenticlog.serving/` com shims identity-preserving, preservando loggers explicitos, comportamento byte-identico, e zero diff nos arquivos-oraculo, para alinhar com a arquitetura-alvo offline/online (ADR-018 SS3/SS4/SS7) e reduzir o pacote raiz.

**Why P1**: E o objetivo unico da fase; sem ele nao ha entrega da Fase 5.

**Acceptance Criteria**:

1. WHEN api.py e movido para `serving/api.py` THEN `agenticlog.serving.api.app is agenticlog.api.app` SHALL ser verdadeiro (identidade `is`), e o mesmo SHALL valer para todos os simbolos publicos: `consultar`, `listar_historico`, `QueryRequest`, `QueryResponse`, `HistoryEntry`, `DocumentInfo`, `lifespan`, `_verificar_vectordb`, `_serializar_documentos`, `_normalizar_estado`, `_resposta_segura`, `_construir_registro`, `handler_lmstudio`, `handler_connect_error`, `handler_generico`. *(SERV-01)*

2. WHEN health.py e movido para `serving/health.py` THEN `agenticlog.serving.health.check_lmstudio_health is agenticlog.health.check_lmstudio_health` SHALL ser verdadeiro, e o mesmo SHALL valer para: `reset_health_check_sentinel`, `LMStudioUnavailableError`, `ModeloNaoCarregadoError`, `_health_checked`, `_extrair_ids_modelos`, `MAX_MODELOS_LOG`. *(SERV-02)*

3. WHEN `logging.getLogger()` e chamado com o nome do modulo THEN `logging.getLogger("agenticlog.api")` e `logging.getLogger("agenticlog.health")` SHALL continuar resolvendo para os loggers corretos — os modulos em `serving/` SHALL usar strings EXPLICITAS `"agenticlog.api"` e `"agenticlog.health"` (NAO `__name__`). *(SERV-03)*

4. WHEN `pytest -m integration tests/test_rag_caracterizacao.py` roda no CI Linux THEN SHALL passar com `git diff` VAZIO no arquivo. *(SERV-04)*

5. WHEN `main_api.py` e executado com `uvicorn agenticlog.api:app` THEN o servidor SHALL iniciar (a string `agenticlog.api:app` resolve via shim ou string atualizada — ver design). *(SERV-05)*

6. WHEN `app.py` e executado e o FastAPI nao esta disponivel THEN a mensagem de erro `MSG_CONNECT_ERROR` SHALL continuar mencionando `agenticlog.api:app` (string mantida ou atualizada — ver design). *(SERV-06)*

7. WHEN mypy e executado nos modulos `serving/` THEN o override em pyproject.toml para `module=["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]` SHALL estar presente. *(SERV-07)*

8. WHEN `@patch("agenticlog.health.httpx.Client")` ou `@patch("agenticlog.health.logger")` sao usados em testes THEN `serving/health.py` (o shim em `agenticlog/health.py`) SHALL manter `import httpx` e `import logging` no topo para que os patches resolvam. *(SERV-08)*

9. WHEN `import agenticlog.serving` e executado THEN SHALL ser viavel importar sem `ImportError`, e `python -c "import agenticlog.serving"` SHALL exitar com codigo 0. *(SERV-09)*

10. WHEN `serving/__init__.py` e carregado THEN SHALL re-exportar todos os simbolos publicos de `api` e `health`, e `__all__` SHALL conter a lista completa. *(SERV-10)*

11. WHEN o novo teste de identidade e executado THEN `agenticlog.api.X is agenticlog.serving.api.X` para todos os ~20 simbolos publicos SHALL ser verdadeiro, e `subprocess` importando `agenticlog.serving` em interpretador frio SHALL exitar 0 sem "circular"/"partially initialized". *(SERV-11)*

12. WHEN a suite completa de testes roda (`pytest --cov=agenticlog --cov-report=term-missing -v`) THEN SHALL passar com 731+ testes verdes (ou o numero atual) e cobertura >= 80%, e `tests/test_rag_caracterizacao.py` com zero diff. *(SERV-12)*

**Independent Test**: `pytest -m integration tests/test_rag_caracterizacao.py -v` verde; `git diff --stat` VAZIO; `pytest tests/acceptance/test_rag_serving_fase5.py -v` verde; `pytest --cov=agenticlog -v` completo verde no CI Linux.

---

## Edge Cases

- WHEN um teste importa `agenticlog.api` usando `import agenticlog.api as api_module` (ex.: `test_observability_history.py`, `test_rag_shared_observability.py`) THEN `api_module.app`, `api_module.X` SHALL continuar resolvendo via shim — `import agenticlog.api` importa o shim `agenticlog/api.py` que re-exporta de `serving/api.py`.
- WHEN um teste faz `from agenticlog.api import app` (ex.: `test_api.py`, `test_modo_seguro_modelo_indisponivel.py`, `test_history_endpoint.py`) THEN o `app` resolvido SHALL ser o mesmo objeto definido em `serving/api.py` (identidade `is`).
- WHEN um teste faz `from agenticlog.health import X` (ex.: `test_api.py:29`, `test_health_check.py:27`) THEN o simbolo SHALL resolver via shim de `agenticlog/health.py` para `serving/health.py`.
- WHEN um teste faz `import agenticlog.health as health_module` (ex.: `test_health.py:23`) e usa `patch.object(health_module, "logger")` THEN o patch SHALL continuar funcionando pois `health_module` referencia o modulo shim `agenticlog/health.py` que mantem `import logging` no topo.
- WHEN um teste faz `@patch("agenticlog.api.inicializar_recursos")` (ocorre ~50 vezes) THEN o patch SHALL continuar funcionando: `inicializar_recursos` e importado de `agenticlog.agent` em `serving/api.py` como nome local, e `agenticlog.api.inicializar_recursos` e shim que re-exporta de `serving.api.inicializar_recursos` -- entao `@patch("agenticlog.api.inicializar_recursos")` patcha o namespace do shim, mas serving/api.py ja tem sua propria referencia local. **Risco critico:** este patch NAO intercepta a chamada dentro de serving/api.py. Ver design SS5 para classificacao detalhada.
- WHEN um teste faz `@patch("agenticlog.health.httpx.Client")` (ocorre ~14 vezes) THEN o patch SHALL continuar funcionando pois o shim `agenticlog/health.py` mantem `import httpx` no topo.

---

## Requirement Traceability

| Requirement ID | Story | AC | Phase | Status |
|----------------|-------|----|-------|--------|
| SERV-01 (api symbols identity) | P1 | AC1 | Design | Pending |
| SERV-02 (health symbols identity) | P1 | AC2 | Design | Pending |
| SERV-03 (logger names explicitos) | P1 | AC3 | Design | Pending |
| SERV-04 (oraculo zero-diff) | P1 | AC4 | Design | Pending |
| SERV-05 (main_api.py string resolve) | P1 | AC5 | Design | Pending |
| SERV-06 (app.py error string) | P1 | AC6 | Design | Pending |
| SERV-07 (mypy override) | P1 | AC7 | Design | Pending |
| SERV-08 (shim health mantem import httpx+logging) | P1 | AC8 | Design | Pending |
| SERV-09 (serving importavel frio) | P1 | AC9 | Design | Pending |
| SERV-10 (serving/__init__.py re-export + __all__) | P1 | AC10 | Design | Pending |
| SERV-11 (novo teste identidade + acicidade) | P1 | AC11 | Design | Pending |
| SERV-12 (suite completa mantida) | P1 | AC12 | Design | Pending |

**ID format:** `SERV-[NUMBER]` (SERV = SERVing).

**Coverage:** 12 IDs totais; todos mapeados a tasks (ver `tasks.md`).

---

## Data Model Changes

Nenhuma. Nenhum schema Pydantic e alterado. `QueryRequest`, `QueryResponse`, `HistoryEntry`, `DocumentInfo` mudam de localizacao fisica mas preservam identidade `is`. `AgentState` permanece em `agenticlog.agent` / `agenticlog.retrieval.state`.

---

## Process / Background Flow

### Happy path — FastAPI startup
`main_api.py` chama `uvicorn.run("agenticlog.api:app")` -> resolve para `agenticlog.serving.api.app` (via shim identity) -> `lifespan` executa -> `_verificar_vectordb()` -> `inicializar_recursos()` (de agent) -> servidor pronto.

### Happy path — POST /query
`app.py` chama `httpx.post(.../query)` -> FastAPI roteia para `consultar()` em `serving/api.py` -> `check_lmstudio_health()` (de serving/health.py) -> `agent_workflow.invoke(AgentState(...))` (de agent.py) -> `_normalizar_estado()` -> response.

### Failure path — pre-flight health check
`consultar()` chama `check_lmstudio_health()` -> levanta `LMStudioUnavailableError` -> capturado por `_ERROS_MODO_SEGURO` -> retorna `_resposta_segura()` com `degraded=True`.

### Failure path — vectordb ausente
`lifespan` chama `_verificar_vectordb()` -> `RuntimeError` -> `app.state.vectordb_pronto = False` -> servidor sobe, mas todas as requisicoes retornam 503.

---

## API Changes

Nao ha alteracoes contratuais. Todos os endpoints (`POST /query`, `GET /history`) e modelos (`QueryRequest`, `QueryResponse`, `HistoryEntry`) permanecem identicos. Exception handlers preservados. FastAPI app e o mesmo objeto `is`.

---

## Frontend Changes

Nao ha alteracoes no frontend. `app.py` continua importando de `agenticlog.api` (shim) e `agenticlog.rag` (shim). A string `MSG_CONNECT_ERROR` em `app.py:27` contem `agenticlog.api:app` — mantida (ver design SS6 para recomendacao).

---

## Tests Required

**Novos (SERV-11):**
- `tests/acceptance/test_rag_serving_fase5.py`:
  - Identidade `is`: `agenticlog.api.X is agenticlog.serving.api.X` para todos os ~21 simbolos de api.py.
  - Identidade `is`: `agenticlog.health.X is agenticlog.serving.health.X` para todos os ~8 simbolos de health.py.
  - Acicidade fresh-interpreter: `import agenticlog.serving` -> exit 0, sem "circular"/"partially initialized".
  - Round-trip `import agenticlog.api` (shim) + `import agenticlog.serving.api` (definicao) -> `api.app is serving.api.app`.

**Migracao de patch-target (CRITICO — ~50 patches agenticlog.api.* + ~16 patches agenticlog.health.*):**

Cada `@patch("agenticlog.api.X")` onde X e importado como nome LOCAL em `serving/api.py` (ex.: `from agenticlog.agent import inicializar_recursos`) precisa migrar para `@patch("agenticlog.serving.api.X")` porque o namespace do shim `agenticlog.api` nao e lido por `serving/api.py`.

**Testes com `assert_called_once()` que sao SENSIVEIS:**
- `test_ac_api_10_singletons_initialized_at_startup` (`assert mock_init.call_count == 1`)

**Testes que NAO se modificam (importam de `agenticlog.api` via shim):**
- `tests/acceptance/test_multi_collection_chromadb.py`: `from agenticlog.api import QueryRequest` (shim preserva)
- `tests/acceptance/test_rag_shared_observability.py`: `import agenticlog.api as api_module` (shim preserva)

**Testes com `@patch("agenticlog.health.*")` que NAO se modificam (shim em agenticlog/health.py mantem imports):**
- `tests/test_health.py`: `agenticlog.health.httpx.Client` (14x), `agenticlog.health.logger` (via `patch.object(health_module, "logger")` — shim mantem `import logging`)
- `tests/acceptance/test_health_check.py`: `agenticlog.health.httpx.Client` (5x), `agenticlog.health.logger` (3x)

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/serving/__init__.py` | Novo | Init package + re-exports API publica + `__all__`. |
| `src/agenticlog/serving/api.py` | Novo | Codigo verbatim de `api.py` (menos os herders de logger). |
| `src/agenticlog/serving/health.py` | Novo | Codigo verbatim de `health.py` (menos os herders de logger). |
| `src/agenticlog/api.py` | Modificado (~30 ln) | Reduzir a shim inline: `from agenticlog.serving.api import ... # Re-export shim (ADR-018 Fase 5) -- remover na Fase 6`. Manter so imports e docstring. |
| `src/agenticlog/health.py` | Modificado (~20 ln) | Reduzir a shim inline: `from agenticlog.serving.health import ... # Re-export shim (ADR-018 Fase 5) -- remover na Fase 6`. Manter `import httpx` + `import logging` no topo. |
| `pyproject.toml` | Modificado (linha 72) | Incluir `"agenticlog.serving.api"` no mypy override: `module = ["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]`. |
| `tests/acceptance/test_rag_serving_fase5.py` | Novo | Teste de identidade + acicidade. |
| `tests/test_api.py` | Modificado | Migrar patch-target de `agenticlog.api.inicializar_recursos`, `agenticlog.api._verificar_vectordb`, `agenticlog.api.check_lmstudio_health`, `agenticlog.api.agent_workflow.invoke`, `agenticlog.api.HistoryStore`, `agenticlog.api.DIR_VECTORDB`, `agenticlog.api.Chroma`, `agenticlog.api._get_rag_embedding_model`, `agenticlog.api.asyncio.to_thread` para `agenticlog.serving.api.*`. |
| `tests/acceptance/test_api_query_endpoint.py` | Modificado | Migrar patch-target para `agenticlog.serving.api.*`. |
| `tests/acceptance/test_modo_seguro_modelo_indisponivel.py` | Modificado | Migrar patch-target para `agenticlog.serving.api.*`. |
| `tests/acceptance/test_history_endpoint.py` | Modificado | Migrar `agenticlog.api.agent_workflow.invoke` e `agenticlog.api.inicializar_recursos` para `agenticlog.serving.api.*`. |
| `tests/acceptance/test_query_history_audit_logging.py` | Modificado | Migrar `agenticlog.api.agent_workflow.invoke` e `agenticlog.api.inicializar_recursos` para `agenticlog.serving.api.*`. |
| `tests/test_rag_caracterizacao.py` | **NAO tocar** | Oraculo de caracterizacao (HARD constraint — zero diff). |
| `tests/ingestion/test_shims_identidade.py` | **NAO tocar** | Oraculo de identidade da Fase 3a (HARD constraint — zero diff). |

---

## Risks

- **[CRITICO] Patch-target migration (LIC AO FASE 4 IMP-02 — obrigatorio):** ~50 patches `agenticlog.api.*` e ~16 `agenticlog.health.*`. Simbolos que `serving/api.py` importa como NOME LOCAL (ex.: `from agenticlog.agent import inicializar_recursos`, `from agenticlog.health import check_lmstudio_health`) NAO sao interceptados por `@patch("agenticlog.api.X")` — porque serving/api.py ja tem sua propria referencia local. Estes patches DEVEM migrar para `agenticlog.serving.api.*`. Testes com `assert_called_once()` (ex.: `test_ac_api_10`) sao os mais sensiveis. O design SS5 apresenta inventario completo por-simbolo classificando cada patch em (a) migrar, (b) manter, (c) manter via shim import.

- **[ALTO] Logger name `__name__` vs string explicita:** Se `serving/api.py` usar `logger = logging.getLogger(__name__)` em vez de `logger = logging.getLogger("agenticlog.api")`, o logger name muda para `agenticlog.serving.api` e os testes/configuracao que referenciam `"agenticlog.api"` no logger quebram. **Mitigacao:** usar STRING EXPLICITA `"agenticlog.api"` e `"agenticlog.health"` em serving/.

- **[ALTO] Import circular via `serving/api.py`:** `serving/api.py` importa de `agenticlog.agent` (que importa de retrieval, que importa de config — folha). `agenticlog.agent` NAO importa `serving/` (shims em agent.py sao para retrieval, nao serving). Nao ha ciclo. Mas `serving/api.py` importa de `agenticlog.health` — se health virar shim que importa de `serving.health`, e `serving.health` importar de config (folha), nao ha ciclo. Confirmado: aciclico.

- **[MEDIO] `import agenticlog.health as health_module` com `patch.object`:** `test_health.py` faz `import agenticlog.health as health_module` e `patch.object(health_module, "logger")`. Isso resolve porque `agenticlog/health.py` (shim) mantem `import logging` no topo e `logger = logging.getLogger(...)`. O nome `logger` ainda e atributo do modulo shim. Preservado.

- **[MEDIO] `from agenticlog.api import app` em `test_history_endpoint.py` e `test_modo_seguro.py`:** esses testes importam `app` diretamente de `agenticlog.api` e criam `TestClient(app)`. O `app` importado e o shim re-exportado — que e o mesmo objeto definido em `serving/api.py`. `TestClient` funciona igual. Nao ha quebra.

- **[MEDIO] `import agenticlog.api as api_module` em `test_rag_shared_observability.py` e `test_observability_history.py`:** esses modulos importam `api_module` e acessam atributos (ex.: `api_module.app.state.history_store`). O `api_module` e o modulo shim `agenticlog/api.py` (reduzido a ~30 ln com shims). O atributo `app` resolve via re-export. Funciona.

- **[MEDIO] `from agenticlog.api import MSG_VECTORDB_AUSENTE, app` em `test_modo_seguro_modelo_indisponivel.py`:** `MSG_VECTORDB_AUSENTE` e constante de modulo em `api.py` que precisa ser re-exportada como shim. **Simbolo nao listado no AC1 mas necessario.**

- **[MEDIO] Oráculo `test_rag_caracterizacao.py` depende de `agenticlog.api` estar no path:** o oraculo NUNCA e modificado, mas se `api.py` shim nao re-exportar `_normalizar_estado` ou outro simbolo usado indiretamente pelo oraculo, o diff aparece. **Mitigacao:** verificar que o shim em api.py re-exporta TODOS os simbolos (incluindo privados `_*`).

- **[Baixo] `MSG_LMSTUDIO_INDISPONIVEL` e `MSG_VECTORDB_AUSENTE` sao constantes de modulo:** precisam ser re-exportadas como shims. Garantir que aparecam no shim de api.py.

- **[Ambiental] hnswlib SAC no Windows:** oraculo/testes Chroma sao *skipped* local (Windows/Smart App Control); o CI Linux e o gate autoritativo.

- **CLAUDE.md conflicts:** nenhum. Alinha com "small files" (serving/api.py ~400 ln < 800), type hints, docstrings PT, imutabilidade, commits Conventional em PT.

---

## Open Questions

Resolvidas no design.md com recomendacao tecnica — usuarix decide no CP2.

- **Q2 — String `agenticlog.api:app` em main_api.py:12 + app.py:27:** Atualizar para `agenticlog.serving.api:app` OU manter via shim? Ver design SS6.
- **Q4 — Import style em serving/api.py:** `from agenticlog.agent import X` (nome local — patch-target migra) vs `import agenticlog.agent` (dot-qualified, patch-safe estilo Fase 3b)? Ver design SS7.

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde no CI Linux; cobertura >= 80%.
- [ ] `git diff --stat tests/test_rag_caracterizacao.py` VAZIO.
- [ ] `agenticlog.api.X is agenticlog.serving.api.X` para todos os ~21 simbolos publicos de api.
- [ ] `agenticlog.health.X is agenticlog.serving.health.X` para todos os ~8 simbolos publicos de health.
- [ ] `import agenticlog.serving` em interpretador frio exit 0.
- [ ] `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings PT (Entrada/Saida/Lanca).
- [ ] `main_api.py` inicia com `uvicorn agenticlog.api:app` (shim ou string atualizada).
