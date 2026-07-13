# ADR-018 Fase 5 — extracao do pacote `serving/` de `api.py` + `health.py` — Tasks

**Path:** `.specs/features/rag-serving-fase5/tasks.md`
**Spec:** `.specs/features/rag-serving-fase5/spec.md` · **Design:** `.specs/features/rag-serving-fase5/design.md`
**TLC scope:** large
**Status:** Awaiting human approval

Tasks para o backend-builder da feature-factory. Ordem projetada para oraculo + suite completa verdes ao final. TLC Execute NAO e usado. Gate autoritativo: **CI Linux** (`pytest --cov=agenticlog -v`). Local Windows pula testes Chroma/hnswlib (Smart App Control) — nao confundir com regressao. Padroes: `ruff`/`black`/`isort` limpos, type hints, docstrings PT (Entrada/Saida/Lanca), imutabilidade, funcoes <50 ln, commits Conventional em PT (`refactor:`), branch ja criada (`feature/rag-serving-fase5`), code-review antes do merge.

**Regra de ouro (todas as tasks):** NUNCA editar `tests/test_rag_caracterizacao.py` nem `tests/ingestion/test_shims_identidade.py`. Preservar comportamento, mensagens, ordem de excecoes verbatim. Migracao de teste = so ALVO de patch (namespace), nunca a assercao de comportamento.

**Marcadores:** todo shim de re-export em `api.py` e `health.py` deve ter comentario:
```python
# Re-export shim (ADR-018 Fase 5) — remover na Fase 6
```

**Ordem macro: serving/health.py -> serving/api.py -> shims api.py/health.py -> pyproject -> serving/__init__.py -> teste identidade novo -> migracao patch-target -> suite completa.**

---

## Execution Plan

### Fase A — Modulos serving/ (sequencial)
```
T1 -> T2
```

### Fase B — Shims + pyproject (sequencial, depende de T1+T2)
```
T2 -> T3 -> T4 -> T5
```

### Fase C — Teste novo (paralelo apos T5)
```
         -> T6 [P]
T5 ------
```

### Fase D — Migracao patch-target (sequencial por arquivo, depende de T5)
```
         -> T7 [P] (test_api.py)
         -> T8 [P] (test_api_query_endpoint.py)
T5 ------ -> T9 [P] (test_modo_seguro_modelo_indisponivel.py)
         -> T10 [P] (test_history_endpoint.py + test_query_history_audit_logging.py)
```

### Fase E — Gate final (sequencial)
```
T6..T10 -> T11
```

---

## Task Breakdown

### T1: Criar `serving/health.py` com logger string explicita (SERV-02, SERV-03, SERV-08)

**What:** Criar `src/agenticlog/serving/health.py` com o conteudo VERBATIM de `src/agenticlog/health.py` exceto:
- Substituir `logger = logging.getLogger(__name__)` por `logger = logging.getLogger("agenticlog.health")`

Manter todas as outras linhas identicas: imports, classes `LMStudioUnavailableError`/`ModeloNaoCarregadoError`, funcao `reset_health_check_sentinel`, `_extrair_ids_modelos`, `check_lmstudio_health`, constantes `MAX_MODELOS_LOG`, global `_health_checked`, docstrings.

**Where:** `src/agenticlog/serving/health.py` (novo)
**Depends on:** None (so importa config + stdlib)
**Reuses:** `src/agenticlog/health.py` (corpo verbatim)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `serving/health.py` tem `logger = logging.getLogger("agenticlog.health")` (NAO `__name__`).
- [ ] `from agenticlog.serving.health import check_lmstudio_health` funciona.
- [ ] `check_lmstudio_health` tem o mesmo comportamento (mesmas excecoes, mesmas mensagens, mesmo retorno).
- [ ] `python -c "import agenticlog.serving.health"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T6) · **Gate:** quick

---

### T2: Criar `serving/api.py` com logger string explicita (SERV-01, SERV-03)

**What:** Criar `src/agenticlog/serving/api.py` com o conteudo VERBATIM de `src/agenticlog/api.py` exceto:
- Substituir `logger = logging.getLogger(__name__)` por `logger = logging.getLogger("agenticlog.api")`

Manter todas as outras linhas identicas: imports, modelos Pydantic, helpers, lifespan, endpoints, exception handlers.

**Where:** `src/agenticlog/serving/api.py` (novo)
**Depends on:** T1 (serving/api.py importa de serving/health.py? NAO — importa de agenticlog.health shim. Nao depende de T1.)
**Reuses:** `src/agenticlog/api.py` (corpo verbatim)

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `serving/api.py` tem `logger = logging.getLogger("agenticlog.api")` (NAO `__name__`).
- [ ] `from agenticlog.serving.api import app, consultar, listar_historico, QueryRequest` funciona.
- [ ] Todos os ~21 simbolos de API sao importaveis.
- [ ] `python -c "import agenticlog.serving.api"` exit 0 (sem ciclo).
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T6) · **Gate:** quick

---

### T3: Criar shim `agenticlog/health.py` com `import httpx` + `import logging` (SERV-08, SERV-02)

**What:** Reescrever `src/agenticlog/health.py` como shim que re-exporta todos os simbolos de `serving/health.py`:
1. Manter docstring (adaptada).
2. Manter `import httpx` e `import logging` no topo (para `@patch("agenticlog.health.httpx.Client")` e `patch.object(health_module, "logger")` resolverem).
3. Adicionar `from agenticlog.serving.health import (...)` com todos os simbolos de health.py.
4. Adicionar `logger = logging.getLogger("agenticlog.health")` APOS o bloco de import do shim.
5. Cada simbolo importado com comentario `# Re-export shim (ADR-018 Fase 5) — remover na Fase 6`.

**Simbolos a re-exportar:**
```python
check_lmstudio_health,
reset_health_check_sentinel,
LMStudioUnavailableError,
ModeloNaoCarregadoError,
_health_checked,
_extrair_ids_modelos,
MAX_MODELOS_LOG,
```

**Where:** `src/agenticlog/health.py` (modificado)
**Depends on:** T1
**Reuses:** padrao de shim de `rag.py` (Fase 3a — mantem `import fitz`), `agent.py` (Fase 4).

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `import httpx` e `import logging` estao no topo de health.py.
- [ ] `logger = logging.getLogger("agenticlog.health")` esta definido no modulo.
- [ ] `from agenticlog.health import check_lmstudio_health` resolve para o objeto em `serving/health.py`.
- [ ] `agenticlog.health.check_lmstudio_health is agenticlog.serving.health.check_lmstudio_health` (identity `is`).
- [ ] `agenticlog.health.httpx` e um atributo do modulo (o modulo httpx, importado via `import httpx` top-level).
- [ ] `agenticlog.health.logger` e o logger com name `"agenticlog.health"`.
- [ ] `python -c "import agenticlog.health"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T6) · **Gate:** quick

---

### T4: Criar shim `agenticlog/api.py` (SERV-01, SERV-05)

**What:** Reescrever `src/agenticlog/api.py` como shim que re-exporta todos os simbolos de `serving/api.py`:
1. Adaptar docstring para indicar shim de compatibilidade.
2. Adicionar `from agenticlog.serving.api import (...)` com todos os simbolos (incluindo privados).
3. Nao manter imports desnecessarios (o shim so importa de serving.api).

**Simbolos a re-exportar:**
```python
# Modelos Pydantic
QueryRequest, QueryResponse, HistoryEntry, DocumentInfo,
# Constantes
MSG_LMSTUDIO_INDISPONIVEL, MSG_VECTORDB_AUSENTE, _ERROS_MODO_SEGURO,
# Helpers
_verificar_vectordb, _serializar_documentos, _normalizar_estado, _resposta_segura, _construir_registro,
# FastAPI
lifespan, app,
# Endpoints
consultar, listar_historico,
# Exception handlers
handler_lmstudio, handler_connect_error, handler_generico,
```

NOTA: `_ERROS_MODO_SEGURO` usa tipos de `agenticlog.health` (LMStudioUnavailableError, ModeloNaoCarregadoError) e `httpx.ConnectError`, `openai.APIConnectionError`. O shim nao precisa importar esses tipos — a definicao original em serving/api.py ja os importa.

**Where:** `src/agenticlog/api.py` (modificado)
**Depends on:** T2
**Reuses:** padrao de shim de `rag.py`, `agent.py`.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `from agenticlog.api import app, consultar, QueryRequest, _normalizar_estado` resolve.
- [ ] `agenticlog.api.app is agenticlog.serving.api.app` (identity `is`).
- [ ] `agenticlog.api.MSG_VECTORDB_AUSENTE is agenticlog.serving.api.MSG_VECTORDB_AUSENTE`.
- [ ] `python -c "import agenticlog.api"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T6 + T7..T10) · **Gate:** quick

---

### T5: Atualizar `pyproject.toml` + criar `serving/__init__.py` (SERV-07, SERV-10)

**What:**
1. Atualizar `pyproject.toml` linha 72: `module = ["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]` (adicionar `"agenticlog.serving.api"`).
2. Criar `src/agenticlog/serving/__init__.py` com re-exports de todos os simbolos publicos de `serving/api.py` e `serving/health.py` + `__all__`.

**Where:** `pyproject.toml` (modificado), `src/agenticlog/serving/__init__.py` (novo)
**Depends on:** T4 (shims), T3 (shim health)
**Reuses:** `ingestion/__init__.py` como referencia de estrutura.

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pyproject.toml` contem `"agenticlog.serving.api"` no mypy override.
- [ ] `from agenticlog.serving import app, consultar, check_lmstudio_health` resolve.
- [ ] `agenticlog.serving.app is agenticlog.serving.api.app`.
- [ ] `agenticlog.serving.check_lmstudio_health is agenticlog.serving.health.check_lmstudio_health`.
- [ ] `agenticlog.serving.__all__` contem todos os simbolos do design SS9.
- [ ] `python -c "import agenticlog.serving"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** (validado em T6) · **Gate:** quick

---

### T6: Novo teste de identidade + acicidade serving/ (SERV-11)

**What:** Criar `tests/acceptance/test_rag_serving_fase5.py` com:

1. **Identidade api:** `agenticlog.api.app is agenticlog.serving.api.app`, e similar para todos os ~21 simbolos de api.py.

2. **Identidade health:** `agenticlog.health.check_lmstudio_health is agenticlog.serving.health.check_lmstudio_health`, e similar para todos os ~8 simbolos de health.py.

3. **Identidade MSG_VECTORDB_AUSENTE:** `agenticlog.api.MSG_VECTORDB_AUSENTE is agenticlog.serving.api.MSG_VECTORDB_AUSENTE`.

4. **Logger names:** `logging.getLogger("agenticlog.api").name == "agenticlog.api"` e `logging.getLogger("agenticlog.health").name == "agenticlog.health"`.

5. **Patch compatibility — health shim:** Verificar que `agenticlog.health.httpx` e o modulo httpx importado (atributo do shim). Verificar que `agenticlog.health.logger` e o Logger (atributo do shim).

6. **Acicidade fresh-interpreter:** `subprocess`:
   - `import agenticlog.serving` -> returncode 0, sem "circular"/"partially initialized".
   - `import agenticlog.api; import agenticlog.serving.api` -> returncode 0.
   - `import agenticlog.health; import agenticlog.serving.health` -> returncode 0.

7. **Round-trip:** Importar de `agenticlog.api` (shim) e de `agenticlog.serving.api` (definicao) e verificar identidade.

**Where:** `tests/acceptance/test_rag_serving_fase5.py` (novo)
**Depends on:** T5 (shims + __init__)
**Reuses:** `tests/ingestion/test_shims_identidade.py` (template de identidade + acicidade).

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_rag_serving_fase5.py -v` verde.
- [ ] Todos os ~21 simbolos de api tem identidade `is` comprovada.
- [ ] Todos os ~8 simbolos de health tem identidade `is` comprovada.
- [ ] Acicidade fresh-interpreter comprovada para `serving/`.

**Tests:** unit (identidade) + integration (subprocess) · **Gate:** full

---

### T7: Migrar patch-target de `tests/test_api.py` (SERV-12)

**What:** Migrar ALVOS de patch em `tests/test_api.py`. Atualizar as duas funcoes helper:
- `_client_vectordb_pronto` (5 patches)
- `_client_vectordb_ausente` (3 patches)

E atualizar patches inline nos testes individuais (teste_9..teste_18).

**Tabela de migracao:**

| Patch atual | Novo alvo |
|-------------|-----------|
| `"agenticlog.api.inicializar_recursos"` | `"agenticlog.serving.api.inicializar_recursos"` |
| `"agenticlog.api._verificar_vectordb"` | `"agenticlog.serving.api._verificar_vectordb"` |
| `"agenticlog.api.check_lmstudio_health"` | `"agenticlog.serving.api.check_lmstudio_health"` |
| `"agenticlog.api.agent_workflow.invoke"` | `"agenticlog.serving.api.agent_workflow.invoke"` |
| `"agenticlog.api.HistoryStore"` | `"agenticlog.serving.api.HistoryStore"` |
| `"agenticlog.api.DIR_VECTORDB"` | `"agenticlog.serving.api.DIR_VECTORDB"` |
| `"agenticlog.api.Chroma"` | `"agenticlog.serving.api.Chroma"` |
| `"agenticlog.api._get_rag_embedding_model"` | `"agenticlog.serving.api._get_rag_embedding_model"` |
| `"agenticlog.api.asyncio.to_thread"` | `"agenticlog.serving.api.asyncio.to_thread"` |

NOTA: Os imports de topo do arquivo (`from agenticlog.api import ...`) NAO mudam — o shim re-exporta. Os patches mudam porque interceptam o namespace usado pelo consumer (serving/api.py).

**Where:** `tests/test_api.py`
**Depends on:** T5

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/test_api.py -v` verde.
- [ ] Contagem de testes preservada (mesmo numero).
- [ ] Todos os `assert_called_once` passam (test_13, test_16, test_9c).

**Tests:** unit · **Gate:** quick

---

### T8: Migrar patch-target de `tests/acceptance/test_api_query_endpoint.py` (SERV-12)

**What:** Migrar ALVOS de patch na funcao helper `_client` (linhas 53-67) e nos testes inline.

A funcao `_client` usa 5 patches:
```python
patch("agenticlog.api.inicializar_recursos")
patch("agenticlog.api._verificar_vectordb")
patch("agenticlog.api.check_lmstudio_health")
patch("agenticlog.api.agent_workflow.invoke")
patch("agenticlog.api.HistoryStore")
```

Migrar todos para `"agenticlog.serving.api.*"`.

**Where:** `tests/acceptance/test_api_query_endpoint.py`
**Depends on:** T5

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_api_query_endpoint.py -v` verde.
- [ ] `test_ac_api_10_singletons_initialized_at_startup` passa com `mock_init.call_count == 1`.

**Tests:** unit · **Gate:** quick

---

### T9: Migrar patch-target de `tests/acceptance/test_modo_seguro_modelo_indisponivel.py` (SERV-12)

**What:** Migrar ALVOS de patch na funcao helper `_client_ctx` (linhas 66-101):

```python
patch("agenticlog.api.inicializar_recursos")
patch("agenticlog.api._verificar_vectordb")
patch("agenticlog.api.check_lmstudio_health")
patch("agenticlog.api.agent_workflow.invoke")
patch("agenticlog.api.HistoryStore")
```

Migrar todos para `"agenticlog.serving.api.*"`.

**Where:** `tests/acceptance/test_modo_seguro_modelo_indisponivel.py`
**Depends on:** T5

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_modo_seguro_modelo_indisponivel.py -v` verde.
- [ ] Todos os 11 testes de modo seguro passam (incluindo os parametrizados).

**Tests:** unit · **Gate:** quick

---

### T10: Migrar patch-target de `tests/acceptance/test_history_endpoint.py` + `test_query_history_audit_logging.py` (SERV-12)

**What:** Migrar patches nestes 2 arquivos.

**test_history_endpoint.py:**
- Linhas 64-66: `"agenticlog.api.agent_workflow.invoke"` -> `"agenticlog.serving.api.agent_workflow.invoke"`
- Linha 66: `"agenticlog.api.inicializar_recursos"` -> `"agenticlog.serving.api.inicializar_recursos"`
- Linhas 92-94: idem.

**test_query_history_audit_logging.py:**
- Linhas 91-93: `"agenticlog.api.agent_workflow.invoke"` -> `"agenticlog.serving.api.agent_workflow.invoke"`
- Linha 93: `"agenticlog.api.inicializar_recursos"` -> `"agenticlog.serving.api.inicializar_recursos"`
- Linhas 127-129: idem.

**Where:** `tests/acceptance/test_history_endpoint.py`, `tests/acceptance/test_query_history_audit_logging.py`
**Depends on:** T5

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `pytest tests/acceptance/test_history_endpoint.py -v` verde.
- [ ] `pytest tests/acceptance/test_query_history_audit_logging.py -v` verde.

**Tests:** unit/integration · **Gate:** full

---

### T11: Gate final — suite completa + lint + verificacao de identidade + oraculo (SERV-12, SERV-04)

**What:** Rodar a suite completa, lint e as verificacoes de diff.

**Where:** todo o repo
**Depends on:** T6, T7, T8, T9, T10

**Tools:** MCP: NONE · Skill: `code-review` (antes do merge)

**Done when:**
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde (CI Linux); cobertura >= 80%.
- [ ] `git diff --stat tests/test_rag_caracterizacao.py` VAZIO.
- [ ] `ruff check .`, `black --check .`, `isort --check .` limpos.
- [ ] Varredura de patch: `rg "@patch\(\"agenticlog\.api\.(.*?)\"\)" tests/` — cada ocorrencia remanescente EXERCEITA um simbolo do shim que serve via re-export (classe `is`-identity). Idealmente ZERO ocorrencias de `agenticlog.api.*` em patches (todos migrados para `agenticlog.serving.api.*`).
- [ ] Varredura de patch health: `rg "@patch\(\"agenticlog\.health\.(.*?)\"\)" tests/` — cada ocorrencia DEVE ser `httpx.Client` ou `logger` (mantidos via shim com `import httpx` + `import logging`). QUALQUER outro simbolo de health patchado deve ser verificado.
- [ ] `logging.getLogger("agenticlog.api").name == "agenticlog.api"` verificado no teste T6.
- [ ] `logging.getLogger("agenticlog.health").name == "agenticlog.health"` verificado no teste T6.
- [ ] `code-review` executado antes do merge.

**Tests:** integration (full suite) · **Gate:** build

**Commit:** `refactor(rag): extrai pacote serving/ de api.py + health.py (ADR-018 Fase 5)`

---

## Diagram-Definition Cross-Check

| Task | Depends On | Fase | Status |
|------|-----------|------|--------|
| T1 (serving/health.py) | None | A | Pendente |
| T2 (serving/api.py) | None | A | Pendente |
| T3 (health.py shim) | T1 | B | Pendente |
| T4 (api.py shim) | T2 | B | Pendente |
| T5 (pyproject + __init__) | T3, T4 | B | Pendente |
| T6 [P] (teste identidade) | T5 | C | Pendente |
| T7 [P] (test_api.py patches) | T5 | D | Pendente |
| T8 [P] (test_api_query_endpoint patches) | T5 | D | Pendente |
| T9 [P] (test_modo_seguro patches) | T5 | D | Pendente |
| T10 [P] (history_endpoint + audit patches) | T5 | D | Pendente |
| T11 (gate final) | T6..T10 | E | Pendente |

Nota: T1 e T2 nao dependem entre si (serving/health.py e serving/api.py sao independentes). T3 depende de T1, T4 depende de T2. T3 e T4 sao paralelizaveis.

## Test Co-location Validation

| Task | Layer criado/modificado | Matriz exige | Task diz | Status |
|------|------------------------|--------------|----------|--------|
| T1 | serving/health.py | unit | validado por T6 | @validado-adiante |
| T2 | serving/api.py | unit | validado por T6 | @validado-adiante |
| T3 | health.py (shim) | unit | validado por T6 + test_health.py | @validado-adiante |
| T4 | api.py (shim) | unit | validado por T6 + T7..T10 | @validado-adiante |
| T5 | serving/__init__.py + pyproject | unit | validado por T6 | @validado-adiante |
| T6 | test_rag_serving_fase5.py (novo) | unit/integration | unit/integration | @ok |
| T7..T10 | testes migrados | unit | unit | @ok |

## Notas de teste / gate (de CLAUDE.md)
- Sempre mockar LLM; sempre testar retrieval vazio.
- Nomes de teste com prefixo `teste_N_` (dominio) / `test_` (erro/seguranca).
- Gate autoritativo: CI Linux `pytest --cov=agenticlog --cov-report=term-missing -v`.
- Commits Conventional em PT (`refactor:`); branch ja criada (`feature/rag-serving-fase5`); code-review antes do merge.
- `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings PT (Entrada/Saida/Lanca); imutabilidade; funcoes <50 ln.
- Nao usar `git add -A` (evita incluir PDFs untracked de data/documents/). Usar `git add <caminhos explicitos>`.
