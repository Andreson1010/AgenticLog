# FastAPI REST API — Tasks

**Path:** `.specs/features/fastapi-rest-api/tasks.md`
**Spec:** `.specs/features/fastapi-rest-api/spec.md`
**Design:** `.specs/features/fastapi-rest-api/design.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## Overview

8 atomic tasks in dependency order. Each task is independently verifiable. Builder should run `pytest --cov=agenticlog --cov-report=term-missing -v` after Tasks 4–8.

---

## Task 1 — Add API constants to config.py

**Requirement IDs:** API-10 (partial)
**Estimated effort:** XS (< 30 min)

**What to do:**
Add two constants to `src/agenticlog/config.py` in the existing constants block:

```python
API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("API_PORT", "8000"))
```

**Done when:**
- `from agenticlog.config import API_HOST, API_PORT` works in a Python REPL.
- No existing test breaks.

**Dependencies:** None

---

## Task 2 — Add `inicializar_recursos()` to agent.py

**Requirement IDs:** API-10
**Estimated effort:** S (30–60 min)

**What to do:**
Add a single public function to `src/agenticlog/agent.py` that triggers the existing lazy-init guards for `_llm`, `_vector_db`, and `_embedding_model`. This function is called once from the FastAPI lifespan; it eliminates the per-request race condition documented in CONCERNS.md.

Function signature (Portuguese naming convention):

```python
def inicializar_recursos() -> None:
    """
    Inicializa singletons do agente (LLM, ChromaDB, embeddings) na inicialização do servidor.

    Entrada: nenhuma
    Saída: nenhuma — efeito colateral: singletons globais inicializados
    """
```

The function body calls the existing lazy-init logic (the `if _resource is None: _resource = ...` blocks) in the correct order: embeddings → vector_db → llm.

**Done when:**
- `inicializar_recursos()` can be called without error when LMStudio is running and vectordb exists.
- Existing unit tests in `tests/test_agentic_rag.py` still pass (mock LLM calls unchanged).

**Dependencies:** Task 1

---

## Task 3 — Create `src/agenticlog/api.py` (skeleton + models + lifespan)

**Requirement IDs:** API-05, API-06, API-07, API-10
**Estimated effort:** M (1–2 h)

**What to do:**
Create `src/agenticlog/api.py` with:

1. Module docstring in Portuguese.
2. Imports (stdlib → third-party → local).
3. Logger: `logger = logging.getLogger(__name__)`.
4. Message constants:
   - `MSG_LMSTUDIO_INDISPONIVEL = "LMStudio indisponível. Inicie o servidor e carregue o modelo."`
   - `MSG_VECTORDB_AUSENTE = "Base vetorial não encontrada. Execute: python -m agenticlog.rag"`
5. Pydantic models: `QueryRequest`, `DocumentInfo`, `QueryResponse` (see design.md).
6. `_verificar_vectordb()` — checks `Path(DIR_VECTORDB).exists()`; raises `RuntimeError` with `MSG_VECTORDB_AUSENTE` if missing.
7. `lifespan()` async context manager:
   - Startup: calls `_verificar_vectordb()` (catches `RuntimeError`, logs, sets `app.state.vectordb_pronto = False`); calls `inicializar_recursos()` only if vectordb is ready; sets `app.state.vectordb_pronto = True`.
   - Yield.
   - Shutdown: no-op.
8. `app = FastAPI(lifespan=lifespan)`.

Do NOT add the endpoint or normalization yet (Task 4).

**Done when:**
- `from agenticlog.api import app` imports without error.
- `app` has a `lifespan` attribute.
- `QueryRequest(query="  ")` raises `ValidationError` (strip_whitespace + min_length=1).
- `QueryRequest(query="")` raises `ValidationError`.

**Dependencies:** Task 1, Task 2

---

## Task 4 — Add normalization helpers to api.py

**Requirement IDs:** API-02, API-03
**Estimated effort:** S (30–60 min)

**What to do:**
Add two private functions to `src/agenticlog/api.py`:

1. `_serializar_documentos(docs: list) -> list[DocumentInfo]` — handles LangChain `Document` objects and plain dicts; returns empty list for empty input (never `null`).
2. `_normalizar_estado(estado: AgentState) -> QueryResponse` — normalizes `ranked_response` (dict → str), `confidence_score` (None → 0.0), calls `_serializar_documentos`.

Both functions must be pure (no I/O, no side effects).

**Done when:**
- `_normalizar_estado` with a mock `AgentState(ranked_response={"answer": "ok"}, confidence_score=None, next_step="retrieve", retrieved_info=[])` returns `QueryResponse(ranked_response="ok", confidence_score=0.0, next_step="retrieve", retrieved_info=[])`.
- `_serializar_documentos([])` returns `[]`.
- `_serializar_documentos` with a mock `Document(page_content="x", metadata={"k": "v"})` returns `[DocumentInfo(page_content="x", metadata={"k": "v"})]`.

**Dependencies:** Task 3

---

## Task 5 — Add `POST /query` endpoint and exception handlers

**Requirement IDs:** API-01, API-04, API-08, API-09
**Estimated effort:** M (1–2 h)

**What to do:**

1. Add `@app.post("/query", response_model=QueryResponse)` async endpoint:
   - Check `request.app.state.vectordb_pronto`; if False, raise `HTTPException(503, detail=MSG_VECTORDB_AUSENTE)`.
   - Call `await asyncio.to_thread(agent_workflow.invoke, AgentState(query=body.query))`.
   - Return `_normalizar_estado(estado)`.

2. Register exception handlers:
   - `@app.exception_handler(LMStudioUnavailableError)` → HTTP 503 + `MSG_LMSTUDIO_INDISPONIVEL`.
   - `@app.exception_handler(httpx.ConnectError)` → HTTP 503 + `MSG_LMSTUDIO_INDISPONIVEL`.
   - `@app.exception_handler(Exception)` → HTTP 500 + `"Erro interno do servidor."` (log full traceback with `logger.exception()`).

**Done when:**
- FastAPI TestClient: `POST /query` with mocked `agent_workflow.invoke` returning a valid `AgentState` returns HTTP 200 with correct JSON keys.
- FastAPI TestClient: `POST /query` with mock raising `LMStudioUnavailableError` returns HTTP 503.
- FastAPI TestClient: `POST /query` with mock raising `RuntimeError("boom")` returns HTTP 500 with `{"detail": "Erro interno do servidor."}` (no "boom" in response body).

**Dependencies:** Task 4

---

## Task 6 — Create `main_api.py` entry point

**Requirement IDs:** API-10 (operational)
**Estimated effort:** XS (< 30 min)

**What to do:**
Create `main_api.py` at project root:

```python
"""
Ponto de entrada do servidor FastAPI do AgenticLog.

Uso: python main_api.py
"""
import uvicorn
from agenticlog.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "agenticlog.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
```

**Done when:**
- `python main_api.py` starts Uvicorn on port 8000 without error (requires LMStudio running and vectordb populated).
- `python -c "import main_api"` does not execute the server (guarded by `if __name__ == "__main__"`).

**Dependencies:** Task 5

---

## Task 7 — Write unit tests for api.py

**Requirement IDs:** API-01 through API-10 (test coverage)
**Estimated effort:** L (2–3 h)

**What to do:**
Create `tests/test_api.py` using FastAPI `TestClient`. All tests must mock `agent_workflow.invoke` at `agenticlog.api` namespace (not `agenticlog.agent`).

Required test functions (naming convention: `teste_N_description` for domain logic, `test_ac_api_NN_description` for AC-mapped tests):

| Test function | What it asserts |
|--------------|----------------|
| `teste_1_query_valida_retorna_200` | HTTP 200 + all response keys present |
| `teste_2_ranked_response_dict_normalizado` | dict `{"answer": "x"}` → string `"x"` in response |
| `teste_3_confidence_score_none_normalizado` | `None` → `0.0` in response |
| `teste_4_retrieved_info_documentos_serializados` | Document objects → `page_content`+`metadata` only |
| `teste_5_retrieved_info_vazia_retorna_lista` | empty list → `[]`, not `null` |
| `teste_6_query_vazia_retorna_422` | `{"query": ""}` → 422 |
| `teste_7_query_espacos_retorna_422` | `{"query": "   "}` → 422 |
| `teste_8_body_malformado_retorna_422` | missing `query` key → 422 |
| `teste_9_lmstudio_indisponivel_retorna_503` | `LMStudioUnavailableError` → 503 + correct detail |
| `teste_10_connect_error_retorna_503` | `httpx.ConnectError` → 503 + correct detail |
| `teste_11_excecao_generica_retorna_500` | `RuntimeError` → 500 + generic message, no traceback |
| `teste_12_vectordb_ausente_retorna_503` | startup check fails → 503 on all requests |
| `teste_13_workflow_executa_em_thread` | `asyncio.to_thread` called (not direct invoke) |

**Done when:**
- `pytest tests/test_api.py -v` passes with 0 failures.
- All mocks are applied at `agenticlog.api` namespace.
- No real LMStudio or ChromaDB calls are made.

**Dependencies:** Task 5

---

## Task 8 — Write acceptance tests

**Requirement IDs:** API-01 through API-10
**Estimated effort:** M (1–2 h)

**What to do:**
Create `tests/acceptance/test_api_query_endpoint.py`. One test per acceptance criterion, named `test_ac_api_NN_description`.

| Test function | AC | Assertion |
|--------------|-----|-----------|
| `test_ac_api_01_response_shape` | AC-API-01 | 200 + ranked_response str, confidence_score float, next_step str, retrieved_info list |
| `test_ac_api_02_ranked_response_dict_normalization` | AC-API-02 | dict ranked_response → string |
| `test_ac_api_03_retrieved_info_document_serialization` | AC-API-03 | only page_content + metadata in each item |
| `test_ac_api_04_lmstudio_unavailable_503` | AC-API-04 | LMStudioUnavailableError → 503 + exact detail string |
| `test_ac_api_05_vectordb_missing_503` | AC-API-05 | vectordb flag False → 503 + exact detail string |
| `test_ac_api_06_malformed_body_422` | AC-API-06 | missing query field → 422 |
| `test_ac_api_07_empty_query_422` | AC-API-07 | empty/whitespace query → 422 |
| `test_ac_api_08_unexpected_exception_500_no_stacktrace` | AC-API-08 | RuntimeError → 500, "traceback" not in response body |
| `test_ac_api_09_workflow_in_threadpool` | AC-API-09 | asyncio.to_thread called with agent_workflow.invoke |
| `test_ac_api_10_singletons_initialized_at_startup` | AC-API-10 | inicializar_recursos called exactly once during lifespan startup |

**Done when:**
- `pytest tests/acceptance/test_api_query_endpoint.py -v` passes all 10 tests.
- Each test description clearly maps to its AC ID in a comment or docstring.

**Dependencies:** Task 7

---

## Task 9 — Coverage gate + smoke test

**Requirement IDs:** All
**Estimated effort:** XS (< 30 min)

**What to do:**

1. Run `pytest --cov=agenticlog --cov-report=term-missing -v` and confirm >= 80% coverage for `agenticlog/api.py`.
2. If coverage is below 80%, add missing test cases to `tests/test_api.py`.
3. Manual smoke test (requires LMStudio running):
   - `python main_api.py` — server starts.
   - `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"Qual é o prazo de entrega SP-RJ?\"}"` — returns 200 with correct JSON shape.

**Done when:**
- Coverage report shows `agenticlog/api.py` >= 80%.
- Smoke test returns HTTP 200 with all four expected keys.
- All existing tests (`test_agentic_rag.py`, `test_rag.py`) still pass.

**Dependencies:** Task 8

---

## Dependency Graph

```
Task 1 (config)
    └── Task 2 (agent.py inicializar_recursos)
            └── Task 3 (api.py skeleton + lifespan)
                    └── Task 4 (normalization helpers)
                            └── Task 5 (endpoint + error handlers)
                                    ├── Task 6 (main_api.py entry point)
                                    └── Task 7 (unit tests)
                                            └── Task 8 (acceptance tests)
                                                    └── Task 9 (coverage gate + smoke)
```

---

## Test Gate Summary

| Gate | Command | Pass condition |
|------|---------|----------------|
| Unit tests | `pytest tests/test_api.py -v` | 0 failures |
| Acceptance tests | `pytest tests/acceptance/test_api_query_endpoint.py -v` | 10/10 pass |
| Full suite + coverage | `pytest --cov=agenticlog --cov-report=term-missing -v` | >= 80% |
| Smoke test | `curl POST /query` with running LMStudio | HTTP 200 + correct shape |
