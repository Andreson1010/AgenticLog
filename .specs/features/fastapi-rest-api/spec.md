# FastAPI REST API — Technical Spec

**Path:** `.specs/features/fastapi-rest-api/spec.md`
**TLC scope:** large
**Based on story:** Como sistema cliente, quero enviar consulta via `POST /query` e receber JSON normalizado com o resultado do workflow do agente.
**Status:** Awaiting human approval

---

## Problem Statement

AgenticLog's agent workflow (`agent_workflow.invoke()`) is currently called directly from `app.py`, tightly coupling the Streamlit UI to the agent logic. This prevents alternative frontends, integrators, or automated pipelines from consuming the agent without embedding the full Python stack. A thin FastAPI layer decouples the transport protocol from the business logic while fixing two known reliability risks: non-atomic singleton initialization and blocking sync calls on the async event loop.

## Goals

- [ ] `POST /query` returns HTTP 200 with normalized JSON payload on success
- [ ] All 10 acceptance criteria (AC-API-01 through AC-API-10) pass in acceptance tests
- [ ] Singletons initialized once at server startup, not per-request
- [ ] Blocking `agent_workflow.invoke()` runs in a threadpool, never blocking the event loop
- [ ] Unit test coverage for `src/agenticlog/api.py` >= 80%

## Out of Scope

| Feature | Reason |
|---------|--------|
| `POST /ingest` endpoint | Separate feature; ingestion flow is CLI-based |
| `GET /health` endpoint | Explicitly excluded from approved story |
| Authentication / API keys | Not in approved story |
| CORS configuration | Not in approved story |
| Rate limiting | Not in approved story |
| Multiple Uvicorn workers | Not in approved story |
| Streaming responses | Not in approved story |
| Migration of `app.py` to HTTP client | Separate story; `app.py` remains calling agent directly |

---

## User Stories

### P1: Query via REST ⭐ MVP

**User Story**: As a client system (Streamlit or any HTTP integrator), I want to send a natural language query via `POST /query`, so that I receive a normalized JSON response with the agent's ranked answer, confidence score, routing decision, and retrieved documents.

**Why P1**: This is the entire scope of the approved story — there is no P2.

**Acceptance Criteria**:

1. WHEN `POST /query` receives `{"query": "logistics question"}` THEN system SHALL return HTTP 200 with body containing `ranked_response` (string), `confidence_score` (float 0.0–1.0), `next_step` (string), `retrieved_info` (array of objects with `page_content` and `metadata`). [AC-API-01]
2. WHEN `ranked_response` in `AgentState` is a dict of form `{"answer": "..."}` THEN system SHALL extract the `"answer"` value and serialize it as a plain string. [AC-API-02]
3. WHEN `retrieved_info` contains LangChain `Document` objects THEN system SHALL serialize only `page_content` and `metadata` fields per document. [AC-API-03]
4. WHEN `agent_workflow.invoke()` raises `LMStudioUnavailableError` or `httpx.ConnectError` THEN system SHALL return HTTP 503 with `{"detail": "LMStudio indisponível. Inicie o servidor e carregue o modelo."}`. [AC-API-04]
5. WHEN the vectordb directory does not exist at startup THEN system SHALL return HTTP 503 with `{"detail": "Base vetorial não encontrada. Execute: python -m agenticlog.rag"}`. [AC-API-05]
6. WHEN request body is absent or malformed JSON THEN system SHALL return HTTP 422 (FastAPI/Pydantic default behavior). [AC-API-06]
7. WHEN `query` field is empty string or contains only whitespace THEN system SHALL return HTTP 422 with validation error. [AC-API-07]
8. WHEN an unexpected exception occurs during workflow invocation THEN system SHALL return HTTP 500 with `{"detail": "Erro interno do servidor."}` without exposing stack trace. [AC-API-08]
9. WHEN `POST /query` is handled THEN `agent_workflow.invoke()` SHALL execute in a separate thread via `asyncio.get_event_loop().run_in_executor()` or `asyncio.to_thread()`, never blocking the async event loop. [AC-API-09]
10. WHEN the FastAPI server starts THEN singletons (LLM client, ChromaDB, embedding model) SHALL be initialized once in the lifespan context manager, not on first request. [AC-API-10]

**Independent Test**: Run `pytest tests/acceptance/test_api_query_endpoint.py -v` — all AC-API-01 through AC-API-10 assertions pass with LMStudio mocked at HTTP level.

---

## Edge Cases

- WHEN `confidence_score` in `AgentState` is `None` THEN system SHALL normalize to `0.0` before serializing.
- WHEN `retrieved_info` is an empty list THEN system SHALL return `[]` in JSON, never `null`.
- WHEN LMStudio does not respond within 60 seconds THEN system SHALL treat as `LMStudioUnavailableError` and return HTTP 503.
- WHEN `ranked_response` is a dict but does not contain key `"answer"` THEN system SHALL serialize the dict as a JSON string (safe fallback).

---

## Requirement Traceability

| Requirement ID | Acceptance Criterion | Phase | Status |
|----------------|----------------------|-------|--------|
| API-01 | AC-API-01 — 200 response shape | Design | Pending |
| API-02 | AC-API-02 — ranked_response dict normalization | Design | Pending |
| API-03 | AC-API-03 — retrieved_info Document serialization | Design | Pending |
| API-04 | AC-API-04 — LMStudio unavailable → 503 | Design | Pending |
| API-05 | AC-API-05 — Vectordb missing → 503 | Design | Pending |
| API-06 | AC-API-06 — Malformed body → 422 | Design | Pending |
| API-07 | AC-API-07 — Empty query → 422 | Design | Pending |
| API-08 | AC-API-08 — Unexpected exception → 500 no stack trace | Design | Pending |
| API-09 | AC-API-09 — Threadpool execution | Design | Pending |
| API-10 | AC-API-10 — Lifespan singleton init | Design | Pending |

**ID format:** `API-[NUMBER]`

---

## Data Model Changes

No database schema or ChromaDB collection changes. Two new Pydantic models added in `src/agenticlog/api.py`:

**Request:**
```
QueryRequest:
  query: str  (min_length=1, strip_whitespace=True)
```

**Response:**
```
QueryResponse:
  ranked_response: str
  confidence_score: float
  next_step: str
  retrieved_info: list[DocumentInfo]

DocumentInfo:
  page_content: str
  metadata: dict
```

No migrations required.

---

## Process / Background Flow

**Happy path:**
1. Client sends `POST /query` with `{"query": "..."}`.
2. FastAPI validates body with `QueryRequest` (Pydantic); empty/whitespace query fails at validation → 422.
3. Endpoint calls `asyncio.to_thread(agent_workflow.invoke, AgentState(query=query))`.
4. Worker thread runs LangGraph FSM; returns `AgentState`.
5. Normalization layer: `ranked_response` dict → string; `confidence_score` None → 0.0; `Document` objects → `DocumentInfo`.
6. Returns `QueryResponse` serialized as JSON with HTTP 200.

**Failure path — LMStudio unavailable:**
1. `agent_workflow.invoke()` raises `LMStudioUnavailableError` or `httpx.ConnectError`.
2. Exception handler catches, logs `logger.error(...)`, returns HTTP 503 with standard detail message.

**Failure path — Vectordb missing:**
1. Lifespan startup detects `DIR_VECTORDB` does not exist.
2. Server logs critical error; sets `vectordb_ready = False` in app state.
3. All `POST /query` requests return HTTP 503 with vectordb detail message.

**Failure path — Unexpected exception:**
1. Any other exception propagates from `agent_workflow.invoke()`.
2. Global exception handler catches, logs full traceback at `logger.exception(...)`, returns HTTP 500 with generic message (no stack trace in response body).

---

## API Changes

### `POST /query`

**Module:** `src/agenticlog/api.py`  
**Entry point:** `main_api.py` (project root) — starts Uvicorn

**Request:**
```
POST /query
Content-Type: application/json

{"query": "Qual é o prazo de entrega para a rota SP-RJ?"}
```

**Response 200:**
```json
{
  "ranked_response": "O prazo médio para a rota SP-RJ é de 2 dias úteis.",
  "confidence_score": 0.87,
  "next_step": "retrieve",
  "retrieved_info": [
    {
      "page_content": "Rota SP-RJ: prazo 2 dias úteis...",
      "metadata": {"source": "logistics_routes.json"}
    }
  ]
}
```

**Response 422** (empty query or malformed body): FastAPI default Pydantic error shape.

**Response 503** (LMStudio or vectordb unavailable):
```json
{"detail": "LMStudio indisponível. Inicie o servidor e carregue o modelo."}
```

**Response 500** (unexpected):
```json
{"detail": "Erro interno do servidor."}
```

No changes to existing endpoints (none existed).

---

## Frontend Changes

No changes to `app.py` or Streamlit UI in this story. The migration of `app.py` to consume the REST API is out of scope.

---

## Tests Required

**Unit tests** (`tests/test_api.py`):
- Happy path: mock `agent_workflow.invoke` returning valid `AgentState` → assert 200 + response shape.
- `ranked_response` is dict → assert string normalization in response.
- `confidence_score` is `None` → assert `0.0` in response.
- `retrieved_info` contains mock `Document` objects → assert only `page_content` + `metadata` serialized.
- Empty `query` → assert 422.
- Whitespace-only `query` → assert 422.
- `LMStudioUnavailableError` raised → assert 503 + correct detail.
- `httpx.ConnectError` raised → assert 503 + correct detail.
- Vectordb missing at startup → assert 503 on all requests.
- Unexpected `RuntimeError` → assert 500 + generic detail, no stack trace in body.

**Acceptance tests** (`tests/acceptance/test_api_query_endpoint.py`):
- One test function per AC-API-01 through AC-API-10, named `test_ac_api_NN_description`.

**Existing tests that will break:**
- `tests/test_app_error_handler.py` — error classification logic moves to `api.py`; test file must be updated to import from new module (out of scope for this task but flagged).
- `tests/test_streamlit_ui.py` — unaffected in this story (app.py migration is out of scope).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/api.py` | **Create** | FastAPI app, lifespan, endpoint, normalization, error handlers |
| `src/agenticlog/config.py` | **Modify** | Add `API_HOST`, `API_PORT` constants |
| `main_api.py` | **Create** | Uvicorn entry point (`uvicorn src.agenticlog.api:app`) |
| `tests/test_api.py` | **Create** | Unit tests for api.py |
| `tests/acceptance/test_api_query_endpoint.py` | **Create** | Acceptance tests AC-API-01 to AC-API-10 |

---

## Risks

| Risk | Severity | Source | Mitigation |
|------|----------|--------|------------|
| Non-atomic singleton init (race condition on concurrent startup) | HIGH | CONCERNS.md + researcher | Initialize in FastAPI lifespan startup (single-threaded); never re-init per request |
| `agent_workflow.invoke()` blocks event loop | HIGH | Researcher finding | `asyncio.to_thread()` wraps all sync invocations |
| `ranked_response` arrives as dict, not str | MEDIUM | Researcher / AgentState warning | Normalization function in api.py before building `QueryResponse` |
| LangChain `Document` not JSON-serializable | MEDIUM | Researcher finding | Extract `.page_content` + `.metadata` explicitly |
| LMStudio single point of failure (no retry) | HIGH | CONCERNS.md | Catch `LMStudioUnavailableError` + `httpx.ConnectError`; 503 response; 60s timeout |
| Stack trace exposure in 500 responses | MEDIUM | CLAUDE.md security | Global exception handler logs full traceback but returns only generic message |
| `confidence_score` None breaks float serialization | LOW | Researcher edge case | Normalize `None → 0.0` in normalization layer |
| `requirements.txt` already has fastapi + uvicorn | INFO | CONCERNS.md LOW item | No dependency changes needed |

---

## Open Questions

None.

---

## Success Criteria

- [ ] `pytest tests/test_api.py -v` passes with 0 failures
- [ ] `pytest tests/acceptance/test_api_query_endpoint.py -v` passes all 10 AC tests
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` reports >= 80% coverage
- [ ] `uvicorn main_api:app` starts without error when LMStudio is running
- [ ] `POST /query` with valid payload returns correct JSON shape (manual smoke test)
- [ ] No stack traces appear in HTTP response bodies under any error condition
