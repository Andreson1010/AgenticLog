# FastAPI REST API — Design

**Path:** `.specs/features/fastapi-rest-api/design.md`
**Spec:** `.specs/features/fastapi-rest-api/spec.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## Architecture Overview

A new thin FastAPI application (`src/agenticlog/api.py`) sits in front of the existing LangGraph workflow. It adds a transport layer without touching agent logic, RAG, or the Streamlit UI.

```
HTTP Client
    │
    ▼
main_api.py (Uvicorn entry point)
    │
    ▼
src/agenticlog/api.py
  ├── lifespan()          — singleton init at startup [API-10]
  ├── POST /query         — async endpoint [API-01]
  ├── _normalizar_estado()— response normalization [API-02, API-03]
  ├── _mapear_erro()      — error → HTTP status [API-04, API-05, API-08]
  └── Pydantic models     — QueryRequest, QueryResponse, DocumentInfo [API-06, API-07]
    │
    ▼  asyncio.to_thread() [API-09]
src/agenticlog/agent.py
  └── agent_workflow.invoke(AgentState)
    │
    ▼
LMStudio (http://127.0.0.1:1234/v1)
ChromaDB (data/vectordb/)
```

The existing `app.py` (Streamlit) continues to call `agent_workflow.invoke()` directly. No change is made to `app.py` in this feature.

---

## Module Structure

### `src/agenticlog/api.py` (new, ~160 lines)

```
1. Module docstring (Portuguese)
2. Imports: os, asyncio, logging, contextlib | fastapi, pydantic | agenticlog.config, agent, health
3. Logger: logging.getLogger(__name__)
4. Pydantic models: QueryRequest, DocumentInfo, QueryResponse
5. lifespan() context manager — startup singleton init + vectordb check
6. app = FastAPI(lifespan=lifespan)
7. _normalizar_resposta() — dict→str, None→0.0, Document→DocumentInfo
8. _serializar_documentos() — list[Document|dict] → list[DocumentInfo]
9. POST /query endpoint — async, calls asyncio.to_thread, returns QueryResponse
10. Exception handlers — LMStudioUnavailableError, httpx.ConnectError, Exception
```

### `main_api.py` (new, ~15 lines, project root)

```python
"""Ponto de entrada para o servidor FastAPI do AgenticLog."""
import uvicorn
from agenticlog.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run("agenticlog.api:app", host=API_HOST, port=API_PORT, reload=False)
```

### `src/agenticlog/config.py` (modify — add 2 constants)

```python
API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("API_PORT", "8000"))
```

---

## Component Designs

### 1. Pydantic Models

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, strip_whitespace=True)

class DocumentInfo(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    ranked_response: str
    confidence_score: float
    next_step: str
    retrieved_info: list[DocumentInfo]
```

`min_length=1` with `strip_whitespace=True` handles AC-API-06 and AC-API-07 together via Pydantic's built-in validation without custom code.

### 2. Lifespan Context Manager (AC-API-10)

```python
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _verificar_vectordb()           # raises RuntimeError → logged, app.state.vectordb_pronto = False
    _inicializar_singletons()       # calls existing lazy-init functions in agent.py once
    app.state.vectordb_pronto = True
    yield
    # Shutdown (no-op for now)
```

`_verificar_vectordb()` checks `Path(DIR_VECTORDB).exists()`. If missing, sets flag; does NOT crash the server — requests receive 503 (allows operator to run `python -m agenticlog.rag` while server is up and restart).

`_inicializar_singletons()` calls `agent.inicializar_recursos()` — a new thin public function added to `agent.py` that triggers the existing lazy-init guards once in a controlled, single-threaded context, eliminating the race condition identified in CONCERNS.md.

### 3. Endpoint (AC-API-01, AC-API-09)

```python
@app.post("/query", response_model=QueryResponse)
async def consultar(request: QueryRequest, req: Request) -> QueryResponse:
    if not req.app.state.vectordb_pronto:
        raise HTTPException(503, detail=MSG_VECTORDB_AUSENTE)
    estado = await asyncio.to_thread(
        agent_workflow.invoke,
        AgentState(query=request.query)
    )
    return _normalizar_estado(estado)
```

`asyncio.to_thread()` (Python 3.9+) is idiomatic and avoids the deprecated `loop.run_in_executor(None, ...)` pattern.

### 4. Normalization Layer (AC-API-02, AC-API-03)

```python
def _normalizar_estado(estado: AgentState) -> QueryResponse:
    ranked = estado.ranked_response
    if isinstance(ranked, dict):
        ranked = ranked.get("answer", json.dumps(ranked, ensure_ascii=False))
    score = estado.confidence_score if estado.confidence_score is not None else 0.0
    docs = _serializar_documentos(estado.retrieved_info or [])
    return QueryResponse(
        ranked_response=ranked,
        confidence_score=score,
        next_step=estado.next_step,
        retrieved_info=docs,
    )

def _serializar_documentos(docs: list) -> list[DocumentInfo]:
    resultado = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            resultado.append(DocumentInfo(
                page_content=doc.page_content,
                metadata=doc.metadata or {},
            ))
        elif isinstance(doc, dict):
            resultado.append(DocumentInfo(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {}),
            ))
    return resultado
```

Both functions are pure (no side effects) and testable in isolation.

### 5. Error Handlers (AC-API-04, AC-API-05, AC-API-08)

```python
# Registered with @app.exception_handler(LMStudioUnavailableError)
async def handler_lmstudio(request, exc):
    logger.error("LMStudio indisponível: %s", exc)
    return JSONResponse(status_code=503, content={"detail": MSG_LMSTUDIO_INDISPONIVEL})

# Registered with @app.exception_handler(httpx.ConnectError)
async def handler_connect_error(request, exc):
    logger.error("Erro de conexão com LMStudio: %s", exc)
    return JSONResponse(status_code=503, content={"detail": MSG_LMSTUDIO_INDISPONIVEL})

# Registered with @app.exception_handler(Exception)
async def handler_generico(request, exc):
    logger.exception("Exceção inesperada no endpoint /query")
    return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor."})
```

Constants `MSG_LMSTUDIO_INDISPONIVEL` and `MSG_VECTORDB_AUSENTE` defined at module level in `api.py` (not in `config.py` — these are API-layer messages, not system config).

---

## Reuse from Codebase

| Existing component | Reused as-is | Notes |
|--------------------|-------------|-------|
| `agent_workflow` from `agent.py` | Yes | Called via `asyncio.to_thread` |
| `AgentState` from `agent.py` | Yes | Input model for workflow |
| `LMStudioUnavailableError` from `health.py` | Yes | Caught in exception handler |
| `DIR_VECTORDB` from `config.py` | Yes | Startup vectordb check |
| `logging.getLogger(__name__)` pattern | Yes | Consistent with all modules |
| `API_HOST`, `API_PORT` | New constants | Added to `config.py` |

No new external dependencies. `fastapi`, `uvicorn`, `httpx` are already in `requirements.txt`.

---

## Mitigations for CONCERNS.md Items

| Concern | Severity | Mitigation in this feature |
|---------|----------|---------------------------|
| Missing error handling at startup | HIGH | Lifespan checks vectordb existence; sets flag; returns 503 if not ready |
| LMStudio single point of failure | HIGH | 503 response with actionable message; 60s timeout via `httpx` (existing in health.py) |
| Non-atomic singleton init (race) | HIGH | Lifespan init runs once in startup phase (single-threaded FastAPI startup) |
| Hardcoded LLM credentials | MEDIUM | Out of scope for this story (separate `load-env-credentials` feature) |
| FastAPI/Uvicorn installed but unused | LOW | Resolved — this feature creates the API server |

---

## Sequence Diagram — Happy Path

```
Client          FastAPI         asyncio.to_thread    agent.py        LMStudio
  │                │                   │                 │               │
  │ POST /query    │                   │                 │               │
  │───────────────▶│                   │                 │               │
  │                │ validate(Query)   │                 │               │
  │                │───────────────────▶                 │               │
  │                │                   │ invoke(state)   │               │
  │                │                   │────────────────▶│               │
  │                │                   │                 │ LLM call      │
  │                │                   │                 │──────────────▶│
  │                │                   │                 │◀──────────────│
  │                │                   │◀────────────────│               │
  │                │ normalize(estado) │                 │               │
  │                │───────────────────▶                 │               │
  │ 200 QueryResp  │                   │                 │               │
  │◀───────────────│                   │                 │               │
```

---

## File Size Estimate

| File | Estimated lines |
|------|----------------|
| `src/agenticlog/api.py` | ~160 |
| `main_api.py` | ~15 |
| `tests/test_api.py` | ~200 |
| `tests/acceptance/test_api_query_endpoint.py` | ~150 |

All within the 200–400 line target from coding-style rules.

---

## Open Questions

None.
