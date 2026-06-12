# Query History Audit Logging — Design

**Path:** `.specs/features/query-history-audit-logging/design.md`
**TLC scope:** large
**Spec:** `.specs/features/query-history-audit-logging/spec.md`
**Status:** Awaiting human approval

---

## Architecture Overview

The feature adds one new module (`history.py`) and modifies two existing files (`config.py`, `api.py`). No new dependencies are introduced — `sqlite3` is Python stdlib.

```
POST /query
     │
     ▼
consultar() [api.py]
     │
     ├─► asyncio.to_thread(agent_workflow.invoke, ...) ──► AgentState
     │
     ├─► _normalizar_estado(estado) ──────────────────────► QueryResponse
     │
     ├─► _construir_registro(query, response) ────────────► dict (registro)
     │
     ├─► asyncio.to_thread(history_store.append, registro)
     │        │
     │        ▼
     │   HistoryStore [history.py]
     │        │  threading.Lock serializes writes
     │        ▼
     │   SQLite: data/history/history.db
     │        table: query_history
     │
     │   (exception → logger.error, swallowed)
     │
     └─► HTTP 200 QueryResponse


GET /history?limit=N
     │
     ▼
listar_historico() [api.py]
     │
     ├─► asyncio.to_thread(history_store.read_all, limit)
     │        │
     │        ▼
     │   HistoryStore.read_all(limit)
     │        │  SELECT ... ORDER BY timestamp DESC LIMIT ?
     │        ▼
     │   list[dict]
     │
     ├─► list[HistoryEntry]
     │
     └─► HTTP 200 list[HistoryEntry]
         (exception → HTTP 503)
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | SQLite via `sqlite3` (stdlib) | No new dependency; persists across restarts; ACID-safe single-file |
| Concurrency | `threading.Lock` in `HistoryStore` | FastAPI uses a thread-pool for sync work via `asyncio.to_thread`; sqlite3 connections are not thread-safe by default |
| Async bridge | `asyncio.to_thread()` | All SQLite I/O is blocking; consistent with existing pattern for `agent_workflow.invoke` |
| Singleton pattern | `app.state.history_store` | Consistent with `app.state.vectordb_pronto` already in api.py |
| Eviction | DELETE oldest row when count = MAX | O(1) on rowid index; simpler than circular buffer |
| Write isolation | `try/except` in `consultar()` | Write failure must never affect HTTP response to client |

---

## Components

### 1. `src/agenticlog/history.py` (new)

**Responsibility:** All SQLite interaction. Zero FastAPI or LangGraph dependencies.

```
HistoryStore
├── __init__(db_path: Path, max_entries: int) -> None
│       Creates DIR if needed. Calls init_db(). Initializes threading.Lock.
├── init_db() -> None
│       CREATE TABLE IF NOT EXISTS query_history (...)
│       Called from __init__. Safe to call multiple times.
├── append(registro: dict) -> None
│       Acquires lock. Checks count vs max_entries.
│       If count >= max_entries: DELETE oldest (MIN rowid).
│       INSERT record. Releases lock.
│       Raises on any sqlite3 error (caller catches).
└── read_all(limit: int | None = None) -> list[dict]
        SELECT id, timestamp, query, next_step, confidence_score, ranked_response
        FROM query_history ORDER BY timestamp DESC [LIMIT ?]
        Returns list of dicts. No lock needed for reads (sqlite3 WAL not needed at this scale).
```

**Full public interface (type-annotated):**

```python
import sqlite3
import threading
from pathlib import Path


class HistoryStore:
    def __init__(self, db_path: Path, max_entries: int) -> None: ...
    def init_db(self) -> None: ...
    def append(self, registro: dict) -> None: ...
    def read_all(self, limit: int | None = None) -> list[dict]: ...
```

**`registro` dict contract (keys):**

| Key | Python type | SQLite column |
|-----|-------------|--------------|
| `timestamp` | `str` | `timestamp TEXT` |
| `query` | `str` | `query TEXT` |
| `next_step` | `str` | `next_step TEXT` |
| `confidence_score` | `float` | `confidence_score REAL` |
| `ranked_response` | `str` | `ranked_response TEXT` |

The caller (`api.py`) is responsible for normalizing `confidence_score` and `ranked_response` before passing to `append()`.

---

### 2. `src/agenticlog/config.py` (modified)

Add three constants after the `API_PORT` block, before the `Logging` block:

```python
# History Audit Log
DIR_HISTORY: Path = PROJECT_ROOT / "data" / "history"   # diretório onde o SQLite é persistido
HISTORY_FILE: Path = DIR_HISTORY / "history.db"          # arquivo SQLite do audit log
HISTORY_MAX_ENTRIES: int = int(os.environ.get("HISTORY_MAX_ENTRIES", "1000"))  # máximo de registros antes de evictar o mais antigo
if HISTORY_MAX_ENTRIES <= 0:
    raise ValueError(
        f"HISTORY_MAX_ENTRIES={HISTORY_MAX_ENTRIES!r} must be > 0."
    )
```

Pattern follows the existing `LOG_LEVEL` validation block exactly.

---

### 3. `src/agenticlog/api.py` (modified)

#### 3a. New imports

```python
from agenticlog.config import DEFAULT_COLLECTION_NAME, DIR_VECTORDB, HISTORY_FILE, HISTORY_MAX_ENTRIES
from agenticlog.history import HistoryStore
```

#### 3b. New Pydantic models

```python
class HistoryEntry(BaseModel):
    """Entrada individual do histórico de queries."""
    id: int
    timestamp: str
    query: str
    next_step: str
    confidence_score: float
    ranked_response: str
```

#### 3c. New helper (pure function, no I/O)

```python
def _construir_registro(query: str, response: QueryResponse) -> dict:
    """Constrói o dict de registro para persistência no histórico.

    Entrada: query original (str), response normalizada (QueryResponse).
    Saída: dict com chaves timestamp, query, next_step, confidence_score, ranked_response.
    """
    return {
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "query": query,
        "next_step": response.next_step,
        "confidence_score": response.confidence_score,
        "ranked_response": response.ranked_response,
    }
```

Note: `datetime` is not currently imported in `api.py`. Add `import datetime` to imports.

#### 3d. Lifespan modification

```python
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing vectordb + inicializar_recursos() block unchanged ...
    app.state.history_store = HistoryStore(db_path=HISTORY_FILE, max_entries=HISTORY_MAX_ENTRIES)
    logger.info("HistoryStore inicializado em %s", HISTORY_FILE)
    yield
```

`HistoryStore.__init__` never raises in normal conditions (directory creation is `exist_ok=True`). If it does raise (e.g., permission error), it surfaces as a startup failure — acceptable behavior.

#### 3e. `consultar()` modification

```python
@app.post("/query", response_model=QueryResponse)
async def consultar(body: QueryRequest, req: Request) -> QueryResponse:
    if not req.app.state.vectordb_pronto:
        raise HTTPException(status_code=503, detail=MSG_VECTORDB_AUSENTE)

    estado = await asyncio.to_thread(
        agent_workflow.invoke,
        AgentState(query=body.query),
    )
    response = _normalizar_estado(estado)

    # Audit write — falha nunca propaga para o cliente
    registro = _construir_registro(body.query, response)
    try:
        await asyncio.to_thread(req.app.state.history_store.append, registro)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao gravar histórico: %s", exc)

    return response
```

#### 3f. New endpoint

```python
@app.get("/history", response_model=list[HistoryEntry])
async def listar_historico(
    req: Request,
    limit: int | None = Query(default=None, ge=1, description="Número máximo de registros a retornar"),
) -> list[HistoryEntry]:
    """Retorna o histórico de queries em ordem decrescente de timestamp.

    Entrada: limit (opcional, >= 1) — limita número de registros.
    Saída: lista de HistoryEntry ordenada por timestamp DESC.

    Errors:
      422 — limit <= 0 (Pydantic/FastAPI Query validation).
      503 — store indisponível.
    """
    try:
        registros = await asyncio.to_thread(
            req.app.state.history_store.read_all, limit
        )
    except Exception as exc:
        logger.error("Falha ao ler histórico: %s", exc)
        raise HTTPException(status_code=503, detail="Histórico indisponível.") from exc
    return [HistoryEntry(**r) for r in registros]
```

Additional import needed: `from fastapi import FastAPI, HTTPException, Query, Request`

---

## SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS query_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    next_step        TEXT    NOT NULL,
    confidence_score REAL    NOT NULL,
    ranked_response  TEXT    NOT NULL
);
```

No indexes beyond primary key. At `HISTORY_MAX_ENTRIES=1000` the table will never exceed 1000 rows; full scan for reads is O(1000) — negligible.

Eviction query:
```sql
DELETE FROM query_history WHERE id = (SELECT MIN(id) FROM query_history);
```

Count query (used before eviction check):
```sql
SELECT COUNT(*) FROM query_history;
```

Read query:
```sql
-- Without limit
SELECT id, timestamp, query, next_step, confidence_score, ranked_response
FROM query_history
ORDER BY timestamp DESC;

-- With limit
SELECT id, timestamp, query, next_step, confidence_score, ranked_response
FROM query_history
ORDER BY timestamp DESC
LIMIT ?;
```

---

## Concurrency Model

```
Thread A (asyncio.to_thread)     Thread B (asyncio.to_thread)
         │                                │
         ▼                                ▼
  lock.acquire() ◄──── blocks ────────── lock.acquire()
         │                                │
  sqlite3 INSERT                    (waiting...)
         │                                │
  lock.release() ─────────────────────► lock.acquire()
                                          │
                                    sqlite3 INSERT
                                          │
                                    lock.release()
```

Single `threading.Lock` instance lives on the `HistoryStore` object. Since there is one singleton per API process, this is safe across all concurrent requests.

`read_all()` does not acquire the lock. SQLite's default serialized mode handles concurrent reads safely. At the scale of this application (local deployment, one process), this is sufficient.

---

## Reuse from Codebase

| Pattern | Source | Applied here |
|---------|--------|--------------|
| `asyncio.to_thread()` for blocking I/O | `api.py` line 221 (`agent_workflow.invoke`) | `history_store.append()` and `read_all()` |
| `app.state.*` singleton | `api.py` line 115 (`app.state.vectordb_pronto`) | `app.state.history_store` |
| `try/except` swallowing non-critical failure | `agent.py` (`usar_ferramenta_web`) | write failure in `consultar()` |
| `ValueError` at config import | `config.py` lines 77-80 (`LOG_LEVEL`) | `HISTORY_MAX_ENTRIES <= 0` |
| `datetime.datetime.now(tz=datetime.UTC).isoformat()` | `config.py` `_JsonFormatter.format()` | `_construir_registro()` timestamp |
| Pydantic response model → pure helper → `@app.get` | `api.py` pattern for `QueryResponse` | `HistoryEntry` → `_construir_registro` → `listar_historico` |
| `tempfile.TemporaryDirectory()` for test isolation | `tests/test_rag.py` | `tests/test_history.py` |

---

## CONCERNS.md Risk Mitigations

| Concern | Mitigation in this feature |
|---------|---------------------------|
| MEDIUM — Incomplete Test Coverage | New `tests/test_history.py` (10 unit tests) + `tests/acceptance/test_history_endpoint.py` (8 ACs) target >= 80% on `history.py` |
| MEDIUM — No Logging Module | `history.py` uses `logging.getLogger(__name__)` — no `print()` statements |
| HIGH — Missing Error Handling at Startup | `HistoryStore` init in lifespan is guarded; failure surfaced via `logger.critical`, not crash |

---

## File Structure After Feature

```
src/agenticlog/
├── config.py          (modified: +3 constants +1 validation)
├── history.py         (new: HistoryStore)
├── api.py             (modified: +imports, +HistoryStore init, +write in consultar, +GET /history)
├── agent.py           (unchanged)
└── rag.py             (unchanged)

data/
├── documents/         (unchanged)
├── vectordb/          (unchanged)
└── history/           (new dir, created by HistoryStore.__init__)
    └── history.db     (new file, created by init_db() on first write)

tests/
├── test_history.py                          (new)
├── test_api.py                              (modified: +mock history_store)
└── acceptance/
    ├── test_api_query_endpoint.py           (modified: +mock history_store)
    └── test_history_endpoint.py             (new)
```
