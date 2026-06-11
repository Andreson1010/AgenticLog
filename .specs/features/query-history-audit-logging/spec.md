# Query History Audit Logging — Technical Spec

**Path:** `.specs/features/query-history-audit-logging/spec.md`
**TLC scope:** large
**Based on story:** As a logistics operator, I want every query recorded with timestamp, route, confidence score, and response, and to retrieve that history via API, so that I can audit what the system answered and how it routed each request.
**Status:** Awaiting human approval

---

## Problem Statement

AgenticLog currently processes queries and returns answers but does not persist any record of what was asked, how it was routed, or what was returned. Logistics operators cannot audit past interactions or verify routing decisions, which is required for operational trust. This feature adds a lightweight SQLite-backed audit log that records every successful query invocation and exposes it via a GET endpoint.

## Goals

- [ ] Every POST /query response is preceded by a durable write of the query record to SQLite before the HTTP response is sent to the client.
- [ ] GET /history returns all persisted records sorted by timestamp DESC, with optional `limit` parameter.
- [ ] Write failures are isolated from the query response: the client always receives HTTP 200 if the agent succeeds.
- [ ] The history store survives API restarts (SQLite file on disk).
- [ ] Concurrent writes are safe (threading.Lock serializes all SQLite writes).

## Out of Scope

| Feature | Reason |
|---------|--------|
| Auth on GET /history | Explicitly accepted: local-only deployment |
| Pagination with offset | Approved story excludes it |
| Search / filter on history | Approved story excludes it |
| Export CSV / log rotation | Approved story excludes it |
| UI in Streamlit | Approved story excludes it |
| Delete individual records | Approved story excludes it |
| Sensitive data scrubbing | Approved story excludes it |
| retrieved_info in records | Approved story excludes it |

---

## User Stories

### P1: Record query on every successful POST /query ⭐ MVP

**User Story:** As a logistics operator, I want each query I submit to be automatically saved with its timestamp, route, confidence score, and final answer, so that I have a permanent audit trail.

**Why P1:** Core audit requirement; all other stories depend on stored records.

**Acceptance Criteria:**
1. WHEN POST /query succeeds THEN system SHALL write a record containing `query` (str), `timestamp` (ISO 8601 UTC), `next_step` (str), `confidence_score` (float), `ranked_response` (str) to the SQLite store before returning HTTP 200.
2. WHEN the write to SQLite fails THEN system SHALL still return HTTP 200 to the client AND log the failure at ERROR level.
3. WHEN `confidence_score` is None THEN system SHALL store `0.0`.
4. WHEN `ranked_response` is a dict THEN system SHALL store its normalized string representation (same logic as `_normalizar_estado`).
5. WHEN `HISTORY_MAX_ENTRIES` is reached THEN system SHALL evict the oldest row (DELETE WHERE rowid = MIN(rowid)) before inserting.
6. WHEN two requests arrive at the same millisecond THEN system SHALL store both records (unique rowid).

**Independent Test:** POST two queries; assert history count = 2 and both timestamps are present.

---

### P2: Retrieve history via GET /history ⭐ MVP

**User Story:** As a logistics operator, I want to call GET /history and receive an ordered list of past queries, so that I can review what was asked and how the system responded.

**Why P2:** Retrieval is the audit access path; without it stored records are inaccessible.

**Acceptance Criteria:**
1. WHEN GET /history is called with records present THEN system SHALL return HTTP 200 with a JSON array sorted by timestamp DESC.
2. WHEN GET /history is called with no records THEN system SHALL return HTTP 200 with `[]`.
3. WHEN GET /history is called with `?limit=N` (N > 0) THEN system SHALL return the N most recent records.
4. WHEN GET /history is called without `limit` THEN system SHALL return all records (bounded by HISTORY_MAX_ENTRIES).
5. WHEN `limit` <= 0 THEN system SHALL return HTTP 422.
6. WHEN history store is unavailable THEN system SHALL return HTTP 503.
7. WHEN `limit` > total records THEN system SHALL return all records without error.

**Independent Test:** Insert 5 records; GET /history?limit=3 returns exactly 3, sorted DESC.

---

### P3: Config constants and store initialization ⭐ MVP

**User Story:** As a developer, I want the history store initialized automatically at API startup with validated config constants, so that no manual setup is required and misconfiguration fails early.

**Why P3:** Prerequisite for P1 and P2; ensures no runtime surprises.

**Acceptance Criteria:**
1. WHEN `HISTORY_MAX_ENTRIES` <= 0 THEN config.py import SHALL raise `ValueError`.
2. WHEN the SQLite file does not exist on first write THEN system SHALL create the file and table automatically.
3. WHEN the API restarts THEN records written in prior sessions SHALL still be present.
4. WHEN lifespan runs THEN system SHALL initialize `HistoryStore` and assign it to `app.state.history_store` as a singleton.

**Independent Test:** Delete SQLite file; POST /query; assert file created and record present.

---

## Edge Cases

- WHEN `HISTORY_MAX_ENTRIES = 0` THEN config.py SHALL raise `ValueError` at import time, before the API starts.
- WHEN `limit` is exactly equal to total record count THEN system SHALL return all records (not an error).
- WHEN `ranked_response` arrives as `{"answer": "text"}` THEN system SHALL store `"text"` (string, not JSON).
- WHEN `ranked_response` arrives as dict without `"answer"` key THEN system SHALL store `json.dumps(dict)`.
- WHEN `confidence_score` is `None` THEN system SHALL store `0.0`.
- WHEN two writes occur concurrently THEN `threading.Lock` SHALL serialize them; both records SHALL be persisted.
- WHEN SQLite DB file exists but table is missing THEN `init_db()` SHALL create the table (idempotent DDL with `CREATE TABLE IF NOT EXISTS`).

---

## Requirement Traceability

| Requirement ID | Story | AC # | Phase | Status |
|----------------|-------|-------|-------|--------|
| HIST-01 | P1 | 1 | Design + Tasks | Pending |
| HIST-02 | P1 | 2 | Design + Tasks | Pending |
| HIST-03 | P1 | 3 | Design + Tasks | Pending |
| HIST-04 | P1 | 4 | Design + Tasks | Pending |
| HIST-05 | P1 | 5 | Design + Tasks | Pending |
| HIST-06 | P1 | 6 | Design + Tasks | Pending |
| HIST-07 | P2 | 1 | Design + Tasks | Pending |
| HIST-08 | P2 | 2 | Design + Tasks | Pending |
| HIST-09 | P2 | 3 | Design + Tasks | Pending |
| HIST-10 | P2 | 4 | Design + Tasks | Pending |
| HIST-11 | P2 | 5 | Design + Tasks | Pending |
| HIST-12 | P2 | 6 | Design + Tasks | Pending |
| HIST-13 | P2 | 7 | Design + Tasks | Pending |
| HIST-14 | P3 | 1 | Tasks | Pending |
| HIST-15 | P3 | 2 | Tasks | Pending |
| HIST-16 | P3 | 3 | Tasks | Pending |
| HIST-17 | P3 | 4 | Tasks | Pending |

**ID format:** `HIST-[NUMBER]`

---

## Data Model Changes

### New: `data/history/history.db` (SQLite)

Table: `query_history`

| Column | SQLite type | Notes |
|--------|------------|-------|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Internal rowid alias |
| `timestamp` | TEXT NOT NULL | ISO 8601 UTC, e.g. `2025-01-15T10:30:00.123456+00:00` |
| `query` | TEXT NOT NULL | Raw user query string |
| `next_step` | TEXT NOT NULL | LangGraph routing decision: `retrieve`, `gerar`, or `usar_web` |
| `confidence_score` | REAL NOT NULL | Float >= 0.0; `None` stored as 0.0 |
| `ranked_response` | TEXT NOT NULL | Final answer as plain string |

DDL:
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

Eviction query (on overflow):
```sql
DELETE FROM query_history WHERE id = (SELECT MIN(id) FROM query_history);
```

No migrations required — new table, new file. No changes to existing ChromaDB schema.

### New constants in `src/agenticlog/config.py`

```python
DIR_HISTORY: Path = PROJECT_ROOT / "data" / "history"   # diretório onde o SQLite é persistido
HISTORY_FILE: Path = DIR_HISTORY / "history.db"          # arquivo SQLite do audit log
HISTORY_MAX_ENTRIES: int = int(os.environ.get("HISTORY_MAX_ENTRIES", "1000"))  # máximo de registros; evicta o mais antigo se atingido
```

Validation (immediately after the constant, mirroring LOG_LEVEL pattern):
```python
if HISTORY_MAX_ENTRIES <= 0:
    raise ValueError(
        f"HISTORY_MAX_ENTRIES={HISTORY_MAX_ENTRIES!r} must be > 0."
    )
```

---

## Process / Background Flow

**Happy path (POST /query with history write):**
1. Client sends POST /query with `{"query": "..."}`.
2. FastAPI validates body via `QueryRequest` Pydantic model (422 if invalid).
3. `consultar()` checks `app.state.vectordb_pronto` (503 if False).
4. `asyncio.to_thread(agent_workflow.invoke, AgentState(query=...))` runs agent.
5. `_normalizar_estado(estado)` produces `QueryResponse`.
6. `_construir_registro(body.query, response)` builds the history dict.
7. `await asyncio.to_thread(app.state.history_store.append, registro)` writes to SQLite.
8. HTTP 200 returned to client with `QueryResponse` body.

**Failure path — write failure:**
1. Steps 1-5 succeed as above.
2. `asyncio.to_thread(history_store.append, ...)` raises any exception.
3. `except Exception` in `consultar()` calls `logger.error(...)`.
4. HTTP 200 still returned to client (write failure is transparent).

**Failure path — history store unavailable on GET /history:**
1. Client sends GET /history.
2. `app.state.history_store` raises exception on `read_all()`.
3. Handler catches exception and raises `HTTPException(status_code=503)`.

---

## API Changes

### POST /query (modified)

No change to request/response contract. Side effect added: history write before return.

### GET /history (new)

```
GET /history?limit=N
```

**Query parameters:**
- `limit` (int, optional, >= 1): Return only the N most recent records. If omitted, returns all.

**Response 200:**
```json
[
  {
    "id": 42,
    "timestamp": "2025-01-15T10:30:00.123456+00:00",
    "query": "qual o prazo de entrega para SP?",
    "next_step": "retrieve",
    "confidence_score": 0.87,
    "ranked_response": "O prazo médio é 3 dias úteis."
  }
]
```
Array sorted by `timestamp` DESC. Empty array `[]` when no records.

**Response 422:** `limit` <= 0 (Pydantic/FastAPI validation).

**Response 503:** History store unavailable.

**Pydantic response model:**
```python
class HistoryEntry(BaseModel):
    id: int
    timestamp: str
    query: str
    next_step: str
    confidence_score: float
    ranked_response: str

class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
```

Note: The endpoint returns the list directly (not wrapped), matching the AC contract `array JSON`.

---

## Frontend Changes

No frontend changes. Streamlit UI (app.py) is explicitly out of scope.

---

## Tests Required

### Unit tests — `tests/test_history.py` (new)

| Test ID | Description | Method naming |
|---------|-------------|---------------|
| teste_1_ | `append()` writes one record; `read_all()` returns it | `teste_1_append_e_read` |
| teste_2_ | `read_all()` returns empty list when DB is empty | `teste_2_read_all_vazio` |
| teste_3_ | `read_all(limit=2)` returns 2 most recent of 5 | `teste_3_read_all_com_limit` |
| teste_4_ | `limit > total` returns all records | `teste_4_limit_maior_que_total` |
| teste_5_ | Eviction: after MAX+1 inserts, count stays at MAX | `teste_5_evicao_max_entries` |
| teste_6_ | `confidence_score=None` stored as 0.0 | `teste_6_confidence_score_none` |
| teste_7_ | `ranked_response` dict stored as normalized string | `teste_7_ranked_response_dict` |
| teste_8_ | Two concurrent `append()` calls; both records persisted | `teste_8_writes_concorrentes` |
| teste_9_ | DB file created on first write if not present | `teste_9_cria_db_no_primeiro_write` |
| test_init_db_idempotente | Calling `init_db()` twice raises no error | `test_init_db_idempotente` |

All tests use `tempfile.TemporaryDirectory()` for isolation — never touch `data/history/`.

### Acceptance tests — `tests/acceptance/test_history_endpoint.py` (new)

| Test ID | AC | Description |
|---------|-----|-------------|
| test_ac_history_01_ | HIST-01 | POST /query → history count increases by 1 |
| test_ac_history_02_ | HIST-02 | Write failure → POST still returns 200 |
| test_ac_history_07_ | HIST-07 | GET /history with records → 200, sorted DESC |
| test_ac_history_08_ | HIST-08 | GET /history empty → 200, [] |
| test_ac_history_09_ | HIST-09 | GET /history?limit=2 → 2 records |
| test_ac_history_11_ | HIST-11 | GET /history?limit=0 → 422 |
| test_ac_history_12_ | HIST-12 | GET /history, store raises → 503 |
| test_ac_history_13_ | HIST-13 | GET /history?limit=100 with 3 records → 3 |

### Existing tests that require patching

- `tests/test_api.py` (13 tests): mock `app.state.history_store` with `MagicMock` so `append()` is a no-op. Namespace: `agenticlog.api`.
- `tests/acceptance/test_api_query_endpoint.py` (10 tests): same mock pattern — inject a `MagicMock` history_store into `app.state` via fixture.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Modify | Add `DIR_HISTORY`, `HISTORY_FILE`, `HISTORY_MAX_ENTRIES` with validation |
| `src/agenticlog/history.py` | Create | `HistoryStore` class: `init_db()`, `append()`, `read_all()` |
| `src/agenticlog/api.py` | Modify | Import `HistoryStore`; init in lifespan; write in `consultar()`; add GET /history |
| `tests/test_history.py` | Create | Unit tests for `HistoryStore` |
| `tests/test_api.py` | Modify | Patch `app.state.history_store` in all existing tests |
| `tests/acceptance/test_api_query_endpoint.py` | Modify | Patch `app.state.history_store` in existing acceptance tests |
| `tests/acceptance/test_history_endpoint.py` | Create | Acceptance tests for GET /history and write isolation |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Race condition on concurrent writes | HIGH | `threading.Lock` in `HistoryStore.append()` — mandatory, not optional |
| Write exception propagates to client | HIGH | `try/except Exception` wrapping `asyncio.to_thread(history_store.append, ...)` in `consultar()` |
| `HISTORY_MAX_ENTRIES = 0` passed via env | HIGH | `ValueError` raised at config import time; API fails to start (correct behavior) |
| `asyncio.to_thread` omitted on blocking SQLite call | MEDIUM | Code review gate: `history_store.append()` must always be called via `asyncio.to_thread` in async context |
| Existing tests broken by `app.state.history_store` attribute | MEDIUM | All 23 existing tests need the mock; verify with `pytest tests/ -v` gate after patch |
| SQLite file path outside project root | LOW | `DIR_HISTORY` anchored to `PROJECT_ROOT` — same pattern as `DIR_VECTORDB` |
| `data/history/` directory not created | LOW | `HistoryStore.__init__` calls `DIR_HISTORY.mkdir(parents=True, exist_ok=True)` before `init_db()` |
| Timestamp timezone inconsistency | LOW | Use `datetime.datetime.now(tz=datetime.UTC).isoformat()` — same pattern as `_JsonFormatter` in config.py |

---

## Open Questions

None. All Q1-Q4 resolved in approved story.

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` passes with >= 80% coverage on `history.py`.
- [ ] POST /query returns HTTP 200 and `tests/acceptance/test_history_endpoint.py::test_ac_history_01_` passes.
- [ ] GET /history returns `[]` for empty store (test_ac_history_08_ passes).
- [ ] GET /history?limit=0 returns 422 (test_ac_history_11_ passes).
- [ ] All 13 existing `tests/test_api.py` tests still pass after adding the mock.
- [ ] All 10 existing `tests/acceptance/test_api_query_endpoint.py` tests still pass.
- [ ] Concurrent write test (teste_8_) passes without deadlock or data loss.
- [ ] `HISTORY_MAX_ENTRIES=0 python -c "import agenticlog.config"` raises `ValueError`.
