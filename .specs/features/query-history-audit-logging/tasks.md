# Query History Audit Logging — Tasks

**Path:** `.specs/features/query-history-audit-logging/tasks.md`
**TLC scope:** large
**Spec:** `.specs/features/query-history-audit-logging/spec.md`
**Design:** `.specs/features/query-history-audit-logging/design.md`
**Status:** Awaiting human approval

---

## Dependency Order

```
TASK-01 (config)
    └─► TASK-02 (history.py)
            └─► TASK-03 (unit tests — test_history.py)
                    └─► TASK-04 (api.py modifications)
                                └─► TASK-05 (patch existing tests)
                                            └─► TASK-06 (acceptance tests)
                                                        └─► TASK-07 (gate)
```

---

## TASK-01 — Add history constants to config.py

**Refs:** HIST-14, HIST-15 (spec.md)
**Depends on:** nothing
**Estimated effort:** XS (< 30 min)

**What to do:**

In `src/agenticlog/config.py`, after the `API_PORT` constant and before the `# Logging` section, add:

```python
# History Audit Log
DIR_HISTORY: Path = PROJECT_ROOT / "data" / "history"   # diretório onde o SQLite é persistido
HISTORY_FILE: Path = DIR_HISTORY / "history.db"          # arquivo SQLite do audit log
HISTORY_MAX_ENTRIES: int = int(os.environ.get("HISTORY_MAX_ENTRIES", "1000"))  # máximo de registros; evicta o mais antigo se atingido
if HISTORY_MAX_ENTRIES <= 0:
    raise ValueError(
        f"HISTORY_MAX_ENTRIES={HISTORY_MAX_ENTRIES!r} must be > 0."
    )
```

Note: `os` is already imported at line 8. `Path` is already imported. No new imports needed.

**Done when:**
- `python -c "from agenticlog.config import DIR_HISTORY, HISTORY_FILE, HISTORY_MAX_ENTRIES; print(HISTORY_MAX_ENTRIES)"` prints `1000`.
- `HISTORY_MAX_ENTRIES=0 python -c "import agenticlog.config"` raises `ValueError`.
- `HISTORY_MAX_ENTRIES=-1 python -c "import agenticlog.config"` raises `ValueError`.
- Existing `pytest tests/ -v` still passes (no regressions from config change).

---

## TASK-02 — Create src/agenticlog/history.py

**Refs:** HIST-01, HIST-02, HIST-03, HIST-04, HIST-05, HIST-06, HIST-15, HIST-16 (spec.md)
**Depends on:** TASK-01
**Estimated effort:** S (1-2 h)

**What to do:**

Create `src/agenticlog/history.py` with the following structure (follow conventions from CONVENTIONS.md — module docstring, imports, class):

```
Module docstring (Portuguese)
Standard imports: import datetime, import json, import logging, import sqlite3, import threading
from pathlib import Path
logger = logging.getLogger(__name__)

class HistoryStore:
    """..."""
    def __init__(self, db_path: Path, max_entries: int) -> None
    def init_db(self) -> None
    def append(self, registro: dict) -> None
    def read_all(self, limit: int | None = None) -> list[dict]
```

**Implementation details:**

`__init__`:
- Store `self._db_path = db_path`, `self._max_entries = max_entries`.
- `db_path.parent.mkdir(parents=True, exist_ok=True)` — creates `data/history/` if absent.
- `self._lock = threading.Lock()`.
- Call `self.init_db()`.

`init_db`:
- Open connection to `self._db_path` (use `sqlite3.connect(str(self._db_path))`).
- Execute DDL: `CREATE TABLE IF NOT EXISTS query_history (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, query TEXT NOT NULL, next_step TEXT NOT NULL, confidence_score REAL NOT NULL, ranked_response TEXT NOT NULL)`.
- `connection.commit()`, `connection.close()`.
- No lock needed here (called only from `__init__` before concurrency begins).

`append`:
- Acquire `self._lock` (use `with self._lock:`).
- Open new connection inside the lock.
- `SELECT COUNT(*) FROM query_history` — if result >= `self._max_entries`: execute eviction DELETE.
- `INSERT INTO query_history (timestamp, query, next_step, confidence_score, ranked_response) VALUES (?, ?, ?, ?, ?)` with tuple from `registro`.
- `connection.commit()`, `connection.close()`.
- Do NOT catch exceptions — let them propagate to caller (api.py catches them).

`read_all`:
- Open connection. No lock for reads.
- Build query: base SELECT + `ORDER BY timestamp DESC` + `LIMIT ?` if `limit` is not None.
- `connection.row_factory = sqlite3.Row` — enables dict-like access.
- Return `[dict(row) for row in cursor.fetchall()]`.
- `connection.close()`.

**Normalizations (caller responsibility, not HistoryStore):**
- `confidence_score=None` → caller passes `0.0`.
- `ranked_response` dict → caller passes normalized string.
HistoryStore only stores what it receives. This keeps the store dumb and testable.

**Done when:**
- `src/agenticlog/history.py` exists and is importable: `python -c "from agenticlog.history import HistoryStore"`.
- `ruff check src/agenticlog/history.py` passes.
- `mypy src/agenticlog/history.py` passes (or matches existing mypy config).

---

## TASK-03 — Write unit tests: tests/test_history.py

**Refs:** HIST-01 through HIST-06, HIST-15 (spec.md)
**Depends on:** TASK-02
**Estimated effort:** S (1-2 h)

**What to do:**

Create `tests/test_history.py`. Use `unittest.TestCase` (project convention). Use `tempfile.TemporaryDirectory()` for DB isolation — all tests must create their own temp dir and `HistoryStore` instance; never use `data/history/`.

Test class: `TestHistoryStore`

| Method | What it asserts |
|--------|-----------------|
| `teste_1_append_e_read` | Append 1 record; `read_all()` returns list of 1 dict with correct keys and values |
| `teste_2_read_all_vazio` | Fresh store; `read_all()` returns `[]` |
| `teste_3_read_all_com_limit` | Insert 5 records with distinct timestamps; `read_all(limit=2)` returns 2; first is most recent |
| `teste_4_limit_maior_que_total` | Insert 3; `read_all(limit=100)` returns 3 without error |
| `teste_5_evicao_max_entries` | Create store with `max_entries=3`; insert 4; `len(read_all())` == 3; oldest not present |
| `teste_6_confidence_score_none_normalizado` | Caller passes `0.0` for None; assert stored value == 0.0 |
| `teste_7_ranked_response_dict_normalizado` | Caller passes `json.dumps({"answer": "text"})`; assert stored string matches |
| `teste_8_writes_concorrentes` | Use `threading.Thread` × 10; all append; assert `len(read_all()) == 10` |
| `teste_9_cria_db_no_primeiro_write` | Point store at nonexistent path; append; assert file exists |
| `test_init_db_idempotente` | Call `init_db()` twice; no exception raised; table exists |

**Helper fixture pattern** (setUp):
```python
def setUp(self):
    self._tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(self._tmpdir.name) / "test.db"
    self.store = HistoryStore(db_path=db_path, max_entries=100)

def tearDown(self):
    self._tmpdir.cleanup()

def _registro(self, query: str = "test query", ts: str | None = None) -> dict:
    return {
        "timestamp": ts or datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "query": query,
        "next_step": "retrieve",
        "confidence_score": 0.9,
        "ranked_response": "resposta teste",
    }
```

**Done when:**
- `pytest tests/test_history.py -v` passes all 10 tests.
- `pytest --cov=agenticlog.history --cov-report=term-missing tests/test_history.py` shows >= 80% branch coverage on `history.py`.

---

## TASK-04 — Modify api.py: lifespan + consultar() + GET /history

**Refs:** HIST-01, HIST-02, HIST-07 through HIST-13, HIST-17 (spec.md)
**Depends on:** TASK-02
**Estimated effort:** S (1-2 h)

**What to do:**

Edit `src/agenticlog/api.py` in this order:

**Step 1 — Imports.** Add:
```python
import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from agenticlog.config import DEFAULT_COLLECTION_NAME, DIR_VECTORDB, HISTORY_FILE, HISTORY_MAX_ENTRIES
from agenticlog.history import HistoryStore
```
(Replace existing `from fastapi import FastAPI, HTTPException, Request` with version including `Query`.)

**Step 2 — New Pydantic model.** After `QueryResponse`, add:
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

**Step 3 — New pure helper.** After `_normalizar_estado`, before the lifespan section, add `_construir_registro(query: str, response: QueryResponse) -> dict` as specified in design.md §3c.

**Step 4 — Lifespan.** Inside the `try` block of `lifespan`, after `app.state.vectordb_pronto = True`, add:
```python
app.state.history_store = HistoryStore(db_path=HISTORY_FILE, max_entries=HISTORY_MAX_ENTRIES)
logger.info("HistoryStore inicializado em %s", HISTORY_FILE)
```
If `HistoryStore.__init__` raises, it is caught by the outer `except Exception` block — acceptable (API startup fails, consistent behavior).

**Step 5 — Modify `consultar()`.** After `response = _normalizar_estado(estado)`, add the audit write block as specified in design.md §3e. Return `response` at the end.

**Step 6 — New endpoint.** Add `listar_historico()` after `consultar()`, before the exception handlers, as specified in design.md §3f.

**Done when:**
- `python -c "from agenticlog.api import app"` imports without error.
- `ruff check src/agenticlog/api.py` passes.
- The app starts: `uvicorn agenticlog.api:app --host 127.0.0.1 --port 8000` (with LMStudio + vectordb available) and `GET /history` returns `[]`.
- Do NOT run full test suite yet — existing tests will fail until TASK-05.

---

## TASK-05 — Patch existing test files for history_store mock

**Refs:** HIST-01, HIST-02 (spec.md — write isolation)
**Depends on:** TASK-04
**Estimated effort:** S (30-60 min)

**What to do:**

Both existing test files create a FastAPI `TestClient(app)`. After TASK-04, `app.state.history_store` must exist. In tests, the store must be replaced with a `MagicMock` so tests don't touch the filesystem.

**Pattern to add as a fixture or setUp in both files:**

```python
from unittest.mock import MagicMock, patch

# Option A — per-test mock via app.state directly (simplest, no namespace patch needed):
app.state.history_store = MagicMock()
app.state.history_store.append.return_value = None
app.state.history_store.read_all.return_value = []
```

Since `app` is a module-level singleton, set this in a `setUpClass` or at module level before the `TestClient` is created.

**Files to modify:**

1. `tests/test_api.py` — Add mock setup for `app.state.history_store` in `setUpClass` or `setUp` of all `TestCase` classes. Verify all 13 tests still pass.

2. `tests/acceptance/test_api_query_endpoint.py` — Same pattern. Verify all 10 acceptance tests still pass.

**Done when:**
- `pytest tests/test_api.py -v` — all 13 tests pass.
- `pytest tests/acceptance/test_api_query_endpoint.py -v` — all 10 tests pass.
- No `AttributeError: 'State' object has no attribute 'history_store'` in either file.

---

## TASK-06 — Write acceptance tests: tests/acceptance/test_history_endpoint.py

**Refs:** HIST-01, HIST-02, HIST-07 through HIST-13 (spec.md)
**Depends on:** TASK-04, TASK-05
**Estimated effort:** S (1-2 h)

**What to do:**

Create `tests/acceptance/test_history_endpoint.py`. Use `unittest.TestCase` + `fastapi.testclient.TestClient`.

**Key setup:** The acceptance tests for GET /history need a real `HistoryStore` (or a controlled mock) to assert record insertion. Use a `HistoryStore` with a `tempfile`-based DB path injected into `app.state`.

**setUp pattern:**
```python
from fastapi.testclient import TestClient
from agenticlog.api import app
from agenticlog.history import HistoryStore
import tempfile
from pathlib import Path

class TestHistoryEndpoint(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "test.db"
        app.state.history_store = HistoryStore(db_path=db_path, max_entries=100)
        app.state.vectordb_pronto = True  # bypass vectordb check
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self):
        self._tmpdir.cleanup()
```

For tests that need agent to run (HIST-01): patch `agenticlog.api.agent_workflow` and `agenticlog.api.inicializar_recursos`.

**Tests to implement:**

| Method | AC ref | What to assert |
|--------|--------|----------------|
| `test_ac_history_01_post_grava_registro` | HIST-01 | POST /query succeeds; `history_store.read_all()` returns 1 record with correct fields |
| `test_ac_history_02_write_failure_nao_afeta_resposta` | HIST-02 | Inject broken store (append raises); POST /query returns 200 |
| `test_ac_history_07_get_retorna_registros_ordenados` | HIST-07 | Insert 3 records; GET /history returns 200; list sorted DESC by timestamp |
| `test_ac_history_08_get_sem_registros_retorna_lista_vazia` | HIST-08 | Fresh store; GET /history returns 200 with `[]` |
| `test_ac_history_09_get_com_limit` | HIST-09 | Insert 5; GET /history?limit=3 returns 3 records |
| `test_ac_history_11_limit_zero_retorna_422` | HIST-11 | GET /history?limit=0 returns 422 |
| `test_ac_history_12_store_indisponivel_retorna_503` | HIST-12 | Replace store with MagicMock where `read_all` raises; GET /history returns 503 |
| `test_ac_history_13_limit_maior_que_total` | HIST-13 | Insert 3; GET /history?limit=100 returns 3 |

**Done when:**
- `pytest tests/acceptance/test_history_endpoint.py -v` — all 8 tests pass.
- No tests touch `data/history/` (all use tempdir).

---

## TASK-07 — Final gate: full test suite + coverage

**Refs:** All HIST-* requirements (spec.md Success Criteria)
**Depends on:** TASK-03, TASK-05, TASK-06
**Estimated effort:** XS (15 min)

**What to do:**

Run full suite and verify all gates from spec.md:

```powershell
# Full suite with coverage
pytest --cov=agenticlog --cov-report=term-missing -v

# Specific coverage check for history.py
pytest --cov=agenticlog.history --cov-report=term-missing tests/test_history.py -v

# Verify config validation
$env:HISTORY_MAX_ENTRIES="0"; python -c "import agenticlog.config"
# Expected: ValueError

# Verify no regressions
pytest tests/test_api.py tests/acceptance/test_api_query_endpoint.py -v
```

**Done when:**
- All tests pass (0 failures, 0 errors).
- `history.py` branch coverage >= 80%.
- `agenticlog` overall coverage >= 80%.
- `HISTORY_MAX_ENTRIES=0` raises `ValueError`.
- No `print()` statements in `history.py` or modified sections of `api.py` (use `logging`).
- `ruff check src/agenticlog/` passes.

---

## Summary Table

| Task | File(s) | Req IDs | Effort | Gate |
|------|---------|---------|--------|------|
| TASK-01 | `config.py` | HIST-14, HIST-15 | XS | `pytest tests/ -v` (no regression) |
| TASK-02 | `history.py` (new) | HIST-01 to HIST-06, HIST-15, HIST-16 | S | Import check + ruff |
| TASK-03 | `test_history.py` (new) | HIST-01 to HIST-06, HIST-15 | S | `pytest tests/test_history.py -v` all pass, >= 80% cov |
| TASK-04 | `api.py` | HIST-01, HIST-02, HIST-07 to HIST-13, HIST-17 | S | Import check + ruff; manual smoke test |
| TASK-05 | `test_api.py`, `test_api_query_endpoint.py` | HIST-01, HIST-02 | S | All 23 existing tests pass |
| TASK-06 | `test_history_endpoint.py` (new) | HIST-01, HIST-02, HIST-07 to HIST-13 | S | All 8 acceptance tests pass |
| TASK-07 | — | All | XS | Full suite green, >= 80% coverage |
