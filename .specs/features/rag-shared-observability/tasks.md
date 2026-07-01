# RAG Shared + Observability Extraction — Tasks

**Design:** `.specs/features/rag-shared-observability/design.md`
**Spec:** `.specs/features/rag-shared-observability/spec.md`
**Status:** Awaiting human approval

> These tasks are for **human + builder planning** in the feature-factory pipeline (Phases 4–5).
> TLC Execute is not used here. Gate commands come from `.specs/codebase/TESTING.md`.
> Baseline note: before starting, record the current passing test count (`pytest -q`) so
> "Test count ≥ baseline" can be checked; the suite has no fixed count in CI.

---

## Execution Plan

### Phase 1: Create canonical packages (Parallel OK)

```
T1  (shared/errors + shared/__init__ + test)          [P]
T3  (observability/logging + observability/__init__ + test)   [P]
```

### Phase 2: Wire shims + add history canonical (Parallel OK)

```
T1 ──→ T2  (rag.py shim for RAGSecurityError)         [P]
T3 ──→ T4  (config.py shim for _JsonFormatter)        [P]
T3 ──→ T5  (observability/history + extend __init__ + test)   [P]
```

### Phase 3: history shim (Sequential)

```
T5 ──→ T6  (history.py shim for HistoryStore)
```

### Phase 4: Full-graph verification gate (Sequential)

```
T2, T4, T6 ──→ T7  (fresh-import + full suite + oracle + rag_eval)
```

---

## Task Breakdown

### T1: Create `shared/` package with `RAGSecurityError` + its test [P]

**What:** Create `src/agenticlog/shared/errors.py` (move `RAGSecurityError` verbatim from `rag.py` L87–92), `src/agenticlog/shared/__init__.py` (re-export with `__all__`), and `tests/test_shared_errors.py`.
**Where:** `src/agenticlog/shared/errors.py`, `src/agenticlog/shared/__init__.py`, `tests/test_shared_errors.py`
**Depends on:** None
**Reuses:** `rag.py` L87–92 body; `health.py` + `agenticlog/__init__.py` re-export precedent
**Requirement:** OBS-01, OBS-03, OBS-05 (canonical part), plus foundation for OBS-04

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `RAGSecurityError` defined verbatim in `shared/errors.py` (plain `Exception` subclass, Portuguese docstring preserved).
- [ ] `shared/__init__.py` re-exports `RAGSecurityError` with `__all__ = ["RAGSecurityError"]`.
- [ ] `tests/test_shared_errors.py` asserts: canonical import works; `agenticlog.shared.RAGSecurityError is agenticlog.shared.errors.RAGSecurityError`; raise/catch round-trip across both names.
- [ ] `shared/errors.py` imports NOTHING from `agenticlog.config`.
- [ ] Gate check passes: `pytest tests/test_shared_errors.py -v`
- [ ] Test count ≥ baseline (no silent deletions).

**Verify:** `python -c "from agenticlog.shared import RAGSecurityError; from agenticlog.shared.errors import RAGSecurityError as C; assert RAGSecurityError is C"`

**Tests:** unit · **Gate:** quick
**Commit:** `refactor(shared): extrai RAGSecurityError para shared/errors.py (ADR-018 Fase 2)`

---

### T2: Convert `rag.py` to `RAGSecurityError` shim [P]

**What:** Replace the `RAGSecurityError` class definition in `rag.py` (L87–92) with a re-export shim import; extend `tests/test_shared_errors.py` with the shim-identity assertion.
**Where:** `src/agenticlog/rag.py` (modify), `tests/test_shared_errors.py` (extend)
**Depends on:** T1
**Reuses:** shim marker convention
**Requirement:** OBS-02, OBS-04

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `rag.py` L87–92 class body replaced with `from agenticlog.shared.errors import RAGSecurityError  # Re-export shim (ADR-018 Fase 2) — remover na Fase 6`.
- [ ] All internal `raise RAGSecurityError(...)` sites in `rag.py` still resolve (unchanged).
- [ ] `tests/test_shared_errors.py` asserts `agenticlog.rag.RAGSecurityError is agenticlog.shared.errors.RAGSecurityError`.
- [ ] Existing `RAGSecurityError` consumers pass: `app.py`, `tests/test_rag.py`, `tests/test_app.py`, `tests/test_rag_integration.py`, `tests/test_pdf_to_json.py`, `tests/acceptance/test_document_ingestion_ui.py`, `tests/acceptance/test_multi_collection_chromadb.py`.
- [ ] Gate check passes: `pytest --cov=agenticlog --cov-report=term-missing -v`
- [ ] Test count ≥ baseline (no silent deletions).

**Verify:** `python -c "import agenticlog.rag as r, agenticlog.shared.errors as s; assert r.RAGSecurityError is s.RAGSecurityError"`

**Tests:** unit · **Gate:** full (rag.py change)
**Commit:** `refactor(rag): rag.py re-exporta RAGSecurityError via shim (ADR-018 Fase 2)`

---

### T3: Create `observability/logging.py` with `_JsonFormatter` + its test [P]

**What:** Create `src/agenticlog/observability/logging.py` (move `_JsonFormatter` verbatim from `config.py` L166–177), `src/agenticlog/observability/__init__.py` (re-export `_JsonFormatter` with `__all__`), and `tests/test_observability_logging.py`.
**Where:** `src/agenticlog/observability/logging.py`, `src/agenticlog/observability/__init__.py`, `tests/test_observability_logging.py`
**Depends on:** None
**Reuses:** `config.py` L166–177 body; `health.py` re-export precedent
**Requirement:** OBS-06, OBS-08 (logging part), OBS-10, OBS-11

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `_JsonFormatter` defined verbatim in `observability/logging.py`, importing only `json`, `logging`, `datetime`.
- [ ] `observability/logging.py` imports NOTHING from `agenticlog.config` (OBS-11); test asserts `"agenticlog.config" not in observability.logging.__dict__`-level dependency (e.g. inspect module source or assert no `config` attribute).
- [ ] `observability/__init__.py` re-exports `_JsonFormatter` with `__all__` including it.
- [ ] `tests/test_observability_logging.py` asserts: canonical import; `agenticlog.observability._JsonFormatter is agenticlog.observability.logging._JsonFormatter`; `format()` output for a fixed `LogRecord` parses to a dict with exactly keys `timestamp/level/logger/message` and expected values (level=levelname, logger=name, message=getMessage(), timestamp is UTC ISO-8601 of record.created).
- [ ] Gate check passes: `pytest tests/test_observability_logging.py -v`
- [ ] Test count ≥ baseline (no silent deletions).

**Verify:** `python -c "from agenticlog.observability import _JsonFormatter; from agenticlog.observability.logging import _JsonFormatter as C; assert _JsonFormatter is C"`

**Tests:** unit · **Gate:** quick
**Commit:** `refactor(observability): extrai _JsonFormatter para observability/logging.py (ADR-018 Fase 2)`

---

### T4: Convert `config.py` to `_JsonFormatter` shim [P]

**What:** Replace `_JsonFormatter` class in `config.py` (L166–177) with a re-export shim; keep `LOG_*`/`_VALID_*` constants; remove now-unused `json`/`datetime` imports; extend `tests/test_observability_logging.py` with config-shim + rag-namespace identity assertions.
**Where:** `src/agenticlog/config.py` (modify), `tests/test_observability_logging.py` (extend)
**Depends on:** T3
**Reuses:** shim marker convention
**Requirement:** OBS-07, OBS-09, OBS-19

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `config.py` L166–177 class body replaced with `from agenticlog.observability.logging import _JsonFormatter  # Re-export shim (ADR-018 Fase 2) — remover na Fase 6`.
- [ ] `LOG_LEVEL`, `LOG_FORMAT`, `_VALID_LOG_LEVELS`, `_VALID_LOG_FORMATS` and their validation remain in `config.py`.
- [ ] Top-of-file `import json` and `import datetime` removed (verified: `ruff check src/agenticlog/config.py` clean; keep `logging` only if still referenced).
- [ ] `tests/test_observability_logging.py` asserts `agenticlog.config._JsonFormatter is agenticlog.observability.logging._JsonFormatter` AND `agenticlog.rag._JsonFormatter is agenticlog.observability.logging._JsonFormatter`, and a reload-identity check mirroring `_reload_rag` (reload `config`+`rag`, NOT `observability.logging`) keeps identity (OBS-19).
- [ ] `tests/acceptance/test_structured_log_config.py` passes unchanged.
- [ ] Gate check passes: `pytest --cov=agenticlog --cov-report=term-missing -v` and `ruff check src/agenticlog/config.py`
- [ ] Test count ≥ baseline (no silent deletions).

**Verify:** `python -c "import agenticlog.config as c, agenticlog.rag as r, agenticlog.observability.logging as o; assert c._JsonFormatter is o._JsonFormatter is r._JsonFormatter"`

**Tests:** unit · **Gate:** full (config.py + reload path)
**Commit:** `refactor(config): config.py re-exporta _JsonFormatter via shim (ADR-018 Fase 2)`

---

### T5: Create `observability/history.py` with `HistoryStore` + its test [P]

**What:** Create `src/agenticlog/observability/history.py` (move the whole current `history.py` body verbatim), extend `observability/__init__.py` to also re-export `HistoryStore`, and create `tests/test_observability_history.py`.
**Where:** `src/agenticlog/observability/history.py`, `src/agenticlog/observability/__init__.py` (extend), `tests/test_observability_history.py`
**Depends on:** T3 (package `__init__` must exist)
**Reuses:** current `history.py` body verbatim
**Requirement:** OBS-12, OBS-14 (history part), OBS-16

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `HistoryStore` (with `__init__`/`init_db`/`append`/`read_all`) defined verbatim in `observability/history.py`; stdlib-only (`sqlite3`, `threading`, `logging`, `pathlib`); imports NOTHING from `agenticlog.config`.
- [ ] `observability/__init__.py` re-exports `HistoryStore` (added to `__all__` alongside `_JsonFormatter`).
- [ ] `tests/test_observability_history.py` asserts: canonical import; `agenticlog.observability.HistoryStore is agenticlog.observability.history.HistoryStore`; append→read_all round-trip on a `tmp_path` SQLite file verifying `query_history` columns, FIFO eviction at `max_entries`, and DESC ordering.
- [ ] Gate check passes: `pytest tests/test_observability_history.py -v`
- [ ] Test count ≥ baseline (no silent deletions).

**Verify:** `python -c "from agenticlog.observability import HistoryStore; from agenticlog.observability.history import HistoryStore as C; assert HistoryStore is C"`

**Tests:** unit · **Gate:** quick
**Commit:** `refactor(observability): extrai HistoryStore para observability/history.py (ADR-018 Fase 2)`

---

### T6: Convert `history.py` to `HistoryStore` shim

**What:** Replace the body of `src/agenticlog/history.py` with a re-export shim; extend `tests/test_observability_history.py` with shim + patch-namespace-safety assertions.
**Where:** `src/agenticlog/history.py` (modify), `tests/test_observability_history.py` (extend)
**Depends on:** T5
**Reuses:** shim marker convention
**Requirement:** OBS-13, OBS-15, OBS-17

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] `history.py` body replaced with module docstring note + `from agenticlog.observability.history import HistoryStore  # Re-export shim (ADR-018 Fase 2) — remover na Fase 6` (+ optional `__all__ = ["HistoryStore"]`).
- [ ] `tests/test_observability_history.py` asserts `agenticlog.history.HistoryStore is agenticlog.observability.history.HistoryStore`.
- [ ] `tests/test_api.py` passes unchanged (11+ `@patch("agenticlog.api.HistoryStore")` sites still target the consumer namespace); `api.py` `from agenticlog.history import HistoryStore` still resolves.
- [ ] `tests/acceptance/test_history_endpoint.py` and `tests/test_query_history_audit_logging.py` pass unchanged.
- [ ] Gate check passes: `pytest --cov=agenticlog --cov-report=term-missing -v`
- [ ] Test count ≥ baseline (no silent deletions).

**Verify:** `python -c "import agenticlog.history as h, agenticlog.api as a, agenticlog.observability.history as o; assert h.HistoryStore is o.HistoryStore is a.HistoryStore"`

**Tests:** unit · **Gate:** full (history.py + api consumers)
**Commit:** `refactor(history): history.py re-exporta HistoryStore via shim (ADR-018 Fase 2)`

---

### T7: Full-graph verification gate

**What:** Prove no circular import, whole-suite green, characterization oracle green, `rag_eval` golden gate green, ruff clean, and `agenticlog/__init__.py` unchanged.
**Where:** repository-wide (no production edits expected; fix-forward only if a gate fails)
**Depends on:** T2, T4, T6
**Reuses:** ADR-018 §5 transition safety net (oracle + rag_eval)
**Requirement:** OBS-18, OBS-20, OBS-21

**Tools:** MCP: NONE · Skill: NONE

**Done when:**
- [ ] Fresh-interpreter import is clean: `python -c "import agenticlog"` exits 0 with no `ImportError`/partially-initialized-module (OBS-18).
- [ ] `observability.logging` proven not to import `config` (assertion in T3/T4 test suite passes).
- [ ] Full suite green: `pytest --cov=agenticlog --cov-report=term-missing -v`; test count ≥ baseline (no silent deletions).
- [ ] Characterization oracle green: `pytest tests/test_rag_caracterizacao.py -v` (file unchanged).
- [ ] `rag_eval` golden gate green (per `.specs/features/rag-golden-eval-ci` / CI workflow) — unaffected, this feature touches no retrieval code.
- [ ] `ruff check src/agenticlog` clean (no leftover unused imports).
- [ ] `git diff --stat src/agenticlog/__init__.py` shows no change (OBS-21).

**Verify:** run the four gate commands above; all exit 0.

**Tests:** integration (import smoke) + full suite · **Gate:** build
**Commit:** (no code commit expected; if a fix was needed) `test(observability): valida ausência de import circular e verde da suíte (ADR-018 Fase 2)`

---

## Pre-Approval Validation

### Check 1 — Task Granularity

| Task | Scope | Status |
|------|-------|--------|
| T1 | 1 new module + its package init + its test (cohesive extraction) | ✅ Granular |
| T2 | 1 file shim edit + 1 test extension | ✅ Granular |
| T3 | 1 new module + package init + its test (cohesive) | ✅ Granular |
| T4 | 1 file shim edit (+ import cleanup) + test extension | ✅ Granular |
| T5 | 1 new module + init extension + its test (cohesive) | ✅ Granular |
| T6 | 1 file shim edit + test extension | ✅ Granular |
| T7 | verification-only gate | ✅ Granular |

### Check 2 — Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram Shows | Status |
|------|-------------------|---------------|--------|
| T1 | None | (root) | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | None | (root) | ✅ Match |
| T4 | T3 | T3 → T4 | ✅ Match |
| T5 | T3 | T3 → T5 | ✅ Match |
| T6 | T5 | T5 → T6 | ✅ Match |
| T7 | T2, T4, T6 | T2,T4,T6 → T7 | ✅ Match |

Parallel-safety of `[P]` in the same phase:
- Phase 1: T1 (`shared/*`) and T3 (`observability/*`) touch disjoint files → `[P]` valid.
- Phase 2: T2 (`rag.py`), T4 (`config.py`), T5 (`observability/history.py` + `observability/__init__.py`) touch disjoint files → `[P]` valid. T5 depends on T3 (init created in Phase 1), so no Phase-2 write conflict on `observability/__init__.py`.
- T6 is sequential (Phase 3); T7 sequential (Phase 4).

### Check 3 — Test Co-location Validation

TESTING.md coverage matrix has no row for the new `shared/` and `observability/` modules; per project rules (80% min, unit for logic modules) and the mocked-unit convention, all logic modules take **unit** tests, co-located in the same task that creates them. No task defers tests.

| Task | Code Layer Created/Modified | Matrix/Rule Requires | Task Says | Status |
|------|-----------------------------|----------------------|-----------|--------|
| T1 | `shared/errors.py`, `shared/__init__.py` | unit | unit (test_shared_errors.py) | ✅ OK |
| T2 | `rag.py` (shim) | unit (rag.py security ⇒ unit) | unit (full gate) | ✅ OK |
| T3 | `observability/logging.py`, `__init__.py` | unit | unit (test_observability_logging.py) | ✅ OK |
| T4 | `config.py` (shim, constants) | unit (behavior-preserving shim; reload path) | unit (full gate + ruff) | ✅ OK |
| T5 | `observability/history.py`, `__init__.py` | unit | unit (test_observability_history.py) | ✅ OK |
| T6 | `history.py` (shim) | unit | unit (full gate) | ✅ OK |
| T7 | none (verification) | none | integration/full gate | ✅ OK |

All checks pass — tasks are ready for Checkpoint-2 review.

---

## Parallelism constraint note

`[P]` tasks are unit-tested and the unit type is Parallel-Safe (all deps mocked, no shared mutable state — per TESTING.md Parallelism Assessment). The only shared file, `observability/__init__.py`, is created in T3 and extended in T5; T5's `Depends on: T3` serializes those writes, so no two concurrent tasks write the same file.
