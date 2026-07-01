# RAG Shared + Observability Extraction — Technical Spec

**Path:** `.specs/features/rag-shared-observability/spec.md`
**TLC scope:** large
**Based on story:** Extract `RAGSecurityError` → `shared/errors.py`, `_JsonFormatter` → `observability/logging.py`, `HistoryStore` → `observability/history.py`, leaving re-export shims at the origin paths until Phase 6 — no behavior change, no circular imports.
**Related:** ADR-018 (§ redesign offline/online, Fase 2), `docs/arquitetura-alvo-rag.md` §3/§4/§7, `.specs/codebase/CONCERNS.md`, `.specs/codebase/TESTING.md`
**Status:** Awaiting human approval

---

## Problem Statement

`rag.py` (1013 lines) and `config.py` mix cross-cutting concerns (a domain exception, a JSON log formatter) with pipeline logic, and the SQLite audit log lives in a top-level `history.py`. ADR-018 targets an offline/online architecture with dedicated `shared/` and `observability/` packages. Phase 2 of that redesign is the **lowest-risk slice**: relocate three self-contained symbols into their target packages while keeping every existing import path working via shims, so no consumer, test, or eval breaks.

## Goals

- [ ] `RAGSecurityError` lives canonically in `src/agenticlog/shared/errors.py`; `_JsonFormatter` in `src/agenticlog/observability/logging.py`; `HistoryStore` in `src/agenticlog/observability/history.py`.
- [ ] Every existing import path (`agenticlog.rag.RAGSecurityError`, `agenticlog.config._JsonFormatter`, `agenticlog.history.HistoryStore`) keeps working through re-export shims, resolving to the **same object** as the canonical location.
- [ ] Ergonomic package re-exports work: `from agenticlog.shared import RAGSecurityError`, `from agenticlog.observability import _JsonFormatter, HistoryStore`.
- [ ] Zero behavior change: formatter output, history schema/thread-safety/returns, and exception raise/catch semantics are byte/behaviour identical.
- [ ] No circular imports; `observability.logging` does not import `agenticlog.config`.
- [ ] Full pytest suite, the characterization oracle (`tests/test_rag_caracterizacao.py`), and the `rag_eval` golden gate stay green.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Removing the shims from `rag.py` / `config.py` / `history.py` | Deferred to ADR-018 Fase 6 (test rewrite with dependency injection). This spec adds shims, does not remove them. |
| Moving `LOG_LEVEL` / `LOG_FORMAT` / `_VALID_LOG_LEVELS` / `_VALID_LOG_FORMATS` | Confirmed at Checkpoint 1: stay in `config.py` to avoid a `config → observability.logging → config` cycle. |
| Moving `_configurar_logging_cli` out of `rag.py` | Confirmed at Checkpoint 1: it moves to `ingestion/cli.py` in ADR-018 Fase 3, not here. |
| Extracting `ingestion/`, `retrieval/`, `serving/` packages | Later ADR-018 phases (Fase 3–5). |
| Changing `_JsonFormatter` fields, `HistoryStore` schema, or `RAGSecurityError` message text | Pure relocation — no functional change permitted. |
| Rewriting existing `@patch("agenticlog.rag.*")` / `@patch("agenticlog.api.HistoryStore")` tests | Shims preserve these namespaces; test rewrite is Fase 6. |

---

## User Stories

### P1: Extract `RAGSecurityError` to `shared/errors.py` ⭐ MVP

**User Story**: As a developer maintaining AgenticLog, I want the domain security exception in `shared/errors.py` with a shim at `rag.py`, so that the offline/online redesign has a `shared/` contracts package without breaking any raiser or catcher of `RAGSecurityError`.

**Why P1**: `shared/errors.py` is a foundational contract other ADR-018 phases depend on, and it is the smallest, lowest-risk extraction (plain `Exception` subclass, stdlib-only, no internal `@patch` on it).

**Acceptance Criteria**:
1. WHEN a caller runs `from agenticlog.shared.errors import RAGSecurityError` THEN the system SHALL resolve the canonical class.
2. WHEN a caller runs `from agenticlog.shared import RAGSecurityError` THEN the system SHALL resolve the same object as the canonical class (ergonomic package re-export).
3. WHEN a caller runs `from agenticlog.rag import RAGSecurityError` (existing shim path) THEN the system SHALL resolve the same object as the canonical class.
4. WHEN `agenticlog.rag` raises `RAGSecurityError` and a caller catches `agenticlog.shared.errors.RAGSecurityError` (or vice-versa) THEN the `except` SHALL catch it (single class identity, one MRO).
5. WHEN the existing consumers (`app.py`, `tests/test_rag.py`, `tests/test_app.py`, `tests/test_rag_integration.py`, `tests/test_pdf_to_json.py`, `tests/acceptance/test_document_ingestion_ui.py`, `tests/acceptance/test_multi_collection_chromadb.py`) run THEN they SHALL pass unchanged.

**Independent Test**: `tests/test_shared_errors.py` asserts canonical import, `agenticlog.rag.RAGSecurityError is agenticlog.shared.errors.RAGSecurityError`, `agenticlog.shared.RAGSecurityError is agenticlog.shared.errors.RAGSecurityError`, and a raise/catch round-trip across both names.

---

### P1: Extract `_JsonFormatter` to `observability/logging.py` ⭐ MVP

**User Story**: As a developer maintaining AgenticLog, I want the JSON log formatter in `observability/logging.py` with a shim at `config.py`, so that observability is a distinct package while `rag.py`'s `from agenticlog.config import _JsonFormatter` and the structured-log tests keep working.

**Why P1**: The formatter is stdlib-only and already free of `LOG_*` references, so it moves without touching the log-config constants — but the move must not introduce a `config ↔ observability` cycle, which is the one real risk of this feature.

**Acceptance Criteria**:
1. WHEN a caller runs `from agenticlog.observability.logging import _JsonFormatter` THEN the system SHALL resolve the canonical class.
2. WHEN a caller runs `from agenticlog.observability import _JsonFormatter` THEN the system SHALL resolve the same object as the canonical class.
3. WHEN a caller runs `from agenticlog.config import _JsonFormatter` (existing shim path) THEN the system SHALL resolve the same object as the canonical class.
4. WHEN `agenticlog.rag._JsonFormatter` is referenced (rag imports it from config) THEN it SHALL be the same object as `agenticlog.observability.logging._JsonFormatter`.
5. WHEN `_JsonFormatter().format(record)` runs on any `LogRecord` THEN the output SHALL be identical to the pre-move implementation: a JSON line with exactly the keys `timestamp` (UTC ISO-8601 from `record.created`), `level` (`record.levelname`), `logger` (`record.name`), `message` (`record.getMessage()`).
6. WHEN `agenticlog.observability.logging` is imported THEN it SHALL NOT import `agenticlog.config` (no back-edge).
7. WHEN `tests/acceptance/test_structured_log_config.py` runs (including its `_reload_rag` helper that reloads `config` + `rag`) THEN it SHALL pass unchanged and identity SHALL survive the reload.

**Independent Test**: `tests/test_observability_logging.py` asserts canonical import, the three identity pairs (config-shim, observability-init, rag namespace), and a `format()` golden comparing the parsed JSON keys/values against a hand-built expected dict for a fixed `LogRecord`.

---

### P1: Extract `HistoryStore` to `observability/history.py` ⭐ MVP

**User Story**: As a developer maintaining AgenticLog, I want `HistoryStore` in `observability/history.py` with a shim at `history.py`, so that the audit log sits with observability while `api.py` and its patched tests keep importing `from agenticlog.history import HistoryStore`.

**Why P1**: The audit store is stdlib-only (SQLite/threading) and consumed by `api.py`; the risk is the `@patch("agenticlog.api.HistoryStore")` sites (11+) in `tests/test_api.py`, which patch the **consumer** namespace and must remain unaffected by relocating the source.

**Acceptance Criteria**:
1. WHEN a caller runs `from agenticlog.observability.history import HistoryStore` THEN the system SHALL resolve the canonical class.
2. WHEN a caller runs `from agenticlog.observability import HistoryStore` THEN the system SHALL resolve the same object as the canonical class.
3. WHEN a caller runs `from agenticlog.history import HistoryStore` (existing shim path) THEN the system SHALL resolve the same object as the canonical class.
4. WHEN `HistoryStore(db_path, max_entries)` is constructed and `init_db` / `append` / `read_all` are exercised THEN the SQLite schema (`query_history` columns), the write-lock thread-safety, the FIFO eviction at `max_entries`, and the `read_all` DESC-ordered `list[dict]` return SHALL be identical to the pre-move implementation.
5. WHEN `tests/test_api.py` runs with its `@patch("agenticlog.api.HistoryStore")` sites THEN they SHALL patch the consumer namespace exactly as before and pass unchanged.
6. WHEN `tests/acceptance/test_history_endpoint.py` and `tests/test_query_history_audit_logging.py` (which import from `agenticlog.history`) run THEN they SHALL pass unchanged.

**Independent Test**: `tests/test_observability_history.py` asserts canonical import, the identity pairs (history-shim, observability-init), and an append→read_all round-trip against a `tmp_path` SQLite file verifying schema and DESC ordering.

---

## Edge Cases

- WHEN `import agenticlog` runs from a fresh interpreter (cold `sys.modules`) THEN the package SHALL import cleanly with no `ImportError`/`partially initialized module` from the new shims.
- WHEN `importlib.reload(config)` and `importlib.reload(rag)` run **without** reloading `observability.logging` (the `_reload_rag` helper) THEN `rag._JsonFormatter is observability.logging._JsonFormatter` SHALL still hold (shim binds to the already-loaded canonical module object).
- WHEN `config.py` is imported and `_JsonFormatter`'s body no longer lives there THEN any now-unused module imports (`json`, `datetime`) SHALL be removed so `ruff` stays clean.
- WHEN `observability/__init__.py` is imported (triggered transitively by `config`'s shim) THEN it SHALL only re-export stdlib-only submodules and SHALL NOT pull in `config`, keeping the `config → observability` edge acyclic.
- WHEN a shim line is read by a future maintainer THEN it SHALL carry the marker `# Re-export shim (ADR-018 Fase 2) — remover na Fase 6`.

---

## Requirement Traceability

| Requirement ID | Story | Type | Phase | Status |
|----------------|-------|------|-------|--------|
| OBS-01 | P1-A errors | FR — canonical import `shared.errors.RAGSecurityError` | Design | Pending |
| OBS-02 | P1-A errors | FR — shim import `rag.RAGSecurityError` works | Design | Pending |
| OBS-03 | P1-A errors | FR — `shared` `__init__` re-export | Design | Pending |
| OBS-04 | P1-A errors | FR — identity: canonical IS shim IS `__init__` re-export | Design | Pending |
| OBS-05 | P1-A errors | NFR — raise/catch semantics identical (one class, one MRO) | Design | Pending |
| OBS-06 | P1-B logging | FR — canonical import `observability.logging._JsonFormatter` | Design | Pending |
| OBS-07 | P1-B logging | FR — shim import `config._JsonFormatter` works | Design | Pending |
| OBS-08 | P1-B logging | FR — `observability` `__init__` re-export | Design | Pending |
| OBS-09 | P1-B logging | FR — identity: canonical IS `config` shim IS `rag._JsonFormatter` IS `__init__` re-export | Design | Pending |
| OBS-10 | P1-B logging | NFR — `format()` output identical (keys timestamp/level/logger/message) | Design | Pending |
| OBS-11 | P1-B logging | NFR — `observability.logging` does not import `config` (no cycle) | Design | Pending |
| OBS-12 | P1-C history | FR — canonical import `observability.history.HistoryStore` | Design | Pending |
| OBS-13 | P1-C history | FR — shim import `history.HistoryStore` works | Design | Pending |
| OBS-14 | P1-C history | FR — `observability` `__init__` re-export | Design | Pending |
| OBS-15 | P1-C history | FR — identity: canonical IS shim IS `__init__` re-export | Design | Pending |
| OBS-16 | P1-C history | NFR — schema/thread-safety/eviction/returns identical | Design | Pending |
| OBS-17 | P1-C history | NFR — `@patch("agenticlog.api.HistoryStore")` consumer-namespace patches unaffected | Design | Pending |
| OBS-18 | all | NFR — fresh `import agenticlog` clean; no circular imports | Design | Pending |
| OBS-19 | all | NFR — reload identity survives `_reload_rag` (config+rag, not observability) | Design | Pending |
| OBS-20 | all | NFR — full suite + characterization oracle + `rag_eval` green; new tests added | Design | Pending |
| OBS-21 | all | NFR — `src/agenticlog/__init__.py` requires no change (verified) | Design | Pending |

**ID format:** `OBS-[NUMBER]`.
**Coverage:** 21 total, all mapped to tasks in `tasks.md` (see cross-check tables there).

---

## Data Model Changes

No data model changes. `HistoryStore` moves byte-for-byte; the `query_history` SQLite schema (`id, timestamp, query, next_step, confidence_score, ranked_response`) is unchanged. No migration.

---

## Process / Background Flow

Pure module relocation with re-export shims — no runtime flow changes.

**Happy path:** Source bodies move to canonical modules; origin modules import the symbol back (shim) and re-export it; package `__init__` files re-export ergonomically. All import paths and the runtime behavior of the three symbols are unchanged.

**Failure path — accidental circular import:** If `observability.logging` (or `observability/__init__.py`) ever imported `agenticlog.config`, the `config → observability.logging → config` cycle would raise `ImportError: partially initialized module` on fresh `import agenticlog`. Mitigation: keep `LOG_*` constants in `config.py`, keep `observability.logging` stdlib-only, and gate on a fresh-interpreter import check (OBS-18).

---

## API Changes

No HTTP/API changes. `api.py` keeps `from agenticlog.history import HistoryStore` (now a shim). No endpoint, request, or response change.

---

## Frontend Changes

No frontend changes. `app.py` keeps `from agenticlog.rag import RAGSecurityError` (now a shim).

---

## Tests Required

**New unit tests** (each asserts canonical import + all identity pairs + behavior round-trip):
- `tests/test_shared_errors.py` — OBS-01..05
- `tests/test_observability_logging.py` — OBS-06..11, OBS-19
- `tests/test_observability_history.py` — OBS-12..17

**Existing tests that MUST stay green unchanged** (shims cover them):
- `tests/test_rag.py`, `tests/test_app.py`, `tests/test_rag_integration.py`, `tests/test_pdf_to_json.py` — `RAGSecurityError` consumers
- `tests/acceptance/test_document_ingestion_ui.py`, `tests/acceptance/test_multi_collection_chromadb.py` — `RAGSecurityError` consumers
- `tests/acceptance/test_structured_log_config.py` — `rag._JsonFormatter` via `_reload_rag` helper (reload identity, OBS-19)
- `tests/test_api.py` — `@patch("agenticlog.api.HistoryStore")` (11+ sites, consumer namespace, OBS-17)
- `tests/acceptance/test_history_endpoint.py`, `tests/test_query_history_audit_logging.py` — `agenticlog.history` consumers
- `tests/test_rag_caracterizacao.py` — characterization oracle (unchanged, OBS-20)

**Golden gate:** `rag_eval` (retrieval golden set) — this feature touches no retrieval code, so it must remain green (OBS-20).

**Integration-level check (no external services):** a fresh-interpreter `import agenticlog` smoke test asserting no circular import (OBS-18).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/shared/__init__.py` | New | Package init; re-export `RAGSecurityError` with `__all__` (health.py precedent). |
| `src/agenticlog/shared/errors.py` | New | Canonical home of `RAGSecurityError` (moved verbatim from `rag.py` L87–92). |
| `src/agenticlog/observability/__init__.py` | New | Package init; re-export `_JsonFormatter` and `HistoryStore` with `__all__`; stdlib-only, must not import `config`. |
| `src/agenticlog/observability/logging.py` | New | Canonical home of `_JsonFormatter` (moved verbatim from `config.py` L166–177); imports only `json`, `logging`, `datetime`. |
| `src/agenticlog/observability/history.py` | New | Canonical home of `HistoryStore` (whole `history.py` body, stdlib-only). |
| `tests/test_shared_errors.py` | New | Unit test for OBS-01..05. |
| `tests/test_observability_logging.py` | New | Unit test for OBS-06..11, OBS-19. |
| `tests/test_observability_history.py` | New | Unit test for OBS-12..17. |
| `src/agenticlog/rag.py` | Modified (shim) | Replace `RAGSecurityError` class body (L87–92) with `from agenticlog.shared.errors import RAGSecurityError` + shim comment. Keeps `from agenticlog.config import _JsonFormatter` (L46) unchanged — identity flows through. |
| `src/agenticlog/config.py` | Modified (shim) | Replace `_JsonFormatter` class body (L166–177) with `from agenticlog.observability.logging import _JsonFormatter` + shim comment. Keep `LOG_*` / `_VALID_*` constants. Remove now-unused imports (`json`, `datetime` — verify via ruff) at top. |
| `src/agenticlog/history.py` | Modified (shim) | Replace body with `from agenticlog.observability.history import HistoryStore` + module docstring note + shim comment. Keep public name available. |
| `src/agenticlog/__init__.py` | Verify no change | It exports `AgentState`, `agent_workflow`, health symbols only — none of the moved symbols. Confirm untouched (OBS-21). |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Circular import** `config → observability.logging → config` | HIGH (the one real risk) | Keep `LOG_LEVEL/LOG_FORMAT/_VALID_*` in `config.py` (Checkpoint 1). Keep `observability.logging` and `observability/__init__.py` stdlib-only — no `config` import. Gate on fresh-interpreter `import agenticlog` (OBS-18) and an explicit assert that `agenticlog.config` is not in the imports of `observability.logging`. |
| **Reload breaks object identity** | MEDIUM | `test_structured_log_config.py::_reload_rag` reloads `config` + `rag` but NOT `observability.logging`; the `config` shim re-binds to the already-loaded canonical module attribute, so identity holds. Add an explicit reload-identity assertion (OBS-19). Do NOT reload `observability.logging` independently in tests, or two class objects would diverge. |
| **`@patch` namespace safety** | MEDIUM | `test_api.py` patches `agenticlog.api.HistoryStore` (consumer namespace, set by `api.py`'s `from agenticlog.history import HistoryStore`). Since the shim re-exports the identical class object, `api.HistoryStore` is unchanged and patches still target the right name. No `@patch("agenticlog.rag.RAGSecurityError")` exists in the suite, so the errors move is patch-safe too. |
| **Unused imports after move** | LOW | `json`/`datetime` in `config.py` are used only by `_JsonFormatter`; after the move they become dead imports. Remove them (ruff-verified) as part of the `config.py` shim task. |
| **Shim removal deferred** | LOW (tracked debt) | Shims are intentionally left at `rag.py`/`config.py`/`history.py` and marked `# Re-export shim (ADR-018 Fase 2) — remover na Fase 6`. Removal is ADR-018 Fase 6 (test rewrite), explicitly out of scope here. |
| **Characterization oracle / rag_eval regression** | LOW | No retrieval or ingestion behavior changes; run the full suite + `tests/test_rag_caracterizacao.py` + `rag_eval` as the transition safety net (ADR-018 §5). |
| **CLAUDE.md conflicts** | LOW | Coding-style honored: new modules are small (<50-line functions, well under 800-line file cap), fully type-hinted, Portuguese docstrings, no silent fails (bodies moved verbatim). No conflict found. |

---

## Open Questions

None. The three Checkpoint-1 open questions were resolved by the user:
1. Move ONLY `_JsonFormatter`; `LOG_*`/`_VALID_*` stay in `config.py`. **Confirmed.**
2. `_configurar_logging_cli` stays in `rag.py` (moves in Fase 3). **Confirmed.**
3. `shared/__init__.py` AND `observability/__init__.py` re-export the symbols, with object-identity assertions covering those `__init__` re-exports. **Confirmed.**

---

## Success Criteria

- [ ] All 6 import paths (3 canonical + 3 shim) resolve, and all `__init__` re-exports resolve to the identical canonical object (`is` checks pass).
- [ ] `_JsonFormatter.format()`, `HistoryStore` schema/thread-safety/returns, and `RAGSecurityError` raise/catch are behaviorally identical (verified by new tests).
- [ ] Fresh `import agenticlog` succeeds with no circular import; `observability.logging` proven not to import `config`.
- [ ] Full suite (`pytest --cov=agenticlog -v`) + `tests/test_rag_caracterizacao.py` + `rag_eval` golden gate all green; test count ≥ pre-change baseline (no silent deletions).
- [ ] `ruff` clean (no unused imports left in `config.py`).
- [ ] `src/agenticlog/__init__.py` confirmed unchanged.
- [ ] Shims carry the `# Re-export shim (ADR-018 Fase 2) — remover na Fase 6` marker.
