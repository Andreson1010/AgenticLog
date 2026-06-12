# Structured Log Config — Technical Spec

**Path:** `.specs/features/structured-log-config/spec.md`
**TLC scope:** medium
**Based on story:** As a developer operating AgenticLog, I want LOG_LEVEL and LOG_FORMAT read from environment variables and configured in config.py, so I can control verbosity and output format at runtime without modifying source code.
**Status:** Awaiting human approval

---

## Problem Statement

The previous logging-module spec (`.specs/features/logging-module/`) introduced `logging.basicConfig` in `rag.py` and the `LOG_LEVEL` constant in `config.py`, but hardcoded both the level and format — no environment-variable override was wired up and no JSON formatter was provided. Operators cannot change log verbosity or format without editing source code, which blocks runtime observability in different deployment contexts (local dev, CI, production).

---

## Goals

- [ ] `config.LOG_LEVEL` is driven by the `LOG_LEVEL` env var, defaulting to `"INFO"` (AC-01, AC-02)
- [ ] `config.LOG_FORMAT` is driven by the `LOG_FORMAT` env var, defaulting to `"text"` (AC-03, AC-04)
- [ ] Both constants are validated at import time; invalid values raise `ValueError` (AC-07, AC-08)
- [ ] When `LOG_FORMAT=json`, `_executar_main()` in `rag.py` formats every log line as valid JSON with fields `timestamp`, `level`, `logger`, `message` (AC-05)
- [ ] When `LOG_FORMAT=text`, existing plain-text behavior is preserved (AC-06)

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| `setup_logging()` as a standalone module | Deferred by approved story |
| `app.py` / `__init__.py` logging setup | Explicitly excluded from story |
| New log statements in `agent.py` / `health.py` | Explicitly excluded from story |
| Log rotation and file handlers | Explicitly excluded from story |
| Third-party logging libraries (structlog, python-json-logger) | Excluded — stdlib only |

---

## User Stories

### P1: Runtime log control via environment variables ⭐ MVP

**User Story**: As a developer operating AgenticLog, I want `LOG_LEVEL` and `LOG_FORMAT` read from environment variables and reflected in `config.py`, so that I can control verbosity and output format at runtime without modifying source code.

**Why P1**: Enables CI pipelines and production deployments to tune logging without code changes; unblocks JSON log ingestion by log aggregators.

**Acceptance Criteria**:

1. WHEN `LOG_LEVEL` env var is not set THEN `config.LOG_LEVEL` SHALL equal `"INFO"` (AC-01)
2. WHEN `LOG_LEVEL=DEBUG` is set in the environment THEN `config.LOG_LEVEL` SHALL equal `"DEBUG"` at runtime (AC-02)
3. WHEN `LOG_FORMAT` env var is not set THEN `config.LOG_FORMAT` SHALL equal `"text"` (AC-03)
4. WHEN `LOG_FORMAT=json` is set in the environment THEN `config.LOG_FORMAT` SHALL equal `"json"` at runtime (AC-04)
5. WHEN `LOG_FORMAT=json` and `rag.py` runs as `__main__` THEN each log line emitted SHALL be valid JSON containing the fields `timestamp`, `level`, `logger`, and `message` (AC-05)
6. WHEN `LOG_FORMAT=text` THEN log output SHALL be plain text, preserving existing behavior (AC-06)
7. WHEN `LOG_LEVEL` is set to an unrecognised value (e.g. `"VERBOSE"`) THEN a `ValueError` SHALL be raised at module import time with no silent fallback (AC-07)
8. WHEN `LOG_FORMAT` is set to an unrecognised value (e.g. `"xml"`) THEN a `ValueError` SHALL be raised at module import time with no silent fallback (AC-08)

**Independent Test**: Set `LOG_FORMAT=json LOG_LEVEL=DEBUG` and run `python -m agenticlog.rag`; pipe output through `python -c "import sys,json; [json.loads(l) for l in sys.stdin]"` — must not raise.

---

## Edge Cases

- WHEN `LOG_LEVEL` env var is set to a lowercase valid name (e.g. `"debug"`) THEN the system SHALL accept it after `.upper()` normalisation and SHALL NOT raise `ValueError`.
- WHEN `LOG_FORMAT` env var is set with surrounding whitespace (e.g. `" json "`) THEN the system SHALL accept it after `.strip()` normalisation.
- WHEN `config.py` is imported multiple times in the same process THEN `LOG_LEVEL` and `LOG_FORMAT` SHALL reflect the values from the first import (standard Python module caching — no re-evaluation of env vars).

---

## Requirement Traceability

| Requirement ID | Story AC | Phase | Status |
|----------------|----------|-------|--------|
| SLC-01 | AC-01 | config.py — default LOG_LEVEL | Pending |
| SLC-02 | AC-02 | config.py — env-var LOG_LEVEL | Pending |
| SLC-03 | AC-03 | config.py — default LOG_FORMAT | Pending |
| SLC-04 | AC-04 | config.py — env-var LOG_FORMAT | Pending |
| SLC-05 | AC-05 | rag.py — JSON formatter in _executar_main | Pending |
| SLC-06 | AC-06 | rag.py — text formatter preserved | Pending |
| SLC-07 | AC-07 | config.py — ValueError on invalid LOG_LEVEL | Pending |
| SLC-08 | AC-08 | config.py — ValueError on invalid LOG_FORMAT | Pending |

**ID format:** `SLC-[NUMBER]` (Structured Log Config)

---

## Data Model Changes

### `src/agenticlog/config.py`

Add `import os` at the top. Replace the existing hardcoded `LOG_LEVEL` line and add `LOG_FORMAT` beneath it:

```python
# Logging
_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_LOG_FORMATS: frozenset[str] = frozenset({"text", "json"})

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
if LOG_LEVEL not in _VALID_LOG_LEVELS:
    raise ValueError(
        f"Invalid LOG_LEVEL={LOG_LEVEL!r}. Must be one of {sorted(_VALID_LOG_LEVELS)}."
    )

LOG_FORMAT: str = os.environ.get("LOG_FORMAT", "text").strip().lower()
if LOG_FORMAT not in _VALID_LOG_FORMATS:
    raise ValueError(
        f"Invalid LOG_FORMAT={LOG_FORMAT!r}. Must be one of {sorted(_VALID_LOG_FORMATS)}."
    )
```

No migration needed — constants only, no database schema involved.

---

## Process / Background Flow

### `_executar_main()` in `rag.py` — happy path (LOG_FORMAT=json)

1. `config.py` is imported → `LOG_LEVEL` and `LOG_FORMAT` are read from env vars and validated.
2. `_executar_main()` is called.
3. A `logging.Formatter` subclass (inline, stdlib only) serialises each `LogRecord` to a JSON string with keys `timestamp` (ISO-8601 from `record.created`), `level` (`record.levelname`), `logger` (`record.name`), `message` (`record.getMessage()`).
4. `logging.basicConfig` is called with `level=LOG_LEVEL` and `handlers=[StreamHandler with JsonFormatter]` when `LOG_FORMAT == "json"`.
5. All subsequent `logger.*` calls in `cria_vectordb()` emit valid JSON lines to `stderr`/`stdout`.

### `_executar_main()` — happy path (LOG_FORMAT=text)

Steps 1–2 same. Step 4: `logging.basicConfig(level=LOG_LEVEL)` is called unchanged (existing behavior).

### Failure path — invalid env var

`config.py` import raises `ValueError` before `_executar_main()` is reached. The process exits with an unhandled exception and a clear message naming the bad value.

---

## API Changes

No API changes. `rag.py` is a CLI entry point, not an HTTP service.

---

## Frontend Changes

No frontend changes.

---

## Tests Required

All new tests go in `tests/test_rag.py` following the `teste_N_` naming convention.

### New unit tests (monkeypatching env vars, no real I/O)

| Test | What it verifies | Req ID |
|------|-----------------|--------|
| `teste_N_log_level_default` | `config.LOG_LEVEL == "INFO"` when env var absent | SLC-01 |
| `teste_N_log_level_from_env` | `config.LOG_LEVEL == "DEBUG"` when `LOG_LEVEL=DEBUG` | SLC-02 |
| `teste_N_log_format_default` | `config.LOG_FORMAT == "text"` when env var absent | SLC-03 |
| `teste_N_log_format_from_env` | `config.LOG_FORMAT == "json"` when `LOG_FORMAT=json` | SLC-04 |
| `teste_N_log_format_json_output` | Each captured log line parses as JSON with required fields | SLC-05 |
| `teste_N_log_format_text_preserved` | Output is plain text (not parseable as JSON) when `LOG_FORMAT=text` | SLC-06 |
| `teste_N_invalid_log_level_raises` | `ValueError` raised on import when `LOG_LEVEL=VERBOSE` | SLC-07 |
| `teste_N_invalid_log_format_raises` | `ValueError` raised on import when `LOG_FORMAT=xml` | SLC-08 |

**Notes on implementation:**
- `config.py` is a module-level constant, so env-var override tests must use `importlib.reload(config)` inside a `monkeypatch.setenv` context, or patch `agenticlog.config.LOG_LEVEL` directly after reload.
- `teste_5` in existing tests asserts `config.LOG_LEVEL == "INFO"` — it passes as long as the test environment does not set `LOG_LEVEL`. Add a note in the test file that this test assumes `LOG_LEVEL` is unset.
- The 8 existing `TestLogging` tests use `assertLogs` which is format-agnostic — no changes required.
- Idempotency tests for handler registration are not needed (no handler management, `basicConfig` is called once per process run).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Modify | Add `import os`, replace hardcoded `LOG_LEVEL`, add validated `LOG_FORMAT` |
| `src/agenticlog/rag.py` | Modify | `_executar_main()` selects JSON or text formatter based on `LOG_FORMAT` |
| `tests/test_rag.py` | Modify | Add 8 new test functions for env-var overrides, JSON output, and ValueError cases |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `config.py` module-level `ValueError` crashes the entire process on bad env var — intended but must be documented in `.env.example` | Low | Document `LOG_LEVEL` and `LOG_FORMAT` in `.env.example`; clear error message names the bad value |
| `importlib.reload(config)` in tests may leave module in a bad state if the reload itself raises `ValueError` | Medium | Wrap reload in `pytest.raises` and restore env vars via `monkeypatch` (auto-undo after test) |
| `logging.basicConfig` is a no-op if any handler is already registered (e.g., when tests run multiple times in same process) | Medium | In `_executar_main()`, call `logging.root.handlers.clear()` before `basicConfig`, or use `force=True` (Python 3.8+) — project uses 3.12, so `force=True` is safe |
| JSON formatter adds a code dependency on `json` stdlib — no risk, already available | None | N/A |
| `teste_5` implicitly depends on `LOG_LEVEL` being unset in the test environment | Low | Add `monkeypatch.delenv("LOG_LEVEL", raising=False)` fixture to the test or add a comment |

---

## Open Questions

None. All acceptance criteria are fully specified and all affected files are known.

---

## Success Criteria

- [ ] `config.LOG_LEVEL` and `config.LOG_FORMAT` read from env vars with correct defaults (SLC-01 through SLC-04)
- [ ] Running `LOG_FORMAT=json python -m agenticlog.rag` produces only valid JSON log lines (SLC-05)
- [ ] Running without `LOG_FORMAT` set produces plain-text output identical to current behavior (SLC-06)
- [ ] Setting `LOG_LEVEL=BAD` or `LOG_FORMAT=bad` causes an immediate `ValueError` at startup (SLC-07, SLC-08)
- [ ] All 8 new tests pass and existing `TestLogging` tests remain green
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` reports >= 80% coverage on changed files
