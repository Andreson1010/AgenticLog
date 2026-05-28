# logging-module — Technical Spec

**Feature slug:** `logging-module`
**TLC scope:** medium
**Status:** Awaiting approval

---

## Problem Statement

`rag.py` uses `print()` throughout, making output uncontrollable when the module is imported as a library. Consumers cannot suppress or redirect messages, and there is no standard way to adjust verbosity. This contradicts the Python convention that library code must never write to stdout directly.

---

## Goals

- [ ] All `print()` calls in `rag.py` replaced with `logging` calls at the correct level (INFO / WARNING / ERROR) — zero stdout writes when imported as library
- [ ] `LOG_LEVEL = "INFO"` added to `config.py` and consumed by `__main__` block in `rag.py`
- [ ] `caplog`-based pytest tests capture all log records from `cria_vectordb()` without touching real I/O

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Log handler in `app.py` | Out of scope per ADR; Streamlit manages its own output |
| Structured/JSON logging | Not required for current operator needs |
| Log rotation | Infrastructure concern |
| `--log-level` CLI argument | `LOG_LEVEL` constant is sufficient |
| Changes to `agent.py` | Already uses correct logging pattern; zero print() calls |

---

## User Stories

### P1: Replace print() with logging in rag.py ⭐ MVP

**User Story**: As a developer importing `agenticlog.rag` as a library, I want all `print()` calls replaced with logging calls, so that my application controls stdout and log verbosity via standard Python logging configuration.

**Acceptance Criteria**:

1. (AC-01) WHEN `rag.py` imported as library AND `cria_vectordb()` runs successfully, THEN system SHALL write no output to stdout.
2. (AC-02) WHEN `rag.py` run as `__main__` AND `cria_vectordb()` completes, THEN system SHALL emit progress messages on console via `basicConfig` called with `LOG_LEVEL`.
3. (AC-03) WHEN `config.py` imported, THEN `LOG_LEVEL` SHALL be a string equal to `"INFO"`.
4. (AC-04) WHEN test uses `caplog` at INFO level AND `cria_vectordb()` runs, THEN records containing `"Gerando"` and `"Criado"` SHALL be in `caplog.records`.
5. (AC-05) WHEN `__main__` AND `RAGSecurityError` raised, THEN `logger.error` (not `print`) SHALL emit message before `SystemExit(1)`.
6. (AC-06) WHEN `__main__` AND generic `Exception` raised, THEN `logger.error` SHALL emit message before `SystemExit(1)`.
7. (AC-07) WHEN test uses `caplog` at WARNING level AND directory empty, THEN WARNING record SHALL be captured.
8. (AC-08) WHEN any `.py` in agenticlog uses logging, THEN `logger = logging.getLogger(__name__)` SHALL be at module level after imports.
9. (AC-09) WHEN log messages contain variable data, THEN `%s`-style formatting SHALL be used (not f-strings).
10. (AC-10) WHEN `rag.py` imported as library, THEN `logging.basicConfig` SHALL NOT be called.
11. (AC-11) WHEN `LOG_LEVEL` read from `config.py`, THEN used in `basicConfig` call in `__main__` block of `rag.py`.

**Independent Test**: Import `rag` with mocked internals, call `cria_vectordb()`, assert `sys.stdout` received nothing.

---

## Edge Cases

- WHEN `LOG_LEVEL = "WARNING"` set in `config.py`, THEN INFO messages suppressed in CLI — expected behavior
- WHEN `__main__` catches `RAGSecurityError`, THEN `logger.error("Erro de segurança: %s", e)` — f-string must be converted
- WHEN `__main__` catches generic `Exception`, THEN `logger.error("Erro ao criar banco vetorial: %s", e)` — same conversion
- WHEN `cria_vectordb()` returns early on empty documents, THEN exactly one WARNING record before return

---

## Requirement Traceability

| Requirement ID | AC | Status |
|----------------|----|--------|
| LG-01 | AC-01 | Pending |
| LG-02 | AC-02 | Pending |
| LG-03 | AC-03 | Pending |
| LG-04 | AC-04 | Pending |
| LG-05 | AC-05 | Pending |
| LG-06 | AC-06 | Pending |
| LG-07 | AC-07 | Pending |
| LG-08 | AC-08 | Pending |
| LG-09 | AC-09 | Pending |
| LG-10 | AC-10 | Pending |
| LG-11 | AC-11 | Pending |

---

## Data Model Changes

None.

---

## Process Flow

**Library import — no output:**
```
import agenticlog.rag
→ logger = logging.getLogger("agenticlog.rag") registered (no handlers)
→ cria_vectordb() → logger.info(...) → silently discarded → zero stdout
```

**CLI run — visible output:**
```
python -m agenticlog.rag
→ __main__: logging.basicConfig(level=LOG_LEVEL)  # "INFO" → root handler attached
→ cria_vectordb() → logger.info(...) → visible on console
```

**Failure — RAGSecurityError:**
```
→ except RAGSecurityError: logger.error("Erro de segurança: %s", e) → SystemExit(1)
```

**Failure — empty documents:**
```
→ loader.load() returns [] → logger.warning("Nenhum documento encontrado.") → return
```

---

## API Changes

None.

---

## Frontend Changes

None.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Modify | Add `LOG_LEVEL: str = "INFO"` |
| `src/agenticlog/rag.py` | Modify | Add logging import + logger, replace 5 print() calls, convert f-strings to %s, add basicConfig in `__main__` |
| `tests/test_rag.py` | Modify | Add 8 caplog-based tests |

`agent.py`, `app.py`, `health.py` — intentionally excluded.

---

## Tests Required

**New tests in `tests/test_rag.py`:**

- `teste_1_log_info_gerando_embeddings` — caplog INFO, cria_vectordb() mocked, asserts "Gerando" in records (LG-04)
- `teste_2_log_info_criado_com_sucesso` — caplog INFO, asserts "Criado" in records (LG-04)
- `teste_3_log_warning_nenhum_documento` — caplog WARNING, loader returns [], asserts WARNING record with "Nenhum documento" (LG-07)
- `teste_4_sem_stdout_quando_importado_como_biblioteca` — capsys, no basicConfig, asserts stdout empty (LG-01)
- `teste_5_log_level_em_config` — asserts `config.LOG_LEVEL == "INFO"` and `logging.getLevelName("INFO") == 20` (LG-03)
- `teste_6_logger_modulo_usa_dunder_name` — asserts `rag.logger.name == "agenticlog.rag"` (LG-08)
- `teste_7_erro_seguranca_usa_logger_error` — RAGSecurityError triggers logger.error (LG-05)
- `teste_8_excecao_generica_usa_logger_error` — generic Exception triggers logger.error (LG-06)

**Note on tests 7 & 8:** `__main__` block not directly importable; use `runpy.run_module("agenticlog.rag", run_name="__main__")` with mocked `cria_vectordb` side_effect, OR extract error-handling into testable helper.

---

## Risks

| Risk | Status |
|------|--------|
| Silent CLI if basicConfig omitted from `__main__` | Mitigated by AC-02 test |
| f-string → %s conversion missed on lines 183, 186 | Flagged — builder must convert both |
| `test_cria_vectordb_sem_documentos_retorna_cedo` may pass silently without caplog assertion | Low risk; existing mock assertions remain valid |
| agent.py accidentally modified | Spec explicitly excludes it |

---

## Open Questions

None. All 3 original open questions resolved:
- OQ1: `LOG_LEVEL = "INFO"` (ADR-003)
- OQ2: "Nenhum documento encontrado" = WARNING (ADR-004)
- OQ3: app.py handler = out of scope

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` passes, coverage >= 80%
- [ ] All 8 new tests pass
- [ ] `python -m agenticlog.rag` emits progress lines to console
- [ ] Library import produces zero bytes on `sys.stdout`
- [ ] `grep -n "print(" src/agenticlog/rag.py` returns no matches
- [ ] `config.LOG_LEVEL == "INFO"`
