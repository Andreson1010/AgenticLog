# health-check-lmstudio — Technical Spec

**Feature slug:** `health-check-lmstudio`
**TLC scope:** medium
**Status:** Implemented (except HC-10 — acceptance tests missing)

---

## Problem Statement

When LMStudio is not running, `agent_workflow.invoke()` enters tenacity's retry cycle (3 attempts, up to ~9 seconds of wait) before raising an HTTP connection error that `app.py` can only classify via string matching as a fallback. The operator has no early signal that the local LLM server is down. This feature adds a fast-fail probe — one GET to `/v1/models` before the workflow — so the operator receives an explicit, actionable error message within 5 seconds instead of waiting through retry cycles to receive a cryptic failure.

---

## Goals

- [x] `check_lmstudio_health()` probes `{LLM_API_BASE}/v1/models` with a single GET (no retries) before `agent_workflow.invoke()` is called
- [x] `LMStudioUnavailableError` is raised on connection error, timeout, or non-2xx response, and is recognised by `app.py`'s `isinstance` check to display a clear Portuguese-language message
- [x] All timeout and URL constants read from `config.py`; zero hardcoded values in `health.py`
- [ ] Acceptance test file `tests/acceptance/test_health_check.py` covering AC-HC-01 to AC-HC-04

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Retry on health check | Fast-fail probe by design; retry already handled by tenacity on workflow calls |
| Background polling / periodic health monitoring | Not in approved story |
| Checking if specific model is loaded | ADR-001: HTTP 200 with empty model list is treated as healthy |
| Caching result across button clicks | Each "Enviar" click performs its own probe |
| Health check at module import time | Lazy pattern — only runs inside the button block |

---

## User Stories

### P1: Fast-fail LMStudio probe before workflow ⭐ MVP

**User Story**: As a logistics operator, I want the system to verify LMStudio is reachable before executing a workflow query, so that I receive an immediate, clear error message instead of waiting through retry cycles only to get a cryptic connection failure.

**Acceptance Criteria**:

1. (AC-01) WHEN LMStudio is running and `/v1/models` returns 2xx, THEN `check_lmstudio_health()` SHALL return without raising AND `agent_workflow.invoke()` SHALL proceed normally.
2. (AC-02) WHEN `check_lmstudio_health()` receives `httpx.ConnectError`, THEN it SHALL raise `LMStudioUnavailableError` AND `app.py` SHALL display "LMStudio não está rodando. Inicie o LMStudio e carregue o modelo hermes-3-llama-3.2-3b antes de usar o sistema." without invoking the workflow.
3. (AC-03) WHEN LMStudio takes longer than `LLM_HEALTH_CHECK_TIMEOUT_SECONDS`, THEN SHALL raise `LMStudioUnavailableError` with a message containing "tempo limite" AND workflow SHALL NOT be invoked.
4. (AC-04) WHEN `LMStudioUnavailableError` is raised, THEN `app.py` `isinstance` check SHALL recognise it and route to the LMStudio error message, NOT the generic fallback.
5. (AC-05) WHEN HTTP response `is_success` is False, THEN SHALL raise `LMStudioUnavailableError` with the status code in the message.
6. (AC-06) WHEN `check_lmstudio_health()` is called, THEN it SHALL perform exactly ONE GET to `{LLM_API_BASE}/v1/models` with NO retries.
7. (AC-07) WHEN timeout constant is needed, THEN it SHALL be read exclusively from `config.py` as `LLM_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0`.
8. (AC-08) WHEN `check_lmstudio_health()` fails, THEN it SHALL call `logger.error` with URL and exception type before raising.
9. (AC-09) WHEN `check_lmstudio_health()` is defined in `health.py`, THEN it SHALL be exported via `agenticlog/__init__.py` alongside `AgentState` and `agent_workflow`.

**Independent Test**: Mock `httpx.Client` at `agenticlog.health` namespace; call `check_lmstudio_health()`; assert no exception for 2xx, `LMStudioUnavailableError` for connection error and timeout.

---

## Edge Cases

- WHEN `httpx.TimeoutException` is raised during GET, THEN `LMStudioUnavailableError` message SHALL contain "tempo limite"
- WHEN `httpx.RemoteProtocolError` is raised, THEN caught by same `(httpx.ConnectError, httpx.RemoteProtocolError)` branch
- WHEN HTTP 200 with empty `data` list is returned, THEN `is_success` is True and health check passes (ADR-001)
- WHEN `reset_health_check_sentinel()` is called, THEN `_health_checked` SHALL be False; used in `setUp`/`tearDown` of every test case
- Health check called inside "Enviar" button block, NOT at module import time

---

## Requirement Traceability

| Requirement ID | Story | AC | Status |
|----------------|-------|----|--------|
| HC-01 | P1: Fast-fail probe | AC-01 | Implemented |
| HC-02 | P1: Fast-fail probe | AC-02 | Implemented |
| HC-03 | P1: Fast-fail probe | AC-03 | Implemented |
| HC-04 | P1: Fast-fail probe | AC-04 | Implemented |
| HC-05 | P1: Fast-fail probe | AC-05 | Implemented |
| HC-06 | P1: Fast-fail probe | AC-06 | Implemented |
| HC-07 | P1: Fast-fail probe | AC-07 | Implemented |
| HC-08 | P1: Fast-fail probe | AC-08 | Implemented |
| HC-09 | P1: Fast-fail probe | AC-09 | Implemented |
| HC-10 | P1: Fast-fail probe | Acceptance tests AC-HC-01..04 | **MISSING** |

---

## Data Model Changes

None. `AgentState` unchanged. No new Pydantic fields.

---

## Process Flow

**Happy path:**
```
st.button("Enviar") clicked
→ check_lmstudio_health()
    → httpx.Client(timeout=5.0).get("{LLM_API_BASE}/v1/models")
    → response.is_success = True → return
→ agent_workflow.invoke(AgentState(query=query))
```

**Failure — ConnectError:**
```
→ check_lmstudio_health()
    → httpx.ConnectError raised
    → logger.error("... url=... exception_type=ConnectError")
    → raise LMStudioUnavailableError("LMStudio não está acessível.")
→ app.py except: isinstance(e, LMStudioUnavailableError) = True
→ st.error("LMStudio não está rodando. Inicie o LMStudio...")
→ agent_workflow.invoke() NOT called
```

**Failure — timeout / non-2xx:** same routing via `isinstance` check.

---

## API Changes

None. Internal HTTP probe only (client-side GET). No new server endpoints.

---

## Frontend Changes

`app.py` — already modified:
- `check_lmstudio_health()` and `LMStudioUnavailableError` imported from `agenticlog`
- `check_lmstudio_health()` called before `agent_workflow.invoke()`
- `LMStudioUnavailableError` added to `isinstance` check

---

## Files That Will Change

| File | Change type | Status |
|------|-------------|--------|
| `src/agenticlog/health.py` | Add | Done |
| `src/agenticlog/config.py` | Modify — `LLM_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0` | Done |
| `src/agenticlog/__init__.py` | Modify — export `check_lmstudio_health`, `LMStudioUnavailableError` | Done |
| `app.py` | Modify — import + call + isinstance | Done |
| `tests/test_health.py` | Add — 7 unit tests | Done |
| `tests/test_app_error_handler.py` | Modify — `teste_12_` classification | Done |
| `tests/acceptance/test_health_check.py` | Add — AC-HC-01..04 | **MISSING** |
| `docs/adr/ADR-001-health-check-empty-model-list.md` | Add | Done |
| `docs/adr/ADR-002-health-check-config-not-params.md` | Add | Done |

---

## Implementation Notes

The researcher brief placed `check_lmstudio_health()` in `agent.py`. Actual implementation extracted it to `src/agenticlog/health.py` — avoids circular imports, keeps `agent.py` focused on LangGraph workflow, isolates test mocking to `agenticlog.health` namespace. `__init__.py` imports from `agenticlog.health`.

Uses `httpx.Client` (sync context manager) rather than bare `httpx.get()` — ensures proper connection cleanup.

---

## Tests Required

**Unit tests (`tests/test_health.py` — done):**
- `teste_1_` happy path — 2xx returns without exception
- `teste_2_` ConnectError raises `LMStudioUnavailableError`; `logger.error` called
- `teste_3_` TimeoutException raises with "tempo limite" in message
- `teste_4_` HTTP 500 raises with "500" in message
- `teste_5_` HTTP 200 + empty data = healthy (ADR-001)
- `teste_6_` `reset_health_check_sentinel()` resets sentinel to False
- `teste_7_` export check via `__init__.py`

**Classification (`tests/test_app_error_handler.py` — done):**
- `teste_12_lmstudio_unavailable_error_classificado_como_lmstudio`

**Acceptance tests (`tests/acceptance/test_health_check.py` — MISSING):**
- `test_ac_hc_01_happy_path` — 2xx; health check passes; workflow called
- `test_ac_hc_02_connect_error_blocks_workflow` — ConnectError; workflow not called
- `test_ac_hc_03_timeout_blocks_workflow` — TimeoutException; message contains "tempo limite"
- `test_ac_hc_04_non_2xx_blocks_workflow` — is_success=False; LMStudioUnavailableError raised

---

## Risks

| Risk | Status |
|------|--------|
| `LMStudioUnavailableError` falls through to generic error if not in `isinstance` check | Mitigated — in check + guarded by `teste_12_` |
| Test isolation — wrong mock namespace | Mitigated — tests mock `agenticlog.health.httpx.Client` + reset sentinel in setUp/tearDown |
| Acceptance test gap (HC-10) | **Open** — no acceptance test file exists |

---

## Open Questions

- Should `tests/acceptance/test_health_check.py` be created now, or does unit test coverage (7 tests + 1 classification) satisfy the acceptance bar? — Resolve before closing feature.

---

## Success Criteria

- [x] `check_lmstudio_health()` in `src/agenticlog/health.py`, exported via `__init__.py`
- [x] `LMStudioUnavailableError` raised on ConnectError, TimeoutException, RemoteProtocolError, non-2xx
- [x] `logger.error` called on every failure path
- [x] `app.py` calls health check before workflow inside button block
- [x] `LMStudioUnavailableError` in `isinstance` check → clear Portuguese message
- [x] `LLM_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0` in `config.py`
- [x] HTTP 200 + empty list = healthy (ADR-001)
- [x] 7 unit tests pass; `reset_health_check_sentinel()` available for test setUp
- [x] `teste_12_` classification test passes
- [ ] `tests/acceptance/test_health_check.py` with AC-HC-01..04 passing
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` >= 80% with all tests passing
