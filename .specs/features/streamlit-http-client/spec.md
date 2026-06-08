# Streamlit HTTP Client — Technical Spec

**Path:** `.specs/features/streamlit-http-client/spec.md`
**TLC scope:** medium
**Based on story:** As a logistics operator, I want the Streamlit UI to query the FastAPI backend over HTTP instead of invoking the LangGraph workflow directly, so that the UI process is decoupled from the agent runtime and both can be started, scaled, and maintained independently.
**Status:** Awaiting human approval

---

## Problem Statement

`app.py` currently imports `AgentState`, `agent_workflow`, `check_lmstudio_health`, `LMStudioUnavailableError`, and `anthropic` and calls `agent_workflow.invoke()` directly inside the button handler. This hard-couples the Streamlit process to the LangGraph runtime, preventing independent deployment, scaling, or restart of each layer. The FastAPI backend already exposes `POST /query`; `app.py` must be migrated to call it over HTTP.

## Goals

- [ ] `app.py` sends `POST /query` via `httpx.post` and stores the JSON response in `st.session_state` — display behaviour identical to today.
- [ ] Every HTTP error code (422, 500, 503) and every network exception (`ConnectError`, `TimeoutException`) produces a distinct, Portuguese-language `st.error` message without mutating session state.
- [ ] All dead imports (`AgentState`, `agent_workflow`, `check_lmstudio_health`, `LMStudioUnavailableError`, `anthropic`) are removed from `app.py`.
- [ ] `API_CLIENT_TIMEOUT_SECONDS = 120` is the sole timeout constant used by `_consultar_api()`.
- [ ] All document field access uses dict-key syntax; zero attribute access on doc objects.
- [ ] Test suites `test_streamlit_ui.py` and `test_app_error_handler.py` are rewritten to patch `httpx.post` and cover all error branches; overall coverage stays at or above 80%.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Changing `POST /query` endpoint contract or `DocumentInfo` schema | Backend API unchanged per approved story |
| Adding `id` field to `DocumentInfo` | Out of scope per approved story |
| Async httpx (`AsyncClient`) | Out of scope per approved story |
| Persistent `httpx.Client` across requests | Out of scope per approved story |
| Client-side retry logic | Out of scope per approved story |
| Caching API responses | Out of scope per approved story |
| Modifying `_ingerir_documento`, `salvar_pdf_enviado`, `reconstruir_vectordb`, `adicionar_documento_incrementalmente` | Ingestion flow unchanged |
| UI layout changes beyond migration requirements | Out of scope per approved story |

---

## User Stories

### P1: HTTP query with success path ⭐ MVP

**User Story**: As a logistics operator, I want to submit a query through the Streamlit UI and receive a ranked answer, so that the UI works identically to today but calls the FastAPI backend over HTTP.

**Why P1**: Core decoupling goal; without this nothing else makes sense.

**Acceptance Criteria**:

1. WHEN the operator submits a non-empty query AND the API returns HTTP 200 THEN `app.py` SHALL have sent `POST http://{API_HOST}:{API_PORT}/query` with body `{"query": "<text>"}` and stored `ranked_response`, `confidence_score`, `next_step`, `retrieved_info` in `st.session_state`.
2. WHEN `retrieved_info` is rendered THEN each document SHALL be accessed as a plain dict using `doc["page_content"]` and `doc["metadata"].get("source", "Desconhecida")`, and the loop index `i` SHALL be used for all widget keys.
3. WHEN `confidence_score` is `None` in the response THEN the existing `or 0.0` guard SHALL be applied before `st.progress()`.
4. WHEN `next_step` value is not in `_ROTAS` THEN the existing guard SHALL suppress the route badge without error.

**Independent Test**: Mock `httpx.post` to return a 200 response with a valid JSON body; assert `st.session_state` is populated and `st.error` is never called.

---

### P2: HTTP error handling ⭐ MVP

**User Story**: As a logistics operator, I want to see a clear Portuguese error message for every failure mode, so that I know what to do next without reading raw tracebacks.

**Why P2**: Without error differentiation, 503 LMStudio and 503 vectordb look identical; operators cannot act.

**Acceptance Criteria**:

1. WHEN the API returns HTTP 503 with `{"detail": "LMStudio indisponível. Inicie o servidor e carregue o modelo."}` THEN `st.error` SHALL display that Portuguese message and session state SHALL NOT be mutated.
2. WHEN the API returns HTTP 503 with `{"detail": "Base vetorial não encontrada. Execute: python -m agenticlog.rag"}` THEN `st.error` SHALL display that Portuguese message and session state SHALL NOT be mutated.
3. WHEN the API returns HTTP 500 THEN `st.error` SHALL show a generic error message AND `st.expander("Detalhes do erro")` SHALL contain the raw `detail` string; session state SHALL NOT be mutated.
4. WHEN the API returns HTTP 422 THEN `st.error` SHALL show a validation error message; session state SHALL NOT be mutated.
5. WHEN `httpx.ConnectError` is raised THEN `st.error` SHALL show "Não foi possível conectar ao servidor FastAPI. Inicie com: uvicorn agenticlog.api:app"; session state SHALL NOT be mutated.
6. WHEN `httpx.TimeoutException` is raised THEN `st.error` SHALL show a timeout message; session state SHALL NOT be mutated.

**Independent Test**: For each branch, mock `httpx.post` with the appropriate status code or side effect; assert the exact error widget is rendered and session state keys retain their pre-call values.

---

### P3: Dead imports removed and config constant added

**User Story**: As a developer, I want `app.py` to import only what it uses and to derive the API URL from `config.py`, so that the module is clean and the URL is configurable without editing source code.

**Why P3**: Prevents import-time failures when `agenticlog.agent` dependencies are not installed in a UI-only deployment.

**Acceptance Criteria**:

1. WHEN `app.py` is imported THEN it SHALL NOT import `AgentState`, `agent_workflow`, `check_lmstudio_health`, `LMStudioUnavailableError`, or `anthropic`.
2. WHEN `_consultar_api()` builds the URL THEN it SHALL use `API_HOST` and `API_PORT` from `config.py` — no hardcoded host or port strings.
3. WHEN `_consultar_api()` sets the timeout THEN it SHALL use `API_CLIENT_TIMEOUT_SECONDS` from `config.py`; `LLM_TIMEOUT_SECONDS` SHALL NOT be referenced for this purpose.

**Independent Test**: Import `app` in a test and assert `AgentState` is not in `app.__dict__`; assert `config.API_CLIENT_TIMEOUT_SECONDS == 120`.

---

## Edge Cases

- WHEN query is empty string AND submitted THEN API returns 422 → 422 handler fires (not a crash).
- WHEN `retrieved_info` is an empty list in 200 response THEN "Nenhum documento relacionado encontrado." renders correctly.
- WHEN `confidence_score` is `None` in 200 response THEN `or 0.0` guard prevents `st.progress(None)`.
- WHEN `next_step` value is absent from `_ROTAS` THEN badge is silently suppressed.
- WHEN `doc["metadata"]` has no `"source"` key THEN `.get("source", "Desconhecida")` returns fallback.

---

## Requirement Traceability

| Requirement ID | Story | AC ref | Status |
|----------------|-------|--------|--------|
| SHC-01 | P1 | AC-01 | Pending |
| SHC-02 | P1 | AC-02 | Pending |
| SHC-03 | P1 | AC-01 (confidence guard) | Pending |
| SHC-04 | P1 | AC-01 (next_step guard) | Pending |
| SHC-05 | P2 | AC-03 | Pending |
| SHC-06 | P2 | AC-04 | Pending |
| SHC-07 | P2 | AC-05 | Pending |
| SHC-08 | P2 | AC-06 | Pending |
| SHC-09 | P2 | AC-07 | Pending |
| SHC-10 | P2 | AC-08 | Pending |
| SHC-11 | P3 | AC-09 | Pending |
| SHC-12 | P3 | AC-10 | Pending |
| SHC-13 | P3 | AC-11 | Pending |
| SHC-14 | P1 | AC-12 | Pending |

**ID format:** `SHC-[NUMBER]` — SHC = Streamlit HTTP Client

---

## Data Model Changes

No data model changes. `DocumentInfo` schema in `api.py` is not modified. The session state keys (`ranked_response`, `confidence_score`, `retrieved_info`, `next_step`) remain identical; their values now come from JSON deserialization rather than `AgentState`.

Post-migration, `retrieved_info` items are plain `dict` objects (from `httpx` JSON deserialization) rather than LangChain `Document` objects. All access must use `doc["page_content"]` and `doc["metadata"].get(...)` — no attribute access.

---

## Process / Background Flow

**Happy path:**
1. Operator types query and clicks "Enviar".
2. `app.py` button handler calls `_consultar_api(query)`.
3. `_consultar_api` calls `httpx.post(f"http://{API_HOST}:{API_PORT}/query", json={"query": query}, timeout=API_CLIENT_TIMEOUT_SECONDS)`.
4. `response.raise_for_status()` passes (HTTP 200).
5. `response.json()` returns dict with `ranked_response`, `confidence_score`, `next_step`, `retrieved_info`.
6. Button handler stores all four keys in `st.session_state`.
7. UI renders response, confidence bar, route badge, and document expanders using dict-key access.

**Failure path — 503 LMStudio:**
1. `httpx.post` returns HTTP 503; `_consultar_api` raises `httpx.HTTPStatusError`.
2. Button handler catches the error, reads `detail` from response JSON.
3. `detail` matches LMStudio sentinel string → `st.error` displays Portuguese LMStudio message.
4. Session state not mutated.

**Failure path — 503 vectordb:**
Same as above but `detail` matches vectordb sentinel string → corresponding Portuguese message shown.

**Failure path — 500:**
`detail` extracted from JSON body → `st.error` generic message + `st.expander("Detalhes do erro")` with raw detail.

**Failure path — 422:**
`st.error` displays validation error message; session state not mutated.

**Failure path — ConnectError:**
`httpx.ConnectError` raised (API not running) → `st.error` shows uvicorn start command.

**Failure path — Timeout:**
`httpx.TimeoutException` raised → `st.error` shows timeout message.

---

## API Changes

No new endpoints. The existing `POST /query` in `src/agenticlog/api.py` is the sole target. Contract (request body `{"query": str}`, response shape with `ranked_response`, `confidence_score`, `next_step`, `retrieved_info`) is unchanged.

---

## Frontend Changes

`app.py` changes:

1. **Remove imports**: `AgentState`, `agent_workflow`, `check_lmstudio_health`, `LMStudioUnavailableError`, `anthropic`.
2. **Add import**: `from agenticlog.config import API_HOST, API_PORT, API_CLIENT_TIMEOUT_SECONDS` (alongside existing config imports).
3. **Add function** `_consultar_api(query: str) -> dict` — calls `httpx.post`, calls `response.raise_for_status()`, returns `response.json()`. Raises `httpx.HTTPStatusError` or network exceptions; does not catch them (caller handles).
4. **Rewrite button handler**: replace `check_lmstudio_health()` + `agent_workflow.invoke()` with `_consultar_api(query)`; branch on `httpx.HTTPStatusError.response.status_code` for 503/500/422; catch `httpx.ConnectError` and `httpx.TimeoutException` separately.
5. **Add module-level string constants** for each error message (follow existing pattern in `app.py`).
6. **Fix document loop**: replace `doc.metadata.get(...)`, `doc.page_content`, and `doc.id` with `doc["metadata"].get(...)`, `doc["page_content"]`, and index `i` respectively.

`config.py` changes:

1. **Add constant** `API_CLIENT_TIMEOUT_SECONDS: int = 120` in the `# API Server` section, after `API_PORT`.

No changes to layout, sidebar, ingestion expander, or any other UI element.

---

## Tests Required

**Unit tests — `tests/test_streamlit_ui.py` (full rewrite)**

All 10 existing tests are obsolete (they mock `agent_workflow.invoke`). Replace with tests that patch `httpx.post` at the `app` module namespace:

- `teste_1_consulta_sucesso` — mock 200, assert session state populated.
- `teste_2_documentos_renderizados_como_dict` — mock 200 with docs, assert dict-key access paths.
- `teste_3_retrieved_info_vazio` — mock 200 empty list, assert fallback message.
- `teste_4_confidence_none` — mock 200 with `confidence_score: null`, assert `or 0.0` guard.
- `teste_5_next_step_invalido` — mock 200 with unknown `next_step`, assert no badge error.

**Unit tests — `tests/test_app_error_handler.py` (full rewrite)**

All 5 existing tests are obsolete (they use `isinstance`-based logic). Replace:

- `teste_1_erro_503_lmstudio` — mock 503 with LMStudio detail, assert correct `st.error` message.
- `teste_2_erro_503_vectordb` — mock 503 with vectordb detail, assert correct `st.error` message.
- `teste_3_erro_500` — mock 500, assert generic error + expander.
- `teste_4_erro_422` — mock 422, assert validation message.
- `teste_5_connect_error` — raise `httpx.ConnectError`, assert uvicorn message.
- `teste_6_timeout` — raise `httpx.TimeoutException`, assert timeout message.

**Existing tests that must not break**

- `tests/test_app.py` — ingestion tests; import of `app` must succeed (dead imports removed means no `ImportError` from missing deps).
- `tests/test_api.py` and `tests/acceptance/test_api_query_endpoint.py` — untouched; must continue passing.

**Mock strategy**: Use `unittest.mock.patch("app.httpx.post")` returning a `MagicMock` with `.status_code`, `.json()`, and `.raise_for_status()` configured per test case. For error status codes, configure `raise_for_status()` to raise `httpx.HTTPStatusError` with a mock response attached.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `app.py` | Modify | Remove dead imports; add `_consultar_api()`; rewrite button handler; fix doc loop dict access; add error message constants |
| `src/agenticlog/config.py` | Modify | Add `API_CLIENT_TIMEOUT_SECONDS = 120` in API Server section |
| `tests/test_streamlit_ui.py` | Rewrite | Mock strategy fully obsolete; patch `httpx.post` instead of `agent_workflow` |
| `tests/test_app_error_handler.py` | Rewrite | `isinstance`-based error logic replaced by HTTP status-code branching |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `doc.id` on line ~235 and `doc.metadata.xxx` / `doc.page_content` attribute access will raise `AttributeError` at runtime after migration because `retrieved_info` items are now plain dicts | High | SHC-14 mandates dict-key access throughout the loop; tests `teste_2_documentos_renderizados_como_dict` validates this path |
| `test_streamlit_ui.py` mock strategy patches `agent_workflow.invoke` which will no longer exist in `app` namespace — all 10 tests will error, not just fail | High | Full rewrite of both test files is a required deliverable |
| 503 disambiguation: two distinct 503 bodies must be routed to different messages; a missing or malformed `detail` key in the 503 body would fall to the wrong branch | Medium | Branch on `detail` string equality/containment after safe `.get("detail", "")` |
| `httpx` is already listed in project dependencies (confirmed present in `app.py` imports) — no new dependency needed | Low | No action required |
| `API_HOST`/`API_PORT` already read from environment in `config.py` — URL construction is safe and configurable | Low | Confirm `_consultar_api()` imports from `config`, not re-reads `os.environ` |
| Removing `check_lmstudio_health()` from `app.py` leaves no pre-flight LMStudio check in the UI layer — the 503 response from the API is now the only signal | Low | Acceptable per approved story; AC-03 and AC-04 cover both 503 variants |
| `bandit` may flag `httpx.post` with a URL built from config variables as a potential SSRF (B113) | Low | URL base is derived from `config.py` constants defaulting to `127.0.0.1`; add `# nosec B113` comment if bandit raises it |

---

## Open Questions

None.

---

## Success Criteria

- [ ] `app.py` contains no imports of `AgentState`, `agent_workflow`, `check_lmstudio_health`, `LMStudioUnavailableError`, or `anthropic`.
- [ ] `config.py` exports `API_CLIENT_TIMEOUT_SECONDS = 120`.
- [ ] All document fields in `app.py` accessed via dict-key syntax.
- [ ] All 6 error branches (503×2, 500, 422, ConnectError, Timeout) have dedicated tests.
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` reports >= 80% coverage with zero failures.
- [ ] `ruff check app.py src/agenticlog/config.py` passes with no errors.
- [ ] `mypy app.py src/agenticlog/config.py` passes with no errors.
