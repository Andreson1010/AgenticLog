# Retry Logic with Exponential Backoff — Technical Spec

**Status:** Approved — In Implementation
**Story:** Logistics operator wants agent to retry failed LLM HTTP calls with exponential backoff so transient LMStudio errors do not crash the workflow.

---

## Problem Statement

When LMStudio is unavailable or slow to respond, `gera_multiplas_respostas()` and `usar_ferramenta_web()` raise unhandled HTTP exceptions that propagate through the LangGraph workflow and crash the Streamlit UI with an uninformative traceback. Transient network failures (connection refused, TCP stalls, remote disconnects) should be retried automatically before surfacing as user-facing errors.

---

## Goals

- [ ] `gera_multiplas_respostas()` retries each of its 5 LLM calls independently up to 3 times on transient HTTP errors, succeeding if any attempt succeeds.
- [ ] `usar_ferramenta_web()` retries the agent executor call up to 3 times on the same transient errors.
- [ ] After all retries are exhausted, the original exception propagates unchanged to `app.py`.
- [ ] `app.py` displays "LMStudio not running" for all transient connection errors, detected by exception type rather than string matching.
- [ ] All retry constants live exclusively in `config.py`. Zero hardcoded values in `agent.py`.
- [ ] TCP stalls are bounded: `ChatOpenAI` is initialized with `request_timeout=LLM_TIMEOUT_SECONDS`.

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Shared retry budget across the 5 calls in `gera_multiplas_respostas()` | Retry is per individual call |
| Circuit breaker / fallback response on full outage | Not in approved story |
| Retry for `retrieve_info()` or ChromaDB calls | ChromaDB is local disk |
| Retry for DuckDuckGo search tool | Only the LLM call inside `usar_ferramenta_web()` is in scope |

---

## Data Model Changes

No `AgentState` changes.

**`src/agenticlog/config.py`** — add to LLM section:

```python
LLM_TIMEOUT_SECONDS: float = 10.0        # per-call TCP timeout for ChatOpenAI
LLM_MAX_RETRY_ATTEMPTS: int = 3          # 1 original + 2 retries
LLM_RETRY_WAIT_INITIAL_SECONDS: float = 1.0  # initial backoff wait
LLM_RETRY_WAIT_MAX_SECONDS: float = 4.0     # backoff cap
```

---

## Process Flow

**Decorator `_llm_retry` (module-level in `agent.py`):**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

_llm_retry = retry(
    stop=stop_after_attempt(LLM_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(min=LLM_RETRY_WAIT_INITIAL_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS),
    retry=retry_if_exception_type((
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.RemoteProtocolError,
        anthropic.APIConnectionError,
    )),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
```

Retryable exceptions: `httpx.ConnectError`, `httpx.TimeoutException`, `httpx.RemoteProtocolError`, `anthropic.APIConnectionError`.
Non-retryable (immediate raise): `anthropic.AuthenticationError`, `anthropic.BadRequestError`.

**`gera_multiplas_respostas()`:** Extract `_invoke_chain(chain, inputs)` helper decorated with `_llm_retry`. Replace `qa_chain_dynamic.invoke(...)` with `_invoke_chain(qa_chain_dynamic, ...)`.

**`usar_ferramenta_web()`:** Two independent blocks:
```python
# bloco 1 — DuckDuckGo (silent fallback — existing behavior)
try:
    resultados = search.run(state.query)
except Exception as e:
    logger.warning("DuckDuckGo search failed: %s", e)
    state.ranked_response = "Busca indisponível no momento."
    return state

# bloco 2 — LLM call (exposed to _llm_retry)
@_llm_retry
def _invoke_executor(executor, inputs):
    return executor.invoke(inputs)
```

---

## API Changes

None. Purely internal resilience change.

---

## Frontend Changes

**`app.py`:** Add `import httpx`. Add `import anthropic`. Replace string-match with `isinstance` as primary check:

```python
if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, anthropic.APIConnectionError)):
    # show LMStudio-not-running message
elif "connection refused" in _msg or ("connect" in _msg and "1234" in _msg):
    # legacy string match fallback
```

---

## Tests Required

| # | Case | File |
|---|------|------|
| T1 | Success on 1st attempt (happy path) | `test_agentic_rag.py` |
| T2 | 1 failure → success on 2nd attempt | `test_agentic_rag.py` |
| T3 | 3 consecutive failures → original exception propagates (not RetryError) | `test_agentic_rag.py` |
| T4 | `AuthenticationError` → no retry, immediate failure | `test_agentic_rag.py` |
| T5 | `RemoteProtocolError` → retryable | `test_agentic_rag.py` |
| T6 | DuckDuckGo failure → returns fallback string, does NOT propagate | `test_agentic_rag.py` |
| T7 | `TimeoutException` respected per call | `test_agentic_rag.py` |
| T8 | `isinstance` classifies `APIConnectionError` correctly | `test_app_error_handler.py` |
| T9 | `isinstance` classifies `AuthenticationError` correctly (generic branch) | `test_app_error_handler.py` |
| T10 | `isinstance` does NOT capture non-LLM exception | `test_app_error_handler.py` |

---

## Files That Will Change

| File | Change |
|------|--------|
| `src/agenticlog/config.py` | +4 retry/timeout constants |
| `src/agenticlog/agent.py` | tenacity/httpx/anthropic imports, logger, `_llm_retry`, `request_timeout` in `_get_llm()`, `_invoke_chain`, `_invoke_executor`, refactored `usar_ferramenta_web()` |
| `app.py` | `import httpx`, `import anthropic`, `isinstance` checks for all 4 retryable types |
| `requirements.txt` | explicit pin for `httpx` and `anthropic` |
| `tests/test_agentic_rag.py` | +T1–T7, update `teste_10`, verify `teste_6`/`teste_6b` |
| `tests/test_app_error_handler.py` | NEW — T8, T9, T10 |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Decorator on node function (not helper) → 5 calls replayed on retry | Extract `_invoke_chain` helper; decorate that, not the node |
| TCP stall hangs indefinitely | `request_timeout=LLM_TIMEOUT_SECONDS` on `ChatOpenAI` |
| `app.py` string-match misses wrapped exceptions | `isinstance` as primary check |
| `anthropic.APIConnectionError` retryable but not classified in `app.py` | `app.py` must include `anthropic.APIConnectionError` in `isinstance` check |
| DuckDuckGo errors newly propagating after removing catch | Separate try/except blocks in `usar_ferramenta_web()` |
| `httpx` and `anthropic` as transitive deps | Pin explicitly in `requirements.txt` |

---

## Success Criteria

- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` passes at 80%+ on changed modules
- [ ] All 10 named tests pass
- [ ] `teste_10` asserts `request_timeout=LLM_TIMEOUT_SECONDS`
- [ ] No `tenacity.RetryError` ever reaches `app.py`
- [ ] `config.py` has exactly 4 new constants; no retry value in `agent.py`
- [ ] `app.py` uses `isinstance` covering all 4 retryable types including `anthropic.APIConnectionError`
- [ ] `before_sleep_log(logger, logging.WARNING)` in `_llm_retry`
