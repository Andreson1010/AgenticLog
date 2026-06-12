# LLM Provider Portability — Technical Spec

**Path:** `.specs/features/llm-provider-portability/spec.md`
**TLC scope:** medium
**Based on story:** As a developer operating AgenticLog across different environments (local LMStudio, alternative OpenAI-compatible endpoints), I want LLM provider configuration (model name, retry-exception handling, LLM client typing) to be correctly portable and free of unrelated SDK dependencies.
**Status:** Awaiting human approval

---

## Problem Statement

`config.py` hardcodes `LLM_MODEL`, preventing override via environment variable. `agent.py` imports the unused `anthropic` SDK solely to reference `anthropic.APIConnectionError` in the `_llm_retry` retryable-exception tuple — but the actual configured provider is OpenAI-compatible (`ChatOpenAI` / LMStudio), so the real connection-error type raised is `openai.APIConnectionError`, not `anthropic.APIConnectionError`. This means connection failures from the real provider are **not retried** as intended, and the codebase carries a dead dependency (`anthropic==0.104.1`). Additionally, `_get_llm()` is typed to return the concrete `ChatOpenAI` class, coupling callers to a specific SDK class rather than the minimal interface `agent.py` actually uses.

---

## Goals

- [ ] `LLM_MODEL` is read from the `LLM_MODEL` environment variable, falling back to the hardcoded default `"hermes-3-llama-3.2-3b"` when unset or empty string.
- [ ] `_llm_retry`'s retryable exception tuple includes `openai.APIConnectionError` instead of `anthropic.APIConnectionError`, so real OpenAI-compatible connection failures are retried with the existing backoff policy.
- [ ] `agent.py` no longer imports `anthropic`; `anthropic==0.104.1` is removed from `requirements.txt`.
- [ ] `_get_llm()`'s return type annotation is a minimal `typing.Protocol` (structural type) instead of the concrete `ChatOpenAI` class; runtime behaviour (construction, singleton, constructor args) is unchanged.

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Additional LLM providers beyond OpenAI-compatible (real Anthropic Claude integration) | Not requested; Protocol is structural typing only |
| Changes to `LLM_API_BASE`, `LLM_API_KEY`, or other LLM connection env vars | Story scopes only `LLM_MODEL` |
| Changes to retry counts, backoff timings, or `LLM_TIMEOUT_SECONDS` | Explicitly unchanged per ACs |
| Wrapper/adapter class or abstraction layer around `ChatOpenAI` | Story requires structural Protocol only, no subclass/wrapper |
| Validation or restriction of acceptable `LLM_MODEL` values | Any non-empty string accepted as-is |
| `data/vectordb/` rebuild or embedding/chunking changes | Unrelated to this feature |
| Auditing/removing other unused deps in `requirements.txt` beyond `anthropic` | Out of scope; only `anthropic` confirmed unused by research |

---

## User Stories

### P1: `LLM_MODEL` is configurable via environment variable ⭐ MVP

**User Story**: As a developer, I want `LLM_MODEL` to be overridable via an `LLM_MODEL` environment variable (with empty string treated as unset), so that I can point AgenticLog at a different model name without editing source code.

**Why P1**: Core portability requirement — the only behavioral change a deploying developer directly interacts with.

**Acceptance Criteria**:

1. WHEN `LLM_MODEL` is unset in the environment AND `config.py` is loaded THEN `LLM_MODEL` SHALL equal the hardcoded default literal `"hermes-3-llama-3.2-3b"`.
2. WHEN `LLM_MODEL` is set to a non-empty value (e.g. `"my-custom-model"`) in the environment AND `config.py` is loaded THEN `LLM_MODEL` SHALL equal that value.
3. WHEN `LLM_MODEL` is set to the empty string (`LLM_MODEL=""`) in the environment AND `config.py` is loaded THEN `LLM_MODEL` SHALL equal the hardcoded default literal `"hermes-3-llama-3.2-3b"` (empty string treated as unset).
4. WHEN `LLM_MODEL` is set to a whitespace-only value (e.g. `" "`) in the environment AND `config.py` is loaded THEN `LLM_MODEL` SHALL equal that whitespace value verbatim (NOT treated as unset — only the exact empty string `""` triggers fallback).

**Independent Test**: `python -c "import os; os.environ['LLM_MODEL']='my-custom-model'; import agenticlog.config as c; print(c.LLM_MODEL)"` prints `my-custom-model`. With `LLM_MODEL` unset or `LLM_MODEL=""`, prints `hermes-3-llama-3.2-3b`.

---

### P1: Retry policy targets the real OpenAI-compatible connection error ⭐ MVP

**User Story**: As a developer, I want `_llm_retry`'s retryable exception tuple to include `openai.APIConnectionError` (the exception the configured `ChatOpenAI`/LMStudio client actually raises), so that connection failures from the real provider are retried with the existing exponential-backoff policy instead of silently NOT being retried.

**Why P1**: This is the functional bug fix — the current `anthropic.APIConnectionError` entry in `_llm_retry`'s tuple is dead code that never matches exceptions raised by the configured `ChatOpenAI` client, meaning real connection failures bypass retry entirely.

**Acceptance Criteria**:

1. WHEN the LLM chain (`prompt | llm | parser`) is invoked via `_invoke_chain` AND the underlying OpenAI-compatible client raises `openai.APIConnectionError` on the first attempt AND a subsequent attempt does not raise THEN `_invoke_chain` SHALL retry per the existing `_llm_retry` backoff policy (up to `LLM_MAX_RETRY_ATTEMPTS`) and SHALL return the successful result.
2. WHEN the LLM chain is invoked via `_invoke_chain` AND the underlying client raises `openai.APIConnectionError` on every attempt up to `LLM_MAX_RETRY_ATTEMPTS` THEN `_invoke_chain` SHALL re-raise the original `openai.APIConnectionError` (not wrapped in `tenacity.RetryError`), consistent with the existing `reraise=True` behaviour for `httpx.ConnectError`, `httpx.TimeoutException`, and `httpx.RemoteProtocolError`.
3. WHEN `_llm_retry`'s retryable exception tuple is inspected THEN it SHALL contain `httpx.ConnectError`, `httpx.TimeoutException`, `httpx.RemoteProtocolError`, and `openai.APIConnectionError` — and SHALL NOT contain `anthropic.APIConnectionError`.

**Independent Test**: Mock `chain.invoke` to raise `openai.APIConnectionError` once then return a value; call `_invoke_chain(chain, inputs)`; assert it returns the value and `chain.invoke` was called twice. Mock `chain.invoke` to always raise `openai.APIConnectionError`; assert `_invoke_chain` raises `openai.APIConnectionError` (not `RetryError`) after exactly `LLM_MAX_RETRY_ATTEMPTS` calls.

---

### P2: Remove unused `anthropic` dependency

**User Story**: As a developer, I want `agent.py`, `tests/test_agentic_rag.py`, and `requirements.txt` to be free of the unused `anthropic` package, so that the codebase has no unrelated SDK dependencies and `import agenticlog.agent` does not pull in an unnecessary library.

**Why P2**: Cleanup that follows directly from P1 (once `anthropic.APIConnectionError` is no longer referenced, the import is dead). Lower priority because it has no runtime behavioral effect beyond P1, but is required by the story's business rules.

**Acceptance Criteria**:

1. WHEN `src/agenticlog/agent.py` is inspected THEN it SHALL NOT contain `import anthropic` or any reference to `anthropic`.
2. WHEN `agenticlog.agent` is imported THEN no `ImportError`/`ModuleNotFoundError` SHALL occur.
3. WHEN `tests/test_agentic_rag.py` is inspected THEN it SHALL NOT contain `import anthropic` (line 15) or any reference to `anthropic`; any test logic that relied on it SHALL be updated to use `openai.APIConnectionError` where applicable.
4. WHEN `requirements.txt` is inspected THEN it SHALL NOT contain the line `anthropic==0.104.1`.

**Independent Test**: `grep -ri anthropic src/agenticlog/agent.py tests/test_agentic_rag.py requirements.txt` returns no matches. `python -c "import agenticlog.agent"` succeeds.

---

### P2: `_get_llm()` returns a minimal structural Protocol type

**User Story**: As a developer, I want `_get_llm()`'s declared return type to be a minimal `typing.Protocol` covering only the operations `agent.py` actually uses on the LLM client (`__or__`/`__ror__` for chain composition, `invoke`), instead of the concrete `ChatOpenAI` class, so that the typed interface reflects actual usage and is provider-agnostic at the type-checking level.

**Why P2**: Type-annotation-only improvement; no runtime behavior change. Lower priority than the functional retry fix (P1) but part of the story's "LLM client typing" portability goal and required by its business rules.

**Acceptance Criteria**:

1. WHEN `_get_llm()` is called THEN it SHALL return a `ChatOpenAI` instance constructed exactly as before — same constructor arguments: `model_name=LLM_MODEL`, `openai_api_base=LLM_API_BASE`, `openai_api_key=LLM_API_KEY`, `temperature=LLM_TEMPERATURE`, `max_tokens=LLM_MAX_TOKENS`, `request_timeout=LLM_TIMEOUT_SECONDS`.
2. WHEN `_get_llm()`'s signature is inspected THEN its declared return type SHALL be the new minimal Protocol type (not `ChatOpenAI`).
3. WHEN the new Protocol type is inspected THEN it SHALL define only: `__or__`, `__ror__` (for `current_prompt | _get_llm() | StrOutputParser()` and `_prompt_web | _get_llm() | StrOutputParser()` pipe composition), and `invoke`.
4. WHEN `ChatOpenAI` is checked against the Protocol THEN it SHALL satisfy it structurally — no wrapper, adapter, or subclass is introduced.
5. WHEN existing tests that `patch("agenticlog.agent.ChatOpenAI")` (`teste_9_import_sem_lmstudio`, `teste_10_get_llm_singleton` in `tests/test_agentic_rag.py`, and `test_ac06_llm_created_with_timeout_from_config` in `tests/acceptance/test_retry_logic.py`) run THEN they SHALL continue to pass without modification to their patch targets.

**Independent Test**: `from agenticlog.agent import _get_llm; import inspect; print(inspect.signature(_get_llm).return_annotation)` shows the new Protocol type name, not `ChatOpenAI`. `pytest tests/test_agentic_rag.py::TestAgenticRAG::teste_9_import_sem_lmstudio tests/test_agentic_rag.py::TestAgenticRAG::teste_10_get_llm_singleton tests/acceptance/test_retry_logic.py -k test_ac06_llm_created_with_timeout_from_config` passes unmodified.

---

## Edge Cases

- WHEN `LLM_MODEL` is set to a whitespace-only string (e.g. `" "`) THEN system SHALL treat it as a non-empty value and use it verbatim — only the exact empty string `""` triggers fallback to the default.
- WHEN `_llm_retry`'s exception tuple is changed THEN `LLM_MAX_RETRY_ATTEMPTS=3`, `LLM_RETRY_WAIT_INITIAL_SECONDS=1.0`, `LLM_RETRY_WAIT_MAX_SECONDS=4.0` SHALL remain unchanged (no retry-bound changes in this feature).
- WHEN a retry occurs (any retryable exception type) THEN `before_sleep_log` (WARNING level) logging behaviour SHALL be unaffected by the `anthropic.APIConnectionError` → `openai.APIConnectionError` exception-type swap.
- WHEN `anthropic==0.104.1` is removed from `requirements.txt` THEN no other module in `src/` or `tests/` SHALL import `anthropic` (verified by research: only `agent.py` and `tests/test_agentic_rag.py` import it, both removed by this feature; not listed as a direct dependency in `pyproject.toml`).
- WHEN the new Protocol type is defined THEN it SHALL be structural/typing-only with no runtime enforcement beyond an optional `@runtime_checkable` decorator if structural `isinstance` checks are needed — it SHALL NOT change the mocking strategy: `patch("agenticlog.agent.ChatOpenAI")` continues to work because the patch target is the `ChatOpenAI` name in `agent.py`'s namespace, unaffected by a return-type annotation change on `_get_llm()`.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| LLMPORT-01 | P1 (LLM_MODEL unset → default) | Implementation | Pending |
| LLMPORT-02 | P1 (LLM_MODEL set → override) | Implementation | Pending |
| LLMPORT-03 | P1 (LLM_MODEL="" → default) | Implementation | Pending |
| LLMPORT-04 | P1 (LLM_MODEL=" " → verbatim) | Implementation | Pending |
| LLMPORT-05 | P1 (openai.APIConnectionError retried, succeeds on retry) | Implementation | Pending |
| LLMPORT-06 | P1 (openai.APIConnectionError exhausts retries → re-raised) | Implementation | Pending |
| LLMPORT-07 | P1 (retry tuple contains openai.APIConnectionError, not anthropic) | Implementation | Pending |
| LLMPORT-08 | P2 (no `anthropic` import in agent.py) | Implementation | Pending |
| LLMPORT-09 | P2 (agent.py imports cleanly, no ImportError) | Implementation | Pending |
| LLMPORT-10 | P2 (no `anthropic` import in test_agentic_rag.py) | Implementation | Pending |
| LLMPORT-11 | P2 (anthropic==0.104.1 removed from requirements.txt) | Implementation | Pending |
| LLMPORT-12 | P2 (_get_llm() returns ChatOpenAI, same constructor args) | Implementation | Pending |
| LLMPORT-13 | P2 (_get_llm() return annotation is new Protocol type) | Implementation | Pending |
| LLMPORT-14 | P2 (Protocol defines only __or__, __ror__, invoke) | Implementation | Pending |
| LLMPORT-15 | P2 (ChatOpenAI satisfies Protocol structurally, no wrapper) | Implementation | Pending |
| LLMPORT-16 | P2 (existing ChatOpenAI-patching tests pass unmodified) | Implementation | Pending |

**ID format:** `LLMPORT-[NUMBER]`

---

## Data Model Changes

No data model changes. `data/vectordb/` is not affected — no chunking, embedding, or document-ingestion changes.

---

## Process / Background Flow

**Happy path — `config.py` load with `LLM_MODEL` unset:**

1. `config.py` module body executes; `load_dotenv()` has already run (existing line 16).
2. `LLM_MODEL = os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL` evaluates: `os.environ.get("LLM_MODEL")` returns `None` (key absent) → `None or DEFAULT_LLM_MODEL` → `DEFAULT_LLM_MODEL`.
3. `LLM_MODEL == "hermes-3-llama-3.2-3b"`.

**Happy path — `config.py` load with `LLM_MODEL="my-custom-model"`:**

1. `os.environ.get("LLM_MODEL")` returns `"my-custom-model"` (truthy, non-empty).
2. `"my-custom-model" or DEFAULT_LLM_MODEL` → `"my-custom-model"` (short-circuit on truthy left operand).
3. `LLM_MODEL == "my-custom-model"`.

**Failure path — `LLM_MODEL=""` (empty string, treated as unset):**

1. `os.environ.get("LLM_MODEL")` returns `""` (key present but empty — falsy in Python).
2. `"" or DEFAULT_LLM_MODEL` → `DEFAULT_LLM_MODEL` (empty string is falsy, short-circuit falls through to right operand).
3. `LLM_MODEL == "hermes-3-llama-3.2-3b"`.

**Happy path — `_invoke_chain` retries `openai.APIConnectionError` and succeeds:**

1. `gera_multiplas_respostas` or `usar_ferramenta_web` builds `chain = current_prompt | _get_llm() | StrOutputParser()`.
2. `_invoke_chain(chain, inputs)` (decorated by `_llm_retry`) calls `chain.invoke(inputs)`.
3. First call raises `openai.APIConnectionError` (e.g. LMStudio temporarily unreachable).
4. `_llm_retry`'s `retry_if_exception_type` tuple now includes `openai.APIConnectionError` → tenacity catches it, logs via `before_sleep_log` at WARNING, waits per `wait_exponential(min=LLM_RETRY_WAIT_INITIAL_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS)`.
5. Second call to `chain.invoke(inputs)` succeeds, returns the response string.
6. `_invoke_chain` returns the response; caller proceeds normally.

**Failure path — `_invoke_chain` exhausts retries on `openai.APIConnectionError`:**

1. Same as above through step 4, but every attempt (up to `LLM_MAX_RETRY_ATTEMPTS=3`) raises `openai.APIConnectionError`.
2. `stop_after_attempt(LLM_MAX_RETRY_ATTEMPTS)` is reached.
3. Because `reraise=True`, tenacity re-raises the **original** `openai.APIConnectionError` (not `tenacity.RetryError`).
4. Caller (e.g. `gera_multiplas_respostas`) sees `openai.APIConnectionError` propagate — same propagation contract as `httpx.ConnectError` today.

---

## API Changes

No API changes. `src/agenticlog/api.py` (FastAPI) is not modified by this feature — `_get_llm()`'s signature change is annotation-only and `_invoke_chain`'s retryable-exception set change does not alter `_invoke_chain`'s external contract (still raises on exhausted retries, still returns `str` on success).

---

## Frontend Changes

No frontend changes. `app.py` is not modified — it does not import `anthropic` directly (confirmed by `TestAC09NoDeadImports`), and does not reference `_get_llm` or `ChatOpenAI`.

---

## Tests Required

### Unit — `tests/test_config_env.py`

Add new test methods to `TestConfigEnv` following the existing `_reload()` pattern (lines 25-48):

| Test | What it verifies | Requirement |
|------|-------------------|--------------|
| `teste_7_llm_model_unset_uses_default` | `LLM_MODEL` env var absent → `cfg.LLM_MODEL == "hermes-3-llama-3.2-3b"` | LLMPORT-01 |
| `teste_8_llm_model_set_uses_override` | `LLM_MODEL="my-custom-model"` → `cfg.LLM_MODEL == "my-custom-model"` | LLMPORT-02 |
| `teste_9_llm_model_empty_string_uses_default` | `LLM_MODEL=""` → `cfg.LLM_MODEL == "hermes-3-llama-3.2-3b"` | LLMPORT-03 |
| `teste_10_llm_model_whitespace_is_verbatim` | `LLM_MODEL=" "` → `cfg.LLM_MODEL == " "` (not treated as unset) | LLMPORT-04 |

Use `_reload({"LLM_MODEL": "..."})` and `_reload({}, remove_keys=("LLM_MODEL",))` exactly as the existing `OPENAI_API_KEY`/`OPENAI_API_BASE` tests do.

### Unit — `tests/test_agentic_rag.py`

| Test | What it verifies | Requirement |
|------|-------------------|--------------|
| Remove `import anthropic` (line 15) | Module imports cleanly without `anthropic` | LLMPORT-08, LLMPORT-09, LLMPORT-10 |
| New `teste_X_openai_api_connection_error_e_retryable` in `TestRetryLogic`, modeled on `teste_5_remote_protocol_error_e_retryable` (lines 275-287) | `chain.invoke` raises `openai.APIConnectionError` once, succeeds on 2nd call; `_invoke_chain` returns success, `call_count == 2` | LLMPORT-05 |
| New `teste_X_openai_api_connection_error_exhaust_reraise` (companion test) | `chain.invoke` always raises `openai.APIConnectionError`; `_invoke_chain` raises `openai.APIConnectionError` (not `RetryError`) after `LLM_MAX_RETRY_ATTEMPTS` calls | LLMPORT-06 |
| `teste_9_import_sem_lmstudio` (existing, unmodified) | Still passes — patches `agenticlog.agent.ChatOpenAI`, asserts not called on reload | LLMPORT-16 |
| `teste_10_get_llm_singleton` (existing, unmodified) | Still passes — patches `ChatOpenAI`, asserts singleton + `request_timeout=LLM_TIMEOUT_SECONDS` kwarg | LLMPORT-12, LLMPORT-16 |

Constructing `openai.APIConnectionError` in tests: it requires a `request` kwarg (httpx.Request), e.g.:
```python
from openai import APIConnectionError
import httpx
err = APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions"))
```

### Unit / Acceptance — `tests/acceptance/test_retry_logic.py`

| Test | What it verifies | Requirement |
|------|-------------------|--------------|
| `test_ac06_llm_created_with_timeout_from_config` (existing, unmodified) | Still passes — patches `ChatOpenAI`, asserts `request_timeout=LLM_TIMEOUT_SECONDS` | LLMPORT-12, LLMPORT-16 |
| Optionally: companion `openai.APIConnectionError` retryable test in `TestAC07`-adjacent section, following the `AuthenticationError`/`BadRequestError` non-retryable template style (~line 365-398) for symmetry — implementer's discretion on file placement (test_agentic_rag.py vs this file); avoid duplicating the same assertion in both files | LLMPORT-05, LLMPORT-06, LLMPORT-07 |

### New — "no dead `anthropic` import" check (optional, consistency with AC-09 precedent)

`tests/acceptance/test_streamlit_http_client.py::TestAC09NoDeadImports` (line 386) already asserts `app.py`'s namespace excludes `anthropic`. Not required by this story's ACs, but the implementer MAY add an analogous assertion for `agent.py`'s module namespace (e.g. `self.assertNotIn("anthropic", agent_module.__dict__.keys())`) in `tests/test_agentic_rag.py` for consistency. Optional — does not block Done.

### Edge case tests

- `teste_10_llm_model_whitespace_is_verbatim` (above) covers the whitespace-vs-empty-string edge case (LLMPORT-04).
- Retry-bound constants (`LLM_MAX_RETRY_ATTEMPTS`, `LLM_RETRY_WAIT_INITIAL_SECONDS`, `LLM_RETRY_WAIT_MAX_SECONDS`) are read from `config.py` in new tests, never hardcoded — consistent with `test_ac06_retry_respects_max_attempts_from_config`.

### Existing tests that will break (and must be fixed as part of this feature)

| File | Test | Why it breaks | Fix |
|------|------|----------------|-----|
| `tests/test_agentic_rag.py` | Module-level `import anthropic` (line 15) | `anthropic` package removed from `requirements.txt`; if uninstalled, `ImportError` at collection time | Remove the import line |
| `tests/test_agentic_rag.py` | None of the existing `teste_N_*` reference `anthropic.APIConnectionError` directly (confirmed: only the import at line 15 references `anthropic`) | N/A — no test logic depends on the import beyond the line itself | Remove import; no other change needed |

No other existing test imports or references `anthropic` (confirmed by research — only `agent.py` line 16 and `test_agentic_rag.py` line 15).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Modify | Add `DEFAULT_LLM_MODEL` constant and change `LLM_MODEL` (line 28) to `os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL` — empty-string-as-unset env-var logic, following the existing `LLM_API_KEY`/`LLM_API_BASE` env-var-with-fallback pattern (lines 29-30) |
| `src/agenticlog/agent.py` | Modify | Remove `import anthropic` (line 16); replace `anthropic.APIConnectionError` with `openai.APIConnectionError` in `_llm_retry`'s exception tuple (line 71) — add `from openai import APIConnectionError as ...` or `import openai` as needed; define new minimal `typing.Protocol` (e.g. `_LLMClient` or similar name, implementer's discretion, placed near top of module or in a small new module per codebase conventions) with `__or__`, `__ror__`, `invoke`; change `_get_llm()`'s return type annotation from `ChatOpenAI` to the new Protocol type (implementation body unchanged) |
| `tests/test_agentic_rag.py` | Modify | Remove `import anthropic` (line 15); add new `teste_X_openai_api_connection_error_e_retryable` and `teste_X_openai_api_connection_error_exhaust_reraise` to `TestRetryLogic`, modeled on `teste_5_remote_protocol_error_e_retryable` (lines 275-287) |
| `tests/test_config_env.py` | Modify | Add `teste_7`..`teste_10` for `LLM_MODEL` env-var behaviour (unset/default, override, empty-string/fallback, whitespace/verbatim), following `_reload()` pattern (lines 25-48) |
| `tests/acceptance/test_retry_logic.py` | No change required | `test_ac06_llm_created_with_timeout_from_config` and other `ChatOpenAI`-patching tests already pass unmodified (LLMPORT-16); implementer MAY optionally add a symmetric `openai.APIConnectionError` retryable test here instead of `test_agentic_rag.py` |
| `requirements.txt` | Modify | Remove line 48: `anthropic==0.104.1` |
| `.env.example` | Modify | Add `LLM_MODEL` documentation, following `OPENAI_API_KEY`/`OPENAI_API_BASE` style (e.g. `# LLM_MODEL: model name passed to the OpenAI-compatible client (default: hermes-3-llama-3.2-3b)`) |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `openai.APIConnectionError` constructor requires a `request` kwarg (`httpx.Request`), unlike simpler exceptions — test authors must construct it correctly or tests will raise `TypeError` instead of testing retry logic | LOW | Document the correct construction pattern in this spec (see Tests Required); `tests/acceptance/test_retry_logic.py` already has working examples of constructing `openai.AuthenticationError`/`BadRequestError` with `response`/`body` kwargs as a reference pattern |
| Removing `anthropic==0.104.1` from `requirements.txt` while it remains installed in `.venv` — tests pass locally but a clean `pip install -r requirements.txt` environment correctly excludes it (no silent re-introduction) | LOW | CLAUDE.md doesn't require venv rebuild for this change; confirmed via research that no other `src/` or `tests/` module imports `anthropic`, and it's not in `pyproject.toml` dependencies — removal is safe |
| `typing.Protocol` type added without `@runtime_checkable` but later code adds an `isinstance()` check against it — would raise `TypeError: Instance and class checks can only be used with @runtime_checkable protocol` | LOW | Out of scope for this feature (no isinstance check is introduced); flag in code comment if `@runtime_checkable` is omitted, so future implementers add it if needed |
| `_llm: SomeProtocolType | None = None` module-level global typing — lazy-singleton pattern (line 79) interacts with the new Protocol annotation | LOW | Annotation-only change; `_llm = None` at module level remains valid regardless of the Protocol's structure — Python does not enforce type annotations at runtime. No special handling needed beyond updating the annotation if one exists (currently `_llm = None` has no explicit annotation) |
| `LLM_MODEL = os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL` pattern silently accepts ANY non-empty string (including malformed model names) — no validation | LOW | Explicitly out of scope per story ("Validation/restriction of acceptable LLM_MODEL values"); if `LLM_MODEL` points to a nonexistent model, `ChatOpenAI`/LMStudio will surface an error at call time (existing behavior for any misconfiguration) |
| `.specs/features/retry-llm-backoff/spec.md` (lines 56-72) documents the now-incorrect `anthropic.APIConnectionError` rationale as the "original design" — future readers may be confused by the discrepancy | LOW | This spec supersedes that design decision for the `_llm_retry` exception tuple; no edit to the old spec is required (historical record), but implementer should ensure code comments in `agent.py` (if any reference `anthropic`) are updated to reference `openai.APIConnectionError` instead |
| CLAUDE.md conflict check | NONE FOUND | This feature does not touch `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `JQ_SCHEMA_CAMPOS_JSON`, or PDF-extraction logic — `data/vectordb/` rebuild is NOT required |
| Multi-tenancy / timezone / cascade / auth risks | NONE FOUND | This feature is config + retry-exception-type + typing only; no data model, no multi-user state, no timezone-sensitive logic |

---

## Open Questions

None. The story specifies the exact env-var fallback logic (`os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL`), the exact exception-tuple change (`anthropic.APIConnectionError` → `openai.APIConnectionError`), the exact files to modify, and the exact test template to follow. The implementer has discretion only on: (1) the new Protocol type's name and exact module location, and (2) whether the new `openai.APIConnectionError` retry tests live in `tests/test_agentic_rag.py` or `tests/acceptance/test_retry_logic.py` — neither choice affects requirement satisfaction.

---

## Success Criteria

- [ ] `config.py` defines `DEFAULT_LLM_MODEL = "hermes-3-llama-3.2-3b"` and `LLM_MODEL = os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL` (or logically equivalent)
- [ ] `LLM_MODEL` env var unset, set to non-empty value, set to `""`, and set to `" "` all behave per LLMPORT-01..04
- [ ] `_llm_retry`'s `retry_if_exception_type` tuple contains `openai.APIConnectionError`, `httpx.ConnectError`, `httpx.TimeoutException`, `httpx.RemoteProtocolError`, and does NOT contain `anthropic.APIConnectionError`
- [ ] `src/agenticlog/agent.py` contains no `import anthropic` and no reference to `anthropic`
- [ ] `tests/test_agentic_rag.py` contains no `import anthropic` and no reference to `anthropic`
- [ ] `requirements.txt` does not contain `anthropic==0.104.1`
- [ ] `_get_llm()` return type annotation is the new minimal Protocol (not `ChatOpenAI`); `_get_llm()` still returns a `ChatOpenAI` instance constructed with the same 6 constructor kwargs
- [ ] New Protocol type defines exactly `__or__`, `__ror__`, `invoke`
- [ ] `teste_9_import_sem_lmstudio`, `teste_10_get_llm_singleton`, and `test_ac06_llm_created_with_timeout_from_config` pass without modification to their patch targets
- [ ] New `openai.APIConnectionError` retry tests (retry-then-succeed, and exhaust-then-reraise) pass
- [ ] `.env.example` documents `LLM_MODEL`
- [ ] Full test suite passes: `pytest --cov=agenticlog --cov-report=term-missing -v`
- [ ] `ruff`, `mypy`, `bandit` pass per existing `pyproject.toml` configuration
