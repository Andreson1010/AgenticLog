# Load ENV Credentials — Technical Spec

**Path:** `.specs/features/load-env-credentials/spec.md`
**TLC scope:** medium
**Based on story:** As a developer, I want LLM_API_KEY and LLM_API_BASE in config.py to be loaded from environment variables via load_dotenv(), so that LLM credentials are never hardcoded in source code.
**Status:** Awaiting human approval

---

## Problem Statement

`config.py` currently hardcodes `LLM_API_KEY = "hermes"` and `LLM_API_BASE = "http://127.0.0.1:1234/v1"` as string literals (lines 22-23). `python-dotenv` is already installed but `load_dotenv()` is never called, so the `.env` file is silently ignored for these credentials. Any environment change requires modifying source code.

---

## Goals

- [ ] AC-01: `config.py` import reads `LLM_API_KEY` and `LLM_API_BASE` from `.env` values
- [ ] AC-02: Custom `.env` values are reflected in the constants
- [ ] AC-03: Shell environment variables take precedence over `.env` (dotenv `override=False`)
- [ ] AC-04: Missing `OPENAI_API_KEY` raises `KeyError` at import with a clear message
- [ ] AC-05: Missing `OPENAI_API_BASE` raises `KeyError` at import with a clear message
- [ ] AC-06: `load_dotenv()` is called in `config.py` before the credential constants are assigned
- [ ] AC-07: Real `.env` file has `OPENAI_API_KEY` and `OPENAI_API_BASE` entries added
- [ ] AC-08: `.env.example` is unchanged (already correct)

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| `LLM_MODEL`, `LLM_TEMPERATURE`, other LLM params as env vars | Not requested; out of story scope |
| Format/reachability validation of `OPENAI_API_BASE` | Separate concern; not in ACs |
| Changes to `agent.py`, `app.py`, or any file other than `config.py` and `.env` | Story explicitly limits scope |
| CI/CD secret injection | Separate infrastructure concern |

---

## User Stories

### P1: Load LLM credentials from environment at startup ⭐ MVP

**User Story**: As a developer, I want `LLM_API_KEY` and `LLM_API_BASE` to be loaded from environment variables via `load_dotenv()`, so that LLM credentials are never hardcoded in source code and can be changed without modifying the codebase.

**Why P1**: Directly addresses the MEDIUM security concern in CONCERNS.md ("Hardcoded LLM Credentials in Source Code") and is the entire scope of this feature.

**Acceptance Criteria**:

1. WHEN `.env` has `OPENAI_API_KEY=hermes` and `OPENAI_API_BASE=http://127.0.0.1:1234/v1` AND `config.py` is imported THEN `LLM_API_KEY == "hermes"` AND `LLM_API_BASE == "http://127.0.0.1:1234/v1"`
2. WHEN `.env` has custom values THEN `LLM_API_KEY` and `LLM_API_BASE` constants reflect those custom values
3. WHEN vars are set in shell AND `.env` also has them THEN shell values take precedence (dotenv called with `override=False`)
4. WHEN `OPENAI_API_KEY` is absent from both `.env` and shell THEN import raises `KeyError` with message identifying the missing variable
5. WHEN `OPENAI_API_BASE` is absent from both `.env` and shell THEN import raises `KeyError` with message identifying the missing variable
6. WHEN `config.py` module loads THEN `load_dotenv()` is called before any `os.environ` read for credentials

**Independent Test**: `python -c "import agenticlog.config; print(agenticlog.config.LLM_API_KEY)"` prints `hermes` when `.env` is present and correct.

---

## Edge Cases

- WHEN `OPENAI_API_KEY` is set to an empty string `""` in `.env` THEN `os.environ["OPENAI_API_KEY"]` returns `""` — this is valid per dotenv semantics; no special handling required (empty string is not absence)
- WHEN `.env` file does not exist THEN `load_dotenv()` silently continues; shell vars still read; missing vars still raise `KeyError`
- WHEN both shell and `.env` define the var THEN shell value wins (`override=False` is the default for `load_dotenv`)

---

## Requirement Traceability

| Requirement ID | Story AC | Phase | Status |
|----------------|----------|-------|--------|
| LOADENV-01 | AC-01 | Implementation | Pending |
| LOADENV-02 | AC-02 | Implementation | Pending |
| LOADENV-03 | AC-03 | Implementation | Pending |
| LOADENV-04 | AC-04 | Implementation | Pending |
| LOADENV-05 | AC-05 | Implementation | Pending |
| LOADENV-06 | AC-06 | Implementation | Pending |
| LOADENV-07 | AC-07 | Implementation | Pending |
| LOADENV-08 | AC-08 | No change needed | Pending |

---

## Data Model Changes

No database or schema changes. Two source changes:

### `src/agenticlog/config.py`

**Add** `load_dotenv` import from `dotenv` (already installed: `python-dotenv==1.0.1`).

**Add** `load_dotenv()` call after the existing `import` block, before any `os.environ` reads. The call must use the default `override=False` so shell variables take precedence.

**Replace** hardcoded string literals on lines 22-23:

```python
# Before
LLM_API_BASE = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "hermes"

# After
LLM_API_KEY: str = os.environ["OPENAI_API_KEY"]
LLM_API_BASE: str = os.environ["OPENAI_API_BASE"]
```

`os.environ["KEY"]` (bracket access, not `.get()`) raises `KeyError` on absence — satisfying AC-04 and AC-05. The `KeyError` message already contains the key name, which is sufficiently clear.

The existing `os.environ.get(...)` pattern for `LOG_LEVEL`/`LOG_FORMAT` is not changed.

### `.env`

**Add** two lines (credentials already referenced in `.env.example`):

```
OPENAI_API_KEY=hermes
OPENAI_API_BASE=http://127.0.0.1:1234/v1
```

The existing `.env` already contains `OLLAMA_API_KEY` and `GITHUB_TOKEN_CURSOR`; these lines are appended without removing anything.

---

## Process / Background Flow

**Happy path — import with `.env` present:**

1. Python executes `config.py` module body.
2. `from dotenv import load_dotenv` resolves (package present).
3. `load_dotenv()` (override=False) reads `.env`, writes vars into `os.environ` for keys not already set.
4. `LLM_API_KEY = os.environ["OPENAI_API_KEY"]` reads the value → assigned.
5. `LLM_API_BASE = os.environ["OPENAI_API_BASE"]` reads the value → assigned.
6. Rest of module (logging, RAG constants) continues unchanged.

**Failure path — missing variable:**

1. Steps 1-3 complete (`.env` may or may not exist).
2. `os.environ["OPENAI_API_KEY"]` raises `KeyError: 'OPENAI_API_KEY'`.
3. Module import fails; Python prints traceback with key name.
4. Application startup aborts (no silent credential gap).

**Failure path — shell override:**

1. Shell has `OPENAI_API_KEY=my-custom-key` exported.
2. `load_dotenv(override=False)` skips that key (already in env).
3. `os.environ["OPENAI_API_KEY"]` returns `"my-custom-key"` from shell.

---

## API Changes

No API changes.

---

## Frontend Changes

No frontend changes.

---

## Tests Required

New test file: `tests/test_config_env.py` (unittest `TestCase`, following existing project patterns).

| Test | Method name | What it verifies |
|------|-------------|-----------------|
| Unit | `teste_1_api_key_loaded_from_env` | `LLM_API_KEY` equals value set in patched `os.environ` | LOADENV-01, LOADENV-02 |
| Unit | `teste_2_api_base_loaded_from_env` | `LLM_API_BASE` equals value set in patched `os.environ` | LOADENV-01, LOADENV-02 |
| Unit | `teste_3_shell_env_takes_precedence` | After `load_dotenv(override=False)`, pre-set shell var is not overwritten | LOADENV-03 |
| Unit | `test_missing_api_key_raises_key_error` | Importing with `OPENAI_API_KEY` absent raises `KeyError` | LOADENV-04 |
| Unit | `test_missing_api_base_raises_key_error` | Importing with `OPENAI_API_BASE` absent raises `KeyError` | LOADENV-05 |
| Unit | `teste_4_load_dotenv_called_before_constants` | `load_dotenv` mock is called before `os.environ` reads (order assertion) | LOADENV-06 |

**Implementation pattern** (env isolation):

```python
import importlib
import sys
from unittest import TestCase
from unittest.mock import patch

class TestConfigEnv(TestCase):
    def _reload_config(self, env: dict):
        """Reload config module with a controlled os.environ."""
        with patch.dict("os.environ", env, clear=True):
            if "agenticlog.config" in sys.modules:
                del sys.modules["agenticlog.config"]
            import agenticlog.config as cfg
            return cfg
```

`importlib.reload` or `del sys.modules` is required because `config.py` executes at module level — constants are assigned once. Tests must isolate environment per case to avoid cross-contamination.

**Existing tests that may break:**

- `tests/test_health.py` and `tests/acceptance/test_health_check.py` import `LLM_API_BASE` at module level. If these tests run in an environment where `OPENAI_API_BASE` is absent from both shell and `.env`, they will fail with `KeyError` during collection. Mitigation: the real `.env` (AC-07) is present in the developer environment; CI must either provide the vars or mock the import.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Modify | Add `load_dotenv` import and call; replace 2 hardcoded string literals with `os.environ[...]` reads |
| `.env` | Modify | Add `OPENAI_API_KEY` and `OPENAI_API_BASE` entries (AC-07) |
| `tests/test_config_env.py` | Create | New unit tests for env-var loading behavior |

`.env.example` — no change (AC-08).
All other files — no change.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `KeyError` breaks existing tests that import `config` without env vars set | MEDIUM | Real `.env` satisfies developer runs; CI must export vars or use a test `.env` fixture |
| Test isolation: `config.py` module-level code runs once per process; stale import in test suite | MEDIUM | Use `del sys.modules["agenticlog.config"]` before each test that reloads config |
| `.env` gitignored — new entries won't reach other developers automatically | LOW | `.env.example` already has correct entries; developers follow documented setup |
| `os.environ["KEY"]` raises `KeyError` not `ValueError` — error message shows only key name, not instructions | LOW | Acceptable per ACs; can be improved in a follow-up with a custom startup check |
| Empty-string value in `.env` passes the read but may break LMStudio auth | LOW | Out of scope per story; no format validation required |
| Removing hardcoded defaults changes behavior for any consumer that relied on them silently | LOW | No external consumers; `agent.py` reads from config constants — unchanged interface |

---

## Open Questions

None.

---

## Success Criteria

- [ ] `config.py` contains no hardcoded string values for `LLM_API_KEY` or `LLM_API_BASE`
- [ ] `load_dotenv()` call appears in `config.py` before `os.environ["OPENAI_API_KEY"]` read
- [ ] `.env` has `OPENAI_API_KEY` and `OPENAI_API_BASE` entries
- [ ] All 6 new tests in `tests/test_config_env.py` pass
- [ ] Full test suite passes: `pytest --cov=agenticlog --cov-report=term-missing -v`
- [ ] `python -c "import agenticlog.config; print(agenticlog.config.LLM_API_KEY)"` prints `hermes` with `.env` present
- [ ] CONCERNS.md item "MEDIUM — Hardcoded LLM Credentials in Source Code" is resolved
