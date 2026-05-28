---
name: build-with-tests
description: How this team builds — match existing patterns, write tests alongside every function, run the full test suite and coverage check before declaring done. Use this skill whenever writing new code, fixing bugs, or extending existing modules in this project. Triggers on any implementation task.
---

# Build With Tests

How we build in this project. Not a methodology — just the rules that stop us from breaking things.

## The pattern before the code

Before writing anything, find the existing pattern for what you're about to do:

- New node in the LangGraph workflow → read `src/agenticlog/agent.py`, match how existing nodes are structured
- New config value → add it to `src/agenticlog/config.py`, never hardcode it inline
- New data access → follow the pattern in `src/agenticlog/rag.py`
- New constant → `config.py` is the single source of truth

If you can't find an existing pattern, ask. Don't invent a new one silently.

## Code rules (non-negotiable)

- **Never mutate.** Return new objects and copies. Never modify in place.
- **Functions under 50 lines.** If it's longer, it's doing too much.
- **Nesting under 4 levels.** Flatten with early returns.
- **No hardcoded values.** Everything goes in `config.py`.
- **No silent failures.** Every exception is caught and logged with context, or re-raised explicitly.
- **Validate at boundaries.** External input (JSON files, user queries, API responses) gets validated on entry. Internal function calls don't need defensive checks.

## AgentState rules

Every LangGraph node receives and returns `AgentState`. Rules:
- Never mutate the state object — return a new dict or updated Pydantic model
- Every node must produce a state that contains all required keys
- Tests must validate that required keys exist in state after each node runs

## Writing tests alongside code

Write the test before or immediately after the function — not at the end of the task.

### Test naming
```
teste_N_descricao_do_que_testa
```
Example: `teste_1_passo_decisao_retrieve`, `teste_2_estado_vazio_retorna_erro`

### What every test must cover
1. Happy path — normal input, expected output
2. Empty/zero edge case — empty list, empty string, zero results
3. Failure path — what happens when something goes wrong

### LLM calls — always mock
```python
# Never make real calls in tests. Always patch the LLM client.
with patch("agenticlog.agent.llm") as mock_llm:
    mock_llm.invoke.return_value = MagicMock(content="mocked response")
    ...
```

### Empty retrieval — always test
Every function that retrieves documents must have a test for when retrieval returns zero results. This is not optional.

### AgentState validation
After calling any node function, assert that the required keys exist:
```python
result = node_function(state)
assert "ranked_response" in result
assert "confidence_score" in result
```

## Gate checks — run before declaring done

In order. Do not skip. Do not declare done if any fail.

```bash
# Full test suite with coverage
pytest --cov=agenticlog --cov-report=term-missing -v
```

Coverage must stay at or above **80%**. If it drops, write more tests — not fewer assertions.

```bash
# Single file (faster feedback during development)
pytest tests/test_<module>.py -v

# Single test
pytest tests/test_<module>.py::TestClass::teste_N_name -v
```

## What "done" means

- All gate checks pass
- Coverage ≥ 80%
- No hardcoded values introduced
- No new patterns invented without documenting why
- Every new function has at least: happy path test, empty edge case test, failure test
