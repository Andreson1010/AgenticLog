# Testing Infrastructure

## Test Frameworks

**Unit/Integration:** pytest >= 8.0.0 + unittest (TestCase base class)  
**Coverage:** pytest-cov >= 4.1.0 (branch coverage enabled)  
**E2E:** None

## Test Organization

**Location:** `tests/`  
**Naming:** `test_*.py` files, `Test*` classes  
**Method naming:** `teste_N_` prefix for domain tests, `test_` prefix for error/security tests  
**Structure:** unittest TestCase classes grouped by module under test

## Testing Patterns

### Unit Tests

**Approach:** All external deps mocked — no real LLM, no real ChromaDB, no real files  
**Location:** `tests/test_rag.py`, `tests/test_agentic_rag.py`  

Mock pattern:
```python
@patch("agenticlog.agent.retriever")
def teste_5_retrieve_info(self, mock_retriever):
    mock_retriever.invoke.return_value = [MagicMock(page_content="doc")]
    result = retrieve_info(self.state)
    self.assertIsInstance(result.retrieved_info, list)
```

Temp file pattern (rag.py tests):
```python
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump({...}, f)
    path = Path(f.name)
try:
    # test assertion
finally:
    path.unlink(missing_ok=True)
```

### Integration Tests

None currently. LMStudio must be running for real workflow tests — not automated.

### E2E Tests

None. Streamlit UI not tested.

## Test Execution

```bash
# All tests with coverage
pytest --cov=agenticlog --cov-report=term-missing -v

# Single file
pytest tests/test_rag.py -v

# Single method
pytest tests/test_rag.py::TestClassName::test_method -v
```

**Configuration (pyproject.toml):**
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*", "teste_*"]
addopts = "-v --tb=short"
```

## Coverage Targets

**Current:** Unknown (not measured in CI)  
**Goals:** 80% minimum (per project rules)  
**Enforcement:** Manual (no CI gate configured)

**Coverage config:**
```ini
[tool.coverage.run]
source = ["src/agenticlog"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
show_missing = true
```

## Test Coverage Matrix

| Code Layer | Required Test Type | Location Pattern | Run Command |
|---|---|---|---|
| config.py | none (constants only) | — | — |
| rag.py security | unit | tests/test_rag.py | pytest tests/test_rag.py -v |
| rag.py ingestion | unit (mocked) | tests/test_rag.py | pytest tests/test_rag.py -v |
| agent.py nodes | unit (mocked) | tests/test_agentic_rag.py | pytest tests/test_agentic_rag.py -v |
| agent_workflow (compiled) | integration | missing | pytest (LMStudio required) |
| app.py | E2E | missing | — |

## Parallelism Assessment

| Test Type | Parallel-Safe? | Isolation Model | Evidence |
|---|---|---|---|
| unit | Yes | All deps mocked, no shared state | @patch decorators, MagicMock |
| integration | No | LMStudio single process | blocking HTTP to localhost:1234 |

## Gate Check Commands

| Gate Level | When to Use | Command |
|---|---|---|
| Quick | After unit-only changes | `pytest tests/ -v --tb=short` |
| Full | After any agent/rag change | `pytest --cov=agenticlog --cov-report=term-missing -v` |
| Build | Before PR merge | `pytest --cov=agenticlog --cov-report=term-missing -v` |
