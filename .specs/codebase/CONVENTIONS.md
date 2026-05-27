# Code Conventions

## Naming Conventions

**Files:** snake_case  
Examples: `agent.py`, `rag.py`, `config.py`, `test_rag.py`, `test_agentic_rag.py`

**Functions/Methods:** snake_case, Portuguese verb phrases  
Examples: `passo_decisao_agente`, `gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas`, `retrieve_info`, `usar_ferramenta_web`

**Classes:** PascalCase  
Examples: `AgentState`, `RAGSecurityError`, `TestRAGSecurityError`, `TestAgenticRAG`

**Variables:** snake_case, Portuguese  
Examples: `state.query`, `estado`, `diretorio`, `chave_proibida`, `arquivo`, `mock_retriever`

**Constants:** UPPER_SNAKE_CASE  
Examples: `LLM_MODEL`, `OPENAI_API_BASE`, `MAX_JSON_FILES`, `DIR_DOCUMENTS`, `PROJECT_ROOT`

**Test methods:** `teste_N_` prefix (domain logic) or `test_` prefix (security/error tests)  
Examples: `teste_1_passo_decisao_retrieve`, `teste_5_retrieve_info`, `test_raise_rag_security_error`

## Code Organization

**Import order:**
1. Standard library (`os`, `json`, `sys`, `pathlib`, `unittest`, `tempfile`)
2. Third-party (`numpy`, `langgraph`, `pydantic`, `langchain*`, `streamlit`)
3. Local (`from agenticlog.config import ...`)

**File structure (modules):**
1. Module docstring
2. Imports
3. Constants / config reads
4. Helper/private functions
5. Main class or public API

## Type Safety

**Approach:** Selective — Pydantic for state model, minimal elsewhere  
- `AgentState` uses Pydantic BaseModel for runtime validation
- `# type: ignore` on numpy and langgraph imports (incompatible stubs)
- Function signatures often lack return type annotations
- `config.py` constants have no type annotations

## Error Handling

**Pattern:** Custom exception + try/except at boundaries  
- `RAGSecurityError(Exception)` for all security violations in `rag.py`
- `try/except Exception` in `usar_ferramenta_web` (agent.py line 157) with fallback string
- `try/except Exception` in `app.py` (line 66) wrapping full workflow invocation
- No error propagation pattern — errors either raise `RAGSecurityError` or return fallback

## Comments / Documentation

**Docstring style:** Triple-quoted, Portuguese, custom format:
```python
"""
Short description.

Entrada: description of input
Saída: description of output
"""
```

**Inline comments:** Portuguese for domain logic, English for generic patterns  
**Print statements:** Used for operational feedback in `rag.py` (not logging module)

## Language Mix

- Function/variable names: Portuguese
- Exception messages: Portuguese
- Docstrings: Portuguese (module + function level)
- Prompts sent to LLM: Portuguese
- Git commit messages: Portuguese (Conventional Commits)
- Constants in config: English abbreviations (LLM, RAG, API)
