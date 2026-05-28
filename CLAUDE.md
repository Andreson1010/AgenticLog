# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AgenticLog** — Agentic RAG system for logistics and supply chain, built with LangGraph. The LLM backend is **LMStudio running locally** (Hermes 3B at `http://127.0.0.1:1234/v1`).

## Commands

### Setup
```bash
uv venv --python 3.12
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell
uv pip install -r requirements-dev.txt
uv pip install -e .
```

### Build VectorDB (first time or after changing documents)
```bash
python -m agenticlog.rag
```

### Run Application
```bash
streamlit run app.py
```

### Tests
```bash
# All tests with coverage
pytest --cov=agenticlog --cov-report=term-missing -v

# Single test file
pytest tests/test_rag.py -v

# Single test function
pytest tests/test_rag.py::TestClassName::test_method -v
```

## Architecture

The system follows an **Agentic RAG** pattern using LangGraph state machines:

```
User Query → Decision Node → [retrieve → generate_multiple → evaluate_similarity → rank]
                           → [gerar (generate directly) → same evaluation pipeline]
                           → [usar_web (DuckDuckGo search)] → END
```

### Key Modules

- **`src/agenticlog/config.py`** — Single source of truth for all constants: paths, model names, LLM settings (temperature, max_tokens), RAG chunk sizes, and security limits. Change behavior here, not scattered throughout the code.

- **`src/agenticlog/rag.py`** — Builds the ChromaDB vector database from JSON files in `data/documents/`. Has security validation (path traversal, forbidden JSON keys, file size limits) raising `RAGSecurityError`. Run once to populate `data/vectordb/`.

- **`src/agenticlog/agent.py`** — LangGraph workflow. `AgentState` (Pydantic model) carries all state between nodes. The decision node routes to `retrieve`, `gerar`, or `usar_web` based on query type. Multiple responses are generated and scored by cosine similarity against the query before returning the best one.

- **`app.py`** — Streamlit UI. Invokes `agent_workflow.invoke(AgentState(query=query))` and displays `ranked_response`, `confidence_score`, and retrieved documents.

### Data Flow
```
data/documents/*.json → rag.py → data/vectordb/ (ChromaDB)
                                         ↓
                              agent.py retriever → LMStudio LLM → ranked answer
```

## Testing Conventions (from `.cursor/rules/`)

- **Always mock LLM calls** — never make real API calls in tests
- **Always test empty retrieval edge cases** (e.g., no documents returned)
- **Validate AgentState** contains required keys after each node
- Test function names use `teste_N_` prefix (e.g., `teste_1_passo_decisao_retrieve`)

## Git Workflow (from `.cursor/rules/`)

- Branch naming: `feature/name` or `fix/name`
- Commit messages follow **Conventional Commits** in **Portuguese**: `feat:`, `fix:`, `docs:`, `refactor:`
- Run `git pull origin main` after merging PRs

## Environment

Requires `.env` file with:
```
OPENAI_API_KEY=hermes
OPENAI_API_BASE=http://127.0.0.1:1234/v1
```

LMStudio must be running with the Hermes 3B model loaded before invoking the agent or running integration tests.
