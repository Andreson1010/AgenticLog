# Project Structure

**Root:** C:\Users\ander\8_projetos\agenticlog

## Directory Tree

```
agenticlog/
├── src/
│   └── agenticlog/
│       ├── __init__.py       # exports AgentState, agent_workflow
│       ├── agent.py          # LangGraph 6-node FSM
│       ├── config.py         # all constants
│       └── rag.py            # document ingestion + vectordb build
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # sys.path setup for pytest
│   ├── test_rag.py           # 8 classes, 15+ methods (rag.py coverage)
│   └── test_agentic_rag.py   # 1 class, 11 methods (agent.py coverage)
├── data/
│   ├── documents/            # source JSON files (3 samples)
│   │   ├── doc1.json
│   │   ├── doc2.json
│   │   └── doc3.json
│   └── vectordb/             # ChromaDB persistent store
│       └── chroma.sqlite3
├── .specs/                   # tlc-spec-driven planning docs
│   ├── project/
│   ├── codebase/
│   ├── features/
│   └── quick/
├── .streamlit/
│   └── config.toml           # dark theme + color config
├── .github/                  # CI/CD workflows (if present)
├── app.py                    # Streamlit entry point
├── pyproject.toml            # project metadata + pytest + coverage config
├── requirements.txt          # 166 pinned runtime deps
├── requirements-dev.txt      # pytest + pytest-cov
├── .env                      # API keys (gitignored)
├── .gitignore
└── CLAUDE.md                 # dev guide for Claude Code
```

## Module Organization

### src/agenticlog — Core Package

**Purpose:** Agentic RAG logic, LangGraph workflow, configuration  
**Key files:**
- `config.py` — change behavior here first (paths, model, limits)
- `agent.py` — the runnable workflow (`agent_workflow`)
- `rag.py` — run once to populate vectordb
- `__init__.py` — public API surface

### tests/ — Test Suite

**Purpose:** Unit and integration tests  
**Key files:**
- `test_rag.py` — security + ingestion tests (all mocked)
- `test_agentic_rag.py` — agent node tests (all mocked)
- `conftest.py` — sys.path fix so `agenticlog` resolves without install

### data/ — Runtime Data

**Purpose:** Source documents and persisted vector index  
**Notes:**
- `documents/` — add JSON files here, then rebuild vectordb
- `vectordb/` — gitignored; must be built locally before first run

## Where Things Live

**LLM configuration:**
- Settings: `src/agenticlog/config.py` (OPENAI_API_BASE, LLM_MODEL, TEMPERATURE, MAX_TOKENS)
- Client: `src/agenticlog/agent.py` lines 44-50

**RAG pipeline:**
- Entry: `python -m agenticlog.rag`
- Logic: `src/agenticlog/rag.py`
- Input: `data/documents/*.json`
- Output: `data/vectordb/`

**Agent workflow:**
- Definition: `src/agenticlog/agent.py`
- Invocation: `app.py` line 68 — `agent_workflow.invoke(AgentState(query=query))`

**UI:**
- Entry: `streamlit run app.py`
- Session state: `app.py` lines 18-25
- Results display: `app.py` (ranked_response, confidence bar, retrieved docs expander)
