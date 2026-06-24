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

### Build VectorDB (first time, after changing documents, or after changing chunking/embedding configuration)
```bash
# Default: incremental ingestion of every file in data/documents/ (skips unchanged, upserts changed)
python -m agenticlog.rag

# Full rebuild from scratch (legacy behavior)
python -m agenticlog.rag --rebuild
```

The CLI is **incremental by default** (REC-04): with no flags it ingests each `*.json` and `*.pdf` in `data/documents/`, skipping files already indexed with the same content-hash and upserting changed ones — no full rebuild. Use `--rebuild` to discard `data/vectordb/` content and recompute every chunk (required after any chunking-strategy or embedding-model change; see Silent-degradation risk below).

**After changing any of the following in `config.py`** — `EMBEDDING_MODEL`, `SEMANTIC_BREAKPOINT_TYPE`, `SEMANTIC_BREAKPOINT_THRESHOLD`, `JQ_SCHEMA_CAMPOS_JSON` (jq_schema), or PDF-extraction logic in `extrair_texto_pdf` (`src/agenticlog/rag.py`) — rebuild the vector DB from scratch:
1. Stop the running app (if any).
2. Delete `data/vectordb/` (gitignored, fully regenerable).
3. Rerun `python -m agenticlog.rag --rebuild`.
4. Resume queries with `streamlit run app.py`.

The current embedding model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (multilingual, 768-dim, optimized for Portuguese among other languages). On first run with this model, expect a larger download (~1.0–1.1 GB, vs ~440 MB for the previous `BAAI/bge-base-en`), so initial setup takes longer.

**Silent-degradation risk:** if `data/vectordb/` is **not** rebuilt after changing `EMBEDDING_MODEL`, `SEMANTIC_BREAKPOINT_TYPE`, `SEMANTIC_BREAKPOINT_THRESHOLD`, the jq_schema, or PDF-extraction logic, the system will **not** raise an error. Existing chunks remain queryable with their original embeddings/dimensions, but they were computed under a different chunking strategy or embedding space than newly ingested content — similarity scores and retrieval results become inconsistent or unreliable, with no warning in logs or the UI. Additionally, incremental ingestion (`adicionar_documento_incrementalmente`) skips files already present in `data/vectordb/` (detected via content-hash dedup), so OLD-strategy chunks for already-ingested files are never replaced without a full rebuild. Always rebuild `data/vectordb/` after any chunking-strategy or embedding-model change.

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

## Testing Conventions

- **Always mock LLM calls** — never make real API calls in tests
- **Always test empty retrieval edge cases** (e.g., no documents returned)
- **Validate AgentState** contains required keys after each node
- Test function names use `teste_N_` prefix (e.g., `teste_1_passo_decisao_retrieve`)

## Git Workflow

- Branch naming: `feature/name` or `fix/name`
- Commit messages follow **Conventional Commits** in **Portuguese**: `feat:`, `fix:`, `docs:`, `refactor:`
- Run `git pull origin main` after merging PRs
- ALWAYS create a feature branch before committing — never commit directly to main. Branch first, then commit, then open a PR.
- After opening a PR, ALWAYS run the code review phase before merging — do not skip it. Use the **`code-review` plugin** (`/code-review` for the working diff, or `/code-review <PR#>` to review a GitHub PR) as the code-review step — not an ad-hoc agent.

## Environment

Requires `.env` file with:
```
OPENAI_API_KEY=hermes
OPENAI_API_BASE=http://127.0.0.1:1234/v1
```

LMStudio must be running with the Hermes 3B model loaded before invoking the agent or running integration tests.

## Editing Config Files
When editing JSON or settings files, validate the result (no trailing commas, valid syntax) before finishing; prefer rewriting the whole file over partial edits for config files.

## Working Style

When asked about a specific tool, command, or status line, inspect the ACTUAL script/config in the repo first before giving generic instructions.
