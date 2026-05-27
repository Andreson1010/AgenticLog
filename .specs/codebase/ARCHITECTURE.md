# Architecture

**Pattern:** Agentic RAG — LangGraph finite state machine with cosine similarity ranking

## High-Level Structure

```
data/documents/*.json
  ↓ rag.py (security validation + chunking + embeddings)
data/vectordb/ (ChromaDB, SQLite)
  ↓ agent.py retriever
LangGraph FSM (6 nodes, 3 routing paths)
  ↓ LMStudio LLM (http://127.0.0.1:1234/v1)
app.py (Streamlit UI)
```

## Identified Patterns

### LangGraph State Machine

**Location:** `src/agenticlog/agent.py`  
**Purpose:** Orchestrate retrieval, generation, evaluation, ranking  
**Implementation:** `StateGraph(AgentState)` with 6 nodes and conditional edges  

State model (`AgentState`, Pydantic BaseModel):
- `query: str` — user input
- `next_step: str` — routing decision ("retrieve" | "gerar" | "usar_web")
- `retrieved_info: list` — docs from ChromaDB
- `possible_responses: list` — 5 candidate LLM responses
- `similarity_scores: list` — cosine similarity per response
- `ranked_response: str` — final selected answer
- `confidence_score: float` — 0.0–1.0

Nodes:
1. `decision` (`passo_decisao_agente`, line 117) — keyword routing
2. `retrieve` (`retrieve_info`, line 166) — ChromaDB similarity search
3. `generate_multiple` (`gera_multiplas_respostas`, line 177) — 5 LLM responses
4. `evaluate_similarity` (`avalia_similaridade`, line 213) — cosine similarity
5. `rank_responses` (`rank_respostas`, line 253) — pick best response
6. `usar_web` (`usar_ferramenta_web`, line 150) — DuckDuckGo search

Routing:
```
decision → retrieve → generate_multiple → evaluate_similarity → rank_responses → END
decision → generate_multiple (same pipeline, no retrieval context)
decision → usar_web → END
```

### RAG Security Pipeline

**Location:** `src/agenticlog/rag.py`  
**Purpose:** Safe document ingestion into ChromaDB  
**Implementation:** 3 validation layers before loading  

1. `_valida_path_documentos()` — path traversal check (resolves to project root)
2. `_valida_json_sem_chaves_proibidas()` — blocks "lc" key (LangChain gadget attack)
3. `_valida_arquivos_json()` — file count (≤1000) and size (≤10MB) limits
4. DirectoryLoader + JSONLoader with jq_schema flattening
5. RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
6. HuggingFaceEmbeddings (BAAI/bge-base-en) → Chroma.from_documents()

### Centralized Config

**Location:** `src/agenticlog/config.py`  
**Purpose:** Single source of truth for all constants  
**Example:** `LLM_MODEL`, `OPENAI_API_BASE`, `TEMPERATURE`, `MAX_TOKENS`, `DIR_DOCUMENTS`, `DIR_VECTORDB`

## Data Flow

### Retrieve path (default)

```
query → decision (keyword: general logistics) → retrieve (ChromaDB top-k)
  → generate_multiple (prompt_rag_retrieve + context, 5 responses)
  → evaluate_similarity (cosine between each response and retrieved docs)
  → rank_responses (highest score wins)
  → app.py displays ranked_response + confidence_score + retrieved docs
```

### Generate path (conceptual queries)

```
query → decision (keywords: explain/summarize/define/concept)
  → generate_multiple (prompt_gerar, no retrieval context)
  → evaluate_similarity → rank_responses → END
```

### Web path

```
query → decision (keywords: search the web/news/recent)
  → usar_web (DuckDuckGo, CHAT_ZERO_SHOT_REACT_DESCRIPTION agent)
  → END (confidence_score = 0.0, no RAG grounding)
```

## Code Organization

**Approach:** Feature-based, flat src layout  

```
src/agenticlog/
  config.py    — constants only, no logic
  rag.py       — data ingestion, one-time setup
  agent.py     — LangGraph workflow, always-on
  __init__.py  — exports AgentState, agent_workflow
app.py         — Streamlit entrypoint, imports from agenticlog
```

**Module boundaries:**  
- `rag.py` runs once to populate vectordb — not imported by agent at runtime  
- `agent.py` imports `vector_db` and `llm` at module level (eager init)  
- `app.py` only imports `agent_workflow` and `AgentState`
