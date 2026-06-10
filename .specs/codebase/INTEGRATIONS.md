# External Integrations

## LLM — LMStudio (Local)

**Service:** LMStudio (local process, OpenAI-compatible API)  
**Purpose:** Generate responses for all 3 routing paths  
**Implementation:** `src/agenticlog/agent.py` lines 44-50  
**Configuration:** `src/agenticlog/config.py` (OPENAI_API_BASE, LLM_MODEL, TEMPERATURE, MAX_TOKENS)  
**Authentication:** Dummy key "hermes" — LMStudio requires the field but does not validate it  
**Model:** hermes-3-llama-3.2-3b (must be pre-loaded in LMStudio GUI before running)  
**Endpoint:** http://127.0.0.1:1234/v1  
**Risk:** Single point of failure — no retry logic, no health check, no fallback model  

## Vector Database — ChromaDB (Local Persistent)

**Service:** ChromaDB (embedded, SQLite backend)  
**Purpose:** Store and retrieve document embeddings for RAG  
**Implementation:** `src/agenticlog/rag.py` (write), `src/agenticlog/agent.py` line 53-58 (read)  
**Configuration:** `config.py` — DIR_VECTORDB = `data/vectordb/`  
**Authentication:** None (local only)  
**Initialization:** Run `python -m agenticlog.rag` once after adding documents  
**Persistence:** `data/vectordb/chroma.sqlite3` (gitignored)  
**Risk:** If `data/vectordb/` missing, agent fails at module import  

## Embeddings — HuggingFace (Local Model)

**Service:** sentence-transformers + FlagEmbedding (local inference)  
**Purpose:** Generate embeddings for documents and queries  
**Model:** sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (multilingual, 768-dim, downloaded on first run, cached locally)  
**Implementation:** `src/agenticlog/rag.py` via `HuggingFaceEmbeddings`  
**Authentication:** None (local model)  
**Risk:** First run requires internet + disk space for model download (~1.0–1.1 GB)  

## Web Search — DuckDuckGo

**Service:** DuckDuckGo Search API (unofficial, via duckduckgo_search library)  
**Purpose:** Answer queries about recent news or web content  
**Implementation:** `src/agenticlog/agent.py` lines 90-98 (`usar_ferramenta_web` node)  
**Configuration:** `DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)`  
**Authentication:** None (no API key required)  
**Trigger:** Query keywords: "search the web", "news", "recent"  
**Risk:** No rate limiting — could hit unofficial API limits; confidence_score always 0.0  

## Environment Variables

**File:** `.env` (gitignored)  
**Loaded by:** Not loaded programmatically in current code — LLM config is hardcoded in `config.py`  

Current `.env` contents (not used by application):
- `OLLAMA_API_KEY` — present but unused
- `GITHUB_TOKEN_CURSOR` — present but unused

**Gap:** CLAUDE.md documents `OPENAI_API_KEY` and `OPENAI_API_BASE` as required `.env` vars, but `config.py` hardcodes these values directly without reading from environment.
