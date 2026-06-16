# AgenticLog — Project Vision

## What

Agentic RAG system for logistics and supply chain domain. Users ask natural language questions; the system retrieves relevant documents, generates multiple candidate answers, ranks them by semantic similarity, and returns the best response with a confidence score.

The system exposes two interfaces: a Streamlit UI for interactive use and a FastAPI REST API for programmatic access.

## Goals

1. Accurate domain answers grounded in logistics documents (RAG path)
2. Conceptual explanations without retrieval overhead (generate path)
3. Real-time web lookup for news/recent events (web path)
4. Transparent confidence scoring so users know when to trust the answer
5. Incremental document ingestion (JSON and PDF) without full vectordb rebuild
6. Observable, auditable query history for traceability

## Tech Stack

- **LLM:** LMStudio local (hermes-3-llama-3.2-3b) — no external API costs
- **Embeddings:** sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (multilingual, 768-dim) — local, optimized for Portuguese
- **Vector DB:** ChromaDB (local persist) — multi-collection support
- **Orchestration:** LangGraph state machine (6-node FSM)
- **UI:** Streamlit — collection selector, JSON + PDF upload, confidence bar, retrieved docs expander
- **API:** FastAPI — POST /query, GET /health, GET /history
- **Audit:** SQLite history log (HistoryStore, thread-safe, eviction at 1000 entries)
- **LLM provider:** Configurable via LLM_MODEL env var; retry with exponential backoff

## Current Capabilities

### Query
- LangGraph decision node routes to retrieve, generate, or web-search path
- 5 candidate responses ranked by cosine similarity; best returned with confidence score
- Health check before first workflow invocation; lazy LLM singleton initialization
- Retry on transient LLM errors (3 attempts, exponential backoff 1–4s)

### Document Ingestion
- JSON upload via Streamlit sidebar (security: path traversal, forbidden keys, size limits, Windows reserved names)
- PDF upload via Streamlit sidebar (PyMuPDF text extraction, password protection detection)
- Incremental ingest (JSON): adds chunks to existing ChromaDB without full rebuild; SHA-256 dedup via `file_hash`
- Incremental ingest (PDF): `adicionar_pdf_incrementalmente()` — same dedup/rollback pattern as JSON; no full rebuild required
- Full rebuild: `python -m agenticlog.rag` — required after embedding/chunking config changes
- Unified chunk metadata on every chunk: `source`, `file_hash`, `chunk_index`, `page`, `doc_type`

### Multi-Collection
- ChromaDB collections named by user (3–63 chars, alphanumeric + `-_`)
- Retriever fan-out across all collections with dedup; per-collection `k` config

### Observability
- Structured logging (text or JSON format via LOG_FORMAT env var)
- Query history audit log: SQLite, GET /history?limit=N, eviction at HISTORY_MAX_ENTRIES
- Confidence score returned on every query response

## Non-Goals (current scope)

- Multi-user authentication
- Fine-tuning the LLM
- Cloud deployment / hosted vector DB
- Upsert (replace existing document chunks on re-upload — REC-03, backlog)
- CLI incremental por padrão (REC-04, backlog)

## Success Criteria

- Query returns a ranked response with confidence score > 0.5 for domain questions
- Test suite passes at 80%+ coverage (currently ~93%)
- App starts without LMStudio if vectordb exists (lazy LLM init)
- All document ingestion paths validated for security at boundaries
- CI green on every merge (lint + test matrix)
