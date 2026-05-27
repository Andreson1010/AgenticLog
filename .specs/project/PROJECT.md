# AgenticLog — Project Vision

## What

Agentic RAG system for logistics and supply chain domain. Users ask natural language questions; the system retrieves relevant documents, generates multiple candidate answers, ranks them by semantic similarity, and returns the best response with a confidence score.

## Goals

1. Accurate domain answers grounded in logistics documents (RAG path)
2. Conceptual explanations without retrieval overhead (generate path)
3. Real-time web lookup for news/recent events (web path)
4. Transparent confidence scoring so users know when to trust the answer

## Tech Constraints

- LLM runs locally via LMStudio (hermes-3-llama-3.2-3b) — no external API costs
- Embeddings run locally (BAAI/bge-base-en) — no external embedding service
- ChromaDB persists locally — no cloud vector DB dependency
- UI is Streamlit — no separate frontend build step

## Non-Goals (current scope)

- Multi-user authentication
- REST API (FastAPI installed but not used)
- Document ingestion via UI (manual `python -m agenticlog.rag` only)
- Fine-tuning the LLM

## Success Criteria

- Query returns a ranked response with confidence score > 0.5 for domain questions
- Test suite passes at 80%+ coverage
- App starts without LMStudio if vectordb exists (lazy LLM init)
