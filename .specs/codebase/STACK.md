# Tech Stack

**Analyzed:** 2026-05-27

## Core

- Language: Python 3.11+
- Package manager: uv
- Build system: setuptools >= 61.0

## AI / ML

- Orchestration: LangGraph 0.2.74
- LLM client: langchain-openai 0.3.6 (OpenAI-compatible, pointing to LMStudio)
- Vector DB: chromadb 0.4.14 (local persistent, SQLite backend)
- Embeddings: langchain-huggingface 0.1.2 + sentence-transformers 3.4.1 + FlagEmbedding 1.0.3
- RAG chain: langchain 0.3.19, langchain-core 0.3.35, langchain-chroma 0.2.2, langchain-community 0.3.17
- Web search: duckduckgo_search 8.0.0
- Similarity: scikit-learn 1.6.1, numpy 1.26.4, torch 2.6.0

## UI

- Framework: streamlit 1.42.1

## Installed but unused in current app

- fastapi 0.115.8, uvicorn 0.34.0

## Testing

- Unit/Integration: pytest >= 8.0.0 + pytest-cov >= 4.1.0
- Style: unittest (test classes inherit unittest.TestCase)
- Runner: pytest (with unittest compatibility)

## External Services

- LLM: LMStudio (local, hermes-3-llama-3.2-3b @ http://127.0.0.1:1234/v1)
- Search: DuckDuckGo (region: br-pt, max_results: 5)

## Development Tools

- Env management: uv venv
- Lint/format: not configured
- Coverage: pytest-cov (branch coverage enabled)
