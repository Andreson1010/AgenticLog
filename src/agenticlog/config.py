# AgenticLog - Configurações centralizadas
"""Constantes, paths e parâmetros de modelos."""

from pathlib import Path

# Raiz do projeto (pasta que contém src/, data/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Diretórios de dados
DIR_DOCUMENTS = PROJECT_ROOT / "data" / "documents"
DIR_VECTORDB = PROJECT_ROOT / "data" / "vectordb"

# Modelo de embeddings (deve ser o mesmo em rag e agent)
EMBEDDING_MODEL = "BAAI/bge-base-en"

# LLM (LMStudio)
LLM_MODEL = "hermes-3-llama-3.2-3b"
LLM_API_BASE = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "hermes"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 2048

# RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
