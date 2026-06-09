# AgenticLog - Configurações centralizadas
"""Constantes, paths e parâmetros de modelos."""

import datetime
import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Raiz do projeto (pasta que contém src/, data/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # âncora para todos os paths relativos

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Diretórios de dados
DIR_DOCUMENTS = PROJECT_ROOT / "data" / "documents"  # JSONs de origem dos documentos
DIR_VECTORDB = PROJECT_ROOT / "data" / "vectordb"    # banco ChromaDB persistido em disco

# Modelo de embeddings (deve ser o mesmo em rag e agent)
EMBEDDING_MODEL = "BAAI/bge-base-en"  # modelo HuggingFace usado para gerar e consultar embeddings

# LLM (LMStudio)
LLM_MODEL = "hermes-3-llama-3.2-3b"
LLM_API_KEY: str = os.environ.get("OPENAI_API_KEY", "hermes")
LLM_API_BASE: str = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1")
LLM_TEMPERATURE = 0                           # temperatura 0 para respostas determinísticas
LLM_MAX_TOKENS = 2048                         # limite de tokens gerados por resposta
LLM_TIMEOUT_SECONDS: float = 60.0            # timeout por chamada ao LLM em segundos
LLM_MAX_RETRY_ATTEMPTS: int = 3              # número máximo de tentativas com retry
LLM_RETRY_WAIT_INITIAL_SECONDS: float = 1.0  # espera inicial do backoff exponencial em segundos
LLM_RETRY_WAIT_MAX_SECONDS: float = 4.0      # espera máxima do backoff exponencial em segundos
LLM_HEALTH_CHECK_TIMEOUT_SECONDS: float = 5.0  # timeout do GET /v1/models antes do workflow

# RAG
CHUNK_SIZE = 500    # tamanho máximo de cada chunk de texto em caracteres
CHUNK_OVERLAP = 50  # sobreposição entre chunks para preservar contexto nas bordas

# Roteamento — palavras-chave que determinam o caminho no grafo LangGraph
ROUTING_KEYWORDS_GERAR: tuple[str, ...] = (
    "explain", "summarize", "define", "concept", "general", "what is",
    "explique", "resuma", "defina", "conceito", "geral", "o que é",
)
ROUTING_KEYWORDS_WEB: tuple[str, ...] = (
    "search the web", "news", "updated", "recent", "latest information",
    "busque na web", "notícias", "atualizado", "recente", "últimas informações",
)

# ChromaDB — nomes de coleção
DEFAULT_COLLECTION_NAME: str = "logistica"
COLLECTION_NAME_MIN_LEN: int = 3
COLLECTION_NAME_MAX_LEN: int = 63
COLLECTION_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$"
)

# Segurança - limites para carregamento de documentos
MAX_JSON_FILES = 1000          # impede carregamento irrestrito de arquivos maliciosos
MAX_JSON_FILE_SIZE_MB = 10     # bloqueia arquivos excessivamente grandes (proteção contra DoS)
MAX_DOCUMENT_FILE_SIZE_MB = MAX_JSON_FILE_SIZE_MB  # limite compartilhado para qualquer formato (JSON, PDF)
FORBIDDEN_JSON_KEYS = ("lc",)  # mitiga injeção via chave "lc" usada pela classe Serializable do LangChain
INVALID_FILENAME_CHARS: frozenset[str] = frozenset('<>:"/\\|?*\x00')
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# API Server
API_HOST: str = os.environ.get("API_HOST", "127.0.0.1")
API_PORT: int = int(os.environ.get("API_PORT", "8000"))
API_CLIENT_TIMEOUT_SECONDS: int = 120  # worst-case: 3 LLM retries × ~24s each = ~73s; client must not time out before server

# History Audit Log
DIR_HISTORY: Path = PROJECT_ROOT / "data" / "history"   # diretório onde o SQLite é persistido
HISTORY_FILE: Path = DIR_HISTORY / "history.db"          # arquivo SQLite do audit log
HISTORY_MAX_ENTRIES: int = int(os.environ.get("HISTORY_MAX_ENTRIES", "1000"))  # máximo de registros; evicta o mais antigo se atingido
if HISTORY_MAX_ENTRIES <= 0:
    raise ValueError(
        f"HISTORY_MAX_ENTRIES={HISTORY_MAX_ENTRIES!r} must be > 0."
    )

# Logging
_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_LOG_FORMATS: frozenset[str] = frozenset({"text", "json"})

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
if LOG_LEVEL not in _VALID_LOG_LEVELS:
    raise ValueError(
        f"Invalid LOG_LEVEL={LOG_LEVEL!r}. Must be one of {sorted(_VALID_LOG_LEVELS)}."
    )

LOG_FORMAT: str = os.environ.get("LOG_FORMAT", "text").strip().lower()
if LOG_FORMAT not in _VALID_LOG_FORMATS:
    raise ValueError(
        f"Invalid LOG_FORMAT={LOG_FORMAT!r}. Must be one of {sorted(_VALID_LOG_FORMATS)}."
    )


class _JsonFormatter(logging.Formatter):
    """Serializa cada LogRecord como uma linha JSON com campos padronizados."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })
