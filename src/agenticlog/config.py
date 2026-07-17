# AgenticLog - Configurações centralizadas
"""Constantes, paths e parâmetros de modelos."""

import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Raiz do projeto (pasta que contém src/, data/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # âncora para todos os paths relativos

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Diretórios de dados
DIR_DOCUMENTS = PROJECT_ROOT / "data" / "documents"  # JSONs de origem dos documentos
DIR_VECTORDB = PROJECT_ROOT / "data" / "vectordb"    # banco ChromaDB persistido em disco

# Modelo de embeddings (deve ser o mesmo em rag e agent)
# Multilíngue (otimizado para português, entre outros idiomas), 768 dimensões — drop-in
# compatível com o modelo anterior (BAAI/bge-base-en, também 768-dim).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # modelo HuggingFace usado para gerar e consultar embeddings

# LLM (LMStudio)
DEFAULT_LLM_MODEL = "hermes-3-llama-3.2-3b"
LLM_MODEL: str = os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL
LLM_API_KEY: str = os.environ.get("OPENAI_API_KEY", "hermes")
LLM_API_BASE: str = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1")
LLM_TEMPERATURE = 0                           # temperatura 0 para respostas determinísticas
LLM_MAX_TOKENS = 2048                         # limite de tokens gerados por resposta
LLM_TIMEOUT_SECONDS: float = 60.0            # timeout por chamada ao LLM em segundos
LLM_MAX_RETRY_ATTEMPTS: int = 3              # número máximo de tentativas com retry
LLM_RETRY_WAIT_INITIAL_SECONDS: float = 1.0  # espera inicial do backoff exponencial em segundos
LLM_RETRY_WAIT_MAX_SECONDS: float = 4.0      # espera máxima do backoff exponencial em segundos
LLM_HEALTH_CHECK_TIMEOUT_SECONDS: float = 5.0  # timeout do GET /v1/models antes do workflow
# Mensagem fixa do modo seguro (200-degraded quando o modelo LMStudio está indisponível)
RESPOSTA_PADRAO_SEGURA: str = (
    "Serviço de IA indisponível no momento. Tente novamente mais tarde."
)

# RAG — Semantic Chunking (ADR-013)
SEMANTIC_BREAKPOINT_TYPE: Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
] = "percentile"  # método de detecção de breakpoints semânticos (tipo exigido pelo SemanticChunker)
SEMANTIC_BREAKPOINT_THRESHOLD: float = 95.0   # percentil de dissimilaridade para corte

# jq_schema compartilhado: 1 valor de saída do jq por chave top-level do JSON -> 1 Document
# por chave (chunking estrutura-aware, ADR-008). `to_entries[]` (com `[]`, sem `map`) faz o jq
# emitir um valor de saída por entrada — JSONLoader itera cada saída como um Document
# separado. Usar `map(...)` (sem `[]`) produziria UMA lista única, e JSONLoader levantaria
# ValueError ("Expected page_content is string, got <class 'list'>").
JQ_SCHEMA_CAMPOS_JSON = 'to_entries[] | .key + ": " + (.value | tostring)'

# Roteamento — palavras-chave que determinam o caminho no grafo LangGraph.
# Estratégia retrieve-first: só "usar_web" é decidido por palavra-chave; todo o
# restante tenta retrieve e cai para geração direta apenas se a base vier vazia.
ROUTING_KEYWORDS_WEB: tuple[str, ...] = (
    "search the web", "news", "updated", "recent", "latest information",
    "busque na web", "notícias", "atualizado", "recente", "últimas informações",
)

# ChromaDB — métrica de distância do índice HNSW.
# Default do Chroma é "l2"; com embeddings normalizados (normalize_embeddings=True) a
# ordem por L2 e por cosseno coincide, mas "cosine" torna os scores interpretáveis (0..1)
# e alinhados ao cosseno usado no ranqueamento de respostas (agent.avalia_similaridade).
CHROMA_DISTANCE_SPACE: str = "cosine"
CHROMA_COLLECTION_METADATA: dict[str, str] = {"hnsw:space": CHROMA_DISTANCE_SPACE}

# Geração — nº de respostas candidatas geradas por consulta antes do ranqueamento.
# Com LLM_TEMPERATURE=0 a geração é determinística: N candidatas seriam idênticas e o
# ranqueamento escolheria entre clones (N× custo de LLM sem ganho). Por isso o default é 1.
# Aumente apenas se LLM_TEMPERATURE > 0 (aí as candidatas divergem e o ranqueamento agrega valor).
NUM_CANDIDATE_RESPONSES: int = 1

# ChromaDB — nomes de coleção
DEFAULT_COLLECTION_NAME: str = "logistica"
COLLECTION_NAME_MIN_LEN: int = 3
COLLECTION_NAME_MAX_LEN: int = 63
COLLECTION_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$"
)

# Retrieval — k (nº de documentos) por coleção no fan-out de _get_retriever.
# Preparação para multi-collection: cada coleção pode ter seu próprio k de busca
# (ex.: coleções menores/mais específicas podem usar k menor). Coleções não listadas
# em RETRIEVAL_K_PER_COLLECTION usam RETRIEVAL_K_DEFAULT.
RETRIEVAL_K_DEFAULT: int = 3
RETRIEVAL_K_PER_COLLECTION: dict[str, int] = {
    DEFAULT_COLLECTION_NAME: 3,
}
RETRIEVAL_K_TOTAL: int = 3  # limite final de docs únicos após mesclar todas as coleções

# RAG Eval — thresholds do gate de qualidade no CI (ajustáveis; frouxos no início).
RAG_EVAL_MIN_HIT_RATE: float = 0.7    # Hit Rate mínimo para o CI passar
RAG_EVAL_MIN_MRR: float = 0.6         # MRR mínimo para o CI passar
RAG_EVAL_MATCH_THRESHOLD: float = 0.6  # cosseno mínimo chunk↔contexto_ref p/ contar "hit"
RAG_EVAL_K: int = 3                   # top-k avaliado (limitado por RETRIEVAL_K_TOTAL)
# Validação fail-fast (mesma convenção de LOG_LEVEL/HISTORY_MAX_ENTRIES).
for _nome, _valor in (
    ("RAG_EVAL_MIN_HIT_RATE", RAG_EVAL_MIN_HIT_RATE),
    ("RAG_EVAL_MIN_MRR", RAG_EVAL_MIN_MRR),
    ("RAG_EVAL_MATCH_THRESHOLD", RAG_EVAL_MATCH_THRESHOLD),
):
    if not 0.0 <= _valor <= 1.0:
        raise ValueError(
            f"{_nome}={_valor!r} must be within [0.0, 1.0]."
        )
if RAG_EVAL_K < 1:
    raise ValueError(
        f"RAG_EVAL_K={RAG_EVAL_K!r} must be >= 1."
    )

# Metadados unificados de chunks (REC-01)
METADATA_FILE_HASH: str = "file_hash"
METADATA_CHUNK_INDEX: str = "chunk_index"
METADATA_PAGE: str = "page"
METADATA_DOC_TYPE: str = "doc_type"
METADATA_DOC_TYPE_JSON: str = "json"
METADATA_DOC_TYPE_PDF: str = "pdf"
METADATA_PAGE_JSON_SENTINEL: int = 0

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


# _JsonFormatter removido na Fase 6 — importar de agenticlog.observability.logging
