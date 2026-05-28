# AgenticLog - Configurações centralizadas
"""Constantes, paths e parâmetros de modelos."""

from pathlib import Path

# Raiz do projeto (pasta que contém src/, data/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # âncora para todos os paths relativos

# Diretórios de dados
DIR_DOCUMENTS = PROJECT_ROOT / "data" / "documents"  # JSONs de origem dos documentos
DIR_VECTORDB = PROJECT_ROOT / "data" / "vectordb"    # banco ChromaDB persistido em disco

# Modelo de embeddings (deve ser o mesmo em rag e agent)
EMBEDDING_MODEL = "BAAI/bge-base-en"  # modelo HuggingFace usado para gerar e consultar embeddings

# LLM (LMStudio)
LLM_MODEL = "hermes-3-llama-3.2-3b"          # identificador do modelo carregado no LMStudio
LLM_API_BASE = "http://127.0.0.1:1234/v1"    # endpoint local do LMStudio (compatível com OpenAI API)
LLM_API_KEY = "hermes"                        # chave fictícia exigida pelo cliente OpenAI
LLM_TEMPERATURE = 0                           # temperatura 0 para respostas determinísticas
LLM_MAX_TOKENS = 2048                         # limite de tokens gerados por resposta
LLM_TIMEOUT_SECONDS: float = 10.0            # timeout por chamada ao LLM em segundos
LLM_MAX_RETRY_ATTEMPTS: int = 3              # número máximo de tentativas com retry
LLM_RETRY_WAIT_INITIAL_SECONDS: float = 1.0  # espera inicial do backoff exponencial em segundos
LLM_RETRY_WAIT_MAX_SECONDS: float = 4.0      # espera máxima do backoff exponencial em segundos
LLM_HEALTH_CHECK_TIMEOUT_SECONDS: float = 5.0  # timeout do GET /v1/models antes do workflow

# RAG
CHUNK_SIZE = 500    # tamanho máximo de cada chunk de texto em caracteres
CHUNK_OVERLAP = 50  # sobreposição entre chunks para preservar contexto nas bordas

# Segurança - limites para carregamento de documentos
MAX_JSON_FILES = 1000          # impede carregamento irrestrito de arquivos maliciosos
MAX_JSON_FILE_SIZE_MB = 10     # bloqueia arquivos excessivamente grandes (proteção contra DoS)
FORBIDDEN_JSON_KEYS = ("lc",)  # mitiga injeção via chave "lc" usada pela classe Serializable do LangChain

# Logging
LOG_LEVEL: str = "INFO"  # nível de log para handlers configurados pelo chamador
