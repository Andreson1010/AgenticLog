# AgenticLog - Estágio de metadados da ingestão
"""Hash de conteúdo e enriquecimento de metadados de chunks (ADR-018 Fase 3a).

Movido verbatim de `agenticlog.rag`. Importa apenas de `agenticlog.config`
(folha do grafo de imports), mantendo o pacote `ingestion` acíclico.
"""

import hashlib
from pathlib import Path

from agenticlog.config import (
    METADATA_CHUNK_INDEX,
    METADATA_DOC_TYPE,
    METADATA_FILE_HASH,
    METADATA_PAGE,
)


def _computar_hash_conteudo(conteudo: bytes) -> str:
    """Computa o hash SHA-256 do conteúdo binário do arquivo.

    Entrada: conteudo — bytes do arquivo.
    Saída: string hexadecimal de 64 caracteres (SHA-256).
    """
    return hashlib.sha256(conteudo).hexdigest()


def _hash_arquivo(path: str) -> str:
    """Lê arquivo do disco e retorna SHA-256 do conteúdo (REC-01)."""
    return _computar_hash_conteudo(Path(path).read_bytes())


def _enriquecer_metadados_chunks(
    chunks: list, file_hash: str, doc_type: str, page: int | None = None
) -> None:
    """Enriquece chunks in-place com campos unificados de metadados (REC-01).

    page=None preserva o valor já presente em chunk.metadata (PDF: herdado do Document pai).
    """
    for idx, chunk in enumerate(chunks):
        chunk.metadata[METADATA_FILE_HASH] = file_hash
        chunk.metadata[METADATA_CHUNK_INDEX] = idx
        chunk.metadata[METADATA_DOC_TYPE] = doc_type
        if page is not None:
            chunk.metadata[METADATA_PAGE] = page
