# AgenticLog - Recuperação vetorial (ADR-018 Fase 4 + Fase 6)
"""
Módulo de recuperação vetorial ChromaDB.

Extraído de `agent.py` (ADR-018 Fase 4). Contém a factory de embedding
(`_build_embedding_model`), parametrização de ChromaDB (`_get_vector_db`,
`_listar_colecoes`), fan-out multi-coleção (`_get_retriever`) e
invalidação de cache (`invalidar_vector_db`).

A partir da Fase 6, os singletons `_vector_dbs`, `_embedding_model` e os
getters associados são definidos LOCALMENTE (não mais importados de agent.py).
"""

import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from agenticlog.config import (
    CHROMA_COLLECTION_METADATA,
    DEFAULT_COLLECTION_NAME,
    DIR_VECTORDB,
    RETRIEVAL_K_DEFAULT,
    RETRIEVAL_K_PER_COLLECTION,
    RETRIEVAL_K_TOTAL,
)
from agenticlog.ingestion.embeddings import criar_embedding_model

logger = logging.getLogger(__name__)

# Singletons lazy — inicializados somente na primeira chamada
_vector_dbs: dict[str, Chroma] = {}
_embedding_model = None


def _build_embedding_model() -> HuggingFaceEmbeddings:
    """Factory do modelo de embedding — sem singleton.

    Usado por _get_embedding_model() (getter com cache local).

    Saída: instância de HuggingFaceEmbeddings criada via criar_embedding_model().
    """
    return criar_embedding_model()


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna o singleton do modelo de embeddings, criando-o na primeira chamada.

    Entrada: nenhuma.
    Saída: instância de HuggingFaceEmbeddings.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _build_embedding_model()
    return _embedding_model


def _get_vector_db(
    collection_name: str = DEFAULT_COLLECTION_NAME, *, vectordb_dir: Path | None = None
) -> Chroma:
    """Retorna singleton do ChromaDB para a coleção indicada, criando-o na primeira chamada.

    Entrada: collection_name — nome da coleção ChromaDB;
             vectordb_dir — diretório persistido (fallback: config.DIR_VECTORDB).
    Saída: instância de Chroma conectada ao diretório persistido para essa coleção.
    """
    actual_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir

    if collection_name not in _vector_dbs:
        _vector_dbs[collection_name] = Chroma(
            persist_directory=str(actual_dir),
            collection_name=collection_name,
            embedding_function=_get_embedding_model(),
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )
    return _vector_dbs[collection_name]


def _listar_colecoes(*, vectordb_dir: Path | None = None) -> list[str]:
    """Retorna os nomes de todas as coleções presentes no ChromaDB persistido em disco.

    Entrada: vectordb_dir — diretório persistido (fallback: config.DIR_VECTORDB).
    Saída: lista de nomes de coleção (strings). Retorna [DEFAULT_COLLECTION_NAME] em caso de
    erro ou resultado vazio para garantir que o agente sempre consulte ao menos uma coleção.

    Usa lazy import de chromadb para evitar efeitos colaterais na importação do módulo.
    """
    actual_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
    try:
        import chromadb  # lazy — evita side-effects na importação do módulo
        client = chromadb.PersistentClient(path=str(actual_dir))
        collections = client.list_collections()
        names = [
            col.name if hasattr(col, "name") else str(col)
            for col in collections
        ]
        if not names:
            return [DEFAULT_COLLECTION_NAME]
        return names
    except Exception as exc:
        logger.warning("Falha ao listar coleções ChromaDB; usando coleção padrão. Detalhe: %s", exc)
        return [DEFAULT_COLLECTION_NAME]


def _get_retriever(query: str) -> list[Document]:
    """Executa fan-out em todas as coleções ChromaDB e retorna até \
RETRIEVAL_K_TOTAL documentos únicos.

    Entrada: query — texto da consulta do usuário.
    Saída: lista de até RETRIEVAL_K_TOTAL documentos únicos (deduplicados por MD5 do page_content).

    Cada coleção usa seu próprio k de busca (RETRIEVAL_K_PER_COLLECTION, com fallback
    RETRIEVAL_K_DEFAULT) antes da deduplicação e do corte final.
    """
    collection_names = _listar_colecoes()
    all_docs: list[Document] = []

    for name in collection_names:
        db = _get_vector_db(name)
        k = RETRIEVAL_K_PER_COLLECTION.get(name, RETRIEVAL_K_DEFAULT)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(query)
        all_docs.extend(docs)

    seen: set[str] = set()
    unique: list[Document] = []
    for doc in all_docs:
        key = hashlib.md5(doc.page_content.encode()).hexdigest()  # nosec B324
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique[:RETRIEVAL_K_TOTAL]


def invalidar_vector_db() -> None:
    """Invalida todos os singletons de ChromaDB para que a próxima chamada reconecte ao disco.

    Entrada: nenhuma.
    Saída: nenhuma.
    Efeito colateral: limpa o dicionário _vector_dbs local.
    """
    _vector_dbs.clear()
