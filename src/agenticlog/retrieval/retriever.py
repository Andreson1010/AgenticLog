# AgenticLog - Recuperação vetorial (ADR-018 Fase 4)
"""
Módulo de recuperação vetorial ChromaDB.

Extraído de `agent.py` (ADR-018 Fase 4). Contém a factory de embedding
(`_build_embedding_model`), parametrização de ChromaDB (`_get_vector_db`,
`_listar_colecoes`), fan-out multi-coleção (`_get_retriever`) e
invalidação de cache (`invalidar_vector_db`).

`_build_embedding_model` é uma factory pura SEM singleton — o singleton
`_embedding_model` fica em `agent.py` e é gerenciado pelo wrapper
`_get_embedding_model` (ver DN-2a).

`_get_vector_db` e `_listar_colecoes` são PARAMETRIZADAS por `vectordb_dir`
— não capturam `DIR_VECTORDB` no import. Os WRAPPERS em `agent.py`
resolvem `agent.DIR_VECTORDB` no call time (RETR-11, ADR-019 D4).

`_get_retriever` acessa `_listar_colecoes` e `_get_vector_db` via lazy
import de `agenticlog.agent` (os wrappers — DN-3), garantindo que
`DIR_VECTORDB` seja resolvido no call time.

NADA importa `agent` em nível de módulo — todos os acessos são lazy
imports DENTRO de funções (DN-2, RETR-13).
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


def _build_embedding_model() -> HuggingFaceEmbeddings:
    """Factory do modelo de embedding — sem singleton.

    Usado por agent._get_embedding_model() (wrapper em agent.py, DN-2a).

    Saída: instância de HuggingFaceEmbeddings criada via criar_embedding_model().
    """
    return criar_embedding_model()


def _get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME, *, vectordb_dir: Path | None = None) -> Chroma:
    """Retorna singleton do ChromaDB para a coleção indicada, criando-o na primeira chamada.

    Entrada: collection_name — nome da coleção ChromaDB;
             vectordb_dir — diretório persistido (call-time, resolvido pelo wrapper de agent).
    Saída: instância de Chroma conectada ao diretório persistido para essa coleção.

    O singleton _vector_dbs fica em agent.py; acessado via lazy import.
    """
    from agenticlog.agent import _vector_dbs  # lazy — singleton de agent (DN-2)
    from agenticlog.agent import _get_embedding_model as _get_emb  # lazy — wrapper de agent (DN-2a)

    actual_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir

    if collection_name not in _vector_dbs:
        _vector_dbs[collection_name] = Chroma(
            persist_directory=str(actual_dir),
            collection_name=collection_name,
            embedding_function=_get_emb(),
            collection_metadata=CHROMA_COLLECTION_METADATA,
        )
    return _vector_dbs[collection_name]


def _listar_colecoes(*, vectordb_dir: Path | None = None) -> list[str]:
    """Retorna os nomes de todas as coleções presentes no ChromaDB persistido em disco.

    Entrada: vectordb_dir — diretório persistido (call-time, resolvido pelo wrapper de agent).
    Saída: lista de nomes de coleção (strings). Retorna [DEFAULT_COLLECTION_NAME] em caso de
    erro ou resultado vazio para garantir que o agente sempre consulte ao menos uma coleção.

    Usa lazy import de chromadb para evitar efeitos colaterais na importação do módulo.
    """
    actual_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
    try:
        import chromadb  # lazy — evita side-effects na importação do módulo
        client = chromadb.PersistentClient(path=str(actual_dir))
        collections = client.list_collections()
        # chromadb 0.4.x returns Collection objects with .name attribute
        # chromadb >= 0.6 may return strings — handle both
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
    """Executa fan-out em todas as coleções ChromaDB e retorna até RETRIEVAL_K_TOTAL documentos únicos.

    Entrada: query — texto da consulta do usuário.
    Saída: lista de até RETRIEVAL_K_TOTAL documentos únicos (deduplicados por MD5 do page_content).

    Cada coleção usa seu próprio k de busca (RETRIEVAL_K_PER_COLLECTION, com fallback
    RETRIEVAL_K_DEFAULT) antes da deduplicação e do corte final.
    Coleção vazia contribui 0 documentos (skip silencioso).
    Erro ChromaDB em qualquer coleção propaga imediatamente (fail-fast).

    Acessa _listar_colecoes e _get_vector_db via lazy import dos WRAPPERS de agent.py
    (DN-3), garantindo que DIR_VECTORDB seja resolvido no call time.
    """
    from agenticlog.agent import _listar_colecoes  # lazy — o wrapper de agent (DN-3)
    from agenticlog.agent import _get_vector_db   # lazy — o wrapper de agent (DN-3)

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
    Efeito colateral: limpa o dicionário _vector_dbs global de agent.py via lazy import.
    """
    import agenticlog.agent as _agent_mod  # lazy — singleton de agent (DN-2)
    _agent_mod._vector_dbs.clear()
