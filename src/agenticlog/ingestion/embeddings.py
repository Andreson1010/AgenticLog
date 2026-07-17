# AgenticLog - Estágio de embeddings da ingestão
"""Factory e singleton do modelo de embedding (ADR-018 Fase 3a + Fase 6).

Contém a construção do modelo (`criar_embedding_model`), o cache singleton
`_rag_embedding_model` e o getter `_get_rag_embedding_model` — realocados de
`agenticlog.rag` na Fase 6.
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from agenticlog.config import EMBEDDING_MODEL

_rag_embedding_model = None


def _get_rag_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna singleton de HuggingFaceEmbeddings para ingestão incremental.

    Entrada: nenhuma.
    Saída: instância de HuggingFaceEmbeddings (criada uma única vez por processo).
    """
    global _rag_embedding_model
    if _rag_embedding_model is None:
        _rag_embedding_model = criar_embedding_model()
    return _rag_embedding_model


def criar_embedding_model() -> HuggingFaceEmbeddings:
    """Constrói o modelo de embedding HuggingFace (device auto + normalize).

    Entrada: nenhuma.
    Saída: instância de HuggingFaceEmbeddings.

    A normalização (normalize_embeddings=True) é crítica: garante o MESMO espaço
    vetorial do rebuild (cria_vectordb) e do agente — sem ela, chunks ingeridos
    incrementalmente teriam normas diferentes, degradando a similaridade silenciosamente.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
