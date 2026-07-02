# AgenticLog - Estágio de embeddings da ingestão
"""Factory de construção do modelo de embedding (ADR-018 Fase 3a).

Contém APENAS a construção do modelo (verbatim de rag.py). O singleton global
`_rag_embedding_model` e o getter `_get_rag_embedding_model` FICAM em `agenticlog.rag`
para preservar o monkeypatch do oráculo de caracterização (ver design §5).
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from agenticlog.config import EMBEDDING_MODEL


def criar_embedding_model() -> HuggingFaceEmbeddings:
    """Constrói o modelo de embedding HuggingFace (device auto + normalize).

    Entrada: nenhuma.
    Saída: instância de HuggingFaceEmbeddings.

    A flag `encode_kwargs={"normalize_embeddings": True}` é crítica: garante o MESMO
    espaço vetorial do rebuild (cria_vectordb) e do agente — sem ela, chunks ingeridos
    incrementalmente teriam normas diferentes, degradando a similaridade silenciosamente.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
