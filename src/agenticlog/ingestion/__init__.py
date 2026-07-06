# AgenticLog - Pacote de ingestão
"""Pacote `ingestion`: estágios de baixo risco do pipeline de ingestão (ADR-018 Fase 3a).

Extraído de `agenticlog.rag` preservando identidade de símbolo via shims em `rag.py`.
Depende exclusivamente de `agenticlog.config` e `agenticlog.shared` (grafo acíclico);
nenhum submódulo importa `rag`/`agent`. Aresta intra-pacote: `security → extraction`.

Re-exporta os símbolos públicos para import ergonômico:
`from agenticlog.ingestion import salvar_documento_enviado, criar_embedding_model`.
"""

from agenticlog.ingestion.chunking import SemanticChunker
from agenticlog.ingestion.cleaning import filtrar_documentos_vazios
from agenticlog.ingestion.embeddings import criar_embedding_model
from agenticlog.ingestion.extraction import carregar_json, extrair_texto_pdf
from agenticlog.ingestion.metadata import (
    _computar_hash_conteudo,
    _enriquecer_metadados_chunks,
    _hash_arquivo,
)
from agenticlog.ingestion.orchestrator import (
    adicionar_documento_incrementalmente,
    adicionar_pdf_incrementalmente,
    cria_vectordb,
    ingerir_incrementalmente,
    reconstruir_vectordb,
)
from agenticlog.ingestion.security import (
    _sanitizar_nome_arquivo,
    _sanitizar_nome_colecao,
    _valida_arquivos_json,
    _valida_json_sem_chaves_proibidas,
    _valida_path_documentos,
    salvar_documento_enviado,
    salvar_pdf_enviado,
    sanitizar_nome_colecao,
)
from agenticlog.ingestion.store import (
    _backup_arquivo,
    _outras_colecoes_existem,
    _resetar_colecao,
    _reverter_disco,
    add_documents_com_rollback,
)

__all__ = [
    "SemanticChunker",
    "filtrar_documentos_vazios",
    "criar_embedding_model",
    "carregar_json",
    "extrair_texto_pdf",
    "_computar_hash_conteudo",
    "_hash_arquivo",
    "_enriquecer_metadados_chunks",
    "_valida_path_documentos",
    "_valida_json_sem_chaves_proibidas",
    "_valida_arquivos_json",
    "_sanitizar_nome_arquivo",
    "_sanitizar_nome_colecao",
    "sanitizar_nome_colecao",
    "salvar_documento_enviado",
    "salvar_pdf_enviado",
    # ADR-018 Fase 3b — store (persistência/atomicidade)
    "_backup_arquivo",
    "_reverter_disco",
    "_outras_colecoes_existem",
    "_resetar_colecao",
    "add_documents_com_rollback",
    # ADR-018 Fase 3b — orchestrator (orquestradores de ingestão)
    "cria_vectordb",
    "adicionar_documento_incrementalmente",
    "adicionar_pdf_incrementalmente",
    "ingerir_incrementalmente",
    "reconstruir_vectordb",
]
