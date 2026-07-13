"""Camada FastAPI do AgenticLog — shim de compatibilidade (ADR-018 Fase 5).

Re-exporta todos os simbolos de agenticlog.serving.api.
Remove na Fase 6.
"""

from agenticlog.serving.api import (  # noqa: F401  # Re-export shim (ADR-018 Fase 5) — remover na Fase 6
    _ERROS_MODO_SEGURO,
    # Constantes
    MSG_LMSTUDIO_INDISPONIVEL,
    MSG_VECTORDB_AUSENTE,
    # Modelos Pydantic
    DocumentInfo,
    HistoryEntry,
    HistoryStore,
    QueryRequest,
    QueryResponse,
    # Helpers
    _construir_registro,
    _normalizar_estado,
    _resposta_segura,
    _serializar_documentos,
    _verificar_vectordb,
    # FastAPI
    app,
    # Endpoints
    consultar,
    # Exception handlers
    handler_connect_error,
    handler_generico,
    handler_lmstudio,
    lifespan,
    listar_historico,
)
