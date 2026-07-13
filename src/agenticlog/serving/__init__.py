"""Pacote `serving`: camada de entrega FastAPI + health check (ADR-018 Fase 5).

Extraido de agenticlog.api e agenticlog.health preservando identidade de simbolo
via shims em api.py e health.py.

Nota: `__getattr__` evita import circular com agenticlog.health (shim), que importa
de agenticlog.serving.health, que por sua vez dispara __init__.py.
"""

from __future__ import annotations

import typing

__all__ = [
    # api
    "app",
    "consultar",
    "listar_historico",
    "QueryRequest",
    "QueryResponse",
    "HistoryEntry",
    "DocumentInfo",
    "lifespan",
    "_verificar_vectordb",
    "_serializar_documentos",
    "_normalizar_estado",
    "_resposta_segura",
    "_construir_registro",
    "handler_lmstudio",
    "handler_connect_error",
    "handler_generico",
    "MSG_LMSTUDIO_INDISPONIVEL",
    "MSG_VECTORDB_AUSENTE",
    "_ERROS_MODO_SEGURO",
    # health
    "check_lmstudio_health",
    "reset_health_check_sentinel",
    "LMStudioUnavailableError",
    "ModeloNaoCarregadoError",
    "_health_checked",
    "_extrair_ids_modelos",
    "MAX_MODELOS_LOG",
]

# Mapa: nome do simbolo -> (modulo de origem, nome no modulo)
_LAZY_MAP: dict[str, tuple[str, str]] = {
    # api symbols
    "app": ("agenticlog.serving.api", "app"),
    "consultar": ("agenticlog.serving.api", "consultar"),
    "listar_historico": ("agenticlog.serving.api", "listar_historico"),
    "QueryRequest": ("agenticlog.serving.api", "QueryRequest"),
    "QueryResponse": ("agenticlog.serving.api", "QueryResponse"),
    "HistoryEntry": ("agenticlog.serving.api", "HistoryEntry"),
    "DocumentInfo": ("agenticlog.serving.api", "DocumentInfo"),
    "lifespan": ("agenticlog.serving.api", "lifespan"),
    "_verificar_vectordb": ("agenticlog.serving.api", "_verificar_vectordb"),
    "_serializar_documentos": ("agenticlog.serving.api", "_serializar_documentos"),
    "_normalizar_estado": ("agenticlog.serving.api", "_normalizar_estado"),
    "_resposta_segura": ("agenticlog.serving.api", "_resposta_segura"),
    "_construir_registro": ("agenticlog.serving.api", "_construir_registro"),
    "handler_lmstudio": ("agenticlog.serving.api", "handler_lmstudio"),
    "handler_connect_error": ("agenticlog.serving.api", "handler_connect_error"),
    "handler_generico": ("agenticlog.serving.api", "handler_generico"),
    "MSG_LMSTUDIO_INDISPONIVEL": ("agenticlog.serving.api", "MSG_LMSTUDIO_INDISPONIVEL"),
    "MSG_VECTORDB_AUSENTE": ("agenticlog.serving.api", "MSG_VECTORDB_AUSENTE"),
    "_ERROS_MODO_SEGURO": ("agenticlog.serving.api", "_ERROS_MODO_SEGURO"),
    # health symbols
    "check_lmstudio_health": ("agenticlog.serving.health", "check_lmstudio_health"),
    "reset_health_check_sentinel": ("agenticlog.serving.health", "reset_health_check_sentinel"),
    "LMStudioUnavailableError": ("agenticlog.serving.health", "LMStudioUnavailableError"),
    "ModeloNaoCarregadoError": ("agenticlog.serving.health", "ModeloNaoCarregadoError"),
    "_health_checked": ("agenticlog.serving.health", "_health_checked"),
    "_extrair_ids_modelos": ("agenticlog.serving.health", "_extrair_ids_modelos"),
    "MAX_MODELOS_LOG": ("agenticlog.serving.health", "MAX_MODELOS_LOG"),
}


def __getattr__(name: str) -> typing.Any:
    """Lazy import de submodulos para evitar import circular com agenticlog.health shim."""
    if name in _LAZY_MAP:
        mod_path, attr_name = _LAZY_MAP[name]
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, attr_name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)
