# AgenticLog - Health check do LMStudio
"""
Verificação de disponibilidade do LMStudio antes de invocar o workflow.

Executa um único GET em {LLM_API_BASE}/models (fast-fail, sem retry) e confirma
que o modelo configurado (LLM_MODEL) está de fato carregado na lista retornada.
"""

import logging
from typing import Any

import httpx

from agenticlog.config import (
    LLM_API_BASE,
    LLM_HEALTH_CHECK_TIMEOUT_SECONDS,
    LLM_MODEL,
)

logger = logging.getLogger("agenticlog.health")

# Limite de ids de modelo logados no erro ModeloNaoCarregadoError (evita log verboso
# em instâncias LMStudio com muitos modelos).
MAX_MODELOS_LOG = 10

_health_checked = False


class LMStudioUnavailableError(Exception):
    """LMStudio inacessível ou resposta HTTP inválida no health check."""


class ModeloNaoCarregadoError(LMStudioUnavailableError):
    """LMStudio responde, mas o modelo configurado (LLM_MODEL) não está carregado."""


def reset_health_check_sentinel() -> None:
    """Reseta o sentinel `_health_checked` — uso exclusivo em testes."""
    global _health_checked
    _health_checked = False


def _extrair_ids_modelos(payload: Any) -> list[str]:
    """Extrai os ids de modelo do payload de GET /models (formato OpenAI).

    Entrada: payload — corpo JSON desserializado (esperado dict com chave "data").
    Saída: lista de ids de modelo (strings não vazias). Lista vazia se o formato
    não for reconhecido.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    ]


def check_lmstudio_health() -> None:
    """Verifica se o LMStudio responde em GET {LLM_API_BASE}/models.

    Raises:
        LMStudioUnavailableError: conexão recusada, timeout, status HTTP não-2xx
            ou corpo JSON inválido.
        ModeloNaoCarregadoError: servidor responde, mas LLM_MODEL não consta na
            lista de /models (subclasse de LMStudioUnavailableError).
    """
    global _health_checked

    url = f"{LLM_API_BASE.rstrip('/')}/models"
    try:
        with httpx.Client(timeout=LLM_HEALTH_CHECK_TIMEOUT_SECONDS) as client:
            response = client.get(url)
    except httpx.TimeoutException as exc:
        logger.error(
            "LMStudio health check falhou: url=%s exception_type=%s",
            url,
            type(exc).__name__,
        )
        raise LMStudioUnavailableError(
            "LMStudio não respondeu dentro do tempo limite do health check."
        ) from exc
    except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
        logger.error(
            "LMStudio health check falhou: url=%s exception_type=%s",
            url,
            type(exc).__name__,
        )
        raise LMStudioUnavailableError("LMStudio não está acessível.") from exc

    if not response.is_success:
        logger.error(
            "LMStudio health check falhou: url=%s exception_type=HTTPStatusError status=%s",
            url,
            response.status_code,
        )
        raise LMStudioUnavailableError(
            f"LMStudio retornou status HTTP {response.status_code}."
        )

    try:
        payload = response.json()
    except (ValueError, httpx.DecodingError) as exc:
        logger.error(
            "LMStudio health check falhou: url=%s exception_type=%s",
            url,
            type(exc).__name__,
        )
        raise LMStudioUnavailableError(
            "LMStudio retornou um corpo de resposta inválido (JSON malformado)."
        ) from exc

    model_ids = _extrair_ids_modelos(payload)
    if LLM_MODEL not in model_ids:
        logger.error(
            "LMStudio health check falhou: url=%s exception_type=ModeloNaoCarregadoError "
            "modelo_esperado=%s modelos_disponiveis=%s",
            url,
            LLM_MODEL,
            model_ids[:MAX_MODELOS_LOG],
        )
        raise ModeloNaoCarregadoError(
            f"O modelo '{LLM_MODEL}' não está carregado no LMStudio. "
            f"Modelos disponíveis: {model_ids or 'nenhum'}."
        )

    _health_checked = True
