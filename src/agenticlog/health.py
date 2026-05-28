# AgenticLog - Health check do LMStudio
"""
Verificação de disponibilidade do LMStudio antes de invocar o workflow.

Executa um único GET em {LLM_API_BASE}/models (fast-fail, sem retry).
"""

import logging

import httpx

from agenticlog.config import LLM_API_BASE, LLM_HEALTH_CHECK_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_health_checked = False


class LMStudioUnavailableError(Exception):
    """LMStudio inacessível ou resposta HTTP inválida no health check."""


def reset_health_check_sentinel() -> None:
    """Reseta o sentinel `_health_checked` — uso exclusivo em testes."""
    global _health_checked
    _health_checked = False


def check_lmstudio_health() -> None:
    """Verifica se o LMStudio responde em GET {LLM_API_BASE}/models.

    Raises:
        LMStudioUnavailableError: conexão recusada, timeout ou status HTTP não-2xx.
    """
    global _health_checked
    _health_checked = True

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
