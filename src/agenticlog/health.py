# AgenticLog - Health check do LMStudio — shim de compatibilidade (ADR-018 Fase 5)
"""Re-exporta todos os simbolos de agenticlog.serving.health.
Remove na Fase 6.
"""

import logging

import httpx  # noqa: F401  # mantido p/ @patch("agenticlog.health.httpx.Client")

from agenticlog.serving.health import (  # noqa: E402,F401  # Re-export shim (ADR-018 Fase 5) — remover na Fase 6
    MAX_MODELOS_LOG,
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    _extrair_ids_modelos,
    _health_checked,
    check_lmstudio_health,
    reset_health_check_sentinel,
)

logger = logging.getLogger("agenticlog.health")
