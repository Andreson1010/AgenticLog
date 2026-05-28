# AgenticLog - Testes do health check LMStudio
"""Testes para check_lmstudio_health() e LMStudioUnavailableError."""

import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import httpx

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from agenticlog.config import LLM_API_BASE, LLM_HEALTH_CHECK_TIMEOUT_SECONDS
from agenticlog.health import (
    LMStudioUnavailableError,
    _health_checked,
    check_lmstudio_health,
    reset_health_check_sentinel,
)
import agenticlog.health as health_module

_MODELS_URL = f"{LLM_API_BASE.rstrip('/')}/models"


class TestLmstudioHealthCheck(TestCase):
    def setUp(self):
        reset_health_check_sentinel()

    def tearDown(self):
        reset_health_check_sentinel()

    @patch.object(health_module, "logger")
    @patch("agenticlog.health.httpx.Client")
    def teste_1_happy_path_retorna_sem_excecao(self, mock_client_cls, mock_logger):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        check_lmstudio_health()

        mock_client_cls.assert_called_once_with(timeout=LLM_HEALTH_CHECK_TIMEOUT_SECONDS)
        mock_client.get.assert_called_once_with(_MODELS_URL)
        self.assertTrue(health_module._health_checked)
        mock_logger.error.assert_not_called()

    @patch.object(health_module, "logger")
    @patch("agenticlog.health.httpx.Client")
    def teste_2_connect_error_levanta_lmstudio_unavailable(self, mock_client_cls, mock_logger):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("connection refused")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with self.assertRaises(LMStudioUnavailableError):
            check_lmstudio_health()

        mock_client.get.assert_called_once()
        mock_logger.error.assert_called_once()
        self.assertIn("url=", mock_logger.error.call_args[0][0])

    @patch.object(health_module, "logger")
    @patch("agenticlog.health.httpx.Client")
    def teste_3_timeout_levanta_mensagem_de_timeout(self, mock_client_cls, mock_logger):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with self.assertRaises(LMStudioUnavailableError) as ctx:
            check_lmstudio_health()

        self.assertIn("tempo limite", str(ctx.exception).lower())
        mock_logger.error.assert_called_once()

    @patch.object(health_module, "logger")
    @patch("agenticlog.health.httpx.Client")
    def teste_4_status_nao_2xx_levanta_lmstudio_unavailable(self, mock_client_cls, mock_logger):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with self.assertRaises(LMStudioUnavailableError) as ctx:
            check_lmstudio_health()

        self.assertIn("500", str(ctx.exception))
        mock_logger.error.assert_called_once()

    @patch.object(health_module, "logger")
    @patch("agenticlog.health.httpx.Client")
    def teste_5_lista_vazia_de_modelos_tratada_como_saudavel(self, mock_client_cls, mock_logger):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        check_lmstudio_health()

        mock_logger.error.assert_not_called()

    @patch("agenticlog.health.httpx.Client")
    def teste_6_sentinel_resetavel_entre_testes(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        self.assertFalse(health_module._health_checked)
        check_lmstudio_health()
        self.assertTrue(health_module._health_checked)
        reset_health_check_sentinel()
        self.assertFalse(health_module._health_checked)

    @patch("agenticlog.health.httpx.Client")
    def teste_7_exportado_via_init(self, mock_client_cls):
        from agenticlog import check_lmstudio_health as exported_check
        from agenticlog import LMStudioUnavailableError as exported_error

        self.assertIs(exported_check, check_lmstudio_health)
        self.assertIs(exported_error, LMStudioUnavailableError)


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
