# AgenticLog — Acceptance Tests: Health Check LMStudio Feature
"""
Verifica todos os critérios de aceite da story:
  "Como operador logístico, quero que o sistema verifique a disponibilidade do
   LMStudio antes de invocar o workflow, para receber feedback imediato quando
   o serviço estiver indisponível."

Cada teste mapeia a exatamente um critério de aceite (AC-HC-01 a AC-HC-04).
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock

import httpx

from agenticlog import check_lmstudio_health, LMStudioUnavailableError
from agenticlog.health import reset_health_check_sentinel
from agenticlog.config import LLM_HEALTH_CHECK_TIMEOUT_SECONDS, LLM_API_BASE


# ---------------------------------------------------------------------------
# AC-HC-01: Happy path — LMStudio responde com 2xx, workflow permanece acessível
# ---------------------------------------------------------------------------

class TestACHC01HappyPath(unittest.TestCase):
    """
    AC-HC-01: WHEN LMStudio responds with HTTP 2xx
    THEN check_lmstudio_health() SHALL return normally without raising.
    """

    def setUp(self) -> None:
        reset_health_check_sentinel()

    def tearDown(self) -> None:
        reset_health_check_sentinel()

    @patch("agenticlog.health.httpx.Client")
    def test_ac_hc_01_healthy_lmstudio_does_not_raise(
        self, mock_client_class: MagicMock
    ) -> None:
        """AC-HC-01: 2xx response → no exception raised, function returns normally."""
        mock_response = MagicMock()
        mock_response.is_success = True

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        try:
            check_lmstudio_health()
        except LMStudioUnavailableError as exc:
            self.fail(
                f"check_lmstudio_health() raised LMStudioUnavailableError unexpectedly: {exc}"
            )

        # AC-06: exactly ONE GET call — no retries
        self.assertEqual(mock_client_instance.get.call_count, 1)

        # AC-07: httpx.Client instantiated with timeout from config
        mock_client_class.assert_called_once_with(
            timeout=LLM_HEALTH_CHECK_TIMEOUT_SECONDS
        )


# ---------------------------------------------------------------------------
# AC-HC-02: ConnectError bloqueia o workflow
# ---------------------------------------------------------------------------

class TestACHC02ConnectErrorBlocksWorkflow(unittest.TestCase):
    """
    AC-HC-02: WHEN LMStudio is unreachable (ConnectError)
    THEN check_lmstudio_health() SHALL raise LMStudioUnavailableError
    AND the agent workflow SHALL NOT be called.
    """

    def setUp(self) -> None:
        reset_health_check_sentinel()

    def tearDown(self) -> None:
        reset_health_check_sentinel()

    @patch("agenticlog.health.logger")
    @patch("agenticlog.agent.agent_workflow")
    @patch("agenticlog.health.httpx.Client")
    def test_ac_hc_02_connect_error_raises_and_blocks_workflow(
        self,
        mock_client_class: MagicMock,
        mock_agent_workflow: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """AC-HC-02: ConnectError → LMStudioUnavailableError raised, agent_workflow not called."""
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = httpx.ConnectError("connection refused")
        mock_client_class.return_value = mock_client_instance

        with self.assertRaises(LMStudioUnavailableError):
            check_lmstudio_health()

        mock_agent_workflow.invoke.assert_not_called()

        # AC-08: logger.error called with URL and exception type name
        mock_logger.error.assert_called_once()
        log_args = mock_logger.error.call_args
        log_str = str(log_args)
        self.assertIn(LLM_API_BASE, log_str)
        self.assertIn("ConnectError", log_str)


# ---------------------------------------------------------------------------
# AC-HC-03: TimeoutException bloqueia o workflow com mensagem "tempo limite"
# ---------------------------------------------------------------------------

class TestACHC03TimeoutBlocksWorkflow(unittest.TestCase):
    """
    AC-HC-03: WHEN LMStudio health check times out (TimeoutException)
    THEN check_lmstudio_health() SHALL raise LMStudioUnavailableError
    AND the exception message SHALL contain "tempo limite".
    """

    def setUp(self) -> None:
        reset_health_check_sentinel()

    def tearDown(self) -> None:
        reset_health_check_sentinel()

    @patch("agenticlog.health.logger")
    @patch("agenticlog.health.httpx.Client")
    def test_ac_hc_03_timeout_raises_with_tempo_limite_message(
        self, mock_client_class: MagicMock, mock_logger: MagicMock
    ) -> None:
        """AC-HC-03: TimeoutException → LMStudioUnavailableError with 'tempo limite' in message."""
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.side_effect = httpx.TimeoutException(
            "request timed out"
        )
        mock_client_class.return_value = mock_client_instance

        with self.assertRaises(LMStudioUnavailableError) as ctx:
            check_lmstudio_health()

        self.assertIn("tempo limite", str(ctx.exception))

        # AC-08: logger.error called with URL and exception type name
        mock_logger.error.assert_called_once()
        log_args = mock_logger.error.call_args
        log_str = str(log_args)
        self.assertIn(LLM_API_BASE, log_str)
        self.assertIn("TimeoutException", log_str)


# ---------------------------------------------------------------------------
# AC-HC-04: Resposta não-2xx bloqueia o workflow com status code na mensagem
# ---------------------------------------------------------------------------

class TestACHC04NonSuccessStatusBlocksWorkflow(unittest.TestCase):
    """
    AC-HC-04: WHEN LMStudio returns a non-2xx HTTP status (e.g. 500)
    THEN check_lmstudio_health() SHALL raise LMStudioUnavailableError
    AND the exception message SHALL contain the status code "500".
    """

    def setUp(self) -> None:
        reset_health_check_sentinel()

    def tearDown(self) -> None:
        reset_health_check_sentinel()

    @patch("agenticlog.health.httpx.Client")
    def test_ac_hc_04_non_2xx_raises_with_status_code_in_message(
        self, mock_client_class: MagicMock
    ) -> None:
        """AC-HC-04: HTTP 500 response → LMStudioUnavailableError with '500' in message."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        with self.assertRaises(LMStudioUnavailableError) as ctx:
            check_lmstudio_health()

        self.assertIn("500", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
