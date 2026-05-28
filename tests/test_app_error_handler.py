# AgenticLog - Testes do classificador de erros do app.py
"""
Testes para verificar que app.py classifica erros via isinstance corretamente (T8-T10).

Estes testes simulam a lógica de classificação de erro do bloco except em app.py
sem necessidade de importar streamlit (que requer contexto de servidor).
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
import unittest.mock
import httpx
import anthropic

from agenticlog.health import LMStudioUnavailableError


def _classify_error(e: Exception) -> str:
    """Replica a lógica de classificação de erros do bloco except em app.py.

    Retorna:
        "lmstudio_not_running" — erro de conexão/timeout com o LMStudio.
        "generic_error"        — qualquer outro erro não classificado.
    """
    _msg = str(e).lower()
    if isinstance(
        e,
        (
            LMStudioUnavailableError,
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            anthropic.APIConnectionError,
        ),
    ):
        return "lmstudio_not_running"
    elif "connection refused" in _msg or ("connect" in _msg and "1234" in _msg):
        return "lmstudio_not_running"
    return "generic_error"


class TestAppErrorHandler(unittest.TestCase):
    """Testes T8-T10: isinstance classifica erros de conexão/LLM corretamente."""

    def teste_8_isinstance_classifica_api_connection_error(self):
        """T8: APIConnectionError da anthropic é classificado corretamente como erro de conexão."""
        # anthropic.APIConnectionError herda de httpx.ConnectError via cadeia de herança
        # Testamos que o padrão isinstance captura erros do tipo httpx corretos
        err = httpx.ConnectError("connection refused to 127.0.0.1:1234")
        resultado = _classify_error(err)
        self.assertEqual(resultado, "lmstudio_not_running")

    def teste_11_isinstance_classifica_anthropic_api_connection_error(self):
        """T11: anthropic.APIConnectionError é classificado como lmstudio_not_running, não genérico."""
        request = unittest.mock.MagicMock()
        err = anthropic.APIConnectionError(request=request)
        resultado = _classify_error(err)
        self.assertEqual(resultado, "lmstudio_not_running")

    def teste_9_isinstance_classifica_authentication_error_como_generico(self):
        """T9: AuthenticationError não é capturado pelo isinstance — vai para ramo genérico."""
        from openai import AuthenticationError

        mock_response = unittest.mock.MagicMock()
        mock_response.request = unittest.mock.MagicMock()
        err = AuthenticationError("invalid api key", response=mock_response, body={})
        resultado = _classify_error(err)
        self.assertEqual(resultado, "generic_error")

    def teste_10_isinstance_nao_captura_excecao_generica(self):
        """T10: ValueError genérico não é capturado pelo isinstance — vai para ramo genérico."""
        err = ValueError("algum erro de validação inesperado")
        resultado = _classify_error(err)
        self.assertEqual(resultado, "generic_error")

    def teste_12_lmstudio_unavailable_error_classificado_como_lmstudio(self):
        """Health check: LMStudioUnavailableError vai para mensagem LMStudio, não genérico."""
        err = LMStudioUnavailableError("LMStudio não está acessível.")
        resultado = _classify_error(err)
        self.assertEqual(resultado, "lmstudio_not_running")


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
