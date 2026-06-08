# AgenticLog - Testes do tratamento de erros HTTP em app.py
"""
Testes para app.py usando streamlit.testing.v1.AppTest.
Cobre todos os ramos de erro do botão "Enviar":
  503 LMStudio, 503 vectordb, 500, 422, ConnectError, TimeoutException.

Mock strategy: patch("app.httpx.post") retornando MagicMock configurado por caso.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(_root / "app.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_error_response(status_code: int, detail: str) -> MagicMock:
    """Cria um MagicMock de resposta HTTP com raise_for_status levantando HTTPStatusError."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {"detail": detail}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=MagicMock(),
        response=mock_response,
    )
    return mock_response


# ---------------------------------------------------------------------------
# Classe de testes
# ---------------------------------------------------------------------------

class TestAppErrorHandler(unittest.TestCase):
    """Testes de tratamento de erros HTTP para app.py."""

    def _run_with_query(self, mock_post: MagicMock) -> AppTest:
        """Executa o app com uma consulta e retorna o AppTest após o clique.

        mock_post deve ser um MagicMock pronto para uso como httpx.post —
        com return_value=response_mock ou side_effect=exception, conforme o caso.
        """
        with patch("app.httpx.post", mock_post):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta de teste").run()
            at.button[0].click().run()
        return at

    def teste_1_erro_503_lmstudio(self) -> None:
        """503 com detalhe LMStudio: st.error exibe MSG_LMSTUDIO_DOWN."""
        from app import MSG_LMSTUDIO_DOWN

        mock_response = _make_error_response(
            503,
            "LMStudio indisponível. Inicie o servidor e carregue o modelo.",
        )
        at = self._run_with_query(MagicMock(return_value=mock_response))

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any(MSG_LMSTUDIO_DOWN in t for t in error_texts),
            msg=f"MSG_LMSTUDIO_DOWN não encontrada em: {error_texts}",
        )
        self.assertIsNone(at.session_state.ranked_response)

    def teste_2_erro_503_vectordb(self) -> None:
        """503 com detalhe vectordb: st.error exibe MSG_VECTORDB_AUSENTE."""
        from app import MSG_VECTORDB_AUSENTE

        mock_response = _make_error_response(
            503,
            "Base vetorial não encontrada. Execute: python -m agenticlog.rag",
        )
        at = self._run_with_query(MagicMock(return_value=mock_response))

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any(MSG_VECTORDB_AUSENTE in t for t in error_texts),
            msg=f"MSG_VECTORDB_AUSENTE não encontrada em: {error_texts}",
        )
        self.assertIsNone(at.session_state.ranked_response)

    def teste_3_erro_500(self) -> None:
        """500: st.error exibe MSG_ERRO_INTERNO e expander contém o detalhe."""
        from app import MSG_ERRO_INTERNO

        mock_response = _make_error_response(500, "Internal server error details.")
        at = self._run_with_query(MagicMock(return_value=mock_response))

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any(MSG_ERRO_INTERNO in t for t in error_texts),
            msg=f"MSG_ERRO_INTERNO não encontrada em: {error_texts}",
        )
        self.assertGreater(len(at.expander), 0, msg="Expander de detalhes não encontrado.")
        self.assertIsNone(at.session_state.ranked_response)

    def teste_4_erro_422(self) -> None:
        """422: st.error exibe MSG_ERRO_VALIDACAO."""
        from app import MSG_ERRO_VALIDACAO

        mock_response = _make_error_response(422, "value is not a valid string")
        at = self._run_with_query(MagicMock(return_value=mock_response))

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any(MSG_ERRO_VALIDACAO in t for t in error_texts),
            msg=f"MSG_ERRO_VALIDACAO não encontrada em: {error_texts}",
        )
        self.assertIsNone(at.session_state.ranked_response)

    def teste_5_connect_error(self) -> None:
        """httpx.ConnectError: st.error exibe MSG_CONNECT_ERROR."""
        from app import MSG_CONNECT_ERROR

        mock_post = MagicMock(side_effect=httpx.ConnectError("connection refused"))
        at = self._run_with_query(mock_post)

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any(MSG_CONNECT_ERROR in t for t in error_texts),
            msg=f"MSG_CONNECT_ERROR não encontrada em: {error_texts}",
        )
        self.assertIsNone(at.session_state.ranked_response)

    def teste_6_timeout(self) -> None:
        """httpx.TimeoutException: st.error exibe MSG_TIMEOUT."""
        from app import MSG_TIMEOUT

        mock_post = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        at = self._run_with_query(mock_post)

        self.assertFalse(at.exception, msg=f"Exceção inesperada: {at.exception}")
        error_texts = [e.value for e in at.error]
        self.assertTrue(
            any(MSG_TIMEOUT in t for t in error_texts),
            msg=f"MSG_TIMEOUT não encontrada em: {error_texts}",
        )
        self.assertIsNone(at.session_state.ranked_response)


if __name__ == "__main__":
    unittest.main()
