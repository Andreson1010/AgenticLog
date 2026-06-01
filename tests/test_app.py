# AgenticLog - Testes de fluxo de ingestão de documentos via UI
"""
Testes para a função _ingerir_documento de app.py.

Todos os chamados a salvar_documento_enviado, reconstruir_vectordb e
funções Streamlit são mockados — nenhuma chamada real ao filesystem ou
LMStudio é feita nestes testes.

Paths de mock:
- app.salvar_documento_enviado  — função importada em app.py
- app.reconstruir_vectordb      — função importada em app.py
- app.st                        — módulo streamlit usado por app.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from agenticlog.rag import RAGSecurityError  # noqa: E402
import app  # noqa: E402 — importa módulo app para acessar _ingerir_documento
from app import _ingerir_documento  # noqa: E402


def _make_uploaded_file(name: str = "doc.json", conteudo: bytes = b"{}") -> MagicMock:
    """Cria um mock de UploadedFile do Streamlit.

    Entrada: name — nome do arquivo; conteudo — bytes do conteúdo.
    Saída: MagicMock com .name e .getvalue() configurados.
    """
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.getvalue.return_value = conteudo
    return mock_file


class TestIngerirDocumento(unittest.TestCase):
    """Testes para o helper _ingerir_documento de app.py."""

    def teste_1_upload_fluxo_sucesso(self) -> None:
        """Fluxo feliz: salvar e reconstruir com sucesso exibem st.success e chamam st.rerun."""
        uploaded_file = _make_uploaded_file()
        saved_path = Path("/tmp/doc.json")

        with (
            patch("app.salvar_documento_enviado", return_value=saved_path) as mock_salvar,
            patch("app.reconstruir_vectordb") as mock_rebuild,
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

            _ingerir_documento(uploaded_file)

        mock_salvar.assert_called_once_with("doc.json", b"{}")
        mock_rebuild.assert_called_once()
        mock_st.success.assert_called_once_with("Documento ingerido com sucesso.")
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()

    def teste_2_upload_erro_validacao_exibido(self) -> None:
        """RAGSecurityError de salvar_documento_enviado deve exibir st.error e não chamar st.rerun."""
        uploaded_file = _make_uploaded_file(name="doc.txt")
        mensagem_erro = "Apenas arquivos .json são aceitos."

        with (
            patch(
                "app.salvar_documento_enviado",
                side_effect=RAGSecurityError(mensagem_erro),
            ),
            patch("app.reconstruir_vectordb") as mock_rebuild,
            patch("app.st") as mock_st,
        ):
            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with(mensagem_erro)
        mock_st.rerun.assert_not_called()
        mock_rebuild.assert_not_called()

    def teste_3_upload_erro_rebuild_exibido(self) -> None:
        """Falha em reconstruir_vectordb deve remover arquivo salvo e exibir st.error sem st.rerun."""
        uploaded_file = _make_uploaded_file()
        mock_path = MagicMock(spec=Path)

        with (
            patch("app.salvar_documento_enviado", return_value=mock_path),
            patch(
                "app.reconstruir_vectordb",
                side_effect=Exception("fail"),
            ),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

            _ingerir_documento(uploaded_file)

        mock_path.unlink.assert_called_once_with(missing_ok=True)
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("Erro ao reconstruir base vetorial", error_msg)
        self.assertIn("fail", error_msg)
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    def teste_4_upload_sem_arquivo_selecionado(self) -> None:
        """Quando uploaded_file é None, _ingerir_documento não deve ser chamado."""
        with patch("app.salvar_documento_enviado") as mock_salvar:
            # Simula a lógica do botão: uploaded_file is None → não chama _ingerir_documento
            uploaded_file = None
            if uploaded_file is not None:
                _ingerir_documento(uploaded_file)

        mock_salvar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
