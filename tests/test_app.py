# AgenticLog - Testes de fluxo de ingestão de documentos via UI
"""
Testes para a função _ingerir_documento de app.py.

Todos os chamados a adicionar_documento_incrementalmente e
funções Streamlit são mockados — nenhuma chamada real ao filesystem ou
LMStudio é feita nestes testes.

Paths de mock:
- app.adicionar_documento_incrementalmente — função importada em app.py
- app.st                                  — módulo streamlit usado por app.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _mock_spinner(mock_st: MagicMock) -> None:
    mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)


class TestIngerirDocumento(unittest.TestCase):
    """Testes para o helper _ingerir_documento de app.py."""

    def teste_1_upload_fluxo_sucesso(self) -> None:
        """Fluxo feliz: adicionado com sucesso exibe st.success e chama st.rerun."""
        uploaded_file = _make_uploaded_file()
        resultado = {"status": "adicionado", "mensagem": "Arquivo doc.json adicionado com sucesso. 3 chunks inseridos."}

        with (
            patch("app.adicionar_documento_incrementalmente", return_value=resultado) as mock_add,
            patch("app.st") as mock_st,
        ):
            _mock_spinner(mock_st)
            _ingerir_documento(uploaded_file)

        mock_add.assert_called_once_with("doc.json", b"{}")
        mock_st.success.assert_called_once_with(resultado["mensagem"])
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()

    def teste_2_upload_erro_validacao_exibido(self) -> None:
        """RAGSecurityError exibe st.error e não chama st.rerun."""
        uploaded_file = _make_uploaded_file(name="doc.txt")
        mensagem_erro = "Apenas arquivos .json são aceitos."

        with (
            patch(
                "app.adicionar_documento_incrementalmente",
                side_effect=RAGSecurityError(mensagem_erro),
            ),
            patch("app.st") as mock_st,
        ):
            _mock_spinner(mock_st)
            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with(mensagem_erro)
        mock_st.rerun.assert_not_called()

    def teste_3_upload_erro_generico_exibido(self) -> None:
        """Exception genérica exibe st.error com detalhe e não chama st.rerun."""
        uploaded_file = _make_uploaded_file()

        with (
            patch(
                "app.adicionar_documento_incrementalmente",
                side_effect=Exception("falha interna"),
            ),
            patch("app.st") as mock_st,
        ):
            _mock_spinner(mock_st)
            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("Erro ao adicionar documento", error_msg)
        self.assertIn("falha interna", error_msg)
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    def teste_4_upload_sem_arquivo_selecionado(self) -> None:
        """Quando uploaded_file é None, _ingerir_documento não deve ser chamado."""
        with patch("app.adicionar_documento_incrementalmente") as mock_add:
            uploaded_file = None
            if uploaded_file is not None:
                _ingerir_documento(uploaded_file)

        mock_add.assert_not_called()

    def teste_5_upload_duplicado_exibe_info(self) -> None:
        """Status 'duplicado' exibe st.info e não chama st.rerun."""
        uploaded_file = _make_uploaded_file()
        resultado = {"status": "duplicado", "mensagem": "Arquivo doc.json já está presente na base vetorial."}

        with (
            patch("app.adicionar_documento_incrementalmente", return_value=resultado),
            patch("app.st") as mock_st,
        ):
            _mock_spinner(mock_st)
            _ingerir_documento(uploaded_file)

        mock_st.info.assert_called_once_with(resultado["mensagem"])
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()

    def teste_6_upload_hash_diferente_exibe_warning(self) -> None:
        """Status 'hash_diferente' exibe st.warning e não chama st.rerun."""
        uploaded_file = _make_uploaded_file()
        resultado = {"status": "hash_diferente", "mensagem": "Arquivo doc.json já existe com conteúdo diferente."}

        with (
            patch("app.adicionar_documento_incrementalmente", return_value=resultado),
            patch("app.st") as mock_st,
        ):
            _mock_spinner(mock_st)
            _ingerir_documento(uploaded_file)

        mock_st.warning.assert_called_once_with(resultado["mensagem"])
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()
        mock_st.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
