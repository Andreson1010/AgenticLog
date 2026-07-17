# AgenticLog - Testes de fluxo de ingestão de documentos via UI
"""
Testes para a função _ingerir_documento de app.py.

Paths de mock:
- app.adicionar_documento_incrementalmente — JSON incremental (novo)
- app.salvar_pdf_enviado                   — salva PDF em disco
- app.reconstruir_vectordb                 — rebuild completo (PDF path)
- app.st                                   — módulo streamlit usado por app.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from agenticlog.shared.errors import RAGSecurityError  # noqa: E402
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
        """JSON feliz: adicionar_documento retorna adicionado → ingest_msg + st.rerun."""
        uploaded_file = _make_uploaded_file()
        resultado = {
            "status": "adicionado",
            "mensagem": "Arquivo doc.json adicionado com sucesso. 3 chunks inseridos.",
        }

        with (
            patch("app.adicionar_documento_incrementalmente", return_value=resultado) as mock_add,
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

            _ingerir_documento(uploaded_file)

        from agenticlog.config import DEFAULT_COLLECTION_NAME
        mock_add.assert_called_once_with("doc.json", b"{}", DEFAULT_COLLECTION_NAME)
        # Sucesso grava a mensagem em session_state (sobrevive ao st.rerun).
        self.assertEqual(mock_st.session_state.ingest_msg, ("success", resultado["mensagem"]))
        mock_st.rerun.assert_called_once()
        mock_st.success.assert_not_called()
        mock_st.error.assert_not_called()

    def teste_2_upload_erro_validacao_exibido(self) -> None:
        """RAGSecurityError de adicionar_documento exibe st.error e não chama st.rerun."""
        uploaded_file = _make_uploaded_file(name="doc.json")
        mensagem_erro = "Apenas arquivos .json são aceitos."

        with (
            patch(
                "app.adicionar_documento_incrementalmente",
                side_effect=RAGSecurityError(mensagem_erro),
            ),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with(mensagem_erro)
        mock_st.rerun.assert_not_called()

    def teste_3_upload_json_erro_add_exibido(self) -> None:
        """Exception genérica de adicionar_documento exibe st.error sem st.rerun (rollback em rag.py)."""
        uploaded_file = _make_uploaded_file()

        with (
            patch(
                "app.adicionar_documento_incrementalmente",
                side_effect=RuntimeError("embed falhou"),
            ),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("embed falhou", error_msg)
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    def teste_3b_extensao_invalida_exibe_erro_antes_de_chamar_backend(self) -> None:
        """Extensão não suportada → st.error antes de qualquer chamada ao backend."""
        uploaded_file = _make_uploaded_file(name="dados.csv", conteudo=b"col1,col2")

        with (
            patch("app.adicionar_documento_incrementalmente") as mock_add,
            patch("app.st") as mock_st,
        ):
            _ingerir_documento(uploaded_file)

        mock_add.assert_not_called()
        mock_st.error.assert_called_once()
        self.assertIn("suportado", mock_st.error.call_args[0][0].lower())

    def teste_4_upload_sem_arquivo_selecionado(self) -> None:
        """Quando uploaded_file é None, _ingerir_documento não deve ser chamado."""
        with patch("app.adicionar_documento_incrementalmente") as mock_add:
            uploaded_file = None
            if uploaded_file is not None:
                _ingerir_documento(uploaded_file)

        mock_add.assert_not_called()

    def teste_5_upload_duplicado_exibe_info(self) -> None:
        """Status duplicado → st.info chamado, st.rerun NÃO chamado."""
        uploaded_file = _make_uploaded_file()
        resultado = {
            "status": "duplicado",
            "mensagem": "Arquivo doc.json já está presente na base vetorial.",
        }

        with (
            patch("app.adicionar_documento_incrementalmente", return_value=resultado),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
            _ingerir_documento(uploaded_file)

        mock_st.info.assert_called_once_with(resultado["mensagem"])
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    def teste_6_upload_substituido_exibe_success_e_rerun(self) -> None:
        """Status substituido → ingest_msg gravado, st.rerun chamado."""
        uploaded_file = _make_uploaded_file()
        resultado = {
            "status": "substituido",
            "mensagem": "Arquivo doc.json atualizado na base vetorial. 3 chunks substituídos.",
        }

        with (
            patch("app.adicionar_documento_incrementalmente", return_value=resultado),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
            mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
            _ingerir_documento(uploaded_file)

        self.assertEqual(mock_st.session_state.ingest_msg, ("success", resultado["mensagem"]))
        mock_st.rerun.assert_called_once()
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()


class TestIngerirDocumentoPDF(unittest.TestCase):
    """Testes para o fluxo PDF de _ingerir_documento usando adicionar_pdf_incrementalmente."""

    def _spinner_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_pdf_happy_path_shows_success(self) -> None:
        """PDF feliz: adicionar_pdf_incrementalmente retorna adicionado → st.success + st.rerun."""
        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake")
        resultado = {
            "status": "adicionado",
            "mensagem": "Arquivo contrato.pdf adicionado com sucesso. 5 chunks inseridos.",
        }

        with (
            patch("app.adicionar_pdf_incrementalmente", return_value=resultado) as mock_add,
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value = self._spinner_ctx()
            _ingerir_documento(uploaded_file)

        from agenticlog.config import DEFAULT_COLLECTION_NAME
        mock_add.assert_called_once_with("contrato.pdf", b"%PDF-1.4 fake", DEFAULT_COLLECTION_NAME)
        tipo, _ = mock_st.session_state.ingest_msg
        self.assertEqual(tipo, "success")
        mock_st.rerun.assert_called_once()
        mock_st.success.assert_not_called()
        mock_st.error.assert_not_called()

    def test_pdf_duplicado_shows_info(self) -> None:
        """Status duplicado → st.info chamado, st.rerun NÃO chamado."""
        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake")
        resultado = {
            "status": "duplicado",
            "mensagem": "Arquivo contrato.pdf já está presente na base vetorial.",
        }

        with (
            patch("app.adicionar_pdf_incrementalmente", return_value=resultado),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value = self._spinner_ctx()
            _ingerir_documento(uploaded_file)

        mock_st.info.assert_called_once_with(resultado["mensagem"])
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    def test_pdf_substituido_shows_success(self) -> None:
        """Status substituido → st.success chamado, st.rerun chamado."""
        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake")
        resultado = {
            "status": "substituido",
            "mensagem": "Arquivo contrato.pdf atualizado na base vetorial. 5 chunks substituídos.",
        }

        with (
            patch("app.adicionar_pdf_incrementalmente", return_value=resultado),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value = self._spinner_ctx()
            _ingerir_documento(uploaded_file)

        tipo, _ = mock_st.session_state.ingest_msg
        self.assertEqual(tipo, "success")
        mock_st.rerun.assert_called_once()
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()

    def test_pdf_security_error_shows_error(self) -> None:
        """adicionar_pdf_incrementalmente lança RAGSecurityError → st.error, sem st.rerun."""
        uploaded_file = _make_uploaded_file("protegido.pdf", b"%PDF-1.4 fake")

        with (
            patch(
                "app.adicionar_pdf_incrementalmente",
                side_effect=RAGSecurityError("PDF protegido por senha."),
            ),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value = self._spinner_ctx()
            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("PDF protegido por senha.")
        mock_st.rerun.assert_not_called()

    def test_pdf_generic_exception_shows_error(self) -> None:
        """adicionar_pdf_incrementalmente lança Exception genérica → st.error, sem st.rerun."""
        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake")

        with (
            patch(
                "app.adicionar_pdf_incrementalmente",
                side_effect=RuntimeError("embed falhou"),
            ),
            patch("app.st") as mock_st,
        ):
            mock_st.spinner.return_value = self._spinner_ctx()
            _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("embed falhou", error_msg)
        mock_st.rerun.assert_not_called()


if __name__ == "__main__":
    unittest.main()
