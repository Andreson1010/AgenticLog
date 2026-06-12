"""Testes para scripts/pdf_to_json.py (wrapper fino sobre agenticlog.rag.extrair_texto_pdf)."""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "scripts"))

import pdf_to_json  # noqa: E402

from agenticlog.rag import RAGSecurityError  # noqa: E402


class TestPdfParaDict:
    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_1_pdf_para_dict_delega_para_extrair_texto_pdf(self, mock_extrair):
        mock_extrair.return_value = {"PÁGINA_1": "conteudo"}
        resultado = pdf_to_json.pdf_para_dict(Path("qualquer.pdf"))
        assert resultado == {"PÁGINA_1": "conteudo"}
        mock_extrair.assert_called_once_with(Path("qualquer.pdf"))

    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_2_pdf_sem_texto_levanta_value_error(self, mock_extrair):
        mock_extrair.side_effect = RAGSecurityError(
            "PDF não contém texto extraível (somente imagem)."
        )
        with pytest.raises(ValueError, match="somente imagem"):
            pdf_to_json.pdf_para_dict(Path("vazio.pdf"))


class TestConverter:
    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_3_converter_escreve_json_ensure_ascii_false(self, mock_extrair, tmp_path):
        mock_extrair.return_value = {"PÁGINA_1": "Texto com acentuação: ção, ã, é"}
        destino = pdf_to_json.converter(Path("doc.pdf"), tmp_path)
        conteudo = destino.read_text(encoding="utf-8")
        assert "PÁGINA_1" in conteudo
        assert "ção" in conteudo  # ensure_ascii=False preserva acentuação
        assert "\\u" not in conteudo  # nao deve haver escapes unicode


class TestMain:
    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_4_main_imprime_ok_em_console_cp1252_sem_unicodeencodeerror(
        self, mock_extrair, tmp_path, monkeypatch
    ):
        """main() nao deve lancar UnicodeEncodeError/SystemExit quando stdout e cp1252 (console Windows)."""
        mock_extrair.return_value = {"PÁGINA_1": "conteudo"}

        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        output_dir = tmp_path / "out"

        stdout_cp1252 = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", write_through=True)
        monkeypatch.setattr(sys, "stdout", stdout_cp1252)
        monkeypatch.setattr(
            sys, "argv", ["pdf_to_json.py", str(pdf_file), "--output", str(output_dir)]
        )

        pdf_to_json.main()

        stdout_cp1252.seek(0)
        saida = stdout_cp1252.buffer.getvalue().decode("cp1252")
        assert "OK" in saida
        assert (output_dir / "doc.json").exists()
