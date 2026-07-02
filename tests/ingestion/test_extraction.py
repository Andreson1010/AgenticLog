# AgenticLog - Testes unitários para ingestion/extraction.py
"""Testes do estágio de extração da ingestão (ADR-018 Fase 3a).

Movidos de tests/test_rag.py; `@patch("agenticlog.rag.fitz.open")` repontado para
`agenticlog.ingestion.extraction.fitz.open` (fitz é lido no corpo movido).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.config as config  # noqa: E402
import agenticlog.rag as rag  # noqa: E402  # shim ainda re-exporta os símbolos (identidade)
from agenticlog.ingestion.extraction import carregar_json, extrair_texto_pdf  # noqa: E402


class TestExtrairTextoPdf(unittest.TestCase):
    """Testes para extrair_texto_pdf."""

    @patch("agenticlog.ingestion.extraction.fitz.open")
    def teste_1_extrair_pdf_valido_retorna_dict(self, mock_fitz_open):
        """PDF com texto retorna dict {"PÁGINA_1": texto}."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "texto do contrato"
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz_open.return_value = mock_doc

        resultado = extrair_texto_pdf(Path("qualquer.pdf"))

        self.assertEqual(resultado, {"PÁGINA_1": "texto do contrato"})

    @patch("agenticlog.ingestion.extraction.fitz.open")
    def teste_2_extrair_pdf_com_senha_lanca_erro(self, mock_fitz_open):
        """PDF com senha lança RAGSecurityError."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.ingestion.extraction.fitz.open")
    def teste_3_extrair_pdf_somente_imagem_lanca_erro(self, mock_fitz_open):
        """PDF somente-imagem (todas as páginas retornam texto vazio) lança RAGSecurityError."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   \n\t  "
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
        mock_fitz_open.return_value = mock_doc

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("somente imagem", str(ctx.exception))

    @patch("agenticlog.ingestion.extraction.fitz.open")
    def teste_4_extrair_pdf_mix_texto_imagem_filtra_pagina_vazia(self, mock_fitz_open):
        """PDF com mix de páginas texto e imagem: só a página com texto aparece no dict."""
        mock_page_texto = MagicMock()
        mock_page_texto.get_text.return_value = "conteúdo real"
        mock_page_imagem = MagicMock()
        mock_page_imagem.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page_texto, mock_page_imagem]))
        mock_fitz_open.return_value = mock_doc

        resultado = extrair_texto_pdf(Path("qualquer.pdf"))

        self.assertEqual(resultado, {"PÁGINA_1": "conteúdo real"})
        self.assertNotIn("PÁGINA_2", resultado)

    @patch("agenticlog.ingestion.extraction.fitz.open")
    def teste_5_extrair_exception_generica_lanca_erro(self, mock_fitz_open):
        """fitz.open() lançando Exception genérica é convertida em RAGSecurityError."""
        mock_fitz_open.side_effect = RuntimeError("unexpected fitz error")

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("corrompido", str(ctx.exception))

    @patch("agenticlog.ingestion.extraction.fitz.open")
    def teste_6_extrair_pdf_multipagina_retorna_dict_ordenado(self, mock_fitz_open):
        """PDF com 3 páginas de texto retorna dict com 3 chaves PÁGINA_1..3 na ordem."""
        mock_pages = []
        for i in range(3):
            p = MagicMock()
            p.get_text.return_value = f"texto da pagina {i + 1}"
            mock_pages.append(p)
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))
        mock_fitz_open.return_value = mock_doc

        resultado = extrair_texto_pdf(Path("qualquer.pdf"))

        self.assertEqual(
            resultado,
            {
                "PÁGINA_1": "texto da pagina 1",
                "PÁGINA_2": "texto da pagina 2",
                "PÁGINA_3": "texto da pagina 3",
            },
        )
        self.assertEqual(list(resultado.keys()), ["PÁGINA_1", "PÁGINA_2", "PÁGINA_3"])


class TestCarregarJson(unittest.TestCase):
    """Testes unitários para carregar_json (RAGING-03 / T16)."""

    @patch("agenticlog.ingestion.extraction.JSONLoader")
    def teste_1_usa_jq_schema_compartilhado(self, mock_loader_cls):
        """carregar_json instancia JSONLoader com jq_schema=config.JQ_SCHEMA_CAMPOS_JSON."""
        caminho = Path("data/documents/doc.json")
        carregar_json(caminho)
        mock_loader_cls.assert_called_once_with(
            str(caminho), jq_schema=config.JQ_SCHEMA_CAMPOS_JSON
        )

    @patch("agenticlog.ingestion.extraction.JSONLoader")
    def teste_2_retorna_o_load_do_loader(self, mock_loader_cls):
        """carregar_json retorna exatamente o resultado de loader.load()."""
        docs_esperados = ["doc_a", "doc_b"]
        mock_loader_cls.return_value.load.return_value = docs_esperados
        resultado = carregar_json(Path("data/documents/outro.json"))
        self.assertIs(resultado, docs_esperados)
        mock_loader_cls.return_value.load.assert_called_once_with()

    @patch("agenticlog.ingestion.extraction.JSONLoader")
    def teste_3_parametrizado_por_path(self, mock_loader_cls):
        """O caminho recebido é repassado (como str) ao JSONLoader."""
        caminho = Path("/tmp/qualquer/arquivo.json")
        carregar_json(caminho)
        args, _ = mock_loader_cls.call_args
        self.assertEqual(args[0], str(caminho))
