# AgenticLog - Testes unitários para ingestion/cleaning.py
"""Testes do estágio de limpeza da ingestão (ADR-011 / ADR-018 Fase 3a, RAGING-13 T14)."""

import sys
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

from langchain_core.documents import Document as LCDocument  # noqa: E402

from agenticlog.ingestion.cleaning import filtrar_documentos_vazios  # noqa: E402


class TestFiltrarDocumentosVazios(unittest.TestCase):
    """Testes para filtrar_documentos_vazios."""

    def teste_1_descarta_string_vazia(self) -> None:
        """Document com page_content '' é descartado."""
        docs = [LCDocument(page_content="", metadata={})]
        self.assertEqual(filtrar_documentos_vazios(docs), [])

    def teste_2_descarta_apenas_whitespace(self) -> None:
        """Document com page_content só de whitespace é descartado (.strip())."""
        docs = [LCDocument(page_content="   \n\t ", metadata={})]
        self.assertEqual(filtrar_documentos_vazios(docs), [])

    def teste_3_preserva_conteudo_com_prefixo_de_campo(self) -> None:
        """'CAMPO_VAZIO: ' tem .strip() não-vazio → preservado."""
        doc = LCDocument(page_content="CAMPO_VAZIO: ", metadata={})
        self.assertEqual(filtrar_documentos_vazios([doc]), [doc])

    def teste_4_preserva_conteudo_normal(self) -> None:
        """Document com conteúdo real é preservado."""
        doc = LCDocument(page_content="texto de logística", metadata={})
        self.assertEqual(filtrar_documentos_vazios([doc]), [doc])

    def teste_5_mistura_preserva_apenas_nao_vazios(self) -> None:
        """Filtra os vazios e mantém a ordem dos não-vazios."""
        d_ok1 = LCDocument(page_content="a", metadata={})
        d_vazio = LCDocument(page_content="  ", metadata={})
        d_ok2 = LCDocument(page_content="b", metadata={})
        resultado = filtrar_documentos_vazios([d_ok1, d_vazio, d_ok2])
        self.assertEqual(resultado, [d_ok1, d_ok2])

    def teste_6_retorna_lista_nova_sem_mutar_entrada(self) -> None:
        """Imutabilidade: entrada não é mutada e a saída é um objeto de lista distinto."""
        entrada = [LCDocument(page_content="x", metadata={})]
        resultado = filtrar_documentos_vazios(entrada)
        self.assertIsNot(resultado, entrada)
        self.assertEqual(len(entrada), 1)
