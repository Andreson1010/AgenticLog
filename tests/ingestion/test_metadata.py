# AgenticLog - Testes unitários para ingestion/metadata.py
"""Testes do estágio de metadados da ingestão (ADR-018 Fase 3a).

Movidos de tests/test_rag.py. Sem alvos de @patch a repontar (funções puras/param.
por argumento). As referências `rag.*` seguem válidas via shim (identidade preservada).
"""

import hashlib
import sys
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

from langchain_core.documents import Document as LCDocument  # noqa: E402

import agenticlog.ingestion.metadata as rag  # noqa: E402
from agenticlog.ingestion.metadata import _computar_hash_conteudo  # noqa: E402


class TestComputarHash(unittest.TestCase):
    """Testes para _computar_hash_conteudo."""

    def teste_1_hash_deterministico(self) -> None:
        """Mesmo input deve gerar mesmo hash de 64 caracteres."""
        h1 = _computar_hash_conteudo(b"hello")
        h2 = _computar_hash_conteudo(b"hello")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def teste_2_hash_diferente_para_conteudo_diferente(self) -> None:
        """Inputs distintos devem gerar hashes diferentes."""
        h1 = _computar_hash_conteudo(b"hello")
        h2 = _computar_hash_conteudo(b"world")
        self.assertNotEqual(h1, h2)


class TestMetadadosUnificados(unittest.TestCase):
    """Testes unitários para _enriquecer_metadados_chunks e campos unificados (REC-01)."""

    def teste_1_enriquece_todos_os_campos(self) -> None:
        """Todos os 5 campos presentes após enriquecimento."""
        chunks = [LCDocument(page_content="a", metadata={})]
        rag._enriquecer_metadados_chunks(chunks, "a" * 64, "json", 0)
        meta = chunks[0].metadata
        self.assertIn("file_hash", meta)
        self.assertIn("chunk_index", meta)
        self.assertIn("doc_type", meta)
        self.assertIn("page", meta)

    def teste_2_chunk_index_sequencial_json(self) -> None:
        """Dois chunks JSON recebem chunk_index [0, 1]."""
        chunks = [
            LCDocument(page_content="c0", metadata={}),
            LCDocument(page_content="c1", metadata={}),
        ]
        rag._enriquecer_metadados_chunks(chunks, "a" * 64, "json", 0)
        self.assertEqual([c.metadata["chunk_index"] for c in chunks], [0, 1])

    def teste_3_chunk_index_single_chunk(self) -> None:
        """Chunk único tem chunk_index == 0."""
        chunk = LCDocument(page_content="x", metadata={})
        rag._enriquecer_metadados_chunks([chunk], "a" * 64, "json", 0)
        self.assertEqual(chunk.metadata["chunk_index"], 0)

    def teste_4_page_sentinel_json(self) -> None:
        """Chunks JSON recebem page=0."""
        chunk = LCDocument(page_content="x", metadata={})
        rag._enriquecer_metadados_chunks([chunk], "a" * 64, "json", 0)
        self.assertEqual(chunk.metadata["page"], 0)

    def teste_5_page_nao_sobrescrito_quando_none(self) -> None:
        """page=None não sobrescreve page já presente (PDF: herdado do Document pai)."""
        chunk = LCDocument(page_content="x", metadata={"page": 3})
        rag._enriquecer_metadados_chunks([chunk], "a" * 64, "pdf")
        self.assertEqual(chunk.metadata["page"], 3)

    def teste_6_doc_type_json(self) -> None:
        """Chunks JSON recebem doc_type='json'."""
        chunk = LCDocument(page_content="x", metadata={})
        rag._enriquecer_metadados_chunks([chunk], "a" * 64, "json", 0)
        self.assertEqual(chunk.metadata["doc_type"], "json")

    def teste_7_doc_type_pdf(self) -> None:
        """Chunks PDF recebem doc_type='pdf'."""
        chunk = LCDocument(page_content="x", metadata={"page": 1})
        rag._enriquecer_metadados_chunks([chunk], "a" * 64, "pdf")
        self.assertEqual(chunk.metadata["doc_type"], "pdf")

    def teste_8_file_hash_sha256_correto(self) -> None:
        """_computar_hash_conteudo retorna SHA-256 de 64 chars identico ao hashlib."""
        conteudo = b"teste de logistica"
        esperado = hashlib.sha256(conteudo).hexdigest()
        resultado = rag._computar_hash_conteudo(conteudo)
        self.assertEqual(resultado, esperado)
        self.assertEqual(len(resultado), 64)

    def teste_9_zero_chunks_sem_erro(self) -> None:
        """Lista vazia não levanta exceção."""
        rag._enriquecer_metadados_chunks([], "a" * 64, "json", 0)

    def teste_10_dois_grupos_chunk_index_independente(self) -> None:
        """Dois grupos separados têm chunk_index independentes partindo de 0."""
        grupo1 = [LCDocument(page_content=f"a{i}", metadata={}) for i in range(2)]
        grupo2 = [LCDocument(page_content=f"b{i}", metadata={}) for i in range(3)]
        rag._enriquecer_metadados_chunks(grupo1, "a" * 64, "json", 0)
        rag._enriquecer_metadados_chunks(grupo2, "b" * 64, "json", 0)
        self.assertEqual([c.metadata["chunk_index"] for c in grupo1], [0, 1])
        self.assertEqual([c.metadata["chunk_index"] for c in grupo2], [0, 1, 2])
