"""
Acceptance tests — REC-01: Unificar metadados de chunks.

Verifies that every chunk produced by cria_vectordb() and
adicionar_documento_incrementalmente() carries the 5 unified metadata fields:
  source, file_hash, chunk_index, page, doc_type
"""

import re
import unittest
from unittest.mock import MagicMock, patch, call

from langchain_core.documents import Document


def _hex64(s: str = "a") -> str:
    """Returns a 64-char hex string (simulated SHA-256)."""
    return s * 64


class TestAC05ConfigConstants(unittest.TestCase):
    """AC-5: All 7 metadata constants are defined in agenticlog.config with correct values."""

    def teste_1_constantes_metadados_existem_e_tem_valores_corretos(self) -> None:
        from agenticlog import config

        self.assertEqual(config.METADATA_FILE_HASH, "file_hash")
        self.assertEqual(config.METADATA_CHUNK_INDEX, "chunk_index")
        self.assertEqual(config.METADATA_PAGE, "page")
        self.assertEqual(config.METADATA_DOC_TYPE, "doc_type")
        self.assertEqual(config.METADATA_DOC_TYPE_JSON, "json")
        self.assertEqual(config.METADATA_DOC_TYPE_PDF, "pdf")
        self.assertEqual(config.METADATA_PAGE_JSON_SENTINEL, 0)

    def teste_2_constantes_sao_tipos_corretos(self) -> None:
        from agenticlog import config

        self.assertIsInstance(config.METADATA_FILE_HASH, str)
        self.assertIsInstance(config.METADATA_CHUNK_INDEX, str)
        self.assertIsInstance(config.METADATA_PAGE, str)
        self.assertIsInstance(config.METADATA_DOC_TYPE, str)
        self.assertIsInstance(config.METADATA_DOC_TYPE_JSON, str)
        self.assertIsInstance(config.METADATA_DOC_TYPE_PDF, str)
        self.assertIsInstance(config.METADATA_PAGE_JSON_SENTINEL, int)


class TestAC01JsonFullRebuild(unittest.TestCase):
    """AC-1: JSON chunks from cria_vectordb() carry all 5 unified metadata fields."""

    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value=_hex64("a"))
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    @patch("agenticlog.ingestion.orchestrator._resetar_colecao", new=MagicMock())
    def teste_1_chunks_json_tem_5_campos_metadados(
        self,
        mock_valida_path: MagicMock,
        mock_valida_json: MagicMock,
        mock_loader: MagicMock,
        mock_dir: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_emb: MagicMock,
        mock_chroma: MagicMock,
        mock_hash: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod

        source_path = "data/documents/test.json"
        doc = Document(page_content="campo1: valor1", metadata={"source": source_path})
        mock_loader.return_value = [doc]
        mock_dir.glob.side_effect = lambda pat: [source_path] if pat == "*.json" else []

        chunk = Document(page_content="campo1: valor1", metadata={"source": source_path})
        splitter_instance = MagicMock()
        # split_documents called twice: first for json_docs, second for pdf_docs (empty)
        splitter_instance.split_documents.side_effect = [[chunk], []]
        mock_splitter_cls.return_value = splitter_instance

        captured_chunks = []
        def capture_from_documents(chunks, *args, **kwargs):
            captured_chunks.extend(chunks)
            return MagicMock()
        mock_chroma.from_documents.side_effect = capture_from_documents

        rag_mod.cria_vectordb()

        self.assertEqual(len(captured_chunks), 1)
        meta = captured_chunks[0].metadata

        self.assertIn("source", meta)
        self.assertIsInstance(meta["source"], str)
        self.assertTrue(len(meta["source"]) > 0)

        self.assertEqual(meta["file_hash"], _hex64("a"))
        self.assertRegex(meta["file_hash"], r"^[0-9a-f]{64}$")

        self.assertIsInstance(meta["chunk_index"], int)
        self.assertGreaterEqual(meta["chunk_index"], 0)

        self.assertEqual(meta["page"], 0)

        self.assertEqual(meta["doc_type"], "json")

    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value=_hex64("a"))
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    @patch("agenticlog.ingestion.orchestrator._resetar_colecao", new=MagicMock())
    def teste_2_chunk_index_e_sequencial_e_zero_based(
        self,
        mock_valida_path: MagicMock,
        mock_valida_json: MagicMock,
        mock_loader: MagicMock,
        mock_dir: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_emb: MagicMock,
        mock_chroma: MagicMock,
        mock_hash: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod

        source_path = "data/documents/test.json"
        chunks = [
            Document(page_content=f"chunk {i}", metadata={"source": source_path})
            for i in range(3)
        ]
        mock_loader.return_value = [
            Document(page_content="original", metadata={"source": source_path})
        ]
        mock_dir.glob.side_effect = lambda pat: [source_path] if pat == "*.json" else []

        splitter_instance = MagicMock()
        # split_documents called twice: first for json_docs, second for pdf_docs (empty)
        splitter_instance.split_documents.side_effect = [chunks, []]
        mock_splitter_cls.return_value = splitter_instance

        captured_chunks = []
        mock_chroma.from_documents.side_effect = lambda c, *a, **kw: captured_chunks.extend(c) or MagicMock()

        rag_mod.cria_vectordb()

        indexes = [c.metadata["chunk_index"] for c in captured_chunks]
        self.assertEqual(indexes, [0, 1, 2])


class TestAC02PdfFullRebuild(unittest.TestCase):
    """AC-2: PDF chunks from cria_vectordb() carry all 5 unified metadata fields."""

    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value=_hex64("b"))
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    @patch("agenticlog.ingestion.orchestrator._resetar_colecao", new=MagicMock())
    def teste_1_chunks_pdf_tem_5_campos_metadados(
        self,
        mock_valida_path: MagicMock,
        mock_valida_json: MagicMock,
        mock_loader: MagicMock,
        mock_dir: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_emb: MagicMock,
        mock_chroma: MagicMock,
        mock_hash: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod

        pdf_source = "data/documents/manual.pdf"
        mock_loader.return_value = []  # no JSON docs

        # Simulate PDF path returned by glob
        mock_pdf_path = MagicMock()
        mock_pdf_path.__str__ = lambda self: pdf_source
        mock_dir.glob.side_effect = lambda pat: [mock_pdf_path] if pat == "*.pdf" else []

        # Patch extrair_texto_pdf to return page data
        with patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf", return_value={"pagina_2": "Texto da página 2"}):
            pdf_chunk = Document(
                page_content="pagina_2: Texto da página 2",
                metadata={"source": pdf_source, "page": 2},
            )
            splitter_instance = MagicMock()
            # split_documents: first call returns [] (json_docs empty), second returns pdf_chunk
            splitter_instance.split_documents.side_effect = [[], [pdf_chunk]]
            mock_splitter_cls.return_value = splitter_instance

            captured_chunks = []
            mock_chroma.from_documents.side_effect = lambda c, *a, **kw: captured_chunks.extend(c) or MagicMock()

            rag_mod.cria_vectordb()

        self.assertEqual(len(captured_chunks), 1)
        meta = captured_chunks[0].metadata

        self.assertIn("source", meta)
        self.assertEqual(meta["file_hash"], _hex64("b"))
        self.assertRegex(meta["file_hash"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(meta["chunk_index"], int)
        self.assertGreaterEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["page"], 2)
        self.assertGreater(meta["page"], 0)
        self.assertEqual(meta["doc_type"], "pdf")

    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value=_hex64("b"))
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    @patch("agenticlog.ingestion.orchestrator._resetar_colecao", new=MagicMock())
    def teste_2_page_pdf_preservada_do_document_pai(
        self,
        mock_valida_path: MagicMock,
        mock_valida_json: MagicMock,
        mock_loader: MagicMock,
        mock_dir: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_emb: MagicMock,
        mock_chroma: MagicMock,
        mock_hash: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod

        pdf_source = "data/documents/relatorio.pdf"
        mock_loader.return_value = []

        mock_pdf_path = MagicMock()
        mock_pdf_path.__str__ = lambda self: pdf_source
        mock_dir.glob.side_effect = lambda pat: [mock_pdf_path] if pat == "*.pdf" else []

        with patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf", return_value={"pagina_5": "Conteúdo p5"}):
            pdf_chunk = Document(
                page_content="pagina_5: Conteúdo p5",
                metadata={"source": pdf_source, "page": 5},
            )
            splitter_instance = MagicMock()
            splitter_instance.split_documents.side_effect = [[], [pdf_chunk]]
            mock_splitter_cls.return_value = splitter_instance

            captured_chunks = []
            mock_chroma.from_documents.side_effect = lambda c, *a, **kw: captured_chunks.extend(c) or MagicMock()

            rag_mod.cria_vectordb()

        self.assertEqual(captured_chunks[0].metadata["page"], 5)


class TestAC03IncrementalJson(unittest.TestCase):
    """AC-3: Incremental JSON ingest via adicionar_documento_incrementalmente() adds metadata."""

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.extraction.JSONLoader")
    @patch("agenticlog.ingestion.orchestrator._valida_json_sem_chaves_proibidas")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.shutil")
    @patch("agenticlog.ingestion.orchestrator.tempfile")
    def teste_1_chunks_incrementais_tem_5_campos_metadados(
        self,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_dir: MagicMock,
        mock_valida_json: MagicMock,
        mock_json_loader: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod
        from pathlib import Path

        # Setup tempfile mock
        tmp_mock = MagicMock()
        tmp_mock.name = "/tmp/test_tmp.json"
        mock_tempfile.NamedTemporaryFile.return_value.__enter__ = lambda s: tmp_mock
        mock_tempfile.NamedTemporaryFile.return_value.__exit__ = MagicMock(return_value=False)

        # planned_path based on DIR_DOCUMENTS / safe_name
        mock_dir.__truediv__ = lambda self, other: Path(f"/fake/docs/{other}")

        # No existing doc → new insert
        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get.return_value = {"ids": [], "metadatas": []}
        mock_chroma.return_value = mock_chroma_instance

        # JSON loader returns one doc
        planned_path = Path("/fake/docs/doc1.json")
        doc = Document(page_content="campo: valor", metadata={"source": str(planned_path)})
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [doc]
        mock_json_loader.return_value = mock_loader_instance

        # Splitter returns one chunk
        chunk = Document(page_content="campo: valor", metadata={"source": str(planned_path)})
        splitter_instance = MagicMock()
        splitter_instance.split_documents.return_value = [chunk]
        mock_splitter_cls.return_value = splitter_instance

        conteudo = b'{"campo": "valor"}'
        result = rag_mod.adicionar_documento_incrementalmente("doc1.json", conteudo)

        self.assertEqual(result["status"], "adicionado")

        # Verify the chunk passed to add_documents has all 5 fields
        call_args = mock_chroma_instance.add_documents.call_args
        stored_chunks = call_args[0][0]
        self.assertEqual(len(stored_chunks), 1)
        meta = stored_chunks[0].metadata

        self.assertIn("file_hash", meta)
        self.assertRegex(meta["file_hash"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(meta["chunk_index"], int)
        self.assertGreaterEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["page"], 0)
        self.assertEqual(meta["doc_type"], "json")


class TestAC04DedupUsesFileHash(unittest.TestCase):
    """AC-4: Deduplication uses file_hash field (not content_hash)."""

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_1_duplicata_detectada_via_file_hash(
        self,
        mock_dir: MagicMock,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod
        from pathlib import Path
        import hashlib

        conteudo = b'{"campo": "valor"}'
        expected_hash = hashlib.sha256(conteudo).hexdigest()

        mock_dir.__truediv__ = lambda self, other: Path(f"/fake/docs/{other}")
        mock_dir.glob.return_value = []

        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get.return_value = {
            "ids": ["existing-id"],
            "metadatas": [{"file_hash": expected_hash, "source": "/fake/docs/doc1.json"}],
        }
        mock_chroma.return_value = mock_chroma_instance

        result = rag_mod.adicionar_documento_incrementalmente("doc1.json", conteudo)

        self.assertEqual(result["status"], "duplicado")

    @patch("agenticlog.agent.invalidar_vector_db")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.extraction.JSONLoader")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_2_hash_diferente_upsert_via_file_hash(
        self,
        mock_dir: MagicMock,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_loader_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        import agenticlog.rag as rag_mod
        from pathlib import Path
        import tempfile as tmpmod

        conteudo_novo = b'{"campo": "valor_novo"}'

        with tmpmod.TemporaryDirectory() as real_tmp:
            real_tmp_path = Path(real_tmp)
            mock_dir.__truediv__ = lambda self, other: real_tmp_path / other
            mock_dir.glob.return_value = []

            mock_chroma_instance = MagicMock()
            mock_chroma_instance.get.return_value = {
                "ids": ["existing-id"],
                "metadatas": [{"file_hash": "0" * 64, "source": str(real_tmp_path / "doc1.json")}],
            }
            mock_chroma.return_value = mock_chroma_instance

            mock_loader_cls.return_value.load.return_value = [
                Document(page_content="valor novo", metadata={})
            ]
            mock_splitter_cls.return_value.split_documents.return_value = [
                Document(page_content="chunk novo", metadata={})
            ]

            result = rag_mod.adicionar_documento_incrementalmente("doc1.json", conteudo_novo)

        self.assertEqual(result["status"], "substituido")
        mock_chroma_instance.delete.assert_called_once_with(ids=["existing-id"])

    @patch("agenticlog.agent.invalidar_vector_db")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.extraction.JSONLoader")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_3_campo_content_hash_nao_e_usado_para_dedup(
        self,
        mock_dir: MagicMock,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_loader_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """Metadata key must be 'file_hash', not the old 'content_hash' — mismatch causes upsert."""
        import agenticlog.rag as rag_mod
        from pathlib import Path
        import hashlib
        import tempfile as tmpmod

        conteudo = b'{"campo": "valor"}'
        expected_hash = hashlib.sha256(conteudo).hexdigest()

        with tmpmod.TemporaryDirectory() as real_tmp:
            real_tmp_path = Path(real_tmp)
            mock_dir.__truediv__ = lambda self, other: real_tmp_path / other
            mock_dir.glob.return_value = []

            # Return metadata with OLD key 'content_hash' — should NOT match file_hash
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.get.return_value = {
                "ids": ["existing-id"],
                "metadatas": [{"content_hash": expected_hash}],  # old key, file_hash missing
            }
            mock_chroma.return_value = mock_chroma_instance

            mock_loader_cls.return_value.load.return_value = [
                Document(page_content="valor", metadata={})
            ]
            mock_splitter_cls.return_value.split_documents.return_value = [
                Document(page_content="chunk", metadata={})
            ]

            # existing_hash = None (file_hash not found) → not equal to expected_hash → upsert
            result = rag_mod.adicionar_documento_incrementalmente("doc1.json", conteudo)

        # old content_hash key is ignored → treated as stale metadata → upsert, not dedup
        self.assertEqual(result["status"], "substituido")


if __name__ == "__main__":
    unittest.main()
