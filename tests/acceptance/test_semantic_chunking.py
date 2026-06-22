# Testes de aceitação — semantic-chunking (ADR-013)
"""
AC-1: cria_vectordb produz chunks com doc_type e file_hash.
AC-2: adicionar_documento_incrementalmente usa SemanticChunker com config de ADR-013.
AC-3: adicionar_pdf_incrementalmente usa SemanticChunker com config de ADR-013.
AC-4: CHUNK_SIZE e CHUNK_OVERLAP não existem em config.py nem são importados em rag.py.
AC-5: SemanticChunker inicializado com SEMANTIC_BREAKPOINT_TYPE/THRESHOLD em todas as funções.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from langchain_core.documents import Document as LCDocument

from agenticlog import config
from agenticlog.rag import (
    adicionar_documento_incrementalmente,
    adicionar_pdf_incrementalmente,
    cria_vectordb,
)


def _mock_vdb(ids=None, metadatas=None):
    m = MagicMock()
    m.get.return_value = {"ids": ids or [], "metadatas": metadatas or []}
    return m


class TestAC1CriaVectordbUsaSemanticChunker(unittest.TestCase):
    """AC-1 — cria_vectordb produz chunks com metadados unificados."""

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    @patch("agenticlog.rag._resetar_colecao", new=MagicMock())
    def teste_1_chunks_tem_doc_type_e_file_hash(
        self, _vp, _vj, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma, _hash
    ):
        """AC-1: chunks passados ao Chroma possuem doc_type e file_hash."""
        doc = LCDocument(
            page_content="CAMPO: valor de teste",
            metadata={"source": "/fake/doc.json"},
        )
        mock_loader.return_value.load.return_value = [doc]
        mock_dir.glob.return_value = []

        mock_splitter_inst = MagicMock()
        mock_splitter_inst.split_documents.side_effect = lambda docs: list(docs)
        mock_splitter.return_value = mock_splitter_inst

        cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        chunks = call_args[0][0]
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIn("doc_type", chunk.metadata)
            self.assertIn("file_hash", chunk.metadata)

    @patch("agenticlog.rag._hash_arquivo", return_value="b" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    @patch("agenticlog.rag._resetar_colecao", new=MagicMock())
    def teste_2_semantic_chunker_inicializado_com_embedding_e_config(
        self, _vp, _vj, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma, _hash
    ):
        """AC-5 (cria_vectordb): SemanticChunker recebe embedding model e config de ADR-013."""
        doc = LCDocument(page_content="CAMPO: x", metadata={"source": "/fake/doc.json"})
        mock_loader.return_value.load.return_value = [doc]
        mock_dir.glob.return_value = []
        mock_splitter.return_value.split_documents.side_effect = lambda docs: list(docs)

        cria_vectordb()

        mock_splitter.assert_called_once_with(
            embeddings=mock_emb.return_value,
            breakpoint_threshold_type=config.SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_THRESHOLD,
        )


class TestAC2JsonIncremental(unittest.TestCase):
    """AC-2 / AC-5 — adicionar_documento_incrementalmente usa SemanticChunker."""

    def _chunk(self, content="campo: valor", source="/fake/doc.json"):
        return LCDocument(page_content=content, metadata={"source": source})

    def teste_3_semantic_chunker_inicializado_com_config_correto(self):
        """AC-5 (JSON): SemanticChunker recebe config de ADR-013."""
        conteudo = b'{"campo": "valor"}'
        mock_vdb = _mock_vdb()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vdb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [self._chunk()]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                adicionar_documento_incrementalmente("doc.json", conteudo)

        mock_splitter_cls.assert_called_once_with(
            embeddings=ANY,
            breakpoint_threshold_type=config.SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_THRESHOLD,
        )

    def teste_4_chunks_recebem_metadados_unificados(self):
        """AC-2: chunks inseridos possuem file_hash e doc_type após ingestão."""
        conteudo = b'{"campo": "valor"}'
        mock_vdb = _mock_vdb()
        chunk = self._chunk()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vdb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [chunk]
                mock_splitter_cls.return_value.split_documents.return_value = [chunk]

                adicionar_documento_incrementalmente("doc.json", conteudo)

        inserted_chunks = mock_vdb.add_documents.call_args[0][0]
        self.assertTrue(len(inserted_chunks) > 0)
        for c in inserted_chunks:
            self.assertIn("file_hash", c.metadata)
            self.assertIn("doc_type", c.metadata)
            self.assertEqual(c.metadata["doc_type"], "json")


class TestAC3PdfIncremental(unittest.TestCase):
    """AC-3 / AC-5 — adicionar_pdf_incrementalmente usa SemanticChunker."""

    _PDF_MAGIC = b"%PDF-1.4 fake content"

    def _chunk(self, content="PÁGINA_1: texto da página"):
        return LCDocument(page_content=content, metadata={"source": "/fake/doc.pdf", "page": 1})

    def teste_5_semantic_chunker_inicializado_com_config_correto(self):
        """AC-5 (PDF): SemanticChunker recebe config de ADR-013."""
        mock_vdb = _mock_vdb()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.extrair_texto_pdf", return_value={"PÁGINA_1": "texto"}),
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vdb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                adicionar_pdf_incrementalmente("doc.pdf", self._PDF_MAGIC)

        mock_splitter_cls.assert_called_once_with(
            embeddings=ANY,
            breakpoint_threshold_type=config.SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_THRESHOLD,
        )

    def teste_6_chunks_pdf_recebem_doc_type_pdf(self):
        """AC-3: chunks de PDF possuem doc_type='pdf' após ingestão."""
        mock_vdb = _mock_vdb()
        chunk = self._chunk()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.extrair_texto_pdf", return_value={"PÁGINA_1": "texto"}),
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vdb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_splitter_cls.return_value.split_documents.return_value = [chunk]

                adicionar_pdf_incrementalmente("doc.pdf", self._PDF_MAGIC)

        inserted_chunks = mock_vdb.add_documents.call_args[0][0]
        self.assertTrue(len(inserted_chunks) > 0)
        for c in inserted_chunks:
            self.assertEqual(c.metadata.get("doc_type"), "pdf")


class TestAC4NaoExisteChunkSize(unittest.TestCase):
    """AC-4 — CHUNK_SIZE e CHUNK_OVERLAP não existem em config."""

    def teste_7_chunk_size_removido_de_config(self):
        """AC-4: config.CHUNK_SIZE não existe após a feature."""
        self.assertFalse(hasattr(config, "CHUNK_SIZE"))

    def teste_8_chunk_overlap_removido_de_config(self):
        """AC-4: config.CHUNK_OVERLAP não existe após a feature."""
        self.assertFalse(hasattr(config, "CHUNK_OVERLAP"))

    def teste_9_semantic_breakpoint_type_existe(self):
        """AC-5: SEMANTIC_BREAKPOINT_TYPE existe em config com valor 'percentile'."""
        self.assertTrue(hasattr(config, "SEMANTIC_BREAKPOINT_TYPE"))
        self.assertEqual(config.SEMANTIC_BREAKPOINT_TYPE, "percentile")

    def teste_10_semantic_breakpoint_threshold_existe(self):
        """AC-5: SEMANTIC_BREAKPOINT_THRESHOLD existe em config com valor 95.0."""
        self.assertTrue(hasattr(config, "SEMANTIC_BREAKPOINT_THRESHOLD"))
        self.assertEqual(config.SEMANTIC_BREAKPOINT_THRESHOLD, 95.0)
