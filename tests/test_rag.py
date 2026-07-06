# AgenticLog - Testes unitários para rag.py
"""
Testes para o pipeline RAG em agenticlog.rag.
Cobre validações de segurança, path traversal e criação do banco vetorial.
"""

import importlib
import io
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY, call

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
import pytest

import hashlib

from langchain_core.documents import Document as LCDocument

from agenticlog.rag import (
    RAGSecurityError,
    _valida_path_documentos,
    _valida_json_sem_chaves_proibidas,
    _valida_arquivos_json,
    cria_vectordb,
    ingerir_incrementalmente,
    _executar_main,
    _sanitizar_nome_arquivo,
    _sanitizar_nome_colecao,
    _computar_hash_conteudo,
    _enriquecer_metadados_chunks,
    adicionar_documento_incrementalmente,
    adicionar_pdf_incrementalmente,
    salvar_documento_enviado,
    salvar_pdf_enviado,
    extrair_texto_pdf,
    reconstruir_vectordb,
)
from agenticlog.config import MAX_JSON_FILES, MAX_JSON_FILE_SIZE_MB
import agenticlog.rag as rag
import agenticlog.config as config
import agenticlog.ingestion.orchestrator as orchestrator
import agenticlog.ingestion.store as store


class TestRAGSecurityError(unittest.TestCase):
    """Testes para a exceção RAGSecurityError."""

    def test_raise_rag_security_error(self):
        """RAGSecurityError pode ser levantada e capturada."""
        with self.assertRaises(RAGSecurityError) as ctx:
            raise RAGSecurityError("Mensagem de teste")
        self.assertIn("Mensagem de teste", str(ctx.exception))


def _glob_side_effect(json_paths=(), pdf_paths=()):
    """Fábrica de side_effect para docs_dir.glob: separa *.json de *.pdf.

    cria_vectordb (Fase 3b) itera `sorted(docs_dir.glob("*.json"))` + `docs_dir.glob("*.pdf")`
    em vez de DirectoryLoader — o glob agora é chamado por-padrão.
    """
    def _g(pattern):
        if pattern == "*.json":
            return list(json_paths)
        if pattern == "*.pdf":
            return list(pdf_paths)
        return []
    return _g


class TestCriaVectordb(unittest.TestCase):
    """Testes para cria_vectordb (com mocks para evitar dependências pesadas)."""

    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_sem_documentos_retorna_cedo(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir, mock_splitter, mock_emb, mock_chroma
    ):
        """Quando não há documentos, cria_vectordb retorna sem criar Chroma."""
        mock_dir.glob.side_effect = _glob_side_effect()  # nenhum json/pdf

        cria_vectordb()

        mock_valida_path.assert_called_once()
        mock_valida_json.assert_called_once()
        mock_carregar.assert_not_called()  # sem arquivos json → carregar_json não é chamado
        mock_chroma.from_documents.assert_not_called()

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_com_documentos_json_carrega_por_arquivo(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir, mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """Com documentos JSON válidos: carregar_json é chamado por-arquivo sobre o glob."""
        from langchain_core.documents import Document

        json_path = Path("/data/documents/pedidos.json")
        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[json_path])
        mock_carregar.return_value = [
            Document(page_content="DESCRIÇÃO: texto da descrição"),
            Document(page_content="CRITÉRIOS: texto dos critérios"),
        ]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs  # passthrough
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        # carregar_json chamado por-arquivo sobre o glob (ING3B-05)
        mock_carregar.assert_called_once_with(json_path)

        # SemanticChunker inicializado com embedding model e config de ADR-013
        mock_splitter.assert_called_once_with(
            embeddings=mock_emb.return_value,
            breakpoint_threshold_type=config.SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_THRESHOLD,
        )

        mock_chroma.from_documents.assert_called_once()
        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        self.assertEqual(len(passed_docs), 2)
        self.assertTrue(passed_docs[0].page_content.startswith("DESCRIÇÃO: "))
        self.assertTrue(passed_docs[1].page_content.startswith("CRITÉRIOS: "))

        # metadados unificados (REC-01)
        self.assertEqual(passed_docs[0].metadata["doc_type"], "json")
        self.assertEqual(passed_docs[0].metadata["page"], 0)
        self.assertEqual(passed_docs[0].metadata["chunk_index"], 0)
        self.assertEqual(passed_docs[1].metadata["chunk_index"], 1)
        self.assertEqual(len(passed_docs[0].metadata["file_hash"]), 64)

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_filtra_documento_json_com_valor_vazio(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """Document JSON com page_content vazio (chave com valor "") é descartado."""
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_carregar.return_value = [
            Document(page_content="DESCRIÇÃO: texto válido"),
            Document(page_content="CAMPO_VAZIO: "),  # .strip() == "CAMPO_VAZIO:" -- nao vazio!
            Document(page_content=""),  # totalmente vazio
            Document(page_content="   "),  # só whitespace
        ]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        contents = [d.page_content for d in passed_docs]
        self.assertIn("DESCRIÇÃO: texto válido", contents)
        self.assertNotIn("", contents)
        self.assertNotIn("   ", contents)
        # "CAMPO_VAZIO: " com .strip() == "CAMPO_VAZIO:" é NAO vazio -> permanece
        self.assertIn("CAMPO_VAZIO: ", contents)

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_pdf_multipagina_um_document_por_pagina(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir,
        mock_extrair, mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """PDF multi-página: 1 Document por página, prefixo PÁGINA_N: ."""
        mock_carregar.return_value = []  # nenhum JSON

        pdf_path = MagicMock()
        pdf_path.name = "materiais_logistica.pdf"
        mock_dir.glob.side_effect = _glob_side_effect(pdf_paths=[pdf_path])

        mock_extrair.return_value = {
            "PÁGINA_1": "conteúdo da primeira página",
            "PÁGINA_2": "conteúdo da segunda página",
        }

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        self.assertEqual(len(passed_docs), 2)
        self.assertEqual(passed_docs[0].page_content, "PÁGINA_1: conteúdo da primeira página")
        self.assertEqual(passed_docs[1].page_content, "PÁGINA_2: conteúdo da segunda página")
        # metadados unificados (REC-01)
        self.assertEqual(passed_docs[0].metadata["page"], 1)
        self.assertEqual(passed_docs[1].metadata["page"], 2)
        self.assertEqual(passed_docs[0].metadata["doc_type"], "pdf")
        self.assertEqual(passed_docs[0].metadata["chunk_index"], 0)
        self.assertEqual(passed_docs[1].metadata["chunk_index"], 1)
        self.assertEqual(len(passed_docs[0].metadata["file_hash"]), 64)

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_pdf_todas_paginas_em_branco_loga_erro_sem_levantar(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir,
        mock_extrair, mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """PDF totalmente em branco: RAGSecurityError é capturada e logada, zero Documents."""
        from langchain_core.documents import Document

        mock_carregar.return_value = [Document(page_content="DESCRIÇÃO: texto válido")]

        pdf_path = MagicMock()
        pdf_path.name = "vazio.pdf"
        mock_dir.glob.side_effect = _glob_side_effect(
            json_paths=[Path("/data/documents/a.json")], pdf_paths=[pdf_path]
        )

        # Usa rag.RAGSecurityError (não o import top-level RAGSecurityError) porque
        # tests/acceptance/test_structured_log_config.py faz importlib.reload(rag_module)
        # quando a suíte completa roda; o `except RAGSecurityError` em cria_vectordb
        # passa a referenciar a classe pós-reload, e a comparação `isinstance` só
        # bate se o side_effect também usar a classe atual de agenticlog.rag.
        mock_extrair.side_effect = rag.RAGSecurityError(
            "PDF não contém texto extraível (somente imagem)."
        )

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level="ERROR") as log_ctx:
            cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        self.assertEqual(len(passed_docs), 1)  # só o Document JSON
        self.assertTrue(any("PDF corrompido ignorado" in m for m in log_ctx.output))

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_reseta_colecao_antes_de_from_documents(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """Rebuild do zero: _resetar_colecao roda com a coleção antes de from_documents."""
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_carregar.return_value = [Document(page_content="X: y")]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        manager = MagicMock()
        manager.attach_mock(mock_resetar, "resetar")
        manager.attach_mock(mock_chroma.from_documents, "from_documents")

        cria_vectordb()

        # o seam vectordb_dir passa a ser propagado ao _resetar_colecao (Fase 3b)
        mock_resetar.assert_called_once_with(config.DEFAULT_COLLECTION_NAME, vectordb_dir=ANY)
        ordem = [nome for nome, _, _ in manager.mock_calls]
        self.assertLess(
            ordem.index("resetar"),
            ordem.index("from_documents"),
            "o descarte da coleção deve ocorrer antes de from_documents",
        )

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_sem_documentos_nao_reseta_colecao(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir, mock_resetar
    ):
        """Sem documentos: retorna cedo sem descartar a coleção existente."""
        mock_dir.glob.side_effect = _glob_side_effect()

        cria_vectordb()

        mock_resetar.assert_not_called()

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_coleção_vazia_pos_rebuild_levanta(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """Guardrail fail-loud: se from_documents persistir 0 chunks, levanta RuntimeError."""
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_carregar.return_value = [Document(page_content="X: y")]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        # coleção persistida reporta 0 chunks → degradação silenciosa que o guardrail deve pegar
        mock_chroma.from_documents.return_value._collection.count.return_value = 0

        with self.assertRaises(RuntimeError):
            cria_vectordb()

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def test_cria_vectordb_seta_metrica_cosine_na_colecao(
        self, mock_valida_path, mock_valida_json, mock_carregar, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """from_documents recebe collection_metadata com hnsw:space=cosine."""
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_carregar.return_value = [Document(page_content="X: y")]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance
        mock_chroma.from_documents.return_value._collection.count.return_value = 1

        cria_vectordb()

        _, kwargs = mock_chroma.from_documents.call_args
        self.assertEqual(
            kwargs["collection_metadata"], config.CHROMA_COLLECTION_METADATA
        )


class TestResetarColecao(unittest.TestCase):
    """Testes para _resetar_colecao (purga segura no rebuild do zero)."""

    @patch("agenticlog.ingestion.store._outras_colecoes_existem", return_value=False)
    @patch("agenticlog.ingestion.store.shutil.rmtree")
    @patch("agenticlog.ingestion.store.DIR_VECTORDB")
    def test_resetar_colecao_unica_remove_diretorio(self, mock_dir, mock_rmtree, mock_outras):
        """Coleção única e diretório presente: rmtree do DIR_VECTORDB (elimina órfãos)."""
        mock_dir.exists.return_value = True

        rag._resetar_colecao("logistica")

        mock_rmtree.assert_called_once_with(mock_dir, ignore_errors=True)

    @patch("agenticlog.ingestion.store._outras_colecoes_existem", return_value=False)
    @patch("agenticlog.ingestion.store.shutil.rmtree")
    @patch("agenticlog.ingestion.store.DIR_VECTORDB")
    def test_resetar_colecao_inexistente_e_no_op(self, mock_dir, mock_rmtree, mock_outras):
        """Diretório ausente: nada a remover (no-op), sem levantar."""
        mock_dir.exists.return_value = False

        rag._resetar_colecao("inexistente")  # não deve levantar

        mock_rmtree.assert_not_called()

    @patch("chromadb.PersistentClient")
    @patch("agenticlog.ingestion.store._outras_colecoes_existem", return_value=True)
    @patch("agenticlog.ingestion.store.shutil.rmtree")
    def test_resetar_colecao_multi_preserva_irmas(
        self, mock_rmtree, mock_outras, mock_client_cls
    ):
        """Multi-coleção: descarta só a coleção alvo (não apaga o diretório/irmãs)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        rag._resetar_colecao("logistica")

        mock_rmtree.assert_not_called()
        mock_client.delete_collection.assert_called_once_with("logistica")

    @patch("chromadb.PersistentClient")
    @patch("agenticlog.ingestion.store._outras_colecoes_existem", return_value=True)
    @patch("agenticlog.ingestion.store.shutil.rmtree")
    def test_resetar_colecao_multi_colecao_inexistente_e_no_op(
        self, mock_rmtree, mock_outras, mock_client_cls
    ):
        """Multi-coleção: exceção de delete_collection (coleção ausente) é engolida."""
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = ValueError("Collection not found")
        mock_client_cls.return_value = mock_client

        rag._resetar_colecao("inexistente")  # não deve levantar

        mock_rmtree.assert_not_called()
        mock_client.delete_collection.assert_called_once_with("inexistente")


class TestOutrasColecoesExistem(unittest.TestCase):
    """Testes para _outras_colecoes_existem (decisão wipe-completo vs delete por coleção)."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._db = Path(self._tmp) / "chroma.sqlite3"

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _criar_db_com_colecoes(self, nomes: list[str]) -> None:
        con = sqlite3.connect(self._db)
        try:
            con.execute("CREATE TABLE collections (name TEXT)")
            con.executemany("INSERT INTO collections (name) VALUES (?)", [(n,) for n in nomes])
            con.commit()
        finally:
            con.close()

    def test_db_inexistente_retorna_false(self) -> None:
        """Sem arquivo SQLite: nenhuma coleção irmã → False (caminho de wipe seguro)."""
        with patch("agenticlog.ingestion.store.DIR_VECTORDB", Path(self._tmp)):
            self.assertFalse(rag._outras_colecoes_existem("logistica"))

    def test_apenas_a_colecao_alvo_retorna_false(self) -> None:
        """Só a coleção alvo presente: não há irmãs → False."""
        self._criar_db_com_colecoes(["logistica"])
        with patch("agenticlog.ingestion.store.DIR_VECTORDB", Path(self._tmp)):
            self.assertFalse(rag._outras_colecoes_existem("logistica"))

    def test_outra_colecao_presente_retorna_true(self) -> None:
        """Existe coleção com nome diferente → True (preservar irmãs)."""
        self._criar_db_com_colecoes(["logistica", "outra"])
        with patch("agenticlog.ingestion.store.DIR_VECTORDB", Path(self._tmp)):
            self.assertTrue(rag._outras_colecoes_existem("logistica"))

    def test_schema_ilegivel_retorna_false(self) -> None:
        """Schema sem tabela collections: degrada para False (wipe seguro)."""
        con = sqlite3.connect(self._db)
        con.execute("CREATE TABLE algo_diferente (x TEXT)")
        con.commit()
        con.close()
        with patch("agenticlog.ingestion.store.DIR_VECTORDB", Path(self._tmp)):
            self.assertFalse(rag._outras_colecoes_existem("logistica"))


class TestLogging(unittest.TestCase):
    """Testes para o módulo de logging em rag.py (LG-01 a LG-11)."""

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def teste_1_log_info_gerando_embeddings(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """assertLogs INFO captura registro contendo 'Gerando' ao executar cria_vectordb (AC-04)."""
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_loader.return_value = [Document(page_content="doc")]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level=logging.INFO) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Gerando" in msg for msg in cm.output),
            f"Esperava 'Gerando' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def teste_2_log_info_criado_com_sucesso(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """assertLogs INFO captura registro contendo 'Criado' ao finalizar cria_vectordb (AC-04)."""
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_loader.return_value = [Document(page_content="doc")]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level=logging.INFO) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Criado" in msg for msg in cm.output),
            f"Esperava 'Criado' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def teste_3_log_warning_nenhum_documento(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir
    ):
        """assertLogs WARNING captura registro com 'Nenhum documento' quando loader retorna [] (AC-07)."""
        mock_dir.glob.side_effect = _glob_side_effect()
        mock_loader.return_value = []

        with self.assertLogs("agenticlog.rag", level=logging.WARNING) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Nenhum documento" in msg for msg in cm.output),
            f"Esperava 'Nenhum documento' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.ingestion.orchestrator._resetar_colecao")
    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    def teste_4_sem_stdout_quando_importado_como_biblioteca(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash, mock_resetar
    ):
        """Nenhuma saída em stdout quando cria_vectordb() chamada como biblioteca (AC-01)."""
        import io
        from langchain_core.documents import Document

        mock_dir.glob.side_effect = _glob_side_effect(json_paths=[Path("/data/documents/a.json")])
        mock_loader.return_value = [Document(page_content="doc")]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            cria_vectordb()

        output = captured_stdout.getvalue()
        self.assertEqual(output, "", f"Esperava stdout vazio, encontrado: {repr(output)}")

    def teste_5_log_level_em_config(self, monkeypatch=None):
        """config.LOG_LEVEL é 'INFO' quando LOG_LEVEL não está definido no ambiente (AC-03).

        Nota: este teste assume que LOG_LEVEL não está definido no ambiente de teste.
        Se a variável estiver definida, use monkeypatch.delenv('LOG_LEVEL', raising=False) antes.
        """
        import os
        original = os.environ.pop("LOG_LEVEL", None)
        try:
            importlib.reload(config)
            self.assertEqual(config.LOG_LEVEL, "INFO")
        finally:
            if original is not None:
                os.environ["LOG_LEVEL"] = original
            importlib.reload(config)
            # Restore rag so RAGSecurityError class identity stays consistent
            # for all tests that execute after this one in the same process.
            importlib.reload(rag)

    def teste_6_logger_modulo_usa_dunder_name(self):
        """logger em rag.py tem name == 'agenticlog.rag' (AC-08)."""
        self.assertEqual(rag.logger.name, "agenticlog.rag")

    def teste_7_erro_seguranca_usa_logger_error(self):
        """RAGSecurityError em _executar_main aciona logger.error com 'Erro de segurança' (AC-05)."""
        with patch.object(rag, "cria_vectordb", side_effect=rag.RAGSecurityError("falha de segurança simulada")):
            with self.assertLogs("agenticlog.rag", level="ERROR") as cm:
                with self.assertRaises(SystemExit) as ctx:
                    _executar_main(["--rebuild"])

        self.assertEqual(ctx.exception.code, 1)
        self.assertTrue(
            any("Erro de segurança" in msg for msg in cm.output),
            f"Esperava 'Erro de segurança' nos logs, encontrado: {cm.output}",
        )

    def teste_8_excecao_generica_usa_logger_error(self):
        """Exception generica em _executar_main aciona logger.error com 'Erro ao criar banco vetorial' (AC-06)."""
        with patch.object(rag, "cria_vectordb", side_effect=RuntimeError("erro generico simulado")):
            with self.assertLogs("agenticlog.rag", level="ERROR") as cm:
                with self.assertRaises(SystemExit) as ctx:
                    _executar_main(["--rebuild"])

        self.assertEqual(ctx.exception.code, 1)
        self.assertTrue(
            any("Erro durante rebuild do banco vetorial" in msg for msg in cm.output),
            f"Esperava 'Erro durante rebuild do banco vetorial' nos logs, encontrado: {cm.output}",
        )


class TestStructuredLogConfig:
    """Testes para configuração de log via variáveis de ambiente (SLC-01 a SLC-08)."""

    @pytest.fixture(autouse=True)
    def _restore_modules(self):
        """Restore config module state after every test in this class.

        Some tests call importlib.reload(config) to exercise env-var validation.
        Only config is reloaded here — reloading rag would create a new
        RAGSecurityError class object, breaking isinstance checks in tests
        that imported RAGSecurityError at module load time.
        """
        yield
        # Teardown: reload config only so LOG_LEVEL / LOG_FORMAT constants are
        # reset.  monkeypatch has already restored env vars at this point
        # (pytest tears down fixtures in LIFO order; monkeypatch scope matches).
        importlib.reload(config)

    def teste_9_log_level_default(self, monkeypatch):
        """config.LOG_LEVEL é 'INFO' quando LOG_LEVEL não está definido no ambiente (SLC-01)."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        importlib.reload(config)
        assert config.LOG_LEVEL == "INFO"

    def teste_10_log_level_from_env(self, monkeypatch):
        """config.LOG_LEVEL é 'DEBUG' quando LOG_LEVEL=DEBUG está definido (SLC-02)."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        importlib.reload(config)
        assert config.LOG_LEVEL == "DEBUG"

    def teste_11_log_format_default(self, monkeypatch):
        """config.LOG_FORMAT é 'text' quando LOG_FORMAT não está definido no ambiente (SLC-03)."""
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(config)
        assert config.LOG_FORMAT == "text"

    def teste_12_log_format_from_env(self, monkeypatch):
        """config.LOG_FORMAT é 'json' quando LOG_FORMAT=json está definido (SLC-04)."""
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(config)
        assert config.LOG_FORMAT == "json"

    def teste_13_log_format_json_output(self, monkeypatch):
        """Cada linha de log emitida por _executar_main é JSON válido com os campos obrigatórios (SLC-05)."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_FORMAT", "json")
        importlib.reload(config)

        # Recarrega rag para que LOG_FORMAT e LOG_LEVEL apontem para os valores atualizados
        importlib.reload(rag)

        captured = io.StringIO()
        captured_handler = logging.StreamHandler(captured)
        from agenticlog.config import _JsonFormatter
        captured_handler.setFormatter(_JsonFormatter())

        # Dispara o log diretamente através do logger do módulo
        rag.logger.addHandler(captured_handler)
        rag.logger.setLevel(logging.DEBUG)
        try:
            rag.logger.info("teste json output")
        finally:
            rag.logger.removeHandler(captured_handler)

        output = captured.getvalue().strip()
        assert output, "Nenhuma saída de log capturada"
        for line in output.splitlines():
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "level" in parsed
            assert "logger" in parsed
            assert "message" in parsed

    def teste_14_log_format_text_preserved(self, monkeypatch):
        """LOG_FORMAT=text não usa _JsonFormatter — o handler padrão é texto simples (SLC-06)."""
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        importlib.reload(config)
        importlib.reload(rag)

        # Em modo text, _executar_main configura o logger 'agenticlog' sem formatter JSON.
        # Verifica que _JsonFormatter NÃO está registrado no logger do pacote após a chamada.
        with patch.object(rag, "cria_vectordb", return_value=None):
            rag._executar_main(["--rebuild"])

        from agenticlog.config import _JsonFormatter
        pkg_logger = logging.getLogger("agenticlog")
        json_handlers = [h for h in pkg_logger.handlers if isinstance(getattr(h, "formatter", None), _JsonFormatter)]
        assert json_handlers == [], f"Não esperava _JsonFormatter em modo text, encontrado: {json_handlers}"
        assert config.LOG_FORMAT == "text"

    def teste_15_invalid_log_level_raises(self, monkeypatch):
        """ValueError é levantado ao importar config quando LOG_LEVEL é inválido (SLC-07)."""
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
            importlib.reload(config)

    def teste_16_invalid_log_format_raises(self, monkeypatch):
        """ValueError é levantado ao importar config quando LOG_FORMAT é inválido (SLC-08)."""
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_FORMAT", "xml")
        with pytest.raises(ValueError, match="Invalid LOG_FORMAT"):
            importlib.reload(config)


class TestReconstruirVectordb(unittest.TestCase):
    """Testes para reconstruir_vectordb."""

    def teste_1_reconstruir_vectordb_chama_cria_vectordb(self):
        """reconstruir_vectordb() chama cria_vectordb() exatamente uma vez."""
        with patch("agenticlog.ingestion.orchestrator.cria_vectordb") as mock_cria:
            reconstruir_vectordb()
        mock_cria.assert_called_once()

    def teste_2_reconstruir_vectordb_propaga_excecao(self):
        """Exceção lançada por cria_vectordb() é propagada pelo reconstruir_vectordb()."""
        with patch("agenticlog.ingestion.orchestrator.cria_vectordb", side_effect=Exception("fail")):
            with self.assertRaises(Exception) as ctx:
                reconstruir_vectordb()
        self.assertEqual(str(ctx.exception), "fail")


class TestAdicionarDocumentoIncrementalmente(unittest.TestCase):
    """Testes para adicionar_documento_incrementalmente."""

    @classmethod
    def setUpClass(cls) -> None:
        import agenticlog.agent  # garante que o módulo está em sys.modules antes dos patches  # noqa: F401

    def _chunk(self, content: str = "chunk") -> LCDocument:
        return LCDocument(page_content=content, metadata={})

    def _setup_vectordb_mock(self, ids: list, metadatas: list) -> MagicMock:
        mock_vdb = MagicMock()
        mock_vdb.get.return_value = {"ids": ids, "metadatas": metadatas}
        return mock_vdb

    def teste_1_adiciona_novo_arquivo_retorna_adicionado(self) -> None:
        """Novo arquivo: chunks adicionados, status adicionado, invalidar chamado."""
        conteudo = b'{"pedido": "P001", "status": "entregue"}'
        mock_vdb = self._setup_vectordb_mock([], [])
        chunks = [self._chunk("chunk1"), self._chunk("chunk2")]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db") as mock_invalidar,
            ):
                mock_loader = MagicMock()
                mock_loader.load.return_value = [LCDocument(page_content="doc", metadata={})]
                mock_loader_cls.return_value = mock_loader
                mock_splitter = MagicMock()
                mock_splitter.split_documents.return_value = chunks
                mock_splitter_cls.return_value = mock_splitter

                result = adicionar_documento_incrementalmente("pedido.json", conteudo)

        self.assertEqual(result["status"], "adicionado")
        self.assertIn("2 chunks", result["mensagem"])
        mock_vdb.add_documents.assert_called_once()
        mock_invalidar.assert_called_once()
        # metadados unificados (REC-01)
        called_chunks = mock_vdb.add_documents.call_args[0][0]
        self.assertEqual(called_chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(called_chunks[1].metadata["chunk_index"], 1)
        self.assertEqual(called_chunks[0].metadata["doc_type"], "json")
        self.assertEqual(called_chunks[0].metadata["page"], 0)
        self.assertEqual(len(called_chunks[0].metadata["file_hash"]), 64)

    def teste_2_primeira_ingestao_sem_colecao_existente(self) -> None:
        """Cold-start (sem coleção existente): cria coleção, ingere sem erro."""
        conteudo = b'{"produto": "caixa"}'
        mock_vdb = self._setup_vectordb_mock([], [])
        chunks = [self._chunk()]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
                mock_splitter_cls.return_value.split_documents.return_value = chunks

                result = adicionar_documento_incrementalmente("produto.json", conteudo)

        self.assertEqual(result["status"], "adicionado")
        mock_vdb.add_documents.assert_called_once()

    def teste_3_detecta_duplicata_mesmo_hash(self) -> None:
        """Mesmo arquivo, mesmo hash: retorna status duplicado, sem adicionar chunks."""
        conteudo = b'{"pedido": "123"}'
        hash_str = hashlib.sha256(conteudo).hexdigest()
        mock_vdb = self._setup_vectordb_mock(
            ["id1"], [{"file_hash": hash_str, "source": "/some/path/doc.json"}]
        )

        with (
            patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
            patch("agenticlog.rag._get_rag_embedding_model"),
            patch("agenticlog.rag.DIR_DOCUMENTS") as mock_dir,
            patch("agenticlog.rag.DIR_VECTORDB"),
        ):
            mock_dir.glob.return_value = []

            result = adicionar_documento_incrementalmente("doc.json", conteudo)

        self.assertEqual(result["status"], "duplicado")
        mock_vdb.add_documents.assert_not_called()

    def teste_4_upsert_hash_diferente(self) -> None:
        """Mesmo nome, hash diferente: upsert — adiciona chunks novos, deleta antigos, retorna 'substituido'."""
        conteudo = b'{"pedido": "123_v2"}'
        hash_antigo = hashlib.sha256(b"versao_anterior").hexdigest()
        old_ids = ["id_antigo_1", "id_antigo_2"]
        mock_vdb = self._setup_vectordb_mock(
            old_ids,
            [{"file_hash": hash_antigo, "source": "/path/doc.json"} for _ in old_ids],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [
                    LCDocument(page_content="pedido v2", metadata={})
                ]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                result = adicionar_documento_incrementalmente("doc.json", conteudo)

        self.assertEqual(result["status"], "substituido")
        mock_vdb.add_documents.assert_called_once()
        mock_vdb.delete.assert_called_once_with(ids=old_ids)

    def teste_5_rejeita_seguranca(self) -> None:
        """Falha de validação de segurança levanta RAGSecurityError antes de tocar Chroma."""
        with self.assertRaises(rag.RAGSecurityError):
            rag.adicionar_documento_incrementalmente("arquivo.txt", b"{}")

    def teste_6_rejeita_limite_arquivos(self) -> None:
        """MAX_JSON_FILES atingido levanta RAGSecurityError."""
        with patch("agenticlog.rag.DIR_DOCUMENTS") as mock_dir:
            mock_dir.glob.return_value = [MagicMock()] * MAX_JSON_FILES
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                rag.adicionar_documento_incrementalmente("novo.json", b'{"k": "v"}')
        self.assertIn(str(MAX_JSON_FILES), str(ctx.exception))

    def teste_7_falha_no_add_dispara_rollback(self) -> None:
        """Exceção no add_documents: delete chamado com IDs pré-gerados, exceção re-levantada."""
        conteudo = b'{"k": "v"}'
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_vdb.add_documents.side_effect = RuntimeError("embed fail")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
            ):
                mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                with self.assertRaises(RuntimeError) as exc_ctx:
                    adicionar_documento_incrementalmente("doc.json", conteudo)

            self.assertNotIn("doc.json", [f.name for f in tmp_path.iterdir()])

        mock_vdb.delete.assert_called_once()
        self.assertEqual(str(exc_ctx.exception), "embed fail")

    def teste_8_rollback_falha_loga_warning_e_relevanva_original(self) -> None:
        """delete falha: WARNING logado com IDs órfãos, exceção original re-levantada."""
        conteudo = b'{"k": "v"}'
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_vdb.add_documents.side_effect = RuntimeError("embed fail")
        mock_vdb.delete.side_effect = RuntimeError("rollback fail")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                self.assertLogs("agenticlog.rag", level="WARNING") as log_ctx,
            ):
                mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                with self.assertRaises(RuntimeError) as exc_ctx:
                    adicionar_documento_incrementalmente("doc.json", conteudo)

        self.assertEqual(str(exc_ctx.exception), "embed fail")
        self.assertTrue(any("IDs órfãos" in m for m in log_ctx.output))

    def teste_9_zero_chunks_retorna_adicionado_com_zero(self) -> None:
        """Documento sem chunks: WARNING logado, status adicionado, arquivo removido do disco, add_documents não chamado."""
        conteudo = b'{"k": "v"}'
        mock_vdb = self._setup_vectordb_mock([], [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                self.assertLogs("agenticlog.rag", level="WARNING") as log_ctx,
            ):
                mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
                mock_splitter_cls.return_value.split_documents.return_value = []

                result = adicionar_documento_incrementalmente("doc.json", conteudo)

            self.assertNotIn("doc.json", [f.name for f in tmp_path.iterdir()])

        self.assertEqual(result["status"], "adicionado")
        self.assertIn("0 chunks", result["mensagem"])
        mock_vdb.add_documents.assert_not_called()
        self.assertTrue(any("zero chunks" in m for m in log_ctx.output))

    def teste_10_usa_jq_schema_compartilhado(self) -> None:
        """JSONLoader é construído com JQ_SCHEMA_CAMPOS_JSON (constante compartilhada)."""
        conteudo = b'{"campo": "valor"}'
        mock_vdb = self._setup_vectordb_mock([], [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [
                    LCDocument(page_content="campo: valor", metadata={})
                ]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                adicionar_documento_incrementalmente("doc.json", conteudo)

        _, kwargs = mock_loader_cls.call_args
        self.assertEqual(kwargs["jq_schema"], config.JQ_SCHEMA_CAMPOS_JSON)

        mock_splitter_cls.assert_called_once_with(
            embeddings=ANY,
            breakpoint_threshold_type=config.SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_THRESHOLD,
        )

    def teste_11_filtra_documents_com_page_content_vazio(self) -> None:
        """Documents com page_content vazio (apos strip) sao descartados antes do split."""
        conteudo = b'{"campo_a": "valor", "campo_b": ""}'
        mock_vdb = self._setup_vectordb_mock([], [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [
                    LCDocument(page_content="campo_a: valor", metadata={}),
                    LCDocument(page_content="campo_b: ", metadata={}),  # .strip() == "campo_b:"
                    LCDocument(page_content="", metadata={}),  # totalmente vazio -- filtrado
                ]
                mock_splitter = mock_splitter_cls.return_value
                mock_splitter.split_documents.side_effect = lambda docs: list(docs)

                adicionar_documento_incrementalmente("doc.json", conteudo)

        passed_docs = mock_splitter.split_documents.call_args[0][0]
        contents = [d.page_content for d in passed_docs]
        self.assertIn("campo_a: valor", contents)
        self.assertIn("campo_b: ", contents)  # .strip() == "campo_b:" -- nao vazio, permanece
        self.assertNotIn("", contents)
        self.assertEqual(len(passed_docs), 2)

    def teste_12_upsert_falha_no_add_restaura_arquivo_antigo(self) -> None:
        """Upsert + falha no add_documents: conteúdo antigo restaurado no disco (consistente
        com chunks antigos ainda no Chroma); exceção re-levantada; chunks antigos NÃO deletados."""
        conteudo_novo = b'{"pedido": "123_v2"}'
        hash_antigo = hashlib.sha256(b"versao_anterior").hexdigest()
        old_ids = ["id_antigo_1"]
        mock_vdb = self._setup_vectordb_mock(
            old_ids, [{"file_hash": hash_antigo, "source": "/path/doc.json"}]
        )
        mock_vdb.add_documents.side_effect = RuntimeError("embed fail")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conteudo_antigo = b'{"pedido": "123_v1"}'
            (tmp_path / "doc.json").write_bytes(conteudo_antigo)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
            ):
                mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                with self.assertRaises(RuntimeError) as exc_ctx:
                    adicionar_documento_incrementalmente("doc.json", conteudo_novo)

                # disco restaurado para o conteúdo antigo, não deletado nem sobrescrito
                self.assertTrue((tmp_path / "doc.json").exists())
                self.assertEqual((tmp_path / "doc.json").read_bytes(), conteudo_antigo)
                # nenhum .bak órfão deixado
                self.assertEqual(list(tmp_path.glob("*.bak")), [])

        self.assertEqual(str(exc_ctx.exception), "embed fail")
        # chunks antigos preservados (delete só chamado no rollback dos chunks novos)
        mock_vdb.delete.assert_called_once()
        self.assertNotIn(call(ids=old_ids), mock_vdb.delete.call_args_list)

    def teste_13_upsert_falha_no_split_restaura_arquivo_antigo(self) -> None:
        """Upsert + falha no split_documents (chunking/embeddings, antes do add): arquivo antigo
        restaurado; exceção propagada; add_documents NÃO chamado; chunks antigos NÃO deletados."""
        conteudo_novo = b'{"pedido": "123_v2"}'
        hash_antigo = hashlib.sha256(b"versao_anterior").hexdigest()
        old_ids = ["id_antigo_1"]
        mock_vdb = self._setup_vectordb_mock(
            old_ids, [{"file_hash": hash_antigo, "source": "/path/doc.json"}]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conteudo_antigo = b'{"pedido": "123_v1"}'
            (tmp_path / "doc.json").write_bytes(conteudo_antigo)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
            ):
                mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
                mock_splitter_cls.return_value.split_documents.side_effect = RuntimeError("embed fail")

                with self.assertRaises(RuntimeError) as exc_ctx:
                    adicionar_documento_incrementalmente("doc.json", conteudo_novo)

                self.assertTrue((tmp_path / "doc.json").exists())
                self.assertEqual((tmp_path / "doc.json").read_bytes(), conteudo_antigo)
                self.assertEqual(list(tmp_path.glob("*.bak")), [])

        self.assertEqual(str(exc_ctx.exception), "embed fail")
        mock_vdb.add_documents.assert_not_called()
        mock_vdb.delete.assert_not_called()


class TestAdicionarPdfIncrementalmente(unittest.TestCase):
    """Testes para adicionar_pdf_incrementalmente (REC-02)."""

    @classmethod
    def setUpClass(cls) -> None:
        import agenticlog.agent  # garante que o módulo está em sys.modules antes dos patches  # noqa: F401

    def _valid_pdf_bytes(self) -> bytes:
        return b"%PDF-1.4 fake content"

    def _chunk(self, content: str = "chunk", page: int = 1) -> LCDocument:
        return LCDocument(page_content=content, metadata={"page": page})

    def _setup_vectordb_mock(self, ids: list, metadatas: list) -> MagicMock:
        mock_vdb = MagicMock()
        mock_vdb.get.return_value = {"ids": ids, "metadatas": metadatas}
        return mock_vdb

    @patch("agenticlog.ingestion.orchestrator.uuid")
    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.ingestion.orchestrator.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_1_happy_path_adiciona_chunks(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_extrair: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_dir: MagicMock,
        mock_shutil: MagicMock,
        mock_tempfile: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """Happy path: PDF válido → arquivo salvo, chunks inseridos, 5 campos de metadados, retorna adicionado."""
        conteudo = self._valid_pdf_bytes()
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_chroma.return_value = mock_vdb

        mock_dir.glob.return_value = []
        fake_path = MagicMock()
        fake_path.__truediv__ = MagicMock(return_value=fake_path)
        fake_path.__str__ = MagicMock(return_value="/data/documents/contrato.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_path)

        mock_extrair.return_value = {"PÁGINA_1": "texto da página 1"}

        # chunk com source já setado — simula o que split_documents faz ao herdar metadados do Document pai
        chunk1 = LCDocument(
            page_content="PÁGINA_1: texto da página 1",
            metadata={"page": 1, "source": "/data/documents/contrato.pdf"},
        )
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [chunk1]
        mock_splitter_cls.return_value = mock_splitter

        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = "/tmp/tmpXXX.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        mock_uuid.uuid4.return_value.hex = "abc123"

        with patch("agenticlog.agent.invalidar_vector_db") as mock_invalidar:
            result = adicionar_pdf_incrementalmente("contrato.pdf", conteudo)

        self.assertEqual(result["status"], "adicionado")
        self.assertIn("contrato.pdf", result["mensagem"])
        mock_vdb.add_documents.assert_called_once()
        mock_invalidar.assert_called_once()
        mock_shutil.move.assert_called_once()
        mock_extrair.assert_called_once()

        called_chunks = mock_vdb.add_documents.call_args[0][0]
        meta = called_chunks[0].metadata
        self.assertIn("file_hash", meta)
        self.assertIn("chunk_index", meta)
        self.assertIn("doc_type", meta)
        self.assertIn("page", meta)
        self.assertIn("source", meta)
        self.assertEqual(meta["doc_type"], "pdf")

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_2_duplicado_mesmo_hash(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_dir: MagicMock,
    ) -> None:
        """Mesmo source + mesmo file_hash no ChromaDB → retorna duplicado, sem escrita em disco."""
        conteudo = self._valid_pdf_bytes()
        hash_str = hashlib.sha256(conteudo).hexdigest()

        fake_path = MagicMock()
        fake_path.__str__ = MagicMock(return_value="/data/documents/contrato.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_path)
        mock_dir.glob.return_value = []

        mock_vdb = self._setup_vectordb_mock(
            ["id1"], [{"file_hash": hash_str}]
        )
        mock_chroma.return_value = mock_vdb

        result = adicionar_pdf_incrementalmente("contrato.pdf", conteudo)

        self.assertEqual(result["status"], "duplicado")
        mock_vdb.add_documents.assert_not_called()

    @patch("agenticlog.ingestion.store.shutil")
    @patch("agenticlog.agent.invalidar_vector_db")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.ingestion.orchestrator.shutil")
    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_3_upsert_hash_diferente_mesmo_nome(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_invalidar: MagicMock,
        mock_store_shutil: MagicMock,
    ) -> None:
        """Mesmo source, hash diferente → upsert: chunks antigos deletados, novos adicionados, retorna 'substituido'."""
        conteudo = self._valid_pdf_bytes()
        hash_antigo = hashlib.sha256(b"conteudo anterior").hexdigest()
        old_ids = ["id1"]

        fake_path = MagicMock()
        fake_path.__str__ = MagicMock(return_value="/data/documents/contrato.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_path)
        mock_dir.glob.return_value = []

        mock_vdb = self._setup_vectordb_mock(old_ids, [{"file_hash": hash_antigo}])
        mock_chroma.return_value = mock_vdb

        fake_tmp = MagicMock()
        fake_tmp.name = "/tmp/fake_contrato.pdf"
        mock_tempfile.NamedTemporaryFile.return_value.__enter__ = MagicMock(return_value=fake_tmp)
        mock_tempfile.NamedTemporaryFile.return_value.__exit__ = MagicMock(return_value=False)

        mock_extrair.return_value = {"PÁGINA_1": "Conteúdo de teste"}
        mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

        result = adicionar_pdf_incrementalmente("contrato.pdf", conteudo)

        self.assertEqual(result["status"], "substituido")
        mock_vdb.add_documents.assert_called_once()
        mock_vdb.delete.assert_called_once_with(ids=old_ids)

    def teste_3b_upsert_falha_no_add_restaura_pdf_antigo(self) -> None:
        """Upsert + falha no add_documents: PDF antigo restaurado no disco (consistente com
        chunks antigos ainda no Chroma); exceção re-levantada; chunks antigos NÃO deletados."""
        conteudo_novo = b"%PDF-1.4 versao nova"
        hash_antigo = hashlib.sha256(b"%PDF-1.4 versao antiga").hexdigest()
        old_ids = ["id1"]
        mock_vdb = self._setup_vectordb_mock(old_ids, [{"file_hash": hash_antigo}])
        mock_vdb.add_documents.side_effect = RuntimeError("embed fail")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conteudo_antigo = b"%PDF-1.4 versao antiga"
            (tmp_path / "contrato.pdf").write_bytes(conteudo_antigo)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf") as mock_extrair,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
            ):
                mock_extrair.return_value = {"PÁGINA_1": "texto"}
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                with self.assertRaises(RuntimeError) as exc_ctx:
                    adicionar_pdf_incrementalmente("contrato.pdf", conteudo_novo)

                self.assertTrue((tmp_path / "contrato.pdf").exists())
                self.assertEqual((tmp_path / "contrato.pdf").read_bytes(), conteudo_antigo)
                self.assertEqual(list(tmp_path.glob("*.bak")), [])

        self.assertEqual(str(exc_ctx.exception), "embed fail")
        mock_vdb.delete.assert_called_once()
        self.assertNotIn(call(ids=old_ids), mock_vdb.delete.call_args_list)

    def teste_3c_upsert_falha_no_split_restaura_pdf_antigo(self) -> None:
        """Upsert + falha no split_documents (antes do add): PDF antigo restaurado; exceção
        propagada; add_documents NÃO chamado; chunks antigos NÃO deletados."""
        conteudo_novo = b"%PDF-1.4 versao nova"
        hash_antigo = hashlib.sha256(b"%PDF-1.4 versao antiga").hexdigest()
        old_ids = ["id1"]
        mock_vdb = self._setup_vectordb_mock(old_ids, [{"file_hash": hash_antigo}])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            conteudo_antigo = b"%PDF-1.4 versao antiga"
            (tmp_path / "contrato.pdf").write_bytes(conteudo_antigo)

            with (
                patch("agenticlog.ingestion.orchestrator.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf") as mock_extrair,
                patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
            ):
                mock_extrair.return_value = {"PÁGINA_1": "texto"}
                mock_splitter_cls.return_value.split_documents.side_effect = RuntimeError("embed fail")

                with self.assertRaises(RuntimeError) as exc_ctx:
                    adicionar_pdf_incrementalmente("contrato.pdf", conteudo_novo)

                self.assertTrue((tmp_path / "contrato.pdf").exists())
                self.assertEqual((tmp_path / "contrato.pdf").read_bytes(), conteudo_antigo)
                self.assertEqual(list(tmp_path.glob("*.bak")), [])

        self.assertEqual(str(exc_ctx.exception), "embed fail")
        mock_vdb.add_documents.assert_not_called()
        mock_vdb.delete.assert_not_called()

    def teste_4_rejeita_extensao_invalida(self) -> None:
        """Extensão .docx → RAGSecurityError antes de qualquer operação em disco."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            adicionar_pdf_incrementalmente("documento.docx", b"%PDF-1.4 content")
        self.assertIn("pdf", str(ctx.exception).lower())

    def teste_5_rejeita_magic_bytes_invalidos(self) -> None:
        """Conteúdo sem magic bytes %PDF → RAGSecurityError antes de qualquer escrita."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            adicionar_pdf_incrementalmente("doc.pdf", b"PK\x03\x04 not a pdf")
        self.assertIn("PDF", str(ctx.exception))

    def teste_6_rejeita_tamanho_excedido(self) -> None:
        """Conteúdo > MAX_DOCUMENT_FILE_SIZE_MB MB → RAGSecurityError antes de escrita em disco."""
        from agenticlog.config import MAX_DOCUMENT_FILE_SIZE_MB
        conteudo = b"%PDF" + b"x" * (MAX_DOCUMENT_FILE_SIZE_MB * 1024 * 1024 + 1)
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            adicionar_pdf_incrementalmente("grande.pdf", conteudo)
        self.assertIn(str(MAX_DOCUMENT_FILE_SIZE_MB), str(ctx.exception))

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_7_rejeita_limite_de_arquivos(self, mock_dir: MagicMock) -> None:
        """json_count + pdf_count >= MAX_JSON_FILES → RAGSecurityError."""
        from agenticlog.config import MAX_JSON_FILES
        mock_dir.glob.return_value = [MagicMock()] * MAX_JSON_FILES
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            adicionar_pdf_incrementalmente("novo.pdf", self._valid_pdf_bytes())
        self.assertIn(str(MAX_JSON_FILES), str(ctx.exception))

    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_8_rejeita_pdf_com_senha(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
    ) -> None:
        """extrair_texto_pdf lança RAGSecurityError(senha) → propagado, tempfile removido."""
        conteudo = self._valid_pdf_bytes()
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_chroma.return_value = mock_vdb
        mock_dir.glob.return_value = []

        fake_path = MagicMock()
        fake_path.__str__ = MagicMock(return_value="/data/documents/protegido.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_path)

        mock_tmp_file = MagicMock()
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp_file.name = "/tmp/tmpXXX.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp_file

        with patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf", side_effect=rag.RAGSecurityError("PDF protegido por senha.")):
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                adicionar_pdf_incrementalmente("protegido.pdf", conteudo)

        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_9_rejeita_pdf_somente_imagem(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
    ) -> None:
        """extrair_texto_pdf lança RAGSecurityError(somente imagem) → propagado."""
        conteudo = self._valid_pdf_bytes()
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_chroma.return_value = mock_vdb
        mock_dir.glob.return_value = []

        fake_path = MagicMock()
        fake_path.__str__ = MagicMock(return_value="/data/documents/scan.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_path)

        mock_tmp_file = MagicMock()
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp_file.name = "/tmp/tmpXXX.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp_file

        with patch(
            "agenticlog.ingestion.orchestrator.extrair_texto_pdf",
            side_effect=rag.RAGSecurityError("PDF não contém texto extraível (somente imagem)."),
        ):
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                adicionar_pdf_incrementalmente("scan.pdf", conteudo)

        self.assertIn("somente imagem", str(ctx.exception))

    @patch("agenticlog.ingestion.orchestrator.uuid")
    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.ingestion.orchestrator.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_10_zero_chunks_retorna_sem_indexar(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_extrair: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_dir: MagicMock,
        mock_shutil: MagicMock,
        mock_tempfile: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """PDF válido mas splitter retorna [] → arquivo removido, retorna adicionado com mensagem de 0 chunks."""
        conteudo = self._valid_pdf_bytes()
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_chroma.return_value = mock_vdb

        mock_dir.glob.return_value = []
        fake_saved_path = MagicMock()
        fake_saved_path.__str__ = MagicMock(return_value="/data/documents/vazio.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_saved_path)

        mock_extrair.return_value = {"PÁGINA_1": "algum texto"}

        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = []
        mock_splitter_cls.return_value = mock_splitter

        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = "/tmp/tmpXXX.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        with self.assertLogs("agenticlog.rag", level="WARNING"):
            result = adicionar_pdf_incrementalmente("vazio.pdf", conteudo)

        self.assertEqual(result["status"], "adicionado")
        self.assertIn("0 chunks", result["mensagem"])
        mock_vdb.add_documents.assert_not_called()
        fake_saved_path.unlink.assert_called()

    @patch("agenticlog.ingestion.orchestrator.uuid")
    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.ingestion.orchestrator.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_11_rollback_em_falha_de_add_documents(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_extrair: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_dir: MagicMock,
        mock_shutil: MagicMock,
        mock_tempfile: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """add_documents lança → vectordb.delete chamado com chunk_ids, arquivo removido, exceção re-levantada."""
        conteudo = self._valid_pdf_bytes()
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_vdb.add_documents.side_effect = RuntimeError("embed falhou")
        mock_chroma.return_value = mock_vdb

        mock_dir.glob.return_value = []
        fake_saved_path = MagicMock()
        fake_saved_path.__str__ = MagicMock(return_value="/data/documents/relatorio.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_saved_path)

        mock_extrair.return_value = {"PÁGINA_1": "texto"}

        chunk = self._chunk("PÁGINA_1: texto", page=1)
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [chunk]
        mock_splitter_cls.return_value = mock_splitter

        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = "/tmp/tmpXXX.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        mock_uuid.uuid4.return_value.hex = "deadbeef"

        with self.assertRaises(RuntimeError) as exc_ctx:
            adicionar_pdf_incrementalmente("relatorio.pdf", conteudo)

        self.assertEqual(str(exc_ctx.exception), "embed falhou")
        mock_vdb.delete.assert_called_once()
        fake_saved_path.unlink.assert_called()

    @patch("agenticlog.ingestion.orchestrator.uuid")
    @patch("agenticlog.ingestion.orchestrator.tempfile")
    @patch("agenticlog.ingestion.orchestrator.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.ingestion.orchestrator.extrair_texto_pdf")
    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    def teste_12_chunk_index_global_entre_paginas(
        self,
        mock_chroma: MagicMock,
        mock_emb: MagicMock,
        mock_extrair: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_dir: MagicMock,
        mock_shutil: MagicMock,
        mock_tempfile: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """PDF 3-páginas: chunk_index é global (0, 1, 2) sem reset por página."""
        conteudo = self._valid_pdf_bytes()
        mock_vdb = self._setup_vectordb_mock([], [])
        mock_chroma.return_value = mock_vdb

        mock_dir.glob.return_value = []
        fake_saved_path = MagicMock()
        fake_saved_path.__str__ = MagicMock(return_value="/data/documents/multi.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=fake_saved_path)

        mock_extrair.return_value = {
            "PÁGINA_1": "texto pagina 1",
            "PÁGINA_2": "texto pagina 2",
            "PÁGINA_3": "texto pagina 3",
        }

        chunks = [
            self._chunk("PÁGINA_1: texto pagina 1", page=1),
            self._chunk("PÁGINA_2: texto pagina 2", page=2),
            self._chunk("PÁGINA_3: texto pagina 3", page=3),
        ]
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = chunks
        mock_splitter_cls.return_value = mock_splitter

        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = "/tmp/tmpXXX.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        mock_uuid.uuid4.return_value.hex = "aabbcc"

        with patch("agenticlog.agent.invalidar_vector_db"):
            adicionar_pdf_incrementalmente("multi.pdf", conteudo)

        called_chunks = mock_vdb.add_documents.call_args[0][0]
        chunk_indices = [c.metadata["chunk_index"] for c in called_chunks]
        self.assertEqual(chunk_indices, [0, 1, 2])


class TestIngerirIncrementalmente(unittest.TestCase):
    """Testes para ingerir_incrementalmente e o CLI incremental por padrão (REC-04)."""

    def _criar_docs(self, tmp: Path, nomes: list[str]) -> None:
        for nome in nomes:
            (tmp / nome).write_bytes(b"conteudo")

    def teste_1_despacha_por_extensao_e_agrega_contadores(self):
        """Itera *.json e *.pdf, despacha por extensão e agrega contadores por status."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._criar_docs(tmp, ["a.json", "b.json", "c.pdf"])

            mock_json = MagicMock(side_effect=[
                {"status": "adicionado", "mensagem": "ok a"},
                {"status": "duplicado", "mensagem": "dup b"},
            ])
            mock_pdf = MagicMock(return_value={"status": "adicionado", "mensagem": "ok c"})

            with patch.object(orchestrator, "DIR_DOCUMENTS", tmp), \
                 patch.object(orchestrator, "adicionar_documento_incrementalmente", mock_json), \
                 patch.object(orchestrator, "adicionar_pdf_incrementalmente", mock_pdf):
                contadores = orchestrator.ingerir_incrementalmente()

        self.assertEqual(contadores, {"adicionado": 2, "duplicado": 1})
        self.assertEqual(mock_json.call_count, 2)
        mock_pdf.assert_called_once()
        # nome do arquivo é passado, não o caminho completo
        self.assertEqual(mock_pdf.call_args[0][0], "c.pdf")

    def teste_2_erro_operacional_nao_aborta_lote(self):
        """Erro operacional (não-segurança) em um arquivo é contado como 'erro'; os demais seguem."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._criar_docs(tmp, ["ok.json", "ruim.json"])

            def _side(nome, conteudo, collection_name, **kwargs):
                if nome == "ruim.json":
                    raise RuntimeError("falha operacional simulada")
                return {"status": "adicionado", "mensagem": "ok"}

            with patch.object(orchestrator, "DIR_DOCUMENTS", tmp), \
                 patch.object(orchestrator, "adicionar_documento_incrementalmente", side_effect=_side), \
                 patch.object(orchestrator, "adicionar_pdf_incrementalmente", MagicMock()):
                contadores = orchestrator.ingerir_incrementalmente()

        self.assertEqual(contadores, {"adicionado": 1, "erro": 1})

    def teste_2b_violacao_seguranca_aborta_lote(self):
        """RAGSecurityError é propagado (fail-fast), abortando o lote em vez de ser contado."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._criar_docs(tmp, ["a.json", "malicioso.json"])

            def _side(nome, conteudo, collection_name, **kwargs):
                if nome == "malicioso.json":
                    raise rag.RAGSecurityError("chave proibida")
                return {"status": "adicionado", "mensagem": "ok"}

            with patch.object(orchestrator, "DIR_DOCUMENTS", tmp), \
                 patch.object(orchestrator, "adicionar_documento_incrementalmente", side_effect=_side), \
                 patch.object(orchestrator, "adicionar_pdf_incrementalmente", MagicMock()):
                with self.assertRaises(rag.RAGSecurityError):
                    orchestrator.ingerir_incrementalmente()

    def teste_2c_propaga_collection_name(self):
        """collection_name é repassado às funções incrementais (json e pdf)."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._criar_docs(tmp, ["a.json", "b.pdf"])

            mock_json = MagicMock(return_value={"status": "adicionado", "mensagem": "ok"})
            mock_pdf = MagicMock(return_value={"status": "adicionado", "mensagem": "ok"})

            with patch.object(orchestrator, "DIR_DOCUMENTS", tmp), \
                 patch.object(orchestrator, "adicionar_documento_incrementalmente", mock_json), \
                 patch.object(orchestrator, "adicionar_pdf_incrementalmente", mock_pdf):
                orchestrator.ingerir_incrementalmente("colecao_custom")

        self.assertEqual(mock_json.call_args[0][2], "colecao_custom")
        self.assertEqual(mock_pdf.call_args[0][2], "colecao_custom")

    def teste_3_sem_arquivos_retorna_dict_vazio(self):
        """Diretório sem documentos retorna contadores vazios sem erro."""
        with tempfile.TemporaryDirectory() as d:
            with patch.object(rag, "DIR_DOCUMENTS", Path(d)):
                self.assertEqual(ingerir_incrementalmente(), {})

    def teste_4_cli_sem_flags_chama_ingestao_incremental(self):
        """_executar_main() sem flags invoca ingerir_incrementalmente, não cria_vectordb (REC-04)."""
        with patch.object(rag, "ingerir_incrementalmente", return_value={}) as mock_inc, \
             patch.object(rag, "cria_vectordb") as mock_rebuild:
            rag._executar_main([])

        mock_inc.assert_called_once()
        mock_rebuild.assert_not_called()

    def teste_5_cli_rebuild_chama_cria_vectordb(self):
        """_executar_main(['--rebuild']) invoca cria_vectordb, não a ingestão incremental."""
        with patch.object(rag, "ingerir_incrementalmente") as mock_inc, \
             patch.object(rag, "cria_vectordb", return_value=None) as mock_rebuild:
            rag._executar_main(["--rebuild"])

        mock_rebuild.assert_called_once()
        mock_inc.assert_not_called()


if __name__ == "__main__":
    print("\nIniciando testes do RAG. Aguarde...\n")
    unittest.main(verbosity=2)
