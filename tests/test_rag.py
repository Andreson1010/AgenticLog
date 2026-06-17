# AgenticLog - Testes unitários para rag.py
"""
Testes para o pipeline RAG em agenticlog.rag.
Cobre validações de segurança, path traversal e criação do banco vetorial.
"""

import importlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY, call

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
import pytest

import hashlib
import tempfile

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


class TestEmbeddingModelConfig(unittest.TestCase):
    """Testes para a constante EMBEDDING_MODEL (PORTPT-01 / AC1)."""

    def test_embedding_model_e_multilingue(self):
        """EMBEDDING_MODEL aponta para o modelo multilíngue (paraphrase-multilingual-mpnet)."""
        self.assertEqual(
            config.EMBEDDING_MODEL,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )


class TestRAGSecurityError(unittest.TestCase):
    """Testes para a exceção RAGSecurityError."""

    def test_raise_rag_security_error(self):
        """RAGSecurityError pode ser levantada e capturada."""
        with self.assertRaises(RAGSecurityError) as ctx:
            raise RAGSecurityError("Mensagem de teste")
        self.assertIn("Mensagem de teste", str(ctx.exception))


class TestValidaPathDocumentos(unittest.TestCase):
    """Testes para _valida_path_documentos."""

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.PROJECT_ROOT")
    def test_path_fora_do_projeto_levanta_erro(self, mock_root, mock_dir):
        """Path fora do PROJECT_ROOT levanta RAGSecurityError."""
        resolved_dir = MagicMock()
        resolved_dir.relative_to.side_effect = ValueError("fora")
        mock_dir.resolve.return_value = resolved_dir

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            _valida_path_documentos()
        self.assertIn("fora do projeto", str(ctx.exception))

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.PROJECT_ROOT")
    def test_diretorio_nao_existe_levanta_erro(self, mock_root, mock_dir):
        """Diretório inexistente levanta RAGSecurityError."""
        resolved_dir = MagicMock()
        resolved_dir.relative_to.return_value = Path("data/documents")
        resolved_dir.exists.return_value = False
        mock_dir.resolve.return_value = resolved_dir

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            _valida_path_documentos()
        self.assertIn("não existe", str(ctx.exception))

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.PROJECT_ROOT")
    def test_caminho_nao_e_diretorio_levanta_erro(self, mock_root, mock_dir):
        """Path que não é diretório levanta RAGSecurityError."""
        resolved_dir = MagicMock()
        resolved_dir.relative_to.return_value = Path("data/documents")
        resolved_dir.exists.return_value = True
        resolved_dir.is_dir.return_value = False
        mock_dir.resolve.return_value = resolved_dir

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            _valida_path_documentos()
        self.assertIn("não é um diretório", str(ctx.exception))

    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.PROJECT_ROOT")
    def test_path_valido_nao_levanta(self, mock_root, mock_dir):
        """Path válido dentro do projeto não levanta exceção."""
        resolved_dir = MagicMock()
        resolved_dir.relative_to.return_value = Path("data/documents")
        resolved_dir.exists.return_value = True
        resolved_dir.is_dir.return_value = True
        mock_dir.resolve.return_value = resolved_dir

        _valida_path_documentos()  # não deve levantar


class TestValidaJsonSemChavesProibidas(unittest.TestCase):
    """Testes para _valida_json_sem_chaves_proibidas."""

    def test_json_invalido_levanta_erro(self):
        """JSON malformado levanta RAGSecurityError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{ invalido }")
            path = Path(f.name)
        try:
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                _valida_json_sem_chaves_proibidas(path)
            self.assertIn("JSON inválido", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_chave_proibida_em_dict_levanta_erro(self):
        """JSON com chave 'lc' em dict levanta RAGSecurityError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"lc": "valor", "outro": "campo"}, f)
            path = Path(f.name)
        try:
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                _valida_json_sem_chaves_proibidas(path)
            self.assertIn("chave proibida", str(ctx.exception))
            self.assertIn("lc", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_chave_proibida_em_lista_levanta_erro(self):
        """JSON com chave 'lc' em item de lista levanta RAGSecurityError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([{"campo": "ok"}, {"lc": "invalido"}], f)
            path = Path(f.name)
        try:
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                _valida_json_sem_chaves_proibidas(path)
            self.assertIn("chave proibida", str(ctx.exception))
            self.assertIn("item 1", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_json_valido_sem_chaves_proibidas_nao_levanta(self):
        """JSON válido sem chaves proibidas não levanta exceção."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"descricao": "texto", "campo": "valor"}, f)
            path = Path(f.name)
        try:
            _valida_json_sem_chaves_proibidas(path)  # não deve levantar
        finally:
            path.unlink(missing_ok=True)


class TestValidaArquivosJson(unittest.TestCase):
    """Testes para _valida_arquivos_json."""

    @patch("agenticlog.rag._valida_json_sem_chaves_proibidas")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.MAX_JSON_FILES", 2)
    def test_excesso_de_arquivos_levanta_erro(self, mock_dir, mock_valida):
        """Mais arquivos que MAX_JSON_FILES levanta RAGSecurityError."""
        mock_dir.glob.return_value = [
            Path("a.json"),
            Path("b.json"),
            Path("c.json"),
        ]
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            _valida_arquivos_json()
        self.assertIn("Excesso de arquivos", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    @patch("agenticlog.rag._valida_json_sem_chaves_proibidas")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.MAX_JSON_FILE_SIZE_MB", 1)
    def test_arquivo_muito_grande_levanta_erro(self, mock_dir, mock_valida):
        """Arquivo maior que MAX_JSON_FILE_SIZE_MB levanta RAGSecurityError."""
        mock_path = MagicMock()
        mock_path.name = "grande.json"
        mock_path.stat.return_value.st_size = 2 * 1024 * 1024  # 2MB
        mock_dir.glob.return_value = [mock_path]

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            _valida_arquivos_json()
        self.assertIn("excede", str(ctx.exception).lower())
        mock_valida.assert_not_called()


class TestCriaVectordb(unittest.TestCase):
    """Testes para cria_vectordb (com mocks para evitar dependências pesadas)."""

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_sem_documentos_retorna_cedo(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma
    ):
        """Quando não há documentos, cria_vectordb retorna sem criar Chroma."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []  # nenhum PDF

        cria_vectordb()

        mock_valida_path.assert_called_once()
        mock_valida_json.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        mock_chroma.from_documents.assert_not_called()

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_com_documentos_json_usa_jq_schema_e_separators(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """Com documentos JSON válidos: usa JQ_SCHEMA_CAMPOS_JSON e separators de ADR-007."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="DESCRIÇÃO: texto da descrição"),
            Document(page_content="CRITÉRIOS: texto dos critérios"),
        ]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []  # nenhum PDF

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs  # passthrough
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        # jq_schema compartilhado usado
        _, loader_kwargs = mock_loader.call_args
        self.assertEqual(
            loader_kwargs["loader_kwargs"]["jq_schema"],
            config.JQ_SCHEMA_CAMPOS_JSON,
        )

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

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_filtra_documento_json_com_valor_vazio(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """Document JSON com page_content vazio (chave com valor "") é descartado."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="DESCRIÇÃO: texto válido"),
            Document(page_content="CAMPO_VAZIO: "),  # .strip() == "CAMPO_VAZIO:" -- nao vazio!
            Document(page_content=""),  # totalmente vazio
            Document(page_content="   "),  # só whitespace
        ]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []

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

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_pdf_multipagina_um_document_por_pagina(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_extrair, mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """PDF multi-página: 1 Document por página, prefixo PÁGINA_N: ."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        pdf_path = MagicMock()
        pdf_path.name = "materiais_logistica.pdf"
        mock_dir.glob.return_value = [pdf_path]

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

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_pdf_todas_paginas_em_branco_loga_erro_sem_levantar(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_extrair, mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """PDF totalmente em branco: RAGSecurityError é capturada e logada, zero Documents."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="DESCRIÇÃO: texto válido")]
        mock_loader.return_value = mock_loader_instance

        pdf_path = MagicMock()
        pdf_path.name = "vazio.pdf"
        mock_dir.glob.return_value = [pdf_path]

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


class TestGetRagEmbeddingModel(unittest.TestCase):
    """Testes para _get_rag_embedding_model (PORTPT-02 / AC2)."""

    def setUp(self) -> None:
        """Reseta o singleton antes de cada teste para garantir isolamento."""
        rag._rag_embedding_model = None

    def tearDown(self) -> None:
        """Reseta o singleton após cada teste."""
        rag._rag_embedding_model = None

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    def test_get_rag_embedding_model_usa_embedding_model_do_config(self, mock_emb):
        """_get_rag_embedding_model() constrói HuggingFaceEmbeddings com model_name=config.EMBEDDING_MODEL."""
        rag._get_rag_embedding_model()

        mock_emb.assert_called_once_with(model_name=config.EMBEDDING_MODEL)

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    def test_get_rag_embedding_model_singleton_reusa_instancia(self, mock_emb):
        """Chamadas subsequentes retornam a mesma instância sem recriar HuggingFaceEmbeddings."""
        primeira = rag._get_rag_embedding_model()
        segunda = rag._get_rag_embedding_model()

        self.assertIs(primeira, segunda)
        mock_emb.assert_called_once_with(model_name=config.EMBEDDING_MODEL)


class TestLogging(unittest.TestCase):
    """Testes para o módulo de logging em rag.py (LG-01 a LG-11)."""

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_1_log_info_gerando_embeddings(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """assertLogs INFO captura registro contendo 'Gerando' ao executar cria_vectordb (AC-04)."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="doc")]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level=logging.INFO) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Gerando" in msg for msg in cm.output),
            f"Esperava 'Gerando' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_2_log_info_criado_com_sucesso(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """assertLogs INFO captura registro contendo 'Criado' ao finalizar cria_vectordb (AC-04)."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="doc")]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []

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
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_3_log_warning_nenhum_documento(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir
    ):
        """assertLogs WARNING captura registro com 'Nenhum documento' quando loader retorna [] (AC-07)."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []

        with self.assertLogs("agenticlog.rag", level=logging.WARNING) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Nenhum documento" in msg for msg in cm.output),
            f"Esperava 'Nenhum documento' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.rag._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_4_sem_stdout_quando_importado_como_biblioteca(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir,
        mock_splitter, mock_emb, mock_chroma, mock_hash
    ):
        """Nenhuma saída em stdout quando cria_vectordb() chamada como biblioteca (AC-01)."""
        import io
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="doc")]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []

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
            any("Erro ao criar banco vetorial" in msg for msg in cm.output),
            f"Esperava 'Erro ao criar banco vetorial' nos logs, encontrado: {cm.output}",
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


class TestSanitizarNomeArquivo(unittest.TestCase):
    """Testes para _sanitizar_nome_arquivo."""

    def teste_1_sanitizar_nome_valido(self):
        """Nome de arquivo válido é retornado sem alteração."""
        resultado = _sanitizar_nome_arquivo("doc.json")
        self.assertEqual(resultado, "doc.json")

    def teste_2_sanitizar_rejeita_path_traversal(self):
        """Nome com '../' levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            _sanitizar_nome_arquivo("../evil.json")

    def teste_3_sanitizar_rejeita_chars_invalidos(self):
        """Nome com caracteres inválidos do Windows levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            _sanitizar_nome_arquivo("file<>.json")

    def teste_4_sanitizar_rejeita_nome_vazio(self):
        """Nome vazio levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            _sanitizar_nome_arquivo("")

    def teste_5_sanitizar_rejeita_nomes_reservados_windows(self):
        """Nomes reservados do Windows levantam RAGSecurityError."""
        reserved = ["CON.json", "PRN.json", "AUX.json", "NUL.json", "COM1.json", "LPT9.json"]
        for name in reserved:
            with self.subTest(name=name):
                with self.assertRaises(rag.RAGSecurityError):
                    _sanitizar_nome_arquivo(name)


class TestSanitizarNomeColecao(unittest.TestCase):
    """Testes para _sanitizar_nome_colecao (MCC-06 a MCC-11)."""

    def _sanitizar(self, name: str) -> str:
        """Chama _sanitizar_nome_colecao via módulo para garantir identidade de classe após reload."""
        return rag._sanitizar_nome_colecao(name)

    def teste_1_nome_vazio_levanta_erro(self) -> None:
        """String vazia levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            self._sanitizar("")
        self.assertIn("vazio", str(ctx.exception))

    def teste_2_nome_muito_curto_dois_chars_levanta_erro(self) -> None:
        """Nome com 2 caracteres levanta RAGSecurityError (mínimo é 3)."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            self._sanitizar("ab")
        self.assertIn("curto", str(ctx.exception))

    def teste_3_nome_exatamente_3_chars_valido(self) -> None:
        """Nome com exatamente 3 caracteres é aceito (fronteira válida)."""
        resultado = self._sanitizar("abc")
        self.assertEqual(resultado, "abc")

    def teste_4_nome_exatamente_63_chars_valido(self) -> None:
        """Nome com exatamente 63 caracteres é aceito (fronteira válida)."""
        nome = "a" * 63
        resultado = self._sanitizar(nome)
        self.assertEqual(resultado, nome)

    def teste_5_nome_64_chars_levanta_erro(self) -> None:
        """Nome com 64 caracteres levanta RAGSecurityError (máximo é 63)."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            self._sanitizar("a" * 64)
        self.assertIn("longo", str(ctx.exception))

    def teste_6_nome_com_espaco_levanta_erro(self) -> None:
        """Nome com espaço levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            self._sanitizar("nome colecao")

    def teste_7_nome_comecando_com_hifen_levanta_erro(self) -> None:
        """Nome iniciando com hífen levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            self._sanitizar("-inicio")

    def teste_8_nome_terminando_com_hifen_levanta_erro(self) -> None:
        """Nome terminando com hífen levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            self._sanitizar("fim-")

    def teste_9_nome_valido_com_hifen_e_underscore(self) -> None:
        """Nome com hífen e underscore internos é aceito."""
        resultado = self._sanitizar("valido-nome_1")
        self.assertEqual(resultado, "valido-nome_1")

    def teste_10_nome_logistica_default_valido(self) -> None:
        """DEFAULT_COLLECTION_NAME='logistica' passa validação."""
        from agenticlog.config import DEFAULT_COLLECTION_NAME
        resultado = self._sanitizar(DEFAULT_COLLECTION_NAME)
        self.assertEqual(resultado, DEFAULT_COLLECTION_NAME)


class TestSalvarDocumentoEnviado(unittest.TestCase):
    """Testes para salvar_documento_enviado."""

    def _valid_json_bytes(self) -> bytes:
        return json.dumps({"conteudo": "test"}).encode()

    def teste_1_salvar_documento_enviado_sucesso(self):
        """Arquivo JSON válido é salvo em DIR_DOCUMENTS."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                result = salvar_documento_enviado("doc.json", self._valid_json_bytes())
            self.assertTrue((tmp_path / "doc.json").exists())
            self.assertEqual(result, tmp_path / "doc.json")

    def teste_2_salvar_rejeita_extensao_invalida(self):
        """Extensão não-.json levanta RAGSecurityError antes de qualquer escrita."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            salvar_documento_enviado("dados.csv", self._valid_json_bytes())
        self.assertIn(".json", str(ctx.exception))

    def teste_3_salvar_rejeita_tamanho_excedido(self):
        """Arquivo maior que 10 MB levanta RAGSecurityError."""
        conteudo_grande = b"x" * (10 * 1024 * 1024 + 1)
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            salvar_documento_enviado("grande.json", conteudo_grande)
        self.assertIn("10", str(ctx.exception))

    def teste_4_salvar_rejeita_chave_proibida(self):
        """JSON com chave proibida 'lc' levanta RAGSecurityError."""
        conteudo = json.dumps({"lc": "bad"}).encode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(rag.RAGSecurityError):
                    salvar_documento_enviado("malicioso.json", conteudo)

    def teste_5_salvar_rejeita_colisao_de_nome(self):
        """Arquivo com nome já existente levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "existente.json").write_bytes(b"{}")
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(rag.RAGSecurityError) as ctx:
                    salvar_documento_enviado("existente.json", self._valid_json_bytes())
            self.assertIn("já existe", str(ctx.exception))

    def teste_6_salvar_rejeita_path_traversal(self):
        """Nome de arquivo com path traversal levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            salvar_documento_enviado("../evil.json", self._valid_json_bytes())

    def teste_7_salvar_rejeita_limite_de_arquivos(self):
        """Quando já há 1000 arquivos .json, levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i in range(1000):
                (tmp_path / f"arquivo_{i:04d}.json").write_bytes(b"{}")
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(rag.RAGSecurityError) as ctx:
                    salvar_documento_enviado("novo.json", self._valid_json_bytes())
            self.assertIn("1000", str(ctx.exception))


class TestReconstruirVectordb(unittest.TestCase):
    """Testes para reconstruir_vectordb."""

    def teste_1_reconstruir_vectordb_chama_cria_vectordb(self):
        """reconstruir_vectordb() chama cria_vectordb() exatamente uma vez."""
        with patch("agenticlog.rag.cria_vectordb") as mock_cria:
            reconstruir_vectordb()
        mock_cria.assert_called_once()

    def teste_2_reconstruir_vectordb_propaga_excecao(self):
        """Exceção lançada por cria_vectordb() é propagada pelo reconstruir_vectordb()."""
        with patch("agenticlog.rag.cria_vectordb", side_effect=Exception("fail")):
            with self.assertRaises(Exception) as ctx:
                reconstruir_vectordb()
        self.assertEqual(str(ctx.exception), "fail")


class TestExtrairTextoPdf(unittest.TestCase):
    """Testes para extrair_texto_pdf."""

    @patch("agenticlog.rag.fitz.open")
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

    @patch("agenticlog.rag.fitz.open")
    def teste_2_extrair_pdf_com_senha_lanca_erro(self, mock_fitz_open):
        """PDF com senha lança RAGSecurityError."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.rag.fitz.open")
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

    @patch("agenticlog.rag.fitz.open")
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

    @patch("agenticlog.rag.fitz.open")
    def teste_5_extrair_exception_generica_lanca_erro(self, mock_fitz_open):
        """fitz.open() lançando Exception genérica é convertida em RAGSecurityError."""
        mock_fitz_open.side_effect = RuntimeError("unexpected fitz error")

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("corrompido", str(ctx.exception))

    @patch("agenticlog.rag.fitz.open")
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


class TestSalvarPdfEnviado(unittest.TestCase):
    """Testes para salvar_pdf_enviado."""

    def _valid_pdf_bytes(self) -> bytes:
        return b"%PDF-1.4 fake content"

    @patch("agenticlog.rag.extrair_texto_pdf")
    def teste_1_salvar_pdf_valido_sucesso(self, mock_extrair):
        """PDF válido é salvo em DIR_DOCUMENTS."""
        mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                result = salvar_pdf_enviado("contrato.pdf", self._valid_pdf_bytes())
            self.assertTrue((tmp_path / "contrato.pdf").exists())
            self.assertEqual(result, tmp_path / "contrato.pdf")

    def teste_2_salvar_rejeita_extensao_invalida(self):
        """Extensão .txt levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            salvar_pdf_enviado("documento.txt", self._valid_pdf_bytes())
        self.assertIn(".pdf", str(ctx.exception))

    def teste_2b_salvar_rejeita_magic_bytes_invalidos(self):
        """Conteúdo sem magic bytes %PDF levanta RAGSecurityError antes de escrita em disco."""
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            salvar_pdf_enviado("fake.pdf", b"PK\x03\x04 not a pdf")
        self.assertIn("PDF válido", str(ctx.exception))

    @patch("agenticlog.rag.extrair_texto_pdf")
    def teste_3_salvar_aceita_extensao_maiuscula(self, mock_extrair):
        """Extensão .PDF (maiúscula) é aceita (case-insensitive)."""
        mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                result = salvar_pdf_enviado("CONTRATO.PDF", self._valid_pdf_bytes())
            self.assertTrue((tmp_path / "CONTRATO.PDF").exists())

    def teste_4_salvar_rejeita_tamanho_excedido(self):
        """Conteúdo maior que 10 MB levanta RAGSecurityError antes de extração."""
        conteudo_grande = b"%PDF" + b"x" * (10 * 1024 * 1024 + 1)
        with self.assertRaises(rag.RAGSecurityError) as ctx:
            salvar_pdf_enviado("grande.pdf", conteudo_grande)
        self.assertIn("10", str(ctx.exception))

    def teste_5_salvar_rejeita_nome_duplicado(self):
        """Arquivo com nome já existente levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "existente.pdf").write_bytes(b"%PDF")
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(rag.RAGSecurityError) as ctx:
                    salvar_pdf_enviado("existente.pdf", self._valid_pdf_bytes())
            self.assertIn("já existe", str(ctx.exception))

    def teste_6_salvar_rejeita_path_traversal(self):
        """Nome com path traversal levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            salvar_pdf_enviado("../evil.pdf", self._valid_pdf_bytes())

    def teste_7_salvar_rejeita_nome_reservado_windows(self):
        """Nome reservado Windows levanta RAGSecurityError."""
        with self.assertRaises(rag.RAGSecurityError):
            salvar_pdf_enviado("CON.pdf", self._valid_pdf_bytes())

    @patch("agenticlog.rag.extrair_texto_pdf")
    def teste_8_salvar_rollback_se_pdf_invalido(self, mock_extrair):
        """Se extrair_texto_pdf lança RAGSecurityError, tempfile é deletado e erro é relançado."""
        mock_extrair.side_effect = RAGSecurityError("PDF protegido por senha.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_pdf_enviado("invalido.pdf", self._valid_pdf_bytes())
            self.assertNotIn("invalido.pdf", [f.name for f in tmp_path.iterdir()])
        # Verifica que o tempfile passado para extrair_texto_pdf foi deletado
        called_path = mock_extrair.call_args[0][0]
        self.assertFalse(called_path.exists(), "Tempfile não foi deletado no rollback")
        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.rag.MAX_JSON_FILES", new=2)
    def teste_9_salvar_rejeita_limite_de_arquivos(self):
        """pdf_count + json_count + 1 > MAX_JSON_FILES levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "doc_1.pdf").write_bytes(b"%PDF")
            (tmp_path / "doc_2.json").write_bytes(b"{}")
            with patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(rag.RAGSecurityError) as ctx:
                    salvar_pdf_enviado("novo.pdf", self._valid_pdf_bytes())
        self.assertIn("Limite", str(ctx.exception))


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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
            patch("agenticlog.rag.Chroma", return_value=mock_vdb),
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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

    @patch("agenticlog.agent.invalidar_vector_db")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.extrair_texto_pdf") as mock_extrair,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.extrair_texto_pdf") as mock_extrair,
                patch("agenticlog.rag.SemanticChunker") as mock_splitter_cls,
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

    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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

        with patch("agenticlog.rag.extrair_texto_pdf", side_effect=rag.RAGSecurityError("PDF protegido por senha.")):
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                adicionar_pdf_incrementalmente("protegido.pdf", conteudo)

        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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
            "agenticlog.rag.extrair_texto_pdf",
            side_effect=rag.RAGSecurityError("PDF não contém texto extraível (somente imagem)."),
        ):
            with self.assertRaises(rag.RAGSecurityError) as ctx:
                adicionar_pdf_incrementalmente("scan.pdf", conteudo)

        self.assertIn("somente imagem", str(ctx.exception))

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.SemanticChunker")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
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

            with patch.object(rag, "DIR_DOCUMENTS", tmp), \
                 patch.object(rag, "adicionar_documento_incrementalmente", mock_json), \
                 patch.object(rag, "adicionar_pdf_incrementalmente", mock_pdf):
                contadores = ingerir_incrementalmente()

        self.assertEqual(contadores, {"adicionado": 2, "duplicado": 1})
        self.assertEqual(mock_json.call_count, 2)
        mock_pdf.assert_called_once()
        # nome do arquivo é passado, não o caminho completo
        self.assertEqual(mock_pdf.call_args[0][0], "c.pdf")

    def teste_2_arquivo_com_erro_nao_aborta_lote(self):
        """Falha em um arquivo é contada como 'erro' e os demais seguem sendo processados."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._criar_docs(tmp, ["ok.json", "ruim.json"])

            def _side(nome, conteudo, collection_name):
                if nome == "ruim.json":
                    raise rag.RAGSecurityError("falha simulada")
                return {"status": "adicionado", "mensagem": "ok"}

            with patch.object(rag, "DIR_DOCUMENTS", tmp), \
                 patch.object(rag, "adicionar_documento_incrementalmente", side_effect=_side), \
                 patch.object(rag, "adicionar_pdf_incrementalmente", MagicMock()):
                contadores = ingerir_incrementalmente()

        self.assertEqual(contadores, {"adicionado": 1, "erro": 1})

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
