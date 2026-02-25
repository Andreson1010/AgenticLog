# AgenticLog - Testes unitários para rag.py
"""
Testes para o pipeline RAG em agenticlog.rag.
Cobre validações de segurança, path traversal e criação do banco vetorial.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest

from agenticlog.rag import (
    RAGSecurityError,
    _valida_path_documentos,
    _valida_json_sem_chaves_proibidas,
    _valida_arquivos_json,
    cria_vectordb,
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

        with self.assertRaises(RAGSecurityError) as ctx:
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

        with self.assertRaises(RAGSecurityError) as ctx:
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

        with self.assertRaises(RAGSecurityError) as ctx:
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
            with self.assertRaises(RAGSecurityError) as ctx:
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
            with self.assertRaises(RAGSecurityError) as ctx:
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
            with self.assertRaises(RAGSecurityError) as ctx:
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
        with self.assertRaises(RAGSecurityError) as ctx:
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

        with self.assertRaises(RAGSecurityError) as ctx:
            _valida_arquivos_json()
        self.assertIn("excede", str(ctx.exception).lower())
        mock_valida.assert_not_called()


class TestCriaVectordb(unittest.TestCase):
    """Testes para cria_vectordb (com mocks para evitar dependências pesadas)."""

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_sem_documentos_retorna_cedo(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_splitter, mock_emb, mock_chroma
    ):
        """Quando não há documentos, cria_vectordb retorna sem criar Chroma."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        cria_vectordb()

        mock_valida_path.assert_called_once()
        mock_valida_json.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        mock_chroma.from_documents.assert_not_called()

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_com_documentos_cria_chroma(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_splitter, mock_emb, mock_chroma
    ):
        """Com documentos válidos, cria_vectordb chama Chroma.from_documents."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Conteúdo de teste"),
        ]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [
            Document(page_content="Chunk 1"),
        ]
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        mock_chroma.from_documents.assert_called_once()
        call_args = mock_chroma.from_documents.call_args
        self.assertEqual(len(call_args[0][0]), 1)
        self.assertEqual(call_args[0][0][0].page_content, "Chunk 1")


if __name__ == "__main__":
    print("\nIniciando testes do RAG. Aguarde...\n")
    unittest.main(verbosity=2)
