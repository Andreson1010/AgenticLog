# AgenticLog - Testes unitários para rag.py
"""
Testes para o pipeline RAG em agenticlog.rag.
Cobre validações de segurança, path traversal e criação do banco vetorial.
"""

import json
import logging
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
import pytest

from agenticlog.rag import (
    RAGSecurityError,
    _valida_path_documentos,
    _valida_json_sem_chaves_proibidas,
    _valida_arquivos_json,
    cria_vectordb,
)
import agenticlog.rag as rag
import agenticlog.config as config


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


class TestLogging(unittest.TestCase):
    """Testes para o módulo de logging em rag.py (LG-01 a LG-11)."""

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_1_log_info_gerando_embeddings(
        self, mock_valida_path, mock_valida_json, mock_loader,
        mock_splitter, mock_emb, mock_chroma
    ):
        """assertLogs INFO captura registro contendo 'Gerando' ao executar cria_vectordb (AC-04)."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="doc")]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level=logging.INFO) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Gerando" in msg for msg in cm.output),
            f"Esperava 'Gerando' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_2_log_info_criado_com_sucesso(
        self, mock_valida_path, mock_valida_json, mock_loader,
        mock_splitter, mock_emb, mock_chroma
    ):
        """assertLogs INFO captura registro contendo 'Criado' ao finalizar cria_vectordb (AC-04)."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="doc")]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level=logging.INFO) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Criado" in msg for msg in cm.output),
            f"Esperava 'Criado' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_3_log_warning_nenhum_documento(
        self, mock_valida_path, mock_valida_json, mock_loader
    ):
        """assertLogs WARNING captura registro com 'Nenhum documento' quando loader retorna [] (AC-07)."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        with self.assertLogs("agenticlog.rag", level=logging.WARNING) as cm:
            cria_vectordb()

        self.assertTrue(
            any("Nenhum documento" in msg for msg in cm.output),
            f"Esperava 'Nenhum documento' nos logs, encontrado: {cm.output}",
        )

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def teste_4_sem_stdout_quando_importado_como_biblioteca(
        self, mock_valida_path, mock_valida_json, mock_loader,
        mock_splitter, mock_emb, mock_chroma
    ):
        """Nenhuma saída em stdout quando cria_vectordb() chamada como biblioteca (AC-01)."""
        import io
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="doc")]
        mock_loader.return_value = mock_loader_instance

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [Document(page_content="chunk")]
        mock_splitter.return_value = mock_splitter_instance

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            cria_vectordb()

        output = captured_stdout.getvalue()
        self.assertEqual(output, "", f"Esperava stdout vazio, encontrado: {repr(output)}")

    def teste_5_log_level_em_config(self):
        """config.LOG_LEVEL é 'INFO' e corresponde ao nível numérico 20 (AC-03)."""
        self.assertEqual(config.LOG_LEVEL, "INFO")
        self.assertEqual(logging.getLevelName("INFO"), 20)

    def teste_6_logger_modulo_usa_dunder_name(self):
        """logger em rag.py tem name == 'agenticlog.rag' (AC-08)."""
        self.assertEqual(rag.logger.name, "agenticlog.rag")

    @staticmethod
    def _exec_rag_main_block(side_effect: Exception) -> None:
        """Executa o corpo do bloco __main__ de rag.py no namespace do módulo já importado.

        Estratégia:
        1. Lê rag.py e extrai o corpo do bloco `if __name__ == "__main__":` via ast.
        2. Aplica patch.object em rag.cria_vectordb antes do exec.
        3. Executa o corpo compilado com o namespace do módulo (vars(rag)) e o arquivo
           real como filename, para que o coverage instrumente as linhas corretas.

        Isso garante que as linhas 182-193 de rag.py sejam exercitadas com mock ativo
        e que o coverage as contabilize corretamente (exec in-process).
        """
        import ast

        rag_path = Path(__file__).resolve().parent.parent / "src" / "agenticlog" / "rag.py"
        with open(rag_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(rag_path))

        main_if = next(
            n for n in tree.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "__name__"
        )
        main_module = ast.Module(body=main_if.body, type_ignores=[])
        ast.fix_missing_locations(main_module)
        code = compile(main_module, filename=str(rag_path), mode="exec")

        with patch.object(rag, "cria_vectordb", side_effect=side_effect):
            exec(code, vars(rag))  # noqa: S102

    def teste_7_erro_seguranca_usa_logger_error(self):
        """RAGSecurityError no bloco __main__ aciona logger.error com 'Erro de segurança' (AC-05).

        Executa o corpo real do bloco __main__ de rag.py (linhas 183-193) via exec in-process
        com cria_vectordb mockada para lancar RAGSecurityError. Verifica SystemExit(1) e log.
        """
        with self.assertLogs("agenticlog.rag", level="ERROR") as cm:
            with self.assertRaises(SystemExit) as ctx:
                self._exec_rag_main_block(RAGSecurityError("falha de segurança simulada"))

        self.assertEqual(ctx.exception.code, 1)
        self.assertTrue(
            any("Erro de segurança" in msg for msg in cm.output),
            f"Esperava 'Erro de segurança' nos logs, encontrado: {cm.output}",
        )

    def teste_8_excecao_generica_usa_logger_error(self):
        """Exception generica no bloco __main__ aciona logger.error com 'Erro ao criar banco vetorial' (AC-06).

        Executa o corpo real do bloco __main__ de rag.py (linhas 183-193) via exec in-process
        com cria_vectordb mockada para lancar Exception generica. Verifica SystemExit(1) e log.
        """
        with self.assertLogs("agenticlog.rag", level="ERROR") as cm:
            with self.assertRaises(SystemExit) as ctx:
                self._exec_rag_main_block(RuntimeError("erro generico simulado"))

        self.assertEqual(ctx.exception.code, 1)
        self.assertTrue(
            any("Erro ao criar banco vetorial" in msg for msg in cm.output),
            f"Esperava 'Erro ao criar banco vetorial' nos logs, encontrado: {cm.output}",
        )


if __name__ == "__main__":
    print("\nIniciando testes do RAG. Aguarde...\n")
    unittest.main(verbosity=2)
