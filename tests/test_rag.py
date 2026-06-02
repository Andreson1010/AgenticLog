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
    _executar_main,
    _sanitizar_nome_arquivo,
    salvar_documento_enviado,
    reconstruir_vectordb,
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
                    _executar_main()

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
                    _executar_main()

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
            rag._executar_main()

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


if __name__ == "__main__":
    print("\nIniciando testes do RAG. Aguarde...\n")
    unittest.main(verbosity=2)
