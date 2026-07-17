# AgenticLog - Testes unitários para ingestion/security.py
"""Testes do estágio de segurança da ingestão (ADR-018 Fase 3a).

Movidos de tests/test_rag.py; alvos de @patch repontados para
`agenticlog.ingestion.security.*` onde o nome é lido no corpo movido.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.ingestion.security as rag  # noqa: E402
from agenticlog.ingestion.security import (  # noqa: E402
    _sanitizar_nome_arquivo,
    _valida_arquivos_json,
    _valida_json_sem_chaves_proibidas,
    _valida_path_documentos,
    salvar_documento_enviado,
    salvar_pdf_enviado,
)
from agenticlog.shared.errors import RAGSecurityError  # noqa: E402


class TestValidaPathDocumentos(unittest.TestCase):
    """Testes para _valida_path_documentos."""

    @patch("agenticlog.ingestion.security.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.security.PROJECT_ROOT")
    def test_path_fora_do_projeto_levanta_erro(self, mock_root, mock_dir):
        """Path fora do PROJECT_ROOT levanta RAGSecurityError."""
        resolved_dir = MagicMock()
        resolved_dir.relative_to.side_effect = ValueError("fora")
        mock_dir.resolve.return_value = resolved_dir

        with self.assertRaises(RAGSecurityError) as ctx:
            _valida_path_documentos()
        self.assertIn("fora do projeto", str(ctx.exception))

    @patch("agenticlog.ingestion.security.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.security.PROJECT_ROOT")
    def test_diretorio_nao_existe_levanta_erro(self, mock_root, mock_dir):
        """Diretório inexistente levanta RAGSecurityError."""
        resolved_dir = MagicMock()
        resolved_dir.relative_to.return_value = Path("data/documents")
        resolved_dir.exists.return_value = False
        mock_dir.resolve.return_value = resolved_dir

        with self.assertRaises(RAGSecurityError) as ctx:
            _valida_path_documentos()
        self.assertIn("não existe", str(ctx.exception))

    @patch("agenticlog.ingestion.security.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.security.PROJECT_ROOT")
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

    @patch("agenticlog.ingestion.security.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.security.PROJECT_ROOT")
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

    @patch("agenticlog.ingestion.security._valida_json_sem_chaves_proibidas")
    @patch("agenticlog.ingestion.security.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.security.MAX_JSON_FILES", 2)
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

    @patch("agenticlog.ingestion.security._valida_json_sem_chaves_proibidas")
    @patch("agenticlog.ingestion.security.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.security.MAX_JSON_FILE_SIZE_MB", 1)
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


class TestSanitizarNomeArquivo(unittest.TestCase):
    """Testes para _sanitizar_nome_arquivo."""

    def teste_1_sanitizar_nome_valido(self):
        """Nome de arquivo válido é retornado sem alteração."""
        resultado = _sanitizar_nome_arquivo("doc.json")
        self.assertEqual(resultado, "doc.json")

    def teste_2_sanitizar_rejeita_path_traversal(self):
        """Nome com '../' levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("../evil.json")

    def teste_3_sanitizar_rejeita_chars_invalidos(self):
        """Nome com caracteres inválidos do Windows levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("file<>.json")

    def teste_4_sanitizar_rejeita_nome_vazio(self):
        """Nome vazio levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("")

    def teste_5_sanitizar_rejeita_nomes_reservados_windows(self):
        """Nomes reservados do Windows levantam RAGSecurityError."""
        reserved = ["CON.json", "PRN.json", "AUX.json", "NUL.json", "COM1.json", "LPT9.json"]
        for name in reserved:
            with self.subTest(name=name):
                with self.assertRaises(RAGSecurityError):
                    _sanitizar_nome_arquivo(name)


class TestSanitizarNomeColecao(unittest.TestCase):
    """Testes para _sanitizar_nome_colecao (MCC-06 a MCC-11)."""

    def _sanitizar(self, name: str) -> str:
        """Chama _sanitizar_nome_colecao via módulo para garantir identidade de classe após reload."""
        return rag._sanitizar_nome_colecao(name)

    def teste_1_nome_vazio_levanta_erro(self) -> None:
        """String vazia levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError) as ctx:
            self._sanitizar("")
        self.assertIn("vazio", str(ctx.exception))

    def teste_2_nome_muito_curto_dois_chars_levanta_erro(self) -> None:
        """Nome com 2 caracteres levanta RAGSecurityError (mínimo é 3)."""
        with self.assertRaises(RAGSecurityError) as ctx:
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
        with self.assertRaises(RAGSecurityError) as ctx:
            self._sanitizar("a" * 64)
        self.assertIn("longo", str(ctx.exception))

    def teste_6_nome_com_espaco_levanta_erro(self) -> None:
        """Nome com espaço levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            self._sanitizar("nome colecao")

    def teste_7_nome_comecando_com_hifen_levanta_erro(self) -> None:
        """Nome iniciando com hífen levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            self._sanitizar("-inicio")

    def teste_8_nome_terminando_com_hifen_levanta_erro(self) -> None:
        """Nome terminando com hífen levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
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
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                result = salvar_documento_enviado("doc.json", self._valid_json_bytes())
            self.assertTrue((tmp_path / "doc.json").exists())
            self.assertEqual(result, tmp_path / "doc.json")

    def teste_2_salvar_rejeita_extensao_invalida(self):
        """Extensão não-.json levanta RAGSecurityError antes de qualquer escrita."""
        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_documento_enviado("dados.csv", self._valid_json_bytes())
        self.assertIn(".json", str(ctx.exception))

    def teste_3_salvar_rejeita_tamanho_excedido(self):
        """Arquivo maior que 10 MB levanta RAGSecurityError."""
        conteudo_grande = b"x" * (10 * 1024 * 1024 + 1)
        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_documento_enviado("grande.json", conteudo_grande)
        self.assertIn("10", str(ctx.exception))

    def teste_4_salvar_rejeita_chave_proibida(self):
        """JSON com chave proibida 'lc' levanta RAGSecurityError."""
        conteudo = json.dumps({"lc": "bad"}).encode()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError):
                    salvar_documento_enviado("malicioso.json", conteudo)

    def teste_5_salvar_rejeita_colisao_de_nome(self):
        """Arquivo com nome já existente levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "existente.json").write_bytes(b"{}")
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_documento_enviado("existente.json", self._valid_json_bytes())
            self.assertIn("já existe", str(ctx.exception))

    def teste_6_salvar_rejeita_path_traversal(self):
        """Nome de arquivo com path traversal levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            salvar_documento_enviado("../evil.json", self._valid_json_bytes())

    def teste_7_salvar_rejeita_limite_de_arquivos(self):
        """Quando já há 1000 arquivos .json, levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i in range(1000):
                (tmp_path / f"arquivo_{i:04d}.json").write_bytes(b"{}")
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_documento_enviado("novo.json", self._valid_json_bytes())
            self.assertIn("1000", str(ctx.exception))


class TestSalvarPdfEnviado(unittest.TestCase):
    """Testes para salvar_pdf_enviado."""

    def _valid_pdf_bytes(self) -> bytes:
        return b"%PDF-1.4 fake content"

    @patch("agenticlog.ingestion.security.extrair_texto_pdf")
    def teste_1_salvar_pdf_valido_sucesso(self, mock_extrair):
        """PDF válido é salvo em DIR_DOCUMENTS."""
        mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                result = salvar_pdf_enviado("contrato.pdf", self._valid_pdf_bytes())
            self.assertTrue((tmp_path / "contrato.pdf").exists())
            self.assertEqual(result, tmp_path / "contrato.pdf")

    def teste_2_salvar_rejeita_extensao_invalida(self):
        """Extensão .txt levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_pdf_enviado("documento.txt", self._valid_pdf_bytes())
        self.assertIn(".pdf", str(ctx.exception))

    def teste_2b_salvar_rejeita_magic_bytes_invalidos(self):
        """Conteúdo sem magic bytes %PDF levanta RAGSecurityError antes de escrita em disco."""
        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_pdf_enviado("fake.pdf", b"PK\x03\x04 not a pdf")
        self.assertIn("PDF válido", str(ctx.exception))

    @patch("agenticlog.ingestion.security.extrair_texto_pdf")
    def teste_3_salvar_aceita_extensao_maiuscula(self, mock_extrair):
        """Extensão .PDF (maiúscula) é aceita (case-insensitive)."""
        mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                salvar_pdf_enviado("CONTRATO.PDF", self._valid_pdf_bytes())
            self.assertTrue((tmp_path / "CONTRATO.PDF").exists())

    def teste_4_salvar_rejeita_tamanho_excedido(self):
        """Conteúdo maior que 10 MB levanta RAGSecurityError antes de extração."""
        conteudo_grande = b"%PDF" + b"x" * (10 * 1024 * 1024 + 1)
        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_pdf_enviado("grande.pdf", conteudo_grande)
        self.assertIn("10", str(ctx.exception))

    def teste_5_salvar_rejeita_nome_duplicado(self):
        """Arquivo com nome já existente levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "existente.pdf").write_bytes(b"%PDF")
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_pdf_enviado("existente.pdf", self._valid_pdf_bytes())
            self.assertIn("já existe", str(ctx.exception))

    def teste_6_salvar_rejeita_path_traversal(self):
        """Nome com path traversal levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            salvar_pdf_enviado("../evil.pdf", self._valid_pdf_bytes())

    def teste_7_salvar_rejeita_nome_reservado_windows(self):
        """Nome reservado Windows levanta RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            salvar_pdf_enviado("CON.pdf", self._valid_pdf_bytes())

    @patch("agenticlog.ingestion.security.extrair_texto_pdf")
    def teste_8_salvar_rollback_se_pdf_invalido(self, mock_extrair):
        """Se extrair_texto_pdf lança RAGSecurityError, tempfile é deletado e erro é relançado."""
        mock_extrair.side_effect = RAGSecurityError("PDF protegido por senha.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_pdf_enviado("invalido.pdf", self._valid_pdf_bytes())
            self.assertNotIn("invalido.pdf", [f.name for f in tmp_path.iterdir()])
        # Verifica que o tempfile passado para extrair_texto_pdf foi deletado
        called_path = mock_extrair.call_args[0][0]
        self.assertFalse(called_path.exists(), "Tempfile não foi deletado no rollback")
        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.ingestion.security.MAX_JSON_FILES", new=2)
    def teste_9_salvar_rejeita_limite_de_arquivos(self):
        """pdf_count + json_count + 1 > MAX_JSON_FILES levanta RAGSecurityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "doc_1.pdf").write_bytes(b"%PDF")
            (tmp_path / "doc_2.json").write_bytes(b"{}")
            with patch("agenticlog.ingestion.security.DIR_DOCUMENTS", new=tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_pdf_enviado("novo.pdf", self._valid_pdf_bytes())
        self.assertIn("Limite", str(ctx.exception))
