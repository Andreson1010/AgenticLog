# AgenticLog — Acceptance Tests: Document Ingestion UI Feature
"""
Verifica todos os critérios de aceite das stories:
  "Como operador logístico, quero fazer upload de um documento JSON pela
   barra lateral do Streamlit para que o ChromaDB seja reconstruído com o
   novo conteúdo sem reiniciar a aplicação."

  "Como operador de logística, quero fazer upload de arquivos PDF diretamente
   pela interface Streamlit para indexação no ChromaDB."

Mapeamento de critérios (DOCING-01 a DOCING-10):
  DOCING-01 — AC-1: happy path (salvar + rebuild + rerun)
  DOCING-02 — AC-2: spinner visível durante rebuild
  DOCING-03 — AC-3: extensão não-.json rejeitada antes de escrita
  DOCING-04 — AC-4: arquivo > 10 MB rejeitado antes de escrita
  DOCING-05 — AC-5: chave proibida "lc" rejeitada antes de escrita
  DOCING-06 — AC-6: colisão de nome rejeitada, zero bytes escritos
  DOCING-07 — AC-7: path traversal / chars inválidos rejeitados
  DOCING-08 — AC-8: limite de 1000 arquivos rejeitado
  DOCING-09 — AC-9: rollback ao falhar rebuild (arquivo removido)
  DOCING-10 — AC-10: verificado por design (st.rerun() aciona re-import do agente)

Mapeamento de critérios PDF (PDF-01 a PDF-11):
  PDF-01 — AC1: happy path PDF (salvar_pdf_enviado + rebuild + rerun)
  PDF-03 — AC3: PDF com senha → RAGSecurityError, st.error, sem rerun
  PDF-04 — AC4: PDF somente-imagem → RAGSecurityError, st.error
  PDF-05 — AC5: PDF > 10 MB → RAGSecurityError, st.error
  PDF-06 — AC6: nome duplicado → RAGSecurityError, st.error
  PDF-08 — AC8: rollback ao falhar rebuild (arquivo removido, st.error)
  PDF-09 — AC9: extensão não-.pdf → RAGSecurityError, st.error
  PDF-10 — AC10: extensão .PDF maiúscula aceita (case-insensitive)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock, call

import agenticlog.rag as _rag_module
from agenticlog.rag import RAGSecurityError


# ---------------------------------------------------------------------------
# Helper — cria um UploadedFile fake do Streamlit
# ---------------------------------------------------------------------------

def _make_uploaded_file(name: str, content: bytes) -> MagicMock:
    """Retorna um MagicMock com a interface mínima do UploadedFile do Streamlit."""
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.getvalue.return_value = content
    return mock_file


# ---------------------------------------------------------------------------
# DOCING-01: Happy path — salva arquivo, reconstrói DB, exibe sucesso, rerun
# ---------------------------------------------------------------------------

class TestDOCING01HappyPath(unittest.TestCase):
    """
    DOCING-01: WHEN operator uploads a valid .json file < 10 MB with no forbidden keys
    AND clicks "Ingerir Documento"
    THEN the file SHALL be saved to data/documents/,
         ChromaDB SHALL be rebuilt,
         a success message SHALL be shown,
         AND st.rerun() SHALL be called.
    """

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_documento_enviado")
    def test_docing_01_happy_path_saves_rebuilds_and_reruns(
        self,
        mock_salvar: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-01: valid upload → salvar + reconstruir chamados, st.success + st.rerun disparados."""
        saved_path = MagicMock(spec=Path)
        mock_salvar.return_value = saved_path
        mock_reconstruir.return_value = None

        # Configura spinner como context manager
        mock_spinner_ctx = MagicMock()
        mock_spinner_ctx.__enter__ = MagicMock(return_value=None)
        mock_spinner_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = mock_spinner_ctx

        uploaded_file = _make_uploaded_file("frete.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_salvar.assert_called_once_with("frete.json", b'{"tipo": "frete"}')
        mock_reconstruir.assert_called_once()
        mock_st.success.assert_called_once_with("Documento ingerido com sucesso.")
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()


# ---------------------------------------------------------------------------
# DOCING-02: Spinner visível durante rebuild
# ---------------------------------------------------------------------------

class TestDOCING02SpinnerDuringRebuild(unittest.TestCase):
    """
    DOCING-02: WHEN rebuild is running
    THEN a spinner with text "Reconstruindo base vetorial..." SHALL be visible.
    """

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_documento_enviado")
    def test_docing_02_spinner_wraps_rebuild(
        self,
        mock_salvar: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-02: st.spinner chamado com texto correto e envolve reconstruir_vectordb."""
        saved_path = MagicMock(spec=Path)
        mock_salvar.return_value = saved_path

        spinner_ctx = MagicMock()
        spinner_ctx.__enter__ = MagicMock(return_value=None)
        spinner_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = spinner_ctx

        uploaded_file = _make_uploaded_file("doc.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.spinner.assert_called_once_with("Reconstruindo base vetorial...")
        # reconstruir foi chamado dentro do bloco with spinner
        mock_reconstruir.assert_called_once()


# ---------------------------------------------------------------------------
# DOCING-03: Extensão não-.json rejeitada antes de escrita em disco
# ---------------------------------------------------------------------------

class TestDOCING03NonJsonExtensionRejected(unittest.TestCase):
    """
    DOCING-03: WHEN operator uploads a file with a non-.json extension
    THEN the system SHALL reject it before any disk write
    AND show "Apenas arquivos .json são aceitos."
    """

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_docing_03_non_json_extension_shows_error_no_disk_write(
        self,
        mock_salvar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-03: extensão .csv → RAGSecurityError capturada, st.error exibido, sem escrita."""
        mock_salvar.side_effect = RAGSecurityError("Apenas arquivos .json são aceitos.")

        uploaded_file = _make_uploaded_file("planilha.csv", b"col1,col2")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("Apenas arquivos .json são aceitos.")
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    def test_docing_03_salvar_rejeita_extensao_nao_json_diretamente(self) -> None:
        """DOCING-03: salvar_documento_enviado lança RAGSecurityError para extensão inválida."""
        from agenticlog.rag import salvar_documento_enviado

        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_documento_enviado("evil.txt", b"data")

        self.assertIn(".json", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# DOCING-04: Arquivo > 10 MB rejeitado antes de escrita em disco
# ---------------------------------------------------------------------------

class TestDOCING04FileSizeRejected(unittest.TestCase):
    """
    DOCING-04: WHEN operator uploads a file larger than 10 MB
    THEN the system SHALL reject it before any disk write
    AND show a message citing the 10 MB limit.
    """

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_docing_04_large_file_shows_error_no_disk_write(
        self,
        mock_salvar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-04: arquivo > 10 MB → RAGSecurityError capturada, st.error exibido."""
        mock_salvar.side_effect = RAGSecurityError("Arquivo excede o limite de 10 MB.")

        uploaded_file = _make_uploaded_file("grande.json", b"x" * (11 * 1024 * 1024))

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("Arquivo excede o limite de 10 MB.")
        mock_st.rerun.assert_not_called()

    def test_docing_04_salvar_rejeita_tamanho_excedido_diretamente(self) -> None:
        """DOCING-04: salvar_documento_enviado lança RAGSecurityError para conteúdo > 10 MB."""
        from agenticlog.rag import salvar_documento_enviado

        conteudo_grande = b"x" * (10 * 1024 * 1024 + 1)

        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_documento_enviado("grande.json", conteudo_grande)

        self.assertIn("10", str(ctx.exception))


# ---------------------------------------------------------------------------
# DOCING-05: Chave proibida "lc" rejeitada antes de escrita em disco
# ---------------------------------------------------------------------------

class TestDOCING05ForbiddenKeyRejected(unittest.TestCase):
    """
    DOCING-05: WHEN operator uploads a JSON file containing the forbidden key "lc"
    THEN the system SHALL reject it before any disk write
    AND show "Arquivo contém chave proibida."
    """

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_docing_05_forbidden_key_shows_error_no_disk_write(
        self,
        mock_salvar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-05: chave 'lc' → RAGSecurityError capturada, st.error exibido."""
        mock_salvar.side_effect = RAGSecurityError("Arquivo contém chave proibida 'lc'")

        uploaded_file = _make_uploaded_file("malicioso.json", b'{"lc": "evil"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("proibida", error_msg.lower())
        mock_st.rerun.assert_not_called()

    def test_docing_05_salvar_rejeita_chave_proibida_diretamente(self) -> None:
        """DOCING-05: salvar_documento_enviado lança RAGSecurityError para chave 'lc'."""
        import json
        import tempfile
        from agenticlog.rag import salvar_documento_enviado

        conteudo = json.dumps({"lc": "evil", "tipo": "frete"}).encode()

        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_documento_enviado("malicioso.json", conteudo)

        self.assertIn("proibida", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# DOCING-06: Colisão de nome de arquivo — zero bytes escritos
# ---------------------------------------------------------------------------

class TestDOCING06FilenameCollisionRejected(unittest.TestCase):
    """
    DOCING-06: WHEN the filename already exists in data/documents/
    THEN the system SHALL reject it, show an error, and write zero bytes.
    """

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_docing_06_collision_shows_error_no_disk_write(
        self,
        mock_salvar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-06: arquivo já existe → RAGSecurityError capturada, st.error exibido."""
        mock_salvar.side_effect = RAGSecurityError("Arquivo com esse nome já existe.")

        uploaded_file = _make_uploaded_file("doc1.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("Arquivo com esse nome já existe.")
        mock_st.rerun.assert_not_called()

    def test_docing_06_salvar_rejeita_colisao_diretamente(self) -> None:
        """DOCING-06: salvar_documento_enviado lança RAGSecurityError se arquivo já existe."""
        import tempfile
        from agenticlog.rag import salvar_documento_enviado, DIR_DOCUMENTS as _REAL_DIR

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Cria arquivo pré-existente
            existing = tmp_path / "existente.json"
            existing.write_bytes(b'{"ok": "1"}')

            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_documento_enviado("existente.json", b'{"ok": "2"}')

            self.assertIn("já existe", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# DOCING-07: Path traversal / caracteres inválidos rejeitados
# ---------------------------------------------------------------------------

class TestDOCING07PathTraversalRejected(unittest.TestCase):
    """
    DOCING-07: WHEN the filename contains path traversal sequences or Windows-invalid
    characters (<>:"/\\|?* and null bytes)
    THEN the system SHALL reject it before any disk write.
    """

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_docing_07_traversal_filename_shows_error_no_disk_write(
        self,
        mock_salvar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-07: '../evil.json' → RAGSecurityError capturada, st.error exibido."""
        mock_salvar.side_effect = RAGSecurityError(
            "Nome de arquivo com path traversal detectado: '../evil.json'"
        )

        uploaded_file = _make_uploaded_file("../evil.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        mock_st.rerun.assert_not_called()

    def test_docing_07_sanitizar_rejeita_path_traversal(self) -> None:
        """DOCING-07: _sanitizar_nome_arquivo lança RAGSecurityError para '../evil.json'."""
        from agenticlog.rag import _sanitizar_nome_arquivo

        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("../evil.json")

    def test_docing_07_sanitizar_rejeita_chars_invalidos_windows(self) -> None:
        """DOCING-07: _sanitizar_nome_arquivo lança RAGSecurityError para chars inválidos do Windows."""
        from agenticlog.rag import _sanitizar_nome_arquivo

        for char in '<>:"|?*':
            with self.subTest(char=char):
                with self.assertRaises(RAGSecurityError):
                    _sanitizar_nome_arquivo(f"arquivo{char}.json")

    def test_docing_07_sanitizar_rejeita_null_byte(self) -> None:
        """DOCING-07: _sanitizar_nome_arquivo lança RAGSecurityError para null byte no nome."""
        from agenticlog.rag import _sanitizar_nome_arquivo

        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("arq\x00ivo.json")


# ---------------------------------------------------------------------------
# DOCING-08: Limite de 1000 arquivos rejeitado
# ---------------------------------------------------------------------------

class TestDOCING08FileCountLimitRejected(unittest.TestCase):
    """
    DOCING-08: WHEN adding the file would make total file count exceed 1000
    THEN the system SHALL reject it with a file-limit message.
    """

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_docing_08_file_count_limit_shows_error(
        self,
        mock_salvar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-08: 1000 arquivos já presentes → RAGSecurityError capturada, st.error exibido."""
        mock_salvar.side_effect = RAGSecurityError("Limite de 1000 arquivos atingido.")

        uploaded_file = _make_uploaded_file("novo.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("Limite de 1000 arquivos atingido.")
        mock_st.rerun.assert_not_called()

    def test_docing_08_salvar_rejeita_limite_arquivos_diretamente(self) -> None:
        """DOCING-08: salvar_documento_enviado lança RAGSecurityError quando há 1000 arquivos."""
        import json
        import tempfile
        from agenticlog.rag import salvar_documento_enviado
        from agenticlog.config import MAX_JSON_FILES

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Cria MAX_JSON_FILES arquivos fictícios (sem conteúdo real — só precisam existir)
            for i in range(MAX_JSON_FILES):
                (tmp_path / f"doc_{i:04d}.json").write_bytes(b'{}')

            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_documento_enviado("novo.json", b'{"ok": "1"}')

            self.assertIn(str(MAX_JSON_FILES), str(ctx.exception))


# ---------------------------------------------------------------------------
# DOCING-09: Rollback ao falhar rebuild — arquivo removido, vectordb intacto
# ---------------------------------------------------------------------------

class TestDOCING09RollbackOnRebuildFailure(unittest.TestCase):
    """
    DOCING-09: WHEN cria_vectordb() raises any exception during rebuild
    THEN the uploaded file SHALL be removed from data/documents/,
         the original vectordb SHALL remain intact,
         AND an error message SHALL be displayed.
    """

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_documento_enviado")
    def test_docing_09_rebuild_failure_removes_file_and_shows_error(
        self,
        mock_salvar: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-09: reconstruir_vectordb lança Exception → saved_path.unlink chamado, st.error exibido."""
        saved_path = MagicMock(spec=Path)
        mock_salvar.return_value = saved_path
        mock_reconstruir.side_effect = RuntimeError("ChromaDB falhou")

        spinner_ctx = MagicMock()
        spinner_ctx.__enter__ = MagicMock(return_value=None)
        spinner_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = spinner_ctx

        uploaded_file = _make_uploaded_file("doc.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        saved_path.unlink.assert_called_once_with(missing_ok=True)
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("reconstruir base vetorial", error_msg.lower())
        self.assertIn("removido", error_msg.lower())
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_documento_enviado")
    def test_docing_09_rebuild_failure_error_includes_detail(
        self,
        mock_salvar: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-09: mensagem de erro inclui detalhe da exceção original."""
        saved_path = MagicMock(spec=Path)
        mock_salvar.return_value = saved_path
        mock_reconstruir.side_effect = RuntimeError("conexão perdida")

        spinner_ctx = MagicMock()
        spinner_ctx.__enter__ = MagicMock(return_value=None)
        spinner_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = spinner_ctx

        uploaded_file = _make_uploaded_file("doc.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("conexão perdida", error_msg)


# ---------------------------------------------------------------------------
# DOCING-10: Verificado por design — st.rerun() reconstrói o retriever
# ---------------------------------------------------------------------------

class TestDOCING10RetrieverRebuiltAfterRerun(unittest.TestCase):
    """
    DOCING-10: WHEN ingest succeeds
    THEN st.rerun() SHALL be called so the next query uses the rebuilt ChromaDB.

    VERIFIED_BY_DESIGN: st.rerun() causa o Streamlit runner reiniciar o script Python
    do zero, o que re-importa agent.py e reconstrói o retriever a partir do vectordb
    atualizado em disco. Não é possível testar o re-import do módulo em um test unitário
    sem simular o runtime do Streamlit; a garantia é estrutural.

    Este teste valida o contrato observável: st.rerun() é chamado exatamente uma vez
    após um ingest bem-sucedido.
    """

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_documento_enviado")
    def test_docing_10_st_rerun_called_after_successful_ingest(
        self,
        mock_salvar: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-10: st.rerun() chamado exatamente uma vez após ingest bem-sucedido."""
        saved_path = MagicMock(spec=Path)
        mock_salvar.return_value = saved_path
        mock_reconstruir.return_value = None

        spinner_ctx = MagicMock()
        spinner_ctx.__enter__ = MagicMock(return_value=None)
        spinner_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.spinner.return_value = spinner_ctx

        uploaded_file = _make_uploaded_file("doc.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        self.assertEqual(mock_st.rerun.call_count, 1)

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_documento_enviado")
    def test_docing_10_st_rerun_not_called_on_validation_error(
        self,
        mock_salvar: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-10: st.rerun() NÃO chamado quando validação falha (retriever não acionado)."""
        mock_salvar.side_effect = RAGSecurityError("Apenas arquivos .json são aceitos.")

        uploaded_file = _make_uploaded_file("bad.txt", b"data")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.rerun.assert_not_called()


# ---------------------------------------------------------------------------
# TestPDFIngestion: PDF-01, PDF-03 a PDF-10
# ---------------------------------------------------------------------------

class TestPDFIngestion(unittest.TestCase):
    """
    Verifica os critérios de aceite do fluxo de upload de PDF (PDF-01 a PDF-10).
    Usa o mesmo padrão de mock que as classes DOCING existentes:
    patch app.salvar_pdf_enviado, app.salvar_documento_enviado, app.reconstruir_vectordb, app.st.
    """

    # ------------------------------------------------------------------
    # Utilitário interno: spinner context manager padrão
    # ------------------------------------------------------------------

    @staticmethod
    def _spinner_ctx() -> MagicMock:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=None)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    # ------------------------------------------------------------------
    # PDF-01: Happy path — salva, reconstrói, exibe sucesso, rerun
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_01_happy_path(
        self,
        mock_salvar_pdf: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-01: PDF válido → salvar_pdf_enviado + reconstruir chamados, success + rerun disparados."""
        saved_path = MagicMock(spec=Path)
        mock_salvar_pdf.return_value = saved_path
        mock_reconstruir.return_value = None
        mock_st.spinner.return_value = self._spinner_ctx()

        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake content")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_salvar_pdf.assert_called_once_with("contrato.pdf", b"%PDF-1.4 fake content")
        mock_reconstruir.assert_called_once()
        mock_st.success.assert_called_once_with("Documento ingerido com sucesso.")
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-03: PDF protegido por senha → st.error, sem rerun
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_03_senha_mostra_erro(
        self,
        mock_salvar_pdf: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-03: salvar_pdf_enviado lança RAGSecurityError(senha) → st.error, sem rerun."""
        mock_salvar_pdf.side_effect = _rag_module.RAGSecurityError("PDF protegido por senha.")

        uploaded_file = _make_uploaded_file("protegido.pdf", b"%PDF-1.4")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("PDF protegido por senha.")
        mock_st.rerun.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-04: PDF somente-imagem → st.error
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_04_somente_imagem_mostra_erro(
        self,
        mock_salvar_pdf: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-04: PDF somente-imagem → RAGSecurityError capturada, st.error exibido."""
        mock_salvar_pdf.side_effect = _rag_module.RAGSecurityError(
            "PDF não contém texto extraível (somente imagem)."
        )

        uploaded_file = _make_uploaded_file("scan.pdf", b"%PDF-1.4")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("somente imagem", error_msg.lower())
        mock_st.rerun.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-05: PDF > 10 MB → st.error
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_05_tamanho_excedido(
        self,
        mock_salvar_pdf: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-05: arquivo > 10 MB → RAGSecurityError capturada, st.error exibido."""
        mock_salvar_pdf.side_effect = _rag_module.RAGSecurityError("Arquivo excede o limite de 10 MB.")

        uploaded_file = _make_uploaded_file("grande.pdf", b"x" * (11 * 1024 * 1024))

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("10", error_msg)
        mock_st.rerun.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-06: Nome duplicado → st.error
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_06_nome_duplicado(
        self,
        mock_salvar_pdf: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-06: nome de arquivo já existe → RAGSecurityError capturada, st.error exibido."""
        mock_salvar_pdf.side_effect = _rag_module.RAGSecurityError("Arquivo com esse nome já existe.")

        uploaded_file = _make_uploaded_file("existente.pdf", b"%PDF-1.4")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("Arquivo com esse nome já existe.")
        mock_st.rerun.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-08: Rollback quando reconstruir_vectordb falha
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_08_rollback_rebuild_falha(
        self,
        mock_salvar_pdf: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-08: salvar_pdf_enviado sucede, reconstruir falha → unlink chamado, st.error exibido."""
        saved_path = MagicMock(spec=Path)
        mock_salvar_pdf.return_value = saved_path
        mock_reconstruir.side_effect = RuntimeError("ChromaDB falhou")
        mock_st.spinner.return_value = self._spinner_ctx()

        uploaded_file = _make_uploaded_file("relatorio.pdf", b"%PDF-1.4")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        saved_path.unlink.assert_called_once_with(missing_ok=True)
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("reconstruir base vetorial", error_msg.lower())
        self.assertIn("removido", error_msg.lower())
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-09: Extensão inválida (.docx) → st.error
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.salvar_documento_enviado")
    def test_pdf_09_extensao_invalida(
        self,
        mock_salvar_json: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-09: arquivo .docx → roteado para salvar_documento_enviado → RAGSecurityError → st.error."""
        mock_salvar_json.side_effect = _rag_module.RAGSecurityError("Apenas arquivos .json são aceitos.")

        uploaded_file = _make_uploaded_file("contrato.docx", b"PK fake docx")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_salvar_json.assert_called_once_with("contrato.docx", b"PK fake docx")
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("aceitos", error_msg.lower())
        mock_st.rerun.assert_not_called()

    # ------------------------------------------------------------------
    # PDF-10: Extensão .PDF maiúscula — salvar_pdf_enviado chamado
    # ------------------------------------------------------------------

    @patch("app.st")
    @patch("app.reconstruir_vectordb")
    @patch("app.salvar_pdf_enviado")
    def test_pdf_10_extensao_maiuscula_aceita(
        self,
        mock_salvar_pdf: MagicMock,
        mock_reconstruir: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """PDF-10: extensão .PDF (maiúsculo) → salvar_pdf_enviado chamado, salvar_documento_enviado NÃO."""
        saved_path = MagicMock(spec=Path)
        mock_salvar_pdf.return_value = saved_path
        mock_reconstruir.return_value = None
        mock_st.spinner.return_value = self._spinner_ctx()

        uploaded_file = _make_uploaded_file("NOTA_FISCAL.PDF", b"%PDF-1.4 content")

        from app import _ingerir_documento
        with patch("app.salvar_documento_enviado") as mock_salvar_json:
            _ingerir_documento(uploaded_file)
            mock_salvar_json.assert_not_called()

        mock_salvar_pdf.assert_called_once_with("NOTA_FISCAL.PDF", b"%PDF-1.4 content")


if __name__ == "__main__":
    unittest.main(verbosity=2)
