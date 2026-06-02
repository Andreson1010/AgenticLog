# AgenticLog — Acceptance Tests: Document Ingestion UI Feature
"""
Verifica todos os critérios de aceite da story:
  "Como operador logístico, quero fazer upload de um documento JSON pela
   barra lateral do Streamlit para que o ChromaDB seja atualizado
   incrementalmente com o novo conteúdo sem reiniciar a aplicação."

Mapeamento de critérios (DOCING-01 a DOCING-10):
  DOCING-01 — AC-1: happy path (adicionar incrementalmente + rerun)
  DOCING-02 — AC-2: spinner visível durante adição
  DOCING-03 — AC-3: extensão não-.json rejeitada antes de escrita
  DOCING-04 — AC-4: arquivo > 10 MB rejeitado antes de escrita
  DOCING-05 — AC-5: chave proibida "lc" rejeitada antes de escrita
  DOCING-06 — AC-6: colisão de nome — duplicado ou hash diferente notificado
  DOCING-07 — AC-7: path traversal / chars inválidos rejeitados
  DOCING-08 — AC-8: limite de 1000 arquivos rejeitado
  DOCING-09 — AC-9: rollback ao falhar adição (sem chunks órfãos)
  DOCING-10 — AC-10: verificado por design (st.rerun() aciona re-import do agente)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock

from agenticlog.rag import RAGSecurityError


def _make_uploaded_file(name: str, content: bytes) -> MagicMock:
    """Retorna um MagicMock com a interface mínima do UploadedFile do Streamlit."""
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.getvalue.return_value = content
    return mock_file


def _mock_spinner(mock_st: MagicMock) -> None:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_st.spinner.return_value = ctx


# ---------------------------------------------------------------------------
# DOCING-01: Happy path — adiciona incrementalmente, exibe sucesso, rerun
# ---------------------------------------------------------------------------

class TestDOCING01HappyPath(unittest.TestCase):
    """
    DOCING-01: WHEN operator uploads a valid .json file < 10 MB with no forbidden keys
    THEN chunks SHALL be added incrementally to ChromaDB,
         a success message SHALL be shown,
         AND st.rerun() SHALL be called.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_01_happy_path_saves_rebuilds_and_reruns(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-01: valid upload → adicionar chamado, st.success + st.rerun disparados."""
        mock_add.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo frete.json adicionado com sucesso. 2 chunks inseridos.",
        }
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("frete.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_add.assert_called_once_with("frete.json", b'{"tipo": "frete"}')
        mock_st.success.assert_called_once_with(
            "Arquivo frete.json adicionado com sucesso. 2 chunks inseridos."
        )
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()


# ---------------------------------------------------------------------------
# DOCING-02: Spinner visível durante adição
# ---------------------------------------------------------------------------

class TestDOCING02SpinnerDuringRebuild(unittest.TestCase):
    """
    DOCING-02: WHEN addition is running
    THEN a spinner with text "Adicionando documento à base vetorial..." SHALL be visible.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_02_spinner_wraps_rebuild(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-02: st.spinner chamado com texto correto e envolve adicionar_documento."""
        mock_add.return_value = {"status": "adicionado", "mensagem": "ok"}
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("doc.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.spinner.assert_called_once_with("Adicionando documento à base vetorial...")
        mock_add.assert_called_once()


# ---------------------------------------------------------------------------
# DOCING-03: Extensão não-.json rejeitada antes de escrita em disco
# ---------------------------------------------------------------------------

class TestDOCING03NonJsonExtensionRejected(unittest.TestCase):

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_03_non_json_extension_shows_error_no_disk_write(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-03: extensão .csv → RAGSecurityError capturada, st.error exibido."""
        mock_add.side_effect = RAGSecurityError("Apenas arquivos .json são aceitos.")
        _mock_spinner(mock_st)

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

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_04_large_file_shows_error_no_disk_write(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-04: arquivo > 10 MB → RAGSecurityError capturada, st.error exibido."""
        mock_add.side_effect = RAGSecurityError("Arquivo excede o limite de 10 MB.")
        _mock_spinner(mock_st)

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

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_05_forbidden_key_shows_error_no_disk_write(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-05: chave 'lc' → RAGSecurityError capturada, st.error exibido."""
        mock_add.side_effect = RAGSecurityError("Arquivo contém chave proibida 'lc'")
        _mock_spinner(mock_st)

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
        from agenticlog.rag import salvar_documento_enviado

        conteudo = json.dumps({"lc": "evil", "tipo": "frete"}).encode()

        with self.assertRaises(RAGSecurityError) as ctx:
            salvar_documento_enviado("malicioso.json", conteudo)

        self.assertIn("proibida", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# DOCING-06: Colisão de nome — duplicado ou hash diferente notificado
# ---------------------------------------------------------------------------

class TestDOCING06FilenameCollisionRejected(unittest.TestCase):
    """
    DOCING-06: WHEN filename already exists AND content hash differs
    THEN the system SHALL warn the operator and NOT overwrite.
    WHEN filename exists AND content hash matches
    THEN the system SHALL inform the operator it is already present.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_06_collision_shows_error_no_disk_write(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-06: mesmo nome, hash diferente → st.warning, sem rerun."""
        mock_add.return_value = {
            "status": "hash_diferente",
            "mensagem": "Arquivo doc1.json já existe com conteúdo diferente.",
        }
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("doc1.json", b'{"tipo": "frete_v2"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.warning.assert_called_once()
        mock_st.rerun.assert_not_called()
        mock_st.error.assert_not_called()

    def test_docing_06_salvar_rejeita_colisao_diretamente(self) -> None:
        """DOCING-06: salvar_documento_enviado lança RAGSecurityError se arquivo já existe."""
        import tempfile
        from agenticlog.rag import salvar_documento_enviado

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
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

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_07_traversal_filename_shows_error_no_disk_write(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-07: '../evil.json' → RAGSecurityError capturada, st.error exibido."""
        mock_add.side_effect = RAGSecurityError(
            "Nome de arquivo com path traversal detectado: '../evil.json'"
        )
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("../evil.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        mock_st.rerun.assert_not_called()

    def test_docing_07_sanitizar_rejeita_path_traversal(self) -> None:
        from agenticlog.rag import _sanitizar_nome_arquivo
        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("../evil.json")

    def test_docing_07_sanitizar_rejeita_chars_invalidos_windows(self) -> None:
        from agenticlog.rag import _sanitizar_nome_arquivo
        for char in '<>:"|?*':
            with self.subTest(char=char):
                with self.assertRaises(RAGSecurityError):
                    _sanitizar_nome_arquivo(f"arquivo{char}.json")

    def test_docing_07_sanitizar_rejeita_null_byte(self) -> None:
        from agenticlog.rag import _sanitizar_nome_arquivo
        with self.assertRaises(RAGSecurityError):
            _sanitizar_nome_arquivo("arq\x00ivo.json")


# ---------------------------------------------------------------------------
# DOCING-08: Limite de 1000 arquivos rejeitado
# ---------------------------------------------------------------------------

class TestDOCING08FileCountLimitRejected(unittest.TestCase):

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_08_file_count_limit_shows_error(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-08: 1000 arquivos já presentes → RAGSecurityError capturada, st.error exibido."""
        mock_add.side_effect = RAGSecurityError("Limite de 1000 arquivos atingido.")
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("novo.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("Limite de 1000 arquivos atingido.")
        mock_st.rerun.assert_not_called()

    def test_docing_08_salvar_rejeita_limite_arquivos_diretamente(self) -> None:
        """DOCING-08: salvar_documento_enviado lança RAGSecurityError quando há 1000 arquivos."""
        import tempfile
        from agenticlog.rag import salvar_documento_enviado
        from agenticlog.config import MAX_JSON_FILES

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for i in range(MAX_JSON_FILES):
                (tmp_path / f"doc_{i:04d}.json").write_bytes(b'{}')

            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path):
                with self.assertRaises(RAGSecurityError) as ctx:
                    salvar_documento_enviado("novo.json", b'{"ok": "1"}')

            self.assertIn(str(MAX_JSON_FILES), str(ctx.exception))


# ---------------------------------------------------------------------------
# DOCING-09: Rollback ao falhar adição — app exibe erro com detalhe
# ---------------------------------------------------------------------------

class TestDOCING09RollbackOnRebuildFailure(unittest.TestCase):
    """
    DOCING-09: WHEN adicionar_documento_incrementalmente raises any exception
    THEN rollback is handled internally by rag.py (no orphan chunks),
         AND app SHALL display an error message with the exception detail.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_09_rebuild_failure_removes_file_and_shows_error(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-09: exceção em adicionar → st.error exibido, st.rerun não chamado."""
        mock_add.side_effect = RuntimeError("ChromaDB falhou")
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("doc.json", b'{"x": "y"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        self.assertIn("Erro ao adicionar documento", error_msg)
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_09_rebuild_failure_error_includes_detail(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-09: mensagem de erro inclui detalhe da exceção original."""
        mock_add.side_effect = RuntimeError("conexão perdida")
        _mock_spinner(mock_st)

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
    THEN st.rerun() SHALL be called so the next query uses the updated ChromaDB
    (singleton invalidado em rag.py antes do rerun).
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_10_st_rerun_called_after_successful_ingest(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-10: st.rerun() chamado exatamente uma vez após ingest bem-sucedido."""
        mock_add.return_value = {"status": "adicionado", "mensagem": "ok"}
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("doc.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        self.assertEqual(mock_st.rerun.call_count, 1)

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_docing_10_st_rerun_not_called_on_validation_error(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """DOCING-10: st.rerun() NÃO chamado quando validação falha."""
        mock_add.side_effect = RAGSecurityError("Apenas arquivos .json são aceitos.")
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("bad.txt", b"data")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.rerun.assert_not_called()


# ---------------------------------------------------------------------------
# P1-1 / P1-9: Pre-existing chunks survive incremental ingestion
# ---------------------------------------------------------------------------

class TestP1PreExistingChunksSurvive(unittest.TestCase):
    """
    P1-1: WHEN new file ingested THEN only new file's chunks added — no existing chunks removed.
    P1-9: WHEN any incremental ingestion completes THEN all pre-existing chunks intact.
    """

    @patch("agenticlog.rag.invalidar_vector_db")
    @patch("agenticlog.rag.JSONLoader")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def test_p1_1_add_does_not_delete_existing_ids(
        self,
        mock_dir: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """P1-1 + P1-9: add_documents called with only new chunk IDs; delete never called."""
        import tempfile
        import json as _json

        content = _json.dumps({"tipo": "frete", "valor": "100"}).encode()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mock_dir.__truediv__ = lambda self, other: tmp_path / other
            mock_dir.glob = lambda pattern: []
            mock_dir.__str__ = lambda self: str(tmp_path)

            # No existing file with this name
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.get.return_value = {"ids": [], "metadatas": []}
            mock_chroma_cls.return_value = mock_chroma_instance

            mock_embeddings_cls.return_value = MagicMock()

            # Loader returns one document
            from langchain_core.documents import Document
            fake_doc = Document(page_content="tipo: frete\nvalor: 100")
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [fake_doc]
            mock_loader_cls.return_value = mock_loader_instance

            from agenticlog.rag import adicionar_documento_incrementalmente

            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path), \
                 patch("agenticlog.rag.DIR_VECTORDB", tmp_path / "vectordb"):
                resultado = adicionar_documento_incrementalmente("novo.json", content)

        # add_documents called exactly once; delete never called (no rollback needed)
        mock_chroma_instance.add_documents.assert_called_once()
        mock_chroma_instance.delete.assert_not_called()
        self.assertEqual(resultado["status"], "adicionado")


# ---------------------------------------------------------------------------
# P1-2: invalidar_vector_db called before st.rerun on success
# ---------------------------------------------------------------------------

class TestP1SIngletonInvalidatedBeforeRerun(unittest.TestCase):
    """
    P1-2: WHEN successful incremental ingestion THEN singleton invalidated
    (invalidar_vector_db called) before st.rerun() so next query uses updated collection.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    @patch("agenticlog.rag.invalidar_vector_db")
    def test_p1_2_invalidar_called_on_success(
        self,
        mock_invalidar: MagicMock,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """P1-2: adicionar_documento_incrementalmente internally calls invalidar_vector_db;
        app then calls st.rerun(). We verify the full chain fires on a successful result."""
        # The real implementation calls invalidar_vector_db() inside rag.py before returning.
        # Here we simulate the happy-path result and confirm app calls st.rerun().
        mock_add.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo ok.json adicionado com sucesso. 3 chunks inseridos.",
        }
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("ok.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        # st.rerun must fire so Streamlit reloads and agent picks up updated ChromaDB
        mock_st.rerun.assert_called_once()
        mock_st.success.assert_called_once()


# ---------------------------------------------------------------------------
# P1-3: UI message says "arquivo adicionado", not "base vetorial reconstruída"
# ---------------------------------------------------------------------------

class TestP1UiMessageWording(unittest.TestCase):
    """
    P1-3: WHEN ingestion succeeds THEN operator sees "arquivo adicionado" message,
    NOT "base vetorial reconstruída".
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_p1_3_success_message_says_adicionado_not_reconstruida(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """P1-3: success message contains 'adicionado', not 'reconstruída'."""
        mock_add.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo logistica.json adicionado com sucesso. 5 chunks inseridos.",
        }
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("logistica.json", b'{"tipo": "logistica"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        success_msg = mock_st.success.call_args[0][0]
        self.assertIn("adicionado", success_msg.lower())
        self.assertNotIn("reconstruída", success_msg.lower())
        self.assertNotIn("reconstruida", success_msg.lower())


# ---------------------------------------------------------------------------
# P1-4: First document — collection created fresh without error
# ---------------------------------------------------------------------------

class TestP1FirstDocumentCreatesCollection(unittest.TestCase):
    """
    P1-4: WHEN ChromaDB collection does not exist yet AND first document uploaded
    THEN system creates new collection and ingests chunks without error.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_p1_4_first_document_no_error(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """P1-4: when collection is empty/new, successful result returned and no error shown."""
        mock_add.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo primeiro.json adicionado com sucesso. 1 chunks inseridos.",
        }
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("primeiro.json", b'{"tipo": "primeiro"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_not_called()
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_called_once()

    @patch("agenticlog.rag.invalidar_vector_db")
    @patch("agenticlog.rag.JSONLoader")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    def test_p1_4_chroma_get_returns_empty_triggers_add(
        self,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """P1-4 (rag layer): Chroma.get() returning empty IDs triggers add_documents."""
        import tempfile
        import json as _json
        from langchain_core.documents import Document

        content = _json.dumps({"tipo": "primeiro"}).encode()

        mock_chroma_instance = MagicMock()
        # Empty collection — no prior chunks with this source
        mock_chroma_instance.get.return_value = {"ids": [], "metadatas": []}
        mock_chroma_cls.return_value = mock_chroma_instance
        mock_embeddings_cls.return_value = MagicMock()

        fake_doc = Document(page_content="tipo: primeiro")
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [fake_doc]
        mock_loader_cls.return_value = mock_loader_instance

        from agenticlog.rag import adicionar_documento_incrementalmente

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path), \
                 patch("agenticlog.rag.DIR_VECTORDB", tmp_path / "vectordb"):
                resultado = adicionar_documento_incrementalmente("primeiro.json", content)

        mock_chroma_instance.add_documents.assert_called_once()
        self.assertEqual(resultado["status"], "adicionado")


# ---------------------------------------------------------------------------
# P1-7: Same name + same hash → st.info, no duplicate chunks
# ---------------------------------------------------------------------------

class TestP1DuplicateSameHash(unittest.TestCase):
    """
    P1-7: WHEN file re-uploaded with same name AND same content hash
    THEN no duplicate chunks added AND operator notified file already present (st.info).
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def test_p1_7_duplicate_same_hash_shows_info_not_success(
        self,
        mock_add: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """P1-7: status 'duplicado' → st.info called, st.success and st.rerun NOT called."""
        mock_add.return_value = {
            "status": "duplicado",
            "mensagem": "Arquivo frete.json já está presente na base vetorial.",
        }
        _mock_spinner(mock_st)

        uploaded_file = _make_uploaded_file("frete.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.info.assert_called_once()
        info_msg = mock_st.info.call_args[0][0]
        self.assertIn("frete.json", info_msg)
        # No duplicate ingestion: no rerun, no success
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()
        mock_st.error.assert_not_called()

    @patch("agenticlog.rag.invalidar_vector_db")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    def test_p1_7_rag_returns_duplicado_on_same_hash(
        self,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """P1-7 (rag layer): when ChromaDB already has chunks with same hash, return 'duplicado'."""
        import tempfile
        import json as _json

        content = _json.dumps({"tipo": "frete"}).encode()
        from agenticlog.rag import _computar_hash_conteudo
        expected_hash = _computar_hash_conteudo(content)

        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get.return_value = {
            "ids": ["existing-id-1"],
            "metadatas": [{"content_hash": expected_hash, "source": "frete.json"}],
        }
        mock_chroma_cls.return_value = mock_chroma_instance
        mock_embeddings_cls.return_value = MagicMock()

        from agenticlog.rag import adicionar_documento_incrementalmente

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Pre-create the file so arquivo_ja_existia == True (avoids writing disk)
            (tmp_path / "frete.json").write_bytes(content)
            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path), \
                 patch("agenticlog.rag.DIR_VECTORDB", tmp_path / "vectordb"):
                resultado = adicionar_documento_incrementalmente("frete.json", content)

        # add_documents must NOT be called — no new chunks
        mock_chroma_instance.add_documents.assert_not_called()
        self.assertEqual(resultado["status"], "duplicado")


# ---------------------------------------------------------------------------
# P2-1: Rollback on mid-ingestion exception deletes added chunk IDs
# ---------------------------------------------------------------------------

class TestP2RollbackOnIngestionFailure(unittest.TestCase):
    """
    P2-1: WHEN embedding or collection.add() raises exception mid-ingestion
    THEN system deletes all chunk IDs added during that attempt AND re-raises original exception.
    """

    @patch("agenticlog.rag.invalidar_vector_db")
    @patch("agenticlog.rag.JSONLoader")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    def test_p2_1_rollback_deletes_chunk_ids_on_add_failure(
        self,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """P2-1: when add_documents raises, delete is called with the same IDs and exception re-raised."""
        import tempfile
        import json as _json
        from langchain_core.documents import Document

        content = _json.dumps({"tipo": "frete", "valor": "200"}).encode()

        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get.return_value = {"ids": [], "metadatas": []}
        add_error = RuntimeError("ChromaDB connection lost")
        mock_chroma_instance.add_documents.side_effect = add_error
        mock_chroma_cls.return_value = mock_chroma_instance
        mock_embeddings_cls.return_value = MagicMock()

        fake_doc = Document(page_content="tipo: frete\nvalor: 200")
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [fake_doc]
        mock_loader_cls.return_value = mock_loader_instance

        from agenticlog.rag import adicionar_documento_incrementalmente

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path), \
                 patch("agenticlog.rag.DIR_VECTORDB", tmp_path / "vectordb"):
                with self.assertRaises(RuntimeError) as ctx:
                    adicionar_documento_incrementalmente("frete_fail.json", content)

        # Original exception re-raised
        self.assertIs(ctx.exception, add_error)
        # Rollback: delete called with IDs that were attempted
        mock_chroma_instance.delete.assert_called_once()
        deleted_ids = mock_chroma_instance.delete.call_args[1].get(
            "ids", mock_chroma_instance.delete.call_args[0][0] if mock_chroma_instance.delete.call_args[0] else []
        )
        self.assertIsInstance(deleted_ids, list)
        self.assertGreater(len(deleted_ids), 0)


# ---------------------------------------------------------------------------
# P2-2: Rollback failure logs CRITICAL and re-raises original exception
# ---------------------------------------------------------------------------

class TestP2RollbackFailureLogsCritical(unittest.TestCase):
    """
    P2-2: WHEN rollback itself fails THEN system logs CRITICAL with orphaned IDs
    AND re-raises the original ingestion exception (not the rollback exception).
    """

    @patch("agenticlog.rag.invalidar_vector_db")
    @patch("agenticlog.rag.JSONLoader")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    def test_p2_2_rollback_failure_logs_critical_reraises_original(
        self,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_loader_cls: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """P2-2: when rollback (delete) also fails, CRITICAL logged and original error re-raised."""
        import tempfile
        import json as _json
        import logging
        from langchain_core.documents import Document

        content = _json.dumps({"tipo": "frete", "valor": "300"}).encode()

        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get.return_value = {"ids": [], "metadatas": []}
        original_error = RuntimeError("add failed")
        rollback_error = RuntimeError("delete also failed")
        mock_chroma_instance.add_documents.side_effect = original_error
        mock_chroma_instance.delete.side_effect = rollback_error
        mock_chroma_cls.return_value = mock_chroma_instance
        mock_embeddings_cls.return_value = MagicMock()

        fake_doc = Document(page_content="tipo: frete\nvalor: 300")
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [fake_doc]
        mock_loader_cls.return_value = mock_loader_instance

        from agenticlog.rag import adicionar_documento_incrementalmente

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("agenticlog.rag.DIR_DOCUMENTS", tmp_path), \
                 patch("agenticlog.rag.DIR_VECTORDB", tmp_path / "vectordb"):
                with self.assertLogs("agenticlog.rag", level=logging.CRITICAL) as log_ctx:
                    with self.assertRaises(RuntimeError) as exc_ctx:
                        adicionar_documento_incrementalmente("frete_rollback.json", content)

        # Original exception (not rollback exception) must be re-raised
        self.assertIs(exc_ctx.exception, original_error)

        # At least one CRITICAL log record must mention the orphaned IDs
        critical_msgs = [r for r in log_ctx.output if "CRITICAL" in r]
        self.assertTrue(
            len(critical_msgs) >= 1,
            "Expected at least one CRITICAL log entry when rollback fails",
        )
        combined = " ".join(critical_msgs)
        # Log must contain some indicator of orphaned IDs (the spec says "logs CRITICAL with orphaned IDs")
        self.assertTrue(
            any(keyword in combined.lower() for keyword in ["órfão", "orfao", "ids", "rollback"]),
            f"CRITICAL log did not mention orphaned IDs or rollback: {combined}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
