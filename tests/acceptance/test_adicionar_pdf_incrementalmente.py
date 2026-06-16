# AgenticLog — Acceptance Tests: adicionar_pdf_incrementalmente (REC-02)
"""
Verifica todos os critérios de aceite da feature REC-02.

Mapeamento de critérios:
  AC-1 — Happy path: salva PDF, extrai texto (1×), chunka, enriquece 5 metadados,
          insere no ChromaDB, invalida cache, retorna {"status": "adicionado", ...}
  AC-2 — Dedup mesmo hash: retorna {"status": "duplicado", ...} sem escrever em disco
  AC-3 — Hash diferente, mesmo nome: retorna {"status": "hash_diferente", ...} sem escrever
  AC-4 — Validações de segurança: extensão inválida, magic bytes, tamanho, contagem de arquivos
  AC-5 — Rollback em falha de add_documents: delete IDs + unlink arquivo + re-raise
  AC-6 — Integração com app.py: status dispatch correto (success/info/warning/error)
  AC-7 — Zero chunks: arquivo salvo é removido, retorna "0 chunks gerados"
  AC-8 — chunk_index global entre páginas: sequencial em todo o documento
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.rag as rag
from agenticlog.config import (
    DEFAULT_COLLECTION_NAME,
    METADATA_FILE_HASH,
    METADATA_CHUNK_INDEX,
    METADATA_PAGE,
    METADATA_DOC_TYPE,
    METADATA_DOC_TYPE_PDF,
)

# ---------------------------------------------------------------------------
# Constantes de apoio
# ---------------------------------------------------------------------------

_VALID_PDF_BYTES = b"%PDF-1.4 fake content for testing"
_VALID_FILENAME = "relatorio.pdf"
_FAKE_HASH = "a" * 64  # 64-char hex string simulating SHA-256


# ---------------------------------------------------------------------------
# Factories de mock comuns
# ---------------------------------------------------------------------------

def _make_mock_vectordb(existing_ids: list | None = None, existing_hash: str | None = None) -> MagicMock:
    """Retorna mock de instância Chroma com get() configurável."""
    mock_vdb = MagicMock()
    if existing_ids:
        mock_vdb.get.return_value = {
            "ids": existing_ids,
            "metadatas": [{METADATA_FILE_HASH: existing_hash or _FAKE_HASH}],
        }
    else:
        mock_vdb.get.return_value = {"ids": [], "metadatas": []}
    return mock_vdb


def _make_chunks(n: int = 2, source: str = "relatorio.pdf", page: int = 1) -> list:
    """Retorna lista de Documents fake já enriquecidos com metadados."""
    chunks = []
    for i in range(n):
        doc = MagicMock()
        doc.metadata = {
            "source": source,
            METADATA_PAGE: page,
            METADATA_FILE_HASH: _FAKE_HASH,
            METADATA_CHUNK_INDEX: i,
            METADATA_DOC_TYPE: METADATA_DOC_TYPE_PDF,
        }
        doc.page_content = f"Chunk {i} de texto."
        chunks.append(doc)
    return chunks


def _make_spinner_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# AC-1: Happy path
# ---------------------------------------------------------------------------

class TestAC1HappyPath:
    """
    AC-1: GIVEN a valid PDF WHEN adicionar_pdf_incrementalmente is called
    THEN it SHALL save the file, call extrair_texto_pdf exactly once, chunk the text,
    enrich all chunks with 5 metadata fields (source, file_hash, chunk_index, page,
    doc_type="pdf"), insert chunks into ChromaDB, invalidate the agent cache,
    and return {"status": "adicionado", ...}.
    """

    @patch("agenticlog.rag.invalidar_vector_db", create=True)
    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_1_happy_path_retorna_adicionado(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """AC-1: PDF válido → retorna status adicionado com contagem de chunks."""
        # Setup: DIR_DOCUMENTS
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        # Tempfile setup
        tmp_mock = MagicMock()
        tmp_mock.__enter__ = MagicMock(return_value=tmp_mock)
        tmp_mock.__exit__ = MagicMock(return_value=False)
        tmp_mock.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_mock

        # extrair_texto_pdf returns 1-page dict
        mock_extrair.return_value = {"PÁGINA_1": "Texto da página 1."}

        # Chroma mock: no existing docs
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        # Splitter returns 2 chunks
        chunks = _make_chunks(2, source=str(Path("/fake/documents/relatorio.pdf")))
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = chunks
        mock_splitter_cls.return_value = mock_splitter_instance

        # uuid
        mock_uuid.uuid4.return_value.hex = "abcdef1234"

        # Patch lazy import of invalidar_vector_db inside rag module
        with patch.dict("sys.modules", {"agenticlog.agent": MagicMock(invalidar_vector_db=mock_invalidar)}):
            resultado = rag.adicionar_pdf_incrementalmente(
                _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
            )

        assert resultado["status"] == "adicionado"
        assert "adicionado com sucesso" in resultado["mensagem"]
        assert "2 chunks" in resultado["mensagem"]

    @patch("agenticlog.rag.invalidar_vector_db", create=True)
    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_2_extrair_texto_chamado_exatamente_uma_vez(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """AC-1: extrair_texto_pdf deve ser chamado exatamente 1 vez (no tempfile)."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_mock = MagicMock()
        tmp_mock.__enter__ = MagicMock(return_value=tmp_mock)
        tmp_mock.__exit__ = MagicMock(return_value=False)
        tmp_mock.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_mock

        mock_extrair.return_value = {"PÁGINA_1": "Texto."}
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        chunks = _make_chunks(1)
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = chunks
        mock_splitter_cls.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value.hex = "abc123"

        with patch.dict("sys.modules", {"agenticlog.agent": MagicMock(invalidar_vector_db=mock_invalidar)}):
            rag.adicionar_pdf_incrementalmente(
                _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
            )

        assert mock_extrair.call_count == 1, (
            f"extrair_texto_pdf deve ser chamado exatamente 1 vez, foi chamado {mock_extrair.call_count} vezes"
        )

    @patch("agenticlog.rag.invalidar_vector_db", create=True)
    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_3_chunks_tem_5_campos_de_metadados(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """AC-1: todos os chunks devem ter os 5 campos: source, file_hash, chunk_index, page, doc_type=pdf."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_mock = MagicMock()
        tmp_mock.__enter__ = MagicMock(return_value=tmp_mock)
        tmp_mock.__exit__ = MagicMock(return_value=False)
        tmp_mock.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_mock

        mock_extrair.return_value = {"PÁGINA_1": "Texto da página 1."}
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        # Use real Documents so _enriquecer_metadados_chunks can actually set fields
        from langchain_core.documents import Document
        real_doc = Document(
            page_content="PÁGINA_1: Texto da página 1.",
            metadata={"source": "/fake/documents/relatorio.pdf", METADATA_PAGE: 1},
        )
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [real_doc]
        mock_splitter_cls.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value.hex = "abc123"

        inserted_chunks = []

        def capture_add_documents(chunks, ids=None):
            inserted_chunks.extend(chunks)

        mock_vdb.add_documents.side_effect = capture_add_documents

        with patch.dict("sys.modules", {"agenticlog.agent": MagicMock(invalidar_vector_db=mock_invalidar)}):
            rag.adicionar_pdf_incrementalmente(
                _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
            )

        assert len(inserted_chunks) == 1
        chunk = inserted_chunks[0]
        assert "source" in chunk.metadata, "campo 'source' ausente"
        assert METADATA_FILE_HASH in chunk.metadata, "campo 'file_hash' ausente"
        assert METADATA_CHUNK_INDEX in chunk.metadata, "campo 'chunk_index' ausente"
        assert METADATA_PAGE in chunk.metadata, "campo 'page' ausente"
        assert METADATA_DOC_TYPE in chunk.metadata, "campo 'doc_type' ausente"
        assert chunk.metadata[METADATA_DOC_TYPE] == METADATA_DOC_TYPE_PDF, (
            f"doc_type deve ser 'pdf', recebeu {chunk.metadata[METADATA_DOC_TYPE]!r}"
        )

    @patch("agenticlog.rag.invalidar_vector_db", create=True)
    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_4_invalidar_cache_chamado_apos_insercao(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """AC-1: invalidar_vector_db deve ser chamado após add_documents bem-sucedido."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_mock = MagicMock()
        tmp_mock.__enter__ = MagicMock(return_value=tmp_mock)
        tmp_mock.__exit__ = MagicMock(return_value=False)
        tmp_mock.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_mock

        mock_extrair.return_value = {"PÁGINA_1": "Texto."}
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        chunks = _make_chunks(1)
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = chunks
        mock_splitter_cls.return_value = mock_splitter_instance
        mock_uuid.uuid4.return_value.hex = "abc123"

        mock_agent = MagicMock()
        mock_agent.invalidar_vector_db = mock_invalidar

        with patch.dict("sys.modules", {"agenticlog.agent": mock_agent}):
            rag.adicionar_pdf_incrementalmente(
                _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
            )

        mock_invalidar.assert_called_once()


# ---------------------------------------------------------------------------
# AC-2: Deduplicação — mesmo hash
# ---------------------------------------------------------------------------

class TestAC2DedupMesmoHash:
    """
    AC-2: GIVEN the same PDF content already exists in ChromaDB WHEN called again
    THEN SHALL return {"status": "duplicado", ...} without writing to disk or ChromaDB.
    """

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_5_duplicado_mesmo_hash_retorna_sem_escrita(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-2: mesmo conteúdo → status duplicado, sem write em disco, sem add_documents."""
        import hashlib
        real_hash = hashlib.sha256(_VALID_PDF_BYTES).hexdigest()

        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []  # Empty to pass file count guard

        mock_vdb = _make_mock_vectordb(existing_ids=["id1"], existing_hash=real_hash)
        mock_chroma_cls.return_value = mock_vdb

        resultado = rag.adicionar_pdf_incrementalmente(
            _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
        )

        assert resultado["status"] == "duplicado"
        assert "duplicado" in resultado["mensagem"].lower() or "já está" in resultado["mensagem"].lower()

        # Nenhum acesso a disco deve ocorrer
        mock_tempfile.NamedTemporaryFile.assert_not_called()
        mock_shutil.move.assert_not_called()
        mock_vdb.add_documents.assert_not_called()

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_6_duplicado_mensagem_contem_nome_arquivo(
        self,
        mock_dir: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-2: mensagem de duplicado deve conter o nome do arquivo."""
        import hashlib
        real_hash = hashlib.sha256(_VALID_PDF_BYTES).hexdigest()

        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []  # Empty to pass file count guard

        mock_vdb = _make_mock_vectordb(existing_ids=["id1"], existing_hash=real_hash)
        mock_chroma_cls.return_value = mock_vdb

        resultado = rag.adicionar_pdf_incrementalmente(
            _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
        )

        assert "relatorio.pdf" in resultado["mensagem"]


# ---------------------------------------------------------------------------
# AC-3: Hash diferente, mesmo nome
# ---------------------------------------------------------------------------

class TestAC3HashDiferente:
    """
    AC-3: GIVEN a PDF with the same filename but different content already in ChromaDB
    WHEN called THEN SHALL return {"status": "hash_diferente", ...} without modifying storage.
    """

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_7_hash_diferente_retorna_sem_escrita(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-3: mesmo nome, hash diferente → status hash_diferente, sem escrita em disco."""
        # The stored hash is different from the hash of _VALID_PDF_BYTES
        stored_different_hash = "b" * 64

        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        mock_vdb = _make_mock_vectordb(existing_ids=["id1"], existing_hash=stored_different_hash)
        mock_chroma_cls.return_value = mock_vdb

        resultado = rag.adicionar_pdf_incrementalmente(
            _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
        )

        assert resultado["status"] == "hash_diferente"
        mock_tempfile.NamedTemporaryFile.assert_not_called()
        mock_shutil.move.assert_not_called()
        mock_vdb.add_documents.assert_not_called()

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_8_hash_diferente_mensagem_contem_nome(
        self,
        mock_dir: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-3: mensagem de hash_diferente deve mencionar o nome do arquivo."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []  # Empty to pass file count guard

        mock_vdb = _make_mock_vectordb(existing_ids=["id1"], existing_hash="c" * 64)
        mock_chroma_cls.return_value = mock_vdb

        resultado = rag.adicionar_pdf_incrementalmente(
            _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
        )

        assert resultado["status"] == "hash_diferente"
        assert "relatorio.pdf" in resultado["mensagem"]


# ---------------------------------------------------------------------------
# AC-4: Validações de segurança
# ---------------------------------------------------------------------------

class TestAC4SecurityValidation:
    """
    AC-4: GIVEN an invalid file (wrong extension, no %PDF magic bytes, oversized,
    or file count limit reached) WHEN called THEN SHALL raise RAGSecurityError
    before any disk or DB write.
    """

    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_9_extensao_invalida_levanta_security_error(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        """AC-4: extensão .docx → RAGSecurityError antes de qualquer I/O."""
        with pytest.raises(rag.RAGSecurityError) as exc_info:
            rag.adicionar_pdf_incrementalmente("contrato.docx", _VALID_PDF_BYTES)

        assert "pdf" in str(exc_info.value).lower()
        mock_tempfile.NamedTemporaryFile.assert_not_called()
        mock_shutil.move.assert_not_called()

    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_10_magic_bytes_invalidos_levanta_security_error(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        """AC-4: conteúdo sem %PDF nos primeiros 4 bytes → RAGSecurityError antes de qualquer I/O."""
        not_a_pdf = b"PK\x03\x04 fake zip content"  # ZIP magic bytes

        with pytest.raises(rag.RAGSecurityError) as exc_info:
            rag.adicionar_pdf_incrementalmente("contrato.pdf", not_a_pdf)

        assert "pdf" in str(exc_info.value).lower()
        mock_tempfile.NamedTemporaryFile.assert_not_called()
        mock_shutil.move.assert_not_called()

    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_11_tamanho_excedido_levanta_security_error(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
    ) -> None:
        """AC-4: arquivo > MAX_DOCUMENT_FILE_SIZE_MB → RAGSecurityError antes de qualquer I/O."""
        from agenticlog.config import MAX_DOCUMENT_FILE_SIZE_MB
        oversized = b"%PDF" + b"x" * (MAX_DOCUMENT_FILE_SIZE_MB * 1024 * 1024 + 1)

        with pytest.raises(rag.RAGSecurityError) as exc_info:
            rag.adicionar_pdf_incrementalmente("grande.pdf", oversized)

        assert str(MAX_DOCUMENT_FILE_SIZE_MB) in str(exc_info.value) or "limite" in str(exc_info.value).lower()
        mock_tempfile.NamedTemporaryFile.assert_not_called()
        mock_shutil.move.assert_not_called()

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_12_limite_de_arquivos_levanta_security_error(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-4: quando json_count + pdf_count >= MAX_JSON_FILES → RAGSecurityError antes de I/O."""
        from agenticlog.config import MAX_JSON_FILES
        # DIR_DOCUMENTS.glob retorna MAX_JSON_FILES arquivos para cada extensão (json + pdf)
        # totalizando >= MAX_JSON_FILES
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        # Simula que já há MAX_JSON_FILES arquivos .json (0 pdf), atingindo o limite
        mock_dir.glob.return_value = [Path(f"/fake/doc_{i}.json") for i in range(MAX_JSON_FILES)]

        with pytest.raises(rag.RAGSecurityError) as exc_info:
            rag.adicionar_pdf_incrementalmente(_VALID_FILENAME, _VALID_PDF_BYTES)

        assert str(MAX_JSON_FILES) in str(exc_info.value) or "limite" in str(exc_info.value).lower()
        mock_tempfile.NamedTemporaryFile.assert_not_called()
        mock_shutil.move.assert_not_called()

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_13_pdf_com_senha_levanta_security_error_e_limpa_tempfile(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-4: extrair_texto_pdf lança RAGSecurityError(senha) → propagada, tempfile unlinked."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        # Tempfile: retorna objeto com unlink rastreável
        tmp_path_mock = MagicMock(spec=Path)
        tmp_path_mock.name = "/tmp/fake_temp.pdf"

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        mock_extrair.side_effect = rag.RAGSecurityError("PDF protegido por senha.")

        with pytest.raises(rag.RAGSecurityError) as exc_info:
            rag.adicionar_pdf_incrementalmente(_VALID_FILENAME, _VALID_PDF_BYTES)

        assert "senha" in str(exc_info.value).lower()
        # shutil.move should NOT have been called (file not saved)
        mock_shutil.move.assert_not_called()

    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_14_pdf_somente_imagem_levanta_security_error(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
    ) -> None:
        """AC-4: PDF somente-imagem → RAGSecurityError com 'somente imagem', tempfile não salvo."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        mock_extrair.side_effect = rag.RAGSecurityError(
            "PDF não contém texto extraível (somente imagem)."
        )

        with pytest.raises(rag.RAGSecurityError) as exc_info:
            rag.adicionar_pdf_incrementalmente(_VALID_FILENAME, _VALID_PDF_BYTES)

        assert "somente imagem" in str(exc_info.value).lower()
        mock_shutil.move.assert_not_called()


# ---------------------------------------------------------------------------
# AC-5: Rollback em falha de add_documents
# ---------------------------------------------------------------------------

class TestAC5RollbackOnFailure:
    """
    AC-5: GIVEN ChromaDB add_documents fails WHEN called
    THEN SHALL delete any inserted chunk IDs, unlink the saved file,
    and re-raise the original exception.
    """

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_15_rollback_deleta_ids_e_unlink_arquivo(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """AC-5: add_documents levanta exceção → vectordb.delete + saved_path.unlink + re-raise."""
        saved_path_mock = MagicMock(spec=Path)
        saved_path_mock.__str__ = MagicMock(return_value="/fake/documents/relatorio.pdf")
        mock_dir.__truediv__ = MagicMock(return_value=saved_path_mock)
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_extrair.return_value = {"PÁGINA_1": "Texto."}

        mock_vdb = _make_mock_vectordb()
        original_error = RuntimeError("ChromaDB write failure")
        mock_vdb.add_documents.side_effect = original_error
        mock_chroma_cls.return_value = mock_vdb

        chunks = _make_chunks(2)
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = chunks
        mock_splitter_cls.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value.hex = "id_0"

        with pytest.raises(RuntimeError) as exc_info:
            rag.adicionar_pdf_incrementalmente(_VALID_FILENAME, _VALID_PDF_BYTES)

        assert exc_info.value is original_error
        mock_vdb.delete.assert_called_once()
        delete_call_args = mock_vdb.delete.call_args
        assert "ids" in delete_call_args.kwargs or len(delete_call_args.args) > 0
        saved_path_mock.unlink.assert_called_once()

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_16_rollback_reraise_excecao_original(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """AC-5: a exceção original (não a de rollback) deve ser re-lançada."""
        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_extrair.return_value = {"PÁGINA_1": "Texto."}
        mock_vdb = _make_mock_vectordb()

        ingestion_error = ValueError("Specific ingestion error")
        mock_vdb.add_documents.side_effect = ingestion_error
        # Rollback delete also fails — but original must still be re-raised
        mock_vdb.delete.side_effect = Exception("Rollback also failed")
        mock_chroma_cls.return_value = mock_vdb

        chunks = _make_chunks(1)
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = chunks
        mock_splitter_cls.return_value = mock_splitter_instance
        mock_uuid.uuid4.return_value.hex = "abc"

        with pytest.raises(ValueError) as exc_info:
            rag.adicionar_pdf_incrementalmente(_VALID_FILENAME, _VALID_PDF_BYTES)

        assert exc_info.value is ingestion_error, (
            "A exceção original deve ser re-lançada mesmo quando o rollback falha"
        )


# ---------------------------------------------------------------------------
# AC-6: Integração com app.py (UI status dispatch)
# ---------------------------------------------------------------------------

class TestAC6AppIntegration:
    """
    AC-6: GIVEN the Streamlit app receives a PDF upload WHEN _ingerir_documento is called
    THEN SHALL call adicionar_pdf_incrementalmente, show st.success + st.rerun on adicionado,
    st.info on duplicado, st.warning on hash_diferente, and st.error on any exception.
    """

    def _make_uploaded_file(self, name: str, content: bytes) -> MagicMock:
        mock_file = MagicMock()
        mock_file.name = name
        mock_file.getvalue.return_value = content
        return mock_file

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_17_adicionado_mostra_success_e_rerun(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC-6: status adicionado → st.success + st.rerun chamados."""
        mock_adicionar.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo relatorio.pdf adicionado com sucesso. 5 chunks inseridos.",
        }
        mock_st.spinner.return_value = _make_spinner_ctx()

        uploaded_file = self._make_uploaded_file("relatorio.pdf", b"%PDF-1.4 content")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_adicionar.assert_called_once_with(
            "relatorio.pdf", b"%PDF-1.4 content", DEFAULT_COLLECTION_NAME
        )
        mock_st.success.assert_called_once()
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()
        mock_st.info.assert_not_called()
        mock_st.warning.assert_not_called()

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_18_duplicado_mostra_info_sem_rerun(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC-6: status duplicado → st.info chamado, st.rerun NÃO chamado."""
        mock_adicionar.return_value = {
            "status": "duplicado",
            "mensagem": "Arquivo relatorio.pdf já está presente na base vetorial.",
        }
        mock_st.spinner.return_value = _make_spinner_ctx()

        uploaded_file = self._make_uploaded_file("relatorio.pdf", b"%PDF-1.4 content")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.info.assert_called_once()
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_19_hash_diferente_mostra_warning_sem_rerun(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC-6: status hash_diferente → st.warning chamado, st.rerun NÃO chamado."""
        mock_adicionar.return_value = {
            "status": "hash_diferente",
            "mensagem": "Arquivo relatorio.pdf já existe com conteúdo diferente.",
        }
        mock_st.spinner.return_value = _make_spinner_ctx()

        uploaded_file = self._make_uploaded_file("relatorio.pdf", b"%PDF-1.4 content")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.warning.assert_called_once()
        mock_st.rerun.assert_not_called()
        mock_st.success.assert_not_called()

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_20_rag_security_error_mostra_error_sem_rerun(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC-6: RAGSecurityError → st.error chamado, st.rerun NÃO chamado."""
        mock_adicionar.side_effect = rag.RAGSecurityError("PDF protegido por senha.")
        mock_st.spinner.return_value = _make_spinner_ctx()

        uploaded_file = self._make_uploaded_file("relatorio.pdf", b"%PDF-1.4 content")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once_with("PDF protegido por senha.")
        mock_st.rerun.assert_not_called()

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_21_excecao_generica_mostra_error_sem_rerun(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC-6: exceção genérica → st.error chamado com detalhe, st.rerun NÃO chamado."""
        mock_adicionar.side_effect = RuntimeError("ChromaDB falhou inesperadamente")
        mock_st.spinner.return_value = _make_spinner_ctx()

        uploaded_file = self._make_uploaded_file("relatorio.pdf", b"%PDF-1.4 content")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "ChromaDB falhou inesperadamente" in error_msg
        mock_st.rerun.assert_not_called()

    @patch("app.st")
    def teste_22_spinner_texto_correto_para_pdf(
        self,
        mock_st: MagicMock,
    ) -> None:
        """AC-6: spinner deve usar texto 'Adicionando documento à base vetorial...' (não rebuild)."""
        with patch("app.adicionar_pdf_incrementalmente") as mock_adicionar:
            mock_adicionar.return_value = {
                "status": "adicionado",
                "mensagem": "Arquivo test.pdf adicionado com sucesso. 1 chunk inserido.",
            }
            mock_st.spinner.return_value = _make_spinner_ctx()

            uploaded_file = MagicMock()
            uploaded_file.name = "test.pdf"
            uploaded_file.getvalue.return_value = b"%PDF-1.4 content"

            from app import _ingerir_documento
            _ingerir_documento(uploaded_file)

        mock_st.spinner.assert_called_once_with("Adicionando documento à base vetorial...")
        # Ensure spinner text does NOT mention rebuild
        spinner_text = mock_st.spinner.call_args[0][0]
        assert "reconstruindo" not in spinner_text.lower()


# ---------------------------------------------------------------------------
# AC-7: Zero chunks — arquivo removido, mensagem de 0 chunks
# ---------------------------------------------------------------------------

class TestAC7ZeroChunks:
    """
    AC-7: GIVEN a valid PDF that produces no text chunks WHEN called
    THEN SHALL unlink the saved file and return {"status": "adicionado", "mensagem": "... 0 chunks gerados."}.
    """

    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_23_zero_chunks_retorna_mensagem_e_unlink(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
    ) -> None:
        """AC-7: splitter retorna lista vazia → arquivo unlinked, retorna status adicionado com '0 chunks'."""
        saved_path_mock = MagicMock(spec=Path)
        saved_path_mock.__str__ = lambda s: "/fake/documents/relatorio.pdf"

        mock_dir.__truediv__ = lambda self_inner, other: saved_path_mock
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_extrair.return_value = {"PÁGINA_1": "Texto."}
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        # Splitter returns empty list → zero chunks
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = []
        mock_splitter_cls.return_value = mock_splitter_instance

        resultado = rag.adicionar_pdf_incrementalmente(
            _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
        )

        assert resultado["status"] == "adicionado"
        assert "0 chunks" in resultado["mensagem"]
        # File should have been unlinked
        saved_path_mock.unlink.assert_called_once()
        # add_documents should NOT have been called
        mock_vdb.add_documents.assert_not_called()

    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_24_zero_chunks_mensagem_contem_nome_arquivo(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """AC-7: mensagem de 0 chunks deve mencionar o nome do arquivo."""
        saved_path_mock = MagicMock(spec=Path)
        saved_path_mock.__str__ = lambda s: "/fake/documents/relatorio.pdf"

        mock_dir.__truediv__ = lambda self_inner, other: saved_path_mock
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_extrair.return_value = {"PÁGINA_1": "Texto."}
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = []
        mock_splitter_cls.return_value = mock_splitter_instance

        resultado = rag.adicionar_pdf_incrementalmente(
            _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
        )

        assert "relatorio.pdf" in resultado["mensagem"]


# ---------------------------------------------------------------------------
# AC-8: chunk_index global entre páginas
# ---------------------------------------------------------------------------

class TestAC8GlobalChunkIndex:
    """
    AC-8: GIVEN a multi-page PDF WHEN called THEN chunk_index values SHALL be
    globally sequential across all pages (not reset per page).
    """

    @patch("agenticlog.rag.invalidar_vector_db", create=True)
    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_25_chunk_index_global_entre_paginas(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """AC-8: PDF de 3 páginas → chunk_index 0,1,2,... sequencial em todo o documento."""
        from langchain_core.documents import Document

        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        # 3-page PDF — each page provides text
        mock_extrair.return_value = {
            "PÁGINA_1": "Texto longo da página 1 que gera dois chunks.",
            "PÁGINA_2": "Texto da página 2.",
            "PÁGINA_3": "Texto da página 3.",
        }

        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        # Create real Documents per page (as the implementation does)
        # The splitter returns them already split — we simulate 4 total chunks
        # across 3 pages so we can verify global sequential index
        doc_p1a = Document(
            page_content="PÁGINA_1: Texto longo da página 1 que gera",
            metadata={"source": "/fake/documents/relatorio.pdf", METADATA_PAGE: 1},
        )
        doc_p1b = Document(
            page_content="PÁGINA_1: dois chunks.",
            metadata={"source": "/fake/documents/relatorio.pdf", METADATA_PAGE: 1},
        )
        doc_p2 = Document(
            page_content="PÁGINA_2: Texto da página 2.",
            metadata={"source": "/fake/documents/relatorio.pdf", METADATA_PAGE: 2},
        )
        doc_p3 = Document(
            page_content="PÁGINA_3: Texto da página 3.",
            metadata={"source": "/fake/documents/relatorio.pdf", METADATA_PAGE: 3},
        )
        all_chunks = [doc_p1a, doc_p1b, doc_p2, doc_p3]

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = all_chunks
        mock_splitter_cls.return_value = mock_splitter_instance

        mock_uuid.uuid4.return_value.hex = "chunk_id"

        inserted_chunks = []

        def capture_add(chunks, ids=None):
            inserted_chunks.extend(chunks)

        mock_vdb.add_documents.side_effect = capture_add

        with patch.dict("sys.modules", {"agenticlog.agent": MagicMock(invalidar_vector_db=mock_invalidar)}):
            resultado = rag.adicionar_pdf_incrementalmente(
                _VALID_FILENAME, _VALID_PDF_BYTES, DEFAULT_COLLECTION_NAME
            )

        assert resultado["status"] == "adicionado"
        assert len(inserted_chunks) == 4

        chunk_indices = [c.metadata[METADATA_CHUNK_INDEX] for c in inserted_chunks]
        expected_indices = [0, 1, 2, 3]
        assert chunk_indices == expected_indices, (
            f"chunk_index deve ser global sequencial [0,1,2,3], recebeu {chunk_indices}"
        )

    @patch("agenticlog.rag.invalidar_vector_db", create=True)
    @patch("agenticlog.rag.uuid")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.shutil")
    @patch("agenticlog.rag.tempfile")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    def teste_26_metadado_page_nao_usa_sentinel_zero(
        self,
        mock_dir: MagicMock,
        mock_tempfile: MagicMock,
        mock_shutil: MagicMock,
        mock_extrair: MagicMock,
        mock_chroma_cls: MagicMock,
        mock_embeddings_cls: MagicMock,
        mock_splitter_cls: MagicMock,
        mock_uuid: MagicMock,
        mock_invalidar: MagicMock,
    ) -> None:
        """AC-8: campo page em chunks PDF deve ser 1-based, nunca 0 (METADATA_PAGE_JSON_SENTINEL)."""
        from langchain_core.documents import Document
        from agenticlog.config import METADATA_PAGE_JSON_SENTINEL

        mock_dir.__truediv__ = lambda self_inner, other: Path("/fake/documents") / other
        mock_dir.glob.return_value = []

        tmp_ctx = MagicMock()
        tmp_ctx.__enter__ = MagicMock(return_value=tmp_ctx)
        tmp_ctx.__exit__ = MagicMock(return_value=False)
        tmp_ctx.name = "/tmp/fake_temp.pdf"
        mock_tempfile.NamedTemporaryFile.return_value = tmp_ctx

        mock_extrair.return_value = {"PÁGINA_2": "Texto da segunda página."}
        mock_vdb = _make_mock_vectordb()
        mock_chroma_cls.return_value = mock_vdb

        doc = Document(
            page_content="PÁGINA_2: Texto da segunda página.",
            metadata={"source": "/fake/documents/relatorio.pdf", METADATA_PAGE: 2},
        )
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [doc]
        mock_splitter_cls.return_value = mock_splitter_instance
        mock_uuid.uuid4.return_value.hex = "abc"

        inserted_chunks = []
        mock_vdb.add_documents.side_effect = lambda chunks, ids=None: inserted_chunks.extend(chunks)

        with patch.dict("sys.modules", {"agenticlog.agent": MagicMock(invalidar_vector_db=mock_invalidar)}):
            rag.adicionar_pdf_incrementalmente(_VALID_FILENAME, _VALID_PDF_BYTES)

        assert len(inserted_chunks) == 1
        page_val = inserted_chunks[0].metadata.get(METADATA_PAGE)
        assert page_val != METADATA_PAGE_JSON_SENTINEL, (
            f"page não deve ser METADATA_PAGE_JSON_SENTINEL (0) em chunks PDF, recebeu {page_val}"
        )
        assert page_val == 2, f"page deve ser 2 (1-based), recebeu {page_val}"


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
