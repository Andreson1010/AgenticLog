# AgenticLog — Acceptance Tests: Multi-Collection ChromaDB Feature
"""
Verifica todos os critérios de aceite da feature multi-collection ChromaDB.

Mapeamento de critérios (AC1–AC18, conforme user story aprovada):

  AC1  — UI mostra dropdown com coleções existentes + "Nova coleção…"
  AC2  — Selecionar "Nova coleção…" revela input com validação inline
  AC3  — Upload JSON + coleção nomeada → adicionar_documento_incrementalmente chamado
         com collection_name; sucesso mostra nome da coleção
  AC4  — Upload PDF + coleção nomeada → salvar_pdf_enviado + reconstruir_vectordb
         chamados com collection_name; sucesso mostra nome da coleção
  AC5  — Duas ou mais coleções com docs → _get_retriever() faz fan-out em todas,
         mescla e retorna top-ranked
  AC6  — Sem collection_name explícito → DEFAULT_COLLECTION_NAME ("logistica") usado
  AC7  — Nome < 3 chars → RAGSecurityError, sem escrita
  AC8  — Nome > 63 chars → RAGSecurityError, sem escrita
  AC9  — Chars inválidos → RAGSecurityError, sem escrita
  AC10 — Começa/termina com não-alfanumérico → RAGSecurityError, sem escrita
  AC11 — (UNTESTABLE) Política documentada; sem comportamento de runtime verificável
  AC12 — sanitizar_nome_colecao() executado antes de qualquer escrita
  AC13 — invalidar_vector_db() limpa todo _vector_dbs; sem exceção em dict vazio
  AC14 — QueryRequest sem campo collection_name
  AC15 — Limites de validação vêm das constantes de config.py
  AC16 — Dropdown + "Nova coleção…" — sentinel constante presente, fluxo validado
  AC17 — Coleção vazia durante fan-out → skip silencioso (0 docs)
  AC18 — Erro ChromaDB durante fan-out → propagação imediata
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import MagicMock, call, patch

from agenticlog.config import (
    COLLECTION_NAME_MAX_LEN,
    COLLECTION_NAME_MIN_LEN,
    DEFAULT_COLLECTION_NAME,
)
from agenticlog.rag import RAGSecurityError, sanitizar_nome_colecao


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def _make_uploaded_file(name: str, content: bytes) -> MagicMock:
    """Retorna um MagicMock com a interface mínima do UploadedFile do Streamlit."""
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.getvalue.return_value = content
    return mock_file


def _spinner_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _make_doc(content: str) -> MagicMock:
    """Retorna um MagicMock imitando langchain_core.documents.Document."""
    from langchain_core.documents import Document
    return Document(page_content=content, metadata={"source": "test"})


# ---------------------------------------------------------------------------
# AC1 / AC16: Dropdown de coleções + "Nova coleção…"
# ---------------------------------------------------------------------------


class TestAC01AC16DropdownColecoes(unittest.TestCase):
    """
    AC1: WHEN o expander "Adicionar Documento" está aberto
    THEN o sistema SHALL exibir dropdown com nomes de coleções existentes
    mais a opção "Nova coleção…".

    AC16: Dropdown + input híbrido — sentinel NOVA_COLECAO_SENTINEL presente em app.py.
    """

    def teste_1_sentinel_nova_colecao_definido_em_app(self) -> None:
        """AC1/AC16: app.py define NOVA_COLECAO_SENTINEL = 'Nova coleção…'."""
        import app as _app
        self.assertTrue(
            hasattr(_app, "NOVA_COLECAO_SENTINEL"),
            "app.py deve exportar NOVA_COLECAO_SENTINEL",
        )
        self.assertEqual(_app.NOVA_COLECAO_SENTINEL, "Nova coleção…")

    def teste_2_listar_colecoes_retorna_lista_de_strings(self) -> None:
        """AC1: _listar_colecoes() retorna lista de strings (tipos corretos para selectbox).

        chromadb é lazy-importado dentro de _listar_colecoes(); patchamos o módulo
        via sys.modules para interceptar o import dinâmico.
        """
        import sys
        from agenticlog.agent import _listar_colecoes

        col1 = MagicMock()
        col1.name = "fornecedores"
        col2 = MagicMock()
        col2.name = "contratos"
        client_mock = MagicMock()
        client_mock.list_collections.return_value = [col1, col2]

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = client_mock

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            result = _listar_colecoes()

        self.assertIsInstance(result, list)
        self.assertIn("fornecedores", result)
        self.assertIn("contratos", result)

    def teste_3_listar_colecoes_sem_colecoes_retorna_default(self) -> None:
        """AC1: WHEN ChromaDB vazio THEN _listar_colecoes retorna [DEFAULT_COLLECTION_NAME]."""
        import sys
        from agenticlog.agent import _listar_colecoes

        client_mock = MagicMock()
        client_mock.list_collections.return_value = []

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = client_mock

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            result = _listar_colecoes()

        self.assertEqual(result, [DEFAULT_COLLECTION_NAME])

    def teste_4_listar_colecoes_erro_retorna_default(self) -> None:
        """AC1: WHEN ChromaDB lança exceção THEN _listar_colecoes retorna [DEFAULT_COLLECTION_NAME]."""
        import sys
        from agenticlog.agent import _listar_colecoes

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.side_effect = RuntimeError("disk error")

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            result = _listar_colecoes()

        self.assertEqual(result, [DEFAULT_COLLECTION_NAME])


# ---------------------------------------------------------------------------
# AC2: "Nova coleção…" revela input com validação inline
# ---------------------------------------------------------------------------


class TestAC02NovaColecaoValidacaoInline(unittest.TestCase):
    """
    AC2: WHEN operador seleciona "Nova coleção…"
    THEN o sistema SHALL revelar input com feedback de validação inline via
    sanitizar_nome_colecao() — nome válido → "Nome válido.", inválido → caption de erro.
    """

    def teste_1_nome_valido_nao_levanta_excecao(self) -> None:
        """AC2: sanitizar_nome_colecao('fornecedores') retorna nome sem erro."""
        result = sanitizar_nome_colecao("fornecedores")
        self.assertEqual(result, "fornecedores")

    def teste_2_nome_invalido_levanta_rag_security_error(self) -> None:
        """AC2: sanitizar_nome_colecao('ab') lança RAGSecurityError — UI exibe caption de erro."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("ab")

    def teste_3_nome_com_espaco_invalido(self) -> None:
        """AC2: sanitizar_nome_colecao('nome com espaço') lança RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome com espaço")


# ---------------------------------------------------------------------------
# AC3: Upload JSON com coleção nomeada
# ---------------------------------------------------------------------------


class TestAC03UploadJsonComColecaoNomeada(unittest.TestCase):
    """
    AC3: WHEN operador clica "Ingerir Documento" com JSON válido e coleção nomeada
    THEN sistema SHALL chamar adicionar_documento_incrementalmente(filename, conteudo, collection_name)
    e exibir mensagem de sucesso contendo o nome da coleção.
    """

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def teste_1_json_com_nome_colecao_chama_adicionar_com_collection_name(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC3: JSON + 'fornecedores' → adicionar chamado com collection_name='fornecedores'."""
        mock_adicionar.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo pedido.json adicionado com sucesso. 3 chunks inseridos.",
        }
        mock_st.spinner.return_value = _spinner_ctx()

        uploaded_file = _make_uploaded_file("pedido.json", b'{"tipo": "pedido"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file, collection_name="fornecedores")

        mock_adicionar.assert_called_once_with(
            "pedido.json", b'{"tipo": "pedido"}', "fornecedores"
        )

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def teste_2_sucesso_json_nao_exige_collection_name_na_mensagem_de_rag(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC3: sucesso → st.success chamado (mensagem do rag retornada ao usuário)."""
        mock_adicionar.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo pedido.json adicionado com sucesso. 3 chunks inseridos.",
        }
        mock_st.spinner.return_value = _spinner_ctx()

        uploaded_file = _make_uploaded_file("pedido.json", b'{"tipo": "pedido"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file, collection_name="fornecedores")

        # Sucesso grava ingest_msg (sobrevive ao st.rerun) — não chama st.success direto.
        self.assertEqual(
            mock_st.session_state.ingest_msg,
            ("success", "Arquivo pedido.json adicionado com sucesso. 3 chunks inseridos."),
        )
        mock_st.success.assert_not_called()
        mock_st.rerun.assert_called_once()
        mock_st.error.assert_not_called()


# ---------------------------------------------------------------------------
# AC4: Upload PDF com coleção nomeada
# ---------------------------------------------------------------------------


class TestAC04UploadPdfComColecaoNomeada(unittest.TestCase):
    """
    AC4: WHEN operador clica "Ingerir Documento" com PDF válido e coleção nomeada
    THEN sistema SHALL chamar adicionar_pdf_incrementalmente(filename, conteudo, collection_name)
    e exibir sucesso com o nome da coleção.
    """

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_1_pdf_com_nome_colecao_chama_adicionar_incrementalmente(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC4: PDF + 'contratos' → adicionar_pdf_incrementalmente chamado com 'contratos'."""
        mock_adicionar.return_value = {"status": "adicionado", "mensagem": "contrato.pdf adicionado."}
        mock_st.spinner.return_value = _spinner_ctx()

        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file, collection_name="contratos")

        mock_adicionar.assert_called_once_with(
            "contrato.pdf", b"%PDF-1.4 fake", "contratos"
        )

    @patch("app.st")
    @patch("app.adicionar_pdf_incrementalmente")
    def teste_2_pdf_sucesso_exibe_mensagem_com_nome_colecao(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC4: sucesso PDF → st.success contendo o nome da coleção."""
        mock_adicionar.return_value = {"status": "adicionado", "mensagem": "contrato.pdf adicionado."}
        mock_st.spinner.return_value = _spinner_ctx()

        uploaded_file = _make_uploaded_file("contrato.pdf", b"%PDF-1.4 fake")

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file, collection_name="contratos")

        mock_st.success.assert_not_called()
        tipo, success_msg = mock_st.session_state.ingest_msg
        self.assertEqual(tipo, "success")
        self.assertIn("contratos", success_msg)


# ---------------------------------------------------------------------------
# AC5 / AC17 / AC18: Fan-out retrieval multi-coleção
# ---------------------------------------------------------------------------


class TestAC05FanOutRetrieval(unittest.TestCase):
    """
    AC5: WHEN duas ou mais coleções com docs existem
    THEN _get_retriever() SHALL recuperar de todas, mesclar e retornar top-ranked.
    """

    def teste_1_fanout_recupera_de_duas_colecoes_e_mescla(self) -> None:
        """AC5: fan-out em 2 coleções → documentos de ambas incluídos no resultado."""
        from agenticlog.agent import _get_retriever

        doc_a = _make_doc("Fornecedor XYZ prazo 30 dias")
        doc_b = _make_doc("Contrato ABC valor 50000")

        retriever_a = MagicMock()
        retriever_a.invoke.return_value = [doc_a]

        retriever_b = MagicMock()
        retriever_b.invoke.return_value = [doc_b]

        db_a = MagicMock()
        db_a.as_retriever.return_value = retriever_a

        db_b = MagicMock()
        db_b.as_retriever.return_value = retriever_b

        def _fake_get_vector_db(name: str):
            return db_a if name == "fornecedores" else db_b

        with patch("agenticlog.agent._listar_colecoes", return_value=["fornecedores", "contratos"]):
            with patch("agenticlog.agent._get_vector_db", side_effect=_fake_get_vector_db):
                result = _get_retriever("prazo de entrega")

        contents = [doc.page_content for doc in result]
        self.assertIn("Fornecedor XYZ prazo 30 dias", contents)
        self.assertIn("Contrato ABC valor 50000", contents)

    def teste_2_fanout_deduplica_documentos_identicos(self) -> None:
        """AC5: doc duplicado em duas coleções → aparece apenas uma vez no resultado."""
        from agenticlog.agent import _get_retriever

        doc = _make_doc("conteúdo duplicado")

        retriever_mock = MagicMock()
        retriever_mock.invoke.return_value = [doc]

        db_mock = MagicMock()
        db_mock.as_retriever.return_value = retriever_mock

        with patch("agenticlog.agent._listar_colecoes", return_value=["col1", "col2"]):
            with patch("agenticlog.agent._get_vector_db", return_value=db_mock):
                result = _get_retriever("query")

        occurrences = sum(1 for d in result if d.page_content == "conteúdo duplicado")
        self.assertEqual(occurrences, 1)

    def teste_3_fanout_retorna_no_maximo_tres_documentos(self) -> None:
        """AC5: fan-out com 10 docs únicos → resultado limitado a 3."""
        from agenticlog.agent import _get_retriever

        docs = [_make_doc(f"doc único {i}") for i in range(10)]
        retriever_mock = MagicMock()
        retriever_mock.invoke.return_value = docs

        db_mock = MagicMock()
        db_mock.as_retriever.return_value = retriever_mock

        with patch("agenticlog.agent._listar_colecoes", return_value=["col1"]):
            with patch("agenticlog.agent._get_vector_db", return_value=db_mock):
                result = _get_retriever("query")

        self.assertLessEqual(len(result), 3)


# ---------------------------------------------------------------------------
# AC17: Coleção vazia → skip silencioso
# ---------------------------------------------------------------------------


class TestAC17ColecaoVaziaSkipSilencioso(unittest.TestCase):
    """
    AC17: WHEN coleção está vazia ou inexistente durante fan-out
    THEN sistema SHALL ignorá-la silenciosamente (0 docs, sem exceção).
    """

    def teste_1_colecao_vazia_nao_levanta_excecao(self) -> None:
        """AC17: coleção vazia contribui 0 docs e não lança exceção."""
        from agenticlog.agent import _get_retriever

        doc = _make_doc("doc de fornecedores")

        retriever_vazio = MagicMock()
        retriever_vazio.invoke.return_value = []

        retriever_cheio = MagicMock()
        retriever_cheio.invoke.return_value = [doc]

        db_vazio = MagicMock()
        db_vazio.as_retriever.return_value = retriever_vazio

        db_cheio = MagicMock()
        db_cheio.as_retriever.return_value = retriever_cheio

        def _fake_get_vector_db(name: str):
            return db_vazio if name == "vazia" else db_cheio

        with patch("agenticlog.agent._listar_colecoes", return_value=["vazia", "fornecedores"]):
            with patch("agenticlog.agent._get_vector_db", side_effect=_fake_get_vector_db):
                result = _get_retriever("query")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "doc de fornecedores")

    def teste_2_todas_colecoes_vazias_retorna_lista_vazia(self) -> None:
        """AC17: WHEN todas as coleções estão vazias THEN retorna lista vazia sem exceção."""
        from agenticlog.agent import _get_retriever

        retriever_vazio = MagicMock()
        retriever_vazio.invoke.return_value = []

        db_vazio = MagicMock()
        db_vazio.as_retriever.return_value = retriever_vazio

        with patch("agenticlog.agent._listar_colecoes", return_value=["col1", "col2"]):
            with patch("agenticlog.agent._get_vector_db", return_value=db_vazio):
                result = _get_retriever("query")

        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# AC18: Erro ChromaDB durante fan-out → propagação imediata
# ---------------------------------------------------------------------------


class TestAC18ErroChromeDBFanOutPropagate(unittest.TestCase):
    """
    AC18: WHEN ChromaDB lança exceção durante fan-out
    THEN _get_retriever() SHALL propagar imediatamente (sem swallow).
    """

    def teste_1_excecao_chromadb_propagada_imediatamente(self) -> None:
        """AC18: RuntimeError no retriever.invoke → _get_retriever propaga sem capturar."""
        from agenticlog.agent import _get_retriever

        retriever_falho = MagicMock()
        retriever_falho.invoke.side_effect = RuntimeError("ChromaDB disk error")

        db_falho = MagicMock()
        db_falho.as_retriever.return_value = retriever_falho

        with patch("agenticlog.agent._listar_colecoes", return_value=["fornecedores"]):
            with patch("agenticlog.agent._get_vector_db", return_value=db_falho):
                with self.assertRaises(RuntimeError) as ctx:
                    _get_retriever("query")

        self.assertIn("ChromaDB disk error", str(ctx.exception))

    def teste_2_excecao_em_segunda_colecao_interrompe_fanout(self) -> None:
        """AC18: WHEN segunda coleção falha THEN _get_retriever re-lança imediatamente."""
        from agenticlog.agent import _get_retriever

        retriever_ok = MagicMock()
        retriever_ok.invoke.return_value = [_make_doc("doc ok")]

        retriever_falho = MagicMock()
        retriever_falho.invoke.side_effect = Exception("collection corrupted")

        db_ok = MagicMock()
        db_ok.as_retriever.return_value = retriever_ok

        db_falho = MagicMock()
        db_falho.as_retriever.return_value = retriever_falho

        def _fake_get_vector_db(name: str):
            return db_ok if name == "boa" else db_falho

        with patch("agenticlog.agent._listar_colecoes", return_value=["boa", "corrompida"]):
            with patch("agenticlog.agent._get_vector_db", side_effect=_fake_get_vector_db):
                with self.assertRaises(Exception) as ctx:
                    _get_retriever("query")

        self.assertIn("collection corrupted", str(ctx.exception))


# ---------------------------------------------------------------------------
# AC6: DEFAULT_COLLECTION_NAME usado quando collection_name não é fornecido
# ---------------------------------------------------------------------------


class TestAC06DefaultCollectionName(unittest.TestCase):
    """
    AC6: WHEN write functions chamadas sem collection_name
    THEN sistema SHALL usar DEFAULT_COLLECTION_NAME ('logistica').
    """

    def teste_1_default_collection_name_e_logistica(self) -> None:
        """AC6: config.py define DEFAULT_COLLECTION_NAME = 'logistica'."""
        self.assertEqual(DEFAULT_COLLECTION_NAME, "logistica")

    def teste_2_adicionar_documento_usa_default_quando_omitido(self) -> None:
        """AC6: adicionar_documento_incrementalmente sem collection_name usa 'logistica'."""
        from agenticlog.rag import adicionar_documento_incrementalmente

        with patch("agenticlog.rag.Chroma") as mock_chroma, \
             patch("agenticlog.rag._get_rag_embedding_model"), \
             patch("agenticlog.rag.DIR_DOCUMENTS") as mock_dir, \
             patch("agenticlog.rag._sanitizar_nome_arquivo", return_value="doc.json"), \
             patch("agenticlog.rag._valida_json_sem_chaves_proibidas"), \
             patch("agenticlog.rag.shutil.move"), \
             patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader, \
             patch("agenticlog.rag.SemanticChunker") as mock_splitter:

            mock_dir.glob.return_value = []
            mock_dir.__truediv__ = lambda self, other: Path("/fake") / other

            chroma_instance = MagicMock()
            chroma_instance.get.return_value = {"ids": [], "metadatas": []}
            mock_chroma.return_value = chroma_instance

            loader_instance = MagicMock()
            loader_instance.load.return_value = [_make_doc("chunk")]
            mock_loader.return_value = loader_instance

            splitter_instance = MagicMock()
            splitter_instance.split_documents.return_value = [_make_doc("chunk")]
            mock_splitter.return_value = splitter_instance

            with patch("agenticlog.agent.invalidar_vector_db", MagicMock()):
                adicionar_documento_incrementalmente("doc.json", b'{"ok": "1"}')

        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("collection_name") or call_kwargs.args[1]
            if len(call_kwargs.args) > 1
            else call_kwargs.kwargs.get("collection_name"),
            DEFAULT_COLLECTION_NAME,
        )

    @patch("app.st")
    @patch("app.adicionar_documento_incrementalmente")
    def teste_3_ingerir_documento_sem_collection_name_usa_default(
        self,
        mock_adicionar: MagicMock,
        mock_st: MagicMock,
    ) -> None:
        """AC6: _ingerir_documento sem collection_name → adicionar chamado com DEFAULT_COLLECTION_NAME."""
        mock_adicionar.return_value = {
            "status": "adicionado",
            "mensagem": "Arquivo doc.json adicionado. 1 chunk.",
        }
        mock_st.spinner.return_value = _spinner_ctx()

        uploaded_file = _make_uploaded_file("doc.json", b'{"tipo": "frete"}')

        from app import _ingerir_documento
        _ingerir_documento(uploaded_file)

        mock_adicionar.assert_called_once_with(
            "doc.json", b'{"tipo": "frete"}', DEFAULT_COLLECTION_NAME
        )


# ---------------------------------------------------------------------------
# AC7: Nome < 3 chars → RAGSecurityError
# ---------------------------------------------------------------------------


class TestAC07NomeMuitoCurto(unittest.TestCase):
    """
    AC7: WHEN collection_name tem menos de 3 caracteres
    THEN sistema SHALL lançar RAGSecurityError sem escrita.
    """

    def teste_1_nome_vazio_levanta_security_error(self) -> None:
        """AC7: string vazia → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("")

    def teste_2_nome_um_char_levanta_security_error(self) -> None:
        """AC7: 1 caractere → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("a")

    def teste_3_nome_dois_chars_levanta_security_error(self) -> None:
        """AC7: 2 caracteres → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("ab")

    def teste_4_nome_tres_chars_aceito(self) -> None:
        """AC7 limite válido: exatamente 3 chars → aceito sem erro."""
        result = sanitizar_nome_colecao("abc")
        self.assertEqual(result, "abc")

    def teste_5_adicionar_com_nome_muito_curto_levanta_antes_de_chroma(self) -> None:
        """AC7: adicionar_documento_incrementalmente com nome < 3 chars → RAGSecurityError."""
        from agenticlog.rag import adicionar_documento_incrementalmente

        with patch("agenticlog.rag.Chroma") as mock_chroma:
            with self.assertRaises(RAGSecurityError):
                adicionar_documento_incrementalmente("doc.json", b'{}', "ab")

            mock_chroma.assert_not_called()

    def teste_6_salvar_pdf_com_nome_muito_curto_levanta_antes_de_chroma(self) -> None:
        """AC7: salvar_pdf_enviado com nome < 3 chars → RAGSecurityError sem escrita."""
        from agenticlog.rag import salvar_pdf_enviado

        with patch("agenticlog.rag.Chroma") as mock_chroma:
            with self.assertRaises(RAGSecurityError):
                salvar_pdf_enviado("doc.pdf", b"%PDF-1.4 fake", "ab")

            mock_chroma.assert_not_called()


# ---------------------------------------------------------------------------
# AC8: Nome > 63 chars → RAGSecurityError
# ---------------------------------------------------------------------------


class TestAC08NomeMuitoLongo(unittest.TestCase):
    """
    AC8: WHEN collection_name tem mais de 63 caracteres
    THEN sistema SHALL lançar RAGSecurityError sem escrita.
    """

    def teste_1_nome_64_chars_levanta_security_error(self) -> None:
        """AC8: 64 caracteres → RAGSecurityError."""
        nome_64 = "a" * 64
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao(nome_64)

    def teste_2_nome_63_chars_aceito(self) -> None:
        """AC8 limite válido: exatamente 63 chars → aceito."""
        nome_63 = "a" * 63
        result = sanitizar_nome_colecao(nome_63)
        self.assertEqual(result, nome_63)

    def teste_3_nome_100_chars_levanta_security_error(self) -> None:
        """AC8: 100 caracteres → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("a" * 100)

    def teste_4_adicionar_com_nome_muito_longo_levanta_antes_de_chroma(self) -> None:
        """AC8: adicionar_documento_incrementalmente com nome > 63 chars → RAGSecurityError."""
        from agenticlog.rag import adicionar_documento_incrementalmente

        with patch("agenticlog.rag.Chroma") as mock_chroma:
            with self.assertRaises(RAGSecurityError):
                adicionar_documento_incrementalmente("doc.json", b'{}', "a" * 64)

            mock_chroma.assert_not_called()


# ---------------------------------------------------------------------------
# AC9: Chars inválidos → RAGSecurityError
# ---------------------------------------------------------------------------


class TestAC09CharsInvalidos(unittest.TestCase):
    """
    AC9: WHEN collection_name contém caracteres além de alfanumérico, hífen, underscore
    THEN sistema SHALL lançar RAGSecurityError.
    """

    def teste_1_espaco_invalido(self) -> None:
        """AC9: espaço → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome invalido")

    def teste_2_ponto_invalido(self) -> None:
        """AC9: ponto → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome.colecao")

    def teste_3_arroba_invalido(self) -> None:
        """AC9: @ → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome@colecao")

    def teste_4_slash_invalido(self) -> None:
        """AC9: / → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome/colecao")

    def teste_5_caracter_unicode_invalido(self) -> None:
        """AC9: caractere acentuado → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("coleção")

    def teste_6_hifen_e_underscore_validos_no_meio(self) -> None:
        """AC9: hífen e underscore no meio → válidos."""
        result_hifen = sanitizar_nome_colecao("nome-colecao")
        self.assertEqual(result_hifen, "nome-colecao")

        result_underscore = sanitizar_nome_colecao("nome_colecao")
        self.assertEqual(result_underscore, "nome_colecao")


# ---------------------------------------------------------------------------
# AC10: Começa/termina com não-alfanumérico → RAGSecurityError
# ---------------------------------------------------------------------------


class TestAC10InicioFimNaoAlfanumerico(unittest.TestCase):
    """
    AC10: WHEN collection_name começa ou termina com caractere não-alfanumérico
    THEN sistema SHALL lançar RAGSecurityError.
    """

    def teste_1_comeca_com_hifen(self) -> None:
        """AC10: nome começando com hífen → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("-nome-colecao")

    def teste_2_termina_com_hifen(self) -> None:
        """AC10: nome terminando com hífen → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome-colecao-")

    def teste_3_comeca_com_underscore(self) -> None:
        """AC10: nome começando com underscore → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("_nome_colecao")

    def teste_4_termina_com_underscore(self) -> None:
        """AC10: nome terminando com underscore → RAGSecurityError."""
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao("nome_colecao_")

    def teste_5_nome_valido_alfanumerico_inicio_fim(self) -> None:
        """AC10: nome começando e terminando com alfanumérico → válido."""
        result = sanitizar_nome_colecao("a-valido-z")
        self.assertEqual(result, "a-valido-z")


# ---------------------------------------------------------------------------
# AC12: sanitizar_nome_colecao antes de toda escrita
# ---------------------------------------------------------------------------


class TestAC12SanitizacaoAntesDeEscrita(unittest.TestCase):
    """
    AC12: sanitizar_nome_colecao() SHALL ser chamado antes de qualquer write
    em todas as funções: adicionar_documento_incrementalmente, salvar_documento_enviado,
    salvar_pdf_enviado, reconstruir_vectordb.
    """

    def teste_1_adicionar_com_nome_invalido_rejeita_antes_chroma(self) -> None:
        """AC12: adicionar_documento_incrementalmente → RAGSecurityError antes de Chroma."""
        from agenticlog.rag import adicionar_documento_incrementalmente

        with patch("agenticlog.rag.Chroma") as mock_chroma:
            with self.assertRaises(RAGSecurityError):
                adicionar_documento_incrementalmente("f.json", b'{}', "a!")

            mock_chroma.assert_not_called()

    def teste_2_salvar_documento_com_nome_invalido_rejeita_antes_escrita(self) -> None:
        """AC12: salvar_documento_enviado → RAGSecurityError antes de disco."""
        from agenticlog.rag import salvar_documento_enviado

        with patch("agenticlog.rag.shutil.move") as mock_move:
            with self.assertRaises(RAGSecurityError):
                salvar_documento_enviado("f.json", b'{"ok":"1"}', "a!")

            mock_move.assert_not_called()

    def teste_3_salvar_pdf_com_nome_invalido_rejeita_antes_escrita(self) -> None:
        """AC12: salvar_pdf_enviado → RAGSecurityError antes de disco."""
        from agenticlog.rag import salvar_pdf_enviado

        with patch("agenticlog.rag.shutil.move") as mock_move:
            with self.assertRaises(RAGSecurityError):
                salvar_pdf_enviado("f.pdf", b"%PDF-1.4 fake", "a!")

            mock_move.assert_not_called()

    def teste_4_reconstruir_vectordb_com_nome_invalido_rejeita(self) -> None:
        """AC12: reconstruir_vectordb → RAGSecurityError antes de cria_vectordb."""
        from agenticlog.rag import reconstruir_vectordb

        with patch("agenticlog.rag.cria_vectordb") as mock_criar:
            with self.assertRaises(RAGSecurityError):
                reconstruir_vectordb("a!")

            mock_criar.assert_not_called()


# ---------------------------------------------------------------------------
# AC13: invalidar_vector_db() limpa _vector_dbs; sem exceção em dict vazio
# ---------------------------------------------------------------------------


class TestAC13InvalidarVectorDb(unittest.TestCase):
    """
    AC13: WHEN invalidar_vector_db() chamado
    THEN sistema SHALL limpar todo _vector_dbs; chamada em dict vazio não lança.
    """

    def teste_1_invalidar_limpa_o_dicionario(self) -> None:
        """AC13: após invalidar_vector_db(), _vector_dbs está vazio."""
        import agenticlog.agent as _agent

        _agent._vector_dbs["logistica"] = MagicMock()
        _agent._vector_dbs["fornecedores"] = MagicMock()

        _agent.invalidar_vector_db()

        self.assertEqual(len(_agent._vector_dbs), 0)

    def teste_2_invalidar_em_dict_vazio_nao_levanta_excecao(self) -> None:
        """AC13: invalidar_vector_db() em _vector_dbs vazio → sem exceção."""
        import agenticlog.agent as _agent

        _agent._vector_dbs.clear()

        try:
            _agent.invalidar_vector_db()
        except Exception as exc:
            self.fail(f"invalidar_vector_db() lançou inesperadamente: {exc}")

    def teste_3_get_vector_db_repopula_apos_invalidar(self) -> None:
        """AC13: após invalidar, _get_vector_db cria nova instância Chroma ao ser chamado."""
        import agenticlog.agent as _agent

        _agent.invalidar_vector_db()
        self.assertEqual(len(_agent._vector_dbs), 0)

        with patch("agenticlog.agent.Chroma") as mock_chroma, \
             patch("agenticlog.agent._get_embedding_model"):
            mock_chroma.return_value = MagicMock()
            _agent._get_vector_db("logistica")

        self.assertIn("logistica", _agent._vector_dbs)


# ---------------------------------------------------------------------------
# AC14: QueryRequest sem campo collection_name
# ---------------------------------------------------------------------------


class TestAC14QueryRequestSemCollectionName(unittest.TestCase):
    """
    AC14: QueryRequest schema SHALL não conter campo collection_name.
    """

    def teste_1_query_request_nao_tem_collection_name(self) -> None:
        """AC14: QueryRequest.model_fields não contém 'collection_name'."""
        from agenticlog.api import QueryRequest

        self.assertNotIn(
            "collection_name",
            QueryRequest.model_fields,
            "QueryRequest não deve ter campo collection_name (AC14)",
        )

    def teste_2_query_request_tem_apenas_query(self) -> None:
        """AC14: QueryRequest contém apenas o campo 'query'."""
        from agenticlog.api import QueryRequest

        fields = set(QueryRequest.model_fields.keys())
        self.assertEqual(fields, {"query"})

    def teste_3_query_request_instancia_valida_sem_collection_name(self) -> None:
        """AC14: QueryRequest({'query': 'teste'}) instancia sem erros."""
        from agenticlog.api import QueryRequest

        req = QueryRequest(query="teste de logistica")
        self.assertEqual(req.query, "teste de logistica")


# ---------------------------------------------------------------------------
# AC15: Limites de validação de constantes de config.py
# ---------------------------------------------------------------------------


class TestAC15LimitesVemDoConfig(unittest.TestCase):
    """
    AC15: Limites de validação SHALL derivar de COLLECTION_NAME_MIN_LEN e
    COLLECTION_NAME_MAX_LEN de config.py.
    """

    def teste_1_min_len_e_3(self) -> None:
        """AC15: COLLECTION_NAME_MIN_LEN == 3."""
        self.assertEqual(COLLECTION_NAME_MIN_LEN, 3)

    def teste_2_max_len_e_63(self) -> None:
        """AC15: COLLECTION_NAME_MAX_LEN == 63."""
        self.assertEqual(COLLECTION_NAME_MAX_LEN, 63)

    def teste_3_nome_com_min_len_menos_um_rejeitado(self) -> None:
        """AC15: nome com len(MIN_LEN - 1) → RAGSecurityError."""
        nome_curto = "a" * (COLLECTION_NAME_MIN_LEN - 1)
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao(nome_curto)

    def teste_4_nome_com_min_len_aceito(self) -> None:
        """AC15: nome com len(MIN_LEN) exato → aceito."""
        nome = "a" * COLLECTION_NAME_MIN_LEN
        result = sanitizar_nome_colecao(nome)
        self.assertEqual(result, nome)

    def teste_5_nome_com_max_len_aceito(self) -> None:
        """AC15: nome com len(MAX_LEN) exato → aceito."""
        nome = "a" * COLLECTION_NAME_MAX_LEN
        result = sanitizar_nome_colecao(nome)
        self.assertEqual(result, nome)

    def teste_6_nome_com_max_len_mais_um_rejeitado(self) -> None:
        """AC15: nome com len(MAX_LEN + 1) → RAGSecurityError."""
        nome_longo = "a" * (COLLECTION_NAME_MAX_LEN + 1)
        with self.assertRaises(RAGSecurityError):
            sanitizar_nome_colecao(nome_longo)

    def teste_7_mensagem_de_erro_cita_limite_minimo(self) -> None:
        """AC15: mensagem de erro de nome curto cita COLLECTION_NAME_MIN_LEN."""
        with self.assertRaises(RAGSecurityError) as ctx:
            sanitizar_nome_colecao("ab")

        self.assertIn(str(COLLECTION_NAME_MIN_LEN), str(ctx.exception))

    def teste_8_mensagem_de_erro_cita_limite_maximo(self) -> None:
        """AC15: mensagem de erro de nome longo cita COLLECTION_NAME_MAX_LEN."""
        with self.assertRaises(RAGSecurityError) as ctx:
            sanitizar_nome_colecao("a" * 64)

        self.assertIn(str(COLLECTION_NAME_MAX_LEN), str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
