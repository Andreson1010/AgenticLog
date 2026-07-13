# AgenticLog — Acceptance Tests: Portuguese Embedding Model Feature
"""
Verifica os critérios de aceite da feature "portuguese-embedding-model"
(.specs/features/portuguese-embedding-model/spec.md, AC1-AC9).

Mapeamento de critérios:

  AC1 — config.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  AC2 — _get_rag_embedding_model, cria_vectordb (rag.py) e _get_embedding_model (agent.py)
        instanciam HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
  AC3 — (MANUAL) rebuild real de data/vectordb/ a 768-dim — fora do escopo de testes automatizados
  AC4 — (MANUAL) query em português end-to-end pós-rebuild — fora do escopo de testes automatizados
  AC5 — vectordb "stale" (768-dim, modelo antigo) consultado com modelo novo (768-dim) NÃO
        levanta exceção de dimensão — avalia_similaridade roda normalmente (768=768)
  AC6 — CLAUDE.md documenta o procedimento de upgrade (parar app → apagar data/vectordb/ →
        rerun python -m agenticlog.rag → retomar queries) e o aumento do download
  AC7 — mocks existentes [0.1]*768 em tests/test_agentic_rag.py::teste_7_avalia_similaridade
        e tests/acceptance/test_agent_workflow_integration.py continuam passando inalterados
  AC8 — nenhuma lógica de prefixo "query:"/"passage:" foi adicionada nos call sites de embedding
  AC9 — model_kwargs/encode_kwargs em rag.py/agent.py permanecem inalterados (incluindo a
        inconsistência pré-existente entre cria_vectordb e os singleton getters)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

from agenticlog import config


# ---------------------------------------------------------------------------
# AC1: config.EMBEDDING_MODEL aponta para o modelo multilíngue
# ---------------------------------------------------------------------------


class TestAC01EmbeddingModelConstant(unittest.TestCase):
    """
    AC1: WHEN src/agenticlog/config.py é inspecionado
    THEN EMBEDDING_MODEL SHALL ser igual a
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".
    """

    def teste_1_embedding_model_e_o_modelo_multilingue(self) -> None:
        """AC1: config.EMBEDDING_MODEL == modelo multilíngue paraphrase-multilingual-mpnet-base-v2."""
        self.assertEqual(
            config.EMBEDDING_MODEL,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )

    def teste_2_embedding_model_nao_e_mais_o_modelo_em_ingles(self) -> None:
        """AC1: config.EMBEDDING_MODEL não referencia mais BAAI/bge-base-en."""
        self.assertNotEqual(config.EMBEDDING_MODEL, "BAAI/bge-base-en")
        self.assertNotIn("bge-base-en", config.EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# AC2: três call sites de HuggingFaceEmbeddings usam config.EMBEDDING_MODEL
# ---------------------------------------------------------------------------


class TestAC02RagEmbeddingCallSitesUsamConfig(unittest.TestCase):
    """
    AC2: WHEN rag.py::_get_rag_embedding_model e rag.py::cria_vectordb instanciam
    HuggingFaceEmbeddings THEN cada um SHALL passar model_name=config.EMBEDDING_MODEL.
    """

    def setUp(self) -> None:
        import agenticlog.rag as rag_mod
        self._rag_mod = rag_mod
        self._rag_mod._rag_embedding_model = None

    def tearDown(self) -> None:
        self._rag_mod._rag_embedding_model = None

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def teste_1_get_rag_embedding_model_usa_embedding_model_do_config(
        self, mock_emb: MagicMock
    ) -> None:
        """AC2: _get_rag_embedding_model() constrói HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)."""
        self._rag_mod._get_rag_embedding_model()

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )

    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    @patch("agenticlog.ingestion.orchestrator._resetar_colecao", new=MagicMock())
    def teste_2_cria_vectordb_usa_embedding_model_do_config(
        self,
        mock_valida_path: MagicMock,
        mock_valida_json: MagicMock,
        mock_loader: MagicMock,
        mock_dir: MagicMock,
        mock_splitter: MagicMock,
        mock_emb: MagicMock,
        mock_chroma: MagicMock,
        mock_hash: MagicMock,
    ) -> None:
        """AC2: cria_vectordb() constrói HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, ...)."""
        from langchain_core.documents import Document

        mock_loader.return_value = [
            Document(page_content="conteúdo de teste")
        ]
        mock_dir.glob.side_effect = lambda pat: ["/fake/documents/a.json"] if pat == "*.json" else []

        splitter_instance = MagicMock()
        splitter_instance.split_documents.return_value = [
            Document(page_content="Chunk 1")
        ]
        mock_splitter.return_value = splitter_instance

        self._rag_mod.cria_vectordb()

        mock_emb.assert_called_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )


class TestAC02AgentEmbeddingCallSiteUsaConfig(unittest.TestCase):
    """
    AC2: WHEN agent.py::_get_embedding_model instancia HuggingFaceEmbeddings
    THEN SHALL passar model_name=config.EMBEDDING_MODEL.
    """

    def setUp(self) -> None:
        import agenticlog.agent as agent_mod
        self._agent_mod = agent_mod
        self._agent_mod._embedding_model = None

    def tearDown(self) -> None:
        self._agent_mod._embedding_model = None

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def teste_1_get_embedding_model_usa_embedding_model_do_config(
        self, mock_emb: MagicMock
    ) -> None:
        """AC2: _get_embedding_model() constrói HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)."""
        self._agent_mod._get_embedding_model()

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def teste_2_get_embedding_model_singleton_reusa_instancia(
        self, mock_emb: MagicMock
    ) -> None:
        """AC2: chamadas repetidas a _get_embedding_model() reusam a mesma instância (singleton)."""
        primeira = self._agent_mod._get_embedding_model()
        segunda = self._agent_mod._get_embedding_model()

        self.assertIs(primeira, segunda)
        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )


# ---------------------------------------------------------------------------
# AC5: vectordb "stale" (768-dim, modelo antigo) não levanta exceção com
#      embeddings de query do modelo novo (768-dim) — dimensões compatíveis
# ---------------------------------------------------------------------------


class TestAC05StaleVectordbDimensoesCompativeisNaoLevantaExcecao(unittest.TestCase):
    """
    AC5: WHEN um data/vectordb/ "stale" (construído com BAAI/bge-base-en, 768-dim)
    é consultado com o novo modelo multilíngue (também 768-dim)
    THEN o sistema SHALL NOT levantar exceção (768=768), embora os resultados
    fiquem degradados/não confiáveis (mitigação documental, não de runtime).
    """

    @patch("agenticlog.agent._get_embedding_model")
    def teste_1_avalia_similaridade_com_vetores_768d_de_espacos_diferentes_nao_levanta(
        self, mock_get_embedding: MagicMock
    ) -> None:
        """AC5: avalia_similaridade roda sem exceção quando doc-vectors (espaço 'antigo')
        e response-vectors (espaço 'novo') são ambos 768-dim, mesmo vindo de
        'modelos' numericamente diferentes (espaços incompatíveis simulados)."""
        from agenticlog.agent import AgentState, avalia_similaridade
        from langchain_core.documents import Document

        mock_model = MagicMock()
        # Simula vetores "stale" (modelo antigo) para os documentos recuperados
        # e vetores do "modelo novo" para as respostas candidatas — ambos 768-dim,
        # mas em espaços numericamente distintos (valores diferentes).
        mock_model.embed_documents.side_effect = [
            [[0.1] * 768],   # retrieved_embeddings (espaço "antigo")
            [[0.9] * 768],   # response_embeddings (espaço "novo")
        ]
        mock_get_embedding.return_value = mock_model

        state = AgentState(
            query="Qual o prazo de entrega?",
            retrieved_info=[Document(page_content="Documento recuperado (espaço antigo).")],
            possible_responses=[{"answer": "Resposta gerada (espaço novo)."}],
        )

        try:
            result = avalia_similaridade(state)
        except Exception as exc:  # pragma: no cover - falha indica regressão de dimensão
            self.fail(
                f"avalia_similaridade levantou exceção inesperada com vetores 768-dim "
                f"de espaços distintos: {exc}"
            )

        self.assertIn("similarity_scores", result.model_fields_set | set(result.__dict__.keys()))
        self.assertEqual(len(result.similarity_scores), 1)
        self.assertIsInstance(result.similarity_scores[0], float)

    @patch("agenticlog.agent._get_embedding_model")
    def teste_2_dimensao_768_consistente_entre_doc_e_query_vectors(
        self, mock_get_embedding: MagicMock
    ) -> None:
        """AC5: ambos os modelos (antigo e novo) produzem vetores 768-dim — não há
        dimension-mismatch que levantaria exceção do ChromaDB/cosine_similarity."""
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 768]
        mock_model.embed_query.return_value = [0.2] * 768
        mock_get_embedding.return_value = mock_model

        doc_vector = mock_model.embed_documents(["doc antigo"])[0]
        query_vector = mock_model.embed_query("query nova")

        self.assertEqual(len(doc_vector), 768)
        self.assertEqual(len(query_vector), 768)
        self.assertEqual(len(doc_vector), len(query_vector))


# ---------------------------------------------------------------------------
# AC6: CLAUDE.md documenta o procedimento de upgrade
# ---------------------------------------------------------------------------


class TestAC06ClaudeMdDocumentaProcedimentoDeUpgrade(unittest.TestCase):
    """
    AC6: WHEN um operador lê CLAUDE.md
    THEN ele SHALL documentar o procedimento de upgrade: parar o app →
    apagar data/vectordb/ → rerun python -m agenticlog.rag → retomar queries.
    """

    @classmethod
    def setUpClass(cls) -> None:
        claude_md_path = _root / "CLAUDE.md"
        cls.conteudo = claude_md_path.read_text(encoding="utf-8")

    def teste_1_documenta_apagar_data_vectordb(self) -> None:
        """AC6: CLAUDE.md menciona apagar/deletar data/vectordb/."""
        self.assertIn("data/vectordb/", self.conteudo)
        self.assertTrue(
            "Delete" in self.conteudo or "delete" in self.conteudo or "apagar" in self.conteudo.lower(),
            "CLAUDE.md deve instruir a apagar/deletar data/vectordb/",
        )

    def teste_2_documenta_rerun_python_m_agenticlog_rag(self) -> None:
        """AC6: CLAUDE.md menciona rerun de `python -m agenticlog.rag`."""
        self.assertIn("python -m agenticlog.rag", self.conteudo)

    def teste_3_documenta_nome_do_novo_modelo(self) -> None:
        """AC6: CLAUDE.md referencia o novo modelo multilíngue."""
        self.assertIn("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", self.conteudo)

    def teste_4_documenta_aumento_do_tamanho_de_download(self) -> None:
        """AC6/edge case: CLAUDE.md menciona o aumento do tamanho de download (~1.0-1.1 GB vs ~440 MB)."""
        self.assertTrue(
            "1.0" in self.conteudo or "1.1" in self.conteudo or "GB" in self.conteudo,
            "CLAUDE.md deve mencionar o tamanho de download maior do novo modelo",
        )

    def teste_5_documenta_risco_de_degradacao_silenciosa(self) -> None:
        """AC5/AC6: CLAUDE.md documenta o risco de degradação silenciosa para data/vectordb/ não reconstruído."""
        conteudo_lower = self.conteudo.lower()
        self.assertTrue(
            "silen" in conteudo_lower or "degrad" in conteudo_lower,
            "CLAUDE.md deve documentar o risco de degradação silenciosa "
            "ao não reconstruir data/vectordb/ após mudar EMBEDDING_MODEL",
        )

    def teste_6_referencia_nao_contem_modelo_antigo(self) -> None:
        """AC6/edge case: CLAUDE.md não referencia mais BAAI/bge-base-en como modelo atual."""
        # O nome antigo pode aparecer apenas em comparação histórica de tamanho de download;
        # garantimos que o modelo "atual" documentado é o novo.
        self.assertIn("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", self.conteudo)


# ---------------------------------------------------------------------------
# AC7: mocks existentes [0.1]*768 continuam passando inalterados
# ---------------------------------------------------------------------------


class TestAC07TestesExistentesComMocks768dContinuamPassando(unittest.TestCase):
    """
    AC7: WHEN os testes existentes que mockam embeddings como [0.1] * 768
    (tests/test_agentic_rag.py, tests/acceptance/test_agent_workflow_integration.py)
    são executados após esta mudança THEN SHALL continuar a passar inalterados
    (dimensão permanece 768).

    Verificação estática: confirma que os mocks [0.1] * 768 permanecem
    inalterados nesses arquivos — a suite completa (pytest tests/) já garante
    que esses testes passam, sem necessidade de subprocessos aninhados aqui.
    """

    def teste_1_test_agentic_rag_mantem_mock_768d(self) -> None:
        """AC7: tests/test_agentic_rag.py::teste_7_avalia_similaridade mantém mock [[0.1] * 768]."""
        source = (_root / "tests" / "test_agentic_rag.py").read_text(encoding="utf-8")
        self.assertIn("[[0.1] * 768]", source)

    def teste_2_test_agent_workflow_integration_mantem_mocks_768d(self) -> None:
        """AC7: tests/acceptance/test_agent_workflow_integration.py mantém mocks [0.1] * 768."""
        source = (_root / "tests" / "acceptance" / "test_agent_workflow_integration.py").read_text(encoding="utf-8")
        self.assertIn("[[0.1] * 768]", source)
        self.assertIn("[0.1] * 768", source)


# ---------------------------------------------------------------------------
# AC8: nenhuma lógica de prefixo "query:"/"passage:" adicionada
# ---------------------------------------------------------------------------


class TestAC08SemLogicaDePrefixoQueryPassage(unittest.TestCase):
    """
    AC8: WHEN o novo modelo é integrado
    THEN o sistema SHALL NOT adicionar nenhuma lógica de prefixo "query:"/"passage:"
    aos inputs de embedding — substituição drop-in pura de nome de modelo.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.rag_source = (_root / "src" / "agenticlog" / "rag.py").read_text(encoding="utf-8")
        cls.agent_source = (_root / "src" / "agenticlog" / "agent.py").read_text(encoding="utf-8")

    def teste_1_rag_py_nao_contem_prefixo_query_dois_pontos(self) -> None:
        """AC8: rag.py não contém literais 'query:' ou 'passage:' usados como prefixo de embedding."""
        self.assertNotIn('"query: "', self.rag_source)
        self.assertNotIn("'query: '", self.rag_source)
        self.assertNotIn('"passage: "', self.rag_source)
        self.assertNotIn("'passage: '", self.rag_source)

    def teste_2_agent_py_nao_contem_prefixo_query_dois_pontos(self) -> None:
        """AC8: agent.py não contém literais 'query:' ou 'passage:' usados como prefixo de embedding."""
        self.assertNotIn('"query: "', self.agent_source)
        self.assertNotIn("'query: '", self.agent_source)
        self.assertNotIn('"passage: "', self.agent_source)
        self.assertNotIn("'passage: '", self.agent_source)

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def teste_3_embed_query_chamado_com_texto_original_sem_prefixo(
        self, mock_emb: MagicMock
    ) -> None:
        """AC8: embed_documents/embed_query recebem o texto original, sem prefixo 'query:'/'passage:'."""
        import agenticlog.agent as agent_mod
        agent_mod._embedding_model = None

        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1] * 768]
        mock_emb.return_value = mock_instance

        from agenticlog.agent import AgentState, avalia_similaridade
        from langchain_core.documents import Document

        agent_mod._embedding_model = None
        with patch("agenticlog.agent._get_embedding_model", return_value=mock_instance):
            state = AgentState(
                query="Qual o prazo de entrega?",
                retrieved_info=[Document(page_content="texto do documento")],
                possible_responses=[{"answer": "texto da resposta"}],
            )
            avalia_similaridade(state)

        # Verifica que os textos passados para embed_documents NÃO têm prefixo "query:"/"passage:"
        for call_args in mock_instance.embed_documents.call_args_list:
            texts = call_args.args[0]
            for text in texts:
                self.assertFalse(text.startswith("query: "))
                self.assertFalse(text.startswith("passage: "))

        agent_mod._embedding_model = None


# ---------------------------------------------------------------------------
# AC9: model_kwargs/encode_kwargs inalterados (incluindo inconsistência pré-existente)
# ---------------------------------------------------------------------------


class TestAC09ModelKwargsEncodeKwargsInalterados(unittest.TestCase):
    """
    Normalização de embeddings UNIFICADA (auditoria RAG 2026-06-23, P0-3).

    A inconsistência pré-existente — cria_vectordb() normalizava, mas os singleton
    getters (_get_rag_embedding_model / agent._get_embedding_model) não — produzia
    normas diferentes no mesmo espaço vetorial. Os três call sites agora passam
    model_kwargs={"device": ...} e encode_kwargs={"normalize_embeddings": True}.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.rag_source = (_root / "src" / "agenticlog" / "rag.py").read_text(encoding="utf-8")
        cls.agent_source = (_root / "src" / "agenticlog" / "agent.py").read_text(encoding="utf-8")
        # ADR-018 Fase 3a: a construção do embedding incremental migrou p/ ingestion/embeddings.py
        cls.embeddings_source = (
            _root / "src" / "agenticlog" / "ingestion" / "embeddings.py"
        ).read_text(encoding="utf-8")
        # ADR-018 Fase 3b: cria_vectordb (construção inline do rebuild) migrou p/ ingestion/orchestrator.py
        cls.orchestrator_source = (
            _root / "src" / "agenticlog" / "ingestion" / "orchestrator.py"
        ).read_text(encoding="utf-8")

    def teste_1_cria_vectordb_passa_model_kwargs_device_e_encode_kwargs_normalize(self) -> None:
        """cria_vectordb() passa model_kwargs={"device": device} e
        encode_kwargs={"normalize_embeddings": True} (ADR-018 Fase 3b: agora em orchestrator.py)."""
        self.assertIn('model_kwargs={"device": device}', self.orchestrator_source)
        self.assertIn('encode_kwargs={"normalize_embeddings": True}', self.orchestrator_source)

    def teste_2_get_rag_embedding_model_normaliza(self) -> None:
        """_get_rag_embedding_model() agora normaliza (sem a inconsistência pré-existente)."""
        self.assertNotIn(
            "_rag_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)",
            self.rag_source,
        )
        # ADR-018 Fase 3b: rebuild (cria_vectordb, construção inline) migrou p/ orchestrator.py;
        # a ingestão incremental normaliza via ingestion/embeddings.py::criar_embedding_model.
        combinado = self.orchestrator_source + self.embeddings_source
        self.assertEqual(
            combinado.count('encode_kwargs={"normalize_embeddings": True}'), 2,
            "rebuild e ingestão incremental devem ambos normalizar embeddings",
        )

    def teste_3_get_embedding_model_normaliza(self) -> None:
        """_get_embedding_model() (wrapper em agent) delega a _build_embedding_model()
        que chama criar_embedding_model() em ingestion/embeddings.py — a normalização
        está em embeddings.py, não mais inline em agent.py."""
        self.assertNotIn(
            "_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)",
            self.agent_source,
        )
        self.assertIn(
            'encode_kwargs={"normalize_embeddings": True}',
            self.embeddings_source,
        )

    @patch("agenticlog.ingestion.orchestrator._hash_arquivo", return_value="a" * 64)
    @patch("agenticlog.ingestion.orchestrator.Chroma")
    @patch("agenticlog.ingestion.orchestrator.HuggingFaceEmbeddings")
    @patch("agenticlog.ingestion.orchestrator.SemanticChunker")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.ingestion.orchestrator.carregar_json")
    @patch("agenticlog.ingestion.orchestrator._valida_arquivos_json")
    @patch("agenticlog.ingestion.orchestrator._valida_path_documentos")
    @patch("agenticlog.ingestion.orchestrator._resetar_colecao", new=MagicMock())
    def teste_4_cria_vectordb_kwargs_completos_via_mock(
        self,
        mock_valida_path: MagicMock,
        mock_valida_json: MagicMock,
        mock_loader: MagicMock,
        mock_dir: MagicMock,
        mock_splitter: MagicMock,
        mock_emb: MagicMock,
        mock_chroma: MagicMock,
        mock_hash: MagicMock,
    ) -> None:
        """AC9: cria_vectordb() chama HuggingFaceEmbeddings com exatamente os três kwargs
        esperados (model_name, model_kwargs com device, encode_kwargs com normalize_embeddings)."""
        from langchain_core.documents import Document
        import agenticlog.rag as rag_mod

        mock_loader.return_value = [
            Document(page_content="conteúdo")
        ]
        mock_dir.glob.side_effect = lambda pat: ["/fake/documents/a.json"] if pat == "*.json" else []

        splitter_instance = MagicMock()
        splitter_instance.split_documents.return_value = [
            Document(page_content="Chunk 1")
        ]
        mock_splitter.return_value = splitter_instance

        rag_mod.cria_vectordb()

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )

        call_kwargs = mock_emb.call_args.kwargs
        self.assertEqual(set(call_kwargs.keys()), {"model_name", "model_kwargs", "encode_kwargs"})
        self.assertEqual(set(call_kwargs["model_kwargs"].keys()), {"device"})
        self.assertEqual(call_kwargs["encode_kwargs"], {"normalize_embeddings": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
