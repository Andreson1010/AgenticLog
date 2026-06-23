# AgenticLog - Testes unitários para agent.py
"""Testes para funções públicas de agenticlog.agent."""

import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.agent as agent_mod
import agenticlog.config as config
from agenticlog.agent import invalidar_vector_db
from langchain_core.documents import Document as LCDocument


class TestGetEmbeddingModel(unittest.TestCase):
    """Testes para _get_embedding_model (PORTPT-02 / AC2)."""

    def setUp(self) -> None:
        """Reseta o singleton antes de cada teste para garantir isolamento."""
        agent_mod._embedding_model = None

    def tearDown(self) -> None:
        """Reseta o singleton após cada teste."""
        agent_mod._embedding_model = None

    @patch("agenticlog.agent.HuggingFaceEmbeddings")
    def test_get_embedding_model_usa_embedding_model_do_config(self, mock_emb):
        """_get_embedding_model() constrói HuggingFaceEmbeddings com model_name=EMBEDDING_MODEL."""
        agent_mod._get_embedding_model()

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )

    @patch("agenticlog.agent.HuggingFaceEmbeddings")
    def test_get_embedding_model_singleton_reusa_instancia(self, mock_emb):
        """Chamadas subsequentes retornam a mesma instância sem recriar HuggingFaceEmbeddings."""
        primeira = agent_mod._get_embedding_model()
        segunda = agent_mod._get_embedding_model()

        self.assertIs(primeira, segunda)
        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )


class TestInvalidarVectorDb(unittest.TestCase):
    """Testes para invalidar_vector_db() com semântica de dicionário."""

    def setUp(self) -> None:
        """Limpa o dicionário antes de cada teste para garantir isolamento."""
        agent_mod._vector_dbs.clear()

    def tearDown(self) -> None:
        """Limpa o dicionário após cada teste."""
        agent_mod._vector_dbs.clear()

    def teste_1_invalidar_limpa_dict(self) -> None:
        """invalidar_vector_db() deve esvaziar o dicionário _vector_dbs."""
        agent_mod._vector_dbs["logistica"] = MagicMock()
        agent_mod._vector_dbs["fornecedores"] = MagicMock()
        invalidar_vector_db()
        self.assertEqual(agent_mod._vector_dbs, {})

    def teste_2_invalidar_dict_vazio_nao_levanta(self) -> None:
        """invalidar_vector_db() com dicionário já vazio não deve lançar exceção."""
        self.assertEqual(agent_mod._vector_dbs, {})
        invalidar_vector_db()  # não deve levantar
        self.assertEqual(agent_mod._vector_dbs, {})


class TestGetRetriever(unittest.TestCase):
    """Testes para _get_retriever() com fan-out multi-coleção (MCC-12 a MCC-15)."""

    def setUp(self) -> None:
        """Limpa o cache de vector_dbs antes de cada teste."""
        agent_mod._vector_dbs.clear()

    def tearDown(self) -> None:
        """Limpa o cache de vector_dbs após cada teste."""
        agent_mod._vector_dbs.clear()

    def _make_doc(self, content: str) -> LCDocument:
        return LCDocument(page_content=content, metadata={})

    def teste_1_fanout_duas_colecoes_mescla_docs(self) -> None:
        """Fan-out em 2 coleções retorna docs mesclados de ambas (até 3 únicos)."""
        doc_a = self._make_doc("fornecedor alfa")
        doc_b = self._make_doc("contrato beta")

        mock_retriever_a = MagicMock()
        mock_retriever_a.invoke.return_value = [doc_a]
        mock_db_a = MagicMock()
        mock_db_a.as_retriever.return_value = mock_retriever_a

        mock_retriever_b = MagicMock()
        mock_retriever_b.invoke.return_value = [doc_b]
        mock_db_b = MagicMock()
        mock_db_b.as_retriever.return_value = mock_retriever_b

        def fake_get_vector_db(name: str):
            return mock_db_a if name == "fornecedores" else mock_db_b

        with (
            patch.object(agent_mod, "_listar_colecoes", return_value=["fornecedores", "contratos"]),
            patch.object(agent_mod, "_get_vector_db", side_effect=fake_get_vector_db),
        ):
            result = agent_mod._get_retriever("query de teste")

        self.assertEqual(len(result), 2)
        conteudos = {d.page_content for d in result}
        self.assertIn("fornecedor alfa", conteudos)
        self.assertIn("contrato beta", conteudos)

    def teste_2_colecao_vazia_contribui_zero_docs(self) -> None:
        """Coleção vazia não lança exceção e contribui 0 documentos."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever

        with (
            patch.object(agent_mod, "_listar_colecoes", return_value=["vazia"]),
            patch.object(agent_mod, "_get_vector_db", return_value=mock_db),
        ):
            result = agent_mod._get_retriever("query")

        self.assertEqual(result, [])

    def teste_3_erro_chromadb_propagado_imediatamente(self) -> None:
        """Exceção do ChromaDB durante retrieval é propagada imediatamente."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("ChromaDB explodiu")
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever

        with (
            patch.object(agent_mod, "_listar_colecoes", return_value=["col1"]),
            patch.object(agent_mod, "_get_vector_db", return_value=mock_db),
        ):
            with self.assertRaises(RuntimeError, msg="ChromaDB explodiu"):
                agent_mod._get_retriever("query")

    def teste_4_zero_colecoes_retorna_lista_vazia(self) -> None:
        """Quando _listar_colecoes retorna lista vazia, _get_retriever retorna []."""
        with patch.object(agent_mod, "_listar_colecoes", return_value=[]):
            result = agent_mod._get_retriever("query")
        self.assertEqual(result, [])

    def teste_5_deduplicacao_de_docs_identicos(self) -> None:
        """Documentos idênticos de coleções diferentes são deduplicados."""
        doc_duplicado = self._make_doc("conteudo igual")

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc_duplicado]
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever

        with (
            patch.object(agent_mod, "_listar_colecoes", return_value=["col1", "col2"]),
            patch.object(agent_mod, "_get_vector_db", return_value=mock_db),
        ):
            result = agent_mod._get_retriever("query")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "conteudo igual")

    def teste_6_retorna_maximo_3_docs(self) -> None:
        """Fan-out com mais de 3 docs únicos retorna apenas os 3 primeiros."""
        docs = [self._make_doc(f"doc {i}") for i in range(5)]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = mock_retriever

        with (
            patch.object(agent_mod, "_listar_colecoes", return_value=["col1"]),
            patch.object(agent_mod, "_get_vector_db", return_value=mock_db),
        ):
            result = agent_mod._get_retriever("query")

        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
