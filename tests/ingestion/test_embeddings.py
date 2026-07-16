# AgenticLog - Testes unitários para ingestion/embeddings.py
"""Testes do estágio de embeddings da ingestão (ADR-018 Fase 3a).

TestGetRagEmbeddingModel/TestEmbeddingModelConfig movidos de tests/test_rag.py;
`@patch("agenticlog.rag.HuggingFaceEmbeddings")` repontado para
`agenticlog.ingestion.embeddings.HuggingFaceEmbeddings` (construção migrou p/ o factory).
O cache `_rag_embedding_model` fica em ingestion/embeddings.py (Fase 6).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.config as config  # noqa: E402
import agenticlog.ingestion.embeddings as rag  # noqa: E402
from agenticlog.ingestion.embeddings import criar_embedding_model  # noqa: E402


class TestEmbeddingModelConfig(unittest.TestCase):
    """Testes para a constante EMBEDDING_MODEL (PORTPT-01 / AC1)."""

    def test_embedding_model_e_multilingue(self):
        """EMBEDDING_MODEL aponta para o modelo multilíngue (paraphrase-multilingual-mpnet)."""
        self.assertEqual(
            config.EMBEDDING_MODEL,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )


class TestGetRagEmbeddingModel(unittest.TestCase):
    """Testes para _get_rag_embedding_model (PORTPT-02 / AC2)."""

    def setUp(self) -> None:
        """Reseta o singleton antes de cada teste para garantir isolamento."""
        rag._rag_embedding_model = None

    def tearDown(self) -> None:
        """Reseta o singleton após cada teste."""
        rag._rag_embedding_model = None

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def test_get_rag_embedding_model_usa_embedding_model_do_config(self, mock_emb):
        """_get_rag_embedding_model() constrói HuggingFaceEmbeddings com model_name=config.EMBEDDING_MODEL."""
        rag._get_rag_embedding_model()

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def test_get_rag_embedding_model_singleton_reusa_instancia(self, mock_emb):
        """Chamadas subsequentes retornam a mesma instância sem recriar HuggingFaceEmbeddings."""
        primeira = rag._get_rag_embedding_model()
        segunda = rag._get_rag_embedding_model()

        self.assertIs(primeira, segunda)
        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )


class TestCriarEmbeddingModel(unittest.TestCase):
    """Testes unitários para criar_embedding_model (RAGING-06 / T15)."""

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def teste_1_constroi_com_kwargs_exatos(self, mock_emb):
        """criar_embedding_model() instancia HuggingFaceEmbeddings com os kwargs verbatim."""
        resultado = criar_embedding_model()

        mock_emb.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.assertIs(resultado, mock_emb.return_value)

    @patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings")
    def teste_2_normalize_embeddings_preservado(self, mock_emb):
        """encode_kwargs preserva normalize_embeddings=True (crítico p/ silent-degradation)."""
        criar_embedding_model()
        _, kwargs = mock_emb.call_args
        self.assertEqual(kwargs["encode_kwargs"], {"normalize_embeddings": True})
