# AgenticLog - Testes unitários
"""
Testes para o sistema Agentic RAG em agenticlog.agent.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from agenticlog.agent import (
    passo_decisao_agente,
    usar_ferramenta_web,
    retrieve_info,
    gera_multiplas_respostas,
    avalia_similaridade,
    rank_respostas,
    AgentState,
    _get_llm,
    _get_vector_db,
)
import agenticlog.agent as agent_module


class TestAgenticRAG(unittest.TestCase):
    def teste_1_passo_decisao_retrieve(self):
        state = AgentState(query="Quais as fases da cadeia de suprimentos?")
        new_state = passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "retrieve")

    def teste_2_passo_decisao_usar_web(self):
        state = AgentState(
            query="Pesquise sobre as novidades recentes em logistica e supply chain."
        )
        new_state = passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "usar_web")

    def teste_3_passo_decisao_gerar(self):
        state = AgentState(query="Resuma o conceito de supply chain.")
        new_state = passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "gerar")

    @patch("agenticlog.agent._get_avk_agent_executor")
    def teste_4_usar_ferramenta_web(self, mock_get_executor):
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Resposta da web."}
        mock_get_executor.return_value = mock_executor
        state = AgentState(query="notícias recentes sobre supply chain")
        new_state = usar_ferramenta_web(state)
        mock_executor.invoke.assert_called_once_with(
            "notícias recentes sobre supply chain"
        )
        self.assertEqual(new_state.ranked_response, "Resposta da web.")

    @patch("agenticlog.agent._get_retriever")
    def teste_5_retrieve_info(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Documento sobre cadeia de suprimentos")
        ]
        mock_get_retriever.return_value = mock_retriever
        state = AgentState(query="fases da cadeia de suprimentos")
        new_state = retrieve_info(state)
        mock_retriever.invoke.assert_called_once_with("fases da cadeia de suprimentos")
        self.assertEqual(len(new_state.retrieved_info), 1)

    @patch("agenticlog.agent._get_retriever")
    def teste_5b_retrieve_info_empty(self, mock_get_retriever):
        """Recuperação vazia: retriever retorna lista vazia."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever
        state = AgentState(query="consulta sem resultados")
        new_state = retrieve_info(state)
        mock_retriever.invoke.assert_called_once_with("consulta sem resultados")
        self.assertEqual(len(new_state.retrieved_info), 0)
        self.assertEqual(new_state.retrieved_info, [])

    @patch("agenticlog.agent.StrOutputParser")
    @patch("agenticlog.agent._get_llm")
    @patch("agenticlog.agent.prompt_rag_retrieve")
    def teste_6_gera_multiplas_respostas(
        self, mock_prompt, mock_get_llm, mock_str_parser_class
    ):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Resposta gerada"
        mock_prompt_llm_chain = MagicMock()
        mock_prompt_llm_chain.__or__ = lambda self, other: mock_chain
        mock_prompt.__or__ = lambda self, other: mock_prompt_llm_chain
        mock_parser_instance = MagicMock()
        mock_parser_instance.invoke.return_value = "Resposta gerada"
        mock_str_parser_class.return_value = mock_parser_instance

        state = AgentState(
            query="fases da cadeia de suprimentos", next_step="retrieve"
        )
        state.retrieved_info = [Document(page_content="Documento teste")]
        new_state = gera_multiplas_respostas(state)
        self.assertEqual(len(new_state.possible_responses), 5)
        self.assertIn("answer", new_state.possible_responses[0])
        self.assertEqual(new_state.possible_responses[0]["answer"], "Resposta gerada")

    @patch("agenticlog.agent.StrOutputParser")
    @patch("agenticlog.agent._get_llm")
    @patch("agenticlog.agent.prompt_rag_retrieve")
    def teste_6b_gera_multiplas_respostas_empty_context(
        self, mock_prompt, mock_get_llm, mock_str_parser_class
    ):
        """Recuperação vazia: context vazio é passado ao LLM (retrieve_info = [])."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Sorry, I did not find that information in the documents."
        mock_prompt_llm_chain = MagicMock()
        mock_prompt_llm_chain.__or__ = lambda self, other: mock_chain
        mock_prompt.__or__ = lambda self, other: mock_prompt_llm_chain
        mock_parser_instance = MagicMock()
        mock_parser_instance.invoke.return_value = "Sorry, I did not find that information in the documents."
        mock_str_parser_class.return_value = mock_parser_instance

        state = AgentState(query="consulta inexistente", next_step="retrieve")
        state.retrieved_info = []
        new_state = gera_multiplas_respostas(state)
        self.assertEqual(len(new_state.possible_responses), 5)
        mock_chain.invoke.assert_called()
        invoke_arg = mock_chain.invoke.call_args[0][0]
        self.assertEqual(invoke_arg.get("context", ""), "")

    @patch("agenticlog.agent._get_embedding_model")
    def teste_7_avalia_similaridade(self, mock_get_embedding_model):
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        mock_get_embedding_model.return_value = mock_embedding_model
        state = AgentState(
            query="cadeia de suprimentos",
            retrieved_info=[Document(page_content="doc")],
            possible_responses=[{"answer": "resposta"}],
        )
        new_state = avalia_similaridade(state)
        self.assertEqual(len(new_state.similarity_scores), 1)

    @patch("agenticlog.agent._get_embedding_model")
    def teste_7b_avalia_similaridade_empty_retrieved(self, mock_get_embedding_model):
        """Recuperação vazia: retrieved_info vazio resulta em similarity_scores zerados."""
        state = AgentState(
            query="consulta sem documentos",
            retrieved_info=[],
            possible_responses=[{"answer": "r1"}, {"answer": "r2"}],
        )
        new_state = avalia_similaridade(state)
        self.assertEqual(new_state.similarity_scores, [0.0, 0.0])

    def teste_8_rank_respostas(self):
        state = AgentState(
            query="cadeia de suprimentos",
            possible_responses=[{"answer": "resp1"}, {"answer": "resp2"}],
            similarity_scores=[0.8, 0.9],
        )
        new_state = rank_respostas(state)
        self.assertEqual(new_state.ranked_response, {"answer": "resp2"})
        self.assertEqual(new_state.confidence_score, 0.9)

    @patch("agenticlog.agent.ChatOpenAI")
    def teste_9_import_sem_lmstudio(self, mock_chat_openai):
        """LAZY-01: recarregar o módulo não deve instanciar ChatOpenAI (init é lazy)."""
        import importlib
        importlib.reload(agent_module)
        mock_chat_openai.assert_not_called()
        self.assertIsNone(agent_module._llm)

    @patch("agenticlog.agent.ChatOpenAI")
    def teste_10_get_llm_singleton(self, mock_chat_openai):
        """LAZY-04: _get_llm() retorna a mesma instância em chamadas repetidas (singleton)."""
        agent_module._llm = None
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        primeira = _get_llm()
        segunda = _get_llm()

        self.assertIs(primeira, segunda)
        mock_chat_openai.assert_called_once()

        # limpar singleton para não vazar entre testes
        agent_module._llm = None

    @patch("agenticlog.agent.Chroma")
    @patch("agenticlog.agent._get_embedding_model")
    def teste_11_get_vector_db_singleton(self, mock_get_embedding, mock_chroma):
        """LAZY-05: _get_vector_db() retorna a mesma instância em chamadas repetidas (singleton)."""
        agent_module._vector_db = None
        mock_db_instance = MagicMock()
        mock_chroma.return_value = mock_db_instance
        mock_get_embedding.return_value = MagicMock()

        primeiro = _get_vector_db()
        segundo = _get_vector_db()

        self.assertIs(primeiro, segundo)
        mock_chroma.assert_called_once()

        # limpar singleton para não vazar entre testes
        agent_module._vector_db = None


if __name__ == "__main__":
    print("\nIniciando os testes. Aguarde...\n")
    unittest.main(verbosity=2)
