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
)


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

    @patch("agenticlog.agent.avk_agent_executor")
    def teste_4_usar_ferramenta_web(self, mock_executor):
        mock_executor.invoke.return_value = {"output": "Resposta da web."}
        state = AgentState(query="notícias recentes sobre supply chain")
        new_state = usar_ferramenta_web(state)
        mock_executor.invoke.assert_called_once_with(
            "notícias recentes sobre supply chain"
        )
        self.assertEqual(new_state.ranked_response, "Resposta da web.")

    @patch("agenticlog.agent.retriever")
    def teste_5_retrieve_info(self, mock_retriever):
        mock_retriever.invoke.return_value = [
            Document(page_content="Documento sobre cadeia de suprimentos")
        ]
        state = AgentState(query="fases da cadeia de suprimentos")
        new_state = retrieve_info(state)
        mock_retriever.invoke.assert_called_once_with("fases da cadeia de suprimentos")
        self.assertEqual(len(new_state.retrieved_info), 1)

    @patch("agenticlog.agent.retriever")
    def teste_5b_retrieve_info_empty(self, mock_retriever):
        """Recuperação vazia: retriever retorna lista vazia."""
        mock_retriever.invoke.return_value = []
        state = AgentState(query="consulta sem resultados")
        new_state = retrieve_info(state)
        mock_retriever.invoke.assert_called_once_with("consulta sem resultados")
        self.assertEqual(len(new_state.retrieved_info), 0)
        self.assertEqual(new_state.retrieved_info, [])

    @patch("agenticlog.agent.StrOutputParser")
    @patch("agenticlog.agent.llm")
    @patch("agenticlog.agent.prompt_rag_retrieve")
    def teste_6_gera_multiplas_respostas(
        self, mock_prompt, mock_llm, mock_str_parser_class
    ):
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
    @patch("agenticlog.agent.llm")
    @patch("agenticlog.agent.prompt_rag_retrieve")
    def teste_6b_gera_multiplas_respostas_empty_context(
        self, mock_prompt, mock_llm, mock_str_parser_class
    ):
        """Recuperação vazia: context vazio é passado ao LLM (retrieve_info = [])."""
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

    @patch("agenticlog.agent.embedding_model")
    def teste_7_avalia_similaridade(self, mock_embedding_model):
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        state = AgentState(
            query="cadeia de suprimentos",
            retrieved_info=[Document(page_content="doc")],
            possible_responses=[{"answer": "resposta"}],
        )
        new_state = avalia_similaridade(state)
        self.assertEqual(len(new_state.similarity_scores), 1)

    @patch("agenticlog.agent.embedding_model")
    def teste_7b_avalia_similaridade_empty_retrieved(self, mock_embedding_model):
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


if __name__ == "__main__":
    print("\nIniciando os testes. Aguarde...\n")
    unittest.main(verbosity=2)
