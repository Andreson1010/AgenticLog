# AgenticLog - Testes de integração do agent_workflow
"""
Testes de integração para agent_workflow.invoke() cobrindo os três caminhos de roteamento:
- retrieve  → busca vetorial → generate_multiple → evaluate_similarity → rank
- gerar     → generate_multiple (sem retrieval) → evaluate_similarity → rank
- usar_web  → DuckDuckGo agent → END
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from agenticlog.agent import AgentState, agent_workflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding_model() -> MagicMock:
    """Retorna um mock do modelo de embeddings que devolve vetores fixos."""
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [[0.1] * 768]
    mock_model.embed_query.return_value = [0.1] * 768
    return mock_model


def _fake_llm(response_text: str = "Resposta integração teste.") -> MagicMock:
    """Retorna um mock do ChatOpenAI cujo pipeline (prompt | llm | parser) devolve response_text."""
    mock_llm = MagicMock()
    mock_llm.__or__ = lambda self, other: other
    return mock_llm


def _fake_retriever(docs: list) -> MagicMock:
    """Retorna um mock de retriever que devolve a lista de documentos fornecida."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs
    return mock_retriever


# ---------------------------------------------------------------------------
# Classe de testes de integração
# ---------------------------------------------------------------------------

class TestAgentWorkflowIntegration(unittest.TestCase):
    """Testes de integração end-to-end para agent_workflow.invoke()."""

    # ------------------------------------------------------------------
    # Caminho 1: retrieve
    # ------------------------------------------------------------------

    @patch("agenticlog.agent._get_embedding_model")
    @patch("agenticlog.agent._get_retriever")
    @patch("agenticlog.agent._invoke_chain")
    def teste_1_caminho_retrieve_retorna_ranked_response(
        self,
        mock_invoke_chain,
        mock_get_retriever,
        mock_get_embedding,
    ):
        """INTEG-01: rota 'retrieve' — ranked_response e confidence_score preenchidos."""
        fake_docs = [
            Document(page_content="Planejamento é a primeira fase."),
            Document(page_content="Distribuição é a fase final."),
        ]
        mock_get_retriever.return_value = _fake_retriever(fake_docs)
        mock_get_embedding.return_value = _fake_embedding_model()
        mock_invoke_chain.return_value = "A cadeia de suprimentos possui cinco fases."

        state = AgentState(query="Quais as fases da cadeia de suprimentos?")

        result = agent_workflow.invoke(state)

        self.assertIn("ranked_response", result)
        self.assertIn("confidence_score", result)
        ranked = result["ranked_response"]
        self.assertTrue(ranked, "ranked_response não pode ser vazio ou None")
        self.assertIsInstance(result["confidence_score"], float)

    @patch("agenticlog.agent._get_embedding_model")
    @patch("agenticlog.agent._get_retriever")
    @patch("agenticlog.agent._invoke_chain")
    def teste_2_caminho_retrieve_agentstate_chaves_obrigatorias(
        self,
        mock_invoke_chain,
        mock_get_retriever,
        mock_get_embedding,
    ):
        """INTEG-02: rota 'retrieve' — AgentState pós-invocação contém todas as chaves obrigatórias."""
        fake_docs = [Document(page_content="Documento de logística.")]
        mock_get_retriever.return_value = _fake_retriever(fake_docs)
        mock_get_embedding.return_value = _fake_embedding_model()
        mock_invoke_chain.return_value = "Resposta sobre logística."

        state = AgentState(query="Quais as fases da cadeia de suprimentos?")
        result = agent_workflow.invoke(state)

        required_keys = {
            "query",
            "next_step",
            "retrieved_info",
            "possible_responses",
            "similarity_scores",
            "ranked_response",
            "confidence_score",
        }
        for key in required_keys:
            self.assertIn(key, result, f"Chave obrigatória ausente: {key}")

        self.assertEqual(result["next_step"], "retrieve")

    # ------------------------------------------------------------------
    # Caminho 2: gerar (geração conceitual sem retrieval)
    # ------------------------------------------------------------------

    @patch("agenticlog.agent._get_embedding_model")
    @patch("agenticlog.agent._get_retriever")
    @patch("agenticlog.agent._invoke_chain")
    def teste_3_caminho_gerar_retorna_ranked_response(
        self,
        mock_invoke_chain,
        mock_get_retriever,
        mock_get_embedding,
    ):
        """INTEG-03: fallback 'gerar' — retrieve vazio cai para geração direta com ranked_response."""
        mock_get_retriever.return_value = []
        mock_get_embedding.return_value = _fake_embedding_model()
        mock_invoke_chain.return_value = "Supply chain é o conjunto de processos de produção e distribuição."

        state = AgentState(query="Defina o conceito de supply chain.")

        result = agent_workflow.invoke(state)

        self.assertIn("ranked_response", result)
        self.assertIn("confidence_score", result)
        ranked = result["ranked_response"]
        self.assertTrue(ranked, "ranked_response não pode ser vazio ou None")
        self.assertIsInstance(result["confidence_score"], float)
        # Retrieve-first: decisão roteia para retrieve; base vazia faz fallback para gerar.
        self.assertEqual(result["next_step"], "gerar")

    @patch("agenticlog.agent._get_embedding_model")
    @patch("agenticlog.agent._get_retriever")
    @patch("agenticlog.agent._invoke_chain")
    def teste_4_caminho_gerar_retrieved_info_vazio(
        self,
        mock_invoke_chain,
        mock_get_retriever,
        mock_get_embedding,
    ):
        """INTEG-04: fallback 'gerar' — retrieved_info deve ser lista vazia (sem retrieval vetorial)."""
        mock_get_retriever.return_value = []
        mock_get_embedding.return_value = _fake_embedding_model()
        mock_invoke_chain.return_value = "Explicação do conceito."

        state = AgentState(query="Explique o que é logística reversa.")

        result = agent_workflow.invoke(state)

        self.assertEqual(result["retrieved_info"], [])
        self.assertIn("ranked_response", result)

    # ------------------------------------------------------------------
    # Caminho 3: usar_web (busca web via DuckDuckGo)
    # ------------------------------------------------------------------

    @patch("agenticlog.agent._invoke_chain")
    @patch("agenticlog.agent.search")
    def teste_5_caminho_usar_web_retorna_ranked_response(
        self,
        mock_search,
        mock_invoke_chain,
    ):
        """INTEG-05: rota 'usar_web' — ranked_response preenchida com resultado do agente web."""
        mock_search.run.return_value = "Resultados de busca web."
        mock_invoke_chain.return_value = "Últimas notícias sobre supply chain global."

        state = AgentState(query="Busque na web as últimas informações sobre supply chain.")

        result = agent_workflow.invoke(state)

        self.assertIn("ranked_response", result)
        self.assertIn("confidence_score", result)
        self.assertEqual(
            result["ranked_response"],
            "Últimas notícias sobre supply chain global.",
        )
        self.assertEqual(result["confidence_score"], 0.0)
        self.assertEqual(result["next_step"], "usar_web")

    @patch("agenticlog.agent._invoke_chain")
    @patch("agenticlog.agent.search")
    def teste_6_caminho_usar_web_agentstate_chaves_obrigatorias(
        self,
        mock_search,
        mock_invoke_chain,
    ):
        """INTEG-06: rota 'usar_web' — AgentState pós-invocação contém todas as chaves obrigatórias."""
        mock_search.run.return_value = "Resultados web."
        mock_invoke_chain.return_value = "Resposta web."

        state = AgentState(query="Últimas notícias sobre logística recente no Brasil.")

        result = agent_workflow.invoke(state)

        required_keys = {"query", "next_step", "ranked_response", "confidence_score"}
        for key in required_keys:
            self.assertIn(key, result, f"Chave obrigatória ausente: {key}")

    @patch("agenticlog.agent._invoke_chain")
    @patch("agenticlog.agent.search")
    def teste_7_caminho_usar_web_duckduckgo_falha_retorna_fallback(
        self,
        mock_search,
        mock_invoke_chain,
    ):
        """INTEG-07: rota 'usar_web' — falha no DuckDuckGo retorna mensagem de fallback."""
        mock_search.run.side_effect = Exception("DuckDuckGo indisponível")

        state = AgentState(query="Últimas notícias sobre supply chain recente.")

        result = agent_workflow.invoke(state)

        self.assertEqual(result["ranked_response"], "Busca indisponível no momento.")
        self.assertEqual(result["confidence_score"], 0.0)
        mock_invoke_chain.assert_not_called()

    # ------------------------------------------------------------------
    # Casos de borda
    # ------------------------------------------------------------------

    @patch("agenticlog.agent._get_embedding_model")
    @patch("agenticlog.agent._get_retriever")
    @patch("agenticlog.agent._invoke_chain")
    def teste_8_caminho_retrieve_sem_documentos_retorna_resposta(
        self,
        mock_invoke_chain,
        mock_get_retriever,
        mock_get_embedding,
    ):
        """INTEG-08: rota 'retrieve' com retriever vazio — workflow completa sem erro."""
        mock_get_retriever.return_value = _fake_retriever([])
        mock_get_embedding.return_value = _fake_embedding_model()
        mock_invoke_chain.return_value = "Sorry, I did not find that information in the documents."

        state = AgentState(query="Informação inexistente na base vetorial.")

        result = agent_workflow.invoke(state)

        self.assertIn("ranked_response", result)
        self.assertIn("confidence_score", result)
        self.assertTrue(
            all(s == 0.0 for s in result["similarity_scores"]),
            "Todos os scores devem ser 0.0 quando não há documentos recuperados",
        )

    @patch("agenticlog.agent._get_embedding_model")
    @patch("agenticlog.agent._get_retriever")
    @patch("agenticlog.agent._invoke_chain")
    def teste_9_caminho_gerar_possible_responses_tem_cinco_itens(
        self,
        mock_invoke_chain,
        mock_get_retriever,
        mock_get_embedding,
    ):
        """INTEG-09: fallback 'gerar' — possible_responses deve conter exatamente 5 candidatas."""
        mock_get_retriever.return_value = []
        mock_get_embedding.return_value = _fake_embedding_model()
        mock_invoke_chain.return_value = "Resposta conceitual."

        state = AgentState(query="O que é gerenciamento de estoque?")

        result = agent_workflow.invoke(state)

        self.assertEqual(len(result["possible_responses"]), 5)
        for item in result["possible_responses"]:
            self.assertIn("answer", item)


if __name__ == "__main__":
    print("\nIniciando testes de integração do agent_workflow. Aguarde...\n")
    unittest.main(verbosity=2)
