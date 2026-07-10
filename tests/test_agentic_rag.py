# AgenticLog - Testes unitários
"""
Testes para o sistema Agentic RAG em agenticlog.agent.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock, call
import httpx

from langchain_core.documents import Document

from agenticlog.agent import (
    passo_decisao_agente,
    usar_ferramenta_web,
    retrieve_info,
    gera_multiplas_respostas,
    avalia_similaridade,
    rank_respostas,
    _invoke_chain,
    AgentState,
    _get_llm,
    _get_vector_db,
)
import agenticlog.agent as agent_module
import agenticlog.config as config
from agenticlog.config import LLM_MAX_RETRY_ATTEMPTS, LLM_TIMEOUT_SECONDS


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

    def teste_3_passo_decisao_conceitual_vai_para_retrieve(self):
        # Consultas conceituais ("Resuma o conceito") NÃO são mais roteadas
        # direto para "gerar": tentam retrieve primeiro (gerar é só fallback).
        state = AgentState(query="Resuma o conceito de supply chain.")
        new_state = passo_decisao_agente(state)
        self.assertEqual(new_state.next_step, "retrieve")

    @patch("agenticlog.retrieval.generation._invoke_chain")
    @patch("agenticlog.agent.search")
    def teste_4_usar_ferramenta_web(self, mock_search, mock_invoke_chain):
        mock_search.run.return_value = "resultados da busca"
        mock_invoke_chain.return_value = "Resposta da web."
        state = AgentState(query="notícias recentes sobre supply chain")
        new_state = usar_ferramenta_web(state)
        mock_invoke_chain.assert_called_once()
        self.assertEqual(new_state.ranked_response, "Resposta da web.")

    @patch("agenticlog.retrieval.retriever._get_retriever")
    def teste_5_retrieve_info(self, mock_get_retriever):
        """_get_retriever(query) retorna list[Document] diretamente — sem .invoke()."""
        mock_get_retriever.return_value = [
            Document(page_content="Documento sobre cadeia de suprimentos")
        ]
        state = AgentState(query="fases da cadeia de suprimentos", next_step="retrieve")
        new_state = retrieve_info(state)
        mock_get_retriever.assert_called_once_with("fases da cadeia de suprimentos")
        self.assertEqual(len(new_state.retrieved_info), 1)
        # Recuperação com resultados: mantém rota retrieve (não cai para gerar).
        self.assertEqual(new_state.next_step, "retrieve")

    @patch("agenticlog.retrieval.retriever._get_retriever")
    def teste_5b_retrieve_info_empty(self, mock_get_retriever):
        """Recuperação vazia: cai para fallback 'gerar' (geração direta sem contexto)."""
        mock_get_retriever.return_value = []
        state = AgentState(query="consulta sem resultados", next_step="retrieve")
        # Fail-loud: retrieval vazio deve logar WARNING (não passar silencioso).
        with self.assertLogs("agenticlog.agent", level="WARNING") as log_ctx:
            new_state = retrieve_info(state)
        mock_get_retriever.assert_called_once_with("consulta sem resultados")
        self.assertEqual(len(new_state.retrieved_info), 0)
        self.assertEqual(new_state.retrieved_info, [])
        # Fallback: sem documentos, rota vira "gerar".
        self.assertEqual(new_state.next_step, "gerar")
        self.assertTrue(any("0 documentos" in m for m in log_ctx.output))

    @patch("time.sleep")
    @patch("agenticlog.retrieval.generation.StrOutputParser")
    @patch("agenticlog.retrieval.generation._get_llm")
    @patch("agenticlog.retrieval.generation.prompt_rag_retrieve")
    def teste_6_gera_multiplas_respostas(
        self, mock_prompt, mock_get_llm, mock_str_parser_class, mock_sleep
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
        self.assertEqual(len(new_state.possible_responses), config.NUM_CANDIDATE_RESPONSES)
        self.assertIn("answer", new_state.possible_responses[0])
        self.assertEqual(new_state.possible_responses[0]["answer"], "Resposta gerada")

    @patch("time.sleep")
    @patch("agenticlog.retrieval.generation.StrOutputParser")
    @patch("agenticlog.retrieval.generation._get_llm")
    @patch("agenticlog.retrieval.generation.prompt_rag_retrieve")
    def teste_6b_gera_multiplas_respostas_empty_context(
        self, mock_prompt, mock_get_llm, mock_str_parser_class, mock_sleep
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
        self.assertEqual(len(new_state.possible_responses), config.NUM_CANDIDATE_RESPONSES)
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

    @patch("agenticlog.retrieval.generation.ChatOpenAI")
    def teste_9_import_sem_lmstudio(self, mock_chat_openai):
        """LAZY-01: recarregar o módulo não deve instanciar ChatOpenAI (init é lazy)."""
        import importlib
        importlib.reload(agent_module)
        mock_chat_openai.assert_not_called()
        self.assertIsNone(agent_module._llm)

    @patch("agenticlog.retrieval.generation.ChatOpenAI")
    def teste_10_get_llm_singleton(self, mock_chat_openai):
        """LAZY-04: _get_llm() retorna a mesma instância em chamadas repetidas (singleton)
        e é criado com request_timeout=LLM_TIMEOUT_SECONDS."""
        agent_module._llm = None
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        primeira = _get_llm()
        segunda = _get_llm()

        self.assertIs(primeira, segunda)
        mock_chat_openai.assert_called_once()
        _, kwargs = mock_chat_openai.call_args
        self.assertEqual(kwargs.get("request_timeout"), LLM_TIMEOUT_SECONDS)

        # limpar singleton para não vazar entre testes
        agent_module._llm = None

    @patch("agenticlog.retrieval.retriever.Chroma")
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

    def teste_12_sem_import_anthropic(self):
        """LLMPORT-08/09/10: agent.py não referencia mais o pacote `anthropic`."""
        self.assertNotIn("anthropic", agent_module.__dict__.keys())


class TestRetryLogic(unittest.TestCase):
    """Testes de retry com backoff exponencial para chamadas ao LLM (T1-T7)."""

    @patch("time.sleep")
    def teste_1_retry_sucesso_primeira_tentativa(self, mock_sleep):
        """T1: _invoke_chain retorna resultado na primeira tentativa (happy path)."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "resposta ok"

        resultado = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(resultado, "resposta ok")
        mock_chain.invoke.assert_called_once_with({"input": "pergunta"})
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def teste_2_retry_uma_falha_sucesso_na_segunda(self, mock_sleep):
        """T2: _invoke_chain falha uma vez com ConnectError e sucede na segunda tentativa."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.ConnectError("connection refused"),
            "resposta na segunda",
        ]

        resultado = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(resultado, "resposta na segunda")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("time.sleep")
    def teste_3_retry_tres_falhas_propaga_excecao_original(self, mock_sleep):
        """T3: 3 falhas consecutivas propagam ConnectError original (não RetryError)."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = httpx.ConnectError("connection refused")

        with self.assertRaises(httpx.ConnectError):
            _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(mock_chain.invoke.call_count, 3)

    @patch("time.sleep")
    def teste_4_authentication_error_nao_faz_retry(self, mock_sleep):
        """T4: AuthenticationError não é retryable — falha imediatamente sem retry."""
        from openai import AuthenticationError

        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        mock_chain.invoke.side_effect = AuthenticationError(
            "invalid api key", response=mock_response, body={}
        )

        with self.assertRaises(AuthenticationError):
            _invoke_chain(mock_chain, {"input": "pergunta"})

        mock_chain.invoke.assert_called_once()

    @patch("time.sleep")
    def teste_5_remote_protocol_error_e_retryable(self, mock_sleep):
        """T5: RemoteProtocolError é retryable — segunda tentativa tem sucesso."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
            "resposta ok",
        ]

        resultado = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(resultado, "resposta ok")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("agenticlog.retrieval.generation._invoke_chain")
    @patch("agenticlog.agent.search")
    def teste_6_duckduckgo_falha_retorna_fallback_sem_propagar(
        self, mock_search, mock_invoke_chain
    ):
        """T6: Falha no DuckDuckGo retorna string de fallback e NÃO propaga a exceção."""
        mock_search.run.side_effect = Exception("DuckDuckGo indisponível")

        state = AgentState(query="últimas informações sobre supply chain")
        new_state = usar_ferramenta_web(state)

        self.assertEqual(new_state.ranked_response, "Busca indisponível no momento.")
        self.assertEqual(new_state.confidence_score, 0.0)
        mock_invoke_chain.assert_not_called()

    @patch("time.sleep")
    def teste_7_timeout_exception_e_retryable(self, mock_sleep):
        """T7: TimeoutException é retryable — respeita LLM_TIMEOUT_SECONDS por chamada."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.TimeoutException("request timed out"),
            "resposta após timeout",
        ]

        resultado = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(resultado, "resposta após timeout")
        self.assertEqual(mock_chain.invoke.call_count, 2)
        self.assertEqual(LLM_TIMEOUT_SECONDS, 60.0)

    @patch("time.sleep")
    def teste_8_openai_api_connection_error_e_retryable(self, mock_sleep):
        """T8: openai.APIConnectionError é retryable — segunda tentativa tem sucesso."""
        from openai import APIConnectionError

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            APIConnectionError(
                request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
            ),
            "resposta ok",
        ]

        resultado = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(resultado, "resposta ok")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("time.sleep")
    def teste_9_openai_api_connection_error_exhaust_reraise(self, mock_sleep):
        """T9: openai.APIConnectionError esgota tentativas e propaga a excecao original
        (nao tenacity.RetryError), apos exatamente LLM_MAX_RETRY_ATTEMPTS chamadas."""
        from openai import APIConnectionError

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = APIConnectionError(
            request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
        )

        with self.assertRaises(APIConnectionError):
            _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(mock_chain.invoke.call_count, LLM_MAX_RETRY_ATTEMPTS)


if __name__ == "__main__":
    print("\nIniciando os testes. Aguarde...\n")
    unittest.main(verbosity=2)
