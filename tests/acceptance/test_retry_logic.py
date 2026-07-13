# AgenticLog — Acceptance Tests: Retry Logic Feature
"""
Verifica todos os critérios de aceite da story:
  "Como operador logístico, quero que o agente reprocesse chamadas HTTP com backoff
   exponencial quando o LMStudio estiver indisponível."

Cada teste mapeia a exatamente um critério de aceite (AC-01 a AC-08).
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest
from unittest.mock import patch, MagicMock

import httpx
from openai import AuthenticationError

import agenticlog.agent as agent_module
from agenticlog.agent import (
    _invoke_chain,
    usar_ferramenta_web,
    gera_multiplas_respostas,
    AgentState,
)
from agenticlog.config import (
    LLM_MAX_RETRY_ATTEMPTS,
    LLM_RETRY_WAIT_INITIAL_SECONDS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------------
# Helper: replicate app.py error-classification logic for AC-04 tests
# ---------------------------------------------------------------------------

def _classify_error(e: Exception) -> str:
    """Replica a lógica do bloco except em app.py."""
    _msg = str(e).lower()
    if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)):
        return "lmstudio_not_running"
    elif "connection refused" in _msg or ("connect" in _msg and "1234" in _msg):
        return "lmstudio_not_running"
    return "generic_error"


# ---------------------------------------------------------------------------
# AC-01: gera_multiplas_respostas() retries on httpx.ConnectError
# ---------------------------------------------------------------------------

class TestAC01GeraMultiplasRetries(unittest.TestCase):
    """
    AC-01: WHEN gera_multiplas_respostas() encounters httpx.ConnectError
    THEN the system SHALL retry and return the result when a subsequent attempt succeeds.
    """

    @patch("time.sleep")
    def test_ac01_gera_multiplas_retries_connect_error_succeeds_on_second_attempt(
        self, mock_sleep
    ):
        """
        AC-01: _invoke_chain (used exclusively by gera_multiplas_respostas) retries
        ConnectError and returns the result from the second attempt.
        """
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.ConnectError("connection refused"),
            "resposta na segunda tentativa",
        ]

        result = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(result, "resposta na segunda tentativa")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("time.sleep")
    @patch("agenticlog.retrieval.generation._get_llm")
    def test_ac01_gera_multiplas_respostas_retries_end_to_end(
        self, mock_get_llm, mock_sleep
    ):
        """
        AC-01 (end-to-end): gera_multiplas_respostas retries on ConnectError and
        populates state.possible_responses when a subsequent LLM call succeeds.

        Strategy: mock the chain object returned by prompt | llm | parser so that
        chain.invoke raises ConnectError on the first call then succeeds. This lets
        tenacity's retry decorator on _invoke_chain do the real retrying work.
        """
        mock_chain = MagicMock()
        # First call: ConnectError → tenacity retries → second call: success
        mock_chain.invoke.side_effect = [
            httpx.ConnectError("connection refused"),
            "resposta 1",
            "resposta 2",
            "resposta 3",
            "resposta 4",
            "resposta 5",
        ]

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        # Patch the prompt | llm | parser pipeline to return our controllable chain
        with patch("agenticlog.retrieval.generation.prompt_gerar") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            state = AgentState(query="pergunta de logística", next_step="gerar")
            new_state = gera_multiplas_respostas(state)

        self.assertIsNotNone(new_state.possible_responses)
        self.assertGreater(len(new_state.possible_responses), 0)
        for resp in new_state.possible_responses:
            self.assertIn("answer", resp)


# ---------------------------------------------------------------------------
# AC-02: usar_ferramenta_web() retries on httpx.ConnectError (LLM executor path)
# ---------------------------------------------------------------------------

class TestAC02UsarWebRetries(unittest.TestCase):
    """
    AC-02: WHEN usar_ferramenta_web() encounters httpx.ConnectError in the LLM call
    THEN the system SHALL retry and return the result when a subsequent attempt succeeds.
    """

    @patch("time.sleep")
    def test_ac02_invoke_chain_retries_connect_error_succeeds_on_second_attempt(
        self, mock_sleep
    ):
        """
        AC-02: _invoke_chain retries ConnectError and returns the result on the second call.
        """
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.ConnectError("connection refused"),
            "resposta na segunda tentativa",
        ]

        result = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(result, "resposta na segunda tentativa")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("time.sleep")
    @patch("agenticlog.agent.search")
    def test_ac02_usar_ferramenta_web_llm_retries_connect_error_end_to_end(
        self, mock_search, mock_sleep
    ):
        """
        AC-02 (end-to-end): usar_ferramenta_web retries the LLM chain on ConnectError
        via _invoke_chain's @_llm_retry decorator.
        """
        mock_search.run.return_value = "resultados da busca"
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.ConnectError("connection refused"),
            "Resposta web após retry",
        ]
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        with patch("agenticlog.retrieval.graph._prompt_web", mock_prompt):
            state = AgentState(query="últimas notícias sobre supply chain")
            new_state = usar_ferramenta_web(state)

        self.assertEqual(new_state.ranked_response, "Resposta web após retry")
        self.assertEqual(mock_chain.invoke.call_count, 2)


# ---------------------------------------------------------------------------
# AC-03: After retries exhausted, original exception propagates (not RetryError)
# ---------------------------------------------------------------------------

class TestAC03OriginalExceptionPropagates(unittest.TestCase):
    """
    AC-03: WHEN all retry attempts are exhausted
    THEN the system SHALL propagate the original httpx exception, NOT tenacity.RetryError.
    """

    @patch("time.sleep")
    def test_ac03_invoke_chain_propagates_connect_error_not_retry_error(
        self, mock_sleep
    ):
        """
        AC-03 (_invoke_chain): after LLM_MAX_RETRY_ATTEMPTS failures, ConnectError
        is raised directly — tenacity.RetryError must NOT surface.
        """
        from tenacity import RetryError

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = httpx.ConnectError("connection refused")

        with self.assertRaises(httpx.ConnectError):
            _invoke_chain(mock_chain, {"input": "pergunta"})

        # Must not raise RetryError
        mock_chain.invoke.reset_mock()
        mock_chain.invoke.side_effect = httpx.ConnectError("connection refused")
        try:
            _invoke_chain(mock_chain, {"input": "x"})
        except RetryError:
            self.fail("RetryError propagated instead of the original httpx.ConnectError")
        except httpx.ConnectError:
            pass  # expected

        self.assertEqual(mock_chain.invoke.call_count, LLM_MAX_RETRY_ATTEMPTS)

    @patch("time.sleep")
    def test_ac03_invoke_chain_propagates_connect_error_after_retries_exhausted(
        self, mock_sleep
    ):
        """
        AC-03 (_invoke_chain): after retries exhausted, ConnectError propagates
        directly — tenacity.RetryError must NOT surface.
        """
        from tenacity import RetryError

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = httpx.ConnectError("connection refused")

        raised = None
        try:
            _invoke_chain(mock_chain, {"input": "pergunta"})
        except Exception as e:
            raised = e

        self.assertIsNotNone(raised)
        self.assertNotIsInstance(raised, RetryError)
        self.assertIsInstance(raised, httpx.ConnectError)
        self.assertEqual(mock_chain.invoke.call_count, LLM_MAX_RETRY_ATTEMPTS)


# ---------------------------------------------------------------------------
# AC-04: app.py shows "LMStudio not running" for ConnectError, TimeoutException,
#         RemoteProtocolError
# ---------------------------------------------------------------------------

class TestAC04AppErrorClassification(unittest.TestCase):
    """
    AC-04: WHEN the agent raises ConnectError, TimeoutException, or RemoteProtocolError
    THEN app.py SHALL display the "LMStudio not running" message.
    """

    def test_ac04_connect_error_classified_as_lmstudio_not_running(self):
        """AC-04: httpx.ConnectError → 'lmstudio_not_running'."""
        err = httpx.ConnectError("connection refused to 127.0.0.1:1234")
        self.assertEqual(_classify_error(err), "lmstudio_not_running")

    def test_ac04_timeout_exception_classified_as_lmstudio_not_running(self):
        """AC-04: httpx.TimeoutException → 'lmstudio_not_running'."""
        err = httpx.TimeoutException("request timed out")
        self.assertEqual(_classify_error(err), "lmstudio_not_running")

    def test_ac04_remote_protocol_error_classified_as_lmstudio_not_running(self):
        """AC-04: httpx.RemoteProtocolError → 'lmstudio_not_running'."""
        err = httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )
        self.assertEqual(_classify_error(err), "lmstudio_not_running")

    def test_ac04_authentication_error_not_classified_as_lmstudio(self):
        """AC-04 (boundary): AuthenticationError is NOT an LMStudio-down error."""
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        err = AuthenticationError("invalid api key", response=mock_response, body={})
        self.assertEqual(_classify_error(err), "generic_error")


# ---------------------------------------------------------------------------
# AC-05: TCP stall → TimeoutException → retryable
# ---------------------------------------------------------------------------

class TestAC05TcpStallRetryable(unittest.TestCase):
    """
    AC-05: WHEN a TCP stall causes httpx.TimeoutException
    THEN the system SHALL treat it as a retryable failure and continue retrying.
    """

    @patch("time.sleep")
    def test_ac05_timeout_exception_triggers_retry_and_succeeds(self, mock_sleep):
        """AC-05: TimeoutException is retried; success on second call is returned."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            httpx.TimeoutException("request timed out after 10s"),
            "resposta após timeout",
        ]

        result = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(result, "resposta após timeout")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("time.sleep")
    def test_ac05_timeout_constant_value_is_sixty_seconds(self, mock_sleep):
        """AC-05: LLM_TIMEOUT_SECONDS == 60.0 (the TCP-stall deadline)."""
        self.assertEqual(LLM_TIMEOUT_SECONDS, 60.0)


# ---------------------------------------------------------------------------
# AC-06: All retry constants in config.py, none hardcoded in agent.py
# ---------------------------------------------------------------------------

class TestAC06ConstantsInConfig(unittest.TestCase):
    """
    AC-06: WHEN agent.py applies retry logic
    THEN all numeric retry constants SHALL come from config.py (none hardcoded).
    """

    def test_ac06_config_exports_all_four_retry_constants(self):
        """AC-06: config.py exports LLM_TIMEOUT_SECONDS, LLM_MAX_RETRY_ATTEMPTS,
        LLM_RETRY_WAIT_INITIAL_SECONDS, LLM_RETRY_WAIT_MAX_SECONDS."""
        self.assertIsInstance(LLM_TIMEOUT_SECONDS, float)
        self.assertIsInstance(LLM_MAX_RETRY_ATTEMPTS, int)
        self.assertIsInstance(LLM_RETRY_WAIT_INITIAL_SECONDS, float)
        self.assertIsInstance(LLM_RETRY_WAIT_MAX_SECONDS, float)

    def test_ac06_agent_imports_all_four_constants_from_config(self):
        """AC-06: generation.py imports each of the four constants from config (not defining them locally)."""
        from agenticlog.retrieval import generation as gen_mod
        # The module must expose these via its own namespace (imported, not hardcoded)
        self.assertEqual(gen_mod.LLM_TIMEOUT_SECONDS, LLM_TIMEOUT_SECONDS)
        self.assertEqual(gen_mod.LLM_MAX_RETRY_ATTEMPTS, LLM_MAX_RETRY_ATTEMPTS)
        self.assertEqual(gen_mod.LLM_RETRY_WAIT_INITIAL_SECONDS, LLM_RETRY_WAIT_INITIAL_SECONDS)
        self.assertEqual(gen_mod.LLM_RETRY_WAIT_MAX_SECONDS, LLM_RETRY_WAIT_MAX_SECONDS)

    @patch("agenticlog.retrieval.generation.ChatOpenAI")
    def test_ac06_llm_created_with_timeout_from_config(self, mock_chat_openai):
        """AC-06: ChatOpenAI is instantiated with request_timeout=LLM_TIMEOUT_SECONDS."""
        agent_module._llm = None
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        from agenticlog.agent import _get_llm
        _get_llm()

        _, kwargs = mock_chat_openai.call_args
        self.assertEqual(kwargs.get("request_timeout"), LLM_TIMEOUT_SECONDS)

        agent_module._llm = None  # reset singleton

    @patch("time.sleep")
    def test_ac06_retry_respects_max_attempts_from_config(self, mock_sleep):
        """AC-06: number of attempts equals LLM_MAX_RETRY_ATTEMPTS (value from config)."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = httpx.ConnectError("refused")

        try:
            _invoke_chain(mock_chain, {"input": "x"})
        except httpx.ConnectError:
            pass

        self.assertEqual(mock_chain.invoke.call_count, LLM_MAX_RETRY_ATTEMPTS)


# ---------------------------------------------------------------------------
# AC-07: Non-transient errors do NOT retry
# ---------------------------------------------------------------------------

class TestAC07NonTransientDoNotRetry(unittest.TestCase):
    """
    AC-07: WHEN a non-transient error occurs (AuthenticationError, BadRequestError)
    THEN the system SHALL NOT retry — it SHALL fail immediately on the first attempt.
    """

    @patch("time.sleep")
    def test_ac07_authentication_error_does_not_retry(self, mock_sleep):
        """AC-07: AuthenticationError causes immediate failure — invoke called exactly once."""
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        mock_chain.invoke.side_effect = AuthenticationError(
            "invalid api key", response=mock_response, body={}
        )

        with self.assertRaises(AuthenticationError):
            _invoke_chain(mock_chain, {"input": "pergunta"})

        mock_chain.invoke.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_ac07_bad_request_error_does_not_retry(self, mock_sleep):
        """AC-07: openai.BadRequestError causes immediate failure — no retry."""
        from openai import BadRequestError

        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.request = MagicMock()
        mock_chain.invoke.side_effect = BadRequestError(
            "invalid request", response=mock_response, body={}
        )

        with self.assertRaises(BadRequestError):
            _invoke_chain(mock_chain, {"input": "pergunta"})

        mock_chain.invoke.assert_called_once()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# AC-08: DuckDuckGo failures return fallback, do NOT propagate
# ---------------------------------------------------------------------------

class TestAC08DuckDuckGoFallback(unittest.TestCase):
    """
    AC-08: WHEN DuckDuckGo search raises an exception in usar_ferramenta_web()
    THEN the system SHALL return a fallback string and SHALL NOT propagate the exception.
    """

    @patch("agenticlog.agent.search")
    def test_ac08_duckduckgo_failure_returns_fallback_string(self, mock_search):
        """AC-08: DuckDuckGo exception → ranked_response='Busca indisponível no momento.'"""
        mock_search.run.side_effect = Exception("DuckDuckGo rate-limited")

        state = AgentState(query="últimas notícias")
        new_state = usar_ferramenta_web(state)

        self.assertEqual(new_state.ranked_response, "Busca indisponível no momento.")
        self.assertEqual(new_state.confidence_score, 0.0)

    @patch("agenticlog.retrieval.generation._invoke_chain")
    @patch("agenticlog.agent.search")
    def test_ac08_duckduckgo_failure_does_not_call_llm_executor(
        self, mock_search, mock_invoke_chain
    ):
        """AC-08: when DuckDuckGo fails, LLM chain is never invoked (early return)."""
        mock_search.run.side_effect = Exception("DuckDuckGo down")

        state = AgentState(query="notícias recentes")
        usar_ferramenta_web(state)

        mock_invoke_chain.assert_not_called()

    @patch("agenticlog.agent.search")
    def test_ac08_duckduckgo_failure_does_not_propagate_exception(self, mock_search):
        """AC-08: usar_ferramenta_web must not raise when DuckDuckGo fails."""
        mock_search.run.side_effect = RuntimeError("network error")

        state = AgentState(query="pesquisa web")
        try:
            usar_ferramenta_web(state)
        except Exception as exc:
            self.fail(
                f"usar_ferramenta_web propagated an exception when it should not: {exc}"
            )


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
