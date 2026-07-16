# AgenticLog — Acceptance Tests: LLM Provider Portability Feature
"""
Verifica os critérios de aceite da story "LLM Provider Portability":
  Como desenvolvedor operando AgenticLog em diferentes ambientes (LMStudio local,
  endpoints OpenAI-compatible alternativos), quero que a configuração do provider LLM
  (nome do modelo, tratamento de exceções de retry, tipagem do cliente LLM) seja
  portável e livre de dependências de SDK não relacionadas.

Cobre LLMPORT-01..16 (spec.md, .specs/features/llm-provider-portability/spec.md):

  P1 — LLM_MODEL via env var:
    LLMPORT-01: unset -> default "hermes-3-llama-3.2-3b"
    LLMPORT-02: "my-custom-model" -> override
    LLMPORT-03: "" (empty) -> default (tratado como unset)
    LLMPORT-04: " " (whitespace) -> verbatim (NAO tratado como unset)

  P1 — Retry alvo do erro real de conexao OpenAI-compatible:
    LLMPORT-05: openai.APIConnectionError uma vez, sucesso na 2a tentativa -> retorna sucesso
    LLMPORT-06: openai.APIConnectionError sempre -> re-raise da excecao original (nao RetryError)
    LLMPORT-07: tupla retry_if_exception_type contem openai.APIConnectionError + os 3 httpx,
                NAO contem anthropic.APIConnectionError

  P2 — Sem dependencia `anthropic`:
    LLMPORT-08: agent.py nao importa/referencia `anthropic`
    LLMPORT-09: `import agenticlog.retrieval.generation` funciona sem ImportError/ModuleNotFoundError
    LLMPORT-10: tests/test_agentic_rag.py nao importa/referencia `anthropic`
    LLMPORT-11: requirements.txt nao contem `anthropic==0.104.1`

  P2 — _get_llm() retorna tipo estrutural Protocol minimo:
    LLMPORT-12: _get_llm() ainda retorna ChatOpenAI construido com os 6 kwargs esperados
    LLMPORT-13: anotacao de retorno de _get_llm() e o novo Protocol, nao ChatOpenAI
    LLMPORT-14: Protocol define exatamente __or__, __ror__, invoke
    LLMPORT-15: ChatOpenAI satisfaz o Protocol estruturalmente (isinstance, @runtime_checkable)
    LLMPORT-16: testes existentes que fazem patch em ChatOpenAI continuam passando sem modificacao

  Edge cases (spec "Edge Cases"):
    - Constantes de retry inalteradas: LLM_MAX_RETRY_ATTEMPTS=3,
      LLM_RETRY_WAIT_INITIAL_SECONDS=1.0, LLM_RETRY_WAIT_MAX_SECONDS=4.0
    - before_sleep_log (WARNING) continua sendo usado no decorator de retry

Convenções seguidas:
  - Nenhuma chamada real a LLM/rede — mocks no limite chain.invoke / ChatOpenAI.
  - openai.APIConnectionError(request=httpx.Request("POST", "...")) (requer kwarg `request`).
  - Padrão `_reload()` de tests/test_config_env.py (linhas 25-48) para testes de config via env var.
  - Um arquivo, classes TestACnn... por grupo de critério de aceite.
"""

import os
import subprocess
import sys
import typing
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_NOOP_LOAD_DOTENV = "dotenv.load_dotenv"


# ---------------------------------------------------------------------------
# Shared helper: reload agenticlog.config with env-var overrides
# (mirrors tests/test_config_env.py::TestConfigEnv._reload, lines 25-48)
# ---------------------------------------------------------------------------

def _reload_config(overrides: dict, remove_keys: tuple = ()):
    """Reload agenticlog.config with specific env-var overrides.

    Uses patch.dict without clear=True so system env vars (HOME, PATH,
    USERPROFILE, etc.) are preserved. load_dotenv is mocked to a no-op
    so the on-disk .env file cannot inject values and interfere.
    """
    sys.modules.pop("agenticlog.config", None)
    with patch(_NOOP_LOAD_DOTENV, return_value=None):
        with patch.dict("os.environ", dict(overrides), clear=False):
            for key in remove_keys:
                os.environ.pop(key, None)
            import agenticlog.config as cfg
    sys.modules["agenticlog.config"] = cfg
    return cfg


# ---------------------------------------------------------------------------
# LLMPORT-01..04: LLM_MODEL env var override
# ---------------------------------------------------------------------------

class TestAC01to04LLMModelEnvVar(TestCase):
    """
    LLMPORT-01..04: WHEN LLM_MODEL is set/unset/empty/whitespace in the environment
    AND config.py is loaded THEN LLM_MODEL SHALL reflect the documented fallback rules.
    """

    def setUp(self):
        self._saved_module = sys.modules.get("agenticlog.config")

    def tearDown(self):
        if self._saved_module is not None:
            sys.modules["agenticlog.config"] = self._saved_module
        else:
            sys.modules.pop("agenticlog.config", None)

    def test_ac01_llm_model_unset_falls_back_to_default(self):
        """
        LLMPORT-01: WHEN LLM_MODEL is unset AND config.py is loaded THEN LLM_MODEL
        SHALL equal the hardcoded default literal "hermes-3-llama-3.2-3b".
        """
        cfg = _reload_config({}, remove_keys=("LLM_MODEL",))

        self.assertEqual(cfg.LLM_MODEL, "hermes-3-llama-3.2-3b")
        self.assertEqual(cfg.LLM_MODEL, cfg.DEFAULT_LLM_MODEL)

    def test_ac02_llm_model_custom_value_overrides_default(self):
        """
        LLMPORT-02: WHEN LLM_MODEL is set to a non-empty value ("my-custom-model")
        AND config.py is loaded THEN LLM_MODEL SHALL equal that value.
        """
        cfg = _reload_config({"LLM_MODEL": "my-custom-model"})

        self.assertEqual(cfg.LLM_MODEL, "my-custom-model")

    def test_ac03_llm_model_empty_string_falls_back_to_default(self):
        """
        LLMPORT-03: WHEN LLM_MODEL is set to the empty string ("") AND config.py
        is loaded THEN LLM_MODEL SHALL equal the hardcoded default (empty string
        treated as unset, NOT used verbatim).
        """
        cfg = _reload_config({"LLM_MODEL": ""})

        self.assertEqual(cfg.LLM_MODEL, "hermes-3-llama-3.2-3b")
        self.assertNotEqual(cfg.LLM_MODEL, "")

    def test_ac04_llm_model_whitespace_used_verbatim_not_treated_as_unset(self):
        """
        LLMPORT-04: WHEN LLM_MODEL is set to a whitespace-only value (" ") AND
        config.py is loaded THEN LLM_MODEL SHALL equal that whitespace value
        verbatim (NOT treated as unset — only the exact empty string triggers
        fallback).
        """
        cfg = _reload_config({"LLM_MODEL": " "})

        self.assertEqual(cfg.LLM_MODEL, " ")
        self.assertNotEqual(cfg.LLM_MODEL, "hermes-3-llama-3.2-3b")

    def test_ac01_to_04_independent_test_command_from_spec(self):
        """
        Spec's exact "Independent Test" command (per spec.md):

            python -c "import os; os.environ['LLM_MODEL']='my-custom-model'; \
                       import agenticlog.config as c; print(c.LLM_MODEL)"

        prints "my-custom-model". With LLM_MODEL unset or LLM_MODEL="", prints
        the default "hermes-3-llama-3.2-3b". Run as a real subprocess so the
        outside-in observation matches exactly what a deploying developer would see.
        """
        override_cmd = (
            "import os; os.environ['LLM_MODEL']='my-custom-model'; "
            "import agenticlog.config as c; print(c.LLM_MODEL)"
        )
        result = subprocess.run(
            [sys.executable, "-c", override_cmd],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "my-custom-model")

        unset_env = {k: v for k, v in os.environ.items() if k != "LLM_MODEL"}
        unset_cmd = "import agenticlog.config as c; print(c.LLM_MODEL)"
        result = subprocess.run(
            [sys.executable, "-c", unset_cmd],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            env=unset_env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "hermes-3-llama-3.2-3b")

        empty_env = dict(unset_env)
        empty_env["LLM_MODEL"] = ""
        result = subprocess.run(
            [sys.executable, "-c", unset_cmd],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            env=empty_env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "hermes-3-llama-3.2-3b")


# ---------------------------------------------------------------------------
# LLMPORT-05..07: Retry targets openai.APIConnectionError, not anthropic
# ---------------------------------------------------------------------------

class TestAC05to07RetryTargetsOpenAIConnectionError(TestCase):
    """
    LLMPORT-05..07: WHEN the LLM chain (prompt | llm | parser) is invoked via
    _invoke_chain AND the underlying OpenAI-compatible client raises
    openai.APIConnectionError THEN the system SHALL retry per the existing
    _llm_retry backoff policy, and the retryable-exception tuple SHALL contain
    openai.APIConnectionError (and the 3 httpx exceptions) but NOT
    anthropic.APIConnectionError.
    """

    @patch("time.sleep")
    def test_ac05_connection_error_once_then_success_returns_result(self, mock_sleep):
        """
        LLMPORT-05: WHEN chain.invoke raises openai.APIConnectionError on the
        first attempt AND a subsequent attempt does not raise THEN
        _invoke_chain SHALL retry and SHALL return the successful result.
        """
        from agenticlog.retrieval.generation import _invoke_chain

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            APIConnectionError(
                request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
            ),
            "resposta apos retry",
        ]

        result = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(result, "resposta apos retry")
        self.assertEqual(mock_chain.invoke.call_count, 2)

    @patch("time.sleep")
    def test_ac06_connection_error_every_attempt_reraises_original_not_retry_error(
        self, mock_sleep
    ):
        """
        LLMPORT-06: WHEN chain.invoke raises openai.APIConnectionError on every
        attempt up to LLM_MAX_RETRY_ATTEMPTS THEN _invoke_chain SHALL re-raise
        the original openai.APIConnectionError (not tenacity.RetryError).
        """
        from tenacity import RetryError

        from agenticlog.config import LLM_MAX_RETRY_ATTEMPTS
        from agenticlog.retrieval.generation import _invoke_chain

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = APIConnectionError(
            request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
        )

        raised = None
        try:
            _invoke_chain(mock_chain, {"input": "pergunta"})
        except Exception as exc:  # noqa: BLE001 — capture for type assertion below
            raised = exc

        self.assertIsNotNone(raised)
        self.assertNotIsInstance(raised, RetryError)
        self.assertIsInstance(raised, APIConnectionError)
        self.assertEqual(mock_chain.invoke.call_count, LLM_MAX_RETRY_ATTEMPTS)

    def test_ac07_retry_tuple_contains_openai_and_httpx_not_anthropic(self):
        """
        LLMPORT-07: WHEN _llm_retry's retryable exception tuple is inspected
        THEN it SHALL contain httpx.ConnectError, httpx.TimeoutException,
        httpx.RemoteProtocolError, and openai.APIConnectionError — and SHALL
        NOT contain anthropic.APIConnectionError.
        """
        import agenticlog.retrieval.generation as agent_module

        retry_predicate = agent_module._invoke_chain.retry.retry
        exception_types = retry_predicate.exception_types

        self.assertIn(httpx.ConnectError, exception_types)
        self.assertIn(httpx.TimeoutException, exception_types)
        self.assertIn(httpx.RemoteProtocolError, exception_types)
        self.assertIn(APIConnectionError, exception_types)

        type_module_names = {f"{t.__module__}.{t.__qualname__}" for t in exception_types}
        for entry in type_module_names:
            self.assertFalse(
                entry.startswith("anthropic."),
                f"Found anthropic exception type in retry tuple: {entry}",
            )


# ---------------------------------------------------------------------------
# LLMPORT-08..11: No `anthropic` dependency
# ---------------------------------------------------------------------------

class TestAC08to11NoAnthropicDependency(TestCase):
    """
    LLMPORT-08..11: WHEN agent.py, tests/test_agentic_rag.py, and
    requirements.txt are inspected THEN none SHALL contain `import anthropic`
    or any reference to `anthropic`; importing agenticlog.retrieval.generation SHALL succeed
    without ImportError/ModuleNotFoundError.
    """

    def test_ac08_agent_source_has_no_anthropic_reference(self):
        """
        LLMPORT-08: src/agenticlog/agent.py SHALL NOT contain `import anthropic`
        or any reference to `anthropic`.
        """
        agent_source = (_ROOT / "src" / "agenticlog" / "retrieval" / "generation.py").read_text(encoding="utf-8")

        self.assertNotIn("anthropic", agent_source.lower())

    def test_ac09_import_agenticlog_agent_succeeds_in_subprocess(self):
        """
        LLMPORT-09: WHEN agenticlog.retrieval.generation is imported THEN no
        ImportError/ModuleNotFoundError SHALL occur. Run in a clean subprocess
        (per spec's "Independent Test": `python -c "import agenticlog.retrieval.generation"`).
        """
        result = subprocess.run(
            [sys.executable, "-c", "import agenticlog.retrieval.generation"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(
            result.returncode, 0,
            f"import agenticlog.retrieval.generation failed:\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        self.assertNotIn("ImportError", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_ac10_test_agentic_rag_source_has_no_anthropic_reference(self):
        """
        LLMPORT-10: tests/test_agentic_rag.py SHALL NOT contain `import anthropic`
        or any other reference to `anthropic` package usage.

        Per spec's "Independent Test" note, a match is acceptable ONLY if it is
        inside a test that specifically asserts the ABSENCE of `anthropic`
        (e.g. `teste_12_sem_import_anthropic` / `assertNotIn("anthropic", ...)`).
        No `import anthropic` line is allowed under any circumstance.
        """
        test_source = (_ROOT / "tests" / "test_agentic_rag.py").read_text(encoding="utf-8")

        self.assertNotIn("import anthropic", test_source.lower())

        for line_number, line in enumerate(test_source.splitlines(), start=1):
            if "anthropic" not in line.lower():
                continue
            is_absence_assertion = (
                "sem_import_anthropic" in line.lower()
                or ('"anthropic"' in line.lower() and "assertnotin" in line.lower())
                or "LLMPORT-08/09/10" in line
            )
            self.assertTrue(
                is_absence_assertion,
                f"Unexpected 'anthropic' reference at line {line_number}: {line!r}",
            )

    def test_ac11_requirements_txt_does_not_contain_anthropic(self):
        """
        LLMPORT-11: requirements.txt SHALL NOT contain the line
        `anthropic==0.104.1` (or any anthropic dependency).
        """
        requirements_source = (_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertNotIn("anthropic", requirements_source.lower())

    def test_ac08_to_11_grep_independent_test_from_spec(self):
        """
        Spec's exact "Independent Test":
        `grep -ri anthropic src/agenticlog/agent.py tests/test_agentic_rag.py
         requirements.txt` returns no matches (or only matches inside a test
        that specifically asserts the ABSENCE of `anthropic`, per spec note).
        """
        targets = [
            _ROOT / "src" / "agenticlog" / "retrieval" / "generation.py",
            _ROOT / "tests" / "test_agentic_rag.py",
            _ROOT / "requirements.txt",
        ]
        for target in targets:
            content = target.read_text(encoding="utf-8")
            for line_number, line in enumerate(content.splitlines(), start=1):
                if "anthropic" not in line.lower():
                    continue
                is_absence_assertion = (
                    "sem_import_anthropic" in line.lower()
                    or ('"anthropic"' in line.lower() and "assertnotin" in line.lower())
                    or "LLMPORT-08/09/10" in line
                )
                self.assertTrue(
                    is_absence_assertion,
                    f"Found 'anthropic' reference in {target}:{line_number}: {line!r}",
                )


# ---------------------------------------------------------------------------
# LLMPORT-12..16: _get_llm() return type is minimal structural Protocol
# ---------------------------------------------------------------------------

class TestAC12to16GetLlmProtocolType(TestCase):
    """
    LLMPORT-12..16: WHEN _get_llm() is inspected THEN it SHALL still return a
    ChatOpenAI instance constructed with the same 6 kwargs as before, but its
    declared return-type annotation SHALL be a minimal structural Protocol
    (only __or__, __ror__, invoke) that ChatOpenAI satisfies structurally
    (@runtime_checkable) — and existing ChatOpenAI-patching tests SHALL
    continue to pass unmodified.
    """

    def setUp(self):
        import agenticlog.retrieval.generation as agent_module
        self.agent_module = agent_module
        self._saved_llm = agent_module._llm
        agent_module._llm = None

    def tearDown(self):
        self.agent_module._llm = self._saved_llm

    @patch("agenticlog.retrieval.generation.ChatOpenAI")
    def test_ac12_get_llm_constructs_chatopenai_with_all_six_kwargs(self, mock_chat_openai):
        """
        LLMPORT-12: _get_llm() SHALL return a ChatOpenAI instance constructed
        exactly as before, with all 6 constructor kwargs:
        model_name=LLM_MODEL, openai_api_base=LLM_API_BASE,
        openai_api_key=LLM_API_KEY, temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS, request_timeout=LLM_TIMEOUT_SECONDS.
        """
        from agenticlog.config import (
            LLM_API_BASE,
            LLM_API_KEY,
            LLM_MAX_TOKENS,
            LLM_MODEL,
            LLM_TEMPERATURE,
            LLM_TIMEOUT_SECONDS,
        )
        from agenticlog.retrieval.generation import _get_llm

        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        result = _get_llm()

        self.assertIs(result, mock_instance)
        mock_chat_openai.assert_called_once()
        _, kwargs = mock_chat_openai.call_args
        self.assertEqual(kwargs.get("model_name"), LLM_MODEL)
        self.assertEqual(kwargs.get("openai_api_base"), LLM_API_BASE)
        self.assertEqual(kwargs.get("openai_api_key"), LLM_API_KEY)
        self.assertEqual(kwargs.get("temperature"), LLM_TEMPERATURE)
        self.assertEqual(kwargs.get("max_tokens"), LLM_MAX_TOKENS)
        self.assertEqual(kwargs.get("request_timeout"), LLM_TIMEOUT_SECONDS)

    def test_ac13_get_llm_return_annotation_is_protocol_not_chatopenai(self):
        """
        LLMPORT-13: WHEN _get_llm()'s signature is inspected THEN its declared
        return type SHALL be the new minimal Protocol type (not ChatOpenAI).
        """
        import inspect

        from agenticlog.retrieval.generation import ChatOpenAI, LLMClient, _get_llm

        return_annotation = inspect.signature(_get_llm).return_annotation

        self.assertIs(return_annotation, LLMClient)
        self.assertIsNot(return_annotation, ChatOpenAI)
        self.assertNotEqual(return_annotation, ChatOpenAI)

    def test_ac14_protocol_defines_exactly_or_ror_and_invoke(self):
        """
        LLMPORT-14: WHEN the new Protocol type is inspected THEN it SHALL
        define exactly: __or__, __ror__, invoke (no more, no less).
        """
        from agenticlog.retrieval.generation import LLMClient

        self.assertTrue(issubclass(LLMClient, typing.Protocol))

        # Members explicitly declared on the Protocol body (excluding dunders
        # inherited from object/Protocol machinery and Python internals).
        protocol_own_members = {
            name
            for name, value in vars(LLMClient).items()
            if callable(value) and not name.startswith("_abc")
        }
        # Filter out Protocol/typing machinery dunders that are not part of
        # the structural interface itself.
        excluded = {
            "__init__", "__new__", "__subclasshook__", "__class_getitem__",
            "__init_subclass__",
        }
        declared_interface = protocol_own_members - excluded

        self.assertEqual(declared_interface, {"__or__", "__ror__", "invoke"})

    def test_ac15_chatopenai_satisfies_llmclient_protocol_structurally(self):
        """
        LLMPORT-15: WHEN a ChatOpenAI instance is checked against the Protocol
        THEN it SHALL satisfy it structurally (isinstance returns True given
        @runtime_checkable) — no wrapper, adapter, or subclass introduced.
        """
        from agenticlog.retrieval.generation import LLMClient

        self.assertTrue(getattr(LLMClient, "_is_runtime_protocol", False))

        # Construct a real ChatOpenAI instance (no network call on construction)
        # to verify structural satisfaction without mocking away the type.
        from langchain_openai import ChatOpenAI

        real_instance = ChatOpenAI(
            model_name="hermes-3-llama-3.2-3b",
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="hermes",
            temperature=0,
            max_tokens=2048,
            request_timeout=60.0,
        )

        self.assertIsInstance(real_instance, LLMClient)

    def test_ac16_existing_chatopenai_patching_tests_pass_unmodified(self):
        """
        LLMPORT-16: WHEN existing tests that patch("agenticlog.retrieval.generation.ChatOpenAI")
        (teste_9_import_sem_lmstudio, teste_10_get_llm_singleton in
        tests/test_agentic_rag.py, and test_ac06_llm_created_with_timeout_from_config
        in tests/acceptance/test_retry_logic.py) run THEN they SHALL continue to
        pass without modification to their patch targets.

        Run as a real pytest subprocess for a true outside-in observation —
        the spec's exact "Independent Test" pytest invocation.

        The two explicit node IDs and the `-k`-filtered file are run as
        separate invocations because `-k` applies globally to all collected
        items and would deselect the explicit node IDs.
        """
        result_explicit = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_agentic_rag.py::TestAgenticRAG::teste_9_import_sem_lmstudio",
                "tests/test_agentic_rag.py::TestAgenticRAG::teste_10_get_llm_singleton",
                "-v",
            ],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        self.assertEqual(
            result_explicit.returncode, 0,
            f"Existing ChatOpenAI-patching tests failed:\n"
            f"{result_explicit.stdout}\n{result_explicit.stderr}",
        )
        self.assertIn("teste_9_import_sem_lmstudio", result_explicit.stdout)
        self.assertIn("teste_10_get_llm_singleton", result_explicit.stdout)

        result_retry_logic = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/acceptance/test_retry_logic.py",
                "-k", "test_ac06_llm_created_with_timeout_from_config",
                "-v",
            ],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        self.assertEqual(
            result_retry_logic.returncode, 0,
            f"test_ac06_llm_created_with_timeout_from_config failed:\n"
            f"{result_retry_logic.stdout}\n{result_retry_logic.stderr}",
        )
        self.assertIn("test_ac06_llm_created_with_timeout_from_config", result_retry_logic.stdout)
        self.assertIn("PASSED", result_retry_logic.stdout)


# ---------------------------------------------------------------------------
# Edge cases: retry-bound constants unchanged; before_sleep_log at WARNING
# ---------------------------------------------------------------------------

class TestEdgeCasesRetryBoundsAndLogging(TestCase):
    """
    Edge cases (spec.md "Edge Cases"):
      - LLM_MAX_RETRY_ATTEMPTS=3, LLM_RETRY_WAIT_INITIAL_SECONDS=1.0,
        LLM_RETRY_WAIT_MAX_SECONDS=4.0 SHALL remain unchanged by the
        anthropic -> openai exception-type swap.
      - before_sleep_log (WARNING level) logging behaviour SHALL be unaffected.
    """

    def test_edge_retry_bound_constants_unchanged(self):
        """Retry-bound constants in config.py remain at their documented values."""
        from agenticlog.config import (
            LLM_MAX_RETRY_ATTEMPTS,
            LLM_RETRY_WAIT_INITIAL_SECONDS,
            LLM_RETRY_WAIT_MAX_SECONDS,
        )

        self.assertEqual(LLM_MAX_RETRY_ATTEMPTS, 3)
        self.assertEqual(LLM_RETRY_WAIT_INITIAL_SECONDS, 1.0)
        self.assertEqual(LLM_RETRY_WAIT_MAX_SECONDS, 4.0)

    def test_edge_invoke_chain_retry_decorator_uses_config_bounds(self):
        """_invoke_chain's tenacity decorator stop/wait bounds match config.py
        (not hardcoded duplicates), confirming the exception-type swap did not
        alter the retry bounds."""
        import agenticlog.retrieval.generation as agent_module
        from agenticlog.config import (
            LLM_MAX_RETRY_ATTEMPTS,
            LLM_RETRY_WAIT_INITIAL_SECONDS,
            LLM_RETRY_WAIT_MAX_SECONDS,
        )

        retry_obj = agent_module._invoke_chain.retry
        self.assertEqual(retry_obj.stop.max_attempt_number, LLM_MAX_RETRY_ATTEMPTS)
        self.assertEqual(retry_obj.wait.min, LLM_RETRY_WAIT_INITIAL_SECONDS)
        self.assertEqual(retry_obj.wait.max, LLM_RETRY_WAIT_MAX_SECONDS)
        self.assertTrue(retry_obj.reraise)

    @patch("time.sleep")
    def test_edge_retry_logs_warning_on_openai_connection_error(self, mock_sleep, caplog=None):
        """WHEN a retry occurs due to openai.APIConnectionError THEN a WARNING-level
        log record SHALL be emitted via before_sleep_log (logging unaffected by the
        anthropic -> openai exception-type swap)."""

        from agenticlog.retrieval.generation import _invoke_chain

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [
            APIConnectionError(
                request=httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
            ),
            "resposta ok",
        ]

        logger_name = "agenticlog.retrieval.generation"
        with self._assert_warning_logged(logger_name):
            result = _invoke_chain(mock_chain, {"input": "pergunta"})

        self.assertEqual(result, "resposta ok")

    def _assert_warning_logged(self, logger_name: str):
        """Context manager asserting at least one WARNING record is logged
        on `logger_name` during the block (before_sleep_log behaviour)."""
        import logging

        class _Capture:
            def __enter__(self_inner):  # noqa: N805
                self_inner.records = []
                self_inner.handler = logging.Handler()
                self_inner.handler.setLevel(logging.WARNING)
                self_inner.handler.emit = self_inner.records.append
                self_inner.logger = logging.getLogger(logger_name)
                self_inner.previous_level = self_inner.logger.level
                self_inner.logger.setLevel(logging.WARNING)
                self_inner.logger.addHandler(self_inner.handler)
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):  # noqa: N805
                self_inner.logger.removeHandler(self_inner.handler)
                self_inner.logger.setLevel(self_inner.previous_level)
                warning_records = [
                    r for r in self_inner.records if r.levelno == logging.WARNING
                ]
                assert warning_records, (
                    f"Expected at least one WARNING log record on '{logger_name}' "
                    "during retry (before_sleep_log behaviour), found none."
                )
                return False

        return _Capture()


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
