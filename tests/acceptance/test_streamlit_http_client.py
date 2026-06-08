# AgenticLog — Acceptance Tests: Streamlit HTTP Client feature
"""
Verifies every acceptance criterion (AC-01 through AC-12) defined in the
approved story for the streamlit-http-client feature.

All tests are black-box: they drive app.py through the same interfaces a
real user would — the Streamlit AppTest harness and module-level inspection.
No implementation internals are imported except for the public constants that
the story explicitly requires to exist (MSG_*, API_CLIENT_TIMEOUT_SECONDS).

Mock strategy: patch("app.httpx.post") — never make real HTTP calls.
"""

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(_ROOT / "app.py")
_APP_SOURCE = Path(_APP_PATH).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _success_mock(
    ranked_response: object = "Resposta OK.",
    confidence_score: float | None = 0.85,
    retrieved_info: list | None = None,
    next_step: str = "retrieve",
) -> MagicMock:
    """Return a MagicMock representing a clean HTTP 200 response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "ranked_response": ranked_response,
        "confidence_score": confidence_score,
        "next_step": next_step,
        "retrieved_info": retrieved_info if retrieved_info is not None else [],
    }
    mock.raise_for_status.return_value = None
    return mock


def _error_mock(status_code: int, detail: str) -> MagicMock:
    """Return a MagicMock whose raise_for_status() raises HTTPStatusError."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {"detail": detail}
    mock.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=MagicMock(),
        response=mock,
    )
    return mock


def _run_query(mock_post: MagicMock, query: str = "pergunta de teste") -> AppTest:
    """Run the Streamlit app with a query click and return the AppTest instance.

    mock_post must be a fully configured callable — either MagicMock(return_value=response)
    for HTTP response cases or MagicMock(side_effect=exception) for network-error cases.
    The patch covers all three .run() calls so every rerun sees the same mock.
    """
    with patch("app.httpx.post", mock_post):
        at = AppTest.from_file(_APP_PATH)
        at.run()
        at.text_input[0].set_value(query).run()
        at.button[0].click().run()
    return at


def _error_texts(at: AppTest) -> list[str]:
    """Extract all st.error widget values, excluding confidence-level badges."""
    return [
        e.value for e in at.error
        if not any(kw in e.value for kw in ("Confiança baixa", "Confiança média", "Confiança alta"))
    ]


# ---------------------------------------------------------------------------
# AC-01: POST /query called on submit, 200 response stored in session_state
# ---------------------------------------------------------------------------

class TestAC01SuccessPath(unittest.TestCase):
    """
    AC-01: WHEN the operator submits a non-empty query AND the API returns
    HTTP 200 THEN app.py SHALL have sent POST /query and stored
    ranked_response, confidence_score, next_step, retrieved_info in
    st.session_state.
    """

    def teste_1_200_response_populates_session_state(self) -> None:
        mock = _success_mock(
            ranked_response="Prazo de entrega é 5 dias úteis.",
            confidence_score=0.9,
            next_step="retrieve",
        )
        at = _run_query(MagicMock(return_value=mock), "Qual o prazo de entrega?")

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        self.assertEqual(at.session_state["ranked_response"], "Prazo de entrega é 5 dias úteis.")
        self.assertEqual(at.session_state["confidence_score"], 0.9)
        self.assertEqual(at.session_state["next_step"], "retrieve")
        self.assertEqual(at.session_state["retrieved_info"], [])
        self.assertEqual(_error_texts(at), [], msg="st.error should not be called on success")

    def teste_2_post_called_once_per_submit(self) -> None:
        mock_post = MagicMock(return_value=_success_mock())
        with patch("app.httpx.post", mock_post):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_input[0].set_value("pergunta").run()
            at.button[0].click().run()
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        # URL must contain /query
        url_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("url", "")
        self.assertIn("/query", url_arg)
        # Body must carry the query text
        json_body = call_kwargs.kwargs.get("json", {})
        self.assertEqual(json_body.get("query"), "pergunta")


# ---------------------------------------------------------------------------
# AC-02: retrieved_info docs accessed as plain dict with correct keys / index
# ---------------------------------------------------------------------------

class TestAC02DictAccess(unittest.TestCase):
    """
    AC-02: WHEN retrieved_info is rendered THEN each document SHALL be
    accessed as a plain dict using doc["page_content"] and
    doc["metadata"].get("source", "Desconhecida"), and loop index i SHALL
    be used for all widget keys.
    """

    def teste_1_docs_rendered_via_dict_key_access(self) -> None:
        docs = [
            {"page_content": "conteúdo A", "metadata": {"source": "frete.json"}},
            {"page_content": "conteúdo B", "metadata": {"source": "estoque.json"}},
        ]
        at = _run_query(MagicMock(return_value=_success_mock(retrieved_info=docs, ranked_response="resp")))

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        expander_labels = [e.label for e in at.expander]
        self.assertTrue(
            any("frete.json" in lbl for lbl in expander_labels),
            msg=f"Source 'frete.json' missing from expanders: {expander_labels}",
        )
        self.assertTrue(
            any("estoque.json" in lbl for lbl in expander_labels),
            msg=f"Source 'estoque.json' missing from expanders: {expander_labels}",
        )

    def teste_2_missing_source_key_falls_back_to_desconhecida(self) -> None:
        docs = [{"page_content": "sem source", "metadata": {}}]
        at = _run_query(MagicMock(return_value=_success_mock(retrieved_info=docs, ranked_response="resp")))

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        expander_labels = [e.label for e in at.expander]
        self.assertTrue(
            any("Desconhecida" in lbl for lbl in expander_labels),
            msg=f"Fallback 'Desconhecida' missing from expanders: {expander_labels}",
        )

    def teste_3_widget_keys_use_loop_index(self) -> None:
        """Loop index i must be used for text_area keys (doc_content_0, doc_content_1, ...)."""
        # The source text must use `key=f"doc_content_{i}"` — inspect statically
        self.assertIn(
            'key=f"doc_content_{i}"',
            _APP_SOURCE,
            msg="app.py must use key=f'doc_content_{i}' for text_area widget keys",
        )


# ---------------------------------------------------------------------------
# AC-03: HTTP 503 + LMStudio detail → st.error with that message
# ---------------------------------------------------------------------------

class TestAC03Http503LMStudio(unittest.TestCase):
    """
    AC-03: WHEN the API returns HTTP 503 with
    {"detail": "LMStudio indisponível..."} THEN st.error SHALL display
    that Portuguese message and session_state SHALL NOT be mutated.
    """

    def teste_1_503_lmstudio_shows_correct_error(self) -> None:
        import app as app_module  # noqa: PLC0415

        mock = _error_mock(503, "LMStudio indisponível. Inicie o servidor e carregue o modelo.")
        at = _run_query(MagicMock(return_value=mock))

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        errors = _error_texts(at)
        self.assertTrue(
            any(app_module.MSG_LMSTUDIO_DOWN in t for t in errors),
            msg=f"MSG_LMSTUDIO_DOWN not found in st.error calls: {errors}",
        )

    def teste_2_503_lmstudio_does_not_mutate_session_state(self) -> None:
        mock = _error_mock(503, "LMStudio indisponível. Inicie o servidor e carregue o modelo.")
        at = _run_query(MagicMock(return_value=mock))

        self.assertIsNone(at.session_state.ranked_response)
        self.assertIsNone(at.session_state.confidence_score)
        self.assertIsNone(at.session_state.next_step)
        self.assertEqual(at.session_state.retrieved_info, [])


# ---------------------------------------------------------------------------
# AC-04: HTTP 503 + vectordb detail → st.error with that message
# ---------------------------------------------------------------------------

class TestAC04Http503Vectordb(unittest.TestCase):
    """
    AC-04: WHEN the API returns HTTP 503 with
    {"detail": "Base vetorial não encontrada..."} THEN st.error SHALL
    display that Portuguese message and session_state SHALL NOT be mutated.
    """

    def teste_1_503_vectordb_shows_correct_error(self) -> None:
        import app as app_module  # noqa: PLC0415

        mock = _error_mock(503, "Base vetorial não encontrada. Execute: python -m agenticlog.rag")
        at = _run_query(MagicMock(return_value=mock))

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        errors = _error_texts(at)
        self.assertTrue(
            any(app_module.MSG_VECTORDB_AUSENTE in t for t in errors),
            msg=f"MSG_VECTORDB_AUSENTE not found in st.error calls: {errors}",
        )

    def teste_2_503_vectordb_does_not_mutate_session_state(self) -> None:
        mock = _error_mock(503, "Base vetorial não encontrada. Execute: python -m agenticlog.rag")
        at = _run_query(MagicMock(return_value=mock))

        self.assertIsNone(at.session_state.ranked_response)


# ---------------------------------------------------------------------------
# AC-05: HTTP 500 → st.error generic + st.expander("Detalhes do erro")
# ---------------------------------------------------------------------------

class TestAC05Http500(unittest.TestCase):
    """
    AC-05: WHEN the API returns HTTP 500 THEN st.error SHALL show a generic
    error message AND st.expander("Detalhes do erro") SHALL contain the raw
    detail string; session state SHALL NOT be mutated.
    """

    def teste_1_500_shows_generic_error_and_expander(self) -> None:
        import app as app_module  # noqa: PLC0415

        mock = _error_mock(500, "Internal traceback detail here.")
        at = _run_query(MagicMock(return_value=mock))

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        errors = _error_texts(at)
        self.assertTrue(
            any(app_module.MSG_ERRO_INTERNO in t for t in errors),
            msg=f"MSG_ERRO_INTERNO not found in st.error calls: {errors}",
        )
        expander_labels = [e.label for e in at.expander]
        self.assertTrue(
            any("Detalhes do erro" in lbl for lbl in expander_labels),
            msg=f"Expander 'Detalhes do erro' not found. Expanders: {expander_labels}",
        )

    def teste_2_500_does_not_mutate_session_state(self) -> None:
        mock = _error_mock(500, "server error")
        at = _run_query(MagicMock(return_value=mock))

        self.assertIsNone(at.session_state.ranked_response)


# ---------------------------------------------------------------------------
# AC-06: HTTP 422 → st.error validation message
# ---------------------------------------------------------------------------

class TestAC06Http422(unittest.TestCase):
    """
    AC-06: WHEN the API returns HTTP 422 THEN st.error SHALL show a
    validation error message; session state SHALL NOT be mutated.
    """

    def teste_1_422_shows_validation_error(self) -> None:
        import app as app_module  # noqa: PLC0415

        mock = _error_mock(422, "value is not a valid string")
        at = _run_query(MagicMock(return_value=mock))

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        errors = _error_texts(at)
        self.assertTrue(
            any(app_module.MSG_ERRO_VALIDACAO in t for t in errors),
            msg=f"MSG_ERRO_VALIDACAO not found in st.error calls: {errors}",
        )

    def teste_2_422_does_not_mutate_session_state(self) -> None:
        mock = _error_mock(422, "value is not a valid string")
        at = _run_query(MagicMock(return_value=mock))

        self.assertIsNone(at.session_state.ranked_response)


# ---------------------------------------------------------------------------
# AC-07: httpx.ConnectError → st.error with "Não foi possível conectar..."
# ---------------------------------------------------------------------------

class TestAC07ConnectError(unittest.TestCase):
    """
    AC-07: WHEN httpx.ConnectError is raised THEN st.error SHALL show
    "Não foi possível conectar ao servidor FastAPI..."; session state
    SHALL NOT be mutated.
    """

    def teste_1_connect_error_shows_correct_message(self) -> None:
        import app as app_module  # noqa: PLC0415

        mock_post = MagicMock(side_effect=httpx.ConnectError("connection refused"))
        at = _run_query(mock_post)

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        errors = _error_texts(at)
        self.assertTrue(
            any(app_module.MSG_CONNECT_ERROR in t for t in errors),
            msg=f"MSG_CONNECT_ERROR not found in st.error calls: {errors}",
        )
        self.assertIn(
            "Não foi possível conectar ao servidor FastAPI",
            app_module.MSG_CONNECT_ERROR,
            msg="MSG_CONNECT_ERROR must contain the exact Portuguese phrase",
        )

    def teste_2_connect_error_does_not_mutate_session_state(self) -> None:
        mock_post = MagicMock(side_effect=httpx.ConnectError("connection refused"))
        at = _run_query(mock_post)

        self.assertIsNone(at.session_state.ranked_response)


# ---------------------------------------------------------------------------
# AC-08: httpx.TimeoutException → st.error timeout message
# ---------------------------------------------------------------------------

class TestAC08TimeoutException(unittest.TestCase):
    """
    AC-08: WHEN httpx.TimeoutException is raised THEN st.error SHALL show
    a timeout message; session state SHALL NOT be mutated.
    """

    def teste_1_timeout_shows_timeout_message(self) -> None:
        import app as app_module  # noqa: PLC0415

        mock_post = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        at = _run_query(mock_post)

        self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
        errors = _error_texts(at)
        self.assertTrue(
            any(app_module.MSG_TIMEOUT in t for t in errors),
            msg=f"MSG_TIMEOUT not found in st.error calls: {errors}",
        )

    def teste_2_timeout_does_not_mutate_session_state(self) -> None:
        mock_post = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        at = _run_query(mock_post)

        self.assertIsNone(at.session_state.ranked_response)


# ---------------------------------------------------------------------------
# AC-09: app.py does not import forbidden names
# ---------------------------------------------------------------------------

class TestAC09NoDeadImports(unittest.TestCase):
    """
    AC-09: WHEN app.py is imported THEN it SHALL NOT import AgentState,
    agent_workflow, check_lmstudio_health, LMStudioUnavailableError, or
    anthropic.
    """

    _FORBIDDEN = (
        "AgentState",
        "agent_workflow",
        "check_lmstudio_health",
        "LMStudioUnavailableError",
        "anthropic",
    )

    def teste_1_forbidden_names_not_in_app_module_dict(self) -> None:
        import app as app_module  # noqa: PLC0415

        app_names = set(app_module.__dict__.keys())
        for name in self._FORBIDDEN:
            self.assertNotIn(
                name,
                app_names,
                msg=f"Forbidden name '{name}' found in app module namespace",
            )

    def teste_2_forbidden_imports_not_in_source(self) -> None:
        """Source-level check: none of the forbidden names appear as imported symbols."""
        import_lines = [
            line for line in _APP_SOURCE.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        import_block = "\n".join(import_lines)
        for name in self._FORBIDDEN:
            self.assertNotIn(
                name,
                import_block,
                msg=f"Forbidden import '{name}' found in app.py import statements",
            )


# ---------------------------------------------------------------------------
# AC-10: URL built from API_HOST and API_PORT (no hardcoded host/port)
# ---------------------------------------------------------------------------

class TestAC10ConfigDrivenUrl(unittest.TestCase):
    """
    AC-10: WHEN _consultar_api() builds the URL THEN it SHALL use API_HOST
    and API_PORT from config.py — no hardcoded host or port strings.
    """

    def teste_1_url_uses_api_host_and_api_port_variables(self) -> None:
        self.assertIn(
            "API_HOST",
            _APP_SOURCE,
            msg="app.py must reference API_HOST from config",
        )
        self.assertIn(
            "API_PORT",
            _APP_SOURCE,
            msg="app.py must reference API_PORT from config",
        )

    def teste_2_no_hardcoded_ip_in_consultar_api(self) -> None:
        """The _consultar_api function body must not contain a literal IP or port."""
        import app as app_module  # noqa: PLC0415

        func_source = inspect.getsource(app_module._consultar_api)
        self.assertNotIn(
            "127.0.0.1",
            func_source,
            msg="_consultar_api must not hardcode 127.0.0.1",
        )
        self.assertNotIn(
            '"8000"',
            func_source,
            msg="_consultar_api must not hardcode port 8000 as string",
        )
        self.assertNotIn(
            ":8000",
            func_source,
            msg="_consultar_api must not hardcode :8000 in URL",
        )

    def teste_3_consultar_api_passes_timeout_from_config(self) -> None:
        """httpx.post must receive timeout= equal to config.API_CLIENT_TIMEOUT_SECONDS."""
        from agenticlog import config  # noqa: PLC0415

        mock_post = MagicMock(return_value=_success_mock())
        with patch("app.httpx.post", mock_post):
            import app as app_module  # noqa: PLC0415
            app_module._consultar_api("test query")

        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(
            call_kwargs.get("timeout"),
            config.API_CLIENT_TIMEOUT_SECONDS,
            msg=(
                "timeout kwarg must equal"
                f" API_CLIENT_TIMEOUT_SECONDS={config.API_CLIENT_TIMEOUT_SECONDS}"
            ),
        )


# ---------------------------------------------------------------------------
# AC-11: API_CLIENT_TIMEOUT_SECONDS = 120 in config.py; used in _consultar_api
# ---------------------------------------------------------------------------

class TestAC11TimeoutConstant(unittest.TestCase):
    """
    AC-11: API_CLIENT_TIMEOUT_SECONDS = 120 SHALL exist in config.py and
    SHALL be the timeout value used by _consultar_api.
    """

    def teste_1_config_exports_api_client_timeout_120(self) -> None:
        from agenticlog import config  # noqa: PLC0415

        self.assertTrue(
            hasattr(config, "API_CLIENT_TIMEOUT_SECONDS"),
            msg="config.py must export API_CLIENT_TIMEOUT_SECONDS",
        )
        self.assertEqual(
            config.API_CLIENT_TIMEOUT_SECONDS,
            120,
            msg=f"API_CLIENT_TIMEOUT_SECONDS must be 120, got {config.API_CLIENT_TIMEOUT_SECONDS}",
        )

    def teste_2_consultar_api_uses_api_client_timeout_not_llm_timeout(self) -> None:
        """_consultar_api source must reference API_CLIENT_TIMEOUT_SECONDS, not LLM_TIMEOUT."""
        import app as app_module  # noqa: PLC0415

        func_source = inspect.getsource(app_module._consultar_api)
        self.assertIn(
            "API_CLIENT_TIMEOUT_SECONDS",
            func_source,
            msg="_consultar_api must use API_CLIENT_TIMEOUT_SECONDS as its timeout",
        )
        self.assertNotIn(
            "LLM_TIMEOUT_SECONDS",
            func_source,
            msg="_consultar_api must NOT use LLM_TIMEOUT_SECONDS",
        )


# ---------------------------------------------------------------------------
# AC-12: Zero attribute access on doc objects in app.py
# ---------------------------------------------------------------------------

class TestAC12NoDotAccessOnDocs(unittest.TestCase):
    """
    AC-12: app.py SHALL use zero attribute access on doc objects.
    All document fields must be accessed via dict-key syntax.
    """

    def teste_1_no_doc_page_content_attribute_access(self) -> None:
        self.assertNotIn(
            "doc.page_content",
            _APP_SOURCE,
            msg="app.py must not use doc.page_content (use doc['page_content'])",
        )

    def teste_2_no_doc_metadata_attribute_access(self) -> None:
        self.assertNotIn(
            "doc.metadata",
            _APP_SOURCE,
            msg="app.py must not use doc.metadata (use doc['metadata'])",
        )

    def teste_3_no_doc_id_attribute_access(self) -> None:
        self.assertNotIn(
            "doc.id",
            _APP_SOURCE,
            msg="app.py must not use doc.id (use loop index i instead)",
        )

    def teste_4_dict_key_access_present_in_source(self) -> None:
        self.assertIn(
            'doc["page_content"]',
            _APP_SOURCE,
            msg='app.py must use doc["page_content"] dict-key syntax',
        )
        self.assertIn(
            'doc["metadata"]',
            _APP_SOURCE,
            msg='app.py must use doc["metadata"] dict-key syntax',
        )


if __name__ == "__main__":
    unittest.main()
