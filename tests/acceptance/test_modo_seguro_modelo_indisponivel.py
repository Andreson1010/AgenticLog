"""
Testes de aceitação — Modo Seguro: Modelo Indisponível

Verifica cada critério de aceite da feature:
  "QUANDO o LLM/modelo estiver indisponível, o sistema SHALL responder 200-degraded
   em vez de propagar um erro, garantindo disponibilidade parcial ao usuário."

Cada teste mapeia a exatamente um critério (AC-01 a AC-11).
Naming: test_ac_modo_seguro_<N>_description (convenção do projeto).

Todos os chamados ao LLM, ChromaDB e rede são mockados — nenhuma chamada real
a serviços externos é feita nestes testes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from agenticlog.agent import AgentState
from agenticlog.api import MSG_VECTORDB_AUSENTE, app
from agenticlog.config import RESPOSTA_PADRAO_SEGURA
from agenticlog.health import (
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    reset_health_check_sentinel,
)

# Raiz do repositório — usada para localizar app.py nos testes de AppTest.
_root = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Shared mock history store — injected before any TestClient is created so
# that the lifespan does not attempt to create a real HistoryStore on disk.
# ---------------------------------------------------------------------------

_mock_history_store = MagicMock()
_mock_history_store.append.return_value = None
_mock_history_store.read_all.return_value = []
app.state.history_store = _mock_history_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_state(**kwargs) -> AgentState:
    """Returns a valid AgentState with sensible defaults for the happy path."""
    defaults = dict(
        query="prazo de entrega SP-RJ",
        next_step="retrieve",
        retrieved_info=[],
        possible_responses=[],
        similarity_scores=[],
        ranked_response="Prazo de entrega: 2 dias úteis.",
        confidence_score=0.88,
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


def _client_ctx(
    *,
    health_side_effect=None,
    invoke_side_effect=None,
    invoke_return=None,
    vectordb_ok: bool = True,
):
    """Context-manager that yields a configured TestClient with mocks applied.

    Parameters
    ----------
    health_side_effect: if set, check_lmstudio_health raises this exception.
    invoke_side_effect: if set, agent_workflow.invoke raises this exception.
    invoke_return: if set, agent_workflow.invoke returns this value (default: happy-path state).
    vectordb_ok: if False, startup marks vectordb_pronto=False → 503 on every request.
    """
    estado = invoke_return if invoke_return is not None else _make_agent_state()
    vectordb_effect = None if vectordb_ok else RuntimeError(MSG_VECTORDB_AUSENTE)

    kwargs_invoke = {}
    if invoke_side_effect is not None:
        kwargs_invoke["side_effect"] = invoke_side_effect
    else:
        kwargs_invoke["return_value"] = estado

    kwargs_health = {}
    if health_side_effect is not None:
        kwargs_health["side_effect"] = health_side_effect

    return (
        patch("agenticlog.serving.api.inicializar_recursos"),
        patch("agenticlog.serving.api._verificar_vectordb", side_effect=vectordb_effect),
        patch("agenticlog.serving.api.check_lmstudio_health", **kwargs_health),
        patch("agenticlog.serving.api.agent_workflow.invoke", **kwargs_invoke),
        patch("agenticlog.serving.api.HistoryStore", return_value=_mock_history_store),
    )


# ---------------------------------------------------------------------------
# AC-01 — Happy path: healthy LLM → 200, degraded=false, real answer present
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_01_happy_path_degraded_false_real_answer():
    """AC-01: WHEN LLM is reachable and check_lmstudio_health passes THEN POST /query
    SHALL return HTTP 200 with degraded=false and a non-empty ranked_response."""
    patches = _client_ctx()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo de entrega SP-RJ"})

    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is False
    assert isinstance(data["ranked_response"], str)
    assert len(data["ranked_response"]) > 0
    assert data["ranked_response"] != RESPOSTA_PADRAO_SEGURA


# ---------------------------------------------------------------------------
# AC-02 — Pre-flight LMStudioUnavailableError → 200-degraded (not 503)
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_02_preflight_lmstudio_unavailable_returns_200_degraded():
    """AC-02: WHEN check_lmstudio_health raises LMStudioUnavailableError THEN POST /query
    SHALL return HTTP 200 (not 503) with degraded=true and RESPOSTA_PADRAO_SEGURA."""
    reset_health_check_sentinel()
    try:
        patches = _client_ctx(
            health_side_effect=LMStudioUnavailableError("LMStudio offline")
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "teste"})
    finally:
        reset_health_check_sentinel()

    assert response.status_code == 200, (
        f"Expected 200 (safe mode), got {response.status_code}. "
        "LMStudioUnavailableError should not propagate as 503."
    )
    data = response.json()
    assert data["degraded"] is True
    assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA


# ---------------------------------------------------------------------------
# AC-03 — Pre-flight ModeloNaoCarregadoError → 200-degraded (new case)
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_03_preflight_modelo_nao_carregado_returns_200_degraded():
    """AC-03: WHEN check_lmstudio_health raises ModeloNaoCarregadoError (model not loaded)
    THEN POST /query SHALL return HTTP 200 with degraded=true and RESPOSTA_PADRAO_SEGURA.
    ModeloNaoCarregadoError is a subclass of LMStudioUnavailableError."""
    reset_health_check_sentinel()
    try:
        patches = _client_ctx(
            health_side_effect=ModeloNaoCarregadoError("Modelo não carregado")
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "teste"})
    finally:
        reset_health_check_sentinel()

    assert response.status_code == 200, (
        f"Expected 200 (safe mode), got {response.status_code}. "
        "ModeloNaoCarregadoError should trigger safe mode, not 503."
    )
    data = response.json()
    assert data["degraded"] is True
    assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA


# ---------------------------------------------------------------------------
# AC-04 — Mid-call openai.APIConnectionError → 200-degraded
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_04_midcall_api_connection_error_returns_200_degraded():
    """AC-04: WHEN pre-flight passes but agent_workflow.invoke raises openai.APIConnectionError
    THEN POST /query SHALL return HTTP 200 with degraded=true and RESPOSTA_PADRAO_SEGURA."""
    patches = _client_ctx(
        invoke_side_effect=APIConnectionError.__new__(APIConnectionError)
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "teste"})

    assert response.status_code == 200, (
        f"Expected 200 (safe mode), got {response.status_code}. "
        "APIConnectionError mid-call should trigger safe mode."
    )
    data = response.json()
    assert data["degraded"] is True
    assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA


# ---------------------------------------------------------------------------
# AC-05 — Mid-call httpx.ConnectError → 200-degraded
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_05_midcall_httpx_connect_error_returns_200_degraded():
    """AC-05: WHEN pre-flight passes but agent_workflow.invoke raises httpx.ConnectError
    THEN POST /query SHALL return HTTP 200 with degraded=true and RESPOSTA_PADRAO_SEGURA."""
    patches = _client_ctx(
        invoke_side_effect=httpx.ConnectError("Connection refused")
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "teste"})

    assert response.status_code == 200, (
        f"Expected 200 (safe mode), got {response.status_code}. "
        "httpx.ConnectError mid-call should trigger safe mode."
    )
    data = response.json()
    assert data["degraded"] is True
    assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA


# ---------------------------------------------------------------------------
# AC-06 — FR-5 invariants: all three safe-mode fields have exact constant values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", [
    pytest.param(
        {"health_side_effect": LMStudioUnavailableError("down")},
        id="lmstudio_unavailable",
    ),
    pytest.param(
        {"health_side_effect": ModeloNaoCarregadoError("no model")},
        id="modelo_nao_carregado",
    ),
    pytest.param(
        {"invoke_side_effect": httpx.ConnectError("refused")},
        id="connect_error_midcall",
    ),
    pytest.param(
        {"invoke_side_effect": APIConnectionError.__new__(APIConnectionError)},
        id="api_connection_error_midcall",
    ),
])
def test_ac_modo_seguro_06_fr5_safe_mode_invariants(trigger):
    """AC-06 (FR-5): WHEN safe mode is triggered by any covered error THEN POST /query
    SHALL return ranked_response==RESPOSTA_PADRAO_SEGURA, confidence_score==0.0,
    retrieved_info==[], next_step=="", and degraded==True — all simultaneously."""
    reset_health_check_sentinel()
    try:
        patches = _client_ctx(**trigger)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "invariant check"})
    finally:
        reset_health_check_sentinel()

    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is True, "degraded must be True in safe mode"
    assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA, (
        f"ranked_response must equal RESPOSTA_PADRAO_SEGURA. Got: {data['ranked_response']!r}"
    )
    assert data["confidence_score"] == 0.0, (
        f"confidence_score must be 0.0 in safe mode. Got: {data['confidence_score']}"
    )
    assert data["retrieved_info"] == [], (
        f"retrieved_info must be [] in safe mode. Got: {data['retrieved_info']}"
    )
    assert data["next_step"] == "", (
        f"next_step must be '' in safe mode. Got: {data['next_step']!r}"
    )


# ---------------------------------------------------------------------------
# AC-07 — FR-6: degraded field always present in normal (non-safe) response
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_07_fr6_degraded_always_present_and_false_on_happy_path():
    """AC-07 (FR-6): WHEN LLM is healthy THEN POST /query 200 response SHALL always
    include the 'degraded' key set to false — not absent, not null."""
    patches = _client_ctx()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo normal"})

    assert response.status_code == 200
    data = response.json()
    assert "degraded" in data, "Response must always include 'degraded' field (FR-6)"
    assert data["degraded"] is False, (
        f"'degraded' must be False on the happy path. Got: {data['degraded']}"
    )


# ---------------------------------------------------------------------------
# AC-08 — FR-7: degraded response is written to audit history
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_08_fr7_degraded_response_written_to_audit_history():
    """AC-08 (FR-7): WHEN safe mode is activated THEN the degraded response SHALL
    be persisted to audit history (history_store.append called exactly once)."""
    _mock_history_store.reset_mock()
    reset_health_check_sentinel()
    try:
        patches = _client_ctx(
            health_side_effect=LMStudioUnavailableError("server down")
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "historia degraded"})
    finally:
        reset_health_check_sentinel()

    assert response.status_code == 200
    assert response.json()["degraded"] is True

    assert _mock_history_store.append.call_count == 1, (
        "history_store.append must be called once even for degraded responses (FR-7). "
        f"Called {_mock_history_store.append.call_count} times."
    )

    # The recorded entry must contain the safe-mode response text
    call_args = _mock_history_store.append.call_args
    recorded = call_args[0][0]  # positional first arg is the registro dict
    assert recorded["ranked_response"] == RESPOSTA_PADRAO_SEGURA, (
        f"Audit record must store RESPOSTA_PADRAO_SEGURA. Got: {recorded['ranked_response']!r}"
    )
    assert recorded["confidence_score"] == 0.0
    assert "query" in recorded
    assert recorded["query"] == "historia degraded"


# ---------------------------------------------------------------------------
# AC-09 — FR-9 UI: degraded=true → "modo seguro" badge rendered
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_09_fr9_ui_degraded_true_renders_modo_seguro_badge():
    """AC-09 (FR-9): WHEN the API returns degraded=true THEN the Streamlit UI SHALL
    render a visible 'modo seguro' badge in the response area."""
    from streamlit.testing.v1 import AppTest

    app_path = str(_root / "app.py")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ranked_response": RESPOSTA_PADRAO_SEGURA,
        "confidence_score": 0.0,
        "next_step": "",
        "retrieved_info": [],
        "degraded": True,
    }
    mock_response.raise_for_status.return_value = None

    with patch("app.httpx.post", return_value=mock_response):
        at = AppTest.from_file(app_path)
        at.run()
        at.text_input[0].set_value("pergunta com modelo indisponivel").run()

    assert not at.exception, f"Unexpected exception in UI: {at.exception}"
    markdown_texts = [m.value for m in at.markdown]
    assert any("modo seguro" in t for t in markdown_texts), (
        f"Badge 'modo seguro' not found in rendered markdown. "
        f"Texts rendered: {markdown_texts}"
    )


# ---------------------------------------------------------------------------
# AC-10 — FR-9 UI: degraded=false/absent → no badge
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_10_fr9_ui_degraded_false_no_modo_seguro_badge():
    """AC-10 (FR-9): WHEN the API returns degraded=false THEN the Streamlit UI SHALL NOT
    render a 'modo seguro' badge — normal answer renders without it."""
    from streamlit.testing.v1 import AppTest

    app_path = str(_root / "app.py")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ranked_response": "Prazo de entrega é 2 dias úteis.",
        "confidence_score": 0.85,
        "next_step": "retrieve",
        "retrieved_info": [],
        "degraded": False,
    }
    mock_response.raise_for_status.return_value = None

    with patch("app.httpx.post", return_value=mock_response):
        at = AppTest.from_file(app_path)
        at.run()
        at.text_input[0].set_value("pergunta normal").run()

    assert not at.exception, f"Unexpected exception in UI: {at.exception}"
    markdown_texts = [m.value for m in at.markdown]
    assert not any("modo seguro" in t for t in markdown_texts), (
        f"Badge 'modo seguro' unexpectedly rendered for degraded=false. "
        f"Texts: {markdown_texts}"
    )


# ---------------------------------------------------------------------------
# AC-11 — Vectordb missing still returns 503 (not swallowed by safe mode)
# ---------------------------------------------------------------------------


def test_ac_modo_seguro_11_vectordb_missing_still_returns_503():
    """AC-11: WHEN vectordb is missing THEN POST /query SHALL return HTTP 503 regardless
    of LLM availability — safe mode does not intercept vectordb errors."""
    patches = _client_ctx(vectordb_ok=False)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "teste vectordb ausente"})

    assert response.status_code == 503, (
        f"Missing vectordb must still return 503, not trigger safe mode. "
        f"Got: {response.status_code}"
    )
    assert MSG_VECTORDB_AUSENTE in response.json().get("detail", "")
