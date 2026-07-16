"""
Testes de aceitação para o endpoint POST /query.

Um teste por critério de aceitação, AC-API-01 a AC-API-10.
Naming: test_ac_api_NN_description (conforme convenção do projeto).
LMStudio e ChromaDB são mockados — nenhuma chamada real é feita.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from agenticlog.config import RESPOSTA_PADRAO_SEGURA
from agenticlog.retrieval.state import AgentState
from agenticlog.serving.api import (
    MSG_VECTORDB_AUSENTE,
    app,
)
from agenticlog.serving.health import (
    LMStudioUnavailableError,
    reset_health_check_sentinel,
)

# Injeta mock do history_store antes de qualquer TestClient ser criado,
# para que o lifespan não tente criar o HistoryStore real em disco.
_mock_history_store = MagicMock()
_mock_history_store.append.return_value = None
_mock_history_store.read_all.return_value = []
app.state.history_store = _mock_history_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_estado(**kwargs) -> AgentState:
    defaults = dict(
        query="prazo SP-RJ",
        next_step="retrieve",
        retrieved_info=[],
        possible_responses=[],
        similarity_scores=[],
        ranked_response="Prazo de 2 dias úteis.",
        confidence_score=0.87,
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


@contextmanager
def _client(estado: AgentState | None = None, *, vectordb_ok: bool = True):
    """Cria TestClient com mocks configurados."""
    estado_ret = estado or _make_estado()
    verificar_side_effect = None if vectordb_ok else RuntimeError(MSG_VECTORDB_AUSENTE)

    with patch("agenticlog.serving.api.inicializar_recursos") as mock_init, patch(
        "agenticlog.serving.api._verificar_vectordb", side_effect=verificar_side_effect
    ), patch("agenticlog.serving.api.check_lmstudio_health"), patch(
        "agenticlog.serving.api.agent_workflow.invoke", return_value=estado_ret
    ) as mock_invoke, patch(
        "agenticlog.serving.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            yield client, mock_invoke, mock_init


# ---------------------------------------------------------------------------
# AC-API-01: Formato da resposta 200
# ---------------------------------------------------------------------------


def test_ac_api_01_response_shape():
    """AC-API-01: POST /query retorna HTTP 200 com ranked_response str,
    confidence_score float, next_step str, retrieved_info list."""
    with _client() as (client, _, __):
        response = client.post("/query", json={"query": "prazo SP-RJ"})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["ranked_response"], str)
    assert isinstance(data["confidence_score"], float)
    assert isinstance(data["next_step"], str)
    assert isinstance(data["retrieved_info"], list)


# ---------------------------------------------------------------------------
# AC-API-02: ranked_response dict → string
# ---------------------------------------------------------------------------


def test_ac_api_02_ranked_response_dict_normalization():
    """AC-API-02: ranked_response do tipo dict {"answer": "..."} é serializado como string."""
    # AgentState é estritamente tipado — usa MagicMock para simular estado com dict
    mock_estado = MagicMock()
    mock_estado.ranked_response = {"answer": "Dois dias úteis."}
    mock_estado.confidence_score = 0.87
    mock_estado.next_step = "retrieve"
    mock_estado.retrieved_info = []

    with patch("agenticlog.serving.api.inicializar_recursos"), patch(
        "agenticlog.serving.api._verificar_vectordb"
    ), patch("agenticlog.serving.api.check_lmstudio_health"), patch(
        "agenticlog.serving.api.agent_workflow.invoke", return_value=mock_estado
    ):
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo"})

    assert response.status_code == 200
    assert isinstance(response.json()["ranked_response"], str)
    assert response.json()["ranked_response"] == "Dois dias úteis."


# ---------------------------------------------------------------------------
# AC-API-03: retrieved_info — serialização de Document LangChain
# ---------------------------------------------------------------------------


def test_ac_api_03_retrieved_info_document_serialization():
    """AC-API-03: Objetos Document LangChain são serializados com apenas page_content e metadata."""
    mock_doc = MagicMock()
    mock_doc.page_content = "Rota SP-RJ: 2 dias úteis."
    mock_doc.metadata = {"source": "rotas.json"}
    estado = _make_estado(retrieved_info=[mock_doc])

    with _client(estado) as (client, _, __):
        response = client.post("/query", json={"query": "prazo"})

    assert response.status_code == 200
    docs = response.json()["retrieved_info"]
    assert len(docs) == 1
    assert set(docs[0].keys()) == {"page_content", "metadata"}
    assert docs[0]["page_content"] == "Rota SP-RJ: 2 dias úteis."
    assert docs[0]["metadata"] == {"source": "rotas.json"}


# ---------------------------------------------------------------------------
# AC-API-04: LMStudio indisponível → 200-degraded (modo seguro)
# ---------------------------------------------------------------------------


def test_ac_api_04_lmstudio_unavailable_200_degraded():
    """AC-API-04 (re-escopado): LMStudioUnavailableError (pre-flight) e httpx.ConnectError
    (mid-call) → HTTP 200 com modo seguro (degraded=True, RESPOSTA_PADRAO_SEGURA)."""
    reset_health_check_sentinel()
    try:
        # Pre-flight levanta LMStudioUnavailableError
        with patch("agenticlog.serving.api.inicializar_recursos"), patch(
            "agenticlog.serving.api._verificar_vectordb"
        ), patch(
            "agenticlog.serving.api.check_lmstudio_health",
            side_effect=LMStudioUnavailableError("LMStudio offline"),
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo"})

        assert response.status_code == 200
        data = response.json()
        assert data["degraded"] is True
        assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA
        assert data["confidence_score"] == 0.0
        assert data["retrieved_info"] == []

        # Mid-call: pre-flight passa, invoke re-levanta httpx.ConnectError
        with patch("agenticlog.serving.api.inicializar_recursos"), patch(
            "agenticlog.serving.api._verificar_vectordb"
        ), patch("agenticlog.serving.api.check_lmstudio_health"), patch(
            "agenticlog.serving.api.agent_workflow.invoke",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with TestClient(app) as client:
                response2 = client.post("/query", json={"query": "prazo"})

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["degraded"] is True
        assert data2["ranked_response"] == RESPOSTA_PADRAO_SEGURA
        assert data2["confidence_score"] == 0.0
        assert data2["retrieved_info"] == []
    finally:
        reset_health_check_sentinel()


# ---------------------------------------------------------------------------
# AC-API-05: Vectordb ausente → 503
# ---------------------------------------------------------------------------


def test_ac_api_05_vectordb_missing_503():
    """AC-API-05: Vectordb ausente no startup → HTTP 503 com mensagem de vectordb ausente."""
    with _client(vectordb_ok=False) as (client, _, __):
        response = client.post("/query", json={"query": "prazo"})

    assert response.status_code == 503
    assert response.json()["detail"] == MSG_VECTORDB_AUSENTE


# ---------------------------------------------------------------------------
# AC-API-06: Body malformado → 422
# ---------------------------------------------------------------------------


def test_ac_api_06_malformed_body_422():
    """AC-API-06: Body sem campo query retorna HTTP 422 (validação Pydantic)."""
    with _client() as (client, _, __):
        response = client.post("/query", json={"pergunta": "valor errado"})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# AC-API-07: Query vazia ou só espaços → 422
# ---------------------------------------------------------------------------


def test_ac_api_07_empty_query_422():
    """AC-API-07: query vazia '' ou só espaços '   ' retorna HTTP 422."""
    with _client() as (client, _, __):
        r1 = client.post("/query", json={"query": ""})
        r2 = client.post("/query", json={"query": "   "})

    assert r1.status_code == 422
    assert r2.status_code == 422


# ---------------------------------------------------------------------------
# AC-API-08: Exceção inesperada → 500 sem stack trace
# ---------------------------------------------------------------------------


def test_ac_api_08_unexpected_exception_500_no_stacktrace():
    """AC-API-08: RuntimeError inesperado retorna 500 sem expor stack trace no body."""
    with patch("agenticlog.serving.api.inicializar_recursos"), patch(
        "agenticlog.serving.api._verificar_vectordb"
    ), patch("agenticlog.serving.api.check_lmstudio_health"), patch(
        "agenticlog.serving.api.agent_workflow.invoke",
        side_effect=RuntimeError("erro interno secreto"),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/query", json={"query": "prazo"})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Erro interno do servidor."
    assert "erro interno secreto" not in response.text
    assert "Traceback" not in response.text
    assert "traceback" not in response.text.lower()


# ---------------------------------------------------------------------------
# AC-API-09: Workflow executa em threadpool
# ---------------------------------------------------------------------------


def test_ac_api_09_workflow_in_threadpool():
    """AC-API-09: asyncio.to_thread é chamado e agent_workflow.invoke executa no thread."""
    import asyncio as _asyncio

    estado = _make_estado()
    with patch("agenticlog.serving.api.inicializar_recursos"), patch(
        "agenticlog.serving.api._verificar_vectordb"
    ), patch("agenticlog.serving.api.check_lmstudio_health"), patch(
        "agenticlog.serving.api.agent_workflow.invoke", return_value=estado
    ) as mock_invoke, patch(
        "agenticlog.serving.api.asyncio.to_thread", wraps=_asyncio.to_thread
    ) as mock_to_thread:
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo"})

    assert response.status_code == 200
    # to_thread deve ter sido chamado exatamente uma vez
    assert mock_to_thread.call_count >= 1
    # agent_workflow.invoke deve ter sido chamado (via to_thread)
    assert mock_invoke.call_count == 1
    # O primeiro argumento posicional de to_thread deve ser callable
    first_arg = mock_to_thread.call_args[0][0]
    assert callable(first_arg)


# ---------------------------------------------------------------------------
# AC-API-10: Singletons inicializados no startup (lifespan)
# ---------------------------------------------------------------------------


def test_ac_api_10_singletons_initialized_at_startup():
    """AC-API-10: inicializar_recursos() é chamado exatamente uma vez durante o lifespan startup."""
    estado = _make_estado()
    with patch("agenticlog.serving.api.inicializar_recursos") as mock_init, patch(
        "agenticlog.serving.api._verificar_vectordb"
    ), patch("agenticlog.serving.api.check_lmstudio_health"), patch(
        "agenticlog.serving.api.agent_workflow.invoke", return_value=estado
    ):
        with TestClient(app) as client:
            # Faz duas requisições para confirmar que init não é chamado por req
            client.post("/query", json={"query": "prazo 1"})
            client.post("/query", json={"query": "prazo 2"})

    # inicializar_recursos deve ter sido chamado exatamente uma vez (no startup)
    assert mock_init.call_count == 1
