"""
Testes unitários para src/agenticlog/api.py.

Todos os mocks são aplicados no namespace agenticlog.api (consumer namespace),
não no namespace de definição (agenticlog.agent).
Nenhuma chamada real ao LMStudio ou ChromaDB é feita.
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from agenticlog.agent import AgentState
from agenticlog.api import (
    MSG_VECTORDB_AUSENTE,
    DocumentInfo,
    _normalizar_estado,
    _resposta_segura,
    _serializar_documentos,
    _verificar_vectordb,
    app,
)
from agenticlog.config import RESPOSTA_PADRAO_SEGURA
from agenticlog.health import (
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    reset_health_check_sentinel,
)

# Injeta mock do history_store antes de qualquer TestClient ser criado,
# para que o lifespan não tente criar o HistoryStore real em disco.
_mock_history_store = MagicMock()
_mock_history_store.append.return_value = None
_mock_history_store.read_all.return_value = []
app.state.history_store = _mock_history_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_estado(**kwargs) -> AgentState:
    """Cria um AgentState com valores padrão para testes."""
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
def _client_vectordb_pronto(estado: AgentState | None = None):
    """Cria TestClient com vectordb_pronto=True, pre-flight saudável e workflow mockado."""
    estado_retornado = estado or _make_estado()
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb"
    ), patch("agenticlog.api.check_lmstudio_health"), patch(
        "agenticlog.api.agent_workflow.invoke", return_value=estado_retornado
    ) as mock_invoke, patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            yield client, mock_invoke


@contextmanager
def _client_vectordb_ausente():
    """Cria TestClient simulando vectordb ausente na inicialização."""
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb",
        side_effect=RuntimeError(MSG_VECTORDB_AUSENTE),
    ), patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Testes do endpoint POST /query
# ---------------------------------------------------------------------------


def teste_1_query_valida_retorna_200():
    """POST /query com estado válido retorna HTTP 200 com todas as chaves esperadas."""
    with _client_vectordb_pronto() as (client, _):
        response = client.post("/query", json={"query": "prazo SP-RJ"})
    assert response.status_code == 200
    data = response.json()
    assert "ranked_response" in data
    assert "confidence_score" in data
    assert "next_step" in data
    assert "retrieved_info" in data
    assert data["degraded"] is False


def teste_2_ranked_response_dict_normalizado():
    """ranked_response do tipo dict {"answer": "x"} é normalizado para string "x"."""
    # AgentState é estritamente tipado — usa MagicMock para simular estado com dict
    mock_estado = MagicMock()
    mock_estado.ranked_response = {"answer": "Dois dias úteis."}
    mock_estado.confidence_score = 0.87
    mock_estado.next_step = "retrieve"
    mock_estado.retrieved_info = []
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb"
    ), patch("agenticlog.api.check_lmstudio_health"), patch(
        "agenticlog.api.agent_workflow.invoke", return_value=mock_estado
    ), patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 200
    assert response.json()["ranked_response"] == "Dois dias úteis."


def teste_3_confidence_score_none_normalizado():
    """confidence_score None é normalizado para 0.0 na resposta."""
    mock_estado = MagicMock()
    mock_estado.ranked_response = "Resposta qualquer."
    mock_estado.confidence_score = None
    mock_estado.next_step = "retrieve"
    mock_estado.retrieved_info = []
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb"
    ), patch("agenticlog.api.check_lmstudio_health"), patch(
        "agenticlog.api.agent_workflow.invoke", return_value=mock_estado
    ), patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 200
    assert response.json()["confidence_score"] == 0.0


def teste_4_retrieved_info_documentos_serializados():
    """Objetos Document LangChain são serializados com page_content e metadata."""
    mock_doc = MagicMock()
    mock_doc.page_content = "Rota SP-RJ: 2 dias."
    mock_doc.metadata = {"source": "rotas.json"}
    estado = _make_estado(retrieved_info=[mock_doc])
    with _client_vectordb_pronto(estado) as (client, _):
        response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 200
    docs = response.json()["retrieved_info"]
    assert len(docs) == 1
    assert docs[0]["page_content"] == "Rota SP-RJ: 2 dias."
    assert docs[0]["metadata"] == {"source": "rotas.json"}


def teste_5_retrieved_info_vazia_retorna_lista():
    """retrieved_info vazia retorna [] no JSON, nunca null."""
    estado = _make_estado(retrieved_info=[])
    with _client_vectordb_pronto(estado) as (client, _):
        response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 200
    assert response.json()["retrieved_info"] == []


def teste_6_query_vazia_retorna_422():
    """query vazia ('') causa erro de validação Pydantic → HTTP 422."""
    with _client_vectordb_pronto() as (client, _):
        response = client.post("/query", json={"query": ""})
    assert response.status_code == 422


def teste_7_query_espacos_retorna_422():
    """query com apenas espaços causa strip + min_length falha → HTTP 422."""
    with _client_vectordb_pronto() as (client, _):
        response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422


def teste_8_body_malformado_retorna_422():
    """Body sem campo query retorna HTTP 422 (Pydantic default)."""
    with _client_vectordb_pronto() as (client, _):
        response = client.post("/query", json={"pergunta": "prazo"})
    assert response.status_code == 422


def _assert_degradado(data: dict) -> None:
    """Valida os invariantes da resposta degradada (modo seguro)."""
    assert data["degraded"] is True
    assert data["ranked_response"] == RESPOSTA_PADRAO_SEGURA
    assert data["confidence_score"] == 0.0
    assert data["retrieved_info"] == []


def teste_9_pre_flight_lmstudio_indisponivel_retorna_200_degradado():
    """Pre-flight LMStudioUnavailableError → 200-degraded (não mais 503)."""
    reset_health_check_sentinel()
    try:
        with patch("agenticlog.api.inicializar_recursos"), patch(
            "agenticlog.api._verificar_vectordb"
        ), patch(
            "agenticlog.api.check_lmstudio_health",
            side_effect=LMStudioUnavailableError("LMStudio offline"),
        ), patch(
            "agenticlog.api.HistoryStore", return_value=_mock_history_store
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo"})
        assert response.status_code == 200
        _assert_degradado(response.json())
    finally:
        reset_health_check_sentinel()


def teste_10_mid_call_connect_error_retorna_200_degradado():
    """invoke re-levanta httpx.ConnectError após pre-flight ok → 200-degraded."""
    reset_health_check_sentinel()
    try:
        with patch("agenticlog.api.inicializar_recursos"), patch(
            "agenticlog.api._verificar_vectordb"
        ), patch("agenticlog.api.check_lmstudio_health"), patch(
            "agenticlog.api.agent_workflow.invoke",
            side_effect=httpx.ConnectError("Connection refused"),
        ), patch(
            "agenticlog.api.HistoryStore", return_value=_mock_history_store
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo"})
        assert response.status_code == 200
        _assert_degradado(response.json())
    finally:
        reset_health_check_sentinel()


def teste_10b_mid_call_api_connection_error_retorna_200_degradado():
    """invoke re-levanta openai.APIConnectionError após pre-flight ok → 200-degraded."""
    reset_health_check_sentinel()
    try:
        with patch("agenticlog.api.inicializar_recursos"), patch(
            "agenticlog.api._verificar_vectordb"
        ), patch("agenticlog.api.check_lmstudio_health"), patch(
            "agenticlog.api.agent_workflow.invoke",
            side_effect=APIConnectionError(request=httpx.Request("POST", "http://x")),
        ), patch(
            "agenticlog.api.HistoryStore", return_value=_mock_history_store
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo"})
        assert response.status_code == 200
        _assert_degradado(response.json())
    finally:
        reset_health_check_sentinel()


def teste_9b_pre_flight_modelo_nao_carregado_retorna_200_degradado():
    """Pre-flight ModeloNaoCarregadoError (modelo errado) → 200-degraded (caso novo)."""
    reset_health_check_sentinel()
    try:
        with patch("agenticlog.api.inicializar_recursos"), patch(
            "agenticlog.api._verificar_vectordb"
        ), patch(
            "agenticlog.api.check_lmstudio_health",
            side_effect=ModeloNaoCarregadoError("modelo X não carregado"),
        ), patch(
            "agenticlog.api.HistoryStore", return_value=_mock_history_store
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo"})
        assert response.status_code == 200
        _assert_degradado(response.json())
    finally:
        reset_health_check_sentinel()


def teste_9c_resposta_degradada_gravada_no_historico():
    """Resposta degradada é gravada no history_store com confidence_score=0.0 e a query."""
    reset_health_check_sentinel()
    store = MagicMock()
    store.append.return_value = None
    store.read_all.return_value = []
    try:
        with patch("agenticlog.api.inicializar_recursos"), patch(
            "agenticlog.api._verificar_vectordb"
        ), patch(
            "agenticlog.api.check_lmstudio_health",
            side_effect=LMStudioUnavailableError("offline"),
        ), patch(
            "agenticlog.api.HistoryStore", return_value=store
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo SP-RJ"})
        assert response.status_code == 200
        store.append.assert_called_once()
        registro = store.append.call_args[0][0]
        assert registro["query"] == "prazo SP-RJ"
        assert registro["confidence_score"] == 0.0
        assert registro["ranked_response"] == RESPOSTA_PADRAO_SEGURA
    finally:
        reset_health_check_sentinel()


def teste_9d_falha_historico_nao_quebra_resposta_degradada():
    """Falha na gravação do histórico não quebra a resposta 200-degraded."""
    reset_health_check_sentinel()
    store = MagicMock()
    store.append.side_effect = OSError("disco cheio")
    store.read_all.return_value = []
    try:
        with patch("agenticlog.api.inicializar_recursos"), patch(
            "agenticlog.api._verificar_vectordb"
        ), patch(
            "agenticlog.api.check_lmstudio_health",
            side_effect=LMStudioUnavailableError("offline"),
        ), patch(
            "agenticlog.api.HistoryStore", return_value=store
        ):
            with TestClient(app) as client:
                response = client.post("/query", json={"query": "prazo"})
        assert response.status_code == 200
        _assert_degradado(response.json())
    finally:
        reset_health_check_sentinel()


def teste_11_excecao_generica_retorna_500():
    """RuntimeError inesperado retorna HTTP 500 com mensagem genérica sem stack trace."""
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb"
    ), patch("agenticlog.api.check_lmstudio_health"), patch(
        "agenticlog.api.agent_workflow.invoke",
        side_effect=RuntimeError("boom interno"),
    ), patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 500
    body_text = response.text
    assert "boom interno" not in body_text
    assert "traceback" not in body_text.lower()
    assert response.json()["detail"] == "Erro interno do servidor."


def teste_12_vectordb_ausente_retorna_503():
    """Quando vectordb não existe na inicialização, todas as requisições retornam 503."""
    with _client_vectordb_ausente() as client:
        response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 503
    assert response.json()["detail"] == MSG_VECTORDB_AUSENTE


def teste_13_workflow_executa_em_thread():
    """agent_workflow.invoke é chamado via asyncio.to_thread (não chamada direta)."""
    estado = _make_estado()
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb"
    ), patch("agenticlog.api.check_lmstudio_health"), patch(
        "agenticlog.api.agent_workflow.invoke", return_value=estado
    ) as mock_invoke, patch(
        "agenticlog.api.asyncio.to_thread", wraps=asyncio.to_thread
    ) as mock_to_thread, patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 200
    # Verifica que asyncio.to_thread foi chamado pelo menos uma vez
    assert mock_to_thread.call_count >= 1
    # Verifica que agent_workflow.invoke (mock) foi realmente invocado pelo to_thread
    assert mock_invoke.call_count == 1


def teste_14_query_tipo_nao_string_retorna_422():
    """query com tipo nao-string (int) e rejeitada com HTTP 422."""
    with _client_vectordb_pronto() as (client, _):
        response = client.post("/query", json={"query": 123})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Testes do lifespan e de _verificar_vectordb
# ---------------------------------------------------------------------------


def teste_15_verificar_vectordb_diretorio_ausente_lanca_runtimeerror(tmp_path):
    """_verificar_vectordb levanta RuntimeError quando DIR_VECTORDB nao existe."""
    caminho_ausente = tmp_path / "vectordb_inexistente"
    with patch("agenticlog.api.DIR_VECTORDB", caminho_ausente):
        with pytest.raises(RuntimeError, match=MSG_VECTORDB_AUSENTE):
            _verificar_vectordb()


def teste_16_verificar_vectordb_diretorio_presente_abre_chroma(tmp_path):
    """_verificar_vectordb abre a colecao Chroma quando DIR_VECTORDB existe."""
    with patch("agenticlog.api.DIR_VECTORDB", tmp_path), patch(
        "agenticlog.api.Chroma"
    ) as mock_chroma, patch(
        "agenticlog.api._get_rag_embedding_model", return_value="embedding-mock"
    ):
        _verificar_vectordb()
    mock_chroma.assert_called_once()


def teste_17_lifespan_erro_generico_marca_vectordb_indisponivel():
    """Exception generica (nao RuntimeError) na inicializacao marca vectordb_pronto=False."""
    with patch(
        "agenticlog.api._verificar_vectordb", side_effect=ValueError("erro inesperado")
    ), patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api.HistoryStore", return_value=_mock_history_store
    ):
        with TestClient(app) as client:
            assert client.app.state.vectordb_pronto is False
            response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 503
    assert response.json()["detail"] == MSG_VECTORDB_AUSENTE


def teste_18_lifespan_historystore_falha_marca_none():
    """Falha ao construir HistoryStore marca history_store=None; /query continua ok."""
    estado = _make_estado()
    with patch("agenticlog.api.inicializar_recursos"), patch(
        "agenticlog.api._verificar_vectordb"
    ), patch("agenticlog.api.check_lmstudio_health"), patch(
        "agenticlog.api.agent_workflow.invoke", return_value=estado
    ), patch(
        "agenticlog.api.HistoryStore", side_effect=OSError("disco cheio")
    ):
        with TestClient(app) as client:
            assert client.app.state.history_store is None
            response = client.post("/query", json={"query": "prazo"})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Testes das funções puras de normalização
# ---------------------------------------------------------------------------


def teste_normalizar_estado_ranked_dict_com_answer():
    """_normalizar_estado extrai valor de "answer" quando ranked_response é dict."""
    estado = MagicMock()
    estado.ranked_response = {"answer": "Resposta correta."}
    estado.confidence_score = 0.5
    estado.next_step = "retrieve"
    estado.retrieved_info = []
    resultado = _normalizar_estado(estado)
    assert resultado.ranked_response == "Resposta correta."


def teste_normalizar_estado_ranked_dict_sem_answer():
    """_normalizar_estado faz json.dumps quando dict não contém "answer"."""
    estado = MagicMock()
    estado.ranked_response = {"outro": "valor"}
    estado.confidence_score = 0.5
    estado.next_step = "retrieve"
    estado.retrieved_info = []
    resultado = _normalizar_estado(estado)
    assert resultado.ranked_response == '{"outro": "valor"}'


def teste_resposta_segura_invariantes():
    """_resposta_segura retorna QueryResponse degradado com invariantes fixos."""
    resultado = _resposta_segura("qualquer query")
    assert resultado.ranked_response == RESPOSTA_PADRAO_SEGURA
    assert resultado.confidence_score == 0.0
    assert resultado.retrieved_info == []
    assert resultado.next_step == ""
    assert resultado.degraded is True


def teste_normalizar_estado_normal_degraded_false():
    """Caminho normal de _normalizar_estado produz degraded=False (default)."""
    resultado = _normalizar_estado(_make_estado())
    assert resultado.degraded is False


def teste_serializar_documentos_vazio():
    """_serializar_documentos([]) retorna []."""
    assert _serializar_documentos([]) == []


def teste_serializar_documentos_langchain_document():
    """_serializar_documentos aceita objetos com atributo page_content."""
    doc = MagicMock()
    doc.page_content = "conteúdo"
    doc.metadata = {"k": "v"}
    resultado = _serializar_documentos([doc])
    assert resultado == [DocumentInfo(page_content="conteúdo", metadata={"k": "v"})]


def teste_serializar_documentos_dict():
    """_serializar_documentos aceita dicts com chave page_content."""
    doc = {"page_content": "texto", "metadata": {"origem": "json"}}
    resultado = _serializar_documentos([doc])
    assert resultado == [DocumentInfo(page_content="texto", metadata={"origem": "json"})]


def teste_serializar_documentos_item_tipo_nao_suportado_ignorado():
    """_serializar_documentos ignora itens sem page_content e que nao sao dict."""
    resultado = _serializar_documentos(["string_qualquer", 123, None])
    assert resultado == []


def teste_normalizar_estado_dict_ranked_string():
    """_normalizar_estado aceita dict (LangGraph) com ranked_response string."""
    estado = {
        "ranked_response": "Resposta direta.",
        "confidence_score": 0.75,
        "next_step": "gerar",
        "retrieved_info": [],
    }
    resultado = _normalizar_estado(estado)
    assert resultado.ranked_response == "Resposta direta."
    assert resultado.confidence_score == 0.75
    assert resultado.next_step == "gerar"
    assert resultado.retrieved_info == []


def teste_normalizar_estado_dict_ranked_dict_com_answer():
    """_normalizar_estado (dict) extrai "answer" quando ranked_response e dict."""
    estado = {
        "ranked_response": {"answer": "Resposta extraida."},
        "confidence_score": 0.6,
        "next_step": "retrieve",
        "retrieved_info": [],
    }
    resultado = _normalizar_estado(estado)
    assert resultado.ranked_response == "Resposta extraida."


def teste_normalizar_estado_dict_valores_ausentes_usa_defaults():
    """_normalizar_estado (dict) usa defaults quando chaves opcionais estao ausentes."""
    estado = {"ranked_response": "Resposta minima."}
    resultado = _normalizar_estado(estado)
    assert resultado.ranked_response == "Resposta minima."
    assert resultado.confidence_score == 0.0
    assert resultado.next_step == ""
    assert resultado.retrieved_info == []
