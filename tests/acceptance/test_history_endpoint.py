"""
Testes de aceitação para o endpoint GET /history.

Um teste por critério de aceitação, AC-HIST-01 a AC-HIST-13 (selecionados).
HistoryStore real com tempfile — nenhum toque em data/history/.
LMStudio e ChromaDB são mockados onde necessário.
"""

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from agenticlog.observability.history import HistoryStore
from agenticlog.retrieval.state import AgentState
from agenticlog.serving.api import app


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


def _registro(query: str = "test query", ts: str | None = None) -> dict:
    return {
        "timestamp": ts or datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "query": query,
        "next_step": "retrieve",
        "confidence_score": 0.9,
        "ranked_response": "resposta teste",
    }


class TestHistoryEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "test.db"
        app.state.history_store = HistoryStore(db_path=db_path, max_entries=100)
        app.state.vectordb_pronto = True
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -----------------------------------------------------------------------
    # AC-HIST-01: POST /query grava registro no histórico
    # -----------------------------------------------------------------------

    def test_ac_history_01_post_grava_registro(self) -> None:
        """HIST-01: POST /query bem-sucedido persiste 1 registro com campos corretos."""
        estado = _make_estado()
        with patch("agenticlog.serving.api.agent_workflow.invoke", return_value=estado), patch(
            "agenticlog.serving.api.check_lmstudio_health"
        ), patch("agenticlog.serving.api.inicializar_recursos"):
            response = self.client.post("/query", json={"query": "prazo SP-RJ"})

        assert response.status_code == 200
        registros = app.state.history_store.read_all()
        assert len(registros) == 1
        r = registros[0]
        assert r["query"] == "prazo SP-RJ"
        assert r["next_step"] == "retrieve"
        assert isinstance(r["confidence_score"], float)
        assert isinstance(r["ranked_response"], str)
        assert isinstance(r["timestamp"], str)
        assert "id" in r

    # -----------------------------------------------------------------------
    # AC-HIST-02: Falha de escrita não afeta resposta do cliente
    # -----------------------------------------------------------------------

    def test_ac_history_02_write_failure_nao_afeta_resposta(self) -> None:
        """HIST-02: Mesmo que append levante exceção, POST /query retorna 200."""
        broken_store = MagicMock()
        broken_store.append.side_effect = RuntimeError("disco cheio")
        broken_store.read_all.return_value = []
        app.state.history_store = broken_store

        estado = _make_estado()
        with patch("agenticlog.serving.api.agent_workflow.invoke", return_value=estado), patch(
            "agenticlog.serving.api.check_lmstudio_health"
        ), patch("agenticlog.serving.api.inicializar_recursos"):
            response = self.client.post("/query", json={"query": "prazo"})

        assert response.status_code == 200

    # -----------------------------------------------------------------------
    # AC-HIST-07: GET /history retorna registros ordenados DESC por timestamp
    # -----------------------------------------------------------------------

    def test_ac_history_07_get_retorna_registros_ordenados(self) -> None:
        """HIST-07: GET /history retorna lista ordenada por timestamp DESC."""
        store = app.state.history_store
        store.append(_registro("q1", "2024-01-01T10:00:00+00:00"))
        store.append(_registro("q2", "2024-01-02T10:00:00+00:00"))
        store.append(_registro("q3", "2024-01-03T10:00:00+00:00"))

        response = self.client.get("/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        timestamps = [r["timestamp"] for r in data]
        assert timestamps == sorted(timestamps, reverse=True)

    # -----------------------------------------------------------------------
    # AC-HIST-08: GET /history com store vazio retorna lista vazia
    # -----------------------------------------------------------------------

    def test_ac_history_08_get_sem_registros_retorna_lista_vazia(self) -> None:
        """HIST-08: Store vazio → GET /history retorna 200 com []."""
        response = self.client.get("/history")

        assert response.status_code == 200
        assert response.json() == []

    # -----------------------------------------------------------------------
    # AC-HIST-09: GET /history?limit=N respeita limite
    # -----------------------------------------------------------------------

    def test_ac_history_09_get_com_limit(self) -> None:
        """HIST-09: GET /history?limit=3 retorna exatamente 3 registros de 5."""
        store = app.state.history_store
        for i in range(5):
            store.append(_registro(f"q{i}", f"2024-01-0{i+1}T10:00:00+00:00"))

        response = self.client.get("/history?limit=3")

        assert response.status_code == 200
        assert len(response.json()) == 3

    # -----------------------------------------------------------------------
    # AC-HIST-11: GET /history?limit=0 retorna 422
    # -----------------------------------------------------------------------

    def test_ac_history_11_limit_zero_retorna_422(self) -> None:
        """HIST-11: limit=0 viola ge=1 → FastAPI retorna 422."""
        response = self.client.get("/history?limit=0")

        assert response.status_code == 422

    # -----------------------------------------------------------------------
    # AC-HIST-12: Store indisponível → 503
    # -----------------------------------------------------------------------

    def test_ac_history_12_store_indisponivel_retorna_503(self) -> None:
        """HIST-12: read_all levanta exceção → GET /history retorna 503."""
        broken_store = MagicMock()
        broken_store.read_all.side_effect = RuntimeError("banco corrompido")
        app.state.history_store = broken_store

        response = self.client.get("/history")

        assert response.status_code == 503

    # -----------------------------------------------------------------------
    # AC-HIST-13: limit maior que total retorna todos os registros
    # -----------------------------------------------------------------------

    def test_ac_history_13_limit_maior_que_total(self) -> None:
        """HIST-13: Insert 3; GET /history?limit=100 retorna os 3 sem erro."""
        store = app.state.history_store
        for i in range(3):
            store.append(_registro(f"q{i}"))

        response = self.client.get("/history?limit=100")

        assert response.status_code == 200
        assert len(response.json()) == 3


if __name__ == "__main__":
    unittest.main()
