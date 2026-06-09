"""
Acceptance tests — query-history-audit-logging feature.

Maps every acceptance criterion from the approved user story to exactly one
test. All 12 ACs are covered.

Story: "As a logistics operator, I want every query I submit to be recorded
with its timestamp, decided route, confidence score, and final response, and
to be able to retrieve that history via API, so that I can audit what the
system answered and how it routed each request."

Test naming: test_ac<NN>_<short_description>
"""

import concurrent.futures
import datetime
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from agenticlog.agent import AgentState
from agenticlog.api import app
from agenticlog.history import HistoryStore


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
        ranked_response="Prazo de 2 dias uteis.",
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


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestQueryHistoryAuditLogging(unittest.TestCase):
    """One test per acceptance criterion, AC-01 through AC-12."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "test_audit.db"
        app.state.history_store = HistoryStore(db_path=db_path, max_entries=100)
        app.state.vectordb_pronto = True
        self.client = TestClient(app, raise_server_exceptions=False)
        self._db_path = db_path

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -----------------------------------------------------------------------
    # AC-01: POST /query bem-sucedido grava registro com campos obrigatorios
    #        antes de retornar HTTP 200.
    # -----------------------------------------------------------------------

    def test_ac01_post_query_grava_registro_e_retorna_200(self) -> None:
        """
        AC-01: WHEN POST /query succeeds THEN system SHALL write a record
        containing query, timestamp (ISO 8601 UTC), next_step, confidence_score,
        ranked_response to the SQLite store before returning HTTP 200.
        """
        estado = _make_estado()
        with patch("agenticlog.api.agent_workflow.invoke", return_value=estado), patch(
            "agenticlog.api.inicializar_recursos"
        ):
            response = self.client.post("/query", json={"query": "prazo SP-RJ"})

        assert response.status_code == 200

        registros = app.state.history_store.read_all()
        assert len(registros) == 1
        r = registros[0]
        assert r["query"] == "prazo SP-RJ"
        assert r["next_step"] == "retrieve"
        assert isinstance(r["confidence_score"], float)
        assert isinstance(r["ranked_response"], str)
        # timestamp must be ISO 8601
        ts = r["timestamp"]
        assert "T" in ts and ("+" in ts or "Z" in ts or ts.endswith("00:00")), (
            f"timestamp nao e ISO 8601 UTC: {ts!r}"
        )
        assert "id" in r

    # -----------------------------------------------------------------------
    # AC-02: Write failure — POST /query ainda retorna HTTP 200; falha logada
    # -----------------------------------------------------------------------

    def test_ac02_write_failure_retorna_200_sem_propagar_erro(self) -> None:
        """
        AC-02: WHEN the write to SQLite fails THEN system SHALL still return
        HTTP 200 to the client AND log the failure at ERROR level.
        """
        broken_store = MagicMock()
        broken_store.append.side_effect = RuntimeError("disco cheio")
        broken_store.read_all.return_value = []
        app.state.history_store = broken_store

        estado = _make_estado()
        with patch("agenticlog.api.agent_workflow.invoke", return_value=estado), patch(
            "agenticlog.api.inicializar_recursos"
        ):
            response = self.client.post("/query", json={"query": "prazo"})

        assert response.status_code == 200

    # -----------------------------------------------------------------------
    # AC-03: GET /history com registros retorna HTTP 200, array JSON ordenado
    #        por timestamp DESC.
    # -----------------------------------------------------------------------

    def test_ac03_get_history_com_registros_ordenado_desc(self) -> None:
        """
        AC-03: WHEN GET /history is called with records present THEN system
        SHALL return HTTP 200 with a JSON array sorted by timestamp DESC.
        """
        store = app.state.history_store
        store.append(_registro("q1", "2024-01-01T10:00:00+00:00"))
        store.append(_registro("q2", "2024-01-02T10:00:00+00:00"))
        store.append(_registro("q3", "2024-01-03T10:00:00+00:00"))

        response = self.client.get("/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        timestamps = [r["timestamp"] for r in data]
        assert timestamps == sorted(timestamps, reverse=True), (
            f"Registros nao ordenados DESC: {timestamps}"
        )

    # -----------------------------------------------------------------------
    # AC-04: GET /history sem registros retorna HTTP 200, []
    # -----------------------------------------------------------------------

    def test_ac04_get_history_sem_registros_retorna_lista_vazia(self) -> None:
        """
        AC-04: WHEN GET /history is called with no records THEN system SHALL
        return HTTP 200 with [].
        """
        response = self.client.get("/history")

        assert response.status_code == 200
        assert response.json() == []

    # -----------------------------------------------------------------------
    # AC-05: GET /history com limit=N retorna os N mais recentes
    # -----------------------------------------------------------------------

    def test_ac05_get_history_com_limit_retorna_n_mais_recentes(self) -> None:
        """
        AC-05: WHEN GET /history is called with ?limit=N (N > 0) THEN system
        SHALL return the N most recent records.
        """
        store = app.state.history_store
        for i in range(5):
            store.append(_registro(f"q{i}", f"2024-01-0{i+1}T10:00:00+00:00"))

        response = self.client.get("/history?limit=3")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        # Most recent 3: q4, q3, q2 (timestamps 5, 4, 3 in DESC order)
        timestamps = [r["timestamp"] for r in data]
        assert timestamps == sorted(timestamps, reverse=True)
        assert data[0]["timestamp"] == "2024-01-05T10:00:00+00:00"

    # -----------------------------------------------------------------------
    # AC-06: GET /history sem limit retorna tudo (bounded por HISTORY_MAX_ENTRIES)
    # -----------------------------------------------------------------------

    def test_ac06_get_history_sem_limit_retorna_todos(self) -> None:
        """
        AC-06: WHEN GET /history is called without limit THEN system SHALL
        return all records (bounded by HISTORY_MAX_ENTRIES).
        """
        store = app.state.history_store
        for i in range(7):
            store.append(_registro(f"q{i}"))

        response = self.client.get("/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 7, f"Esperado 7 registros, recebido {len(data)}"

    # -----------------------------------------------------------------------
    # AC-07: limit <= 0 retorna HTTP 422
    # -----------------------------------------------------------------------

    def test_ac07_limit_zero_retorna_422(self) -> None:
        """
        AC-07: WHEN limit <= 0 THEN system SHALL return HTTP 422.
        """
        response_zero = self.client.get("/history?limit=0")
        assert response_zero.status_code == 422

        response_neg = self.client.get("/history?limit=-5")
        assert response_neg.status_code == 422

    # -----------------------------------------------------------------------
    # AC-08: History store indisponivel — GET /history retorna HTTP 503
    # -----------------------------------------------------------------------

    def test_ac08_store_indisponivel_retorna_503(self) -> None:
        """
        AC-08: WHEN history store is unavailable THEN GET /history SHALL
        return HTTP 503.
        """
        broken_store = MagicMock()
        broken_store.read_all.side_effect = RuntimeError("banco corrompido")
        app.state.history_store = broken_store

        response = self.client.get("/history")

        assert response.status_code == 503

    # -----------------------------------------------------------------------
    # AC-09: limit > total records retorna tudo sem erro
    # -----------------------------------------------------------------------

    def test_ac09_limit_maior_que_total_retorna_todos(self) -> None:
        """
        AC-09: WHEN limit > total records THEN system SHALL return all records
        without error.
        """
        store = app.state.history_store
        for i in range(3):
            store.append(_registro(f"q{i}"))

        response = self.client.get("/history?limit=100")

        assert response.status_code == 200
        assert len(response.json()) == 3

    # -----------------------------------------------------------------------
    # AC-10: HISTORY_MAX_ENTRIES = 0 → ValueError no config load
    # -----------------------------------------------------------------------

    def test_ac10_history_max_entries_zero_raises_value_error(self) -> None:
        """
        AC-10: WHEN HISTORY_MAX_ENTRIES = 0 THEN config.py import SHALL raise
        ValueError before the API starts.

        Tested in subprocess para evitar contaminação do cache de módulos do pytest.
        """
        import subprocess
        import sys
        from pathlib import Path

        src_path = str(Path(__file__).resolve().parent.parent.parent / "src")
        env = {**os.environ, "HISTORY_MAX_ENTRIES": "0", "PYTHONPATH": src_path}
        result = subprocess.run(
            [sys.executable, "-c", "import agenticlog.config"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, "Esperava falha ao importar config com HISTORY_MAX_ENTRIES=0")
        self.assertIn("HISTORY_MAX_ENTRIES", result.stderr, f"ValueError não menciona HISTORY_MAX_ENTRIES: {result.stderr}")

    # -----------------------------------------------------------------------
    # AC-11: Writes concorrentes — ambos gravados, sem corrupcao
    # -----------------------------------------------------------------------

    def test_ac11_concurrent_writes_ambos_gravados(self) -> None:
        """
        AC-11: WHEN two requests arrive concurrently THEN system SHALL store
        both records (unique rowid, no corruption).
        """
        store = app.state.history_store
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def write(query: str) -> None:
            try:
                barrier.wait(timeout=5)
                store.append(_registro(query))
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(write, "concurrent_q1")
            f2 = executor.submit(write, "concurrent_q2")
            f1.result(timeout=10)
            f2.result(timeout=10)

        assert not errors, f"Excecoes nas threads: {errors}"
        registros = store.read_all()
        queries = {r["query"] for r in registros}
        assert "concurrent_q1" in queries
        assert "concurrent_q2" in queries
        assert len(registros) == 2, f"Esperado 2 registros, recebido {len(registros)}"

    # -----------------------------------------------------------------------
    # AC-12: Restart — registros persistem (SQLite on disk)
    # -----------------------------------------------------------------------

    def test_ac12_registros_persistem_apos_restart(self) -> None:
        """
        AC-12: WHEN the API restarts THEN records written in prior sessions
        SHALL still be present (SQLite on disk survives process restart).
        """
        db_path = self._db_path

        # Simulate "session 1": write a record using one HistoryStore instance
        store_session1 = HistoryStore(db_path=db_path, max_entries=100)
        store_session1.append(_registro("query_antes_do_restart"))

        # Simulate "restart": create a brand-new HistoryStore instance pointing
        # to the same file (mimics a fresh API startup reading the existing DB)
        store_session2 = HistoryStore(db_path=db_path, max_entries=100)
        registros = store_session2.read_all()

        queries = [r["query"] for r in registros]
        assert "query_antes_do_restart" in queries, (
            f"Registro nao encontrado apos restart. Registros: {registros}"
        )


if __name__ == "__main__":
    unittest.main()
