"""
Testes unitários para src/agenticlog/history.py.

Todos os testes usam tempfile.TemporaryDirectory() para isolamento.
Nenhum teste toca data/history/.
"""

import datetime
import json
import tempfile
import threading
import unittest
from pathlib import Path

from agenticlog.observability.history import HistoryStore


class TestHistoryStore(unittest.TestCase):
    """Testes unitários do HistoryStore."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "test.db"
        self.store = HistoryStore(db_path=db_path, max_entries=100)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _registro(self, query: str = "test query", ts: str | None = None) -> dict:
        return {
            "timestamp": ts or datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "query": query,
            "next_step": "retrieve",
            "confidence_score": 0.9,
            "ranked_response": "resposta teste",
        }

    # ------------------------------------------------------------------
    # teste_1: append e read_all básico
    # ------------------------------------------------------------------

    def teste_1_append_e_read(self) -> None:
        """append() grava um registro; read_all() retorna lista com 1 dict correto."""
        reg = self._registro(query="prazo SP")
        self.store.append(reg)

        resultado = self.store.read_all()

        self.assertEqual(len(resultado), 1)
        row = resultado[0]
        self.assertIn("id", row)
        self.assertEqual(row["query"], "prazo SP")
        self.assertEqual(row["next_step"], "retrieve")
        self.assertAlmostEqual(row["confidence_score"], 0.9)
        self.assertEqual(row["ranked_response"], "resposta teste")

    # ------------------------------------------------------------------
    # teste_2: read_all vazio
    # ------------------------------------------------------------------

    def teste_2_read_all_vazio(self) -> None:
        """Store recém-criado; read_all() retorna []."""
        self.assertEqual(self.store.read_all(), [])

    # ------------------------------------------------------------------
    # teste_3: read_all com limit
    # ------------------------------------------------------------------

    def teste_3_read_all_com_limit(self) -> None:
        """Inserindo 5 registros com timestamps distintos; read_all(limit=2) retorna 2 mais recentes."""
        base = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        for i in range(5):
            ts = (base + datetime.timedelta(seconds=i)).isoformat()
            self.store.append(self._registro(query=f"query_{i}", ts=ts))

        resultado = self.store.read_all(limit=2)

        self.assertEqual(len(resultado), 2)
        # Ordenados DESC — o mais recente vem primeiro
        self.assertEqual(resultado[0]["query"], "query_4")
        self.assertEqual(resultado[1]["query"], "query_3")

    # ------------------------------------------------------------------
    # teste_4: limit maior que total
    # ------------------------------------------------------------------

    def teste_4_limit_maior_que_total(self) -> None:
        """Insert 3; read_all(limit=100) retorna 3 sem erro."""
        for i in range(3):
            self.store.append(self._registro(query=f"q{i}"))

        resultado = self.store.read_all(limit=100)

        self.assertEqual(len(resultado), 3)

    # ------------------------------------------------------------------
    # teste_5: evicção de max_entries
    # ------------------------------------------------------------------

    def teste_5_evicao_max_entries(self) -> None:
        """Store com max_entries=3; após 4 inserts, len(read_all())==3 e o mais antigo foi removido."""
        tmpdir2 = tempfile.TemporaryDirectory()
        try:
            store = HistoryStore(db_path=Path(tmpdir2.name) / "evict.db", max_entries=3)
            base = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
            for i in range(4):
                ts = (base + datetime.timedelta(seconds=i)).isoformat()
                store.append(self._registro(query=f"q{i}", ts=ts))

            resultado = store.read_all()

            self.assertEqual(len(resultado), 3)
            queries = [r["query"] for r in resultado]
            self.assertNotIn("q0", queries)  # mais antigo evictado
            self.assertIn("q3", queries)
        finally:
            tmpdir2.cleanup()

    # ------------------------------------------------------------------
    # teste_6: confidence_score None normalizado pelo chamador como 0.0
    # ------------------------------------------------------------------

    def teste_6_confidence_score_none_normalizado(self) -> None:
        """Chamador passa 0.0 para confidence_score=None; valor armazenado é 0.0."""
        reg = self._registro()
        reg["confidence_score"] = 0.0
        self.store.append(reg)

        resultado = self.store.read_all()

        self.assertEqual(resultado[0]["confidence_score"], 0.0)

    # ------------------------------------------------------------------
    # teste_7: ranked_response dict normalizado pelo chamador
    # ------------------------------------------------------------------

    def teste_7_ranked_response_dict_normalizado(self) -> None:
        """Chamador passa json.dumps(dict) como ranked_response; string armazenada corretamente."""
        esperado = json.dumps({"answer": "texto"})
        reg = self._registro()
        reg["ranked_response"] = esperado
        self.store.append(reg)

        resultado = self.store.read_all()

        self.assertEqual(resultado[0]["ranked_response"], esperado)

    # ------------------------------------------------------------------
    # teste_8: writes concorrentes
    # ------------------------------------------------------------------

    def teste_8_writes_concorrentes(self) -> None:
        """10 threads appending simultaneamente; todos os 10 registros são persistidos."""
        threads = [
            threading.Thread(target=self.store.append, args=(self._registro(query=f"q{i}"),))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        resultado = self.store.read_all()
        self.assertEqual(len(resultado), 10)

    # ------------------------------------------------------------------
    # teste_9: cria DB no primeiro write se não existir
    # ------------------------------------------------------------------

    def teste_9_cria_db_no_primeiro_write(self) -> None:
        """Store apontando para subdir inexistente cria diretório e arquivo no __init__ (via init_db)."""
        tmpdir3 = tempfile.TemporaryDirectory()
        try:
            db_path = Path(tmpdir3.name) / "subdir" / "novo.db"
            # Confirma que o arquivo ainda não existe antes de criar o store
            self.assertFalse(db_path.exists())
            HistoryStore(db_path=db_path, max_entries=10)
            # Após __init__, o arquivo SQLite deve ter sido criado por init_db()
            self.assertTrue(db_path.exists())
        finally:
            tmpdir3.cleanup()

    # ------------------------------------------------------------------
    # test_init_db_idempotente
    # ------------------------------------------------------------------

    def test_init_db_idempotente(self) -> None:
        """Chamar init_db() duas vezes não levanta exceção; tabela existe após ambas as chamadas."""
        # Primeira chamada já foi feita em setUp via __init__
        # Segunda chamada deve ser idempotente
        self.store.init_db()

        # Tabela deve existir e ser utilizável
        self.store.append(self._registro())
        resultado = self.store.read_all()
        self.assertEqual(len(resultado), 1)


if __name__ == "__main__":
    unittest.main()
