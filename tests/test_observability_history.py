# Unit tests — HistoryStore extraction to observability/history.py (OBS-12..17)
"""Cobre a extração de `HistoryStore` para `observability/history.py` (ADR-018 Fase 2).

Valida:
- OBS-12: import canônico `observability.history.HistoryStore`.
- OBS-14: re-export ergonômico `observability.HistoryStore`.
- OBS-15: identidade de objeto (canônico IS __init__ re-export).
- OBS-16: schema/eviction FIFO/ordenação DESC idênticos (round-trip append/read_all).
"""

import sys
from pathlib import Path

# Garante src/ no path (espelha a convenção do conftest.py).
_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.api as api_module
import agenticlog.history as history_shim
import agenticlog.observability as obs_pkg
import agenticlog.observability.history as obs_history
from agenticlog.observability.history import HistoryStore


def _registro(ts: str, query: str) -> dict:
    return {
        "timestamp": ts,
        "query": query,
        "next_step": "retrieve",
        "confidence_score": 0.9,
        "ranked_response": "resposta",
    }


def test_import_canonico_resolve_classe() -> None:
    """OBS-12: import canônico resolve a classe HistoryStore."""
    assert isinstance(HistoryStore, type)


def test_reexport_pacote_observability_e_mesmo_objeto() -> None:
    """OBS-14/OBS-15: `agenticlog.observability` re-exporta o MESMO objeto canônico."""
    assert obs_pkg.HistoryStore is obs_history.HistoryStore
    assert "HistoryStore" in obs_pkg.__all__


def test_shim_history_e_api_namespace_sao_mesmo_objeto() -> None:
    """OBS-13/OBS-15/OBS-17: shim `history` e consumidor `api` IS o objeto canônico."""
    assert history_shim.HistoryStore is obs_history.HistoryStore
    assert api_module.HistoryStore is obs_history.HistoryStore


def test_append_read_all_round_trip_e_ordenacao_desc(tmp_path: Path) -> None:
    """OBS-16: schema, colunas e ordenação DESC preservados no round-trip."""
    store = HistoryStore(tmp_path / "hist.db", max_entries=100)
    store.append(_registro("2026-01-01T00:00:00", "primeira"))
    store.append(_registro("2026-01-02T00:00:00", "segunda"))

    rows = store.read_all()
    assert [r["query"] for r in rows] == ["segunda", "primeira"]  # DESC por timestamp
    assert set(rows[0].keys()) == {
        "id", "timestamp", "query", "next_step", "confidence_score", "ranked_response",
    }


def test_eviction_fifo_no_max_entries(tmp_path: Path) -> None:
    """OBS-16: eviction FIFO do registro mais antigo ao atingir max_entries."""
    store = HistoryStore(tmp_path / "hist.db", max_entries=2)
    store.append(_registro("2026-01-01T00:00:00", "q1"))
    store.append(_registro("2026-01-02T00:00:00", "q2"))
    store.append(_registro("2026-01-03T00:00:00", "q3"))

    queries = [r["query"] for r in store.read_all()]
    assert queries == ["q3", "q2"]  # q1 evictado (mais antigo)
    assert len(queries) == 2


def test_read_all_com_limit(tmp_path: Path) -> None:
    """OBS-16: `limit` restringe o número de registros retornados."""
    store = HistoryStore(tmp_path / "hist.db", max_entries=100)
    store.append(_registro("2026-01-01T00:00:00", "q1"))
    store.append(_registro("2026-01-02T00:00:00", "q2"))

    rows = store.read_all(limit=1)
    assert len(rows) == 1
    assert rows[0]["query"] == "q2"
