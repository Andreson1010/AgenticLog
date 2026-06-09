"""
Módulo de persistência do histórico de queries do AgenticLog.

Fornece HistoryStore: classe que encapsula todo acesso ao SQLite para o
audit log de queries. Sem dependências de FastAPI ou LangGraph.
"""

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DDL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS query_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    next_step        TEXT    NOT NULL,
    confidence_score REAL    NOT NULL,
    ranked_response  TEXT    NOT NULL
);
"""

_SQL_COUNT = "SELECT COUNT(*) FROM query_history"
_SQL_EVICT = "DELETE FROM query_history WHERE id = (SELECT MIN(id) FROM query_history)"
_SQL_INSERT = (
    "INSERT INTO query_history (timestamp, query, next_step, confidence_score, ranked_response) "
    "VALUES (?, ?, ?, ?, ?)"
)
_SQL_SELECT_ALL = (
    "SELECT id, timestamp, query, next_step, confidence_score, ranked_response "
    "FROM query_history ORDER BY timestamp DESC"
)
_SQL_SELECT_LIMIT = _SQL_SELECT_ALL + " LIMIT ?"


class HistoryStore:
    """Armazenamento SQLite do histórico de queries para auditoria.

    Thread-safe para escritas concorrentes via threading.Lock.
    Leituras não adquirem lock (SQLite serialized mode é suficiente nesta escala).
    """

    def __init__(self, db_path: Path, max_entries: int) -> None:
        """Inicializa o store: cria diretório, tabela e lock de escritas.

        Entrada: db_path — caminho completo do arquivo SQLite.
                 max_entries — número máximo de registros antes de evictar o mais antigo.
        Saída: nenhuma.
        Raises: qualquer exceção de I/O ou sqlite3 propaga para o chamador.
        """
        self._db_path = db_path
        self._max_entries = max_entries
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self) -> None:
        """Cria a tabela query_history se ainda não existir (DDL idempotente).

        Saída: nenhuma.
        Raises: sqlite3.Error se o arquivo não puder ser criado ou acessado.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(_DDL_CREATE_TABLE)
            conn.commit()
        finally:
            conn.close()

    def append(self, registro: dict) -> None:
        """Persiste um registro de query no histórico, evictando o mais antigo se necessário.

        Entrada: registro — dict com chaves: timestamp, query, next_step,
                 confidence_score, ranked_response.
        Saída: nenhuma.
        Raises: sqlite3.Error ou KeyError propagam para o chamador (api.py captura).
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                count = conn.execute(_SQL_COUNT).fetchone()[0]
                if count >= self._max_entries:
                    conn.execute(_SQL_EVICT)
                conn.execute(
                    _SQL_INSERT,
                    (
                        registro["timestamp"],
                        registro["query"],
                        registro["next_step"],
                        registro["confidence_score"],
                        registro["ranked_response"],
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def read_all(self, limit: int | None = None) -> list[dict]:
        """Retorna registros do histórico ordenados por timestamp DESC.

        Entrada: limit — número máximo de registros a retornar; None retorna todos.
        Saída: lista de dicts com chaves id, timestamp, query, next_step,
               confidence_score, ranked_response.
        Raises: sqlite3.Error propaga para o chamador (api.py captura).
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            if limit is not None:
                cursor = conn.execute(_SQL_SELECT_LIMIT, (limit,))
            else:
                cursor = conn.execute(_SQL_SELECT_ALL)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
