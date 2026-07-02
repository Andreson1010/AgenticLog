# AgenticLog - Formatação estruturada de logs (observabilidade)
"""Formatter JSON para logging estruturado.

Somente stdlib (`json`, `logging`, `datetime`); NÃO importa `agenticlog.config`
para preservar a aciclicidade do grafo de imports (config -> observability.logging).
"""

import datetime
import json
import logging


class _JsonFormatter(logging.Formatter):
    """Serializa cada LogRecord como uma linha JSON com campos padronizados."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })
