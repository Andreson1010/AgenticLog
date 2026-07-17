# Unit tests — _JsonFormatter extraction to observability/logging.py (OBS-06..11, OBS-19)
"""Cobre a extração de `_JsonFormatter` para `observability/logging.py` (ADR-018 Fase 2).

Valida:
- OBS-06: import canônico `observability.logging._JsonFormatter`.
- OBS-08: re-export ergonômico `observability._JsonFormatter`.
- OBS-10: saída de `format()` idêntica (chaves timestamp/level/logger/message).
- OBS-11: `observability.logging` NÃO importa `agenticlog.config` (sem ciclo).
"""

import datetime
import inspect
import json
import logging
import sys
from pathlib import Path

# Garante src/ no path (espelha a convenção do conftest.py).
_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.observability as obs_pkg
import agenticlog.observability.logging as obs_logging
from agenticlog.observability.logging import _JsonFormatter


def test_import_canonico_resolve_classe() -> None:
    """OBS-06: import canônico resolve a classe de formatter."""
    assert isinstance(_JsonFormatter, type)
    assert issubclass(_JsonFormatter, logging.Formatter)


def test_reexport_pacote_observability_e_mesmo_objeto() -> None:
    """OBS-08/OBS-09: `agenticlog.observability` re-exporta o MESMO objeto canônico."""
    assert obs_pkg._JsonFormatter is obs_logging._JsonFormatter
    assert "_JsonFormatter" in obs_pkg.__all__


def test_logging_nao_importa_config() -> None:
    """OBS-11: `observability.logging` não importa `agenticlog.config` (aciclicidade)."""
    import_lines = [
        line.strip()
        for line in inspect.getsource(obs_logging).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert all("agenticlog" not in line for line in import_lines), import_lines
    assert not hasattr(obs_logging, "config")


def test_format_output_identico_para_log_record_fixo() -> None:
    """OBS-10: saída de format() é o JSON esperado com exatamente 4 chaves."""
    record = logging.LogRecord(
        name="agenticlog.teste",
        level=logging.WARNING,
        pathname=__file__,
        lineno=42,
        msg="mensagem %s",
        args=("parametrizada",),
        exc_info=None,
    )
    parsed = json.loads(_JsonFormatter().format(record))

    assert set(parsed.keys()) == {"timestamp", "level", "logger", "message"}
    assert parsed["level"] == record.levelname == "WARNING"
    assert parsed["logger"] == record.name == "agenticlog.teste"
    assert parsed["message"] == record.getMessage() == "mensagem parametrizada"
    esperado_ts = datetime.datetime.fromtimestamp(
        record.created, tz=datetime.UTC
    ).isoformat()
    assert parsed["timestamp"] == esperado_ts
