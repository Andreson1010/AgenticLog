# Acceptance tests — structured-log-config feature
"""
Verifies all 8 acceptance criteria from the approved user story:
  AC-01 through AC-08 — LOG_LEVEL / LOG_FORMAT env-var control in config.py
  and JSON/text output in rag._executar_main().

Each test exercises the system through its public surface (config module constants
and the _executar_main() entry point) exactly as an operator would encounter them.
"""

import importlib
import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src/ is on path (mirrors conftest.py convention)
_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.config as config
import agenticlog.rag as rag_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_config(monkeypatch, log_level=None, log_format=None):
    """
    Set (or unset) LOG_LEVEL / LOG_FORMAT then reload config.
    Returns the freshly-loaded module.
    """
    if log_level is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", log_level)

    if log_format is None:
        monkeypatch.delenv("LOG_FORMAT", raising=False)
    else:
        monkeypatch.setenv("LOG_FORMAT", log_format)

    return importlib.reload(config)


def _reload_rag(monkeypatch, log_level=None, log_format=None):
    """Reload both config and rag so module-level imports reflect env changes."""
    _reload_config(monkeypatch, log_level=log_level, log_format=log_format)
    return importlib.reload(rag_module)


# ---------------------------------------------------------------------------
# AC-01: LOG_LEVEL not set → config.LOG_LEVEL == "INFO"
# ---------------------------------------------------------------------------

def test_ac01_log_level_defaults_to_info(monkeypatch):
    """
    AC-01: WHEN LOG_LEVEL env var is not set
    THEN config.LOG_LEVEL SHALL equal 'INFO'.
    """
    cfg = _reload_config(monkeypatch, log_level=None, log_format=None)
    assert cfg.LOG_LEVEL == "INFO", (
        f"Expected config.LOG_LEVEL='INFO' when LOG_LEVEL is unset, got {cfg.LOG_LEVEL!r}"
    )


# ---------------------------------------------------------------------------
# AC-02: LOG_LEVEL=DEBUG → config.LOG_LEVEL == "DEBUG"
# ---------------------------------------------------------------------------

def test_ac02_log_level_reads_from_env(monkeypatch):
    """
    AC-02: WHEN LOG_LEVEL=DEBUG is set in the environment
    THEN config.LOG_LEVEL SHALL equal 'DEBUG' at runtime.
    """
    cfg = _reload_config(monkeypatch, log_level="DEBUG", log_format=None)
    assert cfg.LOG_LEVEL == "DEBUG", (
        f"Expected config.LOG_LEVEL='DEBUG' after LOG_LEVEL=DEBUG, got {cfg.LOG_LEVEL!r}"
    )


# ---------------------------------------------------------------------------
# AC-03: LOG_FORMAT not set → config.LOG_FORMAT == "text"
# ---------------------------------------------------------------------------

def test_ac03_log_format_defaults_to_text(monkeypatch):
    """
    AC-03: WHEN LOG_FORMAT env var is not set
    THEN config.LOG_FORMAT SHALL equal 'text'.
    """
    cfg = _reload_config(monkeypatch, log_level=None, log_format=None)
    assert cfg.LOG_FORMAT == "text", (
        f"Expected config.LOG_FORMAT='text' when LOG_FORMAT is unset, got {cfg.LOG_FORMAT!r}"
    )


# ---------------------------------------------------------------------------
# AC-04: LOG_FORMAT=json → config.LOG_FORMAT == "json"
# ---------------------------------------------------------------------------

def test_ac04_log_format_reads_from_env(monkeypatch):
    """
    AC-04: WHEN LOG_FORMAT=json is set in the environment
    THEN config.LOG_FORMAT SHALL equal 'json' at runtime.
    """
    cfg = _reload_config(monkeypatch, log_level=None, log_format="json")
    assert cfg.LOG_FORMAT == "json", (
        f"Expected config.LOG_FORMAT='json' after LOG_FORMAT=json, got {cfg.LOG_FORMAT!r}"
    )


# ---------------------------------------------------------------------------
# AC-05: LOG_FORMAT=json + _executar_main → each log line is valid JSON
#         with fields: timestamp, level, logger, message
# ---------------------------------------------------------------------------

def test_ac05_json_format_produces_valid_json_lines(monkeypatch, capsys):
    """
    AC-05: WHEN LOG_FORMAT=json and rag._executar_main() runs
    THEN each log line SHALL be valid JSON containing timestamp, level,
    logger, and message fields.

    Uses capsys to capture stderr: _executar_main calls pkg_logger.handlers.clear()
    before installing its own StreamHandler, so any pre-installed capture handler
    is removed. capsys intercepts the real stderr writes instead.
    """
    from unittest.mock import patch as _patch

    _reload_rag(monkeypatch, log_level=None, log_format="json")

    import agenticlog.rag as rag  # noqa: PLC0415

    with _patch("agenticlog.rag._valida_path_documentos"), \
         _patch("agenticlog.rag._valida_arquivos_json"), \
         _patch("agenticlog.rag.DirectoryLoader") as mock_loader:
        mock_loader.return_value.load.return_value = []
        rag._executar_main()

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.strip()]

    assert lines, (
        "No log output captured from _executar_main() — "
        "expected at least one JSON log line on stderr"
    )

    required_fields = {"timestamp", "level", "logger", "message"}
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Log line is not valid JSON: {line!r} — {exc}")
        missing = required_fields - parsed.keys()
        assert not missing, (
            f"Log line missing required fields {missing}: {parsed}"
        )


# ---------------------------------------------------------------------------
# AC-06: LOG_FORMAT=text → plain text output (existing behavior preserved)
# ---------------------------------------------------------------------------

def test_ac06_text_format_does_not_use_json_formatter(monkeypatch):
    """
    AC-06: WHEN LOG_FORMAT=text (or unset)
    THEN log output SHALL be plain text; no _JsonFormatter SHALL be
    attached to the root logger after _executar_main() completes.
    """
    rag = _reload_rag(monkeypatch, log_level=None, log_format=None)

    assert rag.LOG_FORMAT == "text", (
        f"Pre-condition failed: expected LOG_FORMAT='text', got {rag.LOG_FORMAT!r}"
    )

    with patch.object(rag, "cria_vectordb", return_value=None):
        rag._executar_main()

    pkg_logger = logging.getLogger("agenticlog")
    json_handlers = [
        h for h in pkg_logger.handlers
        if isinstance(getattr(h, "formatter", None), rag._JsonFormatter)
    ]
    assert json_handlers == [], (
        f"Expected no _JsonFormatter on 'agenticlog' logger in text mode, found: {json_handlers}"
    )


# ---------------------------------------------------------------------------
# AC-07: Invalid LOG_LEVEL → ValueError at startup
# ---------------------------------------------------------------------------

def test_ac07_invalid_log_level_raises_value_error(monkeypatch):
    """
    AC-07: WHEN LOG_LEVEL is set to an unrecognised value (e.g. 'VERBOSE')
    THEN a ValueError SHALL be raised at module import time with no silent fallback.
    """
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        importlib.reload(config)


def test_ac07_error_message_names_the_bad_value(monkeypatch):
    """
    AC-07 (boundary): The ValueError message SHALL name the bad value,
    giving the operator a clear signal — not just a generic error.
    """
    monkeypatch.setenv("LOG_LEVEL", "TRACE")
    monkeypatch.delenv("LOG_FORMAT", raising=False)

    with pytest.raises(ValueError) as exc_info:
        importlib.reload(config)

    assert "TRACE" in str(exc_info.value), (
        f"Expected error message to name 'TRACE', got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# AC-08: Invalid LOG_FORMAT → ValueError at startup
# ---------------------------------------------------------------------------

def test_ac08_invalid_log_format_raises_value_error(monkeypatch):
    """
    AC-08: WHEN LOG_FORMAT is set to an unrecognised value (e.g. 'xml')
    THEN a ValueError SHALL be raised at module import time with no silent fallback.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("LOG_FORMAT", "xml")

    with pytest.raises(ValueError, match="Invalid LOG_FORMAT"):
        importlib.reload(config)


def test_ac08_error_message_names_the_bad_format(monkeypatch):
    """
    AC-08 (boundary): The ValueError message SHALL name the bad value.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("LOG_FORMAT", "yaml")

    with pytest.raises(ValueError) as exc_info:
        importlib.reload(config)

    assert "yaml" in str(exc_info.value).lower(), (
        f"Expected error message to name 'yaml', got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Edge cases from spec (not full ACs but required by story)
# ---------------------------------------------------------------------------

def test_edge_lowercase_log_level_accepted(monkeypatch):
    """
    Spec edge case: lowercase valid LOG_LEVEL (e.g. 'debug') SHALL be accepted
    after .upper() normalisation and SHALL NOT raise ValueError.
    """
    cfg = _reload_config(monkeypatch, log_level="debug", log_format=None)
    assert cfg.LOG_LEVEL == "DEBUG"


def test_edge_log_format_with_whitespace_accepted(monkeypatch):
    """
    Spec edge case: LOG_FORMAT with surrounding whitespace (e.g. ' json ')
    SHALL be accepted after .strip() normalisation.
    """
    cfg = _reload_config(monkeypatch, log_level=None, log_format=" json ")
    assert cfg.LOG_FORMAT == "json"
