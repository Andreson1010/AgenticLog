# Acceptance tests — RAG Shared + Observability Extraction (ADR-018 Fase 2)
"""
Verifies all acceptance criteria from the approved user story:
  P1-A (RAGSecurityError → shared/errors.py)
  P1-B (_JsonFormatter → observability/logging.py)
  P1-C (HistoryStore → observability/history.py)

Tests exercise the feature from the OUTSIDE — as a downstream developer or consumer
of the agenticlog package would experience it, not through implementation internals.
Each test maps to exactly one acceptance criterion.

OBS-01..21 traceability is noted inline.
"""

import datetime
import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure src/ is on path (mirrors conftest.py convention)
_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.api as api_module
import agenticlog.config as config_module
import agenticlog.history as history_shim
import agenticlog.observability as obs_pkg
import agenticlog.observability.history as obs_history
import agenticlog.observability.logging as obs_logging
import agenticlog.rag as rag_module
import agenticlog.shared as shared_pkg
import agenticlog.shared.errors as shared_errors


# ---------------------------------------------------------------------------
# AC-01: All 3 canonical imports resolve to the right base types
# OBS-01, OBS-06, OBS-12
# ---------------------------------------------------------------------------

def test_ac01_canonical_imports_resolve_to_correct_types() -> None:
    """
    AC-01: WHEN a developer imports the three symbols from their canonical locations
    THEN each SHALL resolve to the expected type with the correct base class or interface.

    Covers OBS-01 (RAGSecurityError<:Exception), OBS-06 (_JsonFormatter<:Formatter),
    OBS-12 (HistoryStore has init_db/append/read_all methods).
    """
    # RAGSecurityError must be a real Exception subclass (OBS-01)
    from agenticlog.shared.errors import RAGSecurityError as CanonicalError
    assert isinstance(CanonicalError, type), "RAGSecurityError must be a class"
    assert issubclass(CanonicalError, Exception), "RAGSecurityError must subclass Exception"

    # _JsonFormatter must be a logging.Formatter subclass (OBS-06)
    from agenticlog.observability.logging import _JsonFormatter as CanonicalFormatter
    assert isinstance(CanonicalFormatter, type), "_JsonFormatter must be a class"
    assert issubclass(CanonicalFormatter, logging.Formatter), (
        "_JsonFormatter must subclass logging.Formatter"
    )

    # HistoryStore must expose the required public interface (OBS-12)
    from agenticlog.observability.history import HistoryStore as CanonicalStore
    assert isinstance(CanonicalStore, type), "HistoryStore must be a class"
    assert callable(getattr(CanonicalStore, "init_db", None)), "HistoryStore.init_db missing"
    assert callable(getattr(CanonicalStore, "append", None)), "HistoryStore.append missing"
    assert callable(getattr(CanonicalStore, "read_all", None)), "HistoryStore.read_all missing"


# ---------------------------------------------------------------------------
# AC-02: All 3 shim (old-path) imports still resolve
# OBS-02, OBS-07, OBS-13
# ---------------------------------------------------------------------------

def test_ac02_shim_imports_still_resolve() -> None:
    """
    AC-02: WHEN a consumer uses the OLD import paths (the ones that existed before
    this feature) THEN they SHALL still resolve without ImportError.

    This ensures backward compatibility is preserved for all existing consumers.
    Covers OBS-02 (rag.RAGSecurityError), OBS-07 (config._JsonFormatter),
    OBS-13 (history.HistoryStore).
    """
    # Existing consumers of rag.RAGSecurityError (app.py, tests/) must keep working
    from agenticlog.rag import RAGSecurityError as ShimError
    assert ShimError is not None

    # Existing consumers of config._JsonFormatter (rag.py internal import) must keep working
    from agenticlog.config import _JsonFormatter as ShimFormatter
    assert ShimFormatter is not None

    # Existing consumers of history.HistoryStore (api.py) must keep working
    from agenticlog.history import HistoryStore as ShimStore
    assert ShimStore is not None


# ---------------------------------------------------------------------------
# AC-03: All 3 package __init__ re-exports resolve
# OBS-03, OBS-08, OBS-14
# ---------------------------------------------------------------------------

def test_ac03_package_init_reexports_resolve() -> None:
    """
    AC-03: WHEN a developer uses the ergonomic package-level import paths
    (from agenticlog.shared import X, from agenticlog.observability import Y)
    THEN they SHALL resolve and the symbols SHALL appear in __all__.

    Covers OBS-03 (shared.__init__), OBS-08, OBS-14 (observability.__init__).
    """
    # shared package re-export (OBS-03)
    from agenticlog.shared import RAGSecurityError as PkgError
    assert PkgError is not None
    assert "RAGSecurityError" in shared_pkg.__all__, (
        "RAGSecurityError must appear in agenticlog.shared.__all__"
    )

    # observability package re-export of _JsonFormatter (OBS-08)
    from agenticlog.observability import _JsonFormatter as PkgFormatter
    assert PkgFormatter is not None
    assert "_JsonFormatter" in obs_pkg.__all__, (
        "_JsonFormatter must appear in agenticlog.observability.__all__"
    )

    # observability package re-export of HistoryStore (OBS-14)
    from agenticlog.observability import HistoryStore as PkgStore
    assert PkgStore is not None
    assert "HistoryStore" in obs_pkg.__all__, (
        "HistoryStore must appear in agenticlog.observability.__all__"
    )


# ---------------------------------------------------------------------------
# AC-04: Object identity — all import paths resolve to the SAME object
# OBS-04, OBS-09, OBS-15, OBS-17 (api consumer namespace)
# ---------------------------------------------------------------------------

def test_ac04_object_identity_canonical_is_shim_is_init_reexport() -> None:
    """
    AC-04: WHEN a developer imports the same symbol from ANY valid path
    (canonical, shim, package __init__) THEN all SHALL resolve to the
    EXACT same Python object (identity, not equality).

    Also verifies:
    - agenticlog.rag._JsonFormatter IS the canonical (rag imports from config shim)
    - agenticlog.api.HistoryStore IS the canonical (api imports from history shim)

    Covers OBS-04 (errors identity), OBS-09 (formatter identity — all 4 paths),
    OBS-15 (history identity — 3 paths), OBS-17 (api consumer namespace).
    """
    canonical_error = shared_errors.RAGSecurityError
    assert shared_pkg.RAGSecurityError is canonical_error, (
        "shared.__init__.RAGSecurityError must be the SAME object as shared.errors.RAGSecurityError"
    )
    assert rag_module.RAGSecurityError is canonical_error, (
        "rag.RAGSecurityError shim must be the SAME object as the canonical"
    )

    canonical_formatter = obs_logging._JsonFormatter
    assert obs_pkg._JsonFormatter is canonical_formatter, (
        "observability.__init__._JsonFormatter must be the SAME object as canonical"
    )
    assert config_module._JsonFormatter is canonical_formatter, (
        "config._JsonFormatter shim must be the SAME object as canonical"
    )
    assert rag_module._JsonFormatter is canonical_formatter, (
        "rag._JsonFormatter (from config shim) must be the SAME object as canonical"
    )

    canonical_store = obs_history.HistoryStore
    assert obs_pkg.HistoryStore is canonical_store, (
        "observability.__init__.HistoryStore must be the SAME object as canonical"
    )
    assert history_shim.HistoryStore is canonical_store, (
        "history shim HistoryStore must be the SAME object as canonical"
    )
    assert api_module.HistoryStore is canonical_store, (
        "api.HistoryStore (consumer namespace) must be the SAME object as canonical"
    )


# ---------------------------------------------------------------------------
# AC-05a: RAGSecurityError — cross-path raise/catch (OBS-05)
# ---------------------------------------------------------------------------

def test_ac05a_ragsecurityerror_cross_path_raise_catch() -> None:
    """
    AC-05a: WHEN agenticlog.rag raises RAGSecurityError (as in production code)
    AND the catcher uses the canonical path (agenticlog.shared.errors),
    THEN the except SHALL catch it — and vice versa.

    This verifies there is ONE class with ONE MRO, not two duplicate classes.
    Covers OBS-05.
    """
    # Shim raises → canonical catches
    caught = False
    try:
        raise rag_module.RAGSecurityError("path traversal detected")
    except shared_errors.RAGSecurityError as exc:
        caught = True
        assert str(exc) == "path traversal detected"
    assert caught, "canonical except did not catch exception raised via rag shim"

    # Canonical raises → shim catches
    caught = False
    try:
        raise shared_errors.RAGSecurityError("forbidden key")
    except rag_module.RAGSecurityError as exc:
        caught = True
        assert str(exc) == "forbidden key"
    assert caught, "rag shim except did not catch exception raised via canonical"

    # Package __init__ raises → canonical catches (full triangle)
    caught = False
    try:
        raise shared_pkg.RAGSecurityError("oversized file")
    except shared_errors.RAGSecurityError as exc:
        caught = True
        assert str(exc) == "oversized file"
    assert caught, "canonical except did not catch exception raised via shared package"


# ---------------------------------------------------------------------------
# AC-05b: _JsonFormatter — format() output unchanged (OBS-10)
# ---------------------------------------------------------------------------

def test_ac05b_json_formatter_format_output_has_exactly_four_keys() -> None:
    """
    AC-05b: WHEN _JsonFormatter().format(record) runs on a fixed LogRecord
    THEN the output SHALL be a valid JSON string with EXACTLY the keys
    'timestamp', 'level', 'logger', 'message' and their expected values.

    Tests both the canonical and the config-shim formatters to confirm
    behavior is identical after relocation. Covers OBS-10.
    """
    record = logging.LogRecord(
        name="agenticlog.acceptance",
        level=logging.ERROR,
        pathname=__file__,
        lineno=99,
        msg="falha crítica %s",
        args=("no pipeline",),
        exc_info=None,
    )

    # Test via canonical import
    canonical_output = obs_logging._JsonFormatter().format(record)
    parsed = json.loads(canonical_output)

    assert set(parsed.keys()) == {"timestamp", "level", "logger", "message"}, (
        f"Expected exactly 4 keys, got: {set(parsed.keys())}"
    )
    assert parsed["level"] == "ERROR"
    assert parsed["logger"] == "agenticlog.acceptance"
    assert parsed["message"] == "falha crítica no pipeline"

    expected_ts = datetime.datetime.fromtimestamp(
        record.created, tz=datetime.timezone.utc
    ).isoformat()
    assert parsed["timestamp"] == expected_ts, (
        f"timestamp mismatch: {parsed['timestamp']} != {expected_ts}"
    )

    # Test via config shim — must produce byte-identical output (OBS-09 + OBS-10)
    shim_output = config_module._JsonFormatter().format(record)
    assert shim_output == canonical_output, (
        "config shim _JsonFormatter().format() must produce identical output to canonical"
    )


# ---------------------------------------------------------------------------
# AC-05c: HistoryStore — round-trip, schema, DESC order, FIFO eviction (OBS-16)
# ---------------------------------------------------------------------------

def test_ac05c_historystore_append_read_all_round_trip_schema_and_fifo(
    tmp_path: Path,
) -> None:
    """
    AC-05c: WHEN HistoryStore is constructed, entries appended, and read_all called
    THEN the SQLite schema, DESC ordering, and FIFO eviction at max_entries SHALL
    be identical to the pre-move implementation.

    Exercises the canonical class through the full consumer lifecycle:
    construct → init_db → append → read_all → schema check → eviction.
    Covers OBS-16.
    """
    db_path = tmp_path / "acceptance_hist.db"
    store = obs_history.HistoryStore(db_path, max_entries=2)

    def _registro(ts: str, query: str) -> dict:
        return {
            "timestamp": ts,
            "query": query,
            "next_step": "retrieve",
            "confidence_score": 0.75,
            "ranked_response": "resultado de teste",
        }

    # Append 3 entries with max_entries=2 → oldest (q1) must be evicted
    store.append(_registro("2026-01-01T00:00:00", "q1"))
    store.append(_registro("2026-01-02T00:00:00", "q2"))
    store.append(_registro("2026-01-03T00:00:00", "q3"))

    rows = store.read_all()

    # Schema: must have exactly these 6 columns (OBS-16 — no schema change)
    expected_columns = {"id", "timestamp", "query", "next_step", "confidence_score", "ranked_response"}
    assert set(rows[0].keys()) == expected_columns, (
        f"Schema mismatch. Got columns: {set(rows[0].keys())}"
    )

    # FIFO eviction: q1 (oldest) must be gone, only q3 and q2 remain
    assert len(rows) == 2, f"Expected 2 rows after eviction, got {len(rows)}"
    queries = [r["query"] for r in rows]
    assert "q1" not in queries, "q1 (oldest) should have been evicted"

    # DESC order: most recent first
    assert queries[0] == "q3", f"Expected q3 first (DESC), got {queries[0]}"
    assert queries[1] == "q2", f"Expected q2 second (DESC), got {queries[1]}"

    # Verify the same behavior via the history shim (backward-compat consumer path)
    store_via_shim = history_shim.HistoryStore(tmp_path / "shim_hist.db", max_entries=100)
    store_via_shim.append(_registro("2026-06-01T00:00:00", "shim_query"))
    shim_rows = store_via_shim.read_all()
    assert len(shim_rows) == 1
    assert shim_rows[0]["query"] == "shim_query"


# ---------------------------------------------------------------------------
# AC-06: No circular import — fresh interpreter exits cleanly (OBS-18)
# ---------------------------------------------------------------------------

def test_ac06_no_circular_import_fresh_interpreter() -> None:
    """
    AC-06: WHEN `import agenticlog` runs in a fresh interpreter (cold sys.modules)
    THEN it SHALL exit 0 with no ImportError or 'partially initialized module' error.

    Uses a subprocess to guarantee a genuinely cold import — the current process
    already has all modules cached and would not catch a circular import.
    Covers OBS-18.
    """
    python = sys.executable
    result = subprocess.run(
        [python, "-c", "import agenticlog"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Fresh `import agenticlog` failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "circular" not in result.stderr.lower(), (
        f"Circular import detected in stderr: {result.stderr!r}"
    )
    assert "partially initialized" not in result.stderr.lower(), (
        f"Partially initialized module in stderr: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# AC-07: observability.logging does not import agenticlog.config (OBS-11)
# ---------------------------------------------------------------------------

def test_ac07_observability_logging_does_not_import_config() -> None:
    """
    AC-07: WHEN agenticlog.observability.logging is imported THEN it SHALL NOT
    import agenticlog.config — keeping the import graph acyclic
    (config → observability.logging must be a forward edge only).

    Verified two ways:
    1. The loaded module's namespace does not contain a 'config' reference.
    2. The module's source has no import line referencing 'agenticlog'.

    Covers OBS-11.
    """
    import inspect

    # Check 1: the loaded module must NOT have brought agenticlog.config into scope
    assert not hasattr(obs_logging, "config"), (
        "observability.logging must not expose a 'config' attribute (would indicate a direct import)"
    )

    # Check 2: no import statement in the source references agenticlog at all
    source_lines = inspect.getsource(obs_logging).splitlines()
    agenticlog_imports = [
        line.strip()
        for line in source_lines
        if line.strip().startswith(("import ", "from ")) and "agenticlog" in line
    ]
    assert agenticlog_imports == [], (
        f"observability.logging contains agenticlog import(s) — cycle risk!\n"
        f"Found: {agenticlog_imports}"
    )

    # Checks 1 and 2 above are the definitive acceptance gate for AC-07:
    # (a) the loaded module has no 'config' attribute in its namespace, and
    # (b) its source contains no import line that references 'agenticlog'.
    # A subprocess check here would require the feature worktree's src/ on PYTHONPATH,
    # which the current test runner does not guarantee; the in-process checks are sufficient.


# ---------------------------------------------------------------------------
# AC-08: ingestion package import is acyclic — fresh interpreter (ADR-018 Fase 3a)
# ---------------------------------------------------------------------------

def test_ac08_ingestion_import_acyclic_fresh_interpreter() -> None:
    """
    AC-08: WHEN `import agenticlog.ingestion` runs in a fresh interpreter (cold
    sys.modules) THEN it SHALL exit 0 with no circular / partially-initialized error.

    `ingestion/*` depende só de `agenticlog.config` e `agenticlog.shared` (folhas do
    grafo) e, intra-pacote, `security → extraction`. Nenhum import de `rag`/`agent`.
    Covers RAGING-10.
    """
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = _src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import agenticlog.ingestion"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Fresh `import agenticlog.ingestion` failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "circular" not in result.stderr.lower(), (
        f"Circular import detected in stderr: {result.stderr!r}"
    )
    assert "partially initialized" not in result.stderr.lower(), (
        f"Partially initialized module in stderr: {result.stderr!r}"
    )
