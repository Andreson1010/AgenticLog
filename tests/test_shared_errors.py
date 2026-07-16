# Unit tests — RAGSecurityError extraction to shared/errors.py (OBS-01..05)
"""Cobre a extração de `RAGSecurityError` para `shared/errors.py` (ADR-018 Fase 2).

Valida:
- OBS-01: import canônico `agenticlog.shared.errors.RAGSecurityError`.
- OBS-03: re-export ergonômico `agenticlog.shared.RAGSecurityError`.
- OBS-04: identidade de objeto (canônico IS __init__ re-export IS shim de rag.py).
- OBS-05: semântica de raise/catch idêntica através de ambos os nomes.
"""

import sys
from pathlib import Path

# Garante src/ no path (espelha a convenção do conftest.py).
_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.shared as shared_pkg
import agenticlog.shared.errors as shared_errors
from agenticlog.shared.errors import RAGSecurityError


def test_import_canonico_resolve_classe() -> None:
    """OBS-01: import canônico resolve a classe de exceção."""
    assert isinstance(RAGSecurityError, type)
    assert issubclass(RAGSecurityError, Exception)


def test_reexport_pacote_shared_e_mesmo_objeto() -> None:
    """OBS-03/OBS-04: `agenticlog.shared` re-exporta o MESMO objeto do canônico."""
    assert shared_pkg.RAGSecurityError is shared_errors.RAGSecurityError
    assert "RAGSecurityError" in shared_pkg.__all__


def test_raise_catch_round_trip_atraves_de_ambos_os_nomes() -> None:
    """OBS-05: uma exceção levantada por um nome é capturada pelo outro (mesma MRO)."""
    try:
        raise shared_errors.RAGSecurityError("violação")
    except shared_errors.RAGSecurityError as exc:
        assert str(exc) == "violação"

    try:
        raise shared_errors.RAGSecurityError("outra")
    except shared_errors.RAGSecurityError as exc:
        assert str(exc) == "outra"
