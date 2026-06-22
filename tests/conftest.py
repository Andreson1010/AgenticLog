# AgenticLog - Configuração pytest
"""
Conftest para pytest:
  1. Garante que src/ esteja no path antes de qualquer teste.
  2. Pula testes marcados @pytest.mark.integration quando o módulo nativo
     hnswlib (dependência do ChromaDB) não carrega neste ambiente — caso do
     Windows com Smart App Control bloqueando o .pyd não-assinado. No CI Linux
     o hnswlib carrega normalmente e os testes de integração rodam.
"""
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def _hnswlib_importavel() -> bool:
    """True se o módulo nativo hnswlib carrega neste ambiente.

    Sob Smart App Control (Windows), o `.pyd` não-assinado é bloqueado e o import
    levanta ImportError — os testes de integração que usam ChromaDB real não
    conseguem rodar localmente, só no CI.
    """
    try:
        import hnswlib  # noqa: F401
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Pula testes @integration quando hnswlib (ChromaDB real) não está disponível."""
    if _hnswlib_importavel():
        return
    skip_integration = pytest.mark.skip(
        reason="hnswlib bloqueado (ex.: Smart App Control no Windows) — "
        "teste de integração com ChromaDB real roda só onde o hnswlib carrega (CI Linux)."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
