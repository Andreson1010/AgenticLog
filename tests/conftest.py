# AgenticLog - Configuração pytest
"""
Conftest para pytest: garante que src esteja no path antes de qualquer teste.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
