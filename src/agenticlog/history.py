"""
Shim de compatibilidade do histórico de queries do AgenticLog.

O corpo canônico de `HistoryStore` foi movido para
`agenticlog.observability.history` (ADR-018 Fase 2). Este módulo re-exporta a
classe para preservar o caminho de import `from agenticlog.history import HistoryStore`
usado por `api.py` e seus testes (`@patch("agenticlog.api.HistoryStore")`).
"""

from agenticlog.observability.history import HistoryStore  # noqa: E501,I001  # Re-export shim (ADR-018 Fase 2) — remover na Fase 6

__all__ = ["HistoryStore"]
