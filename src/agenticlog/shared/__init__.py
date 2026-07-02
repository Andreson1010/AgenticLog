# AgenticLog - Pacote de contratos compartilhados
"""Pacote `shared`: contratos e exceções de domínio reutilizados entre pacotes.

Re-exporta `RAGSecurityError` para import ergonômico:
`from agenticlog.shared import RAGSecurityError`.
"""

from agenticlog.shared.errors import RAGSecurityError

__all__ = ["RAGSecurityError"]
