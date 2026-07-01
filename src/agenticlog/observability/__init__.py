# AgenticLog - Pacote de observabilidade
"""Pacote `observability`: concerns transversais (logging estruturado, audit log).

Somente submódulos stdlib-only; NÃO importa `agenticlog.config`, mantendo o grafo
de imports acíclico (config -> observability.logging é aresta unidirecional).

Re-exporta para import ergonômico:
`from agenticlog.observability import _JsonFormatter`.
"""

from agenticlog.observability.logging import _JsonFormatter

__all__ = ["_JsonFormatter"]
