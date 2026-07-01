# AgenticLog - Exceções de domínio compartilhadas
"""Contratos de exceções compartilhados entre os pacotes do AgenticLog.

Ponto único de definição das exceções de domínio; os módulos de origem
(ex.: rag.py) re-exportam via shim para preservar os caminhos de import.
"""


class RAGSecurityError(Exception):
    """Exceção lançada quando uma violação de segurança é detectada no pipeline RAG.

    Exemplos de violações: path traversal, chaves JSON proibidas, arquivo muito grande.
    """
