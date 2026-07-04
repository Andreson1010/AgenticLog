# AgenticLog - Estágio de limpeza da ingestão
"""Filtragem de Documents vazios antes do chunking (ADR-011 / ADR-018 Fase 3a).

Módulo puro e imutável: não importa `agenticlog.config`, `rag` nem `agent`.
"""


def filtrar_documentos_vazios(docs: list) -> list:
    """Descarta Documents cujo page_content é vazio após .strip() (ADR-011).

    Entrada: docs — lista de Documents.
    Saída: lista NOVA contendo apenas os Documents com conteúdo não-vazio
      (não muta a lista de entrada — imutabilidade).
    """
    return [d for d in docs if d.page_content.strip()]
