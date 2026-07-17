# AgenticLog - Estágio de chunking da ingestão
"""Re-export do SemanticChunker (ADR-013 / ADR-018 Fase 3a).

Re-export bare (sem factory) para que
`@patch("agenticlog.ingestion.chunking.SemanticChunker")` continue funcionando
e a identidade do símbolo seja preservada.
"""

from langchain_experimental.text_splitter import SemanticChunker

__all__ = ["SemanticChunker"]
