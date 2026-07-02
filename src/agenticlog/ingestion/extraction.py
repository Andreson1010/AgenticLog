# AgenticLog - Estágio de extração da ingestão
"""Extração de conteúdo de PDFs e JSONs para Documents (ADR-018 Fase 3a).

`extrair_texto_pdf` movido verbatim de `agenticlog.rag`; `carregar_json` é o wrapper
single-file de `JSONLoader` + jq_schema compartilhado (rota do orquestrador incremental).
"""

from pathlib import Path

import fitz  # PyMuPDF
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from agenticlog.config import JQ_SCHEMA_CAMPOS_JSON
from agenticlog.shared.errors import RAGSecurityError


def extrair_texto_pdf(path: Path) -> dict[str, str]:
    """Extrai texto de um arquivo PDF usando PyMuPDF (fitz), por página.

    Entrada: path — Path para um arquivo PDF já salvo em disco.
    Saída: dict {"PÁGINA_1": "texto da página 1", "PÁGINA_2": "...", ...} —
      apenas páginas com texto não-vazio após .strip() são incluídas.
    Lança RAGSecurityError se:
      - fitz.open() lança qualquer Exception (arquivo corrompido).
      - doc.needs_pass == True (PDF protegido por senha).
      - dict resultante está vazio (nenhuma página com texto extraível — somente imagem).
    """
    try:
        doc_handle = fitz.open(str(path))
    except fitz.FileDataError:
        raise RAGSecurityError("PDF inválido ou corrompido.")
    except Exception as exc:
        raise RAGSecurityError("PDF inválido ou corrompido.") from exc

    with doc_handle:
        if doc_handle.needs_pass:
            raise RAGSecurityError("PDF protegido por senha.")
        pages: dict[str, str] = {}
        for i, page in enumerate(doc_handle):
            texto = page.get_text().strip()
            if texto:
                pages[f"PÁGINA_{i + 1}"] = texto

    if not pages:
        raise RAGSecurityError("PDF não contém texto extraível (somente imagem).")

    return pages


def carregar_json(caminho: Path) -> list[Document]:
    """Carrega um arquivo JSON como Documents via JSONLoader + jq_schema compartilhado.

    Entrada: caminho — Path para um arquivo JSON já salvo em disco.
    Saída: list[Document] — 1 Document por chave top-level (ADR-008).
    """
    loader = JSONLoader(str(caminho), jq_schema=JQ_SCHEMA_CAMPOS_JSON)
    return loader.load()
