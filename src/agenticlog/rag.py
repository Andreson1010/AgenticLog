# AgenticLog - Pipeline RAG (fachada de compatibilidade)
"""
Fachada do pipeline de ingestão RAG (ChromaDB).

Desde a Fase 3a (ADR-018) os estágios de baixo risco (segurança, extração, limpeza,
chunking, embeddings, metadados) vivem em `agenticlog.ingestion` e são re-exportados
abaixo via shims identity-preserving. A Fase 3b moveu a camada de persistência/atomicidade
para `agenticlog.ingestion.store` e os 5 orquestradores para `agenticlog.ingestion.orchestrator`.

`rag.py` permanece como fachada: mantém os globais/seams (`DIR_DOCUMENTS`, `DIR_VECTORDB`,
`_rag_embedding_model` + getter/cache), a CLI, os shims `is`-idênticos (Fase 3a + os 4
símbolos de `store`) e WRAPPERS finos dos orquestradores que ligam os seams NO MOMENTO DA
CHAMADA (para que `monkeypatch.setattr("agenticlog.rag.DIR_*")` continue fluindo).

Execute: python -m agenticlog.rag
"""

import argparse
import logging

import fitz  # noqa: F401  # PyMuPDF — mantido p/ @patch("agenticlog.rag.fitz.open") (singleton sys.modules)
from langchain_huggingface import HuggingFaceEmbeddings

from agenticlog.config import (
    DIR_DOCUMENTS,
    DIR_VECTORDB,
    LOG_LEVEL,
    LOG_FORMAT,
    _JsonFormatter,
    DEFAULT_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

vectordb = None
_rag_embedding_model = None


def _get_rag_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna singleton de HuggingFaceEmbeddings para ingestão incremental.

    Entrada: nenhuma.
    Saída: instância de HuggingFaceEmbeddings (criada uma única vez por processo).

    O cache e o getter FICAM em `rag.py` (ADR-018 Fase 3a): a construção é delegada
    a `ingestion.embeddings.criar_embedding_model`, mas o estado (singleton) permanece
    aqui para preservar `monkeypatch.setattr("agenticlog.rag._rag_embedding_model", ...)`.
    """
    global _rag_embedding_model
    if _rag_embedding_model is None:
        _rag_embedding_model = criar_embedding_model()
    return _rag_embedding_model


# ── Re-export shims (ADR-018 Fase 3a) — remover na Fase 6 ─────────────────────
from agenticlog.shared.errors import RAGSecurityError  # noqa: E402  # Re-export shim (ADR-018 Fase 2) — remover na Fase 6
from agenticlog.ingestion.security import (  # noqa: E402,F401
    _valida_path_documentos,
    _valida_json_sem_chaves_proibidas,
    _valida_arquivos_json,
    _sanitizar_nome_arquivo,
    _sanitizar_nome_colecao,
    sanitizar_nome_colecao,
    salvar_documento_enviado,
    salvar_pdf_enviado,
)  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
from agenticlog.ingestion.extraction import (  # noqa: E402
    extrair_texto_pdf,
    carregar_json,
)  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
from agenticlog.ingestion.cleaning import filtrar_documentos_vazios  # noqa: E402,F401  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
from agenticlog.ingestion.chunking import SemanticChunker  # noqa: E402,F401  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
from agenticlog.ingestion.embeddings import criar_embedding_model  # noqa: E402  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6
from agenticlog.ingestion.metadata import (  # noqa: E402,F401
    _computar_hash_conteudo,
    _hash_arquivo,
    _enriquecer_metadados_chunks,
)  # Re-export shim (ADR-018 Fase 3a) — remover na Fase 6

# ── Re-export shims de store (ADR-018 Fase 3b) — remover na Fase 6 ────────────
# Identidade `is` preservada: agenticlog.rag.X is agenticlog.ingestion.store.X.
from agenticlog.ingestion.store import (  # noqa: E402,F401
    _backup_arquivo,
    _reverter_disco,
    _outras_colecoes_existem,
    _resetar_colecao,
)  # Re-export shim (ADR-018 Fase 3b) — remover na Fase 6

import agenticlog.ingestion.orchestrator as _orch  # noqa: E402


# ── WRAPPERS de orquestrador (ADR-018 Fase 3b) — remover na Fase 6 ────────────
# NÃO são `is`-idênticos aos de `orchestrator`: ligam os seams de rag.py (DIR_DOCUMENTS,
# DIR_VECTORDB, _get_rag_embedding_model) NO MOMENTO DA CHAMADA e os injetam por argumento,
# preservando o monkeypatch do oráculo (§4 do design; ADR-019).
def adicionar_documento_incrementalmente(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict[str, str]:
    """Wrapper: delega ao orquestrador ligando os seams de `rag.py` na chamada."""
    return _orch.adicionar_documento_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=_get_rag_embedding_model(),
    )


def adicionar_pdf_incrementalmente(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict[str, str]:
    """Wrapper: delega ao orquestrador ligando os seams de `rag.py` na chamada."""
    return _orch.adicionar_pdf_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=_get_rag_embedding_model(),
    )


def ingerir_incrementalmente(
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict[str, int]:
    """Wrapper: injeta o singleton de embedding só quando há documentos a ingerir.

    O guard `tem_documentos` preserva a laziness da construção do modelo (~1 GB): com
    o diretório vazio o singleton NÃO é construído. Com documentos, o singleton é
    construído uma única vez e reutilizado por todos os `adicionar_*` do lote.
    """
    tem_documentos = any(DIR_DOCUMENTS.glob("*.json")) or any(DIR_DOCUMENTS.glob("*.pdf"))
    embedding_model = _get_rag_embedding_model() if tem_documentos else None
    return _orch.ingerir_incrementalmente(
        collection_name,
        docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=embedding_model,
    )


def cria_vectordb(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """Wrapper: rebuild delega ao orquestrador (modelo fresco) e reatribui `vectordb`.

    O rebuild constrói um `HuggingFaceEmbeddings` fresco no orquestrador (seam `None`, D2).
    """
    global vectordb  # inicializado como None no nível do módulo; preenchido aqui
    vectordb = _orch.cria_vectordb(
        collection_name, docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=None,
    )


def reconstruir_vectordb(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """Wrapper: reconstrução completa delega ao orquestrador ligando os seams de `rag.py`."""
    return _orch.reconstruir_vectordb(
        collection_name, docs_dir=DIR_DOCUMENTS, vectordb_dir=DIR_VECTORDB,
        embedding_model=None,
    )


def _configurar_logging_cli() -> None:
    """Configura o logger do pacote 'agenticlog' para a execução via CLI."""
    pkg_logger = logging.getLogger("agenticlog")
    pkg_logger.setLevel(LOG_LEVEL)
    # clear existing handlers to avoid duplicates on repeated calls
    pkg_logger.handlers.clear()

    handler = logging.StreamHandler()
    if LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    pkg_logger.addHandler(handler)


def _executar_main(argv: list[str] | None = None) -> None:
    """Ponto de entrada CLI — configura logging e ingere documentos (REC-04).

    Sem flags  → ingestão incremental de todos os arquivos em data/documents/.
    --rebuild  → reconstrução completa do banco vetorial (cria_vectordb, comportamento legado).
    """
    parser = argparse.ArgumentParser(
        prog="python -m agenticlog.rag",
        description="Constrói ou atualiza o banco vetorial ChromaDB do AgenticLog.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstrói o banco vetorial do zero (descarta o índice atual).",
    )
    args = parser.parse_args(argv)

    _configurar_logging_cli()

    try:
        if args.rebuild:
            cria_vectordb()
        else:
            ingerir_incrementalmente()
    except RAGSecurityError as e:
        logger.error("Erro de segurança: %s", e)
        raise SystemExit(1) from e
    except Exception as e:
        operacao = "rebuild do banco vetorial" if args.rebuild else "ingestão incremental"
        logger.error("Erro durante %s: %s", operacao, e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    _executar_main()
