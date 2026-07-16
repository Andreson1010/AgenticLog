# AgenticLog - Orquestradores da ingestão
"""Orquestradores do pipeline RAG: rebuild + ingestão incremental (ADR-018 Fase 3b).

Movido de `agenticlog.rag` preservando comportamento byte-idêntico. Contém os 5
orquestradores (`cria_vectordb`, `adicionar_documento_incrementalmente`,
`adicionar_pdf_incrementalmente`, `ingerir_incrementalmente`, `reconstruir_vectordb`)
e o helper privado compartilhado `_ingerir_arquivo_incrementalmente` que deduplica os
dois `adicionar_*` (parametrizado por passo de extração por-tipo).

Importa `config`, `shared`, `store` e os estágios da Fase 3a (`extraction`, `cleaning`,
`chunking`, `embeddings`, `metadata`, `security`). NUNCA importa `rag`/`agent` no nível
de módulo (`invalidar_vector_db` é importado lazy dentro da função) — mantém o grafo de
imports acíclico. Os seams `docs_dir`/`vectordb_dir`/`embedding_model` têm default `None`
e são resolvidos no CORPO (fallback `config.*`), permitindo injeção via os wrappers de
`rag.py`, que ligam os globais no momento da chamada (§4 do design).
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from agenticlog.config import (
    CHROMA_COLLECTION_METADATA,
    DEFAULT_COLLECTION_NAME,
    DIR_DOCUMENTS,
    DIR_VECTORDB,
    EMBEDDING_MODEL,
    MAX_DOCUMENT_FILE_SIZE_MB,
    MAX_JSON_FILE_SIZE_MB,
    MAX_JSON_FILES,
    METADATA_DOC_TYPE_JSON,
    METADATA_DOC_TYPE_PDF,
    METADATA_FILE_HASH,
    METADATA_PAGE,
    METADATA_PAGE_JSON_SENTINEL,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_BREAKPOINT_TYPE,
)
from agenticlog.ingestion.chunking import SemanticChunker
from agenticlog.ingestion.cleaning import filtrar_documentos_vazios
from agenticlog.ingestion.embeddings import criar_embedding_model
from agenticlog.ingestion.extraction import carregar_json, extrair_texto_pdf
from agenticlog.ingestion.metadata import (
    _computar_hash_conteudo,
    _enriquecer_metadados_chunks,
    _hash_arquivo,
)
from agenticlog.ingestion.security import (
    _sanitizar_nome_arquivo,
    _sanitizar_nome_colecao,
    _valida_arquivos_json,
    _valida_json_sem_chaves_proibidas,
    _valida_path_documentos,
)
from agenticlog.ingestion.store import (
    _backup_arquivo,
    _resetar_colecao,
    _reverter_disco,
    add_documents_com_rollback,
)
from agenticlog.shared.errors import RAGSecurityError

# Loga sob "agenticlog.rag" (não __name__): preserva a saída byte-idêntica dos registros
# e mantém os `assertLogs("agenticlog.rag")` existentes verdes na fase de shims/wrappers.
logger = logging.getLogger("agenticlog.rag")


# --------------------------------------------------------------------------- #
# Helpers compartilhados da ingestão incremental (dedup dos dois `adicionar_*`).
# --------------------------------------------------------------------------- #
def _checar_limite_arquivos(docs_dir: Path) -> None:
    """Levanta RAGSecurityError se adicionar mais um arquivo exceder MAX_JSON_FILES."""
    json_count = len(list(docs_dir.glob("*.json")))
    pdf_count = len(list(docs_dir.glob("*.pdf")))
    if json_count + pdf_count + 1 > MAX_JSON_FILES:
        raise RAGSecurityError(f"Limite de {MAX_JSON_FILES} arquivos atingido.")


def _detectar_duplicata(
    vectordb_instance, planned_path: Path, hash_str: str, safe_name: str
) -> tuple[dict[str, str] | None, list[str]]:
    """Consulta o Chroma por ``source`` e decide duplicado/upsert.

    Saída: (resultado_duplicado | None, old_ids). Se o hash bate, retorna o dict de
    status "duplicado" e old_ids vazio; se difere, retorna (None, old_ids) para upsert.
    """
    existing = vectordb_instance.get(where={"source": {"$eq": str(planned_path)}})
    old_ids: list[str] = []
    if existing["ids"]:
        existing_hash = existing["metadatas"][0].get(METADATA_FILE_HASH)
        if existing_hash == hash_str:
            return {
                "status": "duplicado",
                "mensagem": f"Arquivo {safe_name} já está presente na base vetorial.",
            }, []
        old_ids = list(existing["ids"])
    return None, old_ids


def _gravar_arquivo_no_disco(
    conteudo: bytes,
    planned_path: Path,
    old_ids: list[str],
    tmp_suffix: str,
    preparar_no_tmp,
) -> tuple[Path, Path | None, object]:
    """Grava tmp → valida/extrai (preparar_no_tmp) → backup se upsert → shutil.move.

    Saída: (saved_path, backup_path | None, contexto). ``contexto`` é o retorno de
    ``preparar_no_tmp`` (None para JSON; páginas extraídas para PDF), consumido depois
    por ``construir_docs`` DENTRO do bloco guardado (DN-2).
    """
    tmp_path: Path | None = None
    backup_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
            tmp.write(conteudo)
            tmp_path = Path(tmp.name)
        contexto = preparar_no_tmp(tmp_path)
        if old_ids and planned_path.exists():
            backup_path = _backup_arquivo(planned_path)
        shutil.move(str(tmp_path), planned_path)
        saved_path = planned_path
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return saved_path, backup_path, contexto


def _indexar_no_chroma(
    vectordb_instance,
    saved_path: Path,
    backup_path: Path | None,
    contexto: object,
    *,
    construir_docs,
    embedding_model,
    hash_str: str,
    doc_type: str,
    page_args: tuple,
    old_ids: list[str],
    safe_name: str,
) -> int:
    """Bloco guardado (atomicidade de upsert): constrói docs → chunk → add → delete antigos.

    ``construir_docs(saved_path, contexto)`` é a 1ª instrução DENTRO do guard (DN-2). Em
    falha, ``_reverter_disco`` restaura o disco e a exceção é re-levantada. Saída: número
    de chunks persistidos (0 → disco já revertido; chamador retorna a mensagem de 0 chunks).
    """
    try:
        docs = construir_docs(saved_path, contexto)
        text_splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
        )
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            logger.warning("Arquivo %s produziu zero chunks após divisão.", safe_name)
            _reverter_disco(saved_path, backup_path)
            return 0

        _enriquecer_metadados_chunks(chunks, hash_str, doc_type, *page_args)
        chunk_ids = [uuid.uuid4().hex for _ in chunks]
        add_documents_com_rollback(vectordb_instance, chunks, chunk_ids)

        if old_ids:
            try:
                vectordb_instance.delete(ids=old_ids)
            except Exception as del_exc:
                logger.warning(
                    "Falha ao deletar chunks antigos de %s (IDs: %s). Erro: %s",
                    safe_name, old_ids, del_exc,
                )
    except Exception:
        _reverter_disco(saved_path, backup_path)
        raise
    finally:
        if backup_path is not None and backup_path.exists():
            backup_path.unlink(missing_ok=True)
    return len(chunks)


def _notificar_invalidacao(invalidation_msg: str) -> None:
    """Invalida o singleton de vector DB do agente (lazy import evita ciclo/CLI pesado)."""
    try:
        from agenticlog.retrieval.retriever import (
            invalidar_vector_db,  # lazy — evita importação pesada no CLI
        )
        invalidar_vector_db()
    except ImportError as e:
        logger.warning(invalidation_msg, e)


def _construir_docs_json(saved_path: Path, contexto: object) -> list[Document]:
    """Carrega o JSON salvo e descarta Documents vazios (ADR-011)."""
    return filtrar_documentos_vazios(carregar_json(saved_path))


def _construir_docs_pdf(saved_path: Path, contexto: dict[str, str]) -> list[Document]:
    """Constrói 1 Document por página a partir das páginas extraídas; descarta vazios."""
    pdf_docs = []
    for chave, texto in contexto.items():
        page_num = int(chave.split("_")[1])
        pdf_docs.append(
            Document(
                page_content=f"{chave}: {texto}",
                metadata={"source": str(saved_path), METADATA_PAGE: page_num},
            )
        )
    return filtrar_documentos_vazios(pdf_docs)


def _ingerir_arquivo_incrementalmente(
    filename: str,
    conteudo: bytes,
    collection_name: str,
    *,
    docs_dir: Path,
    vectordb_dir: Path,
    embedding_model,
    tmp_suffix: str,
    preparar_no_tmp,
    construir_docs,
    doc_type: str,
    page_args: tuple,
    invalidation_msg: str,
) -> dict[str, str]:
    """Núcleo compartilhado dos dois `adicionar_*` (ING3B-06/07).

    Preserva EXATAMENTE a sequência de atomicidade de upsert:
    ``backup → move → [guardado: construir_docs → chunk → add → delete antigos] →
    _reverter_disco na exceção``. As diferenças por-tipo entram via callbacks
    (``preparar_no_tmp``/``construir_docs``) e parâmetros (``doc_type``/``page_args``/
    ``invalidation_msg``).
    """
    safe_name = _sanitizar_nome_arquivo(filename)
    _checar_limite_arquivos(docs_dir)
    hash_str = _computar_hash_conteudo(conteudo)
    planned_path = docs_dir / safe_name

    if embedding_model is None:
        embedding_model = criar_embedding_model()

    vectordb_instance = Chroma(
        persist_directory=str(vectordb_dir),
        collection_name=collection_name,
        embedding_function=embedding_model,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )

    duplicado, old_ids = _detectar_duplicata(
        vectordb_instance, planned_path, hash_str, safe_name
    )
    if duplicado is not None:
        return duplicado

    saved_path, backup_path, contexto = _gravar_arquivo_no_disco(
        conteudo, planned_path, old_ids, tmp_suffix, preparar_no_tmp
    )

    num_chunks = _indexar_no_chroma(
        vectordb_instance, saved_path, backup_path, contexto,
        construir_docs=construir_docs, embedding_model=embedding_model,
        hash_str=hash_str, doc_type=doc_type, page_args=page_args,
        old_ids=old_ids, safe_name=safe_name,
    )
    if num_chunks == 0:
        return {
            "status": "adicionado",
            "mensagem": f"Arquivo {safe_name} não pôde ser indexado: 0 chunks gerados.",
        }

    _notificar_invalidacao(invalidation_msg)

    status_final = "substituido" if old_ids else "adicionado"
    mensagem_final = (
        f"Arquivo {safe_name} atualizado na base vetorial. {num_chunks} chunks substituídos."
        if old_ids
        else f"Arquivo {safe_name} adicionado com sucesso. {num_chunks} chunks inseridos."
    )
    return {"status": status_final, "mensagem": mensagem_final}


# --------------------------------------------------------------------------- #
# Orquestradores públicos.
# --------------------------------------------------------------------------- #
def adicionar_documento_incrementalmente(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    docs_dir: Path | None = None,
    vectordb_dir: Path | None = None,
    embedding_model=None,
) -> dict[str, str]:
    """Adiciona chunks de um novo arquivo JSON ao ChromaDB existente sem reconstrução.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
      collection_name — nome da coleção ChromaDB de destino.
      docs_dir/vectordb_dir/embedding_model — seams injetáveis (fallback ``config.*``).
    Saída: dict com chaves "status" e "mensagem".
    Lança RAGSecurityError em qualquer falha de validação de segurança.
    Lança Exception se a ingestão falhar após rollback.
    """
    _sanitizar_nome_colecao(collection_name)

    if Path(filename).suffix.lower() != ".json":
        raise RAGSecurityError("Apenas arquivos .json são aceitos.")

    max_bytes = MAX_JSON_FILE_SIZE_MB * 1024 * 1024
    if len(conteudo) > max_bytes:
        raise RAGSecurityError(f"Arquivo excede o limite de {MAX_JSON_FILE_SIZE_MB} MB.")

    docs_dir = DIR_DOCUMENTS if docs_dir is None else docs_dir
    vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir

    return _ingerir_arquivo_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=docs_dir, vectordb_dir=vectordb_dir, embedding_model=embedding_model,
        tmp_suffix=".json",
        preparar_no_tmp=_valida_json_sem_chaves_proibidas,
        construir_docs=_construir_docs_json,
        doc_type=METADATA_DOC_TYPE_JSON,
        page_args=(METADATA_PAGE_JSON_SENTINEL,),
        invalidation_msg="Não foi possível invalidar o singleton do agente: %s",
    )


def adicionar_pdf_incrementalmente(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    docs_dir: Path | None = None,
    vectordb_dir: Path | None = None,
    embedding_model=None,
) -> dict[str, str]:
    """Adiciona chunks de um PDF ao ChromaDB existente sem reconstrução completa (REC-02).

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo PDF (bytes).
      collection_name — nome da coleção ChromaDB de destino.
      docs_dir/vectordb_dir/embedding_model — seams injetáveis (fallback ``config.*``).
    Saída: dict com chaves "status" e "mensagem".
    Lança RAGSecurityError em qualquer falha de validação de segurança.
    Lança Exception se a ingestão falhar após rollback.
    """
    _sanitizar_nome_colecao(collection_name)

    if Path(filename).suffix.lower() != ".pdf":
        raise RAGSecurityError("Apenas arquivos .pdf são aceitos.")

    if conteudo[:4] != b"%PDF":
        raise RAGSecurityError("Conteúdo não é um arquivo PDF válido.")

    max_bytes = MAX_DOCUMENT_FILE_SIZE_MB * 1024 * 1024
    if len(conteudo) > max_bytes:
        raise RAGSecurityError(
            f"Arquivo excede o limite de {MAX_DOCUMENT_FILE_SIZE_MB} MB."
        )

    docs_dir = DIR_DOCUMENTS if docs_dir is None else docs_dir
    vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir

    return _ingerir_arquivo_incrementalmente(
        filename, conteudo, collection_name,
        docs_dir=docs_dir, vectordb_dir=vectordb_dir, embedding_model=embedding_model,
        tmp_suffix=".pdf",
        preparar_no_tmp=extrair_texto_pdf,
        construir_docs=_construir_docs_pdf,
        doc_type=METADATA_DOC_TYPE_PDF,
        page_args=(),
        invalidation_msg="Não foi possível invalidar o singleton: %s",
    )


def _enriquecer_por_source(chunks: list, doc_type: str, *page_args) -> None:
    """Agrupa chunks por ``source`` e enriquece cada grupo com o hash do arquivo (REC-01)."""
    por_source: dict[str, list] = {}
    for chunk in chunks:
        por_source.setdefault(chunk.metadata.get("source", ""), []).append(chunk)
    for src, group in por_source.items():
        fh = _hash_arquivo(src)
        _enriquecer_metadados_chunks(group, fh, doc_type, *page_args)


def cria_vectordb(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    docs_dir: Path | None = None,
    vectordb_dir: Path | None = None,
    embedding_model=None,
) -> "Chroma | None":
    """Cria e persiste o banco vetorial ChromaDB a partir dos documentos em docs_dir.

    Itera ``carregar_json`` sobre ``sorted(docs_dir.glob("*.json"))`` (carga por-arquivo,
    ADR-018 Fase 3b / ING3B-05) + ``extrair_texto_pdf`` sobre os PDFs, divide em chunks
    (SemanticChunker, ADR-013), reseta a coleção e persiste. Guardrail fail-loud: aborta
    com RuntimeError se a coleção persistir 0 chunks.

    Entrada:
      collection_name — coleção ChromaDB alvo.
      docs_dir/vectordb_dir/embedding_model — seams injetáveis (fallback ``config.*``;
        ``embedding_model=None`` constrói um HuggingFaceEmbeddings fresco — caminho de rebuild).
    Saída: instância Chroma persistida, ou None quando não há documentos.
    Lança RuntimeError se o rebuild persistir 0 chunks.
    """
    _sanitizar_nome_colecao(collection_name)
    docs_dir = DIR_DOCUMENTS if docs_dir is None else docs_dir
    vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir

    _valida_path_documentos()
    _valida_arquivos_json()

    logger.info("Gerando as Embeddings. Aguarde...")

    # jq_schema compartilhado (config.py): 1 Document por chave top-level (ADR-008).
    # sorted() garante ordem determinística por-arquivo (ING3B-05).
    json_docs: list = []
    for json_path in sorted(docs_dir.glob("*.json")):
        json_docs.extend(carregar_json(json_path))
    # Descarta Documents com page_content vazio (ADR-011).
    json_docs = filtrar_documentos_vazios(json_docs)

    pdf_docs = []
    for pdf_path in docs_dir.glob("*.pdf"):
        try:
            paginas = extrair_texto_pdf(pdf_path)
            for chave, texto in paginas.items():
                page_num = int(chave.split("_")[1])
                pdf_docs.append(
                    Document(
                        page_content=f"{chave}: {texto}",
                        metadata={"source": str(pdf_path), METADATA_PAGE: page_num},
                    )
                )
        except RAGSecurityError as e:
            logger.error("PDF corrompido ignorado durante reconstrução: %s — %s", pdf_path.name, e)

    # Descarta Documents com page_content vazio — simetria com o filtro JSON (ADR-011).
    pdf_docs = filtrar_documentos_vazios(pdf_docs)

    documents = json_docs + pdf_docs

    if not documents:
        logger.warning("Nenhum documento encontrado.")
        return None

    if embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
    )

    json_chunks = text_splitter.split_documents(json_docs)
    _enriquecer_por_source(json_chunks, METADATA_DOC_TYPE_JSON, METADATA_PAGE_JSON_SENTINEL)

    pdf_chunks = text_splitter.split_documents(pdf_docs)
    _enriquecer_por_source(pdf_chunks, METADATA_DOC_TYPE_PDF)

    chunks = json_chunks + pdf_chunks

    # Reconstrução do zero: descarta a coleção antiga antes de gravar, senão
    # from_documents anexa e duplica o índice a cada --rebuild.
    _resetar_colecao(collection_name, vectordb_dir=vectordb_dir)

    vectordb_instance = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=str(vectordb_dir),
        collection_name=collection_name,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )

    # Guardrail fail-loud: o rebuild já deixou a coleção ativa vazia no passado (órfãos).
    persistidos = vectordb_instance._collection.count()
    if persistidos == 0:
        raise RuntimeError(
            f"Rebuild gerou coleção vazia ('{collection_name}'): {len(chunks)} chunks "
            "preparados, 0 persistidos. Vector DB não confiável — verifique o ChromaDB."
        )

    logger.info(
        "Banco de Dados Vetorial Criado com sucesso! %s chunks na coleção '%s'.",
        persistidos, collection_name,
    )
    return vectordb_instance


def reconstruir_vectordb(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    docs_dir: Path | None = None,
    vectordb_dir: Path | None = None,
    embedding_model=None,
) -> None:
    """Reconstrói o banco vetorial ChromaDB a partir dos documentos em docs_dir.

    Entrada: collection_name + seams injetáveis (fallback ``config.*``).
    Saída: nenhuma (efeito colateral: atualiza vectordb_dir).
    Lança Exception se cria_vectordb() falhar.
    """
    _sanitizar_nome_colecao(collection_name)
    cria_vectordb(
        collection_name,
        docs_dir=docs_dir,
        vectordb_dir=vectordb_dir,
        embedding_model=embedding_model,
    )


def ingerir_incrementalmente(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    docs_dir: Path | None = None,
    vectordb_dir: Path | None = None,
    embedding_model=None,
) -> dict[str, int]:
    """Ingestão incremental de todos os arquivos em docs_dir sem reconstrução (REC-04).

    Itera *.json e *.pdf, despacha para os dois `adicionar_*` (propagando os seams),
    fail-fast em RAGSecurityError (aborta o lote) e conta 'erro' em falha operacional
    comum sem abortar os demais.

    Entrada: collection_name + seams injetáveis (fallback ``config.*``).
    Saída: dict de contadores agregados por status final.
    Lança RAGSecurityError se algum arquivo falhar na validação de segurança.
    """
    docs_dir = DIR_DOCUMENTS if docs_dir is None else docs_dir
    vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir

    contadores: dict[str, int] = {}
    arquivos = sorted(docs_dir.glob("*.json")) + sorted(docs_dir.glob("*.pdf"))
    for path in arquivos:
        try:
            conteudo = path.read_bytes()
            if path.suffix.lower() == ".json":
                resultado = adicionar_documento_incrementalmente(
                    path.name, conteudo, collection_name,
                    docs_dir=docs_dir, vectordb_dir=vectordb_dir, embedding_model=embedding_model,
                )
            else:
                resultado = adicionar_pdf_incrementalmente(
                    path.name, conteudo, collection_name,
                    docs_dir=docs_dir, vectordb_dir=vectordb_dir, embedding_model=embedding_model,
                )
        except RAGSecurityError:
            # Violação de segurança não é "arquivo ruim" — propaga para abortar o lote.
            logger.error("Violação de segurança ao ingerir %s — abortando lote.", path.name)
            raise
        except Exception as e:  # um arquivo ruim não deve abortar o lote
            logger.error("Falha ao ingerir %s: %s", path.name, e, exc_info=True)
            contadores["erro"] = contadores.get("erro", 0) + 1
            continue
        status = resultado["status"]
        contadores[status] = contadores.get(status, 0) + 1
        logger.info(resultado["mensagem"])

    logger.info("Ingestão incremental concluída: %s", contadores)
    return contadores
