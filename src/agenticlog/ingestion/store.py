# AgenticLog - Camada de persistência/atomicidade da ingestão
"""Escrita/rollback no Chroma e mutação de disco/coleção (ADR-018 Fase 3b).

Movido de `agenticlog.rag` preservando comportamento byte-idêntico. É o DONO das
escritas no Chroma: `add_documents_com_rollback` concentra a atomicidade de escrita
dos NOVOS chunks. As primitivas de disco (`_backup_arquivo`, `_reverter_disco`) e de
coleção (`_outras_colecoes_existem`, `_resetar_colecao`) vivem aqui também.

Importa apenas de `agenticlog.config` (folha do grafo) e stdlib; NUNCA importa
`rag`/`agent`/`orchestrator`, mantendo o pacote `ingestion` acíclico. As primitivas
de coleção são parametrizadas por `vectordb_dir` (resolvido no corpo, §4.2 do design),
permitindo injeção de dependência sem leitura de global de módulo em tempo de import.
"""

import logging
import shutil
import tempfile
from pathlib import Path

from agenticlog.config import DIR_VECTORDB

# Loga sob "agenticlog.rag" (não __name__): preserva saída byte-idêntica dos registros
# (campo `logger`) e mantém os `assertLogs("agenticlog.rag")` existentes verdes durante
# a fase de shims/wrappers (ADR-018 Fase 3b — revisto na Fase 6).
logger = logging.getLogger("agenticlog.rag")


def _backup_arquivo(path: Path) -> Path:
    """Cria cópia temporária de ``path`` para restauração em caso de falha de upsert."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bak") as bak:
        backup_path = Path(bak.name)
    shutil.copy2(path, backup_path)
    return backup_path


def _reverter_disco(saved_path: Path, backup_path: Path | None) -> None:
    """Reverte o estado do disco após falha de ingestão.

    Upsert (``backup_path`` não-nulo): restaura o conteúdo antigo, mantendo o disco
    consistente com os chunks antigos que permanecem no Chroma — evita perda do
    documento original.
    Arquivo novo (``backup_path`` nulo): remove o arquivo recém-gravado.
    """
    if backup_path is not None:
        shutil.move(str(backup_path), saved_path)
    else:
        saved_path.unlink(missing_ok=True)


def _outras_colecoes_existem(
    collection_name: str, vectordb_dir: Path | None = None
) -> bool:
    """Retorna True se o vector DB contém alguma coleção DIFERENTE de ``collection_name``.

    Lê a tabela ``collections`` do SQLite do Chroma em modo read-only — sem abrir um
    cliente Chroma, que seguraria um lock sobre o arquivo e faria o ``rmtree`` falhar no
    Windows. Schema ausente/ilegível ou DB inexistente é tratado como "sem coleções irmãs"
    (False) — o caminho seguro de wipe completo (que purga órfãos).

    Entrada:
      collection_name — coleção alvo do rebuild.
      vectordb_dir — diretório do vector DB (fallback ``config.DIR_VECTORDB`` quando None).
    Saída: True se há ao menos uma coleção com nome diferente; False caso contrário.
    """
    vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
    db_file = vectordb_dir / "chroma.sqlite3"
    if not db_file.exists():
        return False
    import sqlite3  # lazy — leitura pontual, sem lock do cliente Chroma

    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        nomes = [row[0] for row in con.execute("SELECT name FROM collections").fetchall()]
    except sqlite3.Error:
        return False
    finally:
        con.close()
    return any(nome != collection_name for nome in nomes)


def _resetar_colecao(collection_name: str, vectordb_dir: Path | None = None) -> None:
    """Descarta o estado da coleção alvo antes do rebuild, sem destruir coleções irmãs.

    Caso comum (a coleção alvo é a única, ou o DB ainda não existe): remove
    ``vectordb_dir`` inteiro — isso purga os segmentos/embeddings ÓRFÃOS que o
    ``delete_collection`` do Chroma deixa para trás (causa-raiz da coleção vazia / RAG
    silenciosamente offline). Caso multi-coleção: descarta apenas a coleção alvo via
    ``delete_collection``, preservando as demais; a integridade do rebuild é garantida
    pelo guardrail de contagem em ``cria_vectordb`` (aborta se persistir 0 chunks).
    Diretório/coleção inexistente é no-op.

    Entrada:
      collection_name — coleção ChromaDB alvo do rebuild.
      vectordb_dir — diretório do vector DB (fallback ``config.DIR_VECTORDB`` quando None).
    Saída: nenhuma — efeito colateral: estado da coleção alvo removido do disco.
    """
    vectordb_dir = DIR_VECTORDB if vectordb_dir is None else vectordb_dir
    if not _outras_colecoes_existem(collection_name, vectordb_dir=vectordb_dir):
        if vectordb_dir.exists():
            shutil.rmtree(vectordb_dir, ignore_errors=True)
            logger.info(
                "Diretório do vector DB removido para rebuild limpo (coleção '%s'): %s",
                collection_name, vectordb_dir,
            )
        return

    import chromadb  # lazy — evita side-effects na importação do módulo

    client = chromadb.PersistentClient(path=str(vectordb_dir))
    try:
        client.delete_collection(collection_name)
        logger.info(
            "Coleção '%s' descartada (coleções irmãs preservadas no vector DB).",
            collection_name,
        )
    except Exception as exc:  # coleção inexistente → nada a descartar
        logger.debug(
            "Coleção '%s' não descartada (provavelmente inexistente): %s",
            collection_name, exc,
        )


def add_documents_com_rollback(vectordb_instance, chunks, chunk_ids) -> None:
    """Escreve chunks no Chroma com rollback best-effort na falha (ADR-018 Fase 3b, D1).

    Entrada:
      vectordb_instance — instância Chroma (dona da escrita).
      chunks — list[Document] a persistir.
      chunk_ids — list[str] de IDs correspondentes.
    Saída: nenhuma. Efeito: chunks persistidos OU nenhum efeito líquido (rollback).
    Lança: re-levanta a exceção ORIGINAL de ``add_documents`` após tentar o ``delete``;
      se o próprio rollback falhar, loga os IDs órfãos e ainda re-levanta a original.
    """
    try:
        vectordb_instance.add_documents(chunks, ids=chunk_ids)
    except Exception:
        try:
            vectordb_instance.delete(ids=chunk_ids)
        except Exception as rollback_exc:
            logger.warning(
                "Rollback falhou após erro de ingestão. IDs órfãos: %s. Erro de rollback: %s",
                chunk_ids,
                rollback_exc,
            )
        raise
