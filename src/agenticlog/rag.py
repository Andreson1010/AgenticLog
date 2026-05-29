# AgenticLog - Pipeline RAG
"""
Pipeline de construção do banco vetorial (ChromaDB).

Responsabilidades:
- Validar segurança dos documentos JSON antes do carregamento (path traversal, chaves proibidas, tamanho).
- Transformar os documentos em chunks e gerar embeddings com HuggingFace.
- Persistir o banco vetorial em data/vectordb/ para uso pelo agente.

Execute: python -m agenticlog.rag
"""

import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch

from agenticlog.config import (
    PROJECT_ROOT,
    DIR_DOCUMENTS,
    DIR_VECTORDB,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_JSON_FILES,
    MAX_JSON_FILE_SIZE_MB,
    FORBIDDEN_JSON_KEYS,
    LOG_LEVEL,
    LOG_FORMAT,
    _JsonFormatter,
)

logger = logging.getLogger(__name__)

vectordb = None


class RAGSecurityError(Exception):
    """Exceção lançada quando uma violação de segurança é detectada no pipeline RAG.

    Exemplos de violações: path traversal, chaves JSON proibidas, arquivo muito grande.
    """


def _valida_path_documentos() -> None:
    """Verifica que DIR_DOCUMENTS está contido dentro de PROJECT_ROOT.

    Mitiga path traversal: impede que um valor manipulado de DIR_DOCUMENTS aponte para
    diretórios fora do projeto (ex.: /etc/ ou C:\\Windows\\), evitando leitura indevida
    de arquivos do sistema operacional.
    """
    dir_resolved = DIR_DOCUMENTS.resolve()
    root_resolved = PROJECT_ROOT.resolve()
    try:
        dir_resolved.relative_to(root_resolved)
    except ValueError:
        raise RAGSecurityError(
            f"Diretório de documentos fora do projeto: {DIR_DOCUMENTS}"
        )
    if not dir_resolved.exists():
        raise RAGSecurityError(f"Diretório não existe: {DIR_DOCUMENTS}")
    if not dir_resolved.is_dir():
        raise RAGSecurityError(f"Caminho não é um diretório: {DIR_DOCUMENTS}")


def _valida_json_sem_chaves_proibidas(file_path: Path) -> None:
    """Rejeita arquivos JSON que contenham chaves listadas em FORBIDDEN_JSON_KEYS.

    Mitiga injeção de serialização: a chave "lc" é usada internamente pelo LangChain
    para desserializar objetos arbitrários via Serializable. Um documento malicioso com
    essa chave poderia forçar a execução de código inesperado ao ser carregado pelo loader.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RAGSecurityError(f"JSON inválido em {file_path}: {e}") from e
    except OSError as e:
        raise RAGSecurityError(f"Erro ao ler {file_path}: {e}") from e
    if isinstance(data, dict):
        for key in FORBIDDEN_JSON_KEYS:
            if key in data:
                raise RAGSecurityError(
                    f"Arquivo contém chave proibida '{key}': {file_path}"
                )
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                for key in FORBIDDEN_JSON_KEYS:
                    if key in item:
                        raise RAGSecurityError(
                            f"Arquivo contém chave proibida '{key}' no item {i}: {file_path}"
                        )


def _valida_arquivos_json() -> None:
    """Valida todos os arquivos JSON em DIR_DOCUMENTS antes do carregamento no ChromaDB.

    Verificações realizadas:
    - Contagem: rejeita se o número de arquivos exceder MAX_JSON_FILES (proteção contra DoS).
    - Tamanho: rejeita arquivos maiores que MAX_JSON_FILE_SIZE_MB (evita consumo excessivo de memória).
    - Conteúdo: delega a _valida_json_sem_chaves_proibidas para checar chaves proibidas e JSON válido.
    """
    max_bytes = MAX_JSON_FILE_SIZE_MB * 1024 * 1024
    json_files = list(DIR_DOCUMENTS.glob("*.json"))

    if len(json_files) > MAX_JSON_FILES:
        raise RAGSecurityError(
            f"Excesso de arquivos: {len(json_files)} > {MAX_JSON_FILES}"
        )

    for path in json_files:
        try:
            size = path.stat().st_size
        except OSError as e:
            raise RAGSecurityError(f"Erro ao acessar {path.name}: {e}") from e
        if size > max_bytes:
            raise RAGSecurityError(
                f"Arquivo excede {MAX_JSON_FILE_SIZE_MB}MB: {path.name} ({size / (1024*1024):.1f}MB)"
            )
        _valida_json_sem_chaves_proibidas(path)


def cria_vectordb():
    """Cria e persiste o banco vetorial ChromaDB a partir dos documentos em data/documents/.

    Efeito colateral: atribui a variável global `vectordb` com a instância Chroma criada,
    tornando-a disponível para outros módulos que importem este arquivo.

    Fluxo:
    1. Valida segurança dos paths e arquivos JSON.
    2. Carrega documentos com JSONLoader usando jq_schema para achatar chave-valor.
    3. Divide em chunks com RecursiveCharacterTextSplitter.
    4. Gera embeddings com HuggingFace e persiste no ChromaDB.
    """
    global vectordb  # inicializado como None no nível do módulo; preenchido aqui

    _valida_path_documentos()
    _valida_arquivos_json()

    logger.info("Gerando as Embeddings. Aguarde...")

    # jq_schema: achata o JSON em "chave: valor\nchave: valor" para facilitar chunking e busca semântica
    jq_schema = 'to_entries | map(.key + ": " + .value) | join("\\n")'
    loader = DirectoryLoader(
        str(DIR_DOCUMENTS),
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": jq_schema},
    )
    documents = loader.load()

    if not documents:
        logger.warning("Nenhum documento encontrado.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=str(DIR_VECTORDB),
    )

    logger.info("Banco de Dados Vetorial Criado com sucesso!")


def _executar_main() -> None:
    """Ponto de entrada CLI — configura logging e invoca cria_vectordb."""
    pkg_logger = logging.getLogger("agenticlog")
    pkg_logger.setLevel(LOG_LEVEL)
    # clear existing handlers to avoid duplicates on repeated calls
    pkg_logger.handlers.clear()

    if LOG_FORMAT == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        pkg_logger.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        pkg_logger.addHandler(handler)

    try:
        cria_vectordb()
    except RAGSecurityError as e:
        logger.error("Erro de segurança: %s", e)
        raise SystemExit(1) from e
    except Exception as e:
        logger.error("Erro ao criar banco vetorial: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    _executar_main()
