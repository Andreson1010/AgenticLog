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
import shutil
import tempfile
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

INVALID_FILENAME_CHARS: frozenset[str] = frozenset('<>:"/\\|?*\x00')
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


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


def _sanitizar_nome_arquivo(filename: str) -> str:
    """Valida e retorna o basename seguro do filename fornecido.

    Entrada: filename — nome do arquivo como string (pode conter path).
    Saída: basename sanitizado.
    Lança RAGSecurityError se o nome for vazio, contiver caracteres inválidos
    ou indicar path traversal.
    """
    if not filename:
        raise RAGSecurityError("Nome de arquivo vazio.")
    if any(c in INVALID_FILENAME_CHARS for c in filename):
        raise RAGSecurityError(
            f"Nome de arquivo contém caracteres inválidos: {filename!r}"
        )
    basename = Path(filename).name
    if basename != filename or ".." in filename:
        raise RAGSecurityError(
            f"Nome de arquivo com path traversal detectado: {filename!r}"
        )
    stem = Path(basename).stem.upper()
    if stem in WINDOWS_RESERVED_NAMES:
        raise RAGSecurityError(
            f"Nome de arquivo reservado pelo Windows: {filename!r}"
        )
    return basename


def salvar_documento_enviado(filename: str, conteudo: bytes) -> Path:
    """Valida e persiste um arquivo JSON enviado pelo operador.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
    Saída: Path do arquivo salvo em DIR_DOCUMENTS.
    Lança RAGSecurityError em qualquer falha de validação.
    """
    if Path(filename).suffix.lower() != ".json":
        raise RAGSecurityError("Apenas arquivos .json são aceitos.")

    if len(conteudo) > MAX_JSON_FILE_SIZE_MB * 1024 * 1024:
        raise RAGSecurityError(
            f"Arquivo excede o limite de {MAX_JSON_FILE_SIZE_MB} MB."
        )

    safe_name = _sanitizar_nome_arquivo(filename)

    if (DIR_DOCUMENTS / safe_name).exists():
        raise RAGSecurityError("Arquivo com esse nome já existe.")

    if len(list(DIR_DOCUMENTS.glob("*.json"))) + 1 > MAX_JSON_FILES:
        raise RAGSecurityError(
            f"Limite de {MAX_JSON_FILES} arquivos atingido."
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(conteudo)
            tmp_path = Path(tmp.name)
        _valida_json_sem_chaves_proibidas(tmp_path)
        shutil.move(str(tmp_path), DIR_DOCUMENTS / safe_name)
        tmp_path = None  # moved — no cleanup needed
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return DIR_DOCUMENTS / safe_name


def reconstruir_vectordb() -> None:
    """Reconstrói o banco vetorial ChromaDB a partir dos documentos em DIR_DOCUMENTS.

    Entrada: nenhuma.
    Saída: nenhuma (efeito colateral: atualiza data/vectordb/).
    Lança Exception se cria_vectordb() falhar.
    """
    cria_vectordb()


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
