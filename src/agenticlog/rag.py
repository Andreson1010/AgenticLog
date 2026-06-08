# AgenticLog - Pipeline RAG
"""
Pipeline de construção do banco vetorial (ChromaDB).

Responsabilidades:
- Validar segurança dos documentos JSON antes do carregamento (path traversal, chaves proibidas, tamanho).
- Transformar os documentos em chunks e gerar embeddings com HuggingFace.
- Persistir o banco vetorial em data/vectordb/ para uso pelo agente.

Execute: python -m agenticlog.rag
"""

import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document
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
    MAX_DOCUMENT_FILE_SIZE_MB,
    FORBIDDEN_JSON_KEYS,
    INVALID_FILENAME_CHARS,
    WINDOWS_RESERVED_NAMES,
    LOG_LEVEL,
    LOG_FORMAT,
    _JsonFormatter,
    DEFAULT_COLLECTION_NAME,
    COLLECTION_NAME_MIN_LEN,
    COLLECTION_NAME_MAX_LEN,
    COLLECTION_NAME_PATTERN,
)

logger = logging.getLogger(__name__)

vectordb = None
_rag_embedding_model = None


def _get_rag_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna singleton de HuggingFaceEmbeddings para ingestão incremental.

    Entrada: nenhuma.
    Saída: instância de HuggingFaceEmbeddings (criada uma única vez por processo).
    """
    global _rag_embedding_model
    if _rag_embedding_model is None:
        _rag_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _rag_embedding_model


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


def _sanitizar_nome_colecao(name: str) -> str:
    """Valida e retorna o nome de coleção ChromaDB fornecido.

    Entrada: name — nome da coleção como string.
    Saída: name inalterado se válido.
    Lança RAGSecurityError se o nome for vazio, muito curto, muito longo ou
    não corresponder ao padrão de nomes válidos do ChromaDB.
    """
    if not name:
        raise RAGSecurityError("Nome de coleção vazio.")
    if len(name) < COLLECTION_NAME_MIN_LEN:
        raise RAGSecurityError(
            f"Nome de coleção muito curto: mínimo {COLLECTION_NAME_MIN_LEN} caracteres."
        )
    if len(name) > COLLECTION_NAME_MAX_LEN:
        raise RAGSecurityError(
            f"Nome de coleção muito longo: máximo {COLLECTION_NAME_MAX_LEN} caracteres."
        )
    if not COLLECTION_NAME_PATTERN.match(name):
        raise RAGSecurityError(
            "Nome de coleção inválido: use apenas letras, números, hífen e underscore, "
            "começando e terminando com alfanumérico."
        )
    return name


def sanitizar_nome_colecao(name: str) -> str:
    """Valida nome de coleção ChromaDB. Levanta RAGSecurityError se inválido."""
    return _sanitizar_nome_colecao(name)


def salvar_documento_enviado(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Path:
    """Valida e persiste um arquivo JSON enviado pelo operador.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
      collection_name — nome da coleção ChromaDB de destino.
    Saída: Path do arquivo salvo em DIR_DOCUMENTS.
    Lança RAGSecurityError em qualquer falha de validação.
    """
    _sanitizar_nome_colecao(collection_name)

    if Path(filename).suffix.lower() != ".json":
        raise RAGSecurityError("Apenas arquivos .json são aceitos.")

    if len(conteudo) > MAX_DOCUMENT_FILE_SIZE_MB * 1024 * 1024:
        raise RAGSecurityError(
            f"Arquivo excede o limite de {MAX_DOCUMENT_FILE_SIZE_MB} MB."
        )

    safe_name = _sanitizar_nome_arquivo(filename)

    if (DIR_DOCUMENTS / safe_name).exists():
        raise RAGSecurityError("Arquivo com esse nome já existe.")

    pdf_count = len(list(DIR_DOCUMENTS.glob("*.pdf")))
    json_count = len(list(DIR_DOCUMENTS.glob("*.json")))
    if pdf_count + json_count + 1 > MAX_JSON_FILES:
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


def _computar_hash_conteudo(conteudo: bytes) -> str:
    """Computa o hash SHA-256 do conteúdo binário do arquivo.

    Entrada: conteudo — bytes do arquivo.
    Saída: string hexadecimal de 64 caracteres (SHA-256).
    """
    return hashlib.sha256(conteudo).hexdigest()


def adicionar_documento_incrementalmente(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict[str, str]:
    """Adiciona chunks de um novo arquivo JSON ao ChromaDB existente sem reconstrução.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
      collection_name — nome da coleção ChromaDB de destino.
    Saída: dict com chaves "status" e "mensagem":
      {"status": "adicionado", "mensagem": "Arquivo <nome> adicionado com sucesso. N chunks inseridos."}
      {"status": "duplicado", "mensagem": "Arquivo <nome> já está presente na base vetorial."}
      {"status": "hash_diferente", "mensagem": "Arquivo <nome> já existe com conteúdo diferente. Remoção e substituição não são suportadas nesta versão."}
    Lança RAGSecurityError em qualquer falha de validação de segurança.
    Lança Exception se a ingestão falhar após rollback.
    """
    _sanitizar_nome_colecao(collection_name)

    if Path(filename).suffix.lower() != ".json":
        raise RAGSecurityError("Apenas arquivos .json são aceitos.")

    max_bytes = MAX_JSON_FILE_SIZE_MB * 1024 * 1024
    if len(conteudo) > max_bytes:
        raise RAGSecurityError(
            f"Arquivo excede o limite de {MAX_JSON_FILE_SIZE_MB} MB."
        )

    safe_name = _sanitizar_nome_arquivo(filename)

    json_count = len(list(DIR_DOCUMENTS.glob("*.json")))
    pdf_count = len(list(DIR_DOCUMENTS.glob("*.pdf")))
    if json_count + pdf_count + 1 > MAX_JSON_FILES:
        raise RAGSecurityError(
            f"Limite de {MAX_JSON_FILES} arquivos atingido."
        )

    hash_str = _computar_hash_conteudo(conteudo)
    planned_path = DIR_DOCUMENTS / safe_name

    embedding_model = _get_rag_embedding_model()
    vectordb_instance = Chroma(
        persist_directory=str(DIR_VECTORDB),
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    existing = vectordb_instance.get(
        where={"source": {"$eq": str(planned_path)}}
    )
    if existing["ids"]:
        existing_hash = existing["metadatas"][0].get("content_hash")
        if existing_hash == hash_str:
            return {
                "status": "duplicado",
                "mensagem": f"Arquivo {safe_name} já está presente na base vetorial.",
            }
        return {
            "status": "hash_diferente",
            "mensagem": (
                f"Arquivo {safe_name} já existe com conteúdo diferente. "
                "Remoção e substituição não são suportadas nesta versão."
            ),
        }

    saved_path: Path | None = None
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(conteudo)
            tmp_path = Path(tmp.name)
        _valida_json_sem_chaves_proibidas(tmp_path)
        shutil.move(str(tmp_path), planned_path)
        saved_path = planned_path
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    jq_schema = 'to_entries | map(.key + ": " + (.value | tostring)) | join("\\n")'
    loader = JSONLoader(str(saved_path), jq_schema=jq_schema)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        logger.warning("Arquivo %s produziu zero chunks após divisão.", safe_name)
        saved_path.unlink(missing_ok=True)
        return {
            "status": "adicionado",
            "mensagem": f"Arquivo {safe_name} não pôde ser indexado: 0 chunks gerados.",
        }

    for chunk in chunks:
        chunk.metadata["content_hash"] = hash_str

    chunk_ids = [uuid.uuid4().hex for _ in chunks]

    try:
        vectordb_instance.add_documents(chunks, ids=chunk_ids)
    except Exception as ingestion_exc:
        try:
            if chunk_ids:
                vectordb_instance.delete(ids=chunk_ids)
        except Exception as rollback_exc:
            logger.warning(
                "Rollback falhou após erro de ingestão. IDs órfãos: %s. Erro de rollback: %s",
                chunk_ids,
                rollback_exc,
            )
        saved_path.unlink(missing_ok=True)
        raise ingestion_exc

    try:
        from agenticlog.agent import invalidar_vector_db  # lazy — evita importação pesada no CLI
        invalidar_vector_db()
    except ImportError as e:
        logger.warning("Não foi possível invalidar o singleton do agente: %s", e)

    return {
        "status": "adicionado",
        "mensagem": f"Arquivo {safe_name} adicionado com sucesso. {len(chunks)} chunks inseridos.",
    }


def extrair_texto_pdf(path: Path) -> str:
    """Extrai texto de um arquivo PDF usando PyMuPDF (fitz).

    Entrada: path — Path para um arquivo PDF já salvo em disco.
    Saída: string com todo o texto extraível (páginas concatenadas).
    Lança RAGSecurityError se:
      - fitz.open() lança qualquer Exception (arquivo corrompido).
      - doc.needs_pass == True (PDF protegido por senha).
      - Texto extraído tem zero caracteres não-brancos (PDF somente-imagem).
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
        texto = "".join(page.get_text() for page in doc_handle)

    if not texto.strip():
        raise RAGSecurityError("PDF não contém texto extraível (somente imagem).")

    return texto


def salvar_pdf_enviado(
    filename: str,
    conteudo: bytes,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Path:
    """Valida e persiste um arquivo PDF enviado pelo operador.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
      collection_name — nome da coleção ChromaDB de destino.
    Saída: Path do arquivo salvo em DIR_DOCUMENTS.
    Lança RAGSecurityError em qualquer falha de validação.
    """
    _sanitizar_nome_colecao(collection_name)

    if Path(filename).suffix.lower() != ".pdf":
        raise RAGSecurityError("Apenas arquivos .pdf são aceitos.")

    if not conteudo.startswith(b"%PDF"):
        raise RAGSecurityError("Conteúdo não é um arquivo PDF válido.")

    if len(conteudo) > MAX_DOCUMENT_FILE_SIZE_MB * 1024 * 1024:
        raise RAGSecurityError(
            f"Arquivo excede o limite de {MAX_DOCUMENT_FILE_SIZE_MB} MB."
        )

    safe_name = _sanitizar_nome_arquivo(filename)

    if (DIR_DOCUMENTS / safe_name).exists():
        raise RAGSecurityError("Arquivo com esse nome já existe.")

    pdf_count = len(list(DIR_DOCUMENTS.glob("*.pdf")))
    json_count = len(list(DIR_DOCUMENTS.glob("*.json")))
    if pdf_count + json_count + 1 > MAX_JSON_FILES:
        raise RAGSecurityError(
            f"Limite de {MAX_JSON_FILES} arquivos atingido."
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(conteudo)
            tmp_path = Path(tmp.name)
        extrair_texto_pdf(tmp_path)
        shutil.move(str(tmp_path), DIR_DOCUMENTS / safe_name)
        tmp_path = None  # moved — no cleanup needed
    except RAGSecurityError:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
            tmp_path = None
        raise
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return DIR_DOCUMENTS / safe_name


def reconstruir_vectordb(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """Reconstrói o banco vetorial ChromaDB a partir dos documentos em DIR_DOCUMENTS.

    Entrada: collection_name — nome da coleção ChromaDB a reconstruir.
    Saída: nenhuma (efeito colateral: atualiza data/vectordb/).
    Lança Exception se cria_vectordb() falhar.
    """
    _sanitizar_nome_colecao(collection_name)
    cria_vectordb(collection_name)


def cria_vectordb(collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """Cria e persiste o banco vetorial ChromaDB a partir dos documentos em data/documents/.

    Efeito colateral: atribui a variável global `vectordb` com a instância Chroma criada,
    tornando-a disponível para outros módulos que importem este arquivo.

    Fluxo:
    1. Valida segurança dos paths e arquivos JSON.
    2. Carrega documentos com JSONLoader usando jq_schema para achatar chave-valor.
    3. Divide em chunks com RecursiveCharacterTextSplitter.
    4. Gera embeddings com HuggingFace e persiste no ChromaDB.
    """
    _sanitizar_nome_colecao(collection_name)
    global vectordb  # inicializado como None no nível do módulo; preenchido aqui

    _valida_path_documentos()
    _valida_arquivos_json()

    logger.info("Gerando as Embeddings. Aguarde...")

    # jq_schema: achata o JSON em "chave: valor\nchave: valor" para facilitar chunking e busca semântica
    jq_schema = 'to_entries | map(.key + ": " + (.value | tostring)) | join("\\n")'
    loader = DirectoryLoader(
        str(DIR_DOCUMENTS),
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": jq_schema},
    )
    json_docs = loader.load()

    pdf_docs = []
    for pdf_path in DIR_DOCUMENTS.glob("*.pdf"):
        try:
            texto = extrair_texto_pdf(pdf_path)
            pdf_docs.append(Document(page_content=texto, metadata={"source": str(pdf_path)}))
        except RAGSecurityError as e:
            logger.error("PDF corrompido ignorado durante reconstrução: %s — %s", pdf_path.name, e)

    documents = json_docs + pdf_docs

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
        collection_name=collection_name,
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
