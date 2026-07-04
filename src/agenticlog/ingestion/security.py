# AgenticLog - Estágio de segurança da ingestão
"""Validação, sanitização e persistência segura de uploads (ADR-018 Fase 3a).

Funções movidas verbatim de `agenticlog.rag`. Importa de `config`, `shared` e,
intra-pacote, de `ingestion.extraction` (aresta `security → extraction`, não circular).
"""

import json
import shutil
import tempfile
from pathlib import Path

from agenticlog.config import (
    COLLECTION_NAME_MAX_LEN,
    COLLECTION_NAME_MIN_LEN,
    COLLECTION_NAME_PATTERN,
    DEFAULT_COLLECTION_NAME,
    DIR_DOCUMENTS,
    FORBIDDEN_JSON_KEYS,
    INVALID_FILENAME_CHARS,
    MAX_DOCUMENT_FILE_SIZE_MB,
    MAX_JSON_FILE_SIZE_MB,
    MAX_JSON_FILES,
    PROJECT_ROOT,
    WINDOWS_RESERVED_NAMES,
)
from agenticlog.ingestion.extraction import extrair_texto_pdf
from agenticlog.shared.errors import RAGSecurityError


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
        extrair_texto_pdf(tmp_path)  # validação por efeito colateral: levanta RAGSecurityError se sem texto
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
