# Acceptance tests — RAG Ingestion Fase 3a (ADR-018 Fase 3a)
"""
Verifies every acceptance criterion from the approved user story:
  P1-AC1 — Package `agenticlog.ingestion` with 6 modules + `__init__.py` exists.
  P1-AC2 — `agenticlog.rag.X is agenticlog.ingestion.<mod>.X` (shim identity).
  P1-AC3 — Old import paths (`from agenticlog.rag import X`) still resolve.
  P1-AC4 — Oracle `tests/test_rag_caracterizacao.py` passes without modification.
  P1-AC5 — `import agenticlog.ingestion` in a cold interpreter exits 0 (no cycle).

Edge cases exercised alongside the ACs:
  EC-cleaning   — `filtrar_documentos_vazios` discards whitespace docs; returns new list.
  EC-embeddings — `criar_embedding_model` passes `normalize_embeddings=True` verbatim.
  EC-hashes     — hash helpers are deterministic.
  EC-double-shim — monkeypatch on `rag._rag_embedding_model` is honored by getter.
  EC-carregar-json — `carregar_json` uses `JQ_SCHEMA_CAMPOS_JSON` as jq_schema.

RAGING traceability: -01 (AC-01), -02/-03/-04/-05/-06/-07 (AC-01/AC-02),
-08 (AC-02), -09 (AC-01), -10 (AC-05), -11 (AC-04), -13 (edge cases).

Tests exercise the feature from the outside — through module-level imports and
the same interface a developer importing the package would use, never through
implementation internals.

Heavy-boundary mocks (HuggingFace model, JSONLoader) are used to avoid loading
the ~1 GB embedding model or hitting LMStudio in any test.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

# Ensure src/ is on path (mirrors conftest.py convention)
_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import agenticlog.config as config  # noqa: E402
import agenticlog.ingestion as ingestion_pkg  # noqa: E402
import agenticlog.ingestion.chunking as ingestion_chunking  # noqa: E402
import agenticlog.ingestion.cleaning as ingestion_cleaning  # noqa: E402
import agenticlog.ingestion.embeddings as ingestion_embeddings  # noqa: E402
import agenticlog.ingestion.extraction as ingestion_extraction  # noqa: E402
import agenticlog.ingestion.metadata as ingestion_metadata  # noqa: E402
import agenticlog.ingestion.security as ingestion_security  # noqa: E402
import agenticlog.rag as rag  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


# ---------------------------------------------------------------------------
# AC-01: Package structure — 6 modules + __init__.py exist
# RAGING-01
# ---------------------------------------------------------------------------


def test_ac01_pacote_ingestion_expoe_seis_modulos_e_init() -> None:
    """
    AC-01: WHEN o pacote é criado THEN o sistema SHALL expor
    `src/agenticlog/ingestion/{security,extraction,cleaning,chunking,embeddings,metadata}.py`
    + `__init__.py`.

    Verifies that every required module is importable and resides under the
    `agenticlog/ingestion/` directory (structural contract).
    """
    expected_modules = [
        ("security", ingestion_security),
        ("extraction", ingestion_extraction),
        ("cleaning", ingestion_cleaning),
        ("chunking", ingestion_chunking),
        ("embeddings", ingestion_embeddings),
        ("metadata", ingestion_metadata),
    ]
    for mod_name, mod in expected_modules:
        assert mod.__file__ is not None, f"ingestion.{mod_name} has no __file__"
        mod_path = Path(mod.__file__).resolve()
        assert "ingestion" in str(mod_path), (
            f"ingestion.{mod_name} does not live under an 'ingestion' directory: {mod_path}"
        )
        assert mod_path.name.replace(".pyc", ".py") == f"{mod_name}.py" or mod_path.stem == mod_name, (
            f"Unexpected file name for ingestion.{mod_name}: {mod_path.name}"
        )

    # __init__.py must also be resolvable as a proper package
    assert ingestion_pkg.__file__ is not None
    init_path = Path(ingestion_pkg.__file__).resolve()
    assert init_path.name in ("__init__.py", "__init__.pyc"), (
        f"ingestion package __file__ is not __init__: {init_path.name}"
    )


def test_ac01_each_module_exposes_its_core_symbol() -> None:
    """
    AC-01 (sanity): each ingestion module exports its primary symbol,
    confirming the modules are populated — not just importable empty files.
    """
    assert hasattr(ingestion_security, "salvar_documento_enviado")
    assert hasattr(ingestion_security, "sanitizar_nome_colecao")
    assert hasattr(ingestion_extraction, "extrair_texto_pdf")
    assert hasattr(ingestion_extraction, "carregar_json")
    assert hasattr(ingestion_cleaning, "filtrar_documentos_vazios")
    assert hasattr(ingestion_chunking, "SemanticChunker")
    assert hasattr(ingestion_embeddings, "criar_embedding_model")
    assert hasattr(ingestion_metadata, "_computar_hash_conteudo")
    assert hasattr(ingestion_metadata, "_hash_arquivo")
    assert hasattr(ingestion_metadata, "_enriquecer_metadados_chunks")


# ---------------------------------------------------------------------------
# AC-02: Shim identity — rag.X is ingestion.<mod>.X for every moved symbol
# RAGING-08
# ---------------------------------------------------------------------------

# All symbols moved in this phase, paired with their new home module.
_MOVED_SYMBOLS = [
    # security
    ("_valida_path_documentos", ingestion_security),
    ("_valida_json_sem_chaves_proibidas", ingestion_security),
    ("_valida_arquivos_json", ingestion_security),
    ("_sanitizar_nome_arquivo", ingestion_security),
    ("_sanitizar_nome_colecao", ingestion_security),
    ("sanitizar_nome_colecao", ingestion_security),
    ("salvar_documento_enviado", ingestion_security),
    ("salvar_pdf_enviado", ingestion_security),
    # extraction
    ("extrair_texto_pdf", ingestion_extraction),
    ("carregar_json", ingestion_extraction),
    # cleaning
    ("filtrar_documentos_vazios", ingestion_cleaning),
    # chunking
    ("SemanticChunker", ingestion_chunking),
    # embeddings
    ("criar_embedding_model", ingestion_embeddings),
    # metadata
    ("_computar_hash_conteudo", ingestion_metadata),
    ("_hash_arquivo", ingestion_metadata),
    ("_enriquecer_metadados_chunks", ingestion_metadata),
]


@pytest.mark.parametrize("symbol_name,source_module", _MOVED_SYMBOLS)
def test_ac02_identidade_shim_rag_X_is_ingestion_mod_X(
    symbol_name: str, source_module: object
) -> None:
    """
    AC-02: WHEN a symbol is moved THEN `agenticlog.rag.<symbol>` SHALL be the
    SAME object (`is`) as `agenticlog.ingestion.<mod>.<symbol>`.

    The shim in rag.py re-imports directly from ingestion.<mod> so there is only
    one object in memory; monkeypatching either side of the shim affects the other.
    """
    rag_obj = getattr(rag, symbol_name, None)
    assert rag_obj is not None, (
        f"agenticlog.rag.{symbol_name} not found — shim missing?"
    )
    ingestion_obj = getattr(source_module, symbol_name, None)
    assert ingestion_obj is not None, (
        f"{source_module.__name__}.{symbol_name} not found in source module"
    )
    assert rag_obj is ingestion_obj, (
        f"Identity broken for {symbol_name}: "
        f"rag.{symbol_name} is not {source_module.__name__}.{symbol_name}"
    )


# ---------------------------------------------------------------------------
# AC-03: Old import paths still resolve — backward compatibility
# RAGING-08
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol_name,_", _MOVED_SYMBOLS)
def test_ac03_import_antigo_from_rag_resolve_sem_importerror(
    symbol_name: str, _: object
) -> None:
    """
    AC-03: WHEN `from agenticlog.rag import X` is used THEN it SHALL resolve
    without ImportError (agenticlog.rag.X must exist).

    Downstream callers that reference symbols via `agenticlog.rag` must not break.
    """
    assert hasattr(rag, symbol_name), (
        f"`agenticlog.rag.{symbol_name}` is missing — old import paths broken"
    )


# ---------------------------------------------------------------------------
# AC-04: Oracle test_rag_caracterizacao.py passes without modification
# RAGING-11
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ac04_oraculo_caracterizacao_passa_sem_modificacao() -> None:
    """
    AC-04: WHEN the full suite runs on CI Linux THEN the oracle
    `tests/test_rag_caracterizacao.py` SHALL pass without any modification
    to that file.

    Runs the oracle in a subprocess so this test does not pollute the current
    process's ChromaDB state. Skipped automatically on Windows/SAC environments
    by the `@integration` marker (conftest.py).

    Also asserts that the oracle file itself has not been modified (zero diff
    against HEAD), satisfying the HARD constraint of RAGING-11.
    """
    # Gate: oracle file must not have been touched
    oracle_path = _root / "tests" / "test_rag_caracterizacao.py"
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", str(oracle_path)],
        capture_output=True,
        text=True,
        cwd=str(_root),
    )
    assert diff.stdout.strip() == "", (
        f"Oracle file has been modified (HARD constraint violation)!\n"
        f"diff:\n{diff.stdout}"
    )

    # Run the oracle suite
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "integration",
         "tests/test_rag_caracterizacao.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_root),
        env={**os.environ, "PYTHONPATH": _src + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    assert result.returncode == 0, (
        f"Oracle suite FAILED (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# AC-05: No import cycle — cold interpreter exits 0
# RAGING-10
# ---------------------------------------------------------------------------


def test_ac05_import_ingestion_interpretador_frio_sai_zero() -> None:
    """
    AC-05: WHEN `import agenticlog.ingestion` runs in a cold interpreter THEN
    it SHALL exit with code 0 and produce no 'circular'/'partially initialized'
    message in stderr.

    `ingestion/*` only imports from `agenticlog.config` and `agenticlog.shared`
    (DAG leaves); intra-package edge `security → extraction` is not circular.
    A subprocess guarantees a genuinely cold sys.modules.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = _src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", "import agenticlog.ingestion"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Cold `import agenticlog.ingestion` failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "circular" not in result.stderr.lower(), (
        f"Circular import detected in stderr: {result.stderr!r}"
    )
    assert "partially initialized" not in result.stderr.lower(), (
        f"Partially initialized module in stderr: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# AC-05 companion: ingestion/* does NOT import rag or agent (acyclicity guard)
# RAGING-10
# ---------------------------------------------------------------------------


def test_ac05_ingestion_modules_nao_importam_rag_nem_agent() -> None:
    """
    AC-05 (companion): WHEN any ingestion submodule is inspected THEN its source
    SHALL contain no import of `agenticlog.rag` or `agenticlog.agent`.

    Confirms the one-directional dependency: config → shared → ingestion → rag.
    """
    import inspect

    modules_to_check = [
        ingestion_security,
        ingestion_extraction,
        ingestion_cleaning,
        ingestion_chunking,
        ingestion_embeddings,
        ingestion_metadata,
    ]
    for mod in modules_to_check:
        source_lines = inspect.getsource(mod).splitlines()
        forbidden_imports = [
            line.strip()
            for line in source_lines
            if line.strip().startswith(("import ", "from "))
            and ("agenticlog.rag" in line or "agenticlog.agent" in line)
        ]
        assert forbidden_imports == [], (
            f"{mod.__name__} contains forbidden import(s) of rag/agent "
            f"(would create a cycle):\n  " + "\n  ".join(forbidden_imports)
        )


# ---------------------------------------------------------------------------
# EC-cleaning: filtrar_documentos_vazios drops empty/whitespace; returns new list
# RAGING-04
# ---------------------------------------------------------------------------


def test_ac01_ec_cleaning_descarta_page_content_vazio_e_whitespace() -> None:
    """
    Edge case (AC-01/RAGING-04): WHEN a Document has page_content that is empty
    or only whitespace THEN `filtrar_documentos_vazios` SHALL discard it.

    WHEN a Document has non-empty content (even 'CAMPO_VAZIO: ') THEN it SHALL
    be preserved (the filter applies .strip() only to detect truly empty content).

    Returns a NEW list (immutability — does not mutate the input).
    """
    from agenticlog.ingestion.cleaning import filtrar_documentos_vazios

    vazio = Document(page_content="")
    whitespace = Document(page_content="   \n\t  ")
    normal = Document(page_content="carga de dados operacional")
    campo_vazio_label = Document(page_content="CAMPO_VAZIO: ")  # .strip() = "CAMPO_VAZIO:" → kept

    entrada = [vazio, whitespace, normal, campo_vazio_label]
    resultado = filtrar_documentos_vazios(entrada)

    assert vazio not in resultado, "Empty page_content must be discarded"
    assert whitespace not in resultado, "Whitespace-only page_content must be discarded"
    assert normal in resultado, "Non-empty document must be preserved"
    assert campo_vazio_label in resultado, "CAMPO_VAZIO label is non-empty after strip and must be preserved"
    assert len(resultado) == 2

    # Immutability: input list must not be mutated
    assert len(entrada) == 4, "filtrar_documentos_vazios must not mutate the input list"
    assert resultado is not entrada, "filtrar_documentos_vazios must return a new list"


# ---------------------------------------------------------------------------
# EC-embeddings: criar_embedding_model passes normalize_embeddings=True verbatim
# RAGING-06
# ---------------------------------------------------------------------------


def test_ac01_ec_embeddings_normalize_embeddings_true_verbatim() -> None:
    """
    Edge case (AC-01/RAGING-06): WHEN `criar_embedding_model` is called THEN
    it SHALL construct HuggingFaceEmbeddings with:
      - `model_name = config.EMBEDDING_MODEL`
      - `model_kwargs = {"device": <any>}`
      - `encode_kwargs = {"normalize_embeddings": True}`

    The `normalize_embeddings=True` flag is critical: omitting it silently
    degrades cosine-similarity scores across the entire vector space.
    """
    with patch("agenticlog.ingestion.embeddings.HuggingFaceEmbeddings") as mock_hf:
        from agenticlog.ingestion.embeddings import criar_embedding_model

        result = criar_embedding_model()

        mock_hf.assert_called_once_with(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": ANY},
            encode_kwargs={"normalize_embeddings": True},
        )
        assert result is mock_hf.return_value


# ---------------------------------------------------------------------------
# EC-hashes: hash helpers are deterministic
# RAGING-07
# ---------------------------------------------------------------------------


def test_ac01_ec_metadata_hash_helpers_sao_deterministicos() -> None:
    """
    Edge case (AC-01/RAGING-07): WHEN `_computar_hash_conteudo` is called with
    the same bytes THEN it SHALL return the same SHA-256 hex string both times.

    Deterministic hashes are required for the content-dedup logic in the
    incremental ingestion path (REC-01/REC-04).
    """
    from agenticlog.ingestion.metadata import _computar_hash_conteudo

    conteudo = b"dado de teste de logistica: pedido #42"
    hash1 = _computar_hash_conteudo(conteudo)
    hash2 = _computar_hash_conteudo(conteudo)

    assert hash1 == hash2, "Hash must be deterministic for identical input"
    assert len(hash1) == 64, "SHA-256 hex digest must be 64 characters"
    assert all(c in "0123456789abcdef" for c in hash1), "Hash must be lowercase hex"

    # Different content must produce different hash
    hash_other = _computar_hash_conteudo(b"outro conteudo")
    assert hash1 != hash_other, "Different content must produce different hash"


# ---------------------------------------------------------------------------
# EC-double-shim: monkeypatch on rag._rag_embedding_model is honored by getter
# RAGING-06 / design §5
# ---------------------------------------------------------------------------


def test_ac02_ec_double_shim_monkeypatch_rag_embedding_model_honrado_pelo_getter() -> None:
    """
    Edge case (AC-02/RAGING-06): WHEN `agenticlog.rag._rag_embedding_model` is
    monkeypatched to a stub THEN `_get_rag_embedding_model()` SHALL return the
    stub WITHOUT constructing the real 1 GB model.

    This is the oracle's monkeypatch seam. The global + getter MUST remain in
    rag.py (not moved to ingestion.embeddings) so `setattr("agenticlog.rag.*")`
    is visible to the getter — the double-shim design invariant.
    """
    stub = SimpleNamespace(embed_documents=lambda texts: [[0.0] * 16] * len(texts))
    original = rag._rag_embedding_model
    try:
        rag._rag_embedding_model = stub
        returned = rag._get_rag_embedding_model()
        assert returned is stub, (
            "_get_rag_embedding_model() must return the monkeypatched stub; "
            "if it does not, the double-shim is broken and the oracle will fail"
        )
    finally:
        rag._rag_embedding_model = original


# ---------------------------------------------------------------------------
# EC-carregar-json: uses JQ_SCHEMA_CAMPOS_JSON as jq_schema
# RAGING-03
# ---------------------------------------------------------------------------


def test_ac01_ec_carregar_json_usa_jq_schema_de_config() -> None:
    """
    Edge case (AC-01/RAGING-03): WHEN `carregar_json` is called with a path
    THEN it SHALL instantiate `JSONLoader` with `jq_schema=config.JQ_SCHEMA_CAMPOS_JSON`
    and return the documents from `.load()`.

    This confirms that the wrapper uses the same jq_schema as the rest of the
    pipeline — a deviation would silently change which JSON fields are extracted.
    """
    import tempfile

    with patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_docs = [Document(page_content="campo: valor")]
        mock_loader.load.return_value = mock_docs
        mock_loader_cls.return_value = mock_loader

        from agenticlog.ingestion.extraction import carregar_json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = carregar_json(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        mock_loader_cls.assert_called_once_with(
            str(tmp_path),
            jq_schema=config.JQ_SCHEMA_CAMPOS_JSON,
        )
        mock_loader.load.assert_called_once()
        assert result is mock_docs, "carregar_json must return the documents from loader.load()"
