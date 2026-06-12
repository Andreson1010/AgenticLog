# Chunking Estrutura-Aware — Tasks

**Path:** `.specs/features/chunking-estrutura-aware/tasks.md`
**TLC scope:** large
**Links to:** `.specs/features/chunking-estrutura-aware/spec.md`, `.specs/features/chunking-estrutura-aware/design.md`
**Status:** Awaiting human approval

---

## Execution Order

Tasks must be completed in dependency order. Each code task follows TDD: write/update the failing test first, then implement the minimum code to pass it. This pipeline does not use TLC Execute — these tasks are for `backend-builder`, `test-verifier`, and `validator` agents (no `frontend-builder`; this is a backend-only change).

```
T-01 (config constant)
   |
   v
T-02 (extrair_texto_pdf -> dict[str,str])
   |
   +--> T-03 (cria_vectordb: JSON path + filter)
   |        |
   |        v
   |    T-04 (cria_vectordb: PDF path + filter)
   |        |
   |        v
   |    T-05 (cria_vectordb: splitter separators)
   |
   +--> T-06 (adicionar_documento_incrementalmente: jq_schema + filter + separators)
   |
   +--> T-07 (salvar_pdf_enviado: consume dict)
   |
   v
T-08 (scripts/pdf_to_json.py thin wrapper)
   |
   v
T-09 (rewrite tests/test_rag.py: TestExtrairTextoPdf, TestSalvarPdfEnviado)
   |
   v
T-10 (rewrite tests/test_rag.py: TestCriaVectordb)
   |
   v
T-11 (extend tests/test_rag.py: TestAdicionarDocumentoIncrementalmente)
   |
   v
T-12 (new edge-case tests: empty filters, exact-CHUNK_SIZE boundary, residual split prefix)
   |
   v
T-13 (spot-check tests/test_rag_integration.py)
   |
   v
T-14 (CLAUDE.md — generalize rebuild section, ADR-010)
   |
   v
T-15 (coverage gate + final verification)
```

T-03 through T-07 may be implemented in any relative order once T-01/T-02 land, but each must be done BEFORE the corresponding test rewrite tasks (T-09 through T-12) turn green. T-08 depends on T-02 (relocated `extrair_texto_pdf`) but is independent of T-03 through T-07.

---

## Task T-01 — Add `JQ_SCHEMA_CAMPOS_JSON` constant to `config.py`

**Requirement IDs:** CHUNK-10, CHUNK-11
**File:** `src/agenticlog/config.py`
**Dependencies:** none

### What to implement

Add the shared jq_schema constant in the `# RAG` section, immediately after `CHUNK_OVERLAP` (line ~41):

```python
# jq_schema compartilhado: 1 entrada de lista por chave top-level do JSON -> 1 Document
# por chave (chunking estrutura-aware, ADR-008). SEM join — preserva separação por chave.
JQ_SCHEMA_CAMPOS_JSON = 'to_entries | map(.key + ": " + (.value | tostring))'
```

### Done when
- `JQ_SCHEMA_CAMPOS_JSON` is importable from `agenticlog.config`.
- Value is exactly `'to_entries | map(.key + ": " + (.value | tostring))'` (no `join`).
- No existing `config.py` tests break (no test currently asserts on this constant — it's new).

### Test/gate notes
- `config.py` has no dedicated test file per `TESTING.md` ("config.py | none (constants only)") — no new test required for this task alone, but T-09/T-11 will assert this constant is the one passed to `JSONLoader`/`DirectoryLoader`.

---

## Task T-02 — Change `extrair_texto_pdf` to return `dict[str, str]` (relocate `pdf_para_dict` logic)

**Requirement IDs:** CHUNK-06, CHUNK-16, CHUNK-18
**File:** `src/agenticlog/rag.py` (lines 402-427)
**Dependencies:** none (independent of T-01)

### What to implement

Replace the body of `extrair_texto_pdf` per design.md Component 2:

```python
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
```

Do NOT yet update `cria_vectordb` or `salvar_pdf_enviado` call sites in this task — that is T-04 and T-07. This task changes ONLY `extrair_texto_pdf`'s implementation and signature; its two existing callers will be temporarily "broken" (type mismatch) until T-04/T-07 land. Since this is large-scope work done in one PR before merge, this is acceptable — but run `pytest tests/test_rag.py -v` after T-02 alone and expect `TestExtrairTextoPdf` failures (fixed in T-09) and possibly `TestCriaVectordb`/`TestSalvarPdfEnviado` failures (fixed in T-04/T-07/T-09) — this is EXPECTED red state mid-refactor, not a regression to chase down in isolation.

### Test to update (TDD — Red first, see T-09 for full rewrite)

At minimum, before moving on, confirm the new dict-returning implementation is correct in isolation with a throwaway/temporary assertion (formal rewrite is T-09):

```python
# Temporary manual check (formalized in T-09):
# extrair_texto_pdf(Path("data/documents/materiais_logistica.pdf")) returns
# a dict with keys "PÁGINA_1", "PÁGINA_2", ... and no empty-string values.
```

### Done when
- `extrair_texto_pdf` returns `dict[str, str]`.
- All 3 `RAGSecurityError` messages preserved verbatim ("PDF inválido ou corrompido.", "PDF protegido por senha.", "PDF não contém texto extraível (somente imagem).").
- `with doc_handle:` context manager still wraps all `doc_handle` access (including the new page loop).

---

## Task T-03 — `cria_vectordb`: switch JSON path to `JQ_SCHEMA_CAMPOS_JSON` + empty-Document filter

**Requirement IDs:** CHUNK-01, CHUNK-02, CHUNK-03, CHUNK-04, CHUNK-10, CHUNK-11, CHUNK-12
**File:** `src/agenticlog/rag.py` (lines ~520-528, plus import block lines 29-49)
**Dependencies:** T-01

### What to implement

1. Add `JQ_SCHEMA_CAMPOS_JSON` to the `from agenticlog.config import (...)` block (alongside `CHUNK_SIZE`, `CHUNK_OVERLAP`).
2. Replace the inline `jq_schema` literal and comment (lines 520-521) with the shared constant:

```python
    # jq_schema compartilhado (config.py): 1 Document por chave top-level (ADR-008)
    loader = DirectoryLoader(
        str(DIR_DOCUMENTS),
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": JQ_SCHEMA_CAMPOS_JSON},
    )
    json_docs = loader.load()
    # Descarta Documents com page_content vazio (chave JSON com valor "", [] ou {}) — ADR-011
    json_docs = [d for d in json_docs if d.page_content.strip()]
```

### Done when
- No inline `jq_schema` string literal remains in `cria_vectordb`'s JSON-loading block.
- `JQ_SCHEMA_CAMPOS_JSON` is the value passed as `loader_kwargs={"jq_schema": ...}`.
- `json_docs` is filtered for non-empty `page_content` before being combined with `pdf_docs`.

---

## Task T-04 — `cria_vectordb`: PDF path produces 1 Document per page + empty-Document filter

**Requirement IDs:** CHUNK-06, CHUNK-07, CHUNK-08, CHUNK-09, CHUNK-13, CHUNK-15
**File:** `src/agenticlog/rag.py` (lines ~530-538)
**Dependencies:** T-02, T-03

### What to implement

Replace the PDF loop per design.md Component 4:

```python
    pdf_docs = []
    for pdf_path in DIR_DOCUMENTS.glob("*.pdf"):
        try:
            paginas = extrair_texto_pdf(pdf_path)
            for chave, texto in paginas.items():
                pdf_docs.append(
                    Document(page_content=f"{chave}: {texto}", metadata={"source": str(pdf_path)})
                )
        except RAGSecurityError as e:
            logger.error("PDF corrompido ignorado durante reconstrução: %s — %s", pdf_path.name, e)

    # Descarta Documents com page_content vazio — defensivo, simetria com o filtro JSON (ADR-011)
    pdf_docs = [d for d in pdf_docs if d.page_content.strip()]
```

`documents = json_docs + pdf_docs` (existing line) is unchanged in position — it now combines the filtered lists from T-03 and this task.

### Done when
- For a multi-page PDF, `pdf_docs` contains one `Document` per page with extractable text, `page_content = f"PÁGINA_{N}: {texto}"`.
- `RAGSecurityError` from `extrair_texto_pdf` (e.g. all-blank PDF) is still caught per-file via `except RAGSecurityError`, logged via the existing `logger.error(...)` call (message text unchanged), and contributes zero `Document`s for that file.
- The existing `logger.error` call signature/message is byte-for-byte unchanged (`"PDF corrompido ignorado durante reconstrução: %s — %s"`).

---

## Task T-05 — `cria_vectordb`: add sentence-aware separators to splitter

**Requirement IDs:** CHUNK-05, CHUNK-09
**File:** `src/agenticlog/rag.py` (lines ~544-548)
**Dependencies:** T-04

### What to implement

```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
```

### Done when
- `RecursiveCharacterTextSplitter` in `cria_vectordb` is constructed with `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]` in addition to the existing `chunk_size`/`chunk_overlap` kwargs.

---

## Task T-06 — `adicionar_documento_incrementalmente`: shared jq_schema + empty-Document filter + separators

**Requirement IDs:** CHUNK-01, CHUNK-02, CHUNK-03, CHUNK-05, CHUNK-10, CHUNK-11, CHUNK-12, CHUNK-14
**File:** `src/agenticlog/rag.py` (lines ~353-360)
**Dependencies:** T-01

### What to implement

```python
    loader = JSONLoader(str(saved_path), jq_schema=JQ_SCHEMA_CAMPOS_JSON)
    docs = loader.load()
    # Descarta Documents com page_content vazio (chave JSON com valor "", [] ou {}) — ADR-011
    docs = [d for d in docs if d.page_content.strip()]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
```

The zero-chunks branch (`if not chunks:`) and the `content_hash` assignment loop immediately after remain UNCHANGED — do not modify lines after `chunks = text_splitter.split_documents(docs)` in this task.

### Done when
- No inline `jq_schema` string literal (with `join`) remains in `adicionar_documento_incrementalmente`.
- `docs` is filtered for non-empty `page_content` before `split_documents`.
- `RecursiveCharacterTextSplitter` constructed with the same `separators=[...]` list as T-05 (identical list literal in both places — verbatim copy).
- Zero-chunks branch and `content_hash` loop unchanged (existing 9 tests in `TestAdicionarDocumentoIncrementalmente` continue to exercise the same downstream code paths).

---

## Task T-07 — `salvar_pdf_enviado`: consume `dict[str, str]` return from `extrair_texto_pdf`

**Requirement IDs:** CHUNK-18
**File:** `src/agenticlog/rag.py` (line 474)
**Dependencies:** T-02

### What to implement

No functional change — `extrair_texto_pdf(tmp_path)` is called for its validation side-effect only (return value already discarded). Add a clarifying comment per design.md Component 7:

```python
        extrair_texto_pdf(tmp_path)  # validação por efeito colateral: levanta RAGSecurityError se sem texto
```

### Done when
- `salvar_pdf_enviado` compiles and runs correctly against the new `dict[str, str]`-returning `extrair_texto_pdf` (T-02) — no `TypeError` or type-mismatch at runtime.
- `RAGSecurityError` propagation for password-protected / corrupted / image-only PDFs unchanged (verified by `TestSalvarPdfEnviado::teste_8_salvar_rollback_se_pdf_invalido`, which mocks `side_effect = RAGSecurityError(...)` — unaffected by return-type change).

---

## Task T-08 — `scripts/pdf_to_json.py`: thin wrapper delegating to `agenticlog.rag.extrair_texto_pdf`

**Requirement IDs:** CHUNK-16, CHUNK-17, CHUNK-19
**File:** `scripts/pdf_to_json.py`
**Dependencies:** T-02

### What to implement

Per design.md Component 8:

1. Remove `import fitz` (no longer needed — PyMuPDF logic now lives in `agenticlog.rag`).
2. Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))` before importing `agenticlog`.
3. Add `from agenticlog.rag import RAGSecurityError, extrair_texto_pdf`.
4. Replace `pdf_para_dict`'s body with a thin delegation that translates `RAGSecurityError` → `ValueError`:

```python
def pdf_para_dict(pdf_path: Path) -> dict[str, str]:
    """Extrai texto por página de um PDF (wrapper fino sobre agenticlog.rag.extrair_texto_pdf).

    Entrada: pdf_path — Path para o arquivo PDF.
    Saída: dict {"PÁGINA_1": "...", ...}.
    Lança ValueError se o PDF não contém texto extraível ou for inválido/protegido —
    preserva o contrato de exceção original do script CLI.
    """
    try:
        return extrair_texto_pdf(pdf_path)
    except RAGSecurityError as exc:
        raise ValueError(str(exc)) from exc
```

5. `converter()`, `_parse_args()`, `main()` — UNCHANGED. In particular, `converter()`'s `json.dumps(dados, ensure_ascii=False, indent=2)` (line 34) MUST be preserved verbatim — `ensure_ascii=False` is required to keep "PÁGINA" readable in output JSON (per researcher findings, this is a hard requirement, not incidental).

### Test to write/update first (TDD — Red)

**Researcher found no existing test file for `scripts/pdf_to_json.py`.** Confirm this with a final check (`Glob "tests/**/*pdf_to_json*"` and `Grep "pdf_para_dict"` across `tests/`) before proceeding. If genuinely absent, add a new minimal test file `tests/test_pdf_to_json.py`:

```python
"""Testes para scripts/pdf_to_json.py (wrapper fino sobre agenticlog.rag.extrair_texto_pdf)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "scripts"))

from agenticlog.rag import RAGSecurityError
import pdf_to_json


class TestPdfParaDict:
    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_1_pdf_para_dict_delega_para_extrair_texto_pdf(self, mock_extrair):
        mock_extrair.return_value = {"PÁGINA_1": "conteudo"}
        resultado = pdf_to_json.pdf_para_dict(Path("qualquer.pdf"))
        assert resultado == {"PÁGINA_1": "conteudo"}
        mock_extrair.assert_called_once_with(Path("qualquer.pdf"))

    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_2_pdf_sem_texto_levanta_value_error(self, mock_extrair):
        mock_extrair.side_effect = RAGSecurityError("PDF não contém texto extraível (somente imagem).")
        with pytest.raises(ValueError, match="somente imagem"):
            pdf_to_json.pdf_para_dict(Path("vazio.pdf"))


class TestConverter:
    @patch("pdf_to_json.extrair_texto_pdf")
    def teste_3_converter_escreve_json_ensure_ascii_false(self, mock_extrair, tmp_path):
        mock_extrair.return_value = {"PÁGINA_1": "Texto com acentuação: ção, ã, é"}
        destino = pdf_to_json.converter(Path("doc.pdf"), tmp_path)
        conteudo = destino.read_text(encoding="utf-8")
        assert "PÁGINA_1" in conteudo
        assert "ção" in conteudo  # ensure_ascii=False preserva acentuação
        assert "\\u" not in conteudo  # nao deve haver escapes unicode
```

### Done when
- `pdf_para_dict` delegates to `agenticlog.rag.extrair_texto_pdf` — no duplicated PyMuPDF page-iteration code in `scripts/pdf_to_json.py`.
- `RAGSecurityError` is translated to `ValueError` with the same message text.
- `converter()` output is byte-identical in format to before (`ensure_ascii=False, indent=2`).
- `python scripts/pdf_to_json.py <file>.pdf --output <dir>` still runs end-to-end without error against a real PDF.

---

## Task T-09 — Rewrite `TestExtrairTextoPdf` and update `TestSalvarPdfEnviado` mocks

**Requirement IDs:** CHUNK-06, CHUNK-16, CHUNK-18
**File:** `tests/test_rag.py` (lines 765-925)
**Dependencies:** T-02, T-07

### What to implement

**`TestExtrairTextoPdf`** (lines 765-830) — rewrite all 5 methods for `dict[str, str]` return:

```python
class TestExtrairTextoPdf(unittest.TestCase):
    """Testes para extrair_texto_pdf."""

    @patch("agenticlog.rag.fitz.open")
    def teste_1_extrair_pdf_valido_retorna_dict(self, mock_fitz_open):
        """PDF com texto retorna dict {"PÁGINA_1": texto}."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "texto do contrato"
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz_open.return_value = mock_doc

        resultado = extrair_texto_pdf(Path("qualquer.pdf"))

        self.assertEqual(resultado, {"PÁGINA_1": "texto do contrato"})

    @patch("agenticlog.rag.fitz.open")
    def teste_2_extrair_pdf_com_senha_lanca_erro(self, mock_fitz_open):
        """PDF com senha lança RAGSecurityError."""
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("senha", str(ctx.exception))

    @patch("agenticlog.rag.fitz.open")
    def teste_3_extrair_pdf_somente_imagem_lanca_erro(self, mock_fitz_open):
        """PDF somente-imagem (todas as páginas retornam texto vazio) lança RAGSecurityError."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   \n\t  "
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
        mock_fitz_open.return_value = mock_doc

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("somente imagem", str(ctx.exception))

    @patch("agenticlog.rag.fitz.open")
    def teste_4_extrair_pdf_mix_texto_imagem_filtra_pagina_vazia(self, mock_fitz_open):
        """PDF com mix de páginas texto e imagem: só a página com texto aparece no dict."""
        mock_page_texto = MagicMock()
        mock_page_texto.get_text.return_value = "conteúdo real"
        mock_page_imagem = MagicMock()
        mock_page_imagem.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page_texto, mock_page_imagem]))
        mock_fitz_open.return_value = mock_doc

        resultado = extrair_texto_pdf(Path("qualquer.pdf"))

        self.assertEqual(resultado, {"PÁGINA_1": "conteúdo real"})
        self.assertNotIn("PÁGINA_2", resultado)

    @patch("agenticlog.rag.fitz.open")
    def teste_5_extrair_exception_generica_lanca_erro(self, mock_fitz_open):
        """fitz.open() lançando Exception genérica é convertida em RAGSecurityError."""
        mock_fitz_open.side_effect = RuntimeError("unexpected fitz error")

        with self.assertRaises(rag.RAGSecurityError) as ctx:
            extrair_texto_pdf(Path("qualquer.pdf"))
        self.assertIn("corrompido", str(ctx.exception))

    @patch("agenticlog.rag.fitz.open")
    def teste_6_extrair_pdf_multipagina_retorna_dict_ordenado(self, mock_fitz_open):
        """PDF com 3 páginas de texto retorna dict com 3 chaves PÁGINA_1..3 na ordem."""
        mock_pages = []
        for i in range(3):
            p = MagicMock()
            p.get_text.return_value = f"texto da pagina {i + 1}"
            mock_pages.append(p)
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))
        mock_fitz_open.return_value = mock_doc

        resultado = extrair_texto_pdf(Path("qualquer.pdf"))

        self.assertEqual(
            resultado,
            {
                "PÁGINA_1": "texto da pagina 1",
                "PÁGINA_2": "texto da pagina 2",
                "PÁGINA_3": "texto da pagina 3",
            },
        )
        self.assertEqual(list(resultado.keys()), ["PÁGINA_1", "PÁGINA_2", "PÁGINA_3"])
```

**`TestSalvarPdfEnviado`** (lines 833-925) — update only the 2 methods that set a return value:

- `teste_1_salvar_pdf_valido_sucesso` (line 840-848): `mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}` (was `"texto extraído"`).
- `teste_3_salvar_aceita_extensao_maiuscula` (line 863-870): same change, `mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}`.
- `teste_8_salvar_rollback_se_pdf_invalido` (line 900) — NO CHANGE (`side_effect = RAGSecurityError(...)` unaffected by return type).
- All other `TestSalvarPdfEnviado` methods — NO CHANGE.

### Done when
- `pytest tests/test_rag.py::TestExtrairTextoPdf -v` — 6 methods, all green, all asserting `dict` equality (not `assertIn` on a string).
- `pytest tests/test_rag.py::TestSalvarPdfEnviado -v` — 9 methods, all green.

---

## Task T-10 — Rewrite `TestCriaVectordb`

**Requirement IDs:** CHUNK-01, CHUNK-02, CHUNK-03, CHUNK-04, CHUNK-06, CHUNK-07, CHUNK-08, CHUNK-09, CHUNK-10, CHUNK-11, CHUNK-12, CHUNK-13, CHUNK-14, CHUNK-15
**File:** `tests/test_rag.py` (lines 220-282)
**Dependencies:** T-03, T-04, T-05

### What to implement

Rewrite `test_cria_vectordb_sem_documentos_retorna_cedo` and `test_cria_vectordb_com_documentos_cria_chroma`, and add new methods covering: PDF per-page Documents, JSON empty-key filter, PDF blank-page filter, all-blank PDF caught by existing `except RAGSecurityError`, and the `separators` kwarg.

```python
class TestCriaVectordb(unittest.TestCase):
    """Testes para cria_vectordb."""

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_sem_documentos_retorna_cedo(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma
    ):
        """Quando não há documentos, cria_vectordb retorna sem criar Chroma."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []  # nenhum PDF

        cria_vectordb()

        mock_valida_path.assert_called_once()
        mock_valida_json.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        mock_chroma.from_documents.assert_not_called()

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_com_documentos_json_usa_jq_schema_e_separators(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma
    ):
        """Com documentos JSON válidos: usa JQ_SCHEMA_CAMPOS_JSON e separators de ADR-007."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="DESCRIÇÃO: texto da descrição"),
            Document(page_content="CRITÉRIOS: texto dos critérios"),
        ]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []  # nenhum PDF

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs  # passthrough
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        # jq_schema compartilhado usado
        _, loader_kwargs = mock_loader.call_args
        self.assertEqual(
            loader_kwargs["loader_kwargs"]["jq_schema"],
            config.JQ_SCHEMA_CAMPOS_JSON,
        )

        # separators de ADR-007 passados ao splitter
        mock_splitter.assert_called_once_with(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        mock_chroma.from_documents.assert_called_once()
        call_args = mock_chroma.from_documents.call_args
        self.assertEqual(len(call_args[0][0]), 2)
        self.assertTrue(call_args[0][0][0].page_content.startswith("DESCRIÇÃO: "))
        self.assertTrue(call_args[0][0][1].page_content.startswith("CRITÉRIOS: "))

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_filtra_documento_json_com_valor_vazio(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_splitter, mock_emb, mock_chroma
    ):
        """Document JSON com page_content vazio (chave com valor "") é descartado silenciosamente."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="DESCRIÇÃO: texto válido"),
            Document(page_content="CAMPO_VAZIO: "),  # .strip() == "CAMPO_VAZIO:" -- nao vazio!
            Document(page_content=""),  # totalmente vazio
            Document(page_content="   "),  # só whitespace
        ]
        mock_loader.return_value = mock_loader_instance
        mock_dir.glob.return_value = []

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        contents = [d.page_content for d in passed_docs]
        self.assertIn("DESCRIÇÃO: texto válido", contents)
        self.assertNotIn("", contents)
        self.assertNotIn("   ", contents)
        # "CAMPO_VAZIO: " com .strip() == "CAMPO_VAZIO:" é NAO vazio -> permanece
        self.assertIn("CAMPO_VAZIO: ", contents)

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_pdf_multipagina_um_document_por_pagina(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_extrair, mock_splitter, mock_emb, mock_chroma
    ):
        """PDF multi-página: 1 Document por página, prefixo PÁGINA_N: ."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance

        pdf_path = MagicMock()
        pdf_path.name = "materiais_logistica.pdf"
        mock_dir.glob.return_value = [pdf_path]

        mock_extrair.return_value = {
            "PÁGINA_1": "conteúdo da primeira página",
            "PÁGINA_2": "conteúdo da segunda página",
        }

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        self.assertEqual(len(passed_docs), 2)
        self.assertEqual(passed_docs[0].page_content, "PÁGINA_1: conteúdo da primeira página")
        self.assertEqual(passed_docs[1].page_content, "PÁGINA_2: conteúdo da segunda página")

    @patch("agenticlog.rag.Chroma")
    @patch("agenticlog.rag.HuggingFaceEmbeddings")
    @patch("agenticlog.rag.RecursiveCharacterTextSplitter")
    @patch("agenticlog.rag.extrair_texto_pdf")
    @patch("agenticlog.rag.DIR_DOCUMENTS")
    @patch("agenticlog.rag.DirectoryLoader")
    @patch("agenticlog.rag._valida_arquivos_json")
    @patch("agenticlog.rag._valida_path_documentos")
    def test_cria_vectordb_pdf_todas_paginas_em_branco_loga_erro_sem_levantar(
        self, mock_valida_path, mock_valida_json, mock_loader, mock_dir, mock_extrair, mock_splitter, mock_emb, mock_chroma
    ):
        """PDF totalmente em branco: extrair_texto_pdf levanta RAGSecurityError, capturado e logado, zero Documents para esse PDF."""
        from langchain_core.documents import Document

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="DESCRIÇÃO: texto válido")]
        mock_loader.return_value = mock_loader_instance

        pdf_path = MagicMock()
        pdf_path.name = "vazio.pdf"
        mock_dir.glob.return_value = [pdf_path]

        mock_extrair.side_effect = RAGSecurityError("PDF não contém texto extraível (somente imagem).")

        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.side_effect = lambda docs: docs
        mock_splitter.return_value = mock_splitter_instance

        with self.assertLogs("agenticlog.rag", level="ERROR") as log_ctx:
            cria_vectordb()

        call_args = mock_chroma.from_documents.call_args
        passed_docs = call_args[0][0]
        self.assertEqual(len(passed_docs), 1)  # só o Document JSON
        self.assertTrue(any("PDF corrompido ignorado" in m for m in log_ctx.output))
```

### Done when
- `pytest tests/test_rag.py::TestCriaVectordb -v` — all 6 methods green.
- At least one test asserts `JQ_SCHEMA_CAMPOS_JSON` is the `jq_schema` passed to `JSONLoader` via `DirectoryLoader`.
- At least one test asserts `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]` is passed to `RecursiveCharacterTextSplitter`.
- At least one test verifies 1 PDF page = 1 `Document` with `"PÁGINA_N: "` prefix.
- At least one test verifies empty-`page_content` `Document`s (after `.strip()`) are filtered before `Chroma.from_documents`.
- At least one test verifies an all-blank PDF is caught by `except RAGSecurityError`, logged, and contributes zero `Document`s without raising.

---

## Task T-11 — Extend `TestAdicionarDocumentoIncrementalmente` with jq_schema/separators/filter assertions

**Requirement IDs:** CHUNK-01, CHUNK-02, CHUNK-03, CHUNK-05, CHUNK-10, CHUNK-11, CHUNK-12, CHUNK-14
**File:** `tests/test_rag.py` (lines 944-1156)
**Dependencies:** T-06

### What to implement

The 9 existing methods (`teste_1` through `teste_9`) should remain green without modification (they mock `JSONLoader`/`RecursiveCharacterTextSplitter` directly with canned `LCDocument` lists and don't assert on the jq_schema string). Add 2 new methods:

```python
    def teste_10_usa_jq_schema_compartilhado(self) -> None:
        """JSONLoader é construído com JQ_SCHEMA_CAMPOS_JSON (constante compartilhada)."""
        conteudo = b'{"campo": "valor"}'
        mock_vdb = self._setup_vectordb_mock([], [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with (
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.RecursiveCharacterTextSplitter") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [
                    LCDocument(page_content="campo: valor", metadata={})
                ]
                mock_splitter_cls.return_value.split_documents.return_value = [self._chunk()]

                adicionar_documento_incrementalmente("doc.json", conteudo)

        _, kwargs = mock_loader_cls.call_args
        self.assertEqual(kwargs["jq_schema"], config.JQ_SCHEMA_CAMPOS_JSON)

        mock_splitter_cls.assert_called_once_with(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def teste_11_filtra_documents_com_page_content_vazio(self) -> None:
        """Documents com page_content vazio (apos strip) sao descartados antes do split."""
        conteudo = b'{"campo_a": "valor", "campo_b": ""}'
        mock_vdb = self._setup_vectordb_mock([], [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with (
                patch("agenticlog.rag.Chroma", return_value=mock_vdb),
                patch("agenticlog.rag._get_rag_embedding_model"),
                patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
                patch("agenticlog.rag.RecursiveCharacterTextSplitter") as mock_splitter_cls,
                patch("agenticlog.rag.DIR_DOCUMENTS", new=tmp_path),
                patch("agenticlog.rag.DIR_VECTORDB", new=tmp_path / "vectordb"),
                patch("agenticlog.agent.invalidar_vector_db"),
            ):
                mock_loader_cls.return_value.load.return_value = [
                    LCDocument(page_content="campo_a: valor", metadata={}),
                    LCDocument(page_content="campo_b: ", metadata={}),  # .strip() != "" -- "campo_b:"
                    LCDocument(page_content="", metadata={}),  # totalmente vazio -- filtrado
                ]
                mock_splitter = mock_splitter_cls.return_value
                mock_splitter.split_documents.side_effect = lambda docs: list(docs)

                adicionar_documento_incrementalmente("doc.json", conteudo)

        passed_docs = mock_splitter.split_documents.call_args[0][0]
        contents = [d.page_content for d in passed_docs]
        self.assertIn("campo_a: valor", contents)
        self.assertIn("campo_b: ", contents)  # .strip() == "campo_b:" -- nao vazio, permanece
        self.assertNotIn("", contents)
        self.assertEqual(len(passed_docs), 2)
```

### Done when
- `pytest tests/test_rag.py::TestAdicionarDocumentoIncrementalmente -v` — 11 methods (9 existing + 2 new), all green.
- `teste_10` asserts `JQ_SCHEMA_CAMPOS_JSON` and the ADR-007 `separators` list are passed to `JSONLoader`/`RecursiveCharacterTextSplitter`.
- `teste_11` asserts the empty-`page_content` filter runs before `split_documents` is called.

---

## Task T-12 — New edge-case tests: exact-CHUNK_SIZE boundary, residual split prefix-only-on-first-piece

**Requirement IDs:** CHUNK-02, CHUNK-05, CHUNK-09, CHUNK-14, CHUNK-15
**File:** `tests/test_rag.py` (new methods in `TestCriaVectordb` and/or `TestAdicionarDocumentoIncrementalmente`)
**Dependencies:** T-10, T-11

### What to implement

These tests use the REAL `RecursiveCharacterTextSplitter` (not mocked) to verify actual splitting behavior — unlike T-10/T-11 which mock the splitter to assert construction kwargs.

```python
class TestResidualSplitBehavior(unittest.TestCase):
    """Testes de comportamento real do RecursiveCharacterTextSplitter com separators de ADR-007."""

    def _splitter(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def teste_1_campo_no_limite_exato_nao_e_dividido(self) -> None:
        """page_content com len == CHUNK_SIZE exato nao e dividido (1 chunk)."""
        valor = "x" * (config.CHUNK_SIZE - len("CAMPO: "))
        doc = LCDocument(page_content=f"CAMPO: {valor}", metadata={})
        self.assertEqual(len(doc.page_content), config.CHUNK_SIZE)

        chunks = self._splitter().split_documents([doc])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, doc.page_content)

    def teste_2_campo_acima_do_limite_e_dividido_so_primeiro_mantem_prefixo(self) -> None:
        """page_content > CHUNK_SIZE e dividido residualmente; so o 1o pedaco mantem 'CAMPO: '."""
        # Texto com frases completas para exercitar separadores ". ", "! ", "? "
        frase = "Esta e uma frase de exemplo sobre logistica e cadeia de suprimentos. "
        valor = frase * 10  # > CHUNK_SIZE (500)
        doc = LCDocument(page_content=f"CAMPO: {valor}", metadata={})
        self.assertGreater(len(doc.page_content), config.CHUNK_SIZE)

        chunks = self._splitter().split_documents([doc])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].page_content.startswith("CAMPO: "))
        for chunk in chunks[1:]:
            self.assertFalse(chunk.page_content.startswith("CAMPO: "))

    def teste_3_pagina_pdf_acima_do_limite_prefere_fronteira_de_frase(self) -> None:
        """Pagina PDF > CHUNK_SIZE: split residual prefere '. '/'! '/'? ' sobre corte bruto."""
        frase = "Materiais de logistica incluem paletes, embalagens e racks. "
        texto = frase * 10
        doc = LCDocument(page_content=f"PÁGINA_1: {texto}", metadata={})

        chunks = self._splitter().split_documents([doc])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].page_content.startswith("PÁGINA_1: "))
        # Pedacos que nao sao o ultimo devem terminar em fronteira de frase ou espaco
        for chunk in chunks[:-1]:
            stripped = chunk.page_content.rstrip()
            self.assertTrue(
                stripped.endswith((".", "!", "?")) or chunk.page_content.endswith(" "),
                f"Chunk nao termina em fronteira esperada: {chunk.page_content!r}",
            )

    def teste_4_doc1_json_seis_chaves_produz_seis_chunks(self) -> None:
        """Replica AC1/Independent Test: doc1.json (6 chaves, todas <= CHUNK_SIZE) -> 6 chunks."""
        import json as json_module

        doc1_path = config.DIR_DOCUMENTS / "doc1.json"
        if not doc1_path.exists():
            self.skipTest("data/documents/doc1.json nao encontrado neste ambiente")

        dados = json_module.loads(doc1_path.read_text(encoding="utf-8"))
        docs = [
            LCDocument(page_content=f"{chave}: {valor}", metadata={})
            for chave, valor in dados.items()
        ]
        # Confirma premissa: todos os campos <= CHUNK_SIZE (sem split)
        for d in docs:
            self.assertLessEqual(len(d.page_content), config.CHUNK_SIZE)

        chunks = self._splitter().split_documents(docs)

        self.assertEqual(len(chunks), 6)
        for chave, chunk in zip(dados.keys(), chunks):
            self.assertTrue(chunk.page_content.startswith(f"{chave}: "))
```

### Done when
- `pytest tests/test_rag.py::TestResidualSplitBehavior -v` — 4 methods, all green, using the REAL `RecursiveCharacterTextSplitter`.
- `teste_1` confirms `len == CHUNK_SIZE` exactly is NOT split.
- `teste_2`/`teste_3` confirm residual split keeps prefix ONLY on the first piece and prefers sentence boundaries.
- `teste_4` confirms `doc1.json`'s 6 keys produce exactly 6 chunks, each starting with its key — this is the spec.md P1 "Independent Test" made executable. If `data/documents/doc1.json` is unavailable in the test environment, the test SHOULD `skipTest` rather than fail (data-file presence is not a code-correctness gate).

---

## Task T-13 — Spot-check `tests/test_rag_integration.py`

**Requirement IDs:** CHUNK-10, CHUNK-11 (regression verification)
**File:** `tests/test_rag_integration.py`
**Dependencies:** T-06 (code change must be in place)

### What to implement

No code changes anticipated. Run the integration suite and confirm green:

```bash
pytest -m integration tests/test_rag_integration.py -v
```

Per spec.md Risks: `teste_1, 2, 3, 5, 6, 7` use real (non-mocked) `JSONLoader`/`RecursiveCharacterTextSplitter` against small 2-key JSON dicts (e.g. `{"produto": "cadeira", "cor": "azul"}`). With the new `jq_schema`, each now produces 2 `Document`s/chunks instead of 1. Current assertions check only `result["status"] == "adicionado"` and file existence — NOT chunk counts — so these are EXPECTED to remain green.

**If any test fails:**
1. Read the failure — it will most likely be `teste_7_query_retorna_docs_de_multiplas_colecoes`, which asserts retrieved content contains specific substrings (`"Beta SA"` or `"C-999"`). With 2 chunks per file instead of 1, both chunks are still embedded and retrievable — substring assertions on `all_content` (joined across `docs`) should still pass. If it fails, the fix is to confirm `_get_retriever`'s `k` (top-k retrieval count) still returns enough chunks across 2 collections — this would be a PRE-EXISTING `agent.py` configuration concern, NOT a `rag.py` chunking bug. Escalate rather than silently increasing `k`.
2. `teste_4_rollback_nao_deixa_chunks_orfaos` already mocks `JSONLoader`/`RecursiveCharacterTextSplitter` explicitly — no change expected.

### Done when
- `pytest -m integration tests/test_rag_integration.py -v` — all 7 methods green, OR any failure is triaged and either (a) fixed mechanically (count assertion update) or (b) escalated as a pre-existing `agent.py` issue unrelated to this feature's scope.

---

## Task T-14 — Generalize `CLAUDE.md` "Build VectorDB" section (ADR-010)

**Requirement IDs:** CHUNK-20, CHUNK-21
**File:** `CLAUDE.md` (lines 9-22)
**Dependencies:** none (can be done in parallel with any other task; placed last for review-ordering convenience)

### What to implement

Replace the section header and "Silent-degradation risk" paragraph (lines 9-22):

**Before:**
```markdown
### Build VectorDB (first time, after changing documents, or after changing `EMBEDDING_MODEL`)
```bash
python -m agenticlog.rag
```

**After changing `EMBEDDING_MODEL` in `config.py`** (e.g., switching embedding models), rebuild the vector DB from scratch:
1. Stop the running app (if any).
2. Delete `data/vectordb/` (gitignored, fully regenerable).
3. Rerun `python -m agenticlog.rag`.
4. Resume queries with `streamlit run app.py`.

The current model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (multilingual, 768-dim, optimized for Portuguese among other languages). On first run with this model, expect a larger download (~1.0–1.1 GB, vs ~440 MB for the previous `BAAI/bge-base-en`), so initial setup takes longer.

**Silent-degradation risk:** if `data/vectordb/` is **not** rebuilt after an `EMBEDDING_MODEL` change, the system will **not** raise an error — both the old and new models produce 768-dimensional vectors, so dimensions still match. However, the existing vectors were computed in a different (incompatible) embedding space than new query vectors, so similarity scores and retrieval results become unreliable, with no warning in logs or the UI. Always rebuild `data/vectordb/` after changing `EMBEDDING_MODEL`.
```

**After:**
```markdown
### Build VectorDB (first time, after changing documents, or after changing chunking/embedding configuration)
```bash
python -m agenticlog.rag
```

**After changing any of the following in `config.py`** — `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `JQ_SCHEMA_CAMPOS_JSON` (jq_schema), or PDF-extraction logic in `extrair_texto_pdf` (`src/agenticlog/rag.py`) — rebuild the vector DB from scratch:
1. Stop the running app (if any).
2. Delete `data/vectordb/` (gitignored, fully regenerable).
3. Rerun `python -m agenticlog.rag`.
4. Resume queries with `streamlit run app.py`.

The current embedding model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (multilingual, 768-dim, optimized for Portuguese among other languages). On first run with this model, expect a larger download (~1.0–1.1 GB, vs ~440 MB for the previous `BAAI/bge-base-en`), so initial setup takes longer.

**Silent-degradation risk:** if `data/vectordb/` is **not** rebuilt after changing `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, the jq_schema, or PDF-extraction logic, the system will **not** raise an error. Existing chunks remain queryable with their original embeddings/dimensions, but they were computed under a different chunking strategy or embedding space than newly ingested content — similarity scores and retrieval results become inconsistent or unreliable, with no warning in logs or the UI. Additionally, incremental ingestion (`adicionar_documento_incrementalmente`) skips files already present in `data/vectordb/` (detected via content-hash dedup), so OLD-strategy chunks for already-ingested files are never replaced without a full rebuild. Always rebuild `data/vectordb/` after any chunking-strategy or embedding-model change.
```

### Done when
- `CLAUDE.md` "Build VectorDB" section header no longer says "or after changing `EMBEDDING_MODEL`" exclusively — it lists `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, jq_schema, and PDF-extraction logic.
- The "Silent-degradation risk" paragraph covers all 5 triggers generically, including the incremental-ingestion-skip caveat.
- No other section of `CLAUDE.md` is modified.

---

## Task T-15 — Coverage gate and final verification

**Requirement IDs:** All
**File:** none (verification task)
**Dependencies:** T-09, T-10, T-11, T-12, T-13, T-14

### What to verify

1. Run `pytest --cov=agenticlog --cov-report=term-missing -v`.
2. Coverage must be `>= 80%` for `agenticlog/rag.py` (per `TESTING.md` — coverage gate, no regression vs. pre-feature baseline).
3. `Grep "extrair_texto_pdf("` across `src/`, `tests/`, `app.py`, `scripts/` — confirm both production callers (`cria_vectordb`, `salvar_pdf_enviado`) and the new `scripts/pdf_to_json.py` wrapper are updated; confirm no stale `str`-typed usage remains (e.g. string concatenation, `.join()` on the return value).
4. `Grep "join\(\\\\n\)"` (or equivalent) and `Grep "to_entries"` across `src/agenticlog/` — confirm `JQ_SCHEMA_CAMPOS_JSON` is the ONLY jq_schema literal; no duplicated inline jq strings remain in `cria_vectordb` or `adicionar_documento_incrementalmente`.
5. Run `python -m agenticlog.rag` against `data/documents/` (after deleting `data/vectordb/` per T-14's updated instructions) — confirm it completes without error and produces `data/vectordb/`.
6. Run `python scripts/pdf_to_json.py data/documents/materiais_logistica.pdf --output /tmp/pdf_to_json_test/` — confirm output JSON has `"PÁGINA_N"` keys and readable (non-escaped) Portuguese text.
7. Verify all new/modified functions in `rag.py` retain Portuguese docstrings with `Entrada:`/`Saída:`/`Lança` sections per `CONVENTIONS.md`.
8. Verify no `print()` statements were introduced (per `CONCERNS.md` — "No Logging Module" — `rag.py` already uses `logger`).
9. Confirm `pytest -m integration tests/test_rag_integration.py -v` (T-13) is green.

### Done when
- `pytest --cov=agenticlog --cov-report=term-missing -v` exits 0, `rag.py` coverage `>= 80%`.
- No stale `extrair_texto_pdf` `str`-typed usages found.
- Exactly one `jq_schema` literal (`JQ_SCHEMA_CAMPOS_JSON` in `config.py`) exists across `src/agenticlog/`.
- `python -m agenticlog.rag` and `python scripts/pdf_to_json.py ...` both succeed against real data.
- All new/modified `rag.py` functions have Portuguese `Entrada/Saída/Lança` docstrings.
- No new `print()` statements.
- Integration suite green.

---

## Summary Table

| Task | Requirement IDs | Key file | Test class/method |
|------|------------------|----------|---------------------|
| T-01 | CHUNK-10, CHUNK-11 | `config.py` | — (constant only) |
| T-02 | CHUNK-06, CHUNK-16, CHUNK-18 | `rag.py` | (formalized in T-09) |
| T-03 | CHUNK-01..04, 10-12 | `rag.py` | `TestCriaVectordb` (T-10) |
| T-04 | CHUNK-06..09, 13, 15 | `rag.py` | `TestCriaVectordb` (T-10) |
| T-05 | CHUNK-05, CHUNK-09 | `rag.py` | `TestCriaVectordb` (T-10) |
| T-06 | CHUNK-01..03, 05, 10-12, 14 | `rag.py` | `TestAdicionarDocumentoIncrementalmente` (T-11) |
| T-07 | CHUNK-18 | `rag.py` | `TestSalvarPdfEnviado` (T-09) |
| T-08 | CHUNK-16, 17, 19 | `scripts/pdf_to_json.py` | `tests/test_pdf_to_json.py` (new) |
| T-09 | CHUNK-06, 16, 18 | `tests/test_rag.py` | `TestExtrairTextoPdf`, `TestSalvarPdfEnviado` |
| T-10 | CHUNK-01..04, 06-15 | `tests/test_rag.py` | `TestCriaVectordb` |
| T-11 | CHUNK-01..03, 05, 10-12, 14 | `tests/test_rag.py` | `TestAdicionarDocumentoIncrementalmente` |
| T-12 | CHUNK-02, 05, 09, 14, 15 | `tests/test_rag.py` | `TestResidualSplitBehavior` (new) |
| T-13 | CHUNK-10, 11 (regression) | `tests/test_rag_integration.py` | `TestIngestionIntegration` |
| T-14 | CHUNK-20, 21 | `CLAUDE.md` | — (docs only) |
| T-15 | All | — | Coverage gate |
