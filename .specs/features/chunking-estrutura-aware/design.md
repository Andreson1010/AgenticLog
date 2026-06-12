# Chunking Estrutura-Aware — Design

**Path:** `.specs/features/chunking-estrutura-aware/design.md`
**TLC scope:** large
**Links to:** `.specs/features/chunking-estrutura-aware/spec.md`
**Status:** Awaiting human approval

---

## Architecture Overview

This feature is a surgical change to the ingestion layer of `src/agenticlog/rag.py`. No new modules are created and no existing module is split — this aligns with `coding-style.md` ("don't design for hypothetical future requirements") and the researcher's recommendation (no precedent for module splitting in `src/agenticlog/`, only `health.py`, `agent.py`, `history.py`, `rag.py`, `api.py`, `config.py` exist).

```
BEFORE:
  cria_vectordb()                          adicionar_documento_incrementalmente()
    jq_schema (with join) ── duplicated ──── jq_schema (with join)
    JSONLoader -> 1 Document/file            JSONLoader -> 1 Document/file
    extrair_texto_pdf() -> str               (no PDF path)
    RecursiveCharacterTextSplitter           RecursiveCharacterTextSplitter
      (default separators)                     (default separators)

  scripts/pdf_to_json.py
    pdf_para_dict() -> dict[str,str]   <- duplicate PyMuPDF page-loop, NOT used by rag.py


AFTER:
  config.py
    JQ_SCHEMA_CAMPOS_JSON = 'to_entries | map(.key + ": " + (.value | tostring))'  # shared, no join

  cria_vectordb()                          adicionar_documento_incrementalmente()
    JQ_SCHEMA_CAMPOS_JSON ── shared ──────── JQ_SCHEMA_CAMPOS_JSON
    JSONLoader -> 1 Document/key             JSONLoader -> 1 Document/key
    filter empty page_content                filter empty page_content
    extrair_texto_pdf() -> dict[str,str]     (no PDF path — unchanged)
      -> 1 Document/page ("PÁGINA_N: ...")
    filter empty page_content (PDF)
    RecursiveCharacterTextSplitter           RecursiveCharacterTextSplitter
      (+ sentence separators)                  (+ sentence separators)

  extrair_texto_pdf()  [rag.py]  <- pdf_para_dict logic relocated here (single source of truth)

  scripts/pdf_to_json.py
    pdf_para_dict() -> thin wrapper delegating to agenticlog.rag.extrair_texto_pdf
```

---

## Component Design

### 1. New constant: `JQ_SCHEMA_CAMPOS_JSON` (`src/agenticlog/config.py`)

Placed in the existing `# RAG` section, immediately after `CHUNK_OVERLAP` (line ~41), preserving the file's "constants near related constants" organization (`CONVENTIONS.md` — "Constants / config reads" near top of module, grouped by concern):

```python
# RAG
CHUNK_SIZE = 500    # tamanho máximo de cada chunk de texto em caracteres
CHUNK_OVERLAP = 50  # sobreposição entre chunks para preservar contexto nas bordas

# jq_schema compartilhado: 1 entrada de lista por chave top-level do JSON -> 1 Document
# por chave (chunking estrutura-aware, ADR-008). SEM join — preserva separação por chave.
JQ_SCHEMA_CAMPOS_JSON = 'to_entries | map(.key + ": " + (.value | tostring))'
```

**Naming rationale:** `JQ_SCHEMA_CAMPOS_JSON` follows the existing UPPER_SNAKE_CASE convention with English abbreviation prefix (`JQ_SCHEMA_`) + Portuguese domain noun (`CAMPOS_JSON` = "JSON fields"), matching the mixed-language convention documented in `CONVENTIONS.md` ("Constants in config: English abbreviations (LLM, RAG, API)" + Portuguese domain terms elsewhere, e.g. `DIR_DOCUMENTS`, `MAX_JSON_FILES`).

### 2. `extrair_texto_pdf` — relocated and signature-changed (`src/agenticlog/rag.py`)

**Before** (lines 402-427):
```python
def extrair_texto_pdf(path: Path) -> str:
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
```

**After** — merges `pdf_para_dict` (`scripts/pdf_to_json.py:18-28`) into the existing security-checked function, replacing the concatenation with a per-page dict, and replacing the final `not texto.strip()` check with `not pages`:

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

**Notes:**
- The `with doc_handle:` context manager (existing pattern) is preserved and now wraps the page-iteration loop too — improvement over `pdf_para_dict`'s manual `doc.close()` (`scripts/pdf_to_json.py:25`), which is dropped in favor of the existing context-manager pattern already proven in `rag.py`.
- All 3 existing `RAGSecurityError` cases are preserved verbatim (corrupted file, password-protected, no extractable text) — only the *empty-check* changes from `not texto.strip()` to `not pages` (equivalent: `pages` is empty exactly when no page had non-empty `.strip()` text, which is exactly when concatenation would have been empty).
- `enumerate(doc_handle)` matches `pdf_para_dict`'s `for i, page in enumerate(doc)` — page numbering is 1-indexed via `i + 1`.

### 3. `cria_vectordb` — JSON path changes (`src/agenticlog/rag.py`, lines ~520-528)

**Before:**
```python
    # jq_schema: achata o JSON em "chave: valor\nchave: valor" para facilitar chunking e busca semântica
    jq_schema = 'to_entries | map(.key + ": " + (.value | tostring)) | join("\\n")'
    loader = DirectoryLoader(
        str(DIR_DOCUMENTS),
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": jq_schema},
    )
    json_docs = loader.load()
```

**After:**
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

`JQ_SCHEMA_CAMPOS_JSON` added to the `from agenticlog.config import (...)` block at the top of `rag.py` (alongside `CHUNK_SIZE`, `CHUNK_OVERLAP`).

### 4. `cria_vectordb` — PDF path changes (`src/agenticlog/rag.py`, lines ~530-538)

**Before:**
```python
    pdf_docs = []
    for pdf_path in DIR_DOCUMENTS.glob("*.pdf"):
        try:
            texto = extrair_texto_pdf(pdf_path)
            pdf_docs.append(Document(page_content=texto, metadata={"source": str(pdf_path)}))
        except RAGSecurityError as e:
            logger.error("PDF corrompido ignorado durante reconstrução: %s — %s", pdf_path.name, e)
```

**After:**
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

**Why the defensive PDF filter even though `extrair_texto_pdf` already excludes empty pages:** `extrair_texto_pdf` guarantees each `texto` value is non-empty after `.strip()`, so `f"{chave}: {texto}"` is always non-empty (the prefix alone makes `.strip()` non-empty even in a pathological case). This filter line is therefore a no-op for correct `extrair_texto_pdf` output, included for **structural uniformity with the JSON path** (ADR-011 specifies the filter applies to "documentos por página de PDF" in `cria_vectordb`) and as a defense-in-depth invariant if `extrair_texto_pdf`'s contract ever changes. This is the same one-line expression as the JSON filter — copy-paste consistency mitigates the "filter divergence" risk in spec.md.

### 5. `cria_vectordb` — splitter changes (`src/agenticlog/rag.py`, lines ~544-548)

**Before:**
```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
```

**After:**
```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
```

(ADR-007 — applies identically in `adicionar_documento_incrementalmente`, see below.)

### 6. `adicionar_documento_incrementalmente` — changes (`src/agenticlog/rag.py`, lines ~353-371)

**Before:**
```python
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
```

**After:**
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

    if not chunks:
        logger.warning("Arquivo %s produziu zero chunks após divisão.", safe_name)
        saved_path.unlink(missing_ok=True)
        return {
            "status": "adicionado",
            "mensagem": f"Arquivo {safe_name} não pôde ser indexado: 0 chunks gerados.",
        }

    for chunk in chunks:
        chunk.metadata["content_hash"] = hash_str
```

The zero-chunks branch and `content_hash` loop are unchanged — they already handle the "0 chunks" outcome (which now also covers "all keys had empty values" per spec.md edge cases).

### 7. `salvar_pdf_enviado` — consume dict return (`src/agenticlog/rag.py`, line 474)

**Before:**
```python
        extrair_texto_pdf(tmp_path)
```

**After:**
```python
        extrair_texto_pdf(tmp_path)  # validação por efeito colateral: levanta RAGSecurityError se sem texto
```

**No functional code change required.** `extrair_texto_pdf` is called purely for its validation side-effect (the return value was already discarded — researcher confirmed). The only change is the **type** of the (discarded) return value, `str` → `dict[str, str]`. The existing `RAGSecurityError` propagation (corrupted PDF, password-protected, no extractable text → empty dict) is unchanged because all three error conditions are still raised from inside `extrair_texto_pdf` before it returns. A clarifying inline comment is added to make the side-effect-only call explicit (addresses spec.md P4-AC3 — "validates that the returned dict has at least one entry" is satisfied implicitly because `extrair_texto_pdf` raises `RAGSecurityError` rather than returning an empty dict).

### 8. PDF extraction relocation: `scripts/pdf_to_json.py` thin wrapper

**Decision:** `pdf_para_dict` in `scripts/pdf_to_json.py` becomes a thin wrapper that imports and calls `agenticlog.rag.extrair_texto_pdf`, translating `RAGSecurityError` back to the script's pre-existing `ValueError` contract (preserving the CLI's error-message expectations without changing `agenticlog.rag`'s exception type, which is `RAGSecurityError` for ALL rag.py validation failures per `CONVENTIONS.md`).

**Before** (`scripts/pdf_to_json.py:18-28`):
```python
def pdf_para_dict(pdf_path: Path) -> dict[str, str]:
    doc = fitz.open(str(pdf_path))
    pages = {}
    for i, page in enumerate(doc):
        texto = page.get_text().strip()
        if texto:
            pages[f"PÁGINA_{i + 1}"] = texto
    doc.close()
    if not pages:
        raise ValueError(f"Nenhum texto extraível em {pdf_path.name!r}.")
    return pages
```

**After:**
```python
"""Converte arquivos PDF em JSON compatível com o pipeline RAG do AgenticLog.

Formato de saída: {"PÁGINA_1": "texto...", "PÁGINA_2": "texto...", ...}

Uso:
    python scripts/pdf_to_json.py arquivo.pdf [arquivo2.pdf ...]
    python scripts/pdf_to_json.py *.pdf --output data/documents/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agenticlog.rag import RAGSecurityError, extrair_texto_pdf


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


def converter(pdf_path: Path, output_dir: Path) -> Path:
    dados = pdf_para_dict(pdf_path)
    destino = output_dir / pdf_path.with_suffix(".json").name
    destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


# _parse_args() and main() unchanged
```

**Why translate `RAGSecurityError` -> `ValueError` instead of letting it propagate:**
- Preserves the CLI's existing exception-message format and the `except Exception as exc` catch-all in `main()` (line 66) sees the same effective behavior either way (`RAGSecurityError` IS-A `Exception`, so it would also be caught) — BUT translating to `ValueError` keeps `pdf_para_dict`'s **documented contract** stable for any future direct callers of the script's `pdf_para_dict`, without forcing `scripts/` to depend on `agenticlog.rag`'s exception taxonomy.
- `RAGSecurityError`'s 3 messages ("PDF inválido ou corrompido.", "PDF protegido por senha.", "PDF não contém texto extraível (somente imagem).") all read naturally as `ValueError` messages for a CLI conversion script — no message text needs to change.
- `fitz` import is no longer needed in `scripts/pdf_to_json.py` (moved to `rag.py`, which already imports `fitz`) — removed from the script's imports.
- `sys.path.insert(...)` mirrors the existing pattern in `tests/test_rag.py:16-17` and `tests/test_rag_integration.py:15-16` for resolving `agenticlog` from `src/` without installation.

**Alternative considered and rejected:** Re-export `extrair_texto_pdf` directly as `pdf_para_dict = extrair_texto_pdf` (simple alias). Rejected because it would silently change `pdf_para_dict`'s exception contract from `ValueError` to `RAGSecurityError` for any caller that pattern-matches on exception type — the thin-wrapper-with-translation approach is one extra function definition but zero behavioral surprise for the CLI script's contract.

---

## Data Models

No new Pydantic models, no ChromaDB schema changes, no database migrations. Summary of the one type-signature change:

| Symbol | Location | Before | After |
|--------|----------|--------|-------|
| `extrair_texto_pdf(path: Path)` | `src/agenticlog/rag.py` | `-> str` | `-> dict[str, str]` |
| `pdf_para_dict(pdf_path: Path)` | `scripts/pdf_to_json.py` | `-> dict[str, str]` (own impl) | `-> dict[str, str]` (delegates to `agenticlog.rag.extrair_texto_pdf`) |
| `JQ_SCHEMA_CAMPOS_JSON` | `src/agenticlog/config.py` | does not exist | `str` constant, no `join` |

---

## Reuse from Codebase

| Existing component | Reused how |
|---------------------|-----------|
| `RAGSecurityError` | All 3 `extrair_texto_pdf` error paths preserved verbatim; new `JQ_SCHEMA_CAMPOS_JSON`/separators changes raise no new exception types |
| `with doc_handle:` context-manager pattern (already in `extrair_texto_pdf`) | Extended to wrap the page-iteration loop, replacing `pdf_para_dict`'s manual `doc.close()` |
| `logger = logging.getLogger(__name__)` (rag.py:51) | No new logger needed — existing `logger.error` (PDF loop) and `logger.warning` (zero-chunks) calls unchanged; ADR-011 filters add NO new log calls (silent by design) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` from `config.py` | Unchanged values, now passed alongside the new `separators` kwarg at both `RecursiveCharacterTextSplitter` call sites |
| `from agenticlog.config import (...)` block (rag.py:29-49) | Extended with `JQ_SCHEMA_CAMPOS_JSON` |
| `sys.path.insert(0, str(_root / "src"))` pattern (tests/test_rag.py, tests/test_rag_integration.py) | Mirrored in `scripts/pdf_to_json.py` for the new `from agenticlog.rag import ...` |
| `Document` from `langchain_core.documents` (already imported in rag.py:24) | Reused for per-page PDF `Document` construction (was already imported and used for the old single-blob PDF `Document`) |

---

## CONCERNS.md Items Addressed

Reviewed `.specs/codebase/CONCERNS.md` — no items in that file are directly addressed or worsened by this feature:

- **HIGH/MEDIUM items** (LMStudio SPOF, hardcoded credentials, missing error handling at startup) — out of scope, unrelated to ingestion chunking.
- **MEDIUM — Incomplete Test Coverage** — this feature ADDS test coverage (rewritten `TestExtrairTextoPdf`, new edge-case tests for empty-filter and PDF-page chunking) rather than relying on it; net positive but does not close the listed gaps (agent_workflow e2e, app.py, config.py constants — all unrelated to this feature).
- **LOW — Type Hints Incomplete** — this feature IMPROVES type hints for `extrair_texto_pdf` (`str` → `dict[str, str]`, both already and newly type-annotated).
- **MEDIUM — No Logging Module** — `rag.py` already uses `logging.getLogger(__name__)` (not `print()`); this feature adds zero new log statements (ADR-011 silent filters) and does not reintroduce `print()`.

No CONCERNS.md item requires a mitigation plan as part of this design.

---

## ADR Alignment Summary

| ADR | Design section implementing it |
|-----|--------------------------------|
| ADR-007 (sentence separators) | Component 5 (`cria_vectordb` splitter) and Component 6 (`adicionar_documento_incrementalmente` splitter) — both add `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]` |
| ADR-008 (chunking by key/page, shared jq_schema, PDF dict) | Components 1, 2, 3, 4, 6 |
| ADR-009 (no bin-packing) | No component implements bin-packing — confirmed absent by omission |
| ADR-010 (CLAUDE.md generalization) | Not a code component — see tasks.md for the documentation task |
| ADR-011 (silent empty-Document filter) | Components 3, 4, 6 — identical one-line filter `[d for d in X if d.page_content.strip()]` at all 3 sites |
| ADR-012 (page_content format contract `"{chave}: {valor}"`) | Components 3 (JSON, native via jq_schema), 4 (PDF, explicit f-string); residual-split prefix behavior is the DEFAULT `RecursiveCharacterTextSplitter` behavior — no extra code |
