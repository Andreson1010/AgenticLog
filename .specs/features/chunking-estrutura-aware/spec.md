# Chunking Estrutura-Aware — Technical Spec

**Path:** `.specs/features/chunking-estrutura-aware/spec.md`
**TLC scope:** large
**Based on story:** As a developer maintaining the AgenticLog ingestion pipeline, I want JSON and PDF documents chunked along their natural structural boundaries (JSON top-level keys, PDF pages) instead of split mid-sentence by raw character count, so retrieved chunks preserve complete, semantically coherent units of information.
**Status:** Awaiting human approval

---

## Problem Statement

The current ingestion pipeline (`cria_vectordb`, `adicionar_documento_incrementalmente` in `src/agenticlog/rag.py`) flattens each JSON document into a single string (`jq_schema` with `join("\n")`) and each PDF into a single concatenated blob (`extrair_texto_pdf` returns `str`), then lets `RecursiveCharacterTextSplitter` cut both at raw `CHUNK_SIZE` boundaries. This produces chunks that split mid-field and mid-page, mixing unrelated topics or truncating a single topic across two chunks — degrading retrieval quality. The fix re-aligns chunk boundaries with the documents' natural structure: one JSON top-level key = one chunk (pre-split), one PDF page = one chunk (pre-split), with residual splitting only for oversized units and sentence-aware separators.

## Goals

- [ ] Each top-level key of a JSON document produces its own `Document` with `page_content = "{CHAVE}: {valor}"`, before any size-based splitting.
- [ ] Each page of a PDF produces its own `Document` with `page_content = "PÁGINA_{N}: {texto}"`, before any size-based splitting.
- [ ] Units (JSON field or PDF page) that fit within `CHUNK_SIZE` pass through `split_documents` unchanged (1 unit = 1 chunk).
- [ ] Units that exceed `CHUNK_SIZE` are split residually using sentence-aware separators (`["\n\n", "\n", ". ", "! ", "? ", " ", ""]`); only the first residual piece keeps the `"{chave}: "` prefix.
- [ ] Empty-value JSON keys and blank/image-only PDF pages are silently filtered — no `Document`, no chunk, no log.
- [ ] `extrair_texto_pdf` returns `dict[str, str]` (one entry per page with extractable text); both callers (`cria_vectordb`, `salvar_pdf_enviado`) updated accordingly.
- [ ] `pdf_para_dict` logic from `scripts/pdf_to_json.py` relocates into the `agenticlog` package (merged into `extrair_texto_pdf`); `scripts/pdf_to_json.py` becomes a thin wrapper.
- [ ] `jq_schema` (without `join`) is a single shared constant in `config.py`, used identically by `cria_vectordb` and `adicionar_documento_incrementalmente`.
- [ ] `CLAUDE.md` "Build VectorDB" section generalized to cover rebuild triggers beyond `EMBEDDING_MODEL` (CHUNK_SIZE, CHUNK_OVERLAP, jq_schema, PDF-extraction logic).

## Out of Scope

| Feature | Reason |
|---------|--------|
| Bin-packing small JSON fields into shared chunks | ADR-009 — YAGNI, no real small-field case in current data |
| Logging when a Document is filtered for empty content | ADR-011 — silent by design, expected/normal case |
| Migration tooling for existing `data/vectordb/` | Full rebuild only; incremental ingestion skips files by content hash (ADR-010) |
| Re-prefixing residual-split continuation pieces with `"{chave}: "` | ADR-012 — only the first piece keeps the prefix |
| Nested / non-top-level JSON key chunking | Approved story scopes to top-level keys only |
| Changing `EMBEDDING_MODEL`, `CHUNK_SIZE`, or `CHUNK_OVERLAP` values | Approved story: chunking *strategy* changes, not these constants |

---

## User Stories

### P1: JSON chunked by top-level key MVP
**User Story**: As a developer maintaining the ingestion pipeline, I want each top-level key of a JSON document to become its own chunk, so retrieved chunks contain one complete field instead of an arbitrary character-count slice spanning multiple fields.

**Why P1**: Core value of the feature for JSON documents (`doc1.json`, `doc2.json`, `doc3.json`) — eliminates the most common retrieval-quality complaint (a field's text cut mid-sentence with the next field's label appended).

**Acceptance Criteria**:
1. WHEN a JSON document with multiple top-level keys (e.g. `doc1.json`, 6 keys) is ingested, THEN the system SHALL produce one `Document` per top-level key, each with `page_content = "{CHAVE}: {valor}"`.
2. WHEN a JSON-derived `Document`'s `page_content` length is `<= CHUNK_SIZE`, THEN `split_documents` SHALL pass it through unchanged as a single chunk.
3. WHEN a JSON-derived chunk is produced, THEN its `page_content` SHALL start with the literal top-level key followed by `": "` (e.g. `"DESCRIÇÃO: "`).
4. WHEN a JSON document has a single non-empty top-level key, THEN the system SHALL produce exactly 1 chunk with no special-casing relative to multi-key documents.
5. WHEN a JSON top-level key's `page_content` exceeds `CHUNK_SIZE`, THEN the system SHALL split it residually using `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]`, and only the FIRST resulting piece SHALL retain the `"{chave}: "` prefix.

**Independent Test**: Ingest `doc1.json` (6 top-level keys, all values 100-300 chars, all `<= CHUNK_SIZE`) via `cria_vectordb`; assert exactly 6 chunks are produced, each `page_content` starting with its respective key + `": "`, none truncated mid-field.

---

### P2: PDF chunked by page MVP
**User Story**: As a developer maintaining the ingestion pipeline, I want each PDF page to become its own chunk (pre-split), so retrieved chunks reflect one page of source material instead of a blob spanning multiple pages.

**Why P2**: Core value for PDF documents (`materiais_logistica.pdf`) — mirrors P1's benefit for the second supported document type; depends on the same shared filtering/splitting infrastructure as P1.

**Acceptance Criteria**:
1. WHEN a multi-page PDF is processed, THEN `extrair_texto_pdf` SHALL return `dict[str, str]` keyed `"PÁGINA_1"`, `"PÁGINA_2"`, ... for each page with extractable text.
2. WHEN `cria_vectordb` processes the dict returned by `extrair_texto_pdf`, THEN it SHALL build exactly one `Document(page_content=f"{chave}: {texto}")` per page entry, before any size-based splitting.
3. WHEN a PDF-derived `Document` (pre-split) is produced, THEN its `page_content` SHALL start with `"PÁGINA_{N}: "`.
4. WHEN a PDF page's `page_content` exceeds `CHUNK_SIZE`, THEN the system SHALL split it residually using `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]`, preferring sentence boundaries, and only the FIRST resulting piece SHALL retain the `"PÁGINA_{N}: "` prefix.

**Independent Test**: Ingest `materiais_logistica.pdf` via `cria_vectordb`; assert one pre-split `Document` exists per page with extractable text, each starting with `"PÁGINA_{N}: "`; assert any page exceeding `CHUNK_SIZE` is split into multiple chunks where only the first starts with the page prefix and chunk boundaries prefer `". "`/`"! "`/`"? "` over raw character cuts.

---

### P3: Shared jq_schema and silent empty-Document filtering
**User Story**: As a developer maintaining the ingestion pipeline, I want a single shared `jq_schema` constant (no `join`) used by both ingestion entry points, and empty-value JSON keys / blank PDF pages silently filtered out, so the chunking strategy is consistent and free of zero-content noise without extra logging overhead.

**Why P3**: Structural prerequisite for P1/P2 (shared constant avoids the duplication that exists today) and a correctness guard (ADR-011) — without it, empty fields/pages would become useless embeddings polluting retrieval.

**Acceptance Criteria**:
1. WHEN `to_entries | map(.key + ": " + (.value | tostring))` (no `join`) is used as `jq_schema`, THEN `JSONLoader` SHALL emit one list element = one `Document` per top-level key.
2. WHEN `cria_vectordb` and `adicionar_documento_incrementalmente` both build a `JSONLoader`, THEN they SHALL reference the same shared `jq_schema` constant from `config.py` — no duplicated literal.
3. WHEN a JSON top-level key has an empty value (`""`, `[]`, `{}`), THEN the resulting `Document` SHALL be silently filtered before splitting — no chunk produced, no log entry.
4. WHEN a PDF page among otherwise-valid pages is blank or image-only, THEN no `Document` SHALL be produced for that page — no chunk, no log entry.
5. WHEN all top-level keys of a JSON document have empty values, THEN zero `Document`s SHALL be produced for that file and no error SHALL be raised.
6. WHEN all pages of a PDF are blank/image-only AND `extrair_texto_pdf` is called from `cria_vectordb`, THEN `cria_vectordb` SHALL produce zero `Document`s for that PDF without raising (the existing `RAGSecurityError("PDF não contém texto extraível...")` raised by `extrair_texto_pdf` when the dict is empty continues to apply and is caught by `cria_vectordb`'s existing `except RAGSecurityError` — see Edge Cases).

**Independent Test**: Construct a JSON file with one empty-string-valued key and one normal key; ingest via `adicionar_documento_incrementalmente`; assert exactly 1 chunk is produced (for the normal key) and no warning/error is logged for the empty key.

---

### P4: PDF extraction relocation and thin CLI wrapper
**User Story**: As a developer maintaining the ingestion pipeline, I want the PDF-to-dict extraction logic to live in a single place inside the `agenticlog` package (not duplicated between `rag.py` and `scripts/pdf_to_json.py`), so `cria_vectordb`, `salvar_pdf_enviado`, and the standalone CLI converter share one source of truth.

**Why P4**: Structural cleanup required by ADR-008 before P2 can be implemented without duplicating page-extraction logic; lower priority than P1/P2/P3 because it has no behavioral effect on chunking output, but is a prerequisite for `extrair_texto_pdf`'s new `dict[str, str]` signature to be the single implementation.

**Acceptance Criteria**:
1. WHEN `extrair_texto_pdf(path: Path)` is called, THEN it SHALL return `dict[str, str]` (mirroring the current `pdf_para_dict` behavior: key `f"PÁGINA_{i+1}"` for each page where `page.get_text().strip()` is non-empty).
2. WHEN `cria_vectordb` calls `extrair_texto_pdf`, THEN it SHALL use the returned dict to build per-page `Document`s (P2-AC2).
3. WHEN `salvar_pdf_enviado` calls `extrair_texto_pdf` for validation, THEN it SHALL validate that the returned dict has at least one entry with non-empty text (`any(v.strip() for v in dict.values())` or equivalently a non-empty dict, since `extrair_texto_pdf` only includes non-empty-text pages).
4. WHEN `scripts/pdf_to_json.py` is invoked as a CLI, THEN its `pdf_para_dict` SHALL delegate to (or be replaced by an import of) the relocated `agenticlog` implementation — `converter()` and `main()` behavior (including `ensure_ascii=False` JSON output) SHALL be unchanged.

**Independent Test**: Call `extrair_texto_pdf` on a 2-page PDF where page 2 is blank; assert the returned dict has exactly one key (`"PÁGINA_1"`). Run `scripts/pdf_to_json.py sample.pdf --output /tmp` and confirm the produced JSON matches the dict returned by `extrair_texto_pdf` (same keys, same `ensure_ascii=False` formatting).

---

### P5: Documentation — generalized rebuild-trigger guidance
**User Story**: As a developer maintaining the AgenticLog repository, I want `CLAUDE.md`'s "Build VectorDB" section to describe rebuild triggers generically (any change to `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `jq_schema`, or PDF-extraction logic), so future chunking-strategy changes don't require re-documenting the same silent-degradation warning.

**Why P5**: Lowest priority — pure documentation change (ADR-010), no code or test impact, but required for the feature to be considered complete since this PR itself is a chunking-strategy change that triggers the warning it documents.

**Acceptance Criteria**:
1. WHEN `CLAUDE.md`'s "Build VectorDB" section header and body are read, THEN they SHALL state that a full rebuild of `data/vectordb/` is required after changing ANY of: `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `jq_schema`, or PDF-extraction logic — not just `EMBEDDING_MODEL`.
2. WHEN the "Silent-degradation risk" paragraph is read, THEN it SHALL describe the same risk (no error raised, results become unreliable, no warning in logs/UI) generically for any of the triggers in AC1, not solely `EMBEDDING_MODEL`.

**Independent Test**: Read `CLAUDE.md` "Build VectorDB" section; confirm `CHUNK_SIZE`, `CHUNK_OVERLAP`, `jq_schema`, and "PDF extraction" (or equivalent wording) are each mentioned as rebuild triggers alongside `EMBEDDING_MODEL`.

---

## Edge Cases

- WHEN a JSON top-level key has value `""`, `[]`, or `{}`, THEN no `Document`/chunk SHALL be produced for that key, silently (P3-AC3).
- WHEN a PDF page is blank or image-only (among otherwise-valid pages), THEN no `Document`/chunk SHALL be produced for that page, silently (P3-AC4).
- WHEN a JSON field's `page_content` length equals `CHUNK_SIZE` exactly, THEN it SHALL NOT be split (boundary is `> CHUNK_SIZE`, not `>=`).
- WHEN a JSON field or PDF page exceeds `CHUNK_SIZE` by a small margin, THEN residual splitting SHALL occur and only the first resulting piece SHALL retain the `"{chave}: "` / `"PÁGINA_{N}: "` prefix (P1-AC5, P2-AC4).
- WHEN `data/vectordb/` contains chunks produced by the OLD chunking strategy (1 chunk per file) AND the new code runs `adicionar_documento_incrementalmente` against an already-ingested file, THEN the old chunks are NOT automatically replaced — incremental ingestion detects the existing `source` + matching `content_hash` and returns `{"status": "duplicado", ...}` without re-chunking. A full `cria_vectordb()` rebuild is required to replace old-strategy chunks with new-strategy chunks (see Risks).
- WHEN a JSON document has a single non-empty top-level key, THEN exactly 1 chunk SHALL be produced — no special-casing vs. multi-key documents (P1-AC4).
- WHEN ALL top-level keys of a JSON document have empty values, THEN zero `Document`s SHALL be produced for that file and NO error SHALL be raised (P3-AC5). Note: if this is the ONLY file being processed by `adicionar_documento_incrementalmente`, the existing zero-chunks branch (`logger.warning(...)`, `saved_path.unlink()`, returns `{"status": "adicionado", "mensagem": "...0 chunks gerados."}`) applies — this is pre-existing behavior, unchanged by this feature.
- WHEN ALL pages of a PDF are blank/image-only, THEN `extrair_texto_pdf` returns an empty dict, which (per its existing contract — see ADR-008 consequences and current code lines 424-425) raises `RAGSecurityError("PDF não contém texto extraível (somente imagem).")`. In `cria_vectordb`, this is caught by the existing `except RAGSecurityError` around the PDF loop and logged via `logger.error("PDF corrompido ignorado durante reconstrução: ...")`, producing zero `Document`s for that PDF without raising — net effect matches "zero Documents, no error" at the `cria_vectordb` level, though `extrair_texto_pdf` itself does raise (caught immediately by its only structural caller in the PDF loop).

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| CHUNK-01 | P1-AC1 | Design | Pending |
| CHUNK-02 | P1-AC2 | Design | Pending |
| CHUNK-03 | P1-AC3 | Design | Pending |
| CHUNK-04 | P1-AC4 | Design | Pending |
| CHUNK-05 | P1-AC5 | Design | Pending |
| CHUNK-06 | P2-AC1 | Design | Pending |
| CHUNK-07 | P2-AC2 | Design | Pending |
| CHUNK-08 | P2-AC3 | Design | Pending |
| CHUNK-09 | P2-AC4 | Design | Pending |
| CHUNK-10 | P3-AC1 | Design | Pending |
| CHUNK-11 | P3-AC2 | Design | Pending |
| CHUNK-12 | P3-AC3 | Design | Pending |
| CHUNK-13 | P3-AC4 | Design | Pending |
| CHUNK-14 | P3-AC5 | Design | Pending |
| CHUNK-15 | P3-AC6 | Design | Pending |
| CHUNK-16 | P4-AC1 | Design | Pending |
| CHUNK-17 | P4-AC2 | Design | Pending |
| CHUNK-18 | P4-AC3 | Design | Pending |
| CHUNK-19 | P4-AC4 | Design | Pending |
| CHUNK-20 | P5-AC1 | Design | Pending |
| CHUNK-21 | P5-AC2 | Design | Pending |

**ID format:** `CHUNK-[NUMBER]`

---

## Approved Story Traceability

| Approved story AC # | Requirement ID(s) | ADR(s) | Notes |
|---------------------|-------------------|--------|-------|
| AC1 (JSON multi-key -> 1 Document/key, "CHAVE: valor") | CHUNK-01 | ADR-008, ADR-012 | |
| AC2 (Document <= CHUNK_SIZE -> unchanged single chunk) | CHUNK-02 | ADR-008 | |
| AC3 (chunk page_content starts with "CHAVE: ") | CHUNK-03 | ADR-012 | |
| AC4 (multi-page PDF -> dict[str,str] -> 1 Document/page) | CHUNK-06, CHUNK-07 | ADR-008 | |
| AC5 (PDF Document starts with "PÁGINA_{N}: ") | CHUNK-08 | ADR-012 | |
| AC6 (PDF page > CHUNK_SIZE -> residual split, sentence separators) | CHUNK-09 | ADR-007 | |
| AC7 (JSON field > CHUNK_SIZE -> residual split, same separators) | CHUNK-05 | ADR-007 | |
| AC8 (empty JSON key -> silent filter) | CHUNK-12 | ADR-011 | |
| AC9 (blank/image PDF page -> silent filter) | CHUNK-13 | ADR-011 | |
| AC10 (jq_schema = to_entries\|map(...) , no join) | CHUNK-10 | ADR-008 | |
| AC11 (shared jq_schema constant, both callers) | CHUNK-11 | ADR-008 | |
| AC12 (extrair_texto_pdf -> dict[str,str], both callers updated) | CHUNK-06, CHUNK-18 | ADR-008 | |
| AC13 (pdf_para_dict moves into agenticlog package) | CHUNK-16, CHUNK-19 | ADR-008 | |
| AC14 (no bin-packing) | (Out of Scope confirmation) | ADR-009 | No requirement ID -- confirms ADR-009 is honored, not a new behavior |
| AC15 (residual split -> only first piece keeps prefix) | CHUNK-05, CHUNK-09 | ADR-012 | |
| AC16 (CLAUDE.md rebuild section generalized) | CHUNK-20, CHUNK-21 | ADR-010 | |

---

## Data Model Changes

### `src/agenticlog/config.py` — new constant

| Constant | Type | Value | Used by |
|----------|------|-------|---------|
| `JQ_SCHEMA_CAMPOS_JSON` | `str` | `'to_entries \| map(.key + ": " + (.value \| tostring))'` (no `join`) | `cria_vectordb`, `adicionar_documento_incrementalmente` (replaces both inline `jq_schema` literals at rag.py:353 and rag.py:521) |

No new constants for separators — the `RecursiveCharacterTextSplitter(separators=[...])` list from ADR-007 is added inline at both `RecursiveCharacterTextSplitter(...)` instantiations (rag.py:356-359 and rag.py:544-548), consistent with `CHUNK_SIZE`/`CHUNK_OVERLAP` already being inline kwargs there. No separate config constant is introduced for the separators list (YAGNI — not reused elsewhere, and `CHUNK_SIZE`/`CHUNK_OVERLAP` precedent is also inline at the call site, only the *values* live in config.py, not the full splitter kwargs).

### `src/agenticlog/rag.py` — function signature change

| Function | Before | After |
|----------|--------|-------|
| `extrair_texto_pdf(path: Path)` | returns `str` (concatenated all-page text) | returns `dict[str, str]` (`{"PÁGINA_1": "...", "PÁGINA_2": "...", ...}`, only pages with non-empty `.strip()` text) |

No ChromaDB schema/metadata changes. No Pydantic model changes. No database migrations.

---

## Process / Background Flow

### Happy path — `cria_vectordb` (full rebuild)

1. `_valida_path_documentos()` and `_valida_arquivos_json()` run unchanged.
2. `DirectoryLoader` + `JSONLoader` load `*.json` files using `JQ_SCHEMA_CAMPOS_JSON` (no `join`) → `JSONLoader` emits one `Document` per top-level key per file, `page_content = "{CHAVE}: {valor}"`.
3. Filter: `json_docs = [d for d in json_docs if d.page_content.strip()]` — drops `Document`s for empty-value keys (ADR-011).
4. For each `*.pdf` in `DIR_DOCUMENTS`:
   a. `pages = extrair_texto_pdf(pdf_path)` → `dict[str, str]`.
   b. For each `(chave, texto)` in `pages.items()`: build `Document(page_content=f"{chave}: {texto}", metadata={"source": str(pdf_path)})`.
   c. `RAGSecurityError` from `extrair_texto_pdf` (e.g. all-blank PDF, password-protected, corrupted) is caught per-file as today, logged via `logger.error("PDF corrompido ignorado durante reconstrução: %s — %s", pdf_path.name, e)`, and that PDF contributes zero `Document`s.
5. Filter PDF docs: `pdf_docs = [d for d in pdf_docs if d.page_content.strip()]` (defensive — `extrair_texto_pdf` already excludes empty-text pages, but applying the same filter keeps the invariant uniform per ADR-011).
6. `documents = json_docs + pdf_docs`. If empty → `logger.warning("Nenhum documento encontrado.")`, return early (unchanged).
7. `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""])` splits `documents` → `chunks`. Units `<= CHUNK_SIZE` pass through unchanged (1 unit = 1 chunk); units `> CHUNK_SIZE` are split residually with sentence-preferring separators, only the first piece retaining the `"{chave}: "`/`"PÁGINA_{N}: "` prefix (default `RecursiveCharacterTextSplitter` behavior — no re-prefixing code needed).
8. Embeddings + `Chroma.from_documents(...)` as today.

### Happy path — `adicionar_documento_incrementalmente` (incremental, JSON only)

1. Validations, hash computation, dedup check unchanged.
2. `JSONLoader(str(saved_path), jq_schema=JQ_SCHEMA_CAMPOS_JSON)` (shared constant, no `join`) → one `Document` per top-level key.
3. Filter: `docs = [d for d in docs if d.page_content.strip()]` (ADR-011).
4. `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""])` splits filtered `docs` → `chunks`.
5. If `chunks` is empty (e.g. all keys had empty values, or file genuinely produced 0 chunks): existing zero-chunks branch applies unchanged (`logger.warning`, `saved_path.unlink()`, return `{"status": "adicionado", "mensagem": "...0 chunks gerados."}`).
6. `chunk.metadata["content_hash"] = hash_str` for each chunk, `add_documents` etc. — unchanged.

### Failure path — all-blank PDF (`cria_vectordb`)

1. `extrair_texto_pdf(pdf_path)` builds `pages = {}` (no page has non-empty `.strip()` text).
2. Per existing contract (lines 424-425, preserved): `if not pages: raise RAGSecurityError("PDF não contém texto extraível (somente imagem).")`.
3. `cria_vectordb`'s PDF loop catches `RAGSecurityError`, logs `logger.error("PDF corrompido ignorado durante reconstrução: %s — %s", pdf_path.name, e)`, contributes zero `Document`s for that PDF. Other JSON/PDF documents continue processing normally.

### Failure path — `salvar_pdf_enviado` validation (uploaded PDF)

1. `extrair_texto_pdf(tmp_path)` now returns `dict[str, str]`.
2. `salvar_pdf_enviado` validates the dict has at least one page with extractable text. Since `extrair_texto_pdf` only includes pages with non-empty `.strip()` text, a non-empty dict is sufficient: if `extrair_texto_pdf` raises `RAGSecurityError` (all-blank PDF → empty `pages` dict, per its existing contract), that propagates as today — `salvar_pdf_enviado`'s existing `except RAGSecurityError: ... raise` path is unchanged. No new validation code is needed beyond consuming the dict instead of discarding a string return value.

### Failure path — JSON field/PDF page with empty value

1. `JSONLoader` (JSON) or the PDF per-page loop (PDF) produces a `Document` with `page_content` that is empty or whitespace-only after `.strip()` (e.g. `"CHAVE: "`, `"CHAVE: []"` — note: `"CHAVE: []"` has non-empty `.strip()` so it is NOT filtered; only `page_content.strip() == ""` cases are filtered, matching ADR-011's literal filter `docs = [d for d in docs if d.page_content.strip()]`. For PDF, `extrair_texto_pdf` already excludes empty-text pages at the `pdf_para_dict`-derived layer, so the `Document`-level filter is defensive/structural consistency, not expected to trigger in practice for PDF.)
2. Filtered out before `split_documents` — no chunk, no log (silent by design).

---

## API Changes

No HTTP API changes (`src/agenticlog/api.py` not touched by this feature).

### `src/agenticlog/rag.py` — function signature changes

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
```

### `src/agenticlog/config.py` — new constant

```python
# jq_schema compartilhado: 1 entrada de lista por chave top-level -> 1 Document por chave (ADR-008)
JQ_SCHEMA_CAMPOS_JSON = 'to_entries | map(.key + ": " + (.value | tostring))'
```

### `scripts/pdf_to_json.py` — thin wrapper

`pdf_para_dict` either re-exports or thinly delegates to `agenticlog.rag.extrair_texto_pdf` (exact mechanism decided in design.md — must preserve the existing `ValueError` contract for the CLI script if `extrair_texto_pdf` now raises `RAGSecurityError` instead, see Risks).

---

## Frontend Changes

No frontend changes. This is a backend-only ingestion-pipeline change (`rag.py`, `config.py`, `scripts/pdf_to_json.py`). `app.py` is not modified — it calls `adicionar_documento_incrementalmente` and `salvar_pdf_enviado`, whose external call signatures (parameters and return dict shape) are unchanged by this feature.

---

## Tests Required

### Unit — `tests/test_rag.py`

**`TestCriaVectordb`** (rewrite both existing methods + add new):
- `test_cria_vectordb_sem_documentos_retorna_cedo` — unchanged assertions, but mock `JSONLoader`/`DirectoryLoader` consistent with new jq_schema (no behavior change to this specific test since it returns `[]`).
- `test_cria_vectordb_com_documentos_cria_chroma` — update mocked `DirectoryLoader.load()` to return multiple `Document`s (one per simulated top-level key) reflecting the new jq_schema; assert `RecursiveCharacterTextSplitter` is called with `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]` (via `mock_splitter.assert_called_once_with(chunk_size=..., chunk_overlap=..., separators=[...])` or equivalent).
- New: PDF loop produces 1 `Document` per page from `extrair_texto_pdf` dict, each `page_content = f"{chave}: {texto}"`.
- New: empty-value JSON `Document` (`page_content.strip() == ""`) is filtered before `split_documents` — assert it is NOT in the list passed to the splitter.
- New: blank PDF page (among valid pages) does not produce a `Document`.
- New: all-blank PDF → `extrair_texto_pdf` raises `RAGSecurityError`, caught by existing `except RAGSecurityError` in PDF loop, `logger.error` called, zero PDF `Document`s, JSON docs (if any) still processed.

**`TestExtrairTextoPdf`** (rewrite all 5 methods for `dict[str, str]` return):
- `teste_1_extrair_pdf_valido_retorna_texto` — assert `resultado == {"PÁGINA_1": "texto do contrato"}` (exact dict, not substring).
- `teste_2_extrair_pdf_com_senha_lanca_erro` — unchanged (raises before dict construction).
- `teste_3_extrair_pdf_somente_imagem_lanca_erro` — both pages blank → `pages == {}` → `RAGSecurityError("...somente imagem.")`.
- `teste_4_extrair_pdf_mix_texto_imagem_aceita` — 2-page mock (page 1 has text, page 2 blank) → assert `resultado == {"PÁGINA_1": "conteúdo real"}` (only page 1 key present, page 2 silently excluded).
- `teste_5_extrair_exception_generica_lanca_erro` — unchanged.
- New: 3-page mock, all pages have text → assert `resultado == {"PÁGINA_1": ..., "PÁGINA_2": ..., "PÁGINA_3": ...}` (multi-page dict ordering/keys).

**`TestSalvarPdfEnviado`** (update mocks):
- `teste_1_salvar_pdf_valido_sucesso` — `mock_extrair.return_value = {"PÁGINA_1": "texto extraído"}` (was `"texto extraído"`).
- `teste_3_salvar_aceita_extensao_maiuscula` — same dict-return update.
- `teste_8_salvar_rollback_se_pdf_invalido` — unchanged (`side_effect = RAGSecurityError(...)`, return value never reached).
- All other `TestSalvarPdfEnviado` methods — no change needed (don't depend on `extrair_texto_pdf` return value).

**`TestAdicionarDocumentoIncrementalmente`** (update + add):
- All 9 existing methods continue to mock `JSONLoader`/`RecursiveCharacterTextSplitter` directly with canned `LCDocument` lists — these remain green as long as mocks are not asserting on the OLD `jq_schema` string literal (none currently do, per researcher findings).
- New: assert `JSONLoader` is constructed with `jq_schema=JQ_SCHEMA_CAMPOS_JSON` (imported from `agenticlog.config`) — locks in CHUNK-11 (shared constant).
- New: assert `RecursiveCharacterTextSplitter` is constructed with `separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]`.
- New: empty-value-key filter — `JSONLoader.load()` returns a mix of empty and non-empty `page_content` `Document`s; assert only non-empty ones reach `split_documents`.

### New — `scripts/pdf_to_json.py` / relocation tests

- Existing tests for `pdf_para_dict`/`converter`/`main` (if any exist under `tests/` for `scripts/`) must be located and updated to match the new delegation. **Researcher found no existing test file for `scripts/pdf_to_json.py`** — if true, add minimal new tests asserting `pdf_para_dict` (or its replacement) produces output identical to `agenticlog.rag.extrair_texto_pdf` for the same PDF, and that `converter()` still writes `ensure_ascii=False, indent=2` JSON.

### Integration — `tests/test_rag_integration.py`

- `TestIngestionIntegration` teste_1, 2, 3, 5, 6, 7 — currently use real small JSON like `{"produto": "cadeira", "cor": "azul"}` (2 top-level keys) WITHOUT mocking `JSONLoader`/splitter. With the new `jq_schema`, each of these now produces 2 `Document`s (1 per key) → 2 chunks (both well under `CHUNK_SIZE`). Current assertions only check `status == "adicionado"` and file existence — these SHOULD remain green, but must be re-run to confirm (`pytest -m integration tests/test_rag_integration.py -v`). No assertion changes anticipated, but flagged as a verification task (see tasks.md).
- `teste_4_rollback_nao_deixa_chunks_orfaos` — already mocks `JSONLoader`/`RecursiveCharacterTextSplitter` explicitly; update mock to use `JQ_SCHEMA_CAMPOS_JSON` if asserted (currently not asserted — no change needed unless new assertions are added).

### Edge case tests (new, across `TestCriaVectordb` / `TestAdicionarDocumentoIncrementalmente`)

- JSON field with `page_content` length exactly `== CHUNK_SIZE` → not split (1 chunk).
- JSON field with `page_content` length `CHUNK_SIZE + small_margin` → residual split into 2+ chunks; only first chunk's `page_content` starts with `"{chave}: "`.
- JSON document where ALL top-level keys have empty values → 0 `Document`s, 0 chunks, no error, no log (for `cria_vectordb`); for `adicionar_documento_incrementalmente`, existing zero-chunks branch (`logger.warning`, file removed, `status: "adicionado"`, `"0 chunks"` in mensagem).
- All-blank PDF → 0 `Document`s contributed to `cria_vectordb`, `logger.error` called once, no `SystemExit`/unhandled exception.

### Existing tests that will break (must be updated, not just re-verified)

- `tests/test_rag.py::TestCriaVectordb::test_cria_vectordb_com_documentos_cria_chroma` — current mock returns 1 `Document`, splitter call assertions don't check `separators` kwarg.
- `tests/test_rag.py::TestExtrairTextoPdf` — all 5 methods assert string return (`assertIn`); must become dict equality assertions.
- `tests/test_rag.py::TestSalvarPdfEnviado::teste_1_salvar_pdf_valido_sucesso` and `teste_3_salvar_aceita_extensao_maiuscula` — `mock_extrair.return_value = "texto extraído"` must become a dict.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Add constant | New `JQ_SCHEMA_CAMPOS_JSON` (no `join`), shared by both ingestion entry points (CHUNK-10, CHUNK-11) |
| `src/agenticlog/rag.py` | Modify functions | `adicionar_documento_incrementalmente` (jq_schema constant, separators, empty-doc filter), `extrair_texto_pdf` (return `dict[str, str]`, relocated `pdf_para_dict` logic), `salvar_pdf_enviado` (consume dict for validation), `cria_vectordb` (jq_schema constant, separators, per-page PDF Documents, empty-doc filters for both JSON and PDF) |
| `scripts/pdf_to_json.py` | Refactor | `pdf_para_dict` becomes thin wrapper delegating to relocated `agenticlog.rag.extrair_texto_pdf`; `converter`/`main` unchanged in behavior (CHUNK-16, CHUNK-19) |
| `tests/test_rag.py` | Modify classes | `TestCriaVectordb` (rewrite 2 methods + add new), `TestExtrairTextoPdf` (rewrite 5 methods + add 1), `TestSalvarPdfEnviado` (update 2 methods' mocks), `TestAdicionarDocumentoIncrementalmente` (add assertions for shared jq_schema/separators/filter) |
| `tests/test_rag_integration.py` | Spot-check | Re-run `TestIngestionIntegration` teste_1,2,3,5,6,7 to confirm multi-Document-per-file behavior doesn't break existing `status`/file-existence assertions |
| `CLAUDE.md` | Modify section | "Build VectorDB" header + "Silent-degradation risk" paragraph generalized per ADR-010 (CHUNK-20, CHUNK-21) |
| `data/vectordb/` | No code change — operational | Full rebuild required post-merge (gitignored, regenerable; not part of this PR's file changes but documented as a deployment step) |

---

## Risks

| Risk | Severity | Status | Mitigation |
|------|----------|--------|------------|
| Old vectordb chunks (1-per-file, pre-feature) coexist with new chunks (1-per-key/page) if `data/vectordb/` is not fully rebuilt — incremental ingestion (`adicionar_documento_incrementalmente`) skips already-ingested files via content-hash dedup, so old-strategy chunks for those files are NEVER replaced without `cria_vectordb()` full rebuild | High | Found — documented (ADR-010) | CLAUDE.md generalized rebuild section (P5) makes this explicit; out of scope to build auto-migration tooling (Out of Scope table) — operational runbook step, not a code gate |
| `extrair_texto_pdf` signature change (`str` -> `dict[str, str]`) is a breaking change for ANY caller not updated — researcher confirmed only 2 callers exist (`cria_vectordb`, `salvar_pdf_enviado`) and `pdf_para_dict` in `scripts/pdf_to_json.py` has no other callers, but a missed caller would raise `TypeError` (e.g. string concatenation on a dict) at runtime, not at import time | Medium | Clear | Both callers explicitly updated in this spec (P2-AC2, P4-AC3); `Grep` for `extrair_texto_pdf(` across `src/`, `tests/`, `app.py` should be re-run during implementation as a final check |
| `scripts/pdf_to_json.py`'s `pdf_para_dict` currently raises plain `ValueError` on empty dict; relocated `extrair_texto_pdf` raises `RAGSecurityError` (subclass of `Exception`, not `ValueError`) on the same condition — if the CLI script's `main()` `except Exception` catch-all (line 66) doesn't differentiate, behavior is preserved at the CLI level (both are caught generically), but any code relying on catching `ValueError` specifically from `pdf_para_dict` would break | Low | Clear | `scripts/pdf_to_json.py` has no other callers (researcher confirmed); CLI's existing `except Exception as exc` at line 66 catches both exception types identically — no CLI behavior change. Design.md must specify exact wrapper shape to avoid silent divergence |
| `tests/test_rag_integration.py` teste_1,2,3,5,6,7 use REAL (non-mocked) `JSONLoader`/`RecursiveCharacterTextSplitter` against small 2-key JSON dicts — new jq_schema produces 2 `Document`s/chunks per file instead of 1; current assertions (`status == "adicionado"`, file existence) don't assert chunk counts, so likely remain green, but must be explicitly re-run as part of this feature's test gate, not assumed | Medium | Found — flagged for verification | Spot-check task in tasks.md; if any assertion does break, fix is mechanical (update expected counts), not a design change |
| `RecursiveCharacterTextSplitter(separators=[...])` change (ADR-007) alters split boundaries for ANY existing chunk that was already `> CHUNK_SIZE` under the OLD splitter (default separators `["\n\n", "\n", " ", ""]`) — for current data (`doc1-3.json` fields all `100-300` chars, `materiais_logistica.pdf` pages likely `> CHUNK_SIZE`), this primarily affects PDF page residual splits | Low | Found — documented (ADR-007 consequences) | Acceptable per ADR-007 — sentence-aware boundaries are the intended improvement; no mitigation needed beyond test coverage of P1-AC5/P2-AC4 |
| Filter `docs = [d for d in docs if d.page_content.strip()]` (ADR-011) is applied identically in 3 places (`cria_vectordb` JSON path, `cria_vectordb` PDF path, `adicionar_documento_incrementalmente`) — risk of copy-paste divergence (e.g. one site uses `.strip()`, another forgets it) | Low | Clear | Design.md specifies exact one-line filter expression to copy verbatim at all 3 sites; code review / test coverage at each site catches divergence |
| CLAUDE.md conflicts — none found. The "Build VectorDB" section is the only place documenting rebuild triggers; ADR-010 scope is additive (generalize wording), no contradiction with other CLAUDE.md sections (Architecture, Testing Conventions, Git Workflow unaffected) | None | Clear | N/A |
| Multi-tenancy / timezone / auth / retry — not applicable to this feature (pure ingestion-chunking change, no new external calls, no user-facing time-sensitive data, no auth boundary touched) | None | Clear | N/A |

---

## Open Questions

None. All architectural decisions are pre-resolved by ADR-007 through ADR-012 (read in full before writing this spec):

| Topic | Resolution | ADR |
|-------|-----------|-----|
| Sentence-aware separators for residual splits | `["\n\n", "\n", ". ", "! ", "? ", " ", ""]` added to both `RecursiveCharacterTextSplitter` instantiations | ADR-007 |
| JSON jq_schema (no join) + PDF dict-per-page | `to_entries \| map(.key + ": " + (.value \| tostring))`, shared constant; `extrair_texto_pdf` -> `dict[str, str]` | ADR-008 |
| Bin-packing small fields | Not implemented (YAGNI) | ADR-009 |
| CLAUDE.md rebuild section scope | Generalized to cover EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, jq_schema, PDF-extraction logic | ADR-010 |
| Empty-Document filtering | Silent, no log, `docs = [d for d in docs if d.page_content.strip()]` | ADR-011 |
| page_content format contract | `"{chave}: {valor}"` for both JSON and PDF; only first residual piece keeps prefix | ADR-012 |

The only implementation-level decision left to design.md (not a product/architecture decision, purely a code-organization detail already scoped by the researcher's recommendation) is the EXACT shape of the `scripts/pdf_to_json.py` thin wrapper — resolved in design.md "PDF extraction relocation" section.

---

## Success Criteria

- [ ] `pytest tests/test_rag.py -v` passes with all rewritten/new test methods green (TestCriaVectordb, TestExtrairTextoPdf, TestSalvarPdfEnviado, TestAdicionarDocumentoIncrementalmente).
- [ ] `pytest -m integration tests/test_rag_integration.py -v` passes (teste_1-7), confirming multi-Document-per-file behavior doesn't break existing assertions.
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` reports >= 80% coverage, with `rag.py` coverage not regressing.
- [ ] Ingesting `doc1.json` (6 top-level keys, all `<= CHUNK_SIZE`) via `cria_vectordb` produces exactly 6 chunks, each starting with its key + `": "`.
- [ ] Ingesting `materiais_logistica.pdf` via `cria_vectordb` produces 1 pre-split `Document` per page with extractable text, each starting with `"PÁGINA_{N}: "`; any page `> CHUNK_SIZE` is split with sentence-preferring boundaries, only the first piece retaining the prefix.
- [ ] `JQ_SCHEMA_CAMPOS_JSON` constant exists in `config.py` and is the ONLY `jq_schema` literal referenced by `cria_vectordb` and `adicionar_documento_incrementalmente` (no duplicated jq strings remain).
- [ ] `extrair_texto_pdf` returns `dict[str, str]`; `scripts/pdf_to_json.py`'s `pdf_para_dict` delegates to it (no duplicated PyMuPDF page-iteration logic between the two files).
- [ ] `CLAUDE.md` "Build VectorDB" section explicitly lists `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `jq_schema`, and PDF-extraction logic as rebuild triggers.
- [ ] `python -m agenticlog.rag` (full rebuild) and `python scripts/pdf_to_json.py <file>.pdf` both run without error against `data/documents/`.
