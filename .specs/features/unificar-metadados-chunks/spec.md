# Spec: Unificar Metadados de Chunks

**Path:** `.specs/features/unificar-metadados-chunks/spec.md`
**TLC scope:** medium
**Based on story:** Every chunk stored in the vector database carries five consistent metadata fields across all three chunk-generation sites.
**Status:** Awaiting human approval

---

## Overview

Every chunk written to ChromaDB must carry exactly five metadata fields — `source`, `file_hash`, `chunk_index`, `page`, `doc_type` — regardless of which ingest path produced it. This eliminates conditional metadata logic in all downstream features (upsert, incremental PDF ingest, CLI).

---

## Requirements

### Functional Requirements

| ID | Requirement | Traces to AC |
|----|-------------|--------------|
| REC-01-FR-01 | Every chunk written by `cria_vectordb` (JSON path) SHALL have `source` (non-empty string), `file_hash` (SHA-256 hex, 64 chars), `chunk_index` (0-based integer, sequential per file), `page=0` (integer sentinel), `doc_type="json"`. | AC-01, AC-07, AC-08 |
| REC-01-FR-02 | Every chunk written by `cria_vectordb` (PDF path) SHALL have `source`, `file_hash` (SHA-256 hex, 64 chars), `chunk_index` (0-based integer, sequential across entire PDF), `page` (1-based integer from `extrair_texto_pdf` `PÁGINA_N` key), `doc_type="pdf"`. | AC-02, AC-07, AC-08, AC-09 |
| REC-01-FR-03 | Every chunk written by `adicionar_documento_incrementalmente` (JSON path) SHALL have `source`, `file_hash` (SHA-256 hex, 64 chars), `chunk_index` (0-based integer, sequential per file), `page=0`, `doc_type="json"`. | AC-03, AC-07, AC-08 |
| REC-01-FR-04 | The metadata field name used when writing the hash (`file_hash`) SHALL be identical to the field name read by the dedup query in `adicionar_documento_incrementalmente`. The current `content_hash` field SHALL be renamed to `file_hash` in both write and read sites. | AC-11 |
| REC-01-FR-05 | `file_hash` SHALL equal `hashlib.sha256(Path(chunk.metadata["source"]).read_bytes()).hexdigest()` exactly — computed from the file bytes, not the content string. | AC-10 |
| REC-01-FR-06 | `chunk_index` SHALL form a contiguous 0-based integer sequence per file with no gaps and no duplicates. Two files ingested in the same call have independent sequences each starting at 0. | AC-08 |
| REC-01-FR-07 | `chunk_index` for PDF chunks SHALL be sequential across the entire PDF document — not reset per page. | AC-09 |
| REC-01-FR-08 | No metadata field among the five SHALL be absent or `None` in any stored chunk. `page=0` is the valid sentinel for non-PDF sources (ChromaDB does not accept `None` as a metadata value). | AC-07 |
| REC-01-FR-09 | When PDF extraction raises `RAGSecurityError`, the PDF SHALL be skipped and no chunks written — existing behaviour preserved. | AC-04 |
| REC-01-FR-10 | When `adicionar_documento_incrementalmente` detects a duplicate by `file_hash` (same hash), it SHALL return `"duplicado"` and write no new chunks — existing behaviour preserved. | AC-05 |
| REC-01-FR-11 | When `adicionar_documento_incrementalmente` detects a hash mismatch (different hash for same source), it SHALL return `"hash_diferente"` and write no new chunks — existing behaviour preserved. | AC-06 |
| REC-01-FR-12 | A JSON file producing zero chunks after splitting SHALL not mutate any metadata — early-return path preserved. | Edge case |
| REC-01-FR-13 | A single-chunk JSON file SHALL have `chunk_index=[0]`. A PDF with one extractable page SHALL have `page` equal to that page's 1-based number and `chunk_index` starting at 0. | Edge cases |

### Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| REC-01-NFR-01 | `file_hash` computation for the JSON path in `cria_vectordb` SHALL read each unique source file at most once (O(unique files), not O(chunks)). |
| REC-01-NFR-02 | No new public API surface introduced — all changes are internal to `rag.py`. |
| REC-01-NFR-03 | Functions in `rag.py` SHALL remain within the 50-line soft limit defined in project coding style. If metadata enrichment causes `cria_vectordb` or `adicionar_documento_incrementalmente` to exceed 50 lines, a private helper `_enriquecer_metadados_chunks` SHALL be extracted. |
| REC-01-NFR-04 | Metadata field name constants SHALL be defined in `config.py` if the same string literal appears in more than one source file. |
| REC-01-NFR-05 | Test coverage for `rag.py` SHALL remain at or above the 80% project minimum after this change. |

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/rag.py` | Modify | Three chunk-generation sites require metadata enrichment; `content_hash` renamed to `file_hash`; dedup read site updated to match. |
| `src/agenticlog/config.py` | Modify (conditional) | Add metadata field name constants if the same string literal appears in more than one source file. |
| `tests/test_rag.py` | Modify | Add metadata field assertions to existing test methods; update dedup mock setup from `content_hash` to `file_hash`; add new tests for sequential `chunk_index`, `doc_type`, `page` sentinel, and SHA-256 hash value. |

---

## Data Model Changes

### New metadata fields on ChromaDB chunks

| Field | Type | Constraint | Notes |
|-------|------|------------|-------|
| `source` | string | Non-empty | Already present in all paths; no change. |
| `file_hash` | string | SHA-256 hex, exactly 64 chars | Replaces `content_hash` in `adicionar_documento_incrementalmente`; new field for both `cria_vectordb` paths. |
| `chunk_index` | integer | 0-based, contiguous, per-file | New field in all three sites. |
| `page` | integer | 1-based for PDF; 0 as JSON sentinel | New field in all three sites. `0` used instead of `None` (ChromaDB does not store `None` in metadata). |
| `doc_type` | string | `"json"` or `"pdf"` | New field in all three sites. |

### Migration note

After deploying this change, `data/vectordb/` MUST be deleted and rebuilt with `python -m agenticlog.rag`. Existing chunks stored under `content_hash` will not be found by the updated dedup logic (which reads `file_hash`). This follows the silent-degradation policy documented in CLAUDE.md — the rebuild is mandatory, not optional.

---

## Implementation Notes

### Three change sites in `rag.py`

**Site 1 — `cria_vectordb` JSON path (DirectoryLoader + JSONLoader)**

After `text_splitter.split_documents(docs)`, group chunks by `chunk.metadata["source"]`. For each unique source path:
1. Compute `file_hash = _computar_hash_conteudo(Path(source).read_bytes())` once.
2. Enumerate chunks within the group to assign `chunk_index` starting at 0.
3. Set `chunk.metadata["page"] = 0` and `chunk.metadata["doc_type"] = "json"` on each chunk.

Each source file is read at most once for hashing (O(unique files)).

**Site 2 — `cria_vectordb` PDF path (manual Document creation per page)**

The `extrair_texto_pdf` loop iterates over keys of the form `"PÁGINA_N"`. The page number `N` must be extracted as an integer and stored on the pre-split `Document`'s metadata (`page=N`) during construction. LangChain propagates `metadata` from parent `Document` to split chunks automatically.

After `text_splitter.split_documents(pdf_docs)`:
1. Compute `file_hash = _computar_hash_conteudo(pdf_path.read_bytes())` once per PDF.
2. Set `chunk.metadata["file_hash"] = file_hash` and `chunk.metadata["doc_type"] = "pdf"` on all chunks.
3. Enumerate all chunks across the entire PDF to assign a global `chunk_index` (not reset per page).

**Site 3 — `adicionar_documento_incrementalmente` JSON path**

1. Rename the written field from `content_hash` to `file_hash` in the `chunk.metadata` assignment.
2. Update the dedup read from `.get("content_hash")` to `.get("file_hash")`.
3. Enumerate chunks to assign `chunk_index` (0-based).
4. Set `chunk.metadata["page"] = 0` and `chunk.metadata["doc_type"] = "json"` on each chunk.

### Reuse existing patterns

- Hash computation: reuse `_computar_hash_conteudo(conteudo: bytes) -> str` (rag.py lines 265–271). Do not duplicate.
- Metadata loop pattern: `for chunk in chunks: chunk.metadata["key"] = value` already present at rag.py line 374.
- If function lengths exceed 50 lines after changes, extract a private helper: `_enriquecer_metadados_chunks(chunks: list, file_hash: str, doc_type: str) -> list` — sets `file_hash`, `doc_type`, and `chunk_index` via `enumerate`; caller sets `page` separately or passes it as an argument.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `content_hash` rename breaks dedup on existing vectordb | HIGH | Document mandatory `data/vectordb/` rebuild in PR description. CLAUDE.md silent-degradation policy already covers this class of change. |
| `page=0` sentinel semantics ambiguous | MEDIUM | Document in code comment: `page=0` means "not applicable — JSON source". Downstream queries filtering by page must use `page > 0` for PDF-only results. |
| `cria_vectordb` JSON hash reads source files after split | LOW | Files are local; one read per unique file. Acceptable for this deployment scale. |
| `page` extraction from `PÁGINA_N` key format changes | LOW | Use `int(key.split("_")[1])` defensively; raise `RAGSecurityError` or log-and-skip if integer parse fails — consistent with existing security-first approach in `rag.py`. |
| Function length violation after changes | LOW | Extract `_enriquecer_metadados_chunks` helper if either function exceeds 50 lines. |

---

## Open Questions

None. All field names, types, constraints, and sentinel values are resolved by the approved user story and researcher findings.

---

## Test Plan

### Tests to update in `tests/test_rag.py`

**`TestCriaVectordb`** — existing methods (5):
- After each call, inspect `mock_chroma.from_documents.call_args[0][0]` (the chunks list).
- Assert every chunk has all 5 fields present and not `None`.
- Assert `doc_type == "json"` for JSON-path tests; `doc_type == "pdf"` for PDF-path tests.
- Assert `page == 0` for JSON chunks; `page >= 1` (integer) for PDF chunks.
- Assert `file_hash` is a 64-character lowercase hex string.

**`TestAdicionarDocumentoIncrementalmente`** — dedup methods:
- Update any mock `existing["metadatas"][0]` that currently sets `content_hash` to set `file_hash` instead.
- Assert chunks passed to `collection.add` carry `file_hash`, not `content_hash`.

### New tests to add in `tests/test_rag.py`

| Test name | What it asserts |
|-----------|-----------------|
| `teste_N_chunk_index_sequencial_json` | Two-chunk JSON produces `chunk_index` values `[0, 1]`, no gaps |
| `teste_N_chunk_index_single_chunk_json` | Single-chunk JSON produces `chunk_index == 0` |
| `teste_N_chunk_index_sequencial_pdf` | Multi-page PDF produces global `chunk_index` not reset per page |
| `teste_N_file_hash_sha256_correto` | `file_hash` equals `hashlib.sha256(file_bytes).hexdigest()` |
| `teste_N_page_sentinel_json` | All JSON chunks have `page == 0` |
| `teste_N_page_inteiro_pdf` | PDF chunks have `page` matching the source `PÁGINA_N` key integer |
| `teste_N_doc_type_json` | JSON-path chunks have `doc_type == "json"` |
| `teste_N_doc_type_pdf` | PDF-path chunks have `doc_type == "pdf"` |
| `teste_N_zero_chunks_sem_mutacao` | JSON producing zero chunks returns without metadata assignment |
| `teste_N_dois_arquivos_chunk_index_independente` | Two JSON files ingested together have independent `chunk_index` sequences, each starting at 0 |

### Test execution gate

`pytest --cov=agenticlog --cov-report=term-missing -v` must pass with coverage >= 80% after all changes.

---

## Requirement Traceability

| Requirement ID | Acceptance Criteria | Status |
|----------------|---------------------|--------|
| REC-01-FR-01 | AC-01, AC-07, AC-08 | Pending |
| REC-01-FR-02 | AC-02, AC-07, AC-08, AC-09 | Pending |
| REC-01-FR-03 | AC-03, AC-07, AC-08 | Pending |
| REC-01-FR-04 | AC-11 | Pending |
| REC-01-FR-05 | AC-10 | Pending |
| REC-01-FR-06 | AC-08, edge case (two files, independent sequences) | Pending |
| REC-01-FR-07 | AC-09 | Pending |
| REC-01-FR-08 | AC-07 | Pending |
| REC-01-FR-09 | AC-04 | Pending |
| REC-01-FR-10 | AC-05 | Pending |
| REC-01-FR-11 | AC-06 | Pending |
| REC-01-FR-12 | Edge case (zero chunks, no mutation) | Pending |
| REC-01-FR-13 | Edge cases (single chunk; one-page PDF) | Pending |
| REC-01-NFR-01 | — | Pending |
| REC-01-NFR-02 | — | Pending |
| REC-01-NFR-03 | — | Pending |
| REC-01-NFR-04 | — | Pending |
| REC-01-NFR-05 | — | Pending |

---

## Success Criteria

- [ ] All 13 functional requirements verified by passing tests.
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` passes with coverage >= 80%.
- [ ] No reference to `content_hash` remains in `rag.py` or `tests/test_rag.py`.
- [ ] PR description documents mandatory `data/vectordb/` rebuild.
- [ ] All five metadata fields present on every chunk type confirmed by test assertions.
