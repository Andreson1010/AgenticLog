# Incremental ChromaDB Ingestion — Technical Spec

**Path:** `.specs/features/incremental-chroma-ingestion/spec.md`
**TLC scope:** large
**Based on story:** As a logistics operator, I want newly uploaded JSON documents added incrementally to the existing ChromaDB collection without rebuilding it from scratch, so that ingestion is fast and prior document chunks remain unaffected.
**Status:** Approved

---

## Problem Statement

`app.py` currently calls `reconstruir_vectordb()` after every upload, which calls `Chroma.from_documents()` and wipes and recreates the entire ChromaDB collection from all files on disk. This means ingestion time grows linearly with total document count, and any in-flight agent queries see an inconsistent state during the rebuild. The operator also sees a misleading "database rebuilt" mental model.

## Goals

- [ ] Uploading a new JSON file adds only that file's chunks to the existing ChromaDB collection (no full rebuild).
- [ ] Deduplication by SHA-256 content hash: same name + same hash = rejected with friendly "already present" notice; same name + different hash = explicit warning (no silent overwrite).
- [ ] Agent retriever reflects the updated collection in the same Streamlit session without a restart (singleton invalidated after successful ingestion).
- [ ] Full rollback (all newly added chunks removed) if embedding or `collection.add()` fails mid-ingestion.
- [ ] First-time ingestion (no existing collection) works without error.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Replacing / updating an existing document | Approved story defers this; same-name different-hash issues a warning only |
| Batch ingestion of multiple files in one upload | Not in approved story |
| Deleting documents from the collection | Not in approved story |
| Frontend changes beyond success/warning message copy | Message text change only; no new UI widgets |
| PDF incremental ingestion | Approved story: PDF continues with full rebuild |
| Removing `reconstruir_vectordb()` | Approved story: function is NOT removed; CLI entry point must continue to work |
| Upsert / overwrite existing document | Approved story explicitly excludes this |
| Concurrency / multi-user locking | Out of scope for single-user local deployment |
| Splitting `rag.py` into multiple files | Approved story: no `rag.py` split |

---

## User Stories

### P1: Incremental chunk ingestion ⭐ MVP
**User Story**: As a logistics operator, I want a newly uploaded JSON file's chunks added to the existing ChromaDB without a full rebuild, so that ingestion is fast and previous documents are untouched.

**Why P1**: Core value — without this, every upload pays a growing rebuild cost and risks collection loss on partial failure.

**Acceptance Criteria**:
1. WHEN ChromaDB collection exists with prior chunks AND operator uploads a valid new JSON, THEN system SHALL add only the new file's chunks and existing chunks SHALL remain present and unmodified.
2. WHEN incremental ingestion succeeds AND agent is queried in the same Streamlit session, THEN agent SHALL retrieve from the updated collection (singleton invalidated before query).
3. WHEN ingestion completes successfully, THEN UI SHALL display `st.success` with the returned `mensagem` and call `st.rerun()`.
4. WHEN no ChromaDB collection exists yet AND operator uploads the first document, THEN system SHALL create a new collection, ingest chunks, and return without error.
5. WHEN uploaded file fails security validation (size, forbidden keys, path traversal, non-JSON), THEN system SHALL raise `RAGSecurityError`, nothing SHALL be saved, and UI SHALL call `st.error`.
6. WHEN current file count in DIR_DOCUMENTS equals MAX_JSON_FILES AND operator attempts another upload, THEN system SHALL raise `RAGSecurityError`, nothing SHALL be saved, and UI SHALL call `st.error`.
7. WHEN uploaded file's `source` metadata already exists in ChromaDB AND content hash matches, THEN system SHALL add no chunks and SHALL call `st.info` (duplicate).
8. WHEN uploaded file has the same name as an existing file but a DIFFERENT content hash, THEN system SHALL issue `st.warning` and SHALL NOT add any chunks or overwrite the existing file.
9. WHEN any incremental ingestion completes (success or failure), THEN all pre-existing chunks from other files SHALL remain intact.

**Independent Test**: Upload file A, query for A-specific content, upload file B, query for B-specific content while re-querying for A-specific content — both return results; ChromaDB chunk count equals sum of individual ingestions.

---

### P2: Partial-failure rollback
**User Story**: As a logistics operator, I want failed ingestions to leave the collection clean, so that I never have orphan chunks from a half-ingested file.

**Acceptance Criteria**:
1. WHEN embedding or `collection.add()` raises any exception mid-ingestion, THEN system SHALL delete all chunk IDs that were added during that ingestion attempt and SHALL raise the original exception.
2. WHEN rollback itself fails (ChromaDB ignores IDs or raises), THEN system SHALL log a `logger.warning` listing the orphaned IDs, swallow the rollback error, and still re-raise the original ingestion exception.

---

### P3: Return contract and call-site update
**User Story**: As a logistics operator, I want the UI to reflect the exact outcome of an ingestion attempt through a structured response, so that I always know what happened.

**Acceptance Criteria**:
1. `adicionar_documento_incrementalmente()` SHALL return `{"status": "adicionado"|"duplicado"|"hash_diferente", "mensagem": "..."}`.
2. `_ingerir_documento()` in `app.py` SHALL call `adicionar_documento_incrementalmente()` instead of the `salvar_documento_enviado()` + `reconstruir_vectordb()` pair.
3. SHA-256 hash of raw file bytes SHALL be stored as `content_hash` in every chunk's ChromaDB metadata.

---

## Edge Cases

- WHEN DIR_DOCUMENTS is empty AND collection does not exist, THEN `adicionar_documento_incrementalmente` SHALL still save the file and create a collection (cold-start: ChromaDB creates an empty collection automatically via `Chroma(persist_directory=...)`).
- WHEN the JSON document produces zero chunks after splitting, THEN system SHALL log a warning and return without adding anything to ChromaDB.
- WHEN ChromaDB `persist_directory` does not exist on disk, THEN `Chroma(persist_directory=...)` SHALL create it (ChromaDB built-in behaviour — no custom handling needed, verified in integration test).
- WHEN `_vector_db` singleton is None (never initialized), THEN `invalidar_vector_db()` SHALL be a no-op and not raise.
- WHEN rollback `collection.delete(ids=...)` is called but ChromaDB silently ignores the IDs, THEN system SHALL log `logger.warning` with the orphaned IDs and swallow — no secondary error surfaced to the operator.

---

## Requirement Traceability

| Requirement ID | Story AC | Phase | Status |
|----------------|----------|-------|--------|
| INCRM-01 | P1-AC1, P1-AC9 | Design → Tasks | Design |
| INCRM-02 | P1-AC2 | Design → Tasks | Design |
| INCRM-03 | P1-AC3 | Design → Tasks | Design |
| INCRM-04 | P1-AC4 | Design → Tasks | Design |
| INCRM-05 | P1-AC5 | Design → Tasks | Design |
| INCRM-06 | P1-AC6 | Design → Tasks | Design |
| INCRM-07 | P1-AC7 | Design → Tasks | Design |
| INCRM-08 | P1-AC8 | Design → Tasks | Design |
| INCRM-09 | P2-AC1 | Design → Tasks | Design |
| INCRM-10 | P2-AC2 | Design → Tasks | Design |
| INCRM-11 | P3-AC1 | Design → Tasks | Design |
| INCRM-12 | P3-AC2 | Design → Tasks | Design |
| INCRM-13 | P3-AC3 | Design → Tasks | Design |

**ID format:** `INCRM-[NUMBER]`

---

## Approved Story Traceability

| Approved story AC # | Requirement ID(s) | Notes |
|---------------------|-------------------|-------|
| AC1 (save + append + invalidate + st.success + st.rerun) | INCRM-01, INCRM-02, INCRM-03, INCRM-13 | |
| AC2 (agent retrieves without manual rebuild) | INCRM-02 | |
| AC3 (same hash → st.info, no save) | INCRM-07 | |
| AC4 (different hash → st.warning, no save) | INCRM-08 | |
| AC5 (add_documents fails → rollback + st.error) | INCRM-09, INCRM-10 | |
| AC6 (security failure → RAGSecurityError + st.error) | INCRM-05 | |
| AC7 (MAX_JSON_FILES → RAGSecurityError + st.error) | INCRM-06 | |
| AC8 (SHA-256 hash stored in every chunk metadata) | INCRM-13 | |
| AC9 (return dict with status + mensagem) | INCRM-11 | |
| AC10 (_ingerir_documento calls new function) | INCRM-12 | |

---

## Data Model Changes

No new database tables or Pydantic models.

**ChromaDB metadata additions** (per chunk, stored in `document.metadata` before `collection.add()`):

| Field | Type | Description |
|-------|------|-------------|
| `content_hash` | `str` | SHA-256 hex digest of the raw file bytes |
| `source` | `str` | Already set by `JSONLoader` as the file path string |

The `content_hash` field enables deduplication at query time via `collection.get(where={"source": {"$eq": ...}})` before adding.

No migration needed — ChromaDB does not enforce schemas; existing chunks without `content_hash` are unaffected.

---

## Process / Background Flow

**Happy path (new file, collection exists):**
1. `app.py` calls `adicionar_documento_incrementalmente(filename, conteudo)` (replaces `salvar_documento_enviado` + `reconstruir_vectordb` pair).
2. Compute SHA-256 of `conteudo`.
3. Run security validations (extension, size, forbidden keys, path sanitization, file count limit) — raise `RAGSecurityError` on any failure.
4. Open existing ChromaDB collection with `Chroma(persist_directory=..., embedding_function=...)`. If it does not exist, ChromaDB creates it (cold-start).
5. Query collection for existing chunks where `source == saved_path` to detect duplicates.
6. If chunks found and `content_hash` matches: return `{"status": "duplicado", ...}` (no disk write, no ChromaDB modification).
7. If chunks found and `content_hash` differs: return `{"status": "hash_diferente", ...}` (no disk write, no ChromaDB modification).
8. Save file to disk.
9. Load only the new file with `JSONLoader` + `RecursiveCharacterTextSplitter`.
10. Attach `content_hash` to each chunk's metadata.
11. Pre-generate UUIDs as chunk IDs.
12. Call `vectordb_instance.add_documents(chunks, ids=chunk_ids)`. On exception: rollback (delete IDs, unlink saved file), re-raise.
13. Call `invalidar_vector_db()` in `agent.py` via lazy import inside the function body (avoids heavy `agent.py` import at `python -m agenticlog.rag`).
14. Return `{"status": "adicionado", "mensagem": "Arquivo <name> adicionado com sucesso. N chunks inseridos."}`.

**Failure path — mid-ingestion error:**
1. Exception raised during embedding or `collection.add_documents()`.
2. Rollback: call `vectordb_instance.delete(ids=chunk_ids)` for all pre-generated IDs.
3. If rollback itself raises: log `logger.warning` with orphaned IDs list, swallow rollback exception.
4. Remove saved file from `DIR_DOCUMENTS` (`saved_path.unlink(missing_ok=True)`).
5. Re-raise original ingestion exception.

**Cold-start (no existing collection):**
1. `Chroma(persist_directory=str(DIR_VECTORDB), embedding_function=...)` — ChromaDB creates directory and empty collection automatically.
2. `collection.get(...)` returns empty result → no duplicate detected → proceed with ingestion.

---

## API Changes

### New public function in `src/agenticlog/rag.py`

```python
def adicionar_documento_incrementalmente(
    filename: str,
    conteudo: bytes,
) -> dict[str, str]:
    """Adiciona chunks de um novo arquivo JSON ao ChromaDB existente sem reconstrução.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
    Saída: dict com chaves "status" e "mensagem":
      {"status": "adicionado", "mensagem": "Arquivo <nome> adicionado com sucesso. N chunks inseridos."}
      {"status": "duplicado", "mensagem": "Arquivo <nome> já está presente na base vetorial."}
      {"status": "hash_diferente", "mensagem": "Arquivo <nome> já existe com conteúdo diferente. Remoção e substituição não são suportadas nesta versão."}
    Lança RAGSecurityError em qualquer falha de validação de segurança.
    Lança Exception se a ingestão falhar após rollback.
    """
```

### New private function in `src/agenticlog/rag.py`

```python
def _computar_hash_conteudo(conteudo: bytes) -> str:
    """Computa o hash SHA-256 do conteúdo binário do arquivo.

    Entrada: conteudo — bytes do arquivo.
    Saída: string hexadecimal de 64 caracteres (SHA-256).
    """
```

### New public function in `src/agenticlog/agent.py`

```python
def invalidar_vector_db() -> None:
    """Invalida o singleton _vector_db para que a próxima chamada a _get_vector_db() reconecte ao ChromaDB.

    Entrada: nenhuma.
    Saída: nenhuma.
    Efeito colateral: atribui None a _vector_db global.
    """
```

### Changed call in `app.py`

`_ingerir_documento()` replaces the `salvar_documento_enviado` + `reconstruir_vectordb` pair with a single call to `adicionar_documento_incrementalmente`. The UI message is derived from the returned `status` key.

---

## Frontend Changes

`app.py` — `_ingerir_documento()` function only:
- Replace import of `reconstruir_vectordb` with import of `adicionar_documento_incrementalmente`.
- Replace spinner text "Reconstruindo base vetorial..." with "Adicionando documento à base vetorial...".
- Replace status-agnostic `st.success` with status-aware messages:
  - `adicionado` → `st.success(mensagem)` + `st.rerun()`
  - `duplicado` → `st.info(mensagem)` (no `st.rerun()`)
  - `hash_diferente` → `st.warning(mensagem)` (no `st.rerun()`)
- Rollback error path: remove file unlink from `app.py` (now handled inside `rag.py`).

---

## Tests Required

**Unit tests — `tests/test_rag.py` (new class `TestComputarHash`):**
- `teste_1_` — same input produces same 64-char hex digest.
- `teste_2_` — different inputs produce different hashes.

**Unit tests — `tests/test_rag.py` (new class `TestAdicionarDocumentoIncrementalmente`):**
- `teste_1_` — happy path: new file added, chunk count increases, `content_hash` in metadata.
- `teste_2_` — first ingestion (no existing collection): collection created, no error.
- `teste_3_` — duplicate detection (same hash): returns `{"status": "duplicado", ...}`, no new chunks.
- `teste_4_` — hash mismatch (same name, different hash): returns `{"status": "hash_diferente", ...}`, no new chunks.
- `teste_5_` — security validation failure: `RAGSecurityError` raised, collection unchanged.
- `teste_6_` — MAX_JSON_FILES limit: `RAGSecurityError` raised, collection unchanged.
- `teste_7_` — mid-ingestion failure: rollback removes added IDs, original exception re-raised.
- `teste_8_` — rollback failure: `logger.warning` logged with orphaned IDs, original exception re-raised.
- `teste_9_` — zero-chunk document: warning logged, no collection modification.

**Unit tests — `tests/test_agent.py` (new file):**
- `teste_1_` — `invalidar_vector_db()` sets `_vector_db` to None.
- `teste_2_` — `invalidar_vector_db()` when already None does not raise.

**Updated tests — `tests/test_app.py`:**
- Replace all mocks of `app.reconstruir_vectordb` with `app.adicionar_documento_incrementalmente`.
- Add cases for `status == "duplicado"` (`st.info` called, no `st.rerun`) and `status == "hash_diferente"` (`st.warning` called, no `st.rerun`).
- Verify rollback no longer attempted in `app.py` (file unlink removed from app layer).

**Updated tests — `tests/acceptance/test_document_ingestion_ui.py`:**
- Update DOCING-01, 02, 09, 10 and PDF-01, 08 to mock `adicionar_documento_incrementalmente` instead of `salvar_documento_enviado` + `reconstruir_vectordb`.

**Existing tests that will break:**
- `tests/test_app.py::TestIngerirDocumento::teste_1_upload_fluxo_sucesso` — mocks `reconstruir_vectordb`, must be updated.
- `tests/test_app.py::TestIngerirDocumento::teste_3_upload_erro_rebuild_exibido` — tests file unlink in app.py; this moves to rag.py.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/rag.py` | Add functions | New `adicionar_documento_incrementalmente()` + `_computar_hash_conteudo()` |
| `src/agenticlog/agent.py` | Add function | New `invalidar_vector_db()` exposing singleton reset |
| `app.py` | Modify function | `_ingerir_documento()` calls new API; updated messages |
| `tests/test_rag.py` | Add classes | `TestComputarHash` (2 methods) + `TestAdicionarDocumentoIncrementalmente` (9 methods) |
| `tests/test_agent.py` | New file | `TestInvalidarVectorDb` (2 methods) for `invalidar_vector_db()` |
| `tests/test_app.py` | Modify class | Update mocks and add duplicate/hash-mismatch cases |
| `tests/acceptance/test_document_ingestion_ui.py` | Modify tests | Update DOCING-01, 02, 09, 10 and PDF-01, 08 to mock new function |

---

## Risks

| Risk | Severity | Status | Mitigation |
|------|----------|--------|------------|
| `salvar_documento_enviado` raises `RAGSecurityError("Arquivo com esse nome já existe.")` for same-name files, blocking duplicate detection flow | High | **Clear** — `adicionar_documento_incrementalmente` handles save-or-check logic directly; existing `salvar_documento_enviado` not modified | New function conditionally skips disk save and goes straight to ChromaDB dedup check |
| `collection.get(where={"source": ...})` metadata filter syntax may vary across `langchain-chroma` versions | Medium | **Clear** — use `where={"source": {"$eq": str(saved_path)}}` (explicit ChromaDB operator syntax) | Integration test must verify filter works with installed version |
| ChromaDB `collection.add()` is not atomic — partial batch add is possible if embedding list partially succeeds | Medium | **Clear** — pre-compute all IDs before `add_documents()` call; delete on any exception | Covered by INCRM-09 rollback requirement |
| `Chroma.add_documents(ids=[...])` may silently ignore passed IDs | Medium | **Clear** — log `logger.warning` with orphaned IDs and swallow; do not surface secondary error to operator | Approved resolution: warning+swallow pattern |
| `rag.py` importing `agent.py` at module level causes heavy import at `python -m agenticlog.rag` | Medium | **Clear** — import `invalidar_vector_db` lazily inside `adicionar_documento_incrementalmente` function body, not at module top level | See design.md §5 |
| `invalidar_vector_db()` is not thread-safe (Streamlit can serve concurrent users) | Low | **Clear** — Streamlit Community Cloud runs single-threaded per session; not a production multi-user risk for this project | Document as known limitation |
| `Chroma.from_documents()` (used by `cria_vectordb`) vs `Chroma(persist_directory=...)` (new function) collection name collision | Low | **Clear** — both use default collection name `"langchain"` and same persist dir; no `collection_name` parameter is set in either | Verify consistency in integration test |

---

## Open Questions

None. All questions resolved before approval:

| Question | Resolution |
|----------|-----------|
| Cold-start (no existing collection) | `Chroma(persist_directory=...)` creates empty collection automatically; no special handling needed |
| Rollback when ChromaDB silently ignores IDs | Log `logger.warning` with orphaned IDs and swallow; do not surface secondary error to operator |
| Hash over raw bytes vs post-jq text | Hash computed over raw bytes (`conteudo: bytes`) before any parsing |
| Whether `st.rerun()` is called after successful add | Yes — `st.rerun()` IS called for `"adicionado"` status only, consistent with current behavior |

---

## Success Criteria

- [ ] `pytest tests/test_rag.py -v` passes with all 11 new test methods green (2 in TestComputarHash + 9 in TestAdicionarDocumentoIncrementalmente).
- [ ] `pytest tests/test_agent.py -v` passes with both new test methods green.
- [ ] `pytest tests/test_app.py -v` passes with updated mocks.
- [ ] `pytest tests/acceptance/test_document_ingestion_ui.py -v` passes with updated mocks.
- [ ] `pytest --cov=agenticlog --cov-report=term-missing` reports >= 80% coverage.
- [ ] Manual smoke test: upload file A, upload file B, query for each — both return results; ChromaDB chunk count equals A-chunks + B-chunks.
- [ ] Upload same file twice — second upload returns "duplicado" info message; chunk count unchanged.
- [ ] Upload file with same name but edited content — warning message displayed; chunk count unchanged.
- [ ] `python -m agenticlog.rag` still works (CLI entry point not broken).
