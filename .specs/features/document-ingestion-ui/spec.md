# Document Ingestion UI — Technical Spec

**Path:** `.specs/features/document-ingestion-ui/spec.md`
**TLC scope:** large
**Based on story:** As a logistics operator, I want to upload a JSON document through the Streamlit sidebar so that the ChromaDB vector database is rebuilt with the new content and I can immediately query it without restarting the application.
**Status:** Awaiting human approval

---

## Problem Statement

Operators currently must upload JSON files manually to `data/documents/` via the filesystem and run `python -m agenticlog.rag` from the command line to rebuild the vector database. This breaks the operator workflow, requires terminal access, and forces an application restart. A sidebar upload UI would close this gap with zero infrastructure change.

## Goals

- [ ] Operators can upload a `.json` file through the Streamlit sidebar and trigger a ChromaDB rebuild without leaving the browser.
- [ ] All existing security invariants (path traversal, size limit, forbidden keys, file count) are enforced before any byte is written to disk.
- [ ] A rebuild failure leaves `data/documents/` and `data/vectordb/` in their pre-upload state.
- [ ] After a successful ingest, the next query uses the updated ChromaDB without restarting the application.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Delete / replace existing documents via UI | Approved story explicitly excludes this |
| Non-JSON formats (CSV, PDF, TXT) | Approved story explicitly excludes this |
| Incremental vectordb updates | Approved story explicitly excludes this |
| Authentication / authorization | Approved story explicitly excludes this |
| Async / background rebuild | Approved story explicitly excludes this |
| Multi-file batch upload | Not in approved story |

---

## User Stories

### P1: Upload JSON and Rebuild VectorDB ⭐ MVP

**User Story**: As a logistics operator, I want to upload a JSON document through the Streamlit sidebar, so that the ChromaDB vector database is rebuilt and my next query uses the updated content.

**Why P1**: Core feature — the entire story is a single atomic capability.

**Acceptance Criteria**:

1. WHEN operator uploads a valid `.json` file < 10 MB with no forbidden keys AND clicks "Ingerir Documento" THEN the file SHALL be saved to `data/documents/`, ChromaDB SHALL be rebuilt, a success message SHALL be shown, and `st.rerun()` SHALL be called.
2. WHEN rebuild is running THEN a spinner with text "Reconstruindo base vetorial..." SHALL be visible until the operation completes.
3. WHEN operator uploads a file with a non-`.json` extension THEN the system SHALL reject it before any disk write and show "Apenas arquivos .json são aceitos."
4. WHEN operator uploads a file larger than 10 MB THEN the system SHALL reject it before any disk write and show a message citing the 10 MB limit.
5. WHEN operator uploads a JSON file containing the forbidden key `"lc"` THEN the system SHALL reject it before any disk write and show "Arquivo contém chave proibida."
6. WHEN the filename already exists in `data/documents/` THEN the system SHALL reject it, show an error, and write zero bytes.
7. WHEN the filename contains path traversal sequences or Windows-invalid characters (`<>:"/\|?*` and null bytes) THEN the system SHALL reject it before any disk write.
8. WHEN adding the file would make total file count exceed 1000 THEN the system SHALL reject it with a file-limit message.
9. WHEN `cria_vectordb()` raises any exception during rebuild THEN the uploaded file SHALL be removed from `data/documents/`, the original vectordb SHALL remain intact, and an error message SHALL be displayed.
10. WHEN ingest succeeds THEN the next query submitted by the operator SHALL use the rebuilt ChromaDB.

**Independent Test**: Upload a valid JSON, verify file appears in `data/documents/`, verify ChromaDB rebuild succeeds, submit a query that would match the new document content, verify the document appears in retrieved results.

---

## Edge Cases

- WHEN JSON file is syntactically invalid THEN `_valida_json_sem_chaves_proibidas` SHALL raise `RAGSecurityError` and be surfaced as a user-friendly error.
- WHEN `data/documents/` does not exist or is unreadable THEN `_valida_path_documentos` raises `RAGSecurityError` before any write.
- WHEN the uploaded file is 0 bytes THEN size check passes but JSON parse will raise `RAGSecurityError`; operator sees error before disk write.
- WHEN two operators upload simultaneously (two Streamlit sessions) THEN both writes may proceed independently; no locking is in scope — file-count check is a best-effort guard.

---

## Requirement Traceability

| Requirement ID | Acceptance Criterion | Phase | Status |
|----------------|----------------------|-------|--------|
| DOCING-01 | AC-1 (happy path save + rebuild + rerun) | Design | Pending |
| DOCING-02 | AC-2 (spinner during rebuild) | Design | Pending |
| DOCING-03 | AC-3 (non-json extension rejection) | Design | Pending |
| DOCING-04 | AC-4 (size > 10 MB rejection) | Design | Pending |
| DOCING-05 | AC-5 (forbidden key rejection) | Design | Pending |
| DOCING-06 | AC-6 (filename collision rejection) | Design | Pending |
| DOCING-07 | AC-7 (path traversal / invalid char rejection) | Design | Pending |
| DOCING-08 | AC-8 (file count limit rejection) | Design | Pending |
| DOCING-09 | AC-9 (rollback on rebuild failure) | Design | Pending |
| DOCING-10 | AC-10 (retriever uses rebuilt DB after rerun) | Design | Pending |

---

## Data Model Changes

### New constant — `config.py`

No new constant is strictly needed: `MAX_JSON_FILE_SIZE_MB = 10` already exists. However, for clarity the upload UI will reference this constant directly from `config.py` rather than hardcoding 10.

### New functions — `rag.py`

**`_sanitizar_nome_arquivo(filename: str) -> str`** (private)
- Strips directory separators and Windows-invalid characters (`<>:"/\|?*`, null bytes, `..` sequences).
- Returns the sanitized basename or raises `RAGSecurityError` if the result is empty or differs in a way that indicates traversal.

**`salvar_documento_enviado(filename: str, conteudo: bytes) -> Path`** (public)
- Validation pipeline (in order):
  1. Extension check — must be `.json`.
  2. Size check — `len(conteudo)` must not exceed `MAX_JSON_FILE_SIZE_MB * 1024 * 1024`.
  3. Filename sanitization — call `_sanitizar_nome_arquivo`.
  4. Collision check — target path must not already exist in `DIR_DOCUMENTS`.
  5. File-count check — current file count + 1 must not exceed `MAX_JSON_FILES`.
  6. Forbidden-key check — write to a `tempfile`, validate with `_valida_json_sem_chaves_proibidas`, then move to `DIR_DOCUMENTS`.
- Raises `RAGSecurityError` at any step.
- Returns the final `Path` of the saved file.

**`reconstruir_vectordb() -> None`** (public)
- Thin wrapper: calls `cria_vectordb()`.
- Exists to give `app.py` a stable, intention-revealing name and to ease mocking in tests.

### Migrations

No database schema migrations. ChromaDB is rebuilt from scratch on each call to `cria_vectordb()`.

---

## Process / Background Flow

**Happy path:**
1. Operator opens sidebar expander "Adicionar Documento".
2. Operator selects a `.json` file via `st.file_uploader`.
3. Operator clicks "Ingerir Documento".
4. `app.py` reads `uploaded_file.getvalue()` (bytes) and `uploaded_file.name`.
5. `salvar_documento_enviado(filename, conteudo)` validates and writes file to `data/documents/`.
6. `app.py` shows spinner "Reconstruindo base vetorial...".
7. `reconstruir_vectordb()` calls `cria_vectordb()`.
8. On success: `st.success("Documento ingerido com sucesso.")` then `st.rerun()`.
9. Streamlit runner restarts, module-level retriever in `agent.py` is rebuilt from updated ChromaDB.

**Failure path — validation error:**
1. Steps 1–4 same as happy path.
2. `salvar_documento_enviado` raises `RAGSecurityError`.
3. `app.py` catches it, calls `st.error(str(e))`.
4. Zero bytes written, no rebuild triggered.

**Failure path — rebuild error:**
1. Steps 1–7 same as happy path (file is on disk).
2. `cria_vectordb()` raises any exception.
3. `app.py` `finally` block removes the uploaded file from `data/documents/`.
4. Original `data/vectordb/` is untouched (ChromaDB writes atomically to its own directory).
5. `app.py` calls `st.error("Erro ao reconstruir base vetorial. Arquivo removido.")`.

---

## API Changes

No HTTP API changes. All interaction is through the Streamlit UI.

---

## Frontend Changes

**`app.py` — sidebar addition (after existing `st.sidebar.button("Suporte")` block):**

```
st.sidebar.expander("Adicionar Documento")
  └── st.file_uploader("Selecione um arquivo JSON", type=None)   # type=None so we validate manually
  └── st.button("Ingerir Documento")
       └── [validation + save + spinner + rebuild + st.rerun() or st.error()]
```

- `type=None` on `file_uploader`: Streamlit's `type` filter is cosmetic only; manual extension check in `salvar_documento_enviado` is the security gate.
- Spinner wraps only the `reconstruir_vectordb()` call, not the validation step, so validation errors surface immediately.

---

## Tests Required

### Unit — `tests/test_rag.py`

| Test | What it covers |
|------|----------------|
| `teste_1_salvar_documento_enviado_sucesso` | Happy path: valid bytes, valid name → file saved |
| `teste_2_salvar_rejeita_extensao_invalida` | Non-`.json` → `RAGSecurityError` before disk write |
| `teste_3_salvar_rejeita_tamanho_excedido` | > 10 MB → `RAGSecurityError` before disk write |
| `teste_4_salvar_rejeita_chave_proibida` | JSON with `"lc"` key → `RAGSecurityError` |
| `teste_5_salvar_rejeita_colisao_de_nome` | Existing filename → `RAGSecurityError` |
| `teste_6_salvar_rejeita_path_traversal` | `../evil.json` → `RAGSecurityError` |
| `teste_7_salvar_rejeita_limite_de_arquivos` | 1000 files already → `RAGSecurityError` |
| `teste_8_reconstruir_vectordb_chama_cria_vectordb` | Mocked `cria_vectordb` called once |
| `teste_9_sanitizar_nome_arquivo_invalido` | Windows-invalid chars stripped or rejected |

### Integration / UI — `tests/test_app.py`

| Test | What it covers |
|------|----------------|
| `teste_1_upload_fluxo_sucesso` | Full happy path with mocked `salvar_documento_enviado` and `reconstruir_vectordb` |
| `teste_2_upload_erro_validacao_exibido` | `RAGSecurityError` from save → `st.error` called |
| `teste_3_upload_erro_rebuild_exibido` | Exception from rebuild → `st.error` called, file removed |
| `teste_4_upload_sem_arquivo_selecionado` | Button clicked with no file → no action |

### Existing tests that may break

- None expected. No existing function signatures change; only new public functions are added.
- `test_rag.py` existing tests use `@patch("agenticlog.rag.DIR_DOCUMENTS")` — same pattern applies to new tests.

### Coverage gate

```bash
pytest --cov=agenticlog --cov-fail-under=80
```

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/rag.py` | Add functions | `_sanitizar_nome_arquivo`, `salvar_documento_enviado`, `reconstruir_vectordb` |
| `app.py` | Add UI section | Sidebar expander with file uploader, button, spinner, error handling |
| `tests/test_rag.py` | Add test cases | 9 new test methods for new rag.py functions |
| `tests/test_app.py` | Add test cases | 4 new UI upload flow tests (file may be created if it does not exist yet) |
| `src/agenticlog/config.py` | No change required | `MAX_JSON_FILE_SIZE_MB` already exists; no new constant needed |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Race condition: two simultaneous uploads interleave file-count check and write | LOW | No locking in scope per approved story; file-count guard is best-effort; document in code comment |
| `cria_vectordb()` is not atomic — if process is killed mid-rebuild, vectordb may be corrupt | MEDIUM | `st.rerun()` only fires on success; on exception the uploaded file is removed; original vectordb directory is only overwritten by ChromaDB internals if rebuild completes |
| `st.rerun()` in an `st.sidebar.expander` context may behave differently in future Streamlit versions | LOW | Pin Streamlit version in `requirements.txt`; test manually after upgrades |
| Filename sanitization gap: Unicode homoglyphs or NTFS alternate data streams | LOW | Scope is limited to documented Windows-invalid chars and traversal sequences per approved story; document known limitation |
| `salvar_documento_enviado` writes the file before calling `cria_vectordb()` — if the process crashes between write and rebuild, an unindexed file remains | LOW | On next full rebuild (CLI or next successful upload) all files in `data/documents/` are indexed; file is not orphaned permanently |
| Incomplete test coverage of `app.py` (CONCERNS.md MEDIUM risk) | MEDIUM | This feature's test tasks directly address this; `test_app.py` is required delivery |

---

## Open Questions

None. All open questions were resolved by the orchestrator prior to spec authoring (see Approved Story section).

---

## Success Criteria

- [ ] All 9 unit tests in `test_rag.py` for new functions pass.
- [ ] All 4 UI tests in `test_app.py` for upload flow pass.
- [ ] `pytest --cov=agenticlog --cov-fail-under=80` passes.
- [ ] Manual smoke test: upload valid JSON → query matches new content → success.
- [ ] Manual smoke test: upload with forbidden key → zero bytes in `data/documents/`.
- [ ] Manual smoke test: simulate rebuild failure → uploaded file absent from `data/documents/`.
