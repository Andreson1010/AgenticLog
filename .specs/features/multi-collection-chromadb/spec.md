# Multi-Document Collection Support in ChromaDB — Technical Spec

**Path:** `.specs/features/multi-collection-chromadb/spec.md`
**TLC scope:** large
**Based on story:** As a logistics operator, I want to ingest documents into named ChromaDB collections and have the agent query all collections merged, so that I can organise by domain without losing cross-collection search.
**Status:** Awaiting human approval

---

## Problem Statement

Currently AgenticLog stores all documents in a single unnamed ChromaDB collection, making it impossible to organise logistics data by domain (suppliers, routes, contracts). Operators cannot distinguish between document domains at ingestion time, and all retrieval mixes every source. Adding named collection support enables domain isolation at write time while preserving full-corpus search at query time.

## Goals

- [ ] Operator can select or create a named ChromaDB collection from the Streamlit sidebar before uploading a document.
- [ ] Every write function (`adicionar_documento_incrementalmente`, `salvar_documento_enviado`, `salvar_pdf_enviado`) stores documents in the specified named collection only.
- [ ] Agent fan-out retrieval queries all existing collections and returns the highest-ranked result, identical behaviour to today when only one collection exists.
- [ ] All collection-name validation is centralised in `_sanitizar_nome_colecao()` and enforces ChromaDB naming rules before any write.
- [ ] Zero regressions: existing pytest suite passes at ≥ 80 % coverage after changes.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Migration of existing `"langchain"` collection data | AC11 — operator must wipe and re-ingest; out of scope for this feature |
| Per-query collection filtering (`collection_name` on `QueryRequest`) | AC14 — query endpoint unchanged |
| Collection deletion / renaming via UI or API | Not in approved story |
| Access control per collection | Not in approved story |
| Collection listing endpoint in FastAPI | Not requested; listing is read from ChromaDB client directly in UI |

---

## User Stories

### P1: Named Collection Ingestion ⭐ MVP

**User Story**: As a logistics operator, I want to select or create a named collection in the sidebar before uploading a document, so that my document is stored in the correct domain collection.

**Why P1**: Without this, the feature provides no value — ingestion is the prerequisite for everything else.

**Acceptance Criteria**:

1. WHEN the sidebar "Adicionar Documento" expander is open THEN system SHALL display a dropdown listing all existing ChromaDB collection names plus the option "Nova coleção…".
2. WHEN the operator selects "Nova coleção…" THEN system SHALL reveal a text input for the new collection name with inline validation feedback.
3. WHEN the operator clicks "Ingerir Documento" with a valid collection name and a valid JSON file THEN system SHALL call `adicionar_documento_incrementalmente(filename, conteudo, collection_name)` and display a success message that includes the collection name.
4. WHEN the operator clicks "Ingerir Documento" with a valid collection name and a valid PDF file THEN system SHALL call `salvar_pdf_enviado(filename, conteudo, collection_name)` followed by `reconstruir_vectordb(collection_name)` and display a success message that includes the collection name.
5. WHEN no collection is explicitly selected THEN system SHALL use `DEFAULT_COLLECTION_NAME` from `config.py`.

**Independent Test**: Open the sidebar, select "Nova coleção…", type "fornecedores", upload a JSON fixture, click Ingerir — success banner shows "fornecedores". Repeat for a second collection "rotas". Query the agent — results draw from both collections.

---

### P1: Collection Name Validation ⭐ MVP

**User Story**: As a logistics operator, I want invalid collection names rejected with clear messages, so that I cannot create ChromaDB collections with names that will cause downstream errors.

**Why P1**: ChromaDB silently rejects or truncates invalid names, leading to data loss. Server-side validation with `RAGSecurityError` protects data integrity.

**Acceptance Criteria**:

1. WHEN `_sanitizar_nome_colecao()` receives a name with fewer than 3 characters THEN system SHALL raise `RAGSecurityError`.
2. WHEN `_sanitizar_nome_colecao()` receives a name with more than 63 characters THEN system SHALL raise `RAGSecurityError`.
3. WHEN `_sanitizar_nome_colecao()` receives a name containing characters other than alphanumeric, hyphen, or underscore THEN system SHALL raise `RAGSecurityError`.
4. WHEN `_sanitizar_nome_colecao()` receives a name that starts or ends with a non-alphanumeric character THEN system SHALL raise `RAGSecurityError`.
5. WHEN any write function is called with an invalid collection name THEN system SHALL raise `RAGSecurityError` before any ChromaDB write occurs.
6. WHEN `_sanitizar_nome_colecao()` receives exactly a 3-character or 63-character alphanumeric name THEN system SHALL return the name unchanged (boundary = valid).

**Independent Test**: `pytest tests/test_rag.py::TestSanitizarNomeColecao` passes all boundary and rejection cases.

---

### P1: Agent Fan-out Retrieval ⭐ MVP

**User Story**: As a logistics operator, I want the agent to retrieve from all existing collections and return the best-ranked answer, so that I never have to choose which domain to query.

**Why P1**: If the agent only reads one collection, the feature breaks cross-domain queries — operators lose the main benefit.

**Acceptance Criteria**:

1. WHEN two or more named collections each contain at least one document THEN `_get_retriever()` SHALL retrieve from all collections, merge the results, and return the top-ranked documents.
2. WHEN a collection is empty or does not exist during fan-out THEN system SHALL silently skip it (0 documents from that collection, no exception).
3. WHEN ChromaDB raises an exception for a collection during fan-out THEN system SHALL re-raise immediately.
4. WHEN no collections exist or all are empty THEN system SHALL return the same "no results" behaviour as the current single-collection implementation.
5. WHEN `invalidar_vector_db()` is called THEN system SHALL clear all entries in the `_vector_dbs` dict so the next retrieval reconnects to ChromaDB.

**Independent Test**: Populate "fornecedores" with supplier JSON and "contratos" with contract JSON; submit a query that spans both — retrieved documents include hits from both collections.

---

### P2: Default Collection Fallback

**User Story**: As a developer or operator using the CLI ingestion path, I want a sensible default collection name used when I don't specify one, so that the system works unchanged for existing scripts.

**Why P2**: Backward compatibility for CLI and API callers that do not pass `collection_name`.

**Acceptance Criteria**:

1. WHEN `adicionar_documento_incrementalmente`, `salvar_documento_enviado`, or `salvar_pdf_enviado` is called without a `collection_name` argument THEN system SHALL use `DEFAULT_COLLECTION_NAME` from `config.py`.
2. WHEN `config.py` defines `DEFAULT_COLLECTION_NAME = "logistica"` THEN system SHALL use `"logistica"` as the default for all write functions.

**Independent Test**: Call `adicionar_documento_incrementalmente("doc.json", data)` without `collection_name`; verify ChromaDB stores the document in `"logistica"` collection.

---

## Edge Cases

- WHEN collection name is exactly 3 characters (`"abc"`) THEN system SHALL accept it.
- WHEN collection name is exactly 63 characters THEN system SHALL accept it.
- WHEN collection name is 64 characters THEN system SHALL raise `RAGSecurityError`.
- WHEN collection name is empty string (`""`) THEN system SHALL raise `RAGSecurityError`.
- WHEN all collections are empty THEN agent SHALL return the same "no results" response as today.
- WHEN `invalidar_vector_db()` is called on an already-empty dict THEN system SHALL not raise any exception.
- WHEN a valid but unused collection name is passed THEN system SHALL treat it as an empty collection — 0 documents, no error.
- WHEN the dropdown is first loaded and no ChromaDB collections exist yet THEN system SHALL show only "Nova coleção…" in the dropdown.

---

## Requirement Traceability

| Requirement ID | Acceptance Criteria | Story | Phase | Status |
|----------------|--------------------|-----------------------------------------|-------|--------|
| MCC-01 | AC1 | P1: Named Collection Ingestion | Design | Pending |
| MCC-02 | AC2 | P1: Named Collection Ingestion | Design | Pending |
| MCC-03 | AC3 | P1: Named Collection Ingestion | Design | Pending |
| MCC-04 | AC4 | P1: Named Collection Ingestion | Design | Pending |
| MCC-05 | AC5 / AC6 | P1: Named Collection Ingestion + P2 | Design | Pending |
| MCC-06 | AC7 | P1: Collection Name Validation | Design | Pending |
| MCC-07 | AC8 | P1: Collection Name Validation | Design | Pending |
| MCC-08 | AC9 | P1: Collection Name Validation | Design | Pending |
| MCC-09 | AC10 | P1: Collection Name Validation | Design | Pending |
| MCC-10 | AC12 | P1: Collection Name Validation | Design | Pending |
| MCC-11 | AC15 | P1: Collection Name Validation | Design | Pending |
| MCC-12 | AC5 (fan-out) | P1: Agent Fan-out Retrieval | Design | Pending |
| MCC-13 | AC17 | P1: Agent Fan-out Retrieval | Design | Pending |
| MCC-14 | AC18 | P1: Agent Fan-out Retrieval | Design | Pending |
| MCC-15 | AC13 | P1: Agent Fan-out Retrieval | Design | Pending |
| MCC-16 | AC16 | P1: Named Collection Ingestion | Design | Pending |
| MCC-17 | AC6 / P2 | P2: Default Collection Fallback | Design | Pending |
| MCC-18 | AC11 (out of scope) | — migration not supported | — | N/A |
| MCC-19 | AC14 | Out of scope (QueryRequest unchanged) | — | N/A |

**ID format:** `MCC-[NUMBER]` (Multi-Collection ChromaDB)

**Status values:** Pending → In Design → In Tasks → Implementing → Verified

**Coverage:** 17 requirements mapped (2 N/A out-of-scope), 15 actionable pending design.

---

## Data Model Changes

No new database tables or schemas. ChromaDB collection names are handled by the `langchain-chroma` `Chroma` constructor parameter `collection_name`. The `_vector_dbs` dict in `agent.py` maps `collection_name: str → Chroma` instances.

**New constants in `config.py`:**

```python
DEFAULT_COLLECTION_NAME: str = "logistica"
COLLECTION_NAME_MIN_LEN: int = 3
COLLECTION_NAME_MAX_LEN: int = 63
COLLECTION_NAME_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$")
```

Note: single-character names (len 1 or 2) are rejected by the min-len check before the pattern is tested. The pattern anchors require alphanumeric at start and end; for exactly 3-char names the middle character may be `_` or `-`.

---

## Process / Background Flow

**Happy path — named JSON ingestion:**
1. Operator selects collection name (or types new name) in sidebar.
2. UI calls `adicionar_documento_incrementalmente(filename, conteudo, collection_name)`.
3. `_sanitizar_nome_colecao(collection_name)` runs first — rejects invalid names.
4. `Chroma(persist_directory=..., collection_name=collection_name, ...)` opens or creates the collection.
5. Duplicate-hash check runs against that specific collection.
6. Chunks are added; `invalidar_vector_db()` clears the agent's `_vector_dbs` dict.
7. Success message with collection name displayed.

**Happy path — agent fan-out query:**
1. `retrieve_info()` calls `_get_retriever()`.
2. `_get_retriever()` calls `_listar_colecoes()` to get all collection names from ChromaDB.
3. For each collection name, `_get_vector_db(collection_name)` returns cached or new `Chroma` instance.
4. Each Chroma instance produces a retriever; all are invoked with the query.
5. Results are concatenated and deduplicated by document ID; top-k documents returned.
6. Downstream nodes (`gera_multiplas_respostas`, `avalia_similaridade`, `rank_respostas`) unchanged.

**Failure path — invalid collection name:**
1. Any write function receives invalid `collection_name`.
2. `_sanitizar_nome_colecao()` raises `RAGSecurityError` immediately.
3. No ChromaDB write occurs.
4. UI catches `RAGSecurityError` and displays `st.error(str(e))`.

**Failure path — ChromaDB error during fan-out:**
1. `_get_retriever()` iterates collections; one raises a ChromaDB exception.
2. Exception propagates immediately from `_get_retriever()` to `retrieve_info()`.
3. LangGraph node propagates exception; FastAPI returns HTTP 500.

**Failure path — empty or nonexistent collection during fan-out:**
1. `_get_retriever()` opens collection; ChromaDB returns 0 documents.
2. That collection contributes 0 results to the merged list.
3. Processing continues with results from other collections.

---

## API Changes

No changes to the FastAPI API contract. `QueryRequest` and `QueryResponse` schemas are unchanged (AC14). The internal `_verificar_vectordb()` startup check in `api.py` may need updating to check against `DEFAULT_COLLECTION_NAME` instead of the unnamed default collection if it currently relies on the implicit collection name — this is a minor internal change with no external contract impact.

---

## Frontend Changes

**`app.py` — sidebar "Adicionar Documento" expander:**

1. Add `st.selectbox` listing ChromaDB collection names fetched at render time, plus the sentinel option `"Nova coleção…"`.
2. When `"Nova coleção…"` is selected, show `st.text_input("Nome da coleção")` with inline `RAGSecurityError` validation on change.
3. Pass the resolved `collection_name` to `_ingerir_documento()`.
4. Success messages from `_ingerir_documento()` include the collection name.

No changes to the query flow, `_consultar_api()`, or result display sections.

---

## Tests Required

**Unit tests (new):**
- `tests/test_rag.py::TestSanitizarNomeColecao` — boundary values (3, 63, 64 chars), invalid chars, start/end chars, empty string.
- `tests/test_rag.py::TestAdicionarDocumentoIncrementalmente` — existing tests updated to pass `collection_name`; new test for default collection fallback.
- `tests/test_agent.py::TestInvalidarVectorDb` — both existing tests must be updated: `_vector_db` becomes `_vector_dbs: dict`; `invalidar_vector_db()` clears entire dict; verify no exception on empty dict.
- `tests/test_agent.py::TestGetRetriever` — new tests for fan-out: multiple collections, empty collection skip, ChromaDB error propagation.

**Integration tests (new/updated):**
- `tests/test_rag_integration.py` — all `Chroma(...)` calls must pass `collection_name`; add test for two collections with cross-domain query.

**Acceptance tests (updated):**
- `tests/acceptance/test_document_ingestion_ui.py` — mocks on `adicionar_documento_incrementalmente` must include `collection_name` argument.

**Existing tests that will break without update:**
- `tests/test_agent.py::TestInvalidarVectorDb::teste_1_...` and `teste_2_...` — `_vector_db = None` becomes `_vector_dbs = {}`.
- Any test that constructs a `Chroma(...)` mock without `collection_name` and checks it is called correctly.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Add constants | `DEFAULT_COLLECTION_NAME`, `COLLECTION_NAME_MIN_LEN`, `COLLECTION_NAME_MAX_LEN`, `COLLECTION_NAME_PATTERN` |
| `src/agenticlog/rag.py` | Add function + modify signatures | Add `_sanitizar_nome_colecao()`; add `collection_name` param to `adicionar_documento_incrementalmente`, `salvar_documento_enviado`, `salvar_pdf_enviado`, `cria_vectordb` |
| `src/agenticlog/agent.py` | Refactor singleton + fan-out | `_vector_db: Chroma | None` → `_vector_dbs: dict[str, Chroma]`; add `_get_vector_db(collection_name)`; add `_listar_colecoes()`; refactor `_get_retriever()` for fan-out; update `invalidar_vector_db()` |
| `app.py` | UI — collection selector | Add `st.selectbox` + conditional `st.text_input`; pass `collection_name` to ingestion functions |
| `src/agenticlog/api.py` | Minor — startup check | Update `_verificar_vectordb()` to use `DEFAULT_COLLECTION_NAME` if it currently uses implicit collection name |
| `tests/test_rag.py` | Update + add | New `TestSanitizarNomeColecao` class; update all existing tests that mock `Chroma(...)` |
| `tests/test_agent.py` | Update + add | `TestInvalidarVectorDb` updated; new `TestGetRetriever` fan-out tests |
| `tests/acceptance/test_document_ingestion_ui.py` | Update | Mock signatures for `adicionar_documento_incrementalmente` |
| `tests/test_rag_integration.py` | Update | All `Chroma(...)` calls must pass `collection_name` |

---

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| `collection_name` inconsistency — passed to `Chroma(...)` in rag.py but forgotten in agent.py (or vice versa) | High | MCC-03/MCC-04 explicitly require all constructors; tasks.md will call each file out separately |
| Backward compat — existing `data/vectordb/` uses implicit unnamed collection; querying it with `collection_name="logistica"` returns 0 results | High (expected) | AC11 documents this explicitly; operator must wipe and re-ingest; spec notes it in Process flow |
| Fan-out performance — N collections × k=3 retrieval calls multiplies latency | Medium | Acceptable for initial implementation; no SLA change required by story; flagged for future optimisation |
| `invalidar_vector_db()` dict vs None — callers that check `_vector_db is None` will break | High | All callers are internal; tasks.md will update all call sites in the same task |
| ChromaDB `list_collections()` API shape — `langchain-chroma` wraps the underlying client; the exact method to list collection names must be verified against the installed version | Medium | Researcher confirmed pattern exists; builder must verify against installed `langchain-chroma` version before implementing `_listar_colecoes()` |
| Streamlit re-render — `st.selectbox` listing collections fetched at render time may be stale after a new collection is created | Low | `st.rerun()` after successful ingestion already clears session state; collection list will refresh |
| Single-character valid edge case — ChromaDB requires min 3 chars; the `COLLECTION_NAME_PATTERN` regex `^[a-zA-Z0-9][...]*[a-zA-Z0-9]$` requires ≥ 2 chars for the anchors alone (start + end); a 2-char name would need `*` to match zero middle chars, which is correct with `*` but fails with `+` — use `*` not `+` for middle group | Medium | Explicitly documented in Data Model section; builder must use `*` for middle group |

---

## Open Questions

1. **`_listar_colecoes()` implementation** — The underlying ChromaDB client method to list collections must be confirmed against the version in `requirements.txt`. The researcher identified this as a risk. If `langchain-chroma` does not expose a direct list method, the builder should use `chroma_client.list_collections()` on the raw `chromadb.PersistentClient`. Builder must verify and document the method used.

2. **`reconstruir_vectordb()` with `collection_name`** — Currently `reconstruir_vectordb()` calls `cria_vectordb()` with no arguments. After the change, `cria_vectordb(collection_name)` must be called with the correct name. The current PDF ingestion path calls `reconstruir_vectordb()` with no argument. This path must be updated to pass `collection_name` — confirm this is handled in the PDF task.

---

## Success Criteria

- [ ] Operator can ingest a JSON document into "fornecedores" and a second JSON into "contratos" without them mixing in ChromaDB.
- [ ] A query to the agent returns results from both "fornecedores" and "contratos" collections.
- [ ] Collection name "ab" is rejected with `RAGSecurityError`; "abc" is accepted.
- [ ] Collection name 64 characters long is rejected; 63 characters is accepted.
- [ ] `invalidar_vector_db()` called on empty `_vector_dbs` dict raises no exception.
- [ ] All existing tests pass with no silent deletions; coverage remains ≥ 80 %.
- [ ] `QueryRequest` schema is unchanged — no `collection_name` field.
