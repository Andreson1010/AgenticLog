# Multi-Collection ChromaDB — Tasks

**Design:** `.specs/features/multi-collection-chromadb/design.md`
**Status:** Awaiting human approval

---

## Execution Plan

### Phase 1: Foundation — Constants and Validation (Sequential)

Must be done first; every subsequent task depends on the constants and the validation function.

```
T1 → T2
```

### Phase 2: Core Backend Changes (Parallel after T2)

All three can start once T2 is done; they touch different files/functions with no shared mutable state.

```
T2 ──┬──→ T3 [P]
     ├──→ T4 [P]
     └──→ T5 [P]
```

### Phase 3: Agent Refactor (Sequential after T3)

Depends on T3 (which adds `collection_name` to rag.py, consumed by agent).

```
T3 → T6 → T7
```

### Phase 4: Integration (Sequential after Phase 2 and T7)

```
T4, T5, T7 ──→ T8 → T9
```

### Phase 5: UI and Acceptance (Sequential after T8)

```
T8 → T10 → T11
```

---

## Task Breakdown

### T1: Add collection-name constants to config.py [P]

**What:** Add `DEFAULT_COLLECTION_NAME`, `COLLECTION_NAME_MIN_LEN`, `COLLECTION_NAME_MAX_LEN`, and `COLLECTION_NAME_PATTERN` to `config.py`. Add `import re` if not present.
**Where:** `src/agenticlog/config.py`
**Depends on:** None
**Reuses:** Existing constant style (no hardcoded values pattern from CLAUDE.md)
**Requirement:** MCC-05, MCC-11, MCC-17

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `DEFAULT_COLLECTION_NAME: str = "logistica"` present
- [ ] `COLLECTION_NAME_MIN_LEN: int = 3` present
- [ ] `COLLECTION_NAME_MAX_LEN: int = 63` present
- [ ] `COLLECTION_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$")` present
- [ ] `import re` present at top of file
- [ ] Gate check passes: `pytest tests/test_config.py -v` (if file exists) or `python -c "from agenticlog.config import DEFAULT_COLLECTION_NAME, COLLECTION_NAME_MIN_LEN, COLLECTION_NAME_MAX_LEN, COLLECTION_NAME_PATTERN; print('OK')"` exits 0

**Tests:** unit (import smoke test — no dedicated test file needed for constants alone)
**Gate:** quick

**Commit:** `feat(config): adicionar constantes de nomes de coleção ChromaDB`

---

### T2: Add `_sanitizar_nome_colecao()` to rag.py

**What:** Implement the `_sanitizar_nome_colecao(name: str) -> str` function in `rag.py` with the four validation steps from the design doc. Add imports for new config constants. Add `TestSanitizarNomeColecao` test class.
**Where:** `src/agenticlog/rag.py` (new function near `_sanitizar_nome_arquivo`); `tests/test_rag.py` (new test class)
**Depends on:** T1
**Reuses:** `_sanitizar_nome_arquivo()` structure; `RAGSecurityError`; `COLLECTION_NAME_MIN_LEN`, `COLLECTION_NAME_MAX_LEN`, `COLLECTION_NAME_PATTERN` from config
**Requirement:** MCC-06, MCC-07, MCC-08, MCC-09, MCC-10, MCC-11

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `_sanitizar_nome_colecao("")` raises `RAGSecurityError`
- [ ] `_sanitizar_nome_colecao("ab")` raises `RAGSecurityError` (too short)
- [ ] `_sanitizar_nome_colecao("abc")` returns `"abc"` (boundary — valid)
- [ ] `_sanitizar_nome_colecao("a" * 63)` returns the 63-char string (boundary — valid)
- [ ] `_sanitizar_nome_colecao("a" * 64)` raises `RAGSecurityError` (too long)
- [ ] `_sanitizar_nome_colecao("nome colecao")` raises `RAGSecurityError` (space is invalid)
- [ ] `_sanitizar_nome_colecao("-inicio")` raises `RAGSecurityError` (starts with hyphen)
- [ ] `_sanitizar_nome_colecao("fim-")` raises `RAGSecurityError` (ends with hyphen)
- [ ] `_sanitizar_nome_colecao("valido-nome_1")` returns `"valido-nome_1"`
- [ ] Gate check passes: `pytest tests/test_rag.py::TestSanitizarNomeColecao -v` — all tests green
- [ ] Test count: minimum 8 test methods in `TestSanitizarNomeColecao`

**Tests:** unit
**Gate:** quick (`pytest tests/test_rag.py::TestSanitizarNomeColecao -v`)

**Commit:** `feat(rag): adicionar _sanitizar_nome_colecao com validação ChromaDB`

---

### T3: Add `collection_name` param to rag.py write functions [P]

**What:** Add `collection_name: str = DEFAULT_COLLECTION_NAME` parameter to `adicionar_documento_incrementalmente`, `salvar_documento_enviado`, `salvar_pdf_enviado`, `cria_vectordb`, and `reconstruir_vectordb`. Call `_sanitizar_nome_colecao(collection_name)` at the top of each function. Pass `collection_name` to all `Chroma(...)` and `Chroma.from_documents(...)` constructor calls within each function. Update imports in rag.py to include `DEFAULT_COLLECTION_NAME`.
**Where:** `src/agenticlog/rag.py`; `tests/test_rag.py` (update all existing tests that mock `Chroma`)
**Depends on:** T2
**Reuses:** Existing function bodies unchanged except `Chroma(...)` constructor calls
**Requirement:** MCC-03, MCC-04, MCC-05, MCC-17

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `adicionar_documento_incrementalmente("f.json", b"{}", collection_name="teste")` passes `collection_name="teste"` to `Chroma(...)` (verified via mock assert)
- [ ] `adicionar_documento_incrementalmente("f.json", b"{}")` uses `collection_name="logistica"` (default)
- [ ] `adicionar_documento_incrementalmente("f.json", b"{}", collection_name="ab")` raises `RAGSecurityError` (validation called before any write)
- [ ] `salvar_pdf_enviado` and `salvar_documento_enviado` signatures updated and validation called
- [ ] `cria_vectordb()` passes `collection_name` to `Chroma.from_documents(...)`
- [ ] All previously passing tests in `tests/test_rag.py` still pass (no regressions — update mock call signatures to include `collection_name`)
- [ ] Gate check passes: `pytest tests/test_rag.py -v`
- [ ] Test count: no fewer tests than before this task (no silent deletions)

**Tests:** unit
**Gate:** quick (`pytest tests/test_rag.py -v`)

**Commit:** `feat(rag): adicionar collection_name a todas as funções de escrita`

---

### T4: Update `salvar_pdf_enviado` call path in app.py ingestion [P]

**What:** Update the `_ingerir_documento()` helper in `app.py` to accept a `collection_name: str` parameter and pass it to `salvar_pdf_enviado(filename, conteudo, collection_name)` and `reconstruir_vectordb(collection_name)`. Update `salvar_documento_enviado` call if used. Update success messages to include collection name.
**Where:** `app.py`
**Depends on:** T2 (signature defined); T3 can be done in parallel — T4 depends only on the new signatures being defined in T3, so T4 must come after T3
**Reuses:** Existing `_ingerir_documento()` structure; `st.success()`, `st.error()` calls
**Requirement:** MCC-03, MCC-04

**Note:** This task prepares `_ingerir_documento()` for the `collection_name` argument. The UI widget that provides the `collection_name` is wired in T10. For now, `_ingerir_documento()` can accept `collection_name` with a default of `DEFAULT_COLLECTION_NAME` so existing tests continue to pass.

**Depends on:** T3
**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `_ingerir_documento(uploaded_file, collection_name="fornecedores")` calls `salvar_pdf_enviado(filename, conteudo, "fornecedores")`
- [ ] `_ingerir_documento(uploaded_file, collection_name="fornecedores")` calls `reconstruir_vectordb("fornecedores")`
- [ ] `_ingerir_documento(uploaded_file, collection_name="fornecedores")` success message includes `"fornecedores"`
- [ ] `_ingerir_documento(uploaded_file)` still works with default collection (backward compat)
- [ ] Gate check passes: `pytest tests/acceptance/test_document_ingestion_ui.py -v`
- [ ] Test count: no fewer tests than before this task

**Tests:** unit (mock-based; acceptance tests updated in T11)
**Gate:** quick (`pytest tests/acceptance/test_document_ingestion_ui.py -v`)

**Commit:** `feat(app): passar collection_name para funções de ingestão`

---

### T5: Update api.py startup check to use DEFAULT_COLLECTION_NAME [P]

**What:** In `src/agenticlog/api.py`, update `_verificar_vectordb()` (or equivalent startup health-check function) to open ChromaDB with `collection_name=DEFAULT_COLLECTION_NAME` so it does not use the implicit unnamed collection. Import `DEFAULT_COLLECTION_NAME` from `config`.
**Where:** `src/agenticlog/api.py`
**Depends on:** T1
**Reuses:** Existing `_verificar_vectordb()` logic
**Requirement:** MCC-05 (default collection used when none specified)

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `_verificar_vectordb()` opens `Chroma(persist_directory=..., collection_name=DEFAULT_COLLECTION_NAME, ...)` or equivalent
- [ ] No hardcoded collection name string in `api.py`
- [ ] Gate check passes: `pytest tests/ -k "api" -v` (or equivalent api test filter)
- [ ] Test count: no regressions in api tests

**Tests:** unit
**Gate:** quick

**Commit:** `feat(api): usar DEFAULT_COLLECTION_NAME na verificação de startup`

---

### T6: Refactor agent.py — `_vector_dbs` dict + `_get_vector_db(collection_name)`

**What:** Replace `_vector_db: Chroma | None = None` with `_vector_dbs: dict[str, Chroma] = {}`. Add `_get_vector_db(collection_name: str) -> Chroma` function. Update `invalidar_vector_db()` to call `_vector_dbs.clear()`. Update `inicializar_recursos()` to call `_get_vector_db(DEFAULT_COLLECTION_NAME)`. Add `DEFAULT_COLLECTION_NAME` to the `config` import block. Update `tests/test_agent.py::TestInvalidarVectorDb` to reflect dict semantics.
**Where:** `src/agenticlog/agent.py`; `tests/test_agent.py`
**Depends on:** T3 (collection_name available in rag.py; agent imports DEFAULT_COLLECTION_NAME from same config)
**Reuses:** Existing lazy-singleton pattern; `_get_embedding_model()`; `DIR_VECTORDB`
**Requirement:** MCC-15, MCC-17

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `_vector_dbs` is a module-level `dict[str, Chroma]` initialised to `{}`
- [ ] `_get_vector_db("fornecedores")` returns a `Chroma` instance with `collection_name="fornecedores"`; second call returns cached instance
- [ ] `invalidar_vector_db()` calls `_vector_dbs.clear()`; subsequent `_get_vector_db()` call creates new instance
- [ ] `invalidar_vector_db()` on empty dict raises no exception
- [ ] `inicializar_recursos()` pre-warms `_get_vector_db(DEFAULT_COLLECTION_NAME)`
- [ ] `tests/test_agent.py::TestInvalidarVectorDb` updated and passing (both existing tests rewritten for dict semantics)
- [ ] Gate check passes: `pytest tests/test_agent.py::TestInvalidarVectorDb -v`
- [ ] Test count: same or more test methods in `TestInvalidarVectorDb`

**Tests:** unit
**Gate:** quick (`pytest tests/test_agent.py::TestInvalidarVectorDb -v`)

**Commit:** `refactor(agent): migrar singleton _vector_db para dict _vector_dbs`

---

### T7: Implement fan-out `_get_retriever()` and `_listar_colecoes()` in agent.py

**What:** Add `_listar_colecoes() -> list[str]` using lazy `chromadb.PersistentClient`. Refactor `_get_retriever()` to accept `query: str` and return `list[Document]` via fan-out across all collections. Update `retrieve_info()` to call `_get_retriever(state.query)` instead of `_get_retriever().invoke(state.query)`. Add `TestGetRetriever` test class with fan-out, empty-collection-skip, and ChromaDB-error-propagation cases.
**Where:** `src/agenticlog/agent.py`; `tests/test_agent.py`
**Depends on:** T6
**Reuses:** `_get_vector_db()` from T6; `hashlib` (already imported); `Document` from langchain_core
**Requirement:** MCC-12, MCC-13, MCC-14, MCC-15

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `_listar_colecoes()` returns list of collection name strings from `chromadb.PersistentClient(path=...).list_collections()` (or `.name` attribute per installed version); falls back to `[DEFAULT_COLLECTION_NAME]` on exception
- [ ] `_get_retriever("test query")` with two populated collections returns merged `list[Document]` with max 3 unique documents
- [ ] `_get_retriever("test query")` with an empty collection skips it silently (0 docs contributed)
- [ ] `_get_retriever("test query")` with a ChromaDB exception re-raises immediately
- [ ] `_get_retriever("test query")` with zero collections returns `[]`
- [ ] `retrieve_info(state)` calls `_get_retriever(state.query)` (not `.invoke()`)
- [ ] All nodes downstream of `retrieve_info` are unchanged and still pass their existing tests
- [ ] Gate check passes: `pytest tests/test_agent.py -v`
- [ ] Test count: `TestGetRetriever` has minimum 5 test methods; no previously passing tests removed

**Tests:** unit
**Gate:** quick (`pytest tests/test_agent.py -v`)

**Commit:** `feat(agent): fan-out retrieval em múltiplas coleções ChromaDB`

---

### T8: Integration test — two named collections, cross-domain query

**What:** Add integration tests to `tests/test_rag_integration.py` that: (a) ingest a JSON into "fornecedores" collection, (b) ingest a JSON into "contratos" collection, (c) submit a query that should match both, (d) verify retrieved documents contain hits from both collections. Update all existing integration test `Chroma(...)` calls to pass `collection_name`.
**Where:** `tests/test_rag_integration.py`
**Depends on:** T7 (fan-out complete); T3 (rag.py write functions complete)
**Reuses:** Existing integration test fixtures and patterns; `DEFAULT_COLLECTION_NAME`
**Requirement:** MCC-12, MCC-13

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Existing integration tests updated to pass `collection_name` to all `Chroma(...)` mocks or live calls
- [ ] New test: ingest into "fornecedores" + "contratos" → query → documents from both collections present in result
- [ ] New test: ingest into one collection only → query → results only from that collection
- [ ] Gate check passes: `pytest tests/test_rag_integration.py -v`
- [ ] Test count: no previously passing tests removed

**Tests:** integration
**Gate:** full (`pytest tests/test_rag_integration.py -v`)

**Commit:** `test(integration): testes de ingestão e recuperação multi-coleção`

---

### T9: Full coverage gate — run all tests, verify ≥ 80 %

**What:** Run the full test suite with coverage report. Fix any remaining regressions. Document final coverage percentage.
**Where:** All test files
**Depends on:** T8
**Reuses:** Existing pytest + coverage configuration
**Requirement:** All MCC requirements (gate)

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` exits 0
- [ ] Coverage ≥ 80 % overall
- [ ] Zero test failures
- [ ] No previously passing tests removed (test count ≥ baseline)

**Tests:** unit + integration
**Gate:** full (`pytest --cov=agenticlog --cov-report=term-missing -v`)

**Commit:** N/A (gate task — no production code changes expected)

---

### T10: UI — collection selector widget in app.py

**What:** Add `st.selectbox` listing existing ChromaDB collections + "Nova coleção…" option. Add conditional `st.text_input` with inline `RAGSecurityError` validation when "Nova coleção…" is selected. Resolve `collection_name` and pass to `_ingerir_documento()`.
**Where:** `app.py`
**Depends on:** T4 (ingestion function accepts `collection_name`); T7 (`_listar_colecoes()` available)
**Reuses:** Existing `st.rerun()` pattern; `RAGSecurityError`; `_sanitizar_nome_colecao()`
**Requirement:** MCC-01, MCC-02, MCC-16

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Sidebar expander shows `st.selectbox` with collection names from `_listar_colecoes()` + "Nova coleção…"
- [ ] Selecting "Nova coleção…" reveals `st.text_input`
- [ ] Text input shows `st.caption("Nome válido.")` for valid names
- [ ] Text input shows `st.caption(f"Nome inválido: {e}")` for invalid names (caught `RAGSecurityError`)
- [ ] Selected/typed `collection_name` is passed to `_ingerir_documento(uploaded_file, collection_name)`
- [ ] When no collections exist yet, dropdown shows only "Nova coleção…"
- [ ] Gate check passes: `pytest tests/acceptance/test_document_ingestion_ui.py -v`

**Tests:** unit (UI behaviour tested via mocks in acceptance test file)
**Gate:** quick (`pytest tests/acceptance/test_document_ingestion_ui.py -v`)

**Commit:** `feat(ui): seletor de coleção ChromaDB no sidebar de ingestão`

---

### T11: Update acceptance tests for UI collection selector

**What:** Update `tests/acceptance/test_document_ingestion_ui.py` to mock the collection dropdown, test selection of existing collection, test "Nova coleção…" flow, and verify `adicionar_documento_incrementalmente` is called with the correct `collection_name` argument.
**Where:** `tests/acceptance/test_document_ingestion_ui.py`
**Depends on:** T10
**Reuses:** Existing mock patterns in the file
**Requirement:** MCC-01, MCC-02, MCC-03, MCC-04, MCC-16

**Tools:**
- MCP: NONE
- Skill: NONE

**Done when:**
- [ ] Test: selecting "fornecedores" from dropdown and clicking Ingerir → `adicionar_documento_incrementalmente` called with `collection_name="fornecedores"`
- [ ] Test: selecting "Nova coleção…", typing "novos-contratos", clicking Ingerir → called with `collection_name="novos-contratos"`
- [ ] Test: typing invalid name "ab" in new-collection input → `RAGSecurityError` shown as `st.caption` (inline), no ingestion call
- [ ] All previously passing acceptance tests still pass
- [ ] Gate check passes: `pytest tests/acceptance/test_document_ingestion_ui.py -v`
- [ ] Test count: minimum 3 new test methods added

**Tests:** unit (mock-based acceptance)
**Gate:** quick (`pytest tests/acceptance/test_document_ingestion_ui.py -v`)

**Commit:** `test(acceptance): testes de seletor de coleção na UI`

---

## Parallel Execution Map

```
Phase 1 (Sequential — Foundation):
  T1 ──→ T2

Phase 2 (Parallel — Backend, all start after T2):
  T2 complete, then:
    ├── T3 [P]   (rag.py write functions)
    ├── T4 [P]   (app.py ingestion helper)   ← NOTE: T4 depends on T3 signatures; move T4 to after T3
    └── T5 [P]   (api.py startup check)

  Corrected: T3 [P] and T5 [P] run in parallel after T2.
             T4 runs after T3 (depends on T3 signatures).

Phase 3 (Sequential — Agent refactor):
  T3 complete, then:
    T3 → T6 → T7

Phase 4 (Sequential — Integration):
  T5, T7 complete, then:
    T7 → T4 → T8 → T9

  (T4 can be done after T3 but before T8 as long as T3 is done)

Phase 5 (Sequential — UI and Acceptance):
  T9 complete, then:
    T9 → T10 → T11
```

**Revised linear order respecting all dependencies:**

```
T1 → T2 → T3 ──┬──→ T4
               │
               └──→ T6 → T7 → T8 → T9 → T10 → T11
T1 → T5 ──────────────────────────────────────────┘
```

**Parallel opportunities:**
- T3 and T5 can run in parallel after T2 (different files, no shared state).
- T4 must follow T3 (depends on updated signatures).
- T6 must follow T3 (imports `DEFAULT_COLLECTION_NAME` and uses rag.py contract).

---

## Task Granularity Check

| Task | Scope | Status |
|------|-------|--------|
| T1: Add 4 constants to config.py | 1 file, 4 related constants | OK — cohesive constants block |
| T2: Add `_sanitizar_nome_colecao` + unit tests | 1 function + 1 test class | OK — single deliverable with co-located tests |
| T3: Add `collection_name` to 5 rag.py functions | 1 file, related signature changes | OK — cohesive file change; all follow same pattern |
| T4: Update `_ingerir_documento` in app.py | 1 function in 1 file | OK |
| T5: Update `_verificar_vectordb` in api.py | 1 function in 1 file | OK |
| T6: `_vector_dbs` dict + `_get_vector_db` | 1 file, 2 related functions | OK — cohesive agent singleton change |
| T7: `_listar_colecoes` + fan-out `_get_retriever` | 1 file, 2 coupled functions | OK — inseparable; fan-out requires collection list |
| T8: Integration tests | 1 test file | OK |
| T9: Full coverage gate | All tests | OK — gate task |
| T10: UI widget in app.py | 1 file, 1 UI block | OK |
| T11: Acceptance tests update | 1 test file | OK |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
|------|------------------------|---------------|--------|
| T1 | None | No incoming arrows | OK |
| T2 | T1 | T1 → T2 | OK |
| T3 | T2 | T2 → T3 | OK |
| T4 | T3 | T3 → T4 | OK |
| T5 | T1 | T1 → T5 (parallel to T2 path) | OK |
| T6 | T3 | T3 → T6 | OK |
| T7 | T6 | T6 → T7 | OK |
| T8 | T7 (+ T3 implied via T6/T7 chain) | T7 → T8 | OK |
| T9 | T8 | T8 → T9 | OK |
| T10 | T4, T7 (via T9) | T9 → T10 | OK |
| T11 | T10 | T10 → T11 | OK |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Test Type Required | Task Says | Status |
|------|-----------------------------|--------------------|-----------|--------|
| T1 | config.py constants | smoke/import | unit (smoke) | OK |
| T2 | `_sanitizar_nome_colecao()` in rag.py | unit | unit | OK |
| T3 | Modified rag.py write functions | unit | unit | OK |
| T4 | `_ingerir_documento()` in app.py | unit | unit | OK |
| T5 | `_verificar_vectordb()` in api.py | unit | unit | OK |
| T6 | `_vector_dbs` + `_get_vector_db()` in agent.py | unit | unit | OK |
| T7 | `_listar_colecoes()` + `_get_retriever()` in agent.py | unit | unit | OK |
| T8 | Cross-collection integration scenario | integration | integration | OK |
| T9 | All layers — coverage gate | full | full | OK |
| T10 | UI widget in app.py | unit (mock-based) | unit | OK |
| T11 | Acceptance test update | unit (mock-based) | unit | OK |
