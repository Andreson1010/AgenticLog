# RAG E2E Characterization Test Net — Technical Spec

**Path:** `.specs/features/rag-caracterizacao-e2e/spec.md`
**TLC scope:** large
**Based on story:** E2E characterization tests that exercise the real ingestion+retrieval pipeline through public entry points only, so the upcoming RAG refactor (ADR-018) can be validated against pinned behavior instead of internal-patch tests that break on rename.
**Status:** Awaiting human approval

---

## Problem Statement

ADR-018 will extract modules from the monolithic `src/agenticlog/rag.py` and `src/agenticlog/agent.py`, renaming and moving internal functions. The existing safety net (`tests/test_rag_integration.py`, `tests/test_agentic_rag.py`) pins behavior by patching internal seams — `agenticlog.rag._get_rag_embedding_model`, `agenticlog.rag.Chroma`, `agenticlog.rag.JSONLoader`, `agenticlog.rag.SemanticChunker` — which will break on the first rename, giving false failures and zero refactor protection. Before any code moves, we need an end-to-end characterization net that drives only **public entry points** and mocks only **stable module-level boundary seams**, so the tests survive the refactor and act as a stable behavioral oracle.

## Goals

- [ ] Pin 5 observable end-to-end behaviors (AC-1..AC-5) through public entry points only.
- [ ] Mock exclusively the boundary seams that survive the refactor (LLM, embeddings, web search, path bindings); never patch internal code-paths slated for rename/move.
- [ ] Reach the real `Chroma` / `SemanticChunker` / `JSONLoader` so the net characterizes the genuine pipeline, not a mock of it.
- [ ] Ship as one auto-discovered, function-scoped, `@pytest.mark.integration` file with zero production-code change.
- [ ] Reuse the existing `rag_eval` golden gate for retrieval quality — introduce no new quality thresholds.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Any edit to `src/`, `app.py`, `api.py` | Phase 1 adds tests only; zero production change is a locked decision. |
| Extracting/moving/splitting RAG modules | Later ADR-018 phases; this net is the precondition for them. |
| New retrieval-quality thresholds or metrics | Reuse the `rag_eval` golden gate; no duplicate quality assertions. |
| Replacing or deleting existing unit/integration tests | The new net is additive; cleanup of internal-patch tests is a later-phase concern. |
| Real LMStudio / real DuckDuckGo / real HuggingFace download in the net | LLM, embeddings and web search are mocked at the boundary; the net is deterministic and offline. |

---

## User Stories

### P1: E2E Characterization Net via Public Entry Points ⭐ MVP
**User Story**: As a maintainer about to refactor the RAG package, I want E2E characterization tests that exercise the real ingestion+retrieval pipeline through public entry points only, so that the refactor can be validated against pinned behavior instead of internal-patch tests that break on rename.
**Why P1**: This net is the gate that makes the rest of ADR-018 safe. Without it the refactor is unverifiable.

**Acceptance Criteria**:
1. WHEN a JSON doc is ingested into a fresh tmp vector DB and `agent_workflow.invoke(AgentState(query=...))` runs with a matching query, THEN the system SHALL return non-empty retrieval (`retrieved_info`), a present `ranked_response`, and `confidence_score > 0`. *(AC-1)*
2. WHEN a query runs against an empty collection, THEN `retrieve_info` SHALL fall back to the `gerar` path, produce a response, and not crash. *(AC-2)*
3. WHEN a query that triggers the `usar_web` keyword route is invoked with `agenticlog.agent.search` mocked, THEN the web branch SHALL run and the graph SHALL reach END. *(AC-3)*
4. WHEN a NEW doc is added via `adicionar_documento_incrementalmente` / `adicionar_pdf_incrementalmente` into a populated tmp DB within the same test, THEN that doc SHALL be queryable through `agent_workflow.invoke`. *(AC-4)*
5. WHEN an upsert of an existing indexed doc fails because `embed_documents` raises at the `rag._rag_embedding_model` boundary, THEN the disk file AND the Chroma collection SHALL be left unchanged (rollback, per PRs #42/#43). *(AC-5)*

**Independent Test**: Run `pytest -m integration tests/test_rag_caracterizacao.py -v` on CI Linux; all five behaviors pass against the real Chroma store without LMStudio, DuckDuckGo or an HF download.

---

## Edge Cases

- WHEN the embedding stub returns constant vectors THEN `SemanticChunker` SHALL still emit at least one chunk (zero-distance breakpoints collapse to a single chunk) and ingestion SHALL succeed.
- WHEN `retrieve_info` receives 0 documents THEN it SHALL set `next_step="gerar"` and `retrieved_info=[]` (fail-loud WARNING, no exception) — this is the AC-2 oracle.
- WHEN the agent caches `_vector_dbs` from a prior test THEN the fixture SHALL reset it so a fresh tmp store is never shadowed by a stale cached `Chroma` handle.
- WHEN ingestion calls `invalidar_vector_db()` THEN the net SHALL let it run (not mock it) so the agent re-reads the newly-written collection in AC-4.
- WHEN a test runs locally on Windows THEN the unsigned `hnswlib` DLL may be blocked by Smart App Control; the `@pytest.mark.integration` marker SHALL signal that CI Linux is the authoritative gate (local skip/failure is environmental, not a regression).

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| CHAR-01 | P1 / AC-1 | Tasks (T-04) | Pending |
| CHAR-02 | P1 / AC-2 | Tasks (T-05) | Pending |
| CHAR-03 | P1 / AC-3 | Tasks (T-06) | Pending |
| CHAR-04 | P1 / AC-4 | Tasks (T-07) | Pending |
| CHAR-05 | P1 / AC-5 | Tasks (T-08) | Pending |
| CHAR-06 | P1 (mock-boundary contract) | Design §3 / Tasks (T-01..T-08) | Pending |
| CHAR-07 | P1 (`@pytest.mark.integration` on every test) | Tasks (T-01..T-08) | Pending |
| CHAR-08 | P1 (single flat file, function-scoped fixtures) | Tasks (T-01..T-03) | Pending |
| CHAR-09 | P1 (zero production-code change) | Tasks (T-09 verify) | Pending |
| CHAR-10 | P1 (reuse `rag_eval` gate; no new thresholds) | Design §10 / Tasks (T-09) | Pending |

**ID format:** `CHAR-[NUMBER]` — CHAR = characterization-net slug prefix.

---

## Data Model Changes

No data model changes. The net writes only to a `tmp_path`-scoped ChromaDB store and asserts against the existing `AgentState` (Pydantic) fields.

**Observable AgentState fields used as oracles** (defined `src/agenticlog/agent.py` L275-288):

| Field | Type | Oracle role |
|-------|------|-------------|
| `retrieved_info` | `list` | Retrieved source documents. **AC-1 "sources non-empty" maps to this field** — there is no `sources` field on `AgentState`; `retrieved_info` is the canonical source list. |
| `ranked_response` | `str` | Final answer text (present / non-empty). |
| `confidence_score` | `float` | Confidence (`> 0` for AC-1; `0.0` on the web/empty paths). |
| `next_step` | `str` | Route taken: `"retrieve"`, `"gerar"`, `"usar_web"` (AC-2 / AC-3 oracle). |

> **Naming note (carry into design + tests):** the approved story says `sources`; the real field is `retrieved_info`. This is a naming mapping, not a behavioral change — no contradiction with Checkpoint 1.

---

## Process / Background Flow

**Entry point (all read paths):** `agent_workflow.invoke(AgentState(query=...))` — compiled LangGraph at `agent.py` L474.
**Entry point (write/ingest paths):** `rag.adicionar_documento_incrementalmente(...)` and `rag.adicionar_pdf_incrementalmente(...)`.

**Happy path (AC-1):** seed tmp store → ingest JSON via `adicionar_documento_incrementalmente` → `invalidar_vector_db()` clears agent cache → `agent_workflow.invoke` → `passo_decisao_agente` routes `retrieve` → `retrieve_info` → real `_get_retriever` fan-out over real `Chroma` → `gera_multiplas_respostas` (LLM stub) → `avalia_similaridade` (embedding stub) → `rank` → state with non-empty `retrieved_info`, present `ranked_response`, `confidence_score > 0`.

**Failure / fallback path — empty DB (AC-2):** empty collection → `_get_retriever` returns `[]` → `retrieve_info` logs WARNING, sets `next_step="gerar"`, `retrieved_info=[]` → `gera_multiplas_respostas` runs context-free prompt (LLM stub) → response produced, no crash.

**Web path (AC-3):** query containing a `ROUTING_KEYWORDS_WEB` term → `passo_decisao_agente` sets `next_step="usar_web"` → `usar_ferramenta_web` calls `search.run` (mocked via `@patch("agenticlog.agent.search")`) → LLM stub → graph reaches END.

**Rollback path (AC-5):** existing indexed doc re-upserted with different content → after `shutil.move` the guarded block builds `SemanticChunker(embeddings=rag._rag_embedding_model)`; `embed_documents` raises → `_reverter_disco` restores the disk file from backup and the chunk-id `delete` rolls back Chroma → disk file AND collection unchanged.

---

## API Changes

No API changes. No new functions, endpoints, or exported symbols in `src/`. The only new artifact is the test module `tests/test_rag_caracterizacao.py`.

---

## Frontend Changes

No frontend changes.

---

## Tests Required

This feature **is** the test net. All tests live in `tests/test_rag_caracterizacao.py`, each marked `@pytest.mark.integration`, fixtures function-scoped.

**Integration / E2E (the net):**
- `teste_1_*` (CHAR-01 / AC-1) — happy retrieve via `agent_workflow.invoke`; assert `retrieved_info` non-empty, `ranked_response` present, `confidence_score > 0`.
- `teste_2_*` (CHAR-02 / AC-2) — empty-collection fallback; assert `next_step == "gerar"`, `retrieved_info == []`, response produced, no exception.
- `teste_3_*` (CHAR-03 / AC-3) — web route with `agent.search` mocked; assert web branch ran and graph reached END (`ranked_response` set, `confidence_score == 0.0`).
- `teste_4_*` (CHAR-04 / AC-4) — ingest NEW doc into a populated tmp DB within the same test, then query; assert it is retrievable.
- `teste_5_*` (CHAR-05 / AC-5) — stub `rag._rag_embedding_model.embed_documents` to raise during upsert; assert disk file unchanged AND Chroma collection count/ids unchanged.

**Naming:** `teste_N_<descricao>` prefix (project convention, CLAUDE.md / TESTING.md).

**Edge case tests folded in:** empty-retrieval (AC-2), constant-vector chunking (covered implicitly by AC-1/AC-4 ingestion succeeding), cache-reset isolation (fixture).

**Existing tests that will break:** none expected. The net is additive and does not modify production code, so `tests/test_rag_integration.py` and `tests/test_agentic_rag.py` continue to pass unchanged. (Those internal-patch tests are the ones the refactor will later break; replacing them is out of scope here.)

**Quality reuse (CHAR-10):** retrieval-quality assertions are NOT duplicated here. The `rag_eval` golden gate (`scripts/rag_eval.py` + `evals/rag_golden.json`, feature `rag-golden-eval-ci`) remains the authority for Hit Rate / MRR. This net asserts only structural/observable behavior.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `tests/test_rag_caracterizacao.py` | **Create** | The entire characterization net (fixtures + 5 tests). Single flat file (locked decision 3). |
| `src/**`, `app.py`, `api.py` | **No change** | Zero production change (locked, CHAR-09). |
| `pyproject.toml` (`[tool.pytest.ini_options].markers`) | **Verify / possibly add** | Ensure `integration` marker is registered to avoid `PytestUnknownMarkWarning`. Add the marker entry only if not already present; flag if a production-config edit is considered out-of-bounds for this phase. |

---

## Risks

- **R1 — Cross-store miss (agent reads a different dir than ingestion wrote).** The agent resolves its persist directory from `agent.DIR_VECTORDB` (L150) and `_listar_colecoes` from the same binding (L167); ingestion uses `rag.DIR_VECTORDB`. The fixture MUST redirect **both** bindings to the same `tmp_path`. Mitigation: design §3 fixture sets both. *Found.*
- **R2 — Stale agent singletons leak across tests.** `agent._vector_dbs`, `agent._embedding_model`, `agent._llm` are module-level caches. Without reset, a cached `Chroma` handle from a prior test shadows the new tmp store. Mitigation: function-scoped fixture resets all three (locked decision 2). *Found.*
- **R3 — LLM stub not Runnable-composable.** Chains are `prompt | _get_llm() | StrOutputParser()`; a bare `MagicMock` may not compose/return a string. Mitigation: stub `agent._llm` with a LangChain-compatible fake (e.g. `FakeListChatModel` / `RunnableLambda` returning deterministic text) — design §3.3. *Found.*
- **R4 — Embedding-space inconsistency between write and read.** Ingestion (`rag._rag_embedding_model`) and query (`agent._embedding_model`) must share the SAME deterministic, fixed-dimension stub or retrieval is degenerate. Mitigation: a single shared deterministic stub used on both sides — design §3.2. *Found.*
- **R5 — hnswlib / Smart App Control blocks Chroma locally on Windows.** ~15 chroma tests already fail locally for this environmental reason (MEMORY: feedback-hnswlib-SAC-block). Mitigation: `@pytest.mark.integration` on every test; CI Linux is the authoritative gate, documented in tasks. *Found.*
- **R6 — `invalidar_vector_db` accidentally mocked.** Existing integration tests patch it out; if copied blindly, AC-4 would never see the new doc (agent keeps the stale cache). Mitigation: design §3 forbids mocking it; the net lets it run. *Found.*
- **R7 — AC-5 false pass via wrong failure trigger.** Locked decision 1 fixes the trigger at `rag._rag_embedding_model.embed_documents` raising — NOT read-only dirs, NOT patching `Chroma`. Patching `Chroma` is forbidden and would not characterize the real rollback. Mitigation: design §3 ALLOWED/FORBIDDEN table; T-08 asserts both disk and collection invariants. *Found.*
- **R8 — `integration` marker unregistered.** Could emit `PytestUnknownMarkWarning` or be filtered out of CI selection. Mitigation: verify `pyproject.toml` markers; T-01 checks registration. *Found — needs verification at build time.*
- **CLAUDE.md conflicts:** none. Conventions honored — `teste_N_` naming, always-mock-LLM, always-test-empty-retrieval, validate AgentState keys. The net is additive and offline. *Clear.*

---

## Open Questions

None — the 3 locked decisions closed them. One naming mapping is recorded (story `sources` → `AgentState.retrieved_info`) and one config touch is flagged for human ruling (registering the `integration` marker in `pyproject.toml`, if not already present — see Files table).

---

## Success Criteria

- [ ] `tests/test_rag_caracterizacao.py` exists with 5 `@pytest.mark.integration`, function-scoped, `teste_N_`-named tests mapping 1:1 to AC-1..AC-5.
- [ ] Every mock/stub in the file is drawn ONLY from the ALLOWED seam list; zero uses of any FORBIDDEN patch target.
- [ ] Tests drive only public entry points (`agent_workflow.invoke`, `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`).
- [ ] Real `Chroma` / `SemanticChunker` / `JSONLoader` are exercised (not mocked).
- [ ] `pytest -m integration tests/test_rag_caracterizacao.py -v` passes on CI Linux without LMStudio, DuckDuckGo, or an HF model download.
- [ ] `git diff` shows changes confined to `tests/` (and, at most, the `pyproject.toml` markers list) — no `src/`, `app.py`, `api.py` edits.
- [ ] No new retrieval-quality thresholds introduced; `rag_eval` remains the quality authority.
