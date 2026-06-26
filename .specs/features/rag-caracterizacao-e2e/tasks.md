# RAG E2E Characterization Test Net — Tasks

**Path:** `.specs/features/rag-caracterizacao-e2e/tasks.md`
**Spec:** `.specs/features/rag-caracterizacao-e2e/spec.md`
**Design:** `.specs/features/rag-caracterizacao-e2e/design.md`
**TLC scope:** large
**Status:** Awaiting human approval

> Tasks are for human + builder planning in the feature-factory pipeline (not TLC Execute).
> Gate command (authoritative, CI Linux): `pytest -m integration tests/test_rag_caracterizacao.py -v`
> Local quick check (Windows, may skip on hnswlib/SAC): same command — environmental skip/fail is expected, not a regression.
> Guardrail for every task: use ONLY ALLOWED seams (design §3.1); zero FORBIDDEN patches (design §3.2). All tests `@pytest.mark.integration`, `teste_N_` named, function-scoped fixtures.

---

## T-01 — Module scaffold, marker, and stub-composability spike
**Requirements:** CHAR-06, CHAR-07, CHAR-08
**Depends on:** none
**Description:**
- Create `tests/test_rag_caracterizacao.py` with module docstring, imports of public symbols only (`agent_workflow`, `AgentState` from `agenticlog.agent`; `adicionar_documento_incrementalmente`, `adicionar_pdf_incrementalmente`, `DEFAULT_COLLECTION_NAME` from `agenticlog.rag`/`config`), and `pytestmark = pytest.mark.integration`.
- Confirm `integration` is registered in `pyproject.toml` `[tool.pytest.ini_options].markers`. If absent, register it; if registering is judged out-of-scope for a tests-only phase, flag to human (do not leave an unregistered-mark warning).
- Spike the LLM stub: prove `prompt | _stub_llm() | StrOutputParser()` composes and returns a string (R3). Lock the choice (`FakeListChatModel` vs `RunnableLambda`).
**Done when:** file imports cleanly, `pytest --collect-only tests/test_rag_caracterizacao.py` lists the module under the `integration` marker with no `PytestUnknownMarkWarning`; the LLM stub composes in a throwaway chain.
**Gate/test notes:** collection-only; no Chroma yet, so runs even on local Windows.

## T-02 — Deterministic stub helpers
**Requirements:** CHAR-06 (§3.3), R3, R4
**Depends on:** T-01
**Description:** Implement `_stub_embedding()` (shared `embed_documents`/`embed_query`, fixed dim, deterministic), `_stub_embedding_que_falha()` (`embed_documents` raises `RuntimeError`), `_stub_llm()` (locked in T-01). No production imports of forbidden seams.
**Done when:** unit-level sanity (a tiny inline assert) shows `embed_documents`/`embed_query` return correct shapes and the failing variant raises.
**Gate/test notes:** pure-Python; runs anywhere.

## T-03 — `rag_caracterizacao_env` function-scoped fixture
**Requirements:** CHAR-08, R1, R2, R6
**Depends on:** T-02
**Description:** Implement the fixture per design §4 using `monkeypatch.setattr`:
- redirect `rag.DIR_DOCUMENTS`, `rag.DIR_VECTORDB`, **and `agent.DIR_VECTORDB`** to the `tmp_path` store (R1);
- set `rag._rag_embedding_model` and `agent._embedding_model` to the SAME shared stub (R4);
- set `agent._llm` to the LLM stub; reset `agent._vector_dbs = {}` (R2);
- do NOT mock `invalidar_vector_db` (R6).
Yield a namespace exposing `docs`, `vdb`, `emb`.
**Done when:** a smoke test using the fixture can ingest one JSON via `adicionar_documento_incrementalmente` and `agent_workflow.invoke` returns without `AttributeError`/cross-store miss.
**Gate/test notes:** first task that touches real Chroma → integration-marked; local Windows may skip (hnswlib/SAC).

## T-04 — AC-1: happy retrieve E2E  ⭐
**Requirements:** CHAR-01
**Entry point driven:** write `adicionar_documento_incrementalmente`; read `agent_workflow.invoke(AgentState(query=...))`
**Boundaries mocked:** `agent._llm`, shared embedding stub (both sides), path bindings
**Depends on:** T-03
**Description:** Ingest a JSON doc into the fresh tmp store, query with text matching the doc. Assert `retrieved_info` (the story's "sources") is non-empty, `ranked_response` present/non-empty, `confidence_score > 0`, route was `retrieve`. Validate expected AgentState keys present.
**Done when:** `teste_1_*` passes on CI Linux; real Chroma/SemanticChunker/JSONLoader exercised; no forbidden patch.

## T-05 — AC-2: empty-DB fallback to `gerar`
**Requirements:** CHAR-02
**Entry point driven:** `agent_workflow.invoke(AgentState(query=...))` (no ingestion)
**Boundaries mocked:** `agent._llm`, embedding stub, path bindings
**Depends on:** T-03
**Description:** With an empty collection, invoke a non-web query. Assert `retrieve_info` fell back: `next_step == "gerar"`, `retrieved_info == []`, a `ranked_response` is produced, no exception raised. (Exercises the empty-retrieval edge required by CLAUDE.md.)
**Done when:** `teste_2_*` passes; fallback observed via public state, not by patching `_get_retriever`.

## T-06 — AC-3: web route reaches END
**Requirements:** CHAR-03
**Entry point driven:** `agent_workflow.invoke(AgentState(query=<web-keyword query>))`
**Boundaries mocked:** `agent._llm`, embedding stub, path bindings, `@patch("agenticlog.agent.search")`
**Depends on:** T-03
**Description:** Use a query containing a `ROUTING_KEYWORDS_WEB` term so `passo_decisao_agente` routes `usar_web`. Mock `agent.search` (`mock.run.return_value=...`). Assert the web branch ran (`ranked_response` set to the web-derived answer, `confidence_score == 0.0`) and the graph reached END (invoke returns a terminal state).
**Done when:** `teste_3_*` passes; `search.run` asserted called; no DuckDuckGo network call.

## T-07 — AC-4: incremental into populated DB is queryable
**Requirements:** CHAR-04
**Entry point driven:** seed `adicionar_documento_incrementalmente`; add NEW via `adicionar_documento_incrementalmente` and/or `adicionar_pdf_incrementalmente` (same test); read `agent_workflow.invoke`
**Boundaries mocked:** `agent._llm`, shared embedding stub, path bindings; `invalidar_vector_db` runs for real (R6)
**Depends on:** T-03, T-04
**Description:** Populate the store, then within the SAME test add a NEW document and query for content unique to it. Assert the new doc is retrievable (`retrieved_info` contains its content). Covers the gap requiring a DB populated within the same test (locked decision 2). If exercising the PDF path, supply minimal valid PDF bytes through `adicionar_pdf_incrementalmente`.
**Done when:** `teste_4_*` passes; the new doc surfaces through the agent after `invalidar_vector_db` clears the cache.

## T-08 — AC-5: atomic upsert rollback at the embedding boundary
**Requirements:** CHAR-05
**Entry point driven:** seed `adicionar_documento_incrementalmente`; re-upsert same filename with different content
**Boundaries mocked:** `rag._rag_embedding_model` whose `embed_documents` raises (locked decision 1) — NOT read-only dirs, NOT patching `Chroma`
**Depends on:** T-03, T-04
**Description:** Ingest doc A successfully. Capture disk-file bytes and Chroma collection ids/count. Re-upsert the same filename with changed content while `rag._rag_embedding_model.embed_documents` raises (swap to `_stub_embedding_que_falha()` for this call). Expect the raise to propagate; assert the disk file is byte-identical to the pre-upsert state AND the Chroma collection ids/count are unchanged (rollback per PRs #42/#43, `rag.py` L479-528).
**Done when:** `teste_5_*` passes; failure trigger is the embedding boundary only; disk + collection invariants both asserted.

## T-09 — Verification, quality-gate reuse, and zero-production-change proof
**Requirements:** CHAR-09, CHAR-10
**Depends on:** T-04..T-08
**Description:**
- Run the full net: `pytest -m integration tests/test_rag_caracterizacao.py -v` (CI Linux authoritative).
- Confirm `git diff --stat` is confined to `tests/test_rag_caracterizacao.py` (and at most the `pyproject.toml` markers line). No `src/`, `app.py`, `api.py` changes (CHAR-09).
- Confirm NO new retrieval-quality threshold/metric was added; `rag_eval` remains the quality authority (CHAR-10).
- Self-review against the FORBIDDEN-patch table (design §3.2): grep the test file for any forbidden target; expect zero hits.
**Done when:** all five tests green on CI Linux, diff scope verified, forbidden-patch grep is empty.
**Gate/test notes:** also run `pytest -m "not integration" -q` to confirm the new file is correctly excluded from the fast suite and breaks nothing.

---

## Dependency Graph

```
T-01 ─► T-02 ─► T-03 ─┬─► T-04 ─┬─► T-07
                      ├─► T-05  ├─► T-08
                      └─► T-06  └─► (T-04..T-08) ─► T-09
```

## Open Issues for human ruling

- **Marker registration touch:** registering `integration` in `pyproject.toml` (if not already present) is the only candidate change outside `tests/`. Confirm this is acceptable for a tests-only phase, or pre-register it separately.
- **PDF fixture for AC-4 (optional path):** if exercising `adicionar_pdf_incrementalmente`, decide whether to commit a tiny fixture PDF or generate minimal valid PDF bytes inline; the JSON path alone already satisfies AC-4.
- **Pre-existing `tests/test_rag_integration.py`:** left untouched here; its eventual replacement (it uses forbidden internal patches) is a later ADR-018 task, not this feature.
