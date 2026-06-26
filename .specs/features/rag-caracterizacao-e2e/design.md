# RAG E2E Characterization Test Net — Design

**Path:** `.specs/features/rag-caracterizacao-e2e/design.md`
**Spec:** `.specs/features/rag-caracterizacao-e2e/spec.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## 1. Architecture Overview

A single test module builds a **behavioral oracle** around the RAG pipeline by driving public entry points and observing `AgentState`, while substituting only boundary seams. The net never imports or patches the internal functions the refactor will rename.

```
tests/test_rag_caracterizacao.py
  fixtures (function-scoped)
    rag_caracterizacao_env(tmp_path)
      ├─ redirect rag.DIR_DOCUMENTS  → tmp/documents      (module-level binding)
      ├─ redirect rag.DIR_VECTORDB   → tmp/vectordb        (module-level binding)
      ├─ redirect agent.DIR_VECTORDB → tmp/vectordb  ◄─ same store the agent READS
      ├─ set rag._rag_embedding_model   = shared deterministic embedding stub
      ├─ set agent._embedding_model     = SAME shared embedding stub
      ├─ set agent._llm                 = Runnable-compatible LLM stub (deterministic text)
      └─ reset agent._vector_dbs = {}   (drop stale cached Chroma handles)
  │
  ├─ teste_1 (AC-1)  ingest JSON → agent_workflow.invoke(query)         → assert retrieve path
  ├─ teste_2 (AC-2)  empty store → agent_workflow.invoke(query)         → assert gerar fallback
  ├─ teste_3 (AC-3)  @patch agent.search → agent_workflow.invoke(web q) → assert END
  ├─ teste_4 (AC-4)  ingest A → ingest NEW B (same test) → invoke(q→B)  → assert B retrievable
  └─ teste_5 (AC-5)  ingest A → re-upsert A with embed_documents raising → assert disk+Chroma unchanged
        │
        ▼  (every test reaches the REAL stack below — only the seams above are stubbed)
   real Chroma  ·  real SemanticChunker  ·  real JSONLoader  ·  real LangGraph workflow
```

**Design principle:** mock at the *process boundary* (network LLM, network web search, on-disk paths, the HF embedding download), execute everything *inside* the boundary for real. Boundary seams are module-level names that the ADR-018 refactor preserves as the package's public edges; internal helpers are not.

---

## 2. Why the existing tests don't qualify (motivation, not a target to copy)

`tests/test_rag_integration.py` pins behavior with `patch("agenticlog.rag._get_rag_embedding_model")`, `patch("agenticlog.rag.Chroma")`, `patch("agenticlog.rag.JSONLoader")`, `patch("agenticlog.rag.SemanticChunker")`, and `patch("agenticlog.agent._listar_colecoes" / "._get_vector_db")`. Every one of those symbols is an internal code-path the refactor will rename or relocate, so those tests will throw `AttributeError`/`ModuleNotFoundError` on rename — false failures, no protection. This net deliberately avoids all of them.

---

## 3. Mock-Boundary Contract (CHAR-06 — the core constraint)

> This table is normative. A test that patches anything in the FORBIDDEN column fails review even if green.

### 3.1 ALLOWED to mock/stub — survive the refactor (module-level boundary seams)

| Seam | How | Used by |
|------|-----|---------|
| `agent._llm` | **set the module variable** to a Runnable-compatible LLM stub | all read paths (chains) |
| `agent._embedding_model` | **set the module variable** to the shared embedding stub | retrieval + similarity ranking |
| `rag._rag_embedding_model` | **set the module variable**; for AC-5 give it an `embed_documents` that raises | ingestion + chunking |
| `agenticlog.agent.search` | `@patch("agenticlog.agent.search")` (DuckDuckGo wrapper) | AC-3 web branch |
| `rag.DIR_DOCUMENTS` | redirect module-level path binding → `tmp_path/documents` | ingestion file writes |
| `rag.DIR_VECTORDB` | redirect module-level path binding → `tmp_path/vectordb` | ingestion Chroma persist |
| `agent.DIR_VECTORDB` | redirect module-level path binding → SAME `tmp_path/vectordb` | **agent read store + `_listar_colecoes`** |
| `agent._vector_dbs` | reset to `{}` in the fixture | drop stale cached handles |

Setting a module variable (`rag._rag_embedding_model = stub`) is preferred over `patch(...)` for the singletons, because the variable name is the boundary; the *getter* (`_get_rag_embedding_model`) is internal. The getters short-circuit when the module var is already set (see `agent.py` L128-129, `rag.py` L73-74), so setting the var is sufficient and never touches the getter.

### 3.2 FORBIDDEN to patch — internal code-paths the refactor renames/moves

| Forbidden target | Why |
|------------------|-----|
| `rag._get_rag_embedding_model`, `agent._get_rag_embedding_model` | internal getter; renamed by refactor |
| `agent._get_embedding_model`, `agent._get_llm` | internal getter; renamed by refactor |
| `agent._get_retriever`, `agent._get_vector_db`, `agent._listar_colecoes` | internal retrieval plumbing; moved by refactor |
| `Chroma`, `JSONLoader`, `SemanticChunker` (in either module) | the real components the net must exercise |
| `rag._resetar_colecao`, `rag._hash_arquivo` / hashing, `rag.cria_vectordb`, `rag.adicionar_*` | exercised, never mocked (they ARE the behavior under test) |
| `rag.invalidar_vector_db` / `agent.invalidar_vector_db` | must run so AC-4 sees the new doc (R6) |

### 3.3 Stub contracts (builder-facing detail)

- **Embedding stub (shared, R4):** one object reused for both `rag._rag_embedding_model` and `agent._embedding_model`. Must expose `embed_documents(texts) -> list[list[float]]` and `embed_query(text) -> list[float]`, deterministic, fixed dimensionality (e.g. constant `[0.1]*N` as in the existing integration helper `_mock_emb`). Constant vectors are acceptable: Chroma still returns the top-k nearest, and `SemanticChunker` collapses zero-distance breakpoints into a single valid chunk. The SAME stub on both sides keeps write/read in one vector space.
- **LLM stub (R3):** must compose in `prompt | _get_llm() | StrOutputParser()` and yield a string. Use a LangChain fake chat model (`langchain_core.language_models.fake_chat_models.FakeListChatModel`) or a `RunnableLambda` returning a fixed string / `AIMessage`. A bare `MagicMock` is discouraged — verify it composes before relying on it.
- **AC-5 embedding stub:** same shape, but `embed_documents.side_effect = RuntimeError("embedding boundary failure")`. `embed_query` may stay benign. The failure must surface inside the guarded ingestion block (`rag.py` L479-528) so `_reverter_disco` runs.
- **`agent.search` mock (AC-3):** `mock.run.return_value = "resultado web simulado"` (the node calls `search.run(state.query)`, `agent.py` L325).

---

## 4. Fixture Design (CHAR-08, locked decision 2 — function-scoped)

Function scope is mandatory because AC-4 must populate the DB **and** query it within the same test; a session/module fixture would share the store across tests and break isolation (R2). Proposed fixture surface:

```
@pytest.fixture
def rag_caracterizacao_env(tmp_path, monkeypatch):
    # 1. paths
    docs = tmp_path / "documents"; docs.mkdir()
    vdb  = tmp_path / "vectordb";  vdb.mkdir()
    monkeypatch.setattr("agenticlog.rag.DIR_DOCUMENTS", docs)
    monkeypatch.setattr("agenticlog.rag.DIR_VECTORDB",  vdb)
    monkeypatch.setattr("agenticlog.agent.DIR_VECTORDB", vdb)   # R1 — agent reads here
    # 2. seams
    emb = _stub_embedding()                       # shared (R4)
    monkeypatch.setattr("agenticlog.rag._rag_embedding_model", emb)
    monkeypatch.setattr("agenticlog.agent._embedding_model",   emb)
    monkeypatch.setattr("agenticlog.agent._llm", _stub_llm())  # R3
    # 3. cache reset (R2) — restored automatically by monkeypatch teardown
    monkeypatch.setattr("agenticlog.agent._vector_dbs", {})
    yield SimpleNamespace(docs=docs, vdb=vdb, emb=emb)
```

- Use `monkeypatch.setattr` (auto-reverts after each test) rather than manual `setattr`, so module globals are restored and tests stay order-independent.
- `_vector_dbs` is reset to a fresh dict; `invalidar_vector_db()` (run for real during AC-4 ingestion) clears it again mid-test so the agent re-reads the updated collection.
- AC-5 overrides only `rag._rag_embedding_model` with the raising variant (per-test, inside the test body or a parametrized fixture flavor), leaving the rest of the env intact.

---

## 5. Components & Interfaces (test helpers, all in the one file)

| Helper | Signature (intent) | Notes |
|--------|--------------------|-------|
| `_stub_embedding()` | `() -> object` with `embed_documents`, `embed_query` | deterministic, fixed dim; shared across sides |
| `_stub_embedding_que_falha()` | `() -> object` whose `embed_documents` raises | AC-5 only |
| `_stub_llm()` | `() -> Runnable` composing in `prompt | llm | parser` | `FakeListChatModel` or `RunnableLambda` |
| `_ingerir_json(env, nome, dados)` | helper calling `adicionar_documento_incrementalmente` | public entry point only |
| `_invoke(query)` | `agent_workflow.invoke(AgentState(query=query))` | the single read entry point |
| `_colecao_count(vdb, nome)` | reads Chroma collection count via public Chroma ctor | AC-5 invariant check (reading a real Chroma at the tmp dir is allowed — it is not patching the production seam) |

Immutability/style: helpers return new objects, no mutation of shared fixtures; file kept within the 200-400 line target (single flat file, locked decision 3). If it approaches 400 lines, prefer trimming duplication via the helpers above before considering a split — but the locked decision is one file, so a split needs human sign-off.

---

## 6. Public Entry Points Driven (per AC)

| AC | Write entry point | Read entry point | Boundaries mocked |
|----|-------------------|------------------|-------------------|
| AC-1 | `adicionar_documento_incrementalmente` | `agent_workflow.invoke` | `_llm`, both `_embedding_model`/`_rag_embedding_model`, paths |
| AC-2 | none (empty store) | `agent_workflow.invoke` | `_llm`, embeddings, paths |
| AC-3 | none | `agent_workflow.invoke` | `_llm`, embeddings, paths, `@patch agent.search` |
| AC-4 | `adicionar_documento_incrementalmente` (seed) + `adicionar_documento_incrementalmente`/`adicionar_pdf_incrementalmente` (new, same test) | `agent_workflow.invoke` | `_llm`, embeddings, paths; `invalidar_vector_db` runs for real |
| AC-5 | `adicionar_documento_incrementalmente` (seed) + re-upsert with raising embed | direct Chroma read for invariant | raising `rag._rag_embedding_model`, paths |

---

## 7. The `@pytest.mark.integration` marker rationale (CHAR-07)

Every test reaches the real `Chroma` (hnswlib) on disk. On local Windows the unsigned `hnswlib` DLL is blocked by Smart App Control, so these tests skip/fail locally for an *environmental* reason (MEMORY: feedback-hnswlib-SAC-block — ~15 chroma tests already affected). Marking the whole module `@pytest.mark.integration`:

- lets contributors run the fast unmarked suite locally (`pytest -m "not integration"`) without the DLL block,
- designates **CI Linux** as the authoritative gate for this net,
- matches the existing convention (`tests/test_rag_integration.py` uses the same marker).

Apply at module level: `pytestmark = pytest.mark.integration` (or decorate the class). Verify `integration` is a registered marker in `pyproject.toml` `[tool.pytest.ini_options].markers`; register it if absent (flag as the one possible non-`tests/` touch — see spec Files table).

---

## 8. Data Models

No new data models. The net observes the existing `AgentState` (Pydantic, `agent.py` L275-288). `agent_workflow.invoke` returns a dict-like state; AC oracles read `retrieved_info`, `ranked_response`, `confidence_score`, `next_step`. Recall: **story `sources` = `retrieved_info`** (no `sources` field exists).

---

## 9. Reuse from Codebase

- `_mock_emb` pattern from `tests/test_rag_integration.py` L25-29 — reuse the *shape* (`embed_documents`/`embed_query`, constant vectors), NOT the forbidden `patch("...rag._get_rag_embedding_model")` wiring around it.
- `tmp_path` documents/vectordb layout — same as the existing integration tests (L38-41).
- `teste_N_` naming, AgentState-key validation, empty-retrieval edge — CLAUDE.md / `.specs/codebase/TESTING.md` conventions.
- `rag_eval` golden gate (`scripts/rag_eval.py`, `evals/rag_golden.json`, feature `rag-golden-eval-ci`) — remains the retrieval-quality authority; this net does not duplicate Hit Rate / MRR.

---

## 10. CONCERNS.md Mitigations

- **MEDIUM — Incomplete test coverage / `agent_workflow.invoke()` end-to-end absent:** this net is the first E2E exercise of the compiled workflow through the public entry point, covering the retrieve, gerar-fallback and usar_web routes plus the empty-retrieval edge explicitly called out in CONCERNS.
- **HIGH — LMStudio SPOF / startup:** the net is fully offline (LLM, embeddings, web search stubbed at the boundary), so it runs deterministically in CI without LMStudio — it characterizes behavior independent of the SPOF.
- **Silent-degradation (CLAUDE.md):** by sharing one deterministic embedding stub across ingestion and query, the net keeps write/read in a single vector space, side-stepping the cross-space degradation risk for the duration of the test.

---

## 11. Risks & Open Issues for tasks

- R1 cross-store binding (agent.DIR_VECTORDB) — handled in §4 fixture; verify in T-04/T-07.
- R3 LLM-stub composability — validate the chosen fake composes before writing all tests (T-01 spike).
- R8 `integration` marker registration — confirm in `pyproject.toml` (T-01); if a config edit is deemed out-of-scope for a tests-only phase, flag to human.
- Locked-decision guardrails (T-08 must use the embedding-boundary trigger, not Chroma/read-only-dir) — encode as a review checklist item.
