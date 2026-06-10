# Portuguese Embedding Model — Technical Spec

**Path:** `.specs/features/portuguese-embedding-model/spec.md`
**TLC scope:** medium
**Based on story:** Como operador de logística que consulta documentos em português, quero que o sistema use um modelo de embedding otimizado para português, para que o retrieval traga os chunks mais relevantes para minha pergunta.
**Status:** Awaiting human approval

---

## Problem Statement

The system currently uses `BAAI/bge-base-en`, an English-only embedding model, to embed both the document corpus and user queries. Operators query the system in Portuguese, so embeddings of Portuguese text are produced by a model never trained on that language, degrading semantic similarity between queries and relevant chunks. Replacing the model constant with a multilingual model (`paraphrase-multilingual-mpnet-base-v2`, 768-dim, drop-in compatible) fixes this with a config-only change plus documentation updates.

## Goals

- [ ] `EMBEDDING_MODEL` in `config.py` points to `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- [ ] All three `HuggingFaceEmbeddings` call sites continue to use `EMBEDDING_MODEL` from config (verified by tests)
- [ ] After deleting `data/vectordb/` and rerunning `python -m agenticlog.rag`, the DB rebuilds successfully at 768 dimensions and the retrieval pipeline runs end-to-end for Portuguese queries
- [ ] Operator-facing docs (CLAUDE.md, LEIAME.md, brownfield specs) document the model name, the rebuild requirement, and the silent-degradation risk if the DB is not rebuilt

## Out of Scope

| Feature | Reason |
|---------|--------|
| Harmonizing `normalize_embeddings`/`device` kwargs between `cria_vectordb()` and the singleton getters | Pre-existing inconsistency, unrelated to model swap, explicitly deferred by approved story |
| `query:`/`passage:` prefix-based embedding strategies | Not used by `paraphrase-multilingual-mpnet-base-v2`; pure drop-in replacement (AC8) |
| Automated/scripted migration of `data/vectordb/` | Manual delete + rebuild is the established pattern (multi-collection-chromadb precedent) |
| Performance/quality benchmarking old vs new model | Not required for this change; technical direction pre-agreed |
| Changes to `CHUNK_SIZE`, RAG pipeline logic, or `avalia_similaridade` ranking algorithm | Out of scope per approved story |
| New tests that build a real vector DB with the new model | Only call-site/config-name assertions are in scope; real-model tests are too slow/heavy |

---

## User Stories

### P1: Switch to a Portuguese-capable embedding model ⭐ MVP

**User Story**: As an operator querying documents in Portuguese, I want the system to use an embedding model optimized for Portuguese (and other languages), so that retrieval surfaces the most relevant chunks for my question regardless of corpus size or complexity.

**Why P1**: This is the entire feature — without it, the rest of the spec has no purpose. It is also the smallest possible change (one config constant) with the highest semantic-quality impact.

**Acceptance Criteria**:

1. **AC1** — WHEN `src/agenticlog/config.py` is inspected THEN `EMBEDDING_MODEL` SHALL equal `"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"`.
2. **AC2** — WHEN `rag.py::_get_rag_embedding_model`, `rag.py::cria_vectordb`, and `agent.py::_get_embedding_model` instantiate `HuggingFaceEmbeddings` THEN each SHALL pass `model_name=config.EMBEDDING_MODEL` (no signature change — this already happens via the shared import; this AC is about adding tests that lock the behavior).
3. **AC3** — WHEN an operator deletes `data/vectordb/` and runs `python -m agenticlog.rag` THEN the system SHALL rebuild the ChromaDB collection(s) using the new model, producing 768-dimensional embeddings, with no errors.
4. **AC4** — WHEN a Portuguese query is submitted after the rebuild THEN the retrieval → generation → similarity-evaluation → ranking pipeline SHALL run end-to-end without dimension or runtime errors.
5. **AC5** — WHEN an operator does **not** delete/rebuild an existing `data/vectordb/` that was built with `BAAI/bge-base-en` AND then queries the system THEN the system SHALL NOT raise an exception (dimensions match at 768), but results SHALL be degraded/unreliable due to incompatible embedding spaces — this risk SHALL be documented in CLAUDE.md/LEIAME.md.
6. **AC6** — WHEN an operator reads CLAUDE.md THEN it SHALL document the upgrade procedure: stop the app → delete `data/vectordb/` → rerun `python -m agenticlog.rag` → resume queries.
7. **AC7** — WHEN existing tests that mock embeddings as `[0.1] * 768` (`tests/test_agentic_rag.py`, `tests/acceptance/test_agent_workflow_integration.py`) are run after this change THEN they SHALL continue to pass unmodified (dimension is unchanged at 768).
8. **AC8** — WHEN the new model is wired in THEN the system SHALL NOT add any `query:`/`passage:` prefix handling to embedding inputs — this is a pure drop-in model-name replacement.
9. **AC9** — WHEN the diff for this feature is reviewed THEN it SHALL be config-only for behavior (i.e., `model_kwargs`/`encode_kwargs` such as `normalize_embeddings` and `device` in `rag.py`/`agent.py` SHALL remain untouched, including the existing inconsistency between `cria_vectordb()` and the singleton getters).

**Independent Test**: Run `pytest tests/test_rag.py tests/test_agent.py -v` — new assertions confirm `model_name=config.EMBEDDING_MODEL` (which equals the multilingual model) is passed at all three call sites, and existing `[0.1]*768`-based tests still pass. Then manually: delete `data/vectordb/`, run `python -m agenticlog.rag`, run `streamlit run app.py`, submit a Portuguese query, confirm a ranked response with retrieved documents and no errors.

---

## Edge Cases

- WHEN this is the first run after the upgrade (no local HuggingFace cache for the new model) THEN the system SHALL download the model (~1.0–1.1 GB, vs ~440 MB for the previous model) — no caching/pre-fetch mechanism is added beyond what already exists; this SHALL be noted in CLAUDE.md as a longer first-run setup time.
- WHEN an operator runs the app against a stale `data/vectordb/` (built with the old model) without rebuilding THEN the system SHALL NOT crash, but SHALL silently return degraded similarity scores and ranked responses, with no UI-level error or warning (per AC5; UI changes to surface this are out of scope).
- WHEN existing `[0.1]*768` mocks in `tests/test_agentic_rag.py` and `tests/acceptance/test_agent_workflow_integration.py` are executed THEN they SHALL continue to pass (dimension unchanged), even though they do not verify the actual model name in use — the coverage gap for model-name verification is closed by the new assertions added under AC2/PORTPT-02.
- WHEN brownfield docs (`LEIAME.md`, `.specs/codebase/ARCHITECTURE.md`, `.specs/codebase/INTEGRATIONS.md`, `.specs/project/PROJECT.md`, `CLAUDE.md`) reference the old model name `BAAI/bge-base-en` THEN all SHALL be updated to `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` for consistency.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| PORTPT-01 | P1 (AC1) | Design | Pending |
| PORTPT-02 | P1 (AC2, AC7) | Design | Pending |
| PORTPT-03 | P1 (AC3, AC4) | Design | Pending |
| PORTPT-04 | P1 (AC5, AC6) | Design | Pending |
| PORTPT-05 | P1 (AC8, AC9) | Design | Pending |
| PORTPT-06 | P1 (edge cases — doc consistency) | Design | Pending |

**ID format:** `PORTPT-[NUMBER]` (PORTPT = Portuguese embedding feature slug prefix)

---

## Data Model Changes

No data model changes.

The vector dimension remains 768 (both `BAAI/bge-base-en` and `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` are 768-dimensional models), so ChromaDB collection schema/dimension is unaffected. However, **existing vectors in `data/vectordb/` are numerically incompatible** with the new model's embedding space — see Risks.

---

## Process / Background Flow

**Happy path (PORTPT-03 / AC3, AC4):**
1. Operator stops the running app (if any).
2. Operator deletes `data/vectordb/` (gitignored, fully regenerable directory).
3. Operator runs `python -m agenticlog.rag`.
4. `cria_vectordb()` calls `HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})` — `EMBEDDING_MODEL` now resolves to `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
5. On first run, the new model (~1.0–1.1 GB) is downloaded and cached locally (longer than previous ~440 MB download).
6. Documents from `data/documents/` are chunked and embedded with the new model; ChromaDB collection(s) are persisted to `data/vectordb/` at 768 dimensions.
7. Operator runs `streamlit run app.py` and submits a Portuguese query.
8. `agent.py::_get_embedding_model()` and `agent.py::_get_vector_db()` lazily initialize using the same `EMBEDDING_MODEL`, embedding the query in the same (now-aligned) embedding space as the rebuilt collection.
9. Retrieval → multi-response generation → `avalia_similaridade` (cosine similarity) → ranking proceeds end-to-end, returning a `ranked_response` with `confidence_score` and retrieved documents, with no dimension/runtime errors.

**Failure path — stale `data/vectordb/` not rebuilt (PORTPT-04 / AC5):**
1. Operator updates `config.py` (or pulls a commit containing the change) but does **not** delete/rebuild `data/vectordb/`.
2. `data/vectordb/` still contains 768-dim vectors computed with `BAAI/bge-base-en`.
3. Operator runs `streamlit run app.py` and submits a query.
4. `_get_embedding_model()` embeds the query using the new multilingual model (768-dim).
5. ChromaDB performs similarity search between the new-model query vector and old-model document vectors — **dimensions match (768=768) so no exception is raised**, but the cosine-similarity scores are computed across two different (incompatible) embedding spaces, producing **meaningless/degraded relevance rankings**.
6. The UI renders a response and "documentos recuperados" as normal — there is no error message, warning banner, or log entry distinguishing this from a healthy run (no UI/observability change is introduced for this in this feature; documentation is the mitigation per AC5/AC6).

---

## API Changes

No API changes. This feature does not touch `app.py` UI, LangGraph node signatures, or any HTTP/REST surface (none exists in this codebase beyond the Streamlit UI).

---

## Frontend Changes

No frontend changes. `app.py` is unaffected — it continues to call `agent_workflow.invoke(AgentState(query=query))` and render `ranked_response`, `confidence_score`, and retrieved documents exactly as before. No new UI warning is added for the stale-vectordb scenario (documented as an accepted limitation per AC5).

---

## Tests Required

**Unit / config:**
- `tests/test_rag.py` — In the existing `cria_vectordb()` tests that patch `agenticlog.rag.HuggingFaceEmbeddings` as a class mock (`test_cria_vectordb_sem_documentos_retorna_cedo`, `test_cria_vectordb_com_documentos_cria_chroma`, ~lines 217 and 240), ADD an assertion that the mock was called with `model_name=config.EMBEDDING_MODEL` (in addition to whatever `model_kwargs`/`encode_kwargs` are already asserted, if any) — `mock_emb.assert_called_with(model_name=config.EMBEDDING_MODEL, model_kwargs={"device": ANY}, encode_kwargs={"normalize_embeddings": True})` or equivalent matching the existing call shape.
- `tests/test_rag.py` — ADD a test (new or extend an existing singleton test) for `_get_rag_embedding_model()` asserting it constructs `HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)`.
- `tests/test_agent.py` — ADD a test for `_get_embedding_model()` asserting it constructs `HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)`. Mock `agenticlog.agent.HuggingFaceEmbeddings`; reset the module-level singleton between tests if one exists (follow existing patterns for `_get_llm`/`_get_vector_db` singleton tests in this file, if present).
- A direct constant assertion: `from agenticlog import config; assert config.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` — can be folded into one of the above tests or added as a small standalone test in `tests/test_rag.py` or a config test module if one exists.

**Integration / Edge case:**
- `tests/test_agentic_rag.py::teste_7_avalia_similaridade` (~line 138-149) — `[0.1]*768` mock, leave unchanged (AC7); run to confirm it still passes (dimension unchanged).
- `tests/acceptance/test_agent_workflow_integration.py` (`_fake_embedding_model` helper, ~lines 27-32) — `[0.1]*768` mocks, leave unchanged (AC7); run to confirm it still passes.
- `tests/test_rag_integration.py` — 10-dim mocks via `_get_rag_embedding_model` patch, fully insulated from the model-name change; run to confirm no regression.
- `tests/acceptance/test_multi_collection_chromadb.py` — no current references to `EMBEDDING_MODEL`/`HuggingFaceEmbeddings`/`bge-base`/`paraphrase-multilingual` found by search; builder should run this suite to double-check no incidental coupling.

**Existing tests that will break:**
- None expected. All embedding-related tests mock `HuggingFaceEmbeddings` and/or use fixed-dimension vectors (`[0.1]*768` or 10-dim), so the model name change does not alter dimensions or call signatures. The only changes to existing tests are **additive assertions** (model name verification), not behavioral rewrites.

**Manual verification (not automated, per Out of Scope — no real-model tests):**
- Delete `data/vectordb/`, run `python -m agenticlog.rag`, confirm successful rebuild and 768-dim collection.
- Run `streamlit run app.py`, submit a Portuguese query, confirm end-to-end response with no errors (AC4).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `src/agenticlog/config.py` | Modify | Change `EMBEDDING_MODEL` value from `"BAAI/bge-base-en"` to `"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` (AC1) — single source of truth, change ripples to all 3 call sites automatically |
| `tests/test_rag.py` | Modify | Add `model_name=config.EMBEDDING_MODEL` assertions to `cria_vectordb()` tests and `_get_rag_embedding_model()` test (AC2) |
| `tests/test_agent.py` | Modify | Add `model_name=config.EMBEDDING_MODEL` assertion for `_get_embedding_model()` (AC2) |
| `CLAUDE.md` | Modify | Update "Build VectorDB" section to document the rebuild-after-model-change procedure (stop app → delete `data/vectordb/` → rerun `python -m agenticlog.rag` → resume queries), note larger download size, and reference the new model name (AC6, edge cases) |
| `LEIAME.md` | Modify | Update line referencing "Modelo BAAI/bge-base-en" to the new multilingual model name; optionally note the rebuild requirement for end users (AC5, edge cases) |
| `.specs/codebase/INTEGRATIONS.md` | Modify | Update embedding model references (~lines 25-32) from `BAAI/bge-base-en` to the new model name, keep dimension (768) as-is |
| `.specs/codebase/ARCHITECTURE.md` | Modify | Update embedding model reference (~line 60) from `BAAI/bge-base-en` to the new model name |
| `.specs/project/PROJECT.md` | Modify | Update embedding model reference (~line 17) from `BAAI/bge-base-en` to the new model name |

No changes to `src/agenticlog/rag.py` or `src/agenticlog/agent.py` source code — both already import and use `EMBEDDING_MODEL` from `config.py` with no signature changes needed (AC2, AC9).

`requirements.txt` is not expected to change (sentence-transformers, langchain-huggingface, transformers, sentencepiece are already pinned at versions believed compatible with the new model) — builder should verify compatibility at build/test time and flag a version bump only if installation or model loading fails.

---

## Risks

- **Embedding-space incompatibility for stale `data/vectordb/` (AC5)**: Both `BAAI/bge-base-en` and `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` produce 768-dimensional vectors, so ChromaDB will **not** raise a dimension-mismatch error if the old DB is queried with the new model. However, the vectors live in different, numerically incompatible embedding spaces — cosine similarity scores between old document vectors and new query vectors are meaningless. This fails **silently** (no exception, no log warning, no UI indicator) and produces degraded/unreliable retrieval. Mitigation: prominent documentation in CLAUDE.md and LEIAME.md (AC6) is the only mitigation in scope; no automated detection is added.

- **Pre-existing `model_kwargs`/`encode_kwargs` inconsistency — explicitly OUT OF SCOPE, do not touch**: `cria_vectordb()` (rag.py) passes `model_kwargs={"device": device}` and `encode_kwargs={"normalize_embeddings": True}`, while `_get_rag_embedding_model()` (rag.py) and `_get_embedding_model()` (agent.py) pass neither. This means document-corpus embeddings (built via `cria_vectordb`) may be normalized/device-placed differently than query embeddings (built via the singleton getters) — a pre-existing condition unrelated to this feature. This feature SHALL NOT change, harmonize, or comment on this inconsistency in code; only the model **name** changes at all three sites.

- **First-run download size increase**: The new model is ~1.0–1.1 GB vs. ~440 MB for `BAAI/bge-base-en` — roughly 2-3x larger. First-time setup (or any environment without a warm HuggingFace cache, e.g., CI, fresh dev machines, fresh Docker images) will take longer and consume more disk/bandwidth. Mitigation: documented in CLAUDE.md as an edge case; no caching/pre-fetch mechanism is added.

- **`requirements.txt` compatibility (verify-at-build-time, not a spec blocker)**: `sentence-transformers==3.4.1`, `langchain-huggingface==0.1.2`, `transformers==4.48.3`, and `sentencepiece==0.2.0` are already pinned and are very likely compatible with `paraphrase-multilingual-mpnet-base-v2` (a standard sentence-transformers model), but this cannot be confirmed by static analysis alone. Builder should run `python -m agenticlog.rag` and the test suite after the config change; if model loading fails due to a version constraint, a minimal version bump may be required as a follow-up — this is a verification task, not a planned change.

- **No automated drift detection between `EMBEDDING_MODEL` and `data/vectordb/`**: There is no mechanism (metadata file, hash check, etc.) that records which model built an existing `data/vectordb/` and compares it to the current `config.EMBEDDING_MODEL` at startup. This is the root cause enabling the silent-degradation failure mode above. Out of scope for this feature (no automated migration tooling, per approved story); purely a documentation-based mitigation.

- **CLAUDE.md conflicts**: None identified. CLAUDE.md's "Build VectorDB" section already describes "first time or after changing documents" as a rebuild trigger; this feature extends that guidance to also cover "after changing `EMBEDDING_MODEL`," which is consistent with existing project conventions (and the multi-collection-chromadb precedent of "config constant change requires wipe+rebuild, document it, no migration tooling").

---

## Open Questions

None — technical direction (model name, dimension parity, drop-in compatibility, no prefix handling) was pre-agreed with the user before story-writer ran, per the approved story.

---

## Success Criteria

- [ ] `config.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` (AC1)
- [ ] `tests/test_rag.py` and `tests/test_agent.py` assert `model_name=config.EMBEDDING_MODEL` at all three `HuggingFaceEmbeddings` call sites and pass (AC2)
- [ ] After deleting `data/vectordb/` and running `python -m agenticlog.rag`, the rebuild completes successfully producing a 768-dim collection (AC3)
- [ ] A Portuguese query against the rebuilt DB returns a ranked response end-to-end with no dimension/runtime errors (AC4)
- [ ] Querying against a non-rebuilt (stale) `data/vectordb/` does not raise an exception (AC5)
- [ ] CLAUDE.md documents the model-change rebuild procedure (stop app → delete `data/vectordb/` → rerun `python -m agenticlog.rag` → resume queries) and the larger download size (AC6, edge case)
- [ ] `tests/test_agentic_rag.py::teste_7_avalia_similaridade` and `tests/acceptance/test_agent_workflow_integration.py` (`_fake_embedding_model`, `[0.1]*768`) pass unmodified (AC7)
- [ ] No `query:`/`passage:` prefix logic added anywhere in the embedding call paths (AC8)
- [ ] `model_kwargs`/`encode_kwargs` in `rag.py`/`agent.py` are byte-for-byte unchanged from before this feature (AC9)
- [ ] `LEIAME.md`, `.specs/codebase/INTEGRATIONS.md`, `.specs/codebase/ARCHITECTURE.md`, `.specs/project/PROJECT.md` no longer reference `BAAI/bge-base-en` and instead reference the new model name
