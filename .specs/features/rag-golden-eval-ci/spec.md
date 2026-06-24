# RAG Golden Set + CI Regression Gate — Technical Spec

**Path:** `.specs/features/rag-golden-eval-ci/spec.md`
**TLC scope:** large
**Based on story:** Golden set curado (`evals/rag_golden.json`) + job de CI que roda o harness de avaliação RAG em cada PR para detectar regressões de qualidade antes da main.
**Status:** Awaiting human approval

---

## Problem Statement

The RAG quality of AgenticLog is measured only manually via the `rag-pipeline-audit` skill harness, which today (a) requires LMStudio for every metric, (b) falls back to silent synthetic questions when no golden set exists, and (c) always reports `"requer golden set"` for Context Recall and Answer Correctness because `_avaliar_pergunta`/`_agregar` never consume reference answers/contexts. There is no automated, repeatable regression gate: a PR can silently degrade retrieval quality and merge to `main`. We need a curated golden set anchored in the real corpus and a CI job that computes LLM-free retrieval metrics on every PR and fails the build below a quality threshold.

## Goals

- [ ] Curated `evals/rag_golden.json` (8–10 entries) with `pergunta` + `resposta_ref` + `contexto_ref`, covering ≥3 named corpus categories.
- [ ] Harness reports NUMERIC Context Recall and Answer Correctness per entry that has the required refs (no more `"requer golden set"`).
- [ ] Harness computes retrieval metrics (Hit Rate, MRR, Context Precision) WITHOUT an LLM, via embedding cosine match of retrieved chunks against `contexto_ref`.
- [ ] CI job `rag-eval` (Python 3.12 only) builds the index, runs the harness, uploads a JSON artifact, and fails the build (exit ≠ 0) when Hit Rate < threshold or MRR < threshold.
- [ ] Gate applies ONLY to embedding-only retrieval metrics; LLM-judge metrics are skipped (exit 0) when LMStudio is absent and never block CI.
- [ ] Thresholds declared as adjustable constants in `config.py`.

## Out of Scope

| Feature | Reason |
|---------|--------|
| LLM-judge metrics as a gate (Faithfulness / Answer Relevancy / Context Utilization) | Report-only; LMStudio unavailable in CI. Per approved story. |
| Fix of `AgentState.ranked_response` str-vs-dict typing | Only explicit coercion inside harness; no production type change. |
| P2/P3 retrieval improvements (re-ranking, hybrid, query transformation) | Future features. |
| Automatic golden-set generation | Golden is curated by hand from the real corpus. |
| Removing the synthetic baseline mode | Synthetic mode stays for the audit skill; CI never uses it. |
| Adding the eval job to the 3.11+3.12 matrix | Runs on 3.12 only (RESOLVED-04). |

---

## User Stories

### P1: Curated golden set with reference-based metrics ⭐ MVP
**User Story**: As a logistics software engineer, I want a curated golden set with reference answers and contexts so that the harness reports numeric Context Recall and Answer Correctness per entry.
**Why P1**: Without references the harness cannot measure recall/correctness; this is the foundation for the gate.

**Acceptance Criteria**:
1. WHEN a golden entry has `pergunta` + `resposta_ref` + `contexto_ref` THEN the harness SHALL report NUMERIC values (not `"requer golden set"`) for Context Recall and Answer Correctness for that entry.
2. WHEN a golden entry has `resposta_ref` but no `contexto_ref` THEN the harness SHALL compute Answer Correctness numerically AND report Context Recall as the literal `"contexto_ref ausente"` for that entry.
3. WHEN an entry has no `pergunta` THEN the harness SHALL reject it with a warning and exclude it (never pass it through silently).
4. WHEN the golden set is loaded THEN it SHALL cover ≥3 categories of the corpus: controle de estoque, processamento de pedidos, definição de operador logístico.

**Independent Test**: Load a fixture golden with mixed entries; assert per-entry recall/correctness numeric vs `"contexto_ref ausente"`, and that an entry without `pergunta` is dropped with a logged warning.

---

### P1: LLM-free retrieval metrics ⭐ MVP
**User Story**: As an engineer, I want Hit Rate, MRR and Context Precision computed without an LLM so that CI can gate retrieval quality even when LMStudio is offline.
**Why P1**: CI has no LLM; the gated metrics must be embedding-only.

**Acceptance Criteria**:
1. WHEN the harness runs without an LLM client THEN it SHALL still compute Hit Rate, MRR and Context Precision via embedding cosine match of retrieved chunks against `contexto_ref`, AND mark judge metrics as `{"status": "skipped", ...}` with exit 0 for the judge portion.
2. WHEN a retrieved chunk's max cosine similarity to any `contexto_ref` segment is ≥ the match threshold THEN that chunk SHALL be counted as a "hit" (relevant) for that entry.
3. WHEN no `contexto_ref` is present for an entry THEN that entry SHALL be excluded from retrieval-metric aggregation (cannot judge a hit without a reference context).

**Independent Test**: With a stub retriever returning known chunks and a stub embedding model returning fixed vectors, assert Hit Rate / MRR / Context Precision match hand-computed values for a threshold above and below the match.

---

### P1: CI regression gate ⭐ MVP
**User Story**: As an engineer, I want a CI job that runs the harness against the golden set on every PR and fails below threshold so that quality regressions are caught before main.
**Why P1**: This is the automation that delivers the user-visible value.

**Acceptance Criteria**:
1. WHEN a PR is opened THEN a `rag-eval` job (Python 3.12) SHALL checkout, install deps, build the index (`python -m agenticlog.rag --rebuild`), run the harness against `evals/rag_golden.json`, and upload the results JSON as a GitHub Actions artifact.
2. WHEN aggregate Hit Rate < `RAG_EVAL_MIN_HIT_RATE` (0.7) OR MRR < `RAG_EVAL_MIN_MRR` (0.6) THEN the gate step SHALL exit ≠ 0 and fail the job.
3. WHEN aggregate Hit Rate ≥ threshold AND MRR ≥ threshold THEN the gate step SHALL exit 0 even if judge metrics are skipped.
4. WHEN the golden file is absent or empty THEN the harness SHALL exit ≠ 0 with an explicit error (no silent synthetic fallback in CI).
5. WHEN the index is empty (`collection.count() == 0`) THEN the harness SHALL fail with a high-severity explicit error.

**Independent Test**: Run the gate-check function against a results JSON with metrics below threshold (assert exit ≠ 0) and above threshold (assert exit 0); assert missing/empty golden and empty index produce explicit non-zero exits.

---

## Edge Cases

- WHEN `ranked_response` is a `dict` `{"answer": str}` instead of `str` THEN the harness SHALL coerce it to text before evaluation (existing behavior, preserved).
- WHEN `contexto_ref` is a list of strings vs a single string THEN the harness SHALL treat each element as a candidate reference segment for matching.
- WHEN `--k` exceeds `RETRIEVAL_K_TOTAL` (3) THEN retrieval returns at most `RETRIEVAL_K_TOTAL` unique chunks; metrics SHALL be computed over the actually-returned chunks (documented limit, not an error).
- WHEN the HF embedding model download fails in CI THEN the harness SHALL fail with an explicit error (retrieval metrics cannot be computed without embeddings).
- WHEN LMStudio IS reachable locally THEN judge metrics SHALL be computed and reported (report-only), and Answer Correctness MAY use the judge; the gate still applies only to embedding-only retrieval metrics.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
|----------------|-------|-------|--------|
| GOLD-01 | P1 Golden | golden authoring | Pending |
| GOLD-02 | P1 Golden | harness recall/correctness | Pending |
| GOLD-03 | P1 Golden | harness — `contexto_ref ausente` | Pending |
| GOLD-04 | P1 Golden | harness validation (no `pergunta` → drop+warn) | Pending |
| GOLD-05 | P1 Golden | golden coverage ≥3 categories | Pending |
| RETR-01 | P1 Retrieval | embedding-only retrieval metrics | Pending |
| RETR-02 | P1 Retrieval | cosine match threshold → "hit" | Pending |
| RETR-03 | P1 Retrieval | exclude entries without `contexto_ref` from gated agg | Pending |
| GATE-01 | P1 CI | gate-check exit ≠ 0 below threshold | Pending |
| GATE-02 | P1 CI | thresholds as config constants | Pending |
| GATE-03 | P1 CI | missing/empty golden → explicit non-zero exit | Pending |
| GATE-04 | P1 CI | empty index → high-severity explicit fail | Pending |
| CI-01 | P1 CI | `rag-eval` job (3.12) build index + run + upload artifact | Pending |
| CI-02 | P1 CI | judge-skipped never blocks CI | Pending |
| HARN-01 | all | move/copy harness to `scripts/rag_eval.py` (out of `--cov=agenticlog`) | Pending |

**ID format:** `[FEAT]-[NUMBER]` — prefixes: `GOLD` (golden set), `RETR` (retrieval metrics), `GATE` (gate logic/config), `CI` (workflow), `HARN` (harness relocation).

---

## Data Model Changes

No database/schema changes. One new versioned data artifact:

**`evals/rag_golden.json`** — array of golden entries:

```json
[
  {
    "pergunta": "Qual a finalidade do controle de estoques?",
    "resposta_ref": "...resposta de referência curta e factual...",
    "contexto_ref": "...trecho do corpus que contém a resposta...",
    "categoria": "controle de estoque"
  }
]
```

Field contract:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `pergunta` | `str` (non-empty) | Yes | Missing/empty → entry rejected with warning (GOLD-04). |
| `resposta_ref` | `str` (non-empty) | Yes for Answer Correctness | If absent, Answer Correctness reported as `"resposta_ref ausente"`. |
| `contexto_ref` | `str` OR `list[str]` (non-empty) | Yes for Context Recall + retrieval gate | If absent, Context Recall = `"contexto_ref ausente"` and entry excluded from gated retrieval aggregation. |
| `categoria` | `str` | Optional | Documents corpus category for the ≥3-category coverage check; informational. |

No `AgentState`/Pydantic production model changes.

---

## Process / Background Flow

**Happy path (CI, no LLM):**
1. `rag-eval` job checks out, installs deps (Python 3.12), caches/downloads HF embedding model.
2. `python -m agenticlog.rag --rebuild` builds `data/vectordb/` from `data/documents/`.
3. `python scripts/rag_eval.py --golden evals/rag_golden.json --out rag_eval_results.json --k N` runs:
   - Loads + validates golden (drops entries without `pergunta`, warns).
   - For each valid entry: retrieves top-k chunks, embeds chunks + `contexto_ref`, computes per-entry hit/MRR/precision/recall; computes Answer Correctness via embedding cosine `resposta` ↔ `resposta_ref`.
   - Judge client unavailable → judge metrics block = `{"status": "skipped", ...}`; retrieval/correctness block still populated.
   - Aggregates and writes results JSON.
4. Upload `rag_eval_results.json` as artifact.
5. Gate step reads JSON; if Hit Rate ≥ 0.7 AND MRR ≥ 0.6 → exit 0; else exit ≠ 0.

**Failure path — no LMStudio in CI:** judge metrics skipped (exit 0 for judge), embedding-only retrieval metrics still computed and gated. CI passes iff retrieval thresholds met (CI-02).

**Failure path — golden absent/empty:** harness exits ≠ 0 with explicit message; NO synthetic fallback (GATE-03).

**Failure path — empty index:** harness detects `collection.count() == 0`, exits ≠ 0 with high-severity message (GATE-04).

---

## API Changes

No HTTP API changes.

Harness CLI (`scripts/rag_eval.py`) contract changes:
- `--golden` becomes effectively required for CI usage; absent/empty → non-zero exit (CI mode). Synthetic mode is preserved only for the audit skill (when explicitly run without `--golden` outside CI).
- New optional flag `--gate` (or a separate `scripts/rag_eval_gate.py`) returning exit ≠ 0 when thresholds violated — see design.md for the chosen placement.

---

## Frontend Changes

No frontend changes.

---

## Tests Required

**Unit (dedicated test file, NOT counted under `--cov=agenticlog`):**
- Golden loader: valid entry passes; entry without `pergunta` dropped + warning logged (GOLD-04).
- Per-entry metric computation with stub embedding model + stub retriever:
  - Hit / MRR / Context Precision against threshold above and below the match (RETR-01, RETR-02).
  - Answer Correctness numeric from cosine `resposta` ↔ `resposta_ref` (GOLD-02).
  - `contexto_ref` absent → Context Recall = `"contexto_ref ausente"`, entry excluded from gated aggregation (GOLD-03, RETR-03).
- Aggregation: gated metrics ignore entries lacking `contexto_ref`.
- Gate-check: below threshold → exit ≠ 0; above → exit 0 (GATE-01); missing/empty golden → exit ≠ 0 (GATE-03); empty index → exit ≠ 0 (GATE-04).

**Integration (mock LLM at client level; mark `@pytest.mark.integration`):**
- Full harness run against a tiny fixture golden + stub/in-memory retriever; judge client mocked as unavailable → judge block skipped, retrieval block numeric, results JSON shape asserted.
- Acceptance: judge-skipped run with retrieval above threshold → overall pass (CI-02).

**Edge case:**
- `ranked_response` dict coercion preserved.
- `contexto_ref` as list vs str.

**Existing tests that will break:** none expected in `src/agenticlog` (harness lives outside the package and outside `--cov`). The `--cov-fail-under=80` gate stays on the existing `test` job; the new harness tests run in their own invocation so they do not perturb the 80% package coverage number.

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `scripts/rag_eval.py` | Add (moved/copied from `.claude/skills/rag-pipeline-audit/scripts/rag_eval.py`) | Versioned harness outside `--cov=agenticlog` (HARN-01). |
| `scripts/rag_eval.py` | Modify | Extend `_carregar_golden` validation; embedding-only retrieval metrics; `_avaliar_pergunta`/`_agregar` consume `resposta_ref`/`contexto_ref`; empty-index + empty-golden explicit failures; gate logic/flag. |
| `evals/rag_golden.json` | Add | Curated 8–10 entry golden set (GOLD-01, GOLD-05). |
| `src/agenticlog/config.py` | Modify | Add `RAG_EVAL_MIN_HIT_RATE`, `RAG_EVAL_MIN_MRR`, `RAG_EVAL_MATCH_THRESHOLD`, `RAG_EVAL_K` constants (GATE-02). |
| `.github/workflows/ci.yml` | Modify | Add `rag-eval` job (Python 3.12): build index, run harness, upload artifact, gate step (CI-01, GATE-01). |
| `tests/test_rag_eval.py` (or `scripts/tests/`) | Add | Unit + integration tests for the harness (dedicated, not under `--cov=agenticlog`). |
| `.specs/codebase/STATE.md` | (optional, post-merge) | Mark "Add GitHub Actions CI with coverage gate" todo progress. |

Note: importing `config` from the harness must NOT pull config into `--cov=agenticlog` accounting — harness tests run in a separate pytest invocation. See design.md.

---

## Risks

- **Silent-degradation / stale vectordb (CLAUDE.md HIGH):** CI MUST run `python -m agenticlog.rag --rebuild` (not incremental) so the index always matches current chunking/embedding config; an incremental build could leave stale-strategy chunks and produce misleading metrics. Mitigated by `--rebuild` in the workflow (CI-01).
- **HF model download (~1 GB) in CI:** slow first run; mitigate with `actions/cache` keyed on `EMBEDDING_MODEL`. Risk: cache miss → longer job; acceptable.
- **`RETRIEVAL_K_TOTAL` caps retrieval at 3:** `--k > 3` is effectively bounded; metrics computed over ≤3 chunks. Documented limit (Edge Cases). MRR/Hit Rate are still meaningful at k≤3 for a small corpus.
- **Coverage gate interference (CLAUDE.md `--cov-fail-under=80`):** harness lives in `scripts/` (outside `agenticlog` package) and harness tests run separately so they neither inflate nor deflate package coverage. Risk if someone adds harness to `--cov` paths — explicitly forbidden in design.
- **Embedding-only "hit" definition is heuristic:** a cosine threshold can mislabel near-misses. Mitigated by a conservative, configurable `RAG_EVAL_MATCH_THRESHOLD` and report-only judge metrics for cross-check. Threshold is intentionally loose at start (RESOLVED-01) and adjustable.
- **`ranked_response` str/dict ambiguity (out of scope to fix):** harness coerces explicitly; no production change. Risk contained.
- **Determinism:** `LLM_TEMPERATURE=0` and `NUM_CANDIDATE_RESPONSES=1` make generation deterministic; embedding-only metrics are fully deterministic — gate is stable across reruns.
- **CONCERNS.md INFO (no CI coverage gate):** this feature partially addresses the broader CI hardening direction by adding a quality gate, but does not replace the existing `--cov-fail-under=80` test job.

---

## Open Questions

None. The five prior open questions were resolved (gate thresholds, harness location, recall/correctness implementation, CI matrix scope, golden size/content) and are encoded above as RESOLVED decisions. One implementation choice (gate as harness `--gate` flag vs separate `scripts/rag_eval_gate.py`) is deferred to design.md and does not block the spec.

---

## Success Criteria

- [ ] `evals/rag_golden.json` exists with 8–10 entries covering ≥3 named categories; every entry has `pergunta` + `resposta_ref` (+ `contexto_ref` for gated ones).
- [ ] Harness reports numeric Context Recall + Answer Correctness for entries with refs; `"contexto_ref ausente"` where applicable.
- [ ] Harness computes Hit Rate / MRR / Context Precision without an LLM.
- [ ] CI `rag-eval` job runs on Python 3.12, uploads results JSON artifact, and fails when Hit Rate < 0.7 or MRR < 0.6.
- [ ] Judge-skipped (no LMStudio) never blocks CI; missing/empty golden and empty index produce explicit non-zero exits.
- [ ] Thresholds live in `config.py` as adjustable constants.
- [ ] Existing `test` (3.11+3.12, `--cov-fail-under=80`) and `lint` jobs remain green and unaffected.
