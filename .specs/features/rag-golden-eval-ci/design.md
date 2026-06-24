# RAG Golden Set + CI Regression Gate — Design

**Path:** `.specs/features/rag-golden-eval-ci/design.md`
**Spec:** `.specs/features/rag-golden-eval-ci/spec.md`
**TLC scope:** large
**Status:** Awaiting human approval

---

## 1. Architecture Overview

Three cooperating pieces, all anchored in the existing project pipeline:

```
.github/workflows/ci.yml ── rag-eval job (Python 3.12)
   │  checkout → install deps → cache/download HF model
   │  → python -m agenticlog.rag --rebuild        (build data/vectordb/)
   │  → python scripts/rag_eval.py --golden evals/rag_golden.json
   │        --out rag_eval_results.json --k 3
   │  → upload rag_eval_results.json (artifact)
   │  → gate step: read JSON, exit≠0 if HitRate<0.7 or MRR<0.6
   ▼
scripts/rag_eval.py  (relocated + extended harness)
   ├─ _bootstrap()         imports agenticlog.{config,agent,rag}  (unchanged)
   ├─ _carregar_golden()   load + validate (drop no-`pergunta`, warn)   [GOLD-04]
   ├─ _metrica_retrieval() embedding-only Hit/MRR/Precision/Recall      [RETR-*]
   ├─ _avaliar_pergunta()  retrieval metrics + Answer Correctness +     [GOLD-02/03]
   │                       optional judge block (skipped w/o LMStudio)  [CI-02]
   ├─ _agregar()           aggregate; gated metrics ignore no-ctx entries
   ├─ _checar_indice()     collection.count()==0 → explicit fail        [GATE-04]
   └─ portao()/--gate      exit≠0 below threshold                       [GATE-01]
   ▼
evals/rag_golden.json   (curated, versioned)                           [GOLD-01/05]
src/agenticlog/config.py  RAG_EVAL_* constants                         [GATE-02]
```

Design principle: reuse the real pipeline (`_get_retriever`, `_get_rag_embedding_model`, `agent_workflow`) so metrics reflect production behavior. The harness adds NO production code paths; all new logic is in `scripts/` and `evals/` plus pure config constants.

---

## 2. Harness Relocation (HARN-01)

- Copy `.claude/skills/rag-pipeline-audit/scripts/rag_eval.py` → `scripts/rag_eval.py` (versioned in repo).
- The skill copy MAY remain (untracked, audit-only). The versioned `scripts/rag_eval.py` is the single source consumed by CI; keep the skill copy in sync manually or document it as the audit mirror. (Recommendation: make the skill reference `scripts/rag_eval.py` to avoid drift; not required by this feature.)
- `scripts/` is OUTSIDE the `agenticlog` package and OUTSIDE `--cov=agenticlog`. The existing `test` job's coverage number is unaffected.
- Harness tests live in `tests/test_rag_eval.py` but run in a SEPARATE pytest invocation in the `rag-eval` job (or are excluded from the coverage-gated invocation via marker/path). They MUST NOT be included in `pytest --cov=agenticlog --cov-fail-under=80`.

Constraint: `_bootstrap()` already locates `src/` by walking parents for `src/agenticlog/`; from `scripts/rag_eval.py` at repo root this still resolves. No change to `_bootstrap` needed.

---

## 3. Embedding-Only Retrieval Metrics (RETR-01, RETR-02, RETR-03)

### 3.1 Match function

For a retrieved chunk `c` and an entry's `contexto_ref` (normalized to `list[str]` `refs`):

```
sim(c, refs) = max over r in refs of cosine(embed(c), embed(r))
chunk_is_hit(c) = sim(c, refs) >= RAG_EVAL_MATCH_THRESHOLD
```

- `embed` uses `_get_rag_embedding_model().embed_query(text)` — same model/space as production (768-dim, normalized).
- `cosine` reuses the existing `_cosine` helper in the harness.
- `RAG_EVAL_MATCH_THRESHOLD` is a config constant (default proposed 0.6; loose at start, adjustable — RESOLVED-01). It controls when a retrieved chunk "matches" the reference context.

### 3.2 Per-entry retrieval metrics (no LLM)

Given retrieved chunks `c_1..c_n` (n ≤ `RETRIEVAL_K_TOTAL` = 3) in rank order:

- `rels[i] = 1 if chunk_is_hit(c_i) else 0`
- **Context Precision** = `sum(rels) / n` (fraction of retrieved chunks that match the reference).
- **Hit Rate (per entry)** = `1.0 if any(rels) else 0.0`.
- **MRR (per entry)** = `1/(rank of first hit)`, else `0.0`.
- **Context Recall (per entry)** = fraction of reference segments covered by at least one retrieved chunk:
  `recall = (# refs r with max_c cosine(embed(c), embed(r)) >= RAG_EVAL_MATCH_THRESHOLD) / len(refs)`.
  When `contexto_ref` is a single string, `len(refs) == 1`, so recall ∈ {0.0, 1.0}.

This replaces the LLM judge for `context_precision`/`hit`/`mrr` (today computed via `_juiz_json`). The judge-based precision becomes an OPTIONAL report-only cross-check when LMStudio is present.

### 3.3 Exclusion rule (RETR-03)

Entries without `contexto_ref` are EXCLUDED from gated aggregation (Hit Rate, MRR, Context Precision, Context Recall) — a hit cannot be judged without a reference. They still appear in `detalhe` with `context_recall = "contexto_ref ausente"`.

---

## 4. Answer Correctness & Context Recall consumption (GOLD-02, GOLD-03)

Extend `_avaliar_pergunta` to consume refs from the golden item:

- **Answer Correctness** (report-only, numeric):
  - Default (CI / no LLM): `cosine(embed(resposta), embed(resposta_ref))` where `resposta` is the coerced `ranked_response`.
  - When LMStudio present: MAY ALSO compute a judge-based correctness (report-only). The embedding-based value is always present.
  - If `resposta_ref` absent → `answer_correctness = "resposta_ref ausente"`.
- **Context Recall** as defined in §3.2; if `contexto_ref` absent → `"contexto_ref ausente"`.

`_agregar` changes:
- Replace the hardcoded `"requer golden set"` for `context_recall` and `answer_correctness` with the mean of numeric per-entry values (ignoring string sentinels).
- Add embedding-only `hit_rate`, `mrr`, `context_precision` computed from the new per-entry numeric fields, ignoring excluded entries.
- Judge metrics (`faithfulness`, `context_utilization`, `answer_relevancy`) remain, but their block is `{"status": "skipped", "motivo": ...}` when no LLM (CI-02).

Resulting `agregados` shape (CI, no LLM):

```json
{
  "retrieval": {
    "hit_rate": 0.8,
    "mrr": 0.71,
    "context_precision": 0.66,
    "context_recall": 0.75,
    "n_entradas_gated": 8
  },
  "answer_correctness": 0.62,
  "judge": {"status": "skipped", "motivo": "LMStudio inacessível em ..."}
}
```

When LMStudio present, `judge` carries numeric `faithfulness`/`context_utilization`/`answer_relevancy` (report-only, not gated).

---

## 5. Failure Modes (GATE-03, GATE-04)

`main()` ordering changes for CI safety:

1. `_bootstrap()` fails (no `src/agenticlog`) → `{"status": "skipped", ...}` exit 0 (audit-skill behavior preserved for that case only).
2. **Golden absent/empty (CI):** when `--golden` is provided but the file is missing or yields zero valid entries → write `{"status": "error", "motivo": "golden ausente/vazio"}` and return non-zero exit (GATE-03). NO synthetic fallback when `--golden` was explicitly requested. Synthetic mode applies ONLY when `--golden` is omitted (audit skill), never in CI.
3. **Empty index (GATE-04):** new `_checar_indice(h)` calls into the vector DB collection count; if `collection.count() == 0` for all collections → write `{"status": "error", "motivo": "índice vazio (collection.count()==0)", "severidade": "alta"}` and return non-zero exit. (Implementation: reuse `_get_vector_db`/`_listar_colecoes` from `agenticlog.agent`, or probe `_get_retriever("teste")` returning empty after rebuild as a fallback signal.)
4. **Embedding model unavailable:** explicit error exit (cannot compute retrieval metrics).

Exit-code convention for the harness run step:
- `0` = ran (metrics written), regardless of judge skip.
- non-zero = structural failure (no project / golden missing-empty / empty index / embedding model missing). The gate step (§6) is what enforces threshold violations.

---

## 6. Gate Placement (resolves the one deferred implementation choice)

**Decision: a `--gate` flag on the harness `main()`**, keeping a single entrypoint and avoiding a second script that re-reads/re-parses the JSON.

- With `--gate`, after writing results JSON, `main()` evaluates:
  `pass = (agregados.retrieval.hit_rate >= config.RAG_EVAL_MIN_HIT_RATE) and (agregados.retrieval.mrr >= config.RAG_EVAL_MIN_MRR)`.
- `--gate` AND violation → return non-zero exit.
- `--gate` AND judge skipped but retrieval thresholds met → exit 0 (CI-02).
- Without `--gate` (audit skill) → never gates; exit reflects only structural failures.

Rationale: thresholds already live in `config.py`; the harness already imports `config`; a flag keeps logic colocated with the data it just produced and is trivially unit-testable (`portao(agregados, config) -> bool`). The CI workflow calls the harness once with `--gate`.

Alternative considered: separate `scripts/rag_eval_gate.py` reading the artifact. Rejected — duplicates JSON parsing and threshold logic, more drift surface. (If a future need arises to gate on an artifact produced by an earlier job, the `portao()` helper can be reused there.)

---

## 7. CI Workflow Strategy (CI-01, CI-02)

New job `rag-eval` in `.github/workflows/ci.yml`, parallel to `test` and `lint`:

- `runs-on: ubuntu-latest`, Python 3.12 ONLY (no matrix — RESOLVED-04).
- `env`: `OPENAI_API_KEY`, `OPENAI_API_BASE` (harmless; LMStudio absent → judge skipped).
- Steps:
  1. `actions/checkout@v4`.
  2. `actions/setup-python@v5` (3.12).
  3. Install uv; `uv pip install -r requirements-dev.txt --system`; `uv pip install -e . --system`.
  4. Cache HF model: `actions/cache@v4` on `~/.cache/huggingface` (and/or `~/.cache/torch/sentence_transformers`), key includes `EMBEDDING_MODEL` string hash.
  5. Create `.env` from `.env.example`; ensure `data/documents/` is populated (corpus committed) so the index can be built.
  6. `python -m agenticlog.rag --rebuild` — full rebuild (avoids silent-degradation; CLAUDE.md).
  7. `python scripts/rag_eval.py --golden evals/rag_golden.json --out rag_eval_results.json --k 3 --gate`.
  8. `actions/upload-artifact@v4` for `rag_eval_results.json` with `if: always()` so the artifact is uploaded even on gate failure.
- The harness run step exit code (with `--gate`) determines job pass/fail.
- Harness unit/integration tests: run in this job (or a sibling) as a SEPARATE pytest invocation, e.g. `pytest tests/test_rag_eval.py -v` — explicitly NOT `--cov=agenticlog`. The existing `test` job keeps its `--cov-fail-under=80` invocation unchanged; if `tests/test_rag_eval.py` would otherwise be collected there, exclude it via marker (`@pytest.mark.rag_eval`) and `-m "not rag_eval"` in the coverage job, or via path filter.

Corpus availability note: CI must have `data/documents/` content to build an index. If the corpus PDFs/JSONs are committed, step 5 is a no-op; if not, this is a BLOCKER to flag (see Open Issues in tasks.md). The golden contexts are anchored in `doc1/2/3.json` + 3 logistics PDFs per RESOLVED-05.

---

## 8. Config Constants (GATE-02)

Add to `src/agenticlog/config.py` (typed, with Portuguese-comment convention, English constant names):

```python
# RAG Eval — thresholds do gate de qualidade no CI (ajustáveis; frouxos no início).
RAG_EVAL_MIN_HIT_RATE: float = 0.7   # Hit Rate mínimo para o CI passar
RAG_EVAL_MIN_MRR: float = 0.6        # MRR mínimo para o CI passar
RAG_EVAL_MATCH_THRESHOLD: float = 0.6  # cosseno mínimo chunk↔contexto_ref p/ contar "hit"
RAG_EVAL_K: int = 3                  # top-k avaliado (limitado por RETRIEVAL_K_TOTAL)
```

Validation guard (fail-fast, consistent with existing config validators) for ranges `0.0 <= x <= 1.0` and `k >= 1`.

---

## 9. Components & Interfaces (harness internals)

Pure, unit-testable helpers (dependency-injected stubs in tests):

| Function | Signature (intent) | Requirement |
|----------|--------------------|-------------|
| `_carregar_golden(path)` | `Path -> list[dict]`; drops entries without `pergunta`, logs warning per drop | GOLD-04 |
| `_normalizar_contexto_ref(item)` | `dict -> list[str] | None` (str→[str], list→list, absent→None) | RETR-* |
| `_metrica_retrieval(emb, chunks, refs, thr)` | `-> {hit, mrr, precision, recall}` numeric | RETR-01/02 |
| `_answer_correctness(emb, resposta, resposta_ref)` | `-> float | "resposta_ref ausente"` | GOLD-02 |
| `_avaliar_pergunta(h, client, model, item, k)` | per-entry dict incl. new fields + skipped judge block | GOLD-02/03, CI-02 |
| `_agregar(linhas)` | aggregate; gated metrics ignore excluded entries | RETR-03 |
| `_checar_indice(h)` | raises/returns error sentinel when all collections empty | GATE-04 |
| `portao(agregados, config)` | `-> bool` (pass/fail vs thresholds) | GATE-01 |

`embed` is injected (the harness passes `_get_rag_embedding_model()`); tests pass a stub exposing `embed_query(text) -> list[float]`. `_get_retriever` is injected via `h`; tests pass a stub returning `Document`-like objects with `.page_content`.

Immutability: helpers return new dicts/lists; no mutation of input golden entries (coding-style rule). Files kept < 400 lines; if `scripts/rag_eval.py` exceeds the limit after extension, split metric helpers into `scripts/rag_eval_metrics.py`.

---

## 10. Reuse from Codebase

- `_get_retriever`, `_get_rag_embedding_model`, `agent_workflow`, `AgentState` — imported via existing `_bootstrap()` (no change).
- `_cosine` helper — already in harness; reused for all cosine computations.
- `ranked_response` dict/str coercion — existing harness logic (lines ~181–185) preserved verbatim.
- Config validation pattern — mirror existing `LOG_LEVEL`/`HISTORY_MAX_ENTRIES` fail-fast `raise ValueError` style.
- CI conventions — copy structure from existing `test`/`lint` jobs (uv install, `.env.example` copy, cache).

---

## 11. CONCERNS.md Mitigations

- **HIGH SPOF / startup (LMStudio):** harness already degrades to skipped without LMStudio; CI relies on embedding-only gated metrics so the SPOF does not block CI (CI-02).
- **Silent-degradation (CLAUDE.md):** `--rebuild` in CI guarantees the index matches current chunking/embedding config; no incremental stale chunks.
- **INFO no CI quality gate:** this feature adds the RAG-quality gate (complements the existing coverage gate, does not replace it).

---

## 12. Risks & Open Issues for tasks

- Corpus presence in CI (BLOCKER candidate) — see tasks.md T-CORPUS.
- HF model cache effectiveness — measured by job duration; acceptable on cold cache.
- `RAG_EVAL_MATCH_THRESHOLD` calibration — golden authors should sanity-check that the named validated candidates (precision 1.0) yield hits at the chosen threshold before locking it.
