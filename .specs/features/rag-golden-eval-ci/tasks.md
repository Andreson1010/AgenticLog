# RAG Golden Set + CI Regression Gate — Tasks

**Path:** `.specs/features/rag-golden-eval-ci/tasks.md`
**Spec:** `.specs/features/rag-golden-eval-ci/spec.md`
**Design:** `.specs/features/rag-golden-eval-ci/design.md`
**TLC scope:** large
**Status:** Awaiting human approval

> Tasks are atomic, ordered, and traceable to requirement IDs in `spec.md`.
> These plan the feature-factory build phases (Phases 4–5). TLC Execute is not used.
> Test/gate notes follow CLAUDE.md: mock LLM, `teste_N_`-friendly naming, files < 400 lines,
> constants in `config.py`, immutability (return new objects). No `.specs/codebase/TESTING.md` present.

---

## Dependency Graph (summary)

```
T1 (config consts) ─┐
T2 (relocate harness) ─┬─> T3 (retrieval metrics no-LLM) ─> T4 (recall/correctness)
                       │                                       │
T0 (corpus check) ─────┘                                       ├─> T6 (gate flag)
T5 (golden set) ───────────────────────────────────────────────┘     │
                                                                T7 (unit tests) ─> T8 (integration tests)
T1,T6 ─> T9 (CI workflow) ──(depends on T0 corpus, T2,T3,T4,T5,T6)
T10 (docs/state) last
```

---

## T0 — Verify corpus availability in CI  (BLOCKER gate)
**Requirement:** prerequisite for CI-01, GATE-04
**Depends on:** none
**Do:** Confirm `data/documents/` (doc1/2/3.json + 3 logistics PDFs) is committed/available to the CI runner so `python -m agenticlog.rag --rebuild` can build a non-empty index. If the corpus is gitignored or absent, this is a BLOCKER — decide how CI obtains it (commit a minimal eval corpus, or a fetch step) before T9.
**Done when:** Documented confirmation that CI can build a non-empty index from committed data, OR an agreed corpus-provisioning step exists.
**Notes:** `data/vectordb/` is gitignored and rebuilt in CI; `data/documents/` content is the open question (researcher: CI today creates empty `data/vectordb`, does NOT build the index). Per memory note: do NOT `git add -A` (sweeps untracked PDFs) — add corpus files explicitly if committing.

---

## T1 — Add RAG eval threshold constants to config.py
**Requirement:** GATE-02
**Depends on:** none
**Do:** Add `RAG_EVAL_MIN_HIT_RATE=0.7`, `RAG_EVAL_MIN_MRR=0.6`, `RAG_EVAL_MATCH_THRESHOLD=0.6`, `RAG_EVAL_K=3` (typed `float`/`int`, Portuguese comments). Add fail-fast `raise ValueError` guards (0.0–1.0 ranges; k≥1) mirroring existing `LOG_LEVEL`/`HISTORY_MAX_ENTRIES` validators.
**Done when:** `import agenticlog.config` exposes the four constants; invalid override raises `ValueError`. `ruff`/`mypy` clean.
**Test/gate:** Constant existence + validation guard test (can live in harness test file to avoid touching `--cov=agenticlog` semantics, or a small `tests/test_config.py` addition — but ensure it does not drop package coverage below 80%).

---

## T2 — Relocate harness to scripts/rag_eval.py
**Requirement:** HARN-01
**Depends on:** none
**Do:** Copy `.claude/skills/rag-pipeline-audit/scripts/rag_eval.py` to `scripts/rag_eval.py` (versioned). Verify `_bootstrap()` still resolves `src/agenticlog` from repo root. Confirm `scripts/` is outside the `agenticlog` package and outside `--cov=agenticlog`.
**Done when:** `python scripts/rag_eval.py --help` works from repo root; harness imports `agenticlog` successfully when index exists.
**Notes:** Keep the skill copy as the audit mirror (or point the skill at `scripts/`). Do not add `scripts/` to `--cov=agenticlog`.

---

## T3 — Implement embedding-only retrieval metrics
**Requirement:** RETR-01, RETR-02, RETR-03
**Depends on:** T1, T2
**Do:** Add `_normalizar_contexto_ref(item)` (str→[str], list→list, absent→None) and `_metrica_retrieval(emb, chunks, refs, thr)` returning `{hit, mrr, precision, recall}` using `_cosine` and `RAG_EVAL_MATCH_THRESHOLD`. Wire into `_avaliar_pergunta` so Hit Rate / MRR / Context Precision / Context Recall come from embeddings (no LLM). Make `_agregar` ignore entries without `contexto_ref` for gated metrics (RETR-03) and emit a `retrieval` sub-object.
**Done when:** Running harness without LMStudio still produces numeric `retrieval.hit_rate/mrr/context_precision/context_recall`; judge block becomes `{"status":"skipped",...}`.
**Test/gate:** Covered by T7.

---

## T4 — Implement Answer Correctness + Context Recall consumption
**Requirement:** GOLD-02, GOLD-03
**Depends on:** T3
**Do:** Extend `_avaliar_pergunta` to compute Answer Correctness via `cosine(embed(resposta), embed(resposta_ref))`; if `resposta_ref` absent → `"resposta_ref ausente"`. Context Recall per design §3.2; if `contexto_ref` absent → `"contexto_ref ausente"`. Update `_agregar` to replace the hardcoded `"requer golden set"` with means of numeric per-entry values (ignore string sentinels). Preserve existing `ranked_response` dict/str coercion. Optionally compute judge-based correctness as report-only when LMStudio present.
**Done when:** Entries with refs report numeric recall/correctness; entries missing a ref report the correct sentinel; aggregate no longer contains `"requer golden set"`.
**Test/gate:** Covered by T7.

---

## T5 — Author curated golden set evals/rag_golden.json
**Requirement:** GOLD-01, GOLD-05
**Depends on:** none (content), validated against T0 corpus
**Do:** Create `evals/rag_golden.json` with 8–10 entries: `pergunta`, `resposta_ref`, `contexto_ref` (+ optional `categoria`). Cover the 3 named categories (controle de estoque, processamento de pedidos, definição de operador logístico) plus transporte/armazenagem/distribuição. Anchor every `contexto_ref` in real corpus text (doc1/2/3.json + 3 PDFs). Include the 3 validated candidates (precision 1.0/faith 1.0): "finalidade do controle de estoques", "objetivo do Processamento de Pedidos", "definição de operador logístico". Include at least one entry WITHOUT `contexto_ref` to exercise GOLD-03. Validate JSON (no trailing commas; prefer full-file write per CLAUDE.md).
**Done when:** File parses; ≥3 named categories present; refs traceable to corpus; loader (T6 validation) accepts all valid entries.
**Test/gate:** Coverage assertion (≥3 categories) in T7/T8.

---

## T6 — Harness validation, failure modes, and gate flag
**Requirement:** GOLD-04, GATE-01, GATE-03, GATE-04, CI-02
**Depends on:** T1, T3, T4
**Do:**
- Strengthen `_carregar_golden`: drop entries without `pergunta`, log a WARNING per drop (GOLD-04).
- Golden absent/empty WHEN `--golden` is provided → write `{"status":"error",...}` and return non-zero (GATE-03); synthetic mode only when `--golden` omitted.
- Add `_checar_indice(h)`: all collections empty (`collection.count()==0`) → `{"status":"error","severidade":"alta",...}`, non-zero exit (GATE-04).
- Add `--gate` flag and `portao(agregados, config) -> bool` (pass iff `hit_rate>=RAG_EVAL_MIN_HIT_RATE and mrr>=RAG_EVAL_MIN_MRR`); with `--gate`, violation → non-zero exit; judge-skipped alone never fails (CI-02).
**Done when:** Missing/empty golden and empty index exit non-zero with explicit message; `--gate` returns non-zero below threshold, zero above; entry without `pergunta` dropped with warning.
**Test/gate:** Covered by T7.

---

## T7 — Unit tests for harness (dedicated, outside --cov=agenticlog)
**Requirement:** GOLD-02/03/04, RETR-01/02/03, GATE-01/03/04
**Depends on:** T3, T4, T5, T6
**Do:** Add `tests/test_rag_eval.py` with stub embedding model (`embed_query` fixed vectors) and stub retriever (`Document`-like with `.page_content`). Cover: loader drop+warning; per-entry hit/MRR/precision above & below threshold; answer correctness numeric; `contexto_ref`/`resposta_ref` absent sentinels + gated exclusion; aggregation; `portao()` pass/fail; missing/empty golden and empty index non-zero. Mark `@pytest.mark.rag_eval` (or place under a path) so it is excluded from `pytest --cov=agenticlog --cov-fail-under=80`.
**Done when:** Tests pass; running them does NOT alter the package coverage number gated in the `test` job.
**Test/gate:** `pytest tests/test_rag_eval.py -v` green; no real LLM/embedding calls (all stubbed).

---

## T8 — Acceptance / integration test (mock LLM)
**Requirement:** CI-01, CI-02, GOLD-05
**Depends on:** T7
**Do:** Add an integration test (`@pytest.mark.integration` + `rag_eval` marker) running the full harness against a tiny fixture golden with a stub/in-memory retriever and judge client mocked as UNAVAILABLE. Assert: judge block skipped, `retrieval` metrics numeric, results JSON shape, and that retrieval-above-threshold yields overall pass. Assert golden covers ≥3 named categories (GOLD-05) by parsing `evals/rag_golden.json`.
**Done when:** Integration test passes with LLM mocked; demonstrates CI-02 (skipped judge does not fail).
**Test/gate:** No real LMStudio; conftest skip rules for hnswlib respected on Windows.

---

## T9 — Add rag-eval CI job
**Requirement:** CI-01, GATE-01, CI-02
**Depends on:** T0, T1, T2, T3, T4, T5, T6
**Do:** Add job `rag-eval` to `.github/workflows/ci.yml` (ubuntu, Python 3.12, no matrix): checkout; setup-python 3.12; install uv + deps + `-e .`; cache HF model (`~/.cache/huggingface`, key on `EMBEDDING_MODEL`); copy `.env.example`→`.env`; ensure corpus present (T0); `python -m agenticlog.rag --rebuild`; `python scripts/rag_eval.py --golden evals/rag_golden.json --out rag_eval_results.json --k 3 --gate`; `actions/upload-artifact@v4` with `if: always()`. Optionally run harness unit/integration tests here as a separate invocation (NOT `--cov=agenticlog`).
**Done when:** Workflow YAML valid; `rag-eval` job runs on PRs; artifact uploaded; gate fails the job below threshold and passes above; existing `test`/`lint` jobs unchanged.
**Test/gate:** Validate YAML (no syntax errors). Confirm the existing 80% coverage job is not affected by harness tests.

---

## T10 — Docs / state update
**Requirement:** traceability / housekeeping
**Depends on:** T9
**Do:** Document the golden schema + harness usage (README or `scripts/` docstring already covers CLI; ensure `--gate` documented). Update `.specs/codebase/STATE.md` todo "Add GitHub Actions CI with coverage gate" with the RAG-quality-gate addition. Conventional Commit in Portuguese (`feat(ci): ...`, `feat(eval): ...`).
**Done when:** Docs reflect new job + golden schema; STATE.md updated.

---

## Verification checklist (maps to spec Success Criteria)

- [ ] `evals/rag_golden.json` 8–10 entries, ≥3 named categories, refs from corpus (T5, GOLD-01/05).
- [ ] Numeric Context Recall + Answer Correctness; `"contexto_ref ausente"` where applicable (T4, GOLD-02/03).
- [ ] Hit Rate / MRR / Context Precision computed without LLM (T3, RETR-01/02).
- [ ] `rag-eval` job (3.12) builds index, runs harness, uploads artifact, gates on 0.7/0.6 (T9, CI-01, GATE-01).
- [ ] Judge-skipped never blocks CI; missing/empty golden + empty index non-zero (T6, CI-02, GATE-03/04).
- [ ] Thresholds in `config.py` (T1, GATE-02).
- [ ] Existing `test` (3.11+3.12, 80%) and `lint` jobs green and unaffected (T7, T9).
