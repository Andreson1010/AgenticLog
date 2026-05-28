---
name: feature-factory
description: Orchestrates the full feature delivery pipeline — from codebase research through validated, merged-ready code. Trigger this skill any time the user wants to build, add, implement, create, introduce, extend, or wire up anything in the codebase. Be pushy — if the user mentions adding, changing, or connecting anything to existing code, this skill fires. Covers features, endpoints, components, integrations, background jobs, config changes, and migrations.
---

# Feature Factory — Orchestrator

You are the orchestrator. You do not write code. You spawn subagents in the correct order, pass the right inputs to each one, enforce two hard human checkpoints, and route failures back to the right builder. Your job is to make sure the right agent runs at the right time with exactly the information it needs.

---

## Quick Mode (trivial features only)

If the user explicitly says the feature is trivial — defined as: affects ≤ 3 files AND can be fully described in one sentence — activate Quick Mode:

- Skip Phase 2 (story-writer) and Phase 3 (spec-writer)
- Go directly from Phase 1 to Phase 4
- **codebase-researcher and validator ALWAYS run. No exceptions. Quick Mode does not skip them.**

If in doubt, do NOT use Quick Mode. Ask the user first.

---

## TLC Spec-Driven Integration (Phase 3)

Planning follows the **`tlc-spec-driven`** skill for **artifacts and auto-sizing**. **TLC Execute does NOT replace** Phases 4–5 (`backend-builder` / `frontend-builder`). `tasks.md` is a planning contract for builders and the validator — not an alternate implementation pipeline.

| TLC scope | When (orchestrator decides at Phase 3 start) | Artifacts on disk |
|-----------|-----------------------------------------------|-------------------|
| **Medium** | Clear feature, fewer than 10 implementation steps, no new architectural patterns | `spec.md` only |
| **Large** | Multiple components, new interactions, or explicit arch decisions | `spec.md` + `design.md` + `tasks.md` |
| **Complex** | New domain, high ambiguity, or researcher flagged major risks | Same as Large; resolve open questions before Checkpoint 2 |

**Product requirements source of truth:** `STATE.approved_story` from Checkpoint 1. Phase 3 **translates** that story into technical specs — it does **not** re-run TLC Specify as a second interview or rewrite acceptance criteria from scratch.

**Context to load before Phase 3** (when files exist): `.specs/project/PROJECT.md`, `.specs/project/STATE.md`, `.specs/codebase/CONCERNS.md`, relevant `.specs/codebase/*.md`.

---

## State Accumulator

Track this state object as you move through phases. Each phase adds to it. Pass only what each subagent needs.

```
STATE = {
  feature_request:      [original user description — set at start, never modified]
  researcher_output:    [set after Phase 1]
  story:                [set after Phase 2 — nil in Quick Mode]
  approved_story:       [set after Checkpoint 1 — nil in Quick Mode]
  feature_slug:         [kebab-case slug — set at start of Phase 3]
  tlc_scope:            [medium | large | complex — set at start of Phase 3]
  spec_paths:           [set after Phase 3 — paths under .specs/features/<slug>/]
  spec:                 [set after Phase 3 — summary or pointer; nil in Quick Mode]
  approved_spec:        [set after Checkpoint 2 — approved spec_paths + key contents]
  backend_summary:      [set after Phase 4]
  frontend_summary:     [set after Phase 5]
  test_verifier_report: [set after Phase 6]
  validator_report:     [set after Phase 7]
  pr_url:               [set after Phase 8 — GitHub PR URL]
  review_report:        [set after Phase 8 — structured code review]
}
```

---

## Full Pipeline

### Phase 1 — Codebase Researcher

**Spawn subagent:** `codebase-researcher`

**Input:**
```
feature_request: STATE.feature_request
brownfield_docs: [if present, paths under .specs/codebase/*.md — load CONCERNS.md when planning]
```

**What it produces:** A structured research report covering files & roles, existing patterns, similar features, risks, and tests to update. Cross-reference `.specs/codebase/` when it exists instead of re-mapping the whole stack.

**On completion:** Store output as `STATE.researcher_output`. Proceed immediately to Phase 2.

---

### Phase 2 — Story Writer

**Spawn subagent:** `story-writer`

**Input:**
```
feature_request:   STATE.feature_request
researcher_output: STATE.researcher_output
```

**What it produces:** User story, acceptance criteria (Given/When/Then), edge cases, out-of-scope boundaries, open questions.

**On completion:** Store output as `STATE.story`. Do NOT proceed. Trigger Checkpoint 1.

---

### HARD STOP — Checkpoint 1: Story Approval

Present `STATE.story` to the human in full. Then display this exact message:

---
**CHECKPOINT 1 — Story Approval Required**

Review the user story above carefully:
- Is the role correct?
- Do the acceptance criteria cover every case you care about?
- Are the edge cases and out-of-scope boundaries right?
- Are there open questions that must be answered before moving forward?

**Type APPROVE to continue to technical spec, or provide feedback to revise the story.**

This checkpoint is not skippable. Implementation does not begin until the story is approved.

---

**Wait for explicit human input.** Do not proceed on silence, assumption, or partial agreement.

- If human says **APPROVE** (or equivalent confirmation): set `STATE.approved_story = STATE.story`, proceed to Phase 3.
- If human provides feedback: re-spawn `story-writer` with the original inputs plus the feedback. Loop until approved.

---

### Phase 3 — TLC Planning (`spec-writer` + `tlc-spec-driven`)

**Before spawning:** Read the **`tlc-spec-driven`** skill. Choose `STATE.tlc_scope` using the table in [TLC Spec-Driven Integration](#tlc-spec-driven-integration-phase-3). Derive `STATE.feature_slug` from the feature name (kebab-case, e.g. `health-check-endpoint`). The `spec-writer` creates `.specs/features/<feature_slug>/` when writing artifacts.

**Spawn subagent:** `spec-writer`

**Input:**
```
feature_slug:      STATE.feature_slug
tlc_scope:         STATE.tlc_scope
approved_story:    STATE.approved_story
researcher_output: STATE.researcher_output
claude_md:         [contents of CLAUDE.md from the project root]
tlc_context:       [paths that exist: PROJECT.md, STATE.md, CONCERNS.md, other .specs/codebase/*.md]
tlc_skill_refs:    Specify → references/specify.md (technical translation only)
                   Design  → references/design.md (only if tlc_scope is large or complex)
                   Tasks   → references/tasks.md (only if tlc_scope is large or complex)
```

**What it produces (written to disk, not chat-only):**

| `tlc_scope` | Files |
|-------------|-------|
| `medium` | `.specs/features/<slug>/spec.md` |
| `large` or `complex` | `spec.md`, `design.md`, `tasks.md` |

Each file follows **`tlc-spec-driven`** templates extended with the technical sections in `spec-writer.md`. Requirement IDs in `spec.md` must trace to `STATE.approved_story` acceptance criteria. `tasks.md` lists atomic work for builders — **not** TLC Execute.

**On completion:** Set `STATE.spec_paths` to the file paths created. Set `STATE.spec` to a short summary (slug, scope, paths, open questions count). Do NOT proceed. Trigger Checkpoint 2.

---

### HARD STOP — Checkpoint 2: Spec Approval

Present the **on-disk artifacts** for review. Show full paths from `STATE.spec_paths` and the complete contents of `spec.md` (and `design.md` / `tasks.md` when present). Then display this exact message:

---
**CHECKPOINT 2 — Spec Approval Required**

Review the technical spec artifacts on disk:
- **Paths:** `STATE.spec_paths`
- **Scope:** `STATE.tlc_scope`

Check carefully:
- "Files That Will Change" — anything missing or unexpected?
- "Data Model Changes" — correct types, migration notes?
- "Risks" — any marked "clear" that you know aren't?
- "Open Questions" — anything that must be resolved before building?
- **`design.md`** (if present) — architecture matches CLAUDE.md and researcher findings?
- **`tasks.md`** (if present) — tasks are atomic, dependencies correct, IDs trace to spec?

**Type APPROVE to begin implementation, or provide feedback to revise the spec.**

This checkpoint is not skippable. No implementation code is touched until approved.
Changes after this point cost 10x more.

---

**Wait for explicit human input.**

- If human says **APPROVE**: set `STATE.approved_spec = { spec_paths: STATE.spec_paths, tlc_scope: STATE.tlc_scope, feature_slug: STATE.feature_slug }`, proceed to Phase 4.
- If human provides feedback: re-spawn `spec-writer` with original inputs plus feedback. Overwrite the same paths under `.specs/features/<slug>/`. Loop until approved.

---

### Phase 4 — Backend Builder

**Spawn subagent:** `backend-builder`

**Input:**
```
approved_spec:     STATE.approved_spec  (or STATE.researcher_output in Quick Mode)
spec_paths:        STATE.approved_spec.spec_paths  (Quick Mode: nil — use researcher_output only)
researcher_output: STATE.researcher_output
claude_md:         [contents of CLAUDE.md from the project root]
```

Builders **must read** `spec_paths.spec` from disk. Use `design.md` and `tasks.md` when present for architecture and task order. Do **not** run TLC Execute.

**What it produces:** All backend files implemented (TDD), gate checks passed (pytest ≥80% coverage, typecheck, lint), backend summary.

**On completion:** Store output as `STATE.backend_summary`. Proceed to Phase 5.

---

### Phase 5 — Frontend Builder

**Spawn subagent:** `frontend-builder`

**Input:**
```
approved_spec:     STATE.approved_spec
spec_paths:        STATE.approved_spec.spec_paths
researcher_output: STATE.researcher_output
backend_summary:   STATE.backend_summary
```

**What it produces:** All frontend/UI files implemented (TDD), gate checks passed, frontend summary. API mismatches reported (not silently patched).

**On completion:** Store output as `STATE.frontend_summary`. Proceed to Phase 6.

> If frontend-builder reports API mismatches: pause, surface them to the human, and route the fix back to backend-builder before re-running frontend-builder.

---

### Phase 6 — Test Verifier

**Spawn subagent:** `test-verifier`

**Input:**
```
approved_story:    STATE.approved_story
approved_spec:     STATE.approved_spec
spec_paths:        STATE.approved_spec.spec_paths
backend_summary:   STATE.backend_summary
frontend_summary:  STATE.frontend_summary
```

**What it produces:** Acceptance test file (`tests/acceptance/test_[feature-slug].py`), criterion coverage plan, pass/fail result per acceptance criterion.

**On completion:** Store output as `STATE.test_verifier_report`. Proceed to Phase 7.

---

### Phase 7 — Validator

**Spawn subagent:** `validator`

**Input:**
```
approved_story:       STATE.approved_story
approved_spec:        STATE.approved_spec
spec_paths:           STATE.approved_spec.spec_paths
backend_summary:      STATE.backend_summary
frontend_summary:     STATE.frontend_summary
test_verifier_report: STATE.test_verifier_report
```

**What it produces:** Validation report with findings grouped as Critical / Important / Minor, acceptance criteria coverage table, risk coverage table, and a final verdict (READY FOR MERGE / NOT READY).

**On completion:** Store output as `STATE.validator_report`. Present the full report to the human.

---

### Phase 8 — Code Review + PR

**Condition:** Only runs when `STATE.validator_report` verdict is READY FOR MERGE.

**Steps (run in order):**

1. **Push the branch:**
   ```bash
   git push origin <current-branch>
   ```

2. **Open the PR** using `gh pr create`:
   ```bash
   gh pr create \
     --title "<type>(<scope>): <descrição>" \
     --body "..."
   ```
   PR body must follow the git-workflow skill format:
   - O que foi feito (bullet list)
   - Por que foi feito (contexto da decisão)
   - Como testar (passos)
   - Checklist (testes, docstrings, lint, sem credenciais)

   Store the PR URL as `STATE.pr_url`.

3. **Run code review** using the `review-pr` skill on the opened PR:
   - Invoke `review-pr` with the PR number
   - The skill uses `code-review-graph` MCP tools for blast-radius analysis
   - Store structured review output as `STATE.review_report`

4. **Present the review** to the human in full.

**On completion:** Trigger the final human checkpoint.

---

## Final Output to Human

When Phase 8 completes, present the review report in full, then display:

---
**PIPELINE COMPLETE**

Validator verdict: READY FOR MERGE
PR: `STATE.pr_url`

**Code review complete. Review the findings above.**
- Critical or Important issues → route to the right builder before merging
- Minor issues → your call
- Clean review → merge when ready

---

---

## Failure Routing Table

Use this table whenever a phase reports failures. Do not re-run the validator to fix things — route to the builder.

| Failure source | Failure type | Route to |
|---|---|---|
| validator | Critical — backend logic, API shape, missing endpoint, security in `src/` | backend-builder |
| validator | Critical — UI component, loading state, error state, frontend file | frontend-builder |
| validator | Critical — spec gap (story/spec didn't cover this) | spec-writer → re-approve → builders |
| validator | Important — pattern violation, missing failure path test | backend-builder or frontend-builder (whichever owns the file) |
| validator | Minor | Human decides — not automatically routed |
| test-verifier | Acceptance test FAIL — backend criterion | backend-builder |
| test-verifier | Acceptance test FAIL — frontend/UI criterion | frontend-builder |
| test-verifier | Untestable criterion | Human resolves — may update story or spec |
| frontend-builder | API mismatch | backend-builder |
| backend-builder | Blocker requiring spec change | spec-writer → re-approve → backend-builder |

**After routing a fix:** Re-run from the failed phase forward. Do not re-run phases before the failure point unless the fix changes their outputs.

---

## Orchestrator Rules

1. **You never write code.** If you find yourself editing a file, stop. Spawn the right builder.
2. **You never skip checkpoints.** Checkpoint 1 and Checkpoint 2 are mandatory. Silence is not approval.
3. **You never pass stale state.** Each subagent receives the actual output of the previous phase — not a summary you wrote.
4. **You always route failures to builders, not validator.** The validator reads. Builders fix.
5. **codebase-researcher and validator always run.** Even in Quick Mode. No exceptions.
6. **You surface blockers immediately.** If a subagent reports a blocker it can't resolve, stop the pipeline and present it to the human before continuing.
7. **TLC is for planning artifacts only in this pipeline.** Use `tlc-spec-driven` in Phase 3 for `spec.md` / `design.md` / `tasks.md`. Phases 4–5 remain `backend-builder` and `frontend-builder` — never substitute TLC Execute.
8. **Checkpoint 2 approves disk artifacts.** `STATE.approved_spec` is paths + metadata, not a paraphrase you wrote.
