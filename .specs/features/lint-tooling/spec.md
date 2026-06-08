# Lint Tooling — Technical Spec

**Path:** `.specs/features/lint-tooling/spec.md`
**TLC scope:** medium
**Based on story:** As a project maintainer, I want ruff, mypy, and bandit enforced on every commit via pre-commit hooks and on every push/PR via a CI lint job, so that code quality and security issues are caught automatically before they reach the main branch.
**Status:** Awaiting human approval

---

## Problem Statement

AgenticLog has no automated static analysis or security scanning. Code quality issues, type errors, and security findings (e.g., hardcoded credentials patterns) can reach `main` undetected. Adding ruff, mypy, and bandit as pre-commit hooks and a parallel CI job closes this gap without blocking the existing test pipeline.

## Goals

- [ ] `pre-commit run --all-files` exits 0 with ruff, mypy, and bandit hooks configured
- [ ] CI lint job runs all three tools against `src/agenticlog/` and exits 0 on every push/PR
- [ ] PRs to `main` are blocked when the CI lint job fails
- [ ] All tool configuration lives in `pyproject.toml` — no separate config files
- [ ] Known false positive B105 on `LLM_API_KEY` is suppressed and does not cause build failure

## Out of Scope

| Feature | Reason |
|---------|--------|
| ruff --fix (auto-fix) | Approved story: check-only mode only |
| ruff format | Explicitly excluded in approved story |
| isort / black | Not part of this feature |
| mypy strict mode | Approved story: permissive mode only |
| test.yml changes | Approved story: lint job in ci.yml only, test.yml untouched |
| New tests | Lint tooling requires no new pytest tests |

---

## User Stories

### P1: Pre-commit hooks for ruff, mypy, bandit ⭐ MVP

**User Story**: As a project maintainer, I want ruff, mypy, and bandit to run automatically on every commit, so that violations are caught locally before they reach CI.

**Why P1**: Local enforcement is the first line of defence; CI is the safety net.

**Acceptance Criteria**:
1. WHEN `pre-commit run --all-files` is executed THEN it SHALL exit 0 with all three hooks passing
2. WHEN a ruff violation exists in a staged file THEN the hook SHALL block the commit and report the offending line
3. WHEN a mypy type error exists in a staged file THEN the hook SHALL block the commit and report the error
4. WHEN a bandit finding at or above the configured threshold exists THEN the hook SHALL block the commit and report the finding
5. WHEN `LLM_API_KEY = "hermes"` triggers B105 THEN the suppression SHALL prevent a build failure

**Independent Test**: Run `pre-commit run --all-files` on the current codebase after applying all changes; verify exit code 0 and no unresolved findings.

---

### P2: CI lint job parallel to test job

**User Story**: As a project maintainer, I want the lint job to run in CI on every push and PR to `main`, so that violations are caught even if pre-commit was bypassed.

**Why P2**: CI is the authoritative gate; local hooks can be skipped with `--no-verify`.

**Acceptance Criteria**:
1. WHEN a push or PR triggers CI THEN the lint job SHALL run ruff, mypy, and bandit against `src/agenticlog/` in sequence
2. WHEN any lint tool exits non-zero THEN the CI lint job SHALL fail and block the PR
3. WHEN the lint job fails THEN the test job SHALL continue unaffected (parallel, no dependency)
4. WHEN the lint job passes THEN it SHALL exit 0 with all three tools reporting no findings

**Independent Test**: Introduce a deliberate ruff violation in a branch, push, and verify the lint job fails while the test job still runs.

---

## Edge Cases

- WHEN `# type: ignore[reportMissingImports]` (Pyright syntax) is present THEN ruff PGH003 rule SHALL flag it — must be replaced with `# type: ignore[import-untyped]`
- WHEN `import numpy as np  # type: ignore` lacks an error code THEN ruff PGH003 SHALL flag it — must add `[import-untyped]`
- WHEN bandit runs with `-ll -ii` THEN only medium-confidence + medium-severity findings are reported; low-severity false positives are suppressed
- WHEN `AgentState` bare list fields are present THEN mypy permissive mode (`disallow_untyped_defs = false`) SHALL not fail on them

---

## Requirement Traceability

| Requirement ID | Acceptance Criterion | Story | Status |
|----------------|----------------------|-------|--------|
| LINT-01 | pre-commit run --all-files exits 0 | P1-AC1 | Pending |
| LINT-02 | ruff violation blocks commit with line report | P1-AC2 | Pending |
| LINT-03 | mypy error blocks commit with error report | P1-AC3 | Pending |
| LINT-04 | bandit finding blocks commit with report | P1-AC4 | Pending |
| LINT-05 | B105 on LLM_API_KEY suppressed via # nosec B105 | P1-AC5 | Pending |
| LINT-06 | CI lint job runs all three tools | P2-AC1 | Pending |
| LINT-07 | CI lint job failure blocks PR | P2-AC2 | Pending |
| LINT-08 | CI lint and test jobs run in parallel | P2-AC3 | Pending |
| LINT-09 | All tool config in pyproject.toml only | Story AC9 | Pending |
| LINT-10 | requirements-dev.txt includes ruff, mypy, bandit, pre-commit | Story AC10 | Pending |
| LINT-11 | test.yml untouched | Story AC11 | Pending |

**ID format:** `LINT-[NUMBER]`

---

## Data Model Changes

No data model changes.

---

## Process / Background Flow

**Happy path (pre-commit):**
1. Developer runs `git commit`
2. pre-commit triggers hooks in order: ruff → mypy → bandit
3. Each hook runs against staged files only (via `files: ^src/agenticlog/`)
4. All tools exit 0 → commit proceeds

**Failure path — ruff violation:**
1. ruff detects a linting rule violation
2. Hook exits non-zero, prints offending file + line + rule code
3. Commit is aborted; developer fixes violation and retries

**Failure path — mypy type error:**
1. mypy detects a type mismatch or missing annotation
2. Hook exits non-zero, prints file + line + error message
3. Commit is aborted

**Failure path — bandit finding:**
1. bandit detects a medium-severity + medium-confidence finding (above `-ll -ii` threshold)
2. Hook exits non-zero, prints file + line + issue code + severity/confidence
3. Commit is aborted; developer adds `# nosec <CODE>` with justification comment if it is a known false positive

**CI happy path:**
1. Push or PR triggers `ci.yml`
2. `lint` job and `test` job start in parallel
3. lint job: Checkout → Setup Python 3.12 → Install uv → Install deps → ruff check → mypy → bandit
4. All tools exit 0 → lint job passes → PR is unblocked (combined with test job result)

---

## API Changes

No API changes.

---

## Frontend Changes

No frontend changes.

---

## Tests Required

**No new pytest tests** — lint tooling is validated by running the tools themselves.

**Manual validation steps (part of Done criteria):**
- `pre-commit run --all-files` exits 0 on current codebase
- `ruff check src/agenticlog/` exits 0
- `mypy src/agenticlog/` exits 0
- `bandit -r src/agenticlog/ -ll -ii` exits 0
- CI lint job passes on a clean push to a branch

**Existing tests that could break:**
- None — lint tools do not affect pytest execution. `test.yml` is untouched (LINT-11).

---

## Files That Will Change

| File | Change type | Why |
|------|-------------|-----|
| `pyproject.toml` | Modify | Add `[tool.ruff]`, `[tool.mypy]`, `[tool.bandit]` config sections (LINT-09) |
| `.pre-commit-config.yaml` | Create | New file; defines ruff, mypy, bandit hooks (LINT-01) |
| `.github/workflows/ci.yml` | Modify | Add `lint` job parallel to existing `test` job (LINT-06, LINT-07, LINT-08) |
| `requirements-dev.txt` | Modify | Add `ruff`, `mypy`, `bandit`, `pre-commit` with pinned versions (LINT-10) |
| `src/agenticlog/config.py` | Modify | Add `# nosec B105` inline comment to `LLM_API_KEY` line (LINT-05) |
| `src/agenticlog/agent.py` | Modify | Replace Pyright-syntax `# type: ignore` with mypy/ruff-compatible codes (Edge case) |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `agent.py` uses `# type: ignore[reportMissingImports]` (Pyright syntax) — ruff PGH003 will flag it | High | Replace with `# type: ignore[import-untyped]` before first pre-commit run |
| `import numpy as np  # type: ignore` missing error code — ruff PGH003 will flag it | Medium | Add `[import-untyped]` to the comment |
| `AgentState` bare list fields may fail mypy strict mode | Low | Mitigated by `disallow_untyped_defs = false` in permissive mypy config |
| bandit B105 false positive on `LLM_API_KEY = "hermes"` | Medium | Suppressed with `# nosec B105` inline; documented as intentional dummy key for LMStudio |
| Pre-commit hook versions drifting from requirements-dev.txt versions | Low | Pin same versions in both `.pre-commit-config.yaml` additional_dependencies and requirements-dev.txt |
| Developer bypasses pre-commit with `--no-verify` | Low | CI lint job is the authoritative gate; local hooks are convenience only |
| New third-party import added later may have no stubs — mypy would error | Low | `ignore_missing_imports = true` in mypy config suppresses this class of error globally |

---

## Open Questions

None. All questions resolved in approved story:
- ruff: check-only, no --fix
- mypy: `ignore_missing_imports = true`, `disallow_untyped_defs = false`
- CI: lint job parallel to test job, no dependency gate between them
- bandit: `-ll -ii` threshold, `# nosec B105` suppression inline in config.py
- ruff format: out of scope

---

## Success Criteria

- [ ] `pre-commit run --all-files` exits 0 on the current codebase with all three hooks active
- [ ] `ruff check src/agenticlog/` exits 0 (no violations, check-only mode)
- [ ] `mypy src/agenticlog/` exits 0 (permissive mode, no type errors)
- [ ] `bandit -r src/agenticlog/ -ll -ii` exits 0 (B105 on LLM_API_KEY suppressed)
- [ ] CI lint job appears in `ci.yml` as a parallel job alongside `test`
- [ ] CI lint job fails on a deliberate ruff violation introduced in a test branch
- [ ] `test.yml` is byte-for-byte unchanged
- [ ] No tool configuration exists outside `pyproject.toml`
- [ ] `requirements-dev.txt` lists ruff, mypy, bandit, pre-commit
