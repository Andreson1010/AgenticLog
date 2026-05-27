# Portuguese Docstrings & Inline Comments — Technical Spec

**Intended path:** `.specs/features/portuguese-docstrings/spec.md`
**Based on story:** Add complete, consistent Portuguese docstrings and inline comments to all five source modules so developers can understand intent, design decisions, and security rationale without external docs.
**Status:** Awaiting human approval

---

## Problem Statement

The five source modules (`config.py`, `rag.py`, `agent.py`, `app.py`, `__init__.py`) have inconsistent documentation coverage. Several non-obvious decisions — why `FORBIDDEN_JSON_KEYS = ("lc",)` blocks LangChain serialization injection, why cosine similarity is used to rank multiple LLM responses, and the lifecycle of `session_state` keys in Streamlit — are not explained in any comment or docstring. Onboarding developers must reverse-engineer intent from code alone.

---

## Goals

1. Every public constant, function, class, and LangGraph node in the five source files has a Portuguese docstring or inline comment.
2. Security-sensitive logic (path traversal, forbidden JSON keys) has explicit rationale comments.
3. `AgentState` fields are self-describing via inline comment or `Field(description=...)`.
4. `app.py` global variables and `session_state` keys have lifecycle comments.
5. `__init__.py` has a module-level docstring describing the public API.
6. `pytest` suite passes unchanged after all additions.

---

## Out of Scope

- New `.md` documentation files.
- Documenting `tests/` directory.
- Translating `CLAUDE.md` to Portuguese.
- Adding or changing type annotations.
- Any logic changes whatsoever.
- Documenting third-party library internals.

---

## User Stories (P1 MVP)

### US-1 — config.py constants documented

**WHEN** a developer opens `config.py`
**THEN** every constant SHALL have an inline comment explaining its purpose
**AND** `FORBIDDEN_JSON_KEYS = ("lc",)` SHALL have a comment stating: mitigates LangChain serialization injection via the `"lc"` key used by LangChain's `Serializable` base class.

### US-2 — rag.py functions documented

**WHEN** a developer reads any function in `rag.py`
**THEN** each function SHALL have a docstring with: one-line purpose, params/return (where non-obvious), and security rationale for `_valida_path_documentos`, `_valida_json_sem_chaves_proibidas`, and `_valida_arquivos_json`.
**AND** `RAGSecurityError` class SHALL have a docstring explaining when it is raised.
**AND** `cria_vectordb` SHALL document the `global vectordb` side effect and the `jq_schema` transformation logic.

### US-3 — agent.py nodes and AgentState documented

**WHEN** a developer reads `agent.py`
**THEN** each of the six node functions SHALL have a docstring describing its role in the LangGraph graph and its inputs/outputs on `AgentState`.
**AND** `AgentState` SHALL have a class-level docstring explaining it is the immutable-by-convention state carrier between LangGraph nodes.
**AND** each field of `AgentState` SHALL have an inline comment explaining its role:
  - `query` — pergunta original do usuário
  - `next_step` — rota decidida: "retrieve", "gerar" ou "usar_web"
  - `retrieved_info` — documentos retornados pelo retriever vetorial
  - `possible_responses` — 5 respostas geradas pelo LLM para ranqueamento
  - `similarity_scores` — scores de cosseno de cada resposta vs. contexto recuperado
  - `ranked_response` — melhor resposta após ranqueamento por similaridade
  - `confidence_score` — score de confiança da resposta ranqueada (0.0–1.0)
**AND** `avalia_similaridade` SHALL have a comment explaining WHY cosine similarity is used (respostas semanticamente próximas ao contexto recuperado têm maior fidelidade factual).
**AND** `passo_decisao_agente` SHALL have a comment documenting the keyword lists that drive routing.
**AND** global variables `llm`, `embedding_model`, `vector_db`, `retriever`, `prompt_rag_retrieve`, `prompt_gerar`, `search`, `web_search_tool`, `avk_agent_executor`, `workflow`, `agent_workflow` SHALL each have a one-line lifecycle comment (inicializado na importação do módulo).
**AND** REGRAS sections inside prompt templates SHALL have a brief inline comment explaining their enforcement role.

### US-4 — app.py session_state and globals documented

**WHEN** a developer reads `app.py`
**THEN** the four `session_state` initializations SHALL be preceded by a block comment listing all keys and their purpose/lifecycle (inicializados uma vez por sessão Streamlit; persistem entre reruns).
**AND** the `_ROTAS` dict SHALL have a comment explaining it maps `next_step` values to user-facing Portuguese labels.
**AND** the `output.get(...)` block SHALL have a comment explaining that `agent_workflow.invoke` returns a dict and keys mirror `AgentState` fields.

### US-5 — __init__.py module docstring

**WHEN** a developer imports `agenticlog`
**THEN** the module-level docstring SHALL describe: what the package does, the two public exports (`AgentState`, `agent_workflow`), and point to `agent.py` for full workflow details.

### US-6 — pytest passes unchanged

**WHEN** docstrings are added to any module
**THEN** `pytest --cov=agenticlog --cov-report=term-missing -v` SHALL pass with the same result as before
**AND** no test file SHALL be modified.

---

## Edge Cases

| Edge case | Required handling |
|-----------|-------------------|
| Docstring added to a function whose name starts with `test_` or `teste_` in source modules | Not applicable — no such functions exist in the five target modules |
| Multi-line docstring indentation | Follow existing project convention: `"""` on its own line for multi-line, single-line `"""text."""` for brief |
| `global vectordb` declared twice in `cria_vectordb` (lines 101 and 133) | Document the duplicate as a known redundancy; do NOT remove it (out of scope) |
| `passo_decisao_agente` mutates `state` directly instead of returning a new object | Document the existing mutation pattern; do NOT refactor (out of scope) |
| `avalia_similaridade` returns early with `[0.0] * len(response_texts)` when embeddings are empty | Docstring SHALL cover this fallback path |

---

## Requirement Traceability

| Acceptance criterion | Maps to |
|----------------------|---------|
| `config.py`: every constant has inline comment | US-1 |
| `FORBIDDEN_JSON_KEYS` rationale | US-1 |
| `rag.py`: every function has docstring with security rationale | US-2 |
| `agent.py`: every node + AgentState field documented | US-3 |
| Cosine similarity rationale | US-3 |
| `app.py`: session_state block comment + globals | US-4 |
| `__init__.py`: module-level docstring | US-5 |
| pytest passes unchanged | US-6 |

---

## Data Model Changes

None. `AgentState` fields are documented only; no new fields, no type changes.

If inline `Field(description=...)` style is chosen over inline comments for `AgentState`, the Pydantic field default values must be preserved exactly:

```python
# Before (existing)
next_step: str = ""

# After (option A — inline comment, preferred for brevity)
next_step: str = ""  # rota decidida: "retrieve", "gerar" ou "usar_web"

# After (option B — Field, only if team prefers Pydantic style)
next_step: str = Field(default="", description="Rota decidida: 'retrieve', 'gerar' ou 'usar_web'.")
```

Option A (inline comment) is preferred to avoid changing the public API surface.

---

## Process / Background Flow

No process or background flow changes. This feature is purely additive text.

---

## API Changes

None. No function signatures, return types, or module exports change.

---

## Frontend Changes

None. `app.py` receives comment additions only; no UI behavior changes.

---

## Tests Required

No new tests are required. The existing suite (`tests/test_rag.py`, `tests/test_agent.py`) must continue to pass. The implementation task includes a mandatory final step: run `pytest --cov=agenticlog --cov-report=term-missing -v` and confirm green before closing the story.

Failure path to verify: if any added docstring inadvertently contains a syntax error (e.g., unclosed triple-quote), pytest collection will fail with `SyntaxError`. Fix the syntax; do not modify test files.

---

## Files That Will Change

| File | Lines (current) | Changes |
|------|-----------------|---------|
| `src/agenticlog/config.py` | 31 | Inline comments on all constants; expand `FORBIDDEN_JSON_KEYS` comment |
| `src/agenticlog/rag.py` | 151 | Docstrings on `RAGSecurityError`, `_valida_path_documentos`, `_valida_json_sem_chaves_proibidas`, `_valida_arquivos_json`, `cria_vectordb` |
| `src/agenticlog/agent.py` | 236 | Class docstring + field comments on `AgentState`; docstrings on all 6 node functions; lifecycle comments on all module-level globals; REGRAS comment in prompt templates |
| `app.py` | 150 | Block comment for `session_state` keys; comment on `_ROTAS`; comment on `output.get(...)` block |
| `src/agenticlog/__init__.py` | 6 | Expand module-level docstring |

**Files that will NOT change:** `tests/`, `CLAUDE.md`, `requirements*.txt`, `pyproject.toml`, `.github/`, `data/`.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Triple-quote docstring breaks f-string or existing string inside function | Low | High (SyntaxError) | Run `python -m py_compile <file>` after editing each file before running pytest |
| `global vectordb` declared twice in `cria_vectordb` causes confusion if comment is added only once | Low | Low | Add identical comment to both declarations |
| Inline comment on `AgentState` field chosen vs `Field(description=...)` introduces API drift if future code inspects field metadata | Low | Low | Use inline comment (Option A) consistently; document this choice here |
| `passo_decisao_agente` mutation of `state` is flagged by reviewer as anti-pattern | Medium | Low | Docstring acknowledges the pattern; refactor is explicitly out of scope |

---

## Open Questions

1. **Field style for AgentState:** Inline comment (Option A) vs `Field(description=...)` (Option B)? Recommendation: Option A — consistent with existing project style and avoids Pydantic API surface changes. Needs human confirmation before implementation.

2. **`prompt_rag_retrieve` and `prompt_gerar` REGRAS comment placement:** Should the inline comment appear above each `REGRAS` line or as a block before the template string? Recommendation: one-line comment directly above the `REGRAS DE RESPOSTA:` section inside the template. Needs human confirmation.

3. **`cria_vectordb` `jq_schema` comment:** The jq expression `to_entries | map(.key + ": " + .value) | join("\n")` flattens JSON key-value pairs into plain text for embedding. Should this be a docstring addition or an inline comment on the assignment line? Recommendation: inline comment on the assignment line.

---

## Success Criteria

- [ ] All five files have been edited with docstrings/comments matching US-1 through US-5.
- [ ] `python -m py_compile` passes on each of the five files with no errors.
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` exits 0 with the same test count as before.
- [ ] `FORBIDDEN_JSON_KEYS` comment explicitly mentions LangChain `Serializable` serialization injection as the threat.
- [ ] `avalia_similaridade` docstring explicitly states why cosine similarity is the ranking metric.
- [ ] `session_state` keys are listed in a block comment with lifecycle description in `app.py`.
- [ ] No `.md` files were created; no test files were modified; no logic was changed.

---

**This spec is the second human checkpoint.**

Please review the following before approving:

- [ ] **Open Question 1** resolved: inline comment or `Field(description=...)` for `AgentState` fields?
- [ ] **Open Question 2** resolved: REGRAS comment placement (above each line vs. block before template)?
- [ ] **Open Question 3** resolved: `jq_schema` inline comment vs. docstring addition?
- [ ] All five target files listed under "Files That Will Change" are correct and complete.
- [ ] "Out of Scope" list matches your intent (no type annotations, no test changes, no CLAUDE.md changes).
- [ ] Success criteria are measurable and sufficient for story acceptance.

Once you approve, implementation begins. Changes after this point cost 10x more.
