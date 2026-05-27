# State

## Decisions

- **2026-05-27 — Adopted tlc-spec-driven for spec-driven development.** Brownfield mapping done. All 7 codebase docs created. Use these docs as context when specifying new features.
- **LMStudio local-only.** No external LLM API. hermes-3-llama-3.2-3b must be pre-loaded in LMStudio GUI before running any agent path.
- **Portuguese naming convention.** Function and variable names in Portuguese; constants in English. Docstrings use custom "Entrada/Saída" format in Portuguese.
- **All tests mock external deps.** No real LLM or ChromaDB calls in tests. Follows project rule: "Always mock LLM calls".

## Blockers

None currently.

## Todos

- [x] Fix SPOF: lazy LLM init — merged PR #8 (2026-05-27)
- [ ] Load LLM credentials from .env instead of config.py hardcodes
- [ ] Add GitHub Actions CI with coverage gate

## Deferred Ideas

- REST API via FastAPI (dependency installed, not implemented)
- Streamlit document ingestion UI
- Multi-collection ChromaDB support

## Lessons

- `.specs/` directory was already present before tlc-spec-driven adoption — pre-existing spec in `.specs/features/portuguese-docstrings/spec.md` was created outside the process and can remain as-is.

## Preferences

- Lightweight tasks (state updates, session handoff) work well with faster/cheaper models.
