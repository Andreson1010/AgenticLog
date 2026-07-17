# ADR-018 Fase 6 — Remocao de shims de compatibilidade + reescrita de testes — Tasks

**Path:** `.specs/features/rag-fase6-shims-remocao/tasks.md`
**Spec:** `.specs/features/rag-fase6-shims-remocao/spec.md`
**Design:** `.specs/features/rag-fase6-shims-remocao/design.md`
**TLC scope:** complex
**Status:** Awaiting human approval

Tasks para o backend-builder da feature-factory. Ordem CRITICA: singletons primeiro, depois consumers, depois testes, DEPOIS deletar shims (ultimo passo). Gate autoritativo: **CI Linux** (`pytest --cov=agenticlog -v`). Local Windows pula testes Chroma/hnswlib (Smart App Control) — nao confundir com regressao. Padroes: `ruff`/`black`/`isort` limpos, type hints, docstrings PT (Entrada/Saida/Lanca), imutabilidade, funcoes <50 ln, commits Conventional em PT (`refactor:`), branch ja criada (`feature/rag-fase6-shims-remocao`), code-review antes do merge.

**REGRA DE OURO:** NUNCA deletar shims (rag.py, agent.py, api.py, health.py, history.py, config.py:166) antes de:
1. Realocar todos singletons/globals/CLI/side-effects (Fase A)
2. Migrar todos consumers (Fase B)
3. Migrar todos patches de teste (Fase C)
4. Reescrever oracle monkeypatches (Fase C)
5. Remover identity tests + atualizar acicidade (Fase D)

shims sao deletados na Fase E (ULTIMA).

---

## Execution Plan

### Fase A — Realocar singletons/globals/CLI/side-effects (sequencial)
```
T1(RAG:embeddings singleton) ──→ T2(RAG:CLI+side-effects) ──→ T3(RAG:import fitz)
                                                                       │
                                                                       ↓
T4(AGENT:retriever globals) ──→ T5(AGENT:generation _llm) ──→ T6(AGENT:graph search) ──→ T7(AGENT:startup init)
                                                                                                                                       │
                                                                                                                                       ↓
T8(invalidar_vector_db local) ──→ T9(orchestrator lazy import update)
```

### Fase B — Migrar consumers + scripts (paralelo apos T9)
```
                           ┌→ T10 (app.py imports)
                           ├→ T11 (main_api.py string)
T9 (completa) ────────────┼→ T12 (__init__.py imports)
                           ├→ T13 (serving/api.py imports + HistoryStore)
                           ├→ T14 (config.py:166 remove)
                           ├→ T15 (scripts imports)
                           └→ T16 (pyproject.toml mypy override)
```

### Fase C — Migrar patches de teste + oracle (paralelo apos Fase B)
```
Fase B ────┼→ T17 (test_rag.py patches)
            ├→ T18 (test_agentic_rag.py patches)
            ├→ T19 (oracle caracterizacao monkeypatches)
            ├→ T20 (test_adicionar_pdf patches)
            ├→ T21 (acceptance test patches)
            └→ T22 (ingestion test patches)
```

### Fase D — Remover identity tests + atualizar acicidade (paralelo apos Fase C)
```
Fase C ────┼→ T23 (remove TestShimsIdentidade)
            ├→ T24 (remove TestServingShimsIdentidade)
            ├→ T25 (remove TestShimIdentity)
            └→ T26 (update acicidade MODULES)
```

### Fase E — DELETAR SHIMS (ULTIMO passo, sequencial)
```
Fase D ────→ T27 (delete rag.py) ──→ T28 (delete agent.py) ──→ T29 (delete api.py)
             ──→ T30 (delete health.py) ──→ T31 (delete history.py)
```

### Fase F — Gate final (sequencial apos T31)
```
T31 ──→ T32 (gate final suite completa)
```

---

## Task Breakdown

### FASE A — Realocar singletons/globals/CLI/side-effects

#### T1: Unificar `_get_rag_embedding_model` singleton em `ingestion/embeddings.py` (SHIMS-08)

**What:** Mover `_rag_embedding_model` (cache global) e `_get_rag_embedding_model()` (getter+cache) de `rag.py:35-52` para `ingestion/embeddings.py`. Unificar com `criar_embedding_model` ja existente. Adicionar `_rag_embedding_model = None` no topo e `def _get_rag_embedding_model() -> HuggingFaceEmbeddings` que usa `global _rag_embedding_model` e delega a `criar_embedding_model()`.

**Where:** `src/agenticlog/ingestion/embeddings.py` (modificado)
**Depends on:** None
**Reuses:** `criar_embedding_model()` ja existente no mesmo modulo.
**Requirement:** SHIMS-08

**Done when:**
- [ ] `ingestion/embeddings.py` tem `_rag_embedding_model = None` e `def _get_rag_embedding_model()`.
- [ ] `from agenticlog.ingestion.embeddings import _get_rag_embedding_model` funciona e retorna o mesmo singleton em chamadas subsequentes.
- [ ] `_get_rag_embedding_model()` retorna `HuggingFaceEmbeddings` com os mesmos params de criação.
- [ ] `ruff`/`black`/`isort` limpos.
- [ ] `python -c "from agenticlog.ingestion.embeddings import _get_rag_embedding_model"` exit 0.

**Tests:** unit (validado por T17, T19)
**Gate:** quick

---

#### T2: Mover `_executar_main` + `_configurar_logging_cli` + `import fitz` + `RAGSecurityError` para `ingestion/cli.py` (SHIMS-08)

**What:** Mover de `rag.py:160-209` para `ingestion/cli.py`:
1. `_configurar_logging_cli()` — copiar verbatim (adapta logger name para `__name__` se necessario).
2. `_executar_main(argv=None)` — copiar verbatim, mantendo `import argparse`, try/except com `RAGSecurityError` (import operacional `from agenticlog.shared.errors import RAGSecurityError`).
3. `if __name__ == "__main__": _executar_main()` — entrypoint.
4. Adicionar `import fitz` (migrado de rag.py) se nao existir em extraction.py (T3 cobre).

Verificar se `ingestion/cli.py` ja existe com conteudo; se sim, adicionar as funcoes. Se nao, criar.

**Where:** `src/agenticlog/ingestion/cli.py` (modificado ou criado)
**Depends on:** None
**Reuses:** Padrao de `rag.py` verbatim.
**Requirement:** SHIMS-08

**Done when:**
- [ ] `ingestion/cli.py` contem `_configurar_logging_cli()`, `_executar_main(argv=None)`, e `if __name__ ...`.
- [ ] `python -m agenticlog.ingestion` exit 0 (ou mostra help).
- [ ] `python -m agenticlog.ingestion --help` mostra help do CLI.
- [ ] `RAGSecurityError` importado de `agenticlog.shared.errors`.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (validado por T17, T32)
**Gate:** quick

---

#### T3: Garantir `import fitz` em `ingestion/extraction.py` (SHIMS-08)

**What:** Verificar se `import fitz` ja existe em `ingestion/extraction.py`. Se nao, adicionar. O import e necessario para `@patch("agenticlog.rag.fitz.open")` que migra para `@patch("agenticlog.ingestion.extraction.fitz.open")`.

**Where:** `src/agenticlog/ingestion/extraction.py` (modificado)
**Depends on:** None
**Reuses:** `import fitz` ja existente em `rag.py`.
**Requirement:** SHIMS-08

**Done when:**
- [ ] `import fitz` presente em `ingestion/extraction.py`.
- [ ] `python -c "from agenticlog.ingestion.extraction import extrair_texto_pdf"` exit 0.

**Tests:** none
**Gate:** quick

---

#### T4: Realocar `_vector_dbs`, `_embedding_model`, `_get_embedding_model`, `_listar_colecoes`, `_get_vector_db` para `retrieval/retriever.py` (SHIMS-09)

**What:**
1. Adicionar `_vector_dbs: dict[str, Chroma] = {}` em `retrieval/retriever.py`.
2. Adicionar `_embedding_model = None` em `retrieval/retriever.py`.
3. Adicionar `def _get_embedding_model() -> HuggingFaceEmbeddings` (usando `global _embedding_model`, delegando a `_build_embedding_model()`).
4. Verificar que `_build_embedding_model` ja existe em retriever.py — se sim, manter.
5. Verificar que `_listar_colecoes` e `_get_vector_db` ja existem como funcoes parametrizadas. Se existirem, verificar que os consumers usam os parametros corretos. Se NAO existirem como definicoes reais (so como targets dos wrappers), copiar a logica.

NOTA: Estes simbolos sao as DEFINICOES REAIS agora, nao wrappers. Os consumers que importavam de `agenticlog.agent.X` agora importam de `agenticlog.retrieval.retriever.X`.

**Where:** `src/agenticlog/retrieval/retriever.py` (modificado)
**Depends on:** None
**Reuses:** `_build_embedding_model()` ja existente no mesmo modulo.
**Requirement:** SHIMS-09

**Done when:**
- [ ] `_vector_dbs` definido como dict vazio em `retrieval/retriever.py`.
- [ ] `_embedding_model` definido como None em `retrieval/retriever.py`.
- [ ] `_get_embedding_model()` criado e funcional.
- [ ] `_listar_colecoes()` e `_get_vector_db()` sao definicoes reais (nao wrappers).
- [ ] `python -c "from agenticlog.retrieval.retriever import _listar_colecoes, _get_vector_db, _get_embedding_model, invalidar_vector_db"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (validado por T17, T18, T19)
**Gate:** quick

---

#### T5: Realocar `_llm` singleton + `_get_llm` para `retrieval/generation.py` (SHIMS-09)

**What:** Verificar se `_llm` e `_get_llm` ja existem em `retrieval/generation.py`. Se existirem como definicoes reais (nao lazy imports de agent.py), estao corretos. Se forem shims que importam de agent.py, copiar as definicoes reais.

Atualmente `generation.py` importa _llm/_get_llm via lazy import de agent.py dentro de funcoes (DN-2). Com agent.py deletado, estes singletons precisam ser definidos LOCALMENTE em generation.py.

**Where:** `src/agenticlog/retrieval/generation.py` (modificado)
**Depends on:** None
**Reuses:** Ja tem o Protocol `LLMClient`, decorator `_llm_retry`, etc.
**Requirement:** SHIMS-09

**Done when:**
- [ ] `_llm` definido como `None` (ou a definicao real) em `generation.py`.
- [ ] `_get_llm` definido localmente (NAO faz lazy import de agent.py).
- [ ] `from agenticlog.retrieval.generation import _llm, _get_llm` funciona.
- [ ] `python -c "from agenticlog.retrieval.generation import _get_llm"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (validado por T18, T19)
**Gate:** quick

---

#### T6: Realocar `search` (DuckDuckGoSearchAPIWrapper) para `retrieval/graph.py` (SHIMS-09)

**What:** Mover `search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)` de `agent.py:49` para o topo de `retrieval/graph.py` (apos os imports). Adicionar `from langchain_community.utilities import DuckDuckGoSearchAPIWrapper` aos imports de `graph.py`.

**Where:** `src/agenticlog/retrieval/graph.py` (modificado)
**Depends on:** None
**Reuses:** Import de `DuckDuckGoSearchAPIWrapper` ja existe em agent.py — copiar import para graph.py.
**Requirement:** SHIMS-09

**Done when:**
- [ ] `search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)` presente em `retrieval/graph.py`.
- [ ] `from agenticlog.retrieval.graph import search` funciona.
- [ ] `python -c "from agenticlog.retrieval.graph import agent_workflow"` exit 0 (sem ciclo).
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (validado por T19)
**Gate:** quick

---

#### T7: Adicionar side-effects de startup em `retrieval/__init__.py` (SHIMS-09)

**What:** Copiar de `agent.py:31-34` para `retrieval/__init__.py`:
```python
import warnings
warnings.filterwarnings("ignore")

import torch
torch.classes.__path__ = []

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

Colocar ANTES de outros imports para que os side-effects executem na primeira carga do pacote.

**Where:** `src/agenticlog/retrieval/__init__.py` (modificado)
**Depends on:** None
**Reuses:** Codigo verbatim de agent.py.
**Requirement:** SHIMS-09

**Done when:**
- [ ] Side-effects presentes em `retrieval/__init__.py`.
- [ ] `python -c "import agenticlog.retrieval"` exit 0 e executa os side-effects sem erro.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** none (side-effects apenas alteram env vars globais)
**Gate:** quick

---

#### T8: Atualizar `invalidar_vector_db` em `retrieval/retriever.py` para limpar dict LOCAL (SHIMS-10)

**What:** Modificar `invalidar_vector_db()` em `retrieval/retriever.py:152-160`:
- REMOVER `import agenticlog.agent as _agent_mod`
- REMOVER `_agent_mod._vector_dbs.clear()`
- ADICIONAR `_vector_dbs.clear()` — usando o `_vector_dbs` LOCAL definido em T4.

**Where:** `src/agenticlog/retrieval/retriever.py` (modificado)
**Depends on:** T4 (para `_vector_dbs` estar definido em retriever.py)
**Requirement:** SHIMS-10

**Done when:**
- [ ] `invalidar_vector_db()` nao importa `agent`.
- [ ] `invalidar_vector_db()` limpa `_vector_dbs` local.
- [ ] `from agenticlog.retrieval.retriever import invalidar_vector_db; invalidar_vector_db()` funciona sem erro.

**Tests:** unit (validado por T17, T20)
**Gate:** quick

---

#### T9: Atualizar `orchestrator.py` lazy import para `retrieval.retriever` (LM1, SHIMS-17)

**What:** Modificar `ingestion/orchestrator.py:196`:
- ANTES: `from agenticlog.agent import invalidar_vector_db`
- DEPOIS: `from agenticlog.retrieval.retriever import invalidar_vector_db`

**Where:** `src/agenticlog/ingestion/orchestrator.py` (modificado, ln 196)
**Depends on:** T8
**Requirement:** SHIMS-10, SHIMS-17

**Done when:**
- [ ] `orchestrator.py` lazy import aponta para `agenticlog.retrieval.retriever`.
- [ ] `python -c "from agenticlog.ingestion.orchestrator import _notificar_invalidacao"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** integration (validado por T32)
**Gate:** quick

---

### FASE B — Migrar consumers + scripts

#### T10: Migrar imports de `app.py` (SHIMS-07)

**What:** Atualizar `app.py:9` e `app.py:16-21`:
- `from agenticlog.agent import _listar_colecoes` → `from agenticlog.retrieval.retriever import _listar_colecoes`
- `from agenticlog.rag import RAGSecurityError, adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente, sanitizar_nome_colecao` → `from agenticlog.shared.errors import RAGSecurityError; from agenticlog.ingestion.orchestrator import adicionar_documento_incrementalmente, adicionar_pdf_incrementalmente; from agenticlog.ingestion.security import sanitizar_nome_colecao`
- `app.py:27` `MSG_CONNECT_ERROR = "... uvicorn agenticlog.api:app"` → `"... uvicorn agenticlog.serving.api:app"` (LM2)

**Where:** `app.py` (modificado)
**Depends on:** Fase A completa
**Requirement:** SHIMS-07, SHIMS-17

**Done when:**
- [ ] Todos os imports de `agenticlog.agent` e `agenticlog.rag` removidos.
- [ ] Novos imports funcionam.
- [ ] `python app.py` (Streamlit) carrega sem ImportError.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** none (Streamlit nao tem testes automatizados)
**Gate:** quick

---

#### T11: Atualizar `main_api.py` string entrypoint (LM2, SHIMS-17)

**What:** `main_api.py:12` — `"agenticlog.api:app"` → `"agenticlog.serving.api:app"`

**Where:** `main_api.py` (modificado)
**Depends on:** None
**Requirement:** SHIMS-17

**Done when:**
- [ ] `main_api.py` contem `"agenticlog.serving.api:app"`.
- [ ] `python main_api.py` (com FastAPI) carrega sem ImportError.

**Tests:** none
**Gate:** quick

---

#### T12: Migrar imports de `src/agenticlog/__init__.py` (SHIMS-07)

**What:**
- `from agenticlog.agent import AgentState, agent_workflow` → `from agenticlog.retrieval.state import AgentState; from agenticlog.retrieval.graph import agent_workflow`
- `from agenticlog.health import LMStudioUnavailableError, ModeloNaoCarregadoError, check_lmstudio_health` → `from agenticlog.serving.health import ...`

**Where:** `src/agenticlog/__init__.py` (modificado)
**Depends on:** Fase A completa
**Requirement:** SHIMS-07

**Done when:**
- [ ] `from agenticlog import AgentState, agent_workflow, check_lmstudio_health` funciona.
- [ ] `python -c "import agenticlog"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (validado por T32)
**Gate:** quick

---

#### T13: Migrar imports de `serving/api.py` (LM4, SHIMS-07)

**What:**
- `from agenticlog.history import HistoryStore` → `from agenticlog.observability.history import HistoryStore`
- `from agenticlog.rag import _get_rag_embedding_model` → `from agenticlog.ingestion.embeddings import _get_rag_embedding_model`

**Where:** `src/agenticlog/serving/api.py` (modificado, ln 36-37)
**Depends on:** T1
**Requirement:** SHIMS-07, SHIMS-17

**Done when:**
- [ ] `from agenticlog.serving.api import app` funciona (os imports do topo resolvem).
- [ ] `python -c "from agenticlog.serving.api import consultar"` exit 0.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** unit (validado por T32)
**Gate:** quick

---

#### T14: Remover `_JsonFormatter` shim de `config.py` (SHIMS-06, SHIMS-11)

**What:** Remover `config.py:166` — `from agenticlog.observability.logging import _JsonFormatter`. Todo consumer que precisar de `_JsonFormatter` importa de `agenticlog.observability.logging` diretamente.

Verificar quem usa `from agenticlog.config import _JsonFormatter`:
- `ingestion/cli.py:_configurar_logging_cli` → mudar para `from agenticlog.observability.logging import _JsonFormatter`

**Where:** `src/agenticlog/config.py` (modificado, ln 166)
**Depends on:** T2 (cli.py atualizado)
**Requirement:** SHIMS-06, SHIMS-11

**Done when:**
- [ ] Linha 166 de config.py removida.
- [ ] `from agenticlog.config import DIR_DOCUMENTS` funciona (outros imports intactos).
- [ ] Nenhum consumer importa `_JsonFormatter` de `config.py`.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** none
**Gate:** quick

---

#### T15: Migrar imports de `scripts/rag_eval.py` + `scripts/pdf_to_json.py` (SHIMS-07)

**What:**
`scripts/rag_eval.py:84-89`:
- `from agenticlog.agent import AgentState, _get_retriever, agent_workflow` → `from agenticlog.retrieval.state import AgentState; from agenticlog.retrieval.retriever import _get_retriever; from agenticlog.retrieval.graph import agent_workflow`
- `from agenticlog.rag import _get_rag_embedding_model` → `from agenticlog.ingestion.embeddings import _get_rag_embedding_model`

`scripts/pdf_to_json.py:17`:
- `from agenticlog.rag import RAGSecurityError, extrair_texto_pdf` → `from agenticlog.shared.errors import RAGSecurityError; from agenticlog.ingestion.extraction import extrair_texto_pdf`

**Where:** `scripts/rag_eval.py`, `scripts/pdf_to_json.py` (modificados)
**Depends on:** Fase A completa
**Requirement:** SHIMS-07

**Done when:**
- [ ] `scripts/rag_eval.py` imports atualizados dentro do try/except.
- [ ] `scripts/pdf_to_json.py` imports atualizados.
- [ ] `python scripts/pdf_to_json.py --help` funciona.
- [ ] `ruff`/`black`/`isort` limpos.

**Tests:** none (scripts sao ferramentas auxiliares)
**Gate:** quick

---

#### T16: Atualizar `pyproject.toml` mypy override (SHIMS-18)

**What:** Alterar `pyproject.toml` mypy override:
- ANTES: `module = ["agenticlog.rag", "agenticlog.api", "agenticlog.serving.api"]`
- DEPOIS: `module = ["agenticlog.serving.api"]`

**Where:** `pyproject.toml` (modificado)
**Depends on:** None
**Requirement:** SHIMS-18

**Done when:**
- [ ] mypy override contem so `"agenticlog.serving.api"`.
- [ ] `mypy src/agenticlog/` limpo (ou secoes ignoradas via config).

**Tests:** none
**Gate:** quick

---

### FASE C — Migrar patches de teste + oracle

#### T17: Migrar patches de `tests/test_rag.py` para namespaces canonicos (SHIMS-13)

**What:** Migrar todos os `@patch("agenticlog.rag.X")` e `@patch("agenticlog.agent.X")` em `tests/test_rag.py` para os modulos canonicos.

**Tabela de migracao (por simbolo):**

| Patch ATUAL | Patch DESTINO |
|-------------|--------------|
| `agenticlog.rag.adicionar_documento_incrementalmente` | `agenticlog.ingestion.orchestrator.adicionar_documento_incrementalmente` |
| `agenticlog.rag.adicionar_pdf_incrementalmente` | `agenticlog.ingestion.orchestrator.adicionar_pdf_incrementalmente` |
| `agenticlog.rag._get_rag_embedding_model` | `agenticlog.ingestion.embeddings._get_rag_embedding_model` |
| `agenticlog.rag.ingerir_incrementalmente` | `agenticlog.ingestion.orchestrator.ingerir_incrementalmente` |
| `agenticlog.rag.cria_vectordb` | `agenticlog.ingestion.orchestrator.cria_vectordb` |
| `agenticlog.rag._resetar_colecao` | `agenticlog.ingestion.store._resetar_colecao` |
| `agenticlog.rag._valida_path_documentos` | `agenticlog.ingestion.security._valida_path_documentos` |
| `agenticlog.rag.extrair_texto_pdf` | `agenticlog.ingestion.extraction.extrair_texto_pdf` |
| `agenticlog.rag.RAGSecurityError` | `agenticlog.shared.errors.RAGSecurityError` |
| `agenticlog.rag.invalidar_vector_db` | `agenticlog.retrieval.retriever.invalidar_vector_db` |
| `agenticlog.agent._get_embedding_model` | `agenticlog.retrieval.retriever._get_embedding_model` |
| `agenticlog.agent._embedding_model` | `agenticlog.retrieval.retriever._embedding_model` |
| `agenticlog.agent.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` |
| `agenticlog.agent.invalidar_vector_db` | `agenticlog.retrieval.retriever.invalidar_vector_db` |

**Where:** `tests/test_rag.py` (modificado)
**Depends on:** Fase A + B completas
**Requirement:** SHIMS-13

**Done when:**
- [ ] `rg "@patch\(\"agenticlog\.rag\.\w" tests/test_rag.py` retorna 0 matches.
- [ ] `rg "@patch\(\"agenticlog\.agent\.\w" tests/test_rag.py` retorna 0 matches.
- [ ] `rg "from agenticlog\.rag import|from agenticlog\.agent import" tests/test_rag.py` — se houver imports de topo, manter (virao de ingestion/ ou retrieval/).
- [ ] `pytest tests/test_rag.py -v` verde.
- [ ] Contagem de testes preservada (mesmo numero, sem silent deletions).

**Tests:** unit
**Gate:** quick

---

#### T18: Migrar patches de `tests/test_agentic_rag.py` (SHIMS-13)

**What:** Migrar todos `@patch("agenticlog.agent.X")` para canonicos.

**Tabela de migracao:**

| Patch ATUAL | Patch DESTINO |
|-------------|--------------|
| `agenticlog.agent.retriever` | `agenticlog.retrieval.retriever._get_retriever` |
| `agenticlog.agent.agent_workflow` | `agenticlog.retrieval.graph.agent_workflow` |
| `agenticlog.agent.gera_multiplas_respostas` | `agenticlog.retrieval.generation.gera_multiplas_respostas` |
| `agenticlog.agent.avalia_similaridade` | `agenticlog.retrieval.generation.avalia_similaridade` |
| `agenticlog.agent.rank_respostas` | `agenticlog.retrieval.generation.rank_respostas` |
| `agenticlog.agent._get_llm` | `agenticlog.retrieval.generation._get_llm` |
| `agenticlog.agent._llm` | `agenticlog.retrieval.generation._llm` |
| `agenticlog.agent.search` | `agenticlog.retrieval.graph.search` |
| `agenticlog.agent._get_embedding_model` | `agenticlog.retrieval.retriever._get_embedding_model` |
| `agenticlog.agent._embedding_model` | `agenticlog.retrieval.retriever._embedding_model` |
| `agenticlog.agent.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` |

**Where:** `tests/test_agentic_rag.py` (modificado)
**Depends on:** Fase A + B completas
**Requirement:** SHIMS-13

**Done when:**
- [ ] `rg "@patch\(\"agenticlog\.agent\.\w" tests/test_agentic_rag.py` retorna 0 matches.
- [ ] `pytest tests/test_agentic_rag.py -v` verde.
- [ ] Contagem de testes preservada.

**Tests:** unit
**Gate:** quick

---

#### T19: Reescrever monkeypatches do oraculo `tests/test_rag_caracterizacao.py` (SHIMS-16)

**What:** Reescrever os 8 `monkeypatch.setattr(...)` na fixture `rag_caracterizacao_env` e nos testes que usam `monkeypatch.setattr("agenticlog.agent.search", ...)`:

| monkeypatch ATUAL | monkeypatch DESTINO |
|-------------------|---------------------|
| `agenticlog.rag.DIR_DOCUMENTS` | `agenticlog.config.DIR_DOCUMENTS` |
| `agenticlog.rag.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` |
| `agenticlog.agent.DIR_VECTORDB` | `agenticlog.config.DIR_VECTORDB` |
| `agenticlog.rag._rag_embedding_model` | `agenticlog.ingestion.embeddings._rag_embedding_model` |
| `agenticlog.agent._embedding_model` | `agenticlog.retrieval.retriever._embedding_model` |
| `agenticlog.agent._llm` | `agenticlog.retrieval.generation._llm` |
| `agenticlog.agent._vector_dbs` | `agenticlog.retrieval.retriever._vector_dbs` |
| `agenticlog.agent.search` | `agenticlog.retrieval.graph.search` |

E atualizar os imports de topo do oraculo:
- `from agenticlog.agent import AgentState, agent_workflow` → `from agenticlog.retrieval.state import AgentState; from agenticlog.retrieval.graph import agent_workflow`
- `from agenticlog.rag import adicionar_documento_incrementalmente` → `from agenticlog.ingestion.orchestrator import adicionar_documento_incrementalmente`

**IMPORTANTE:** Nao alterar LOGICA do oraculo (asserts, estrutura de teste, fixtures). So mudar os namespaces dos monkeypatches e imports.

**Where:** `tests/test_rag_caracterizacao.py` (modificado)
**Depends on:** Fase A + B completas
**Requirement:** SHIMS-16

**Done when:**
- [ ] `monkeypatch.setattr("agenticlog.rag.*")` e `monkeypatch.setattr("agenticlog.agent.*")` — 0 ocorrencias.
- [ ] `pytest -m integration tests/test_rag_caracterizacao.py -v` verde no CI Linux.
- [ ] Import de `AgentState`/`agent_workflow` de `agenticlog.retrieval.*`.
- [ ] Import de `adicionar_documento_incrementalmente` de `agenticlog.ingestion.orchestrator`.
- [ ] Nenhuma linha de assert alterada.

**Tests:** integration (marcado @integration)
**Gate:** full

---

#### T20: Migrar patches de `tests/adicionar_pdf_incrementalmente.py` + `tests/acceptance/test_adicionar_pdf_incrementalmente.py` (SHIMS-13)

**What:** Migrar patches de `agenticlog.rag.invalidar_vector_db` e `agenticlog.rag.adicionar_pdf_incrementalmente` e `agenticlog.rag._get_rag_embedding_model` para modulos canonicos (ver tabela no design SS7).

**Where:** `tests/adicionar_pdf_incrementalmente.py`, `tests/acceptance/test_adicionar_pdf_incrementalmente.py` (modificados)
**Depends on:** Fase A + B completas
**Requirement:** SHIMS-13

**Done when:**
- [ ] `rg "@patch\(\"agenticlog\.rag\.\w" tests/adicionar_pdf_incrementalmente.py` 0 matches.
- [ ] `rg "@patch\(\"agenticlog\.rag\.\w" tests/acceptance/test_adicionar_pdf_incrementalmente.py` 0 matches.
- [ ] `pytest tests/adicionar_pdf_incrementalmente.py -v` verde.
- [ ] `pytest tests/acceptance/test_adicionar_pdf_incrementalmente.py -v` verde.

**Tests:** unit
**Gate:** quick

---

#### T21: Migrar patches de acceptance tests (test_api, test_health, test_modo_seguro, test_history, test_query_history)* (SHIMS-13)

**What:** Migrar patches nos acceptance tests que referenciam `agenticlog.rag.*`, `agenticlog.agent.*`, `agenticlog.api.*`, `agenticlog.health.*`, `agenticlog.history.*`. Verificar arquivos:
- `tests/acceptance/test_api.py` — patches `agenticlog.api.X` → `agenticlog.serving.api.X` (JA MIGRADO na Fase 5? Se sim, verificar se `agenticlog.api.*` remanescentes existem)
- `tests/acceptance/test_api_query_endpoint.py` — idem
- `tests/acceptance/test_modo_seguro_modelo_indisponivel.py` — idem
- `tests/acceptance/test_history_endpoint.py` — idem
- `tests/acceptance/test_query_history_audit_logging.py` — idem
- `tests/acceptance/test_health_check.py` — patches `agenticlog.health.*` (shim deletado)

NOTA: Patches `agenticlog.health.httpx.Client` precisam ser migrados para `agenticlog.serving.health.httpx.Client` porque o shim health.py foi deletado. O import `import httpx` agora esta em `serving/health.py`.

**Where:** `tests/acceptance/` (multiplos arquivos)
**Depends on:** Fase A + B completas
**Requirement:** SHIMS-13

**Done when:**
- [ ] `rg "@patch\(\"agenticlog\.(rag|agent|api|health|history)\.\w" tests/acceptance/` retorna 0 matches.
- [ ] `pytest tests/acceptance/ -v` verde (menos os testes removidos em Fase D).

**Tests:** unit/integration
**Gate:** full

---

#### T22: Migrar patches de `tests/ingestion/test_*.py` (SHIMS-13)

**What:** Arquivos `tests/ingestion/test_security.py`, `tests/ingestion/test_metadata.py`, `tests/ingestion/test_extraction.py` fazem `import agenticlog.rag as rag` no topo e usam `rag.X`. Com rag.py deletado, estes imports precisam ser atualizados para importar diretamente dos modulos canonicos (ex.: `from agenticlog.ingestion.security import _valida_path_documentos`).

**Where:** `tests/ingestion/test_security.py`, `tests/ingestion/test_metadata.py`, `tests/ingestion/test_extraction.py` (modificados)
**Depends on:** Fase B completa
**Requirement:** SHIMS-13

**Done when:**
- [ ] `rg "import agenticlog\.rag|from agenticlog\.rag import" tests/ingestion/` retorna 0 matches.
- [ ] `pytest tests/ingestion/ -v` verde.

**Tests:** unit
**Gate:** quick

---

### FASE D — Remover identity tests + atualizar acicidade

#### T23: Remover `TestShimsIdentidade` de `tests/ingestion/test_shims_identidade.py` (SHIMS-14)

**What:** Remover a classe `TestShimsIdentidade` (e seu metodo `teste_1_identidade_de_cada_simbolo_movido`, `teste_2_caminho_de_import_antigo_resolve`). Manter `TestIngestionAcyclic`.

Tambem remover os imports de modulo que so serviam ao TestShimsIdentidade:
- `import agenticlog.rag as rag` (remover)
- Imports de `agenticlog.ingestion.*` (manter se TestIngestionAcyclic usa)

**Where:** `tests/ingestion/test_shims_identidade.py` (modificado)
**Depends on:** Fase C completa (para garantir que nenhum teste quebrado usa identidade)
**Requirement:** SHIMS-14

**Done when:**
- [ ] `TestShimsIdentidade` removido.
- [ ] `TestIngestionAcyclic` continua presente e verde.
- [ ] `pytest tests/ingestion/test_shims_identidade.py -v` verde.

**Tests:** unit
**Gate:** quick

---

#### T24: Remover `TestServingShimsIdentidade` de `tests/acceptance/test_rag_serving_fase5.py` (SHIMS-14)

**What:** Remover a classe `TestServingShimsIdentidade` (7 testes). Manter `TestServingAcyclic`.

**Where:** `tests/acceptance/test_rag_serving_fase5.py` (modificado)
**Depends on:** Fase C completa
**Requirement:** SHIMS-14

**Done when:**
- [ ] `TestServingShimsIdentidade` removido.
- [ ] `TestServingAcyclic` continua presente e verde.
- [ ] `pytest tests/acceptance/test_rag_serving_fase5.py -v` verde.

**Tests:** unit
**Gate:** quick

---

#### T25: Remover `TestShimIdentity` de `tests/acceptance/test_rag_retrieval_fase4.py` (SHIMS-14)

**What:** Remover a classe `TestShimIdentity` (1 teste parametrizado). Manter `TestAcyclicImport`, `TestWrapperDelegation`.

**Where:** `tests/acceptance/test_rag_retrieval_fase4.py` (modificado)
**Depends on:** Fase C completa
**Requirement:** SHIMS-14

**Done when:**
- [ ] `TestShimIdentity` removido.
- [ ] `TestAcyclicImport` e `TestWrapperDelegation` continuam presentes e verdes.
- [ ] `pytest tests/acceptance/test_rag_retrieval_fase4.py -v` verde.

**Tests:** unit
**Gate:** quick

---

#### T26: Atualizar `TestAcyclicImport.MODULES` — remover `agenticlog.agent` (SHIMS-15)

**What:** Em `tests/acceptance/test_rag_retrieval_fase4.py:TestAcyclicImport`:
1. Remover `"agenticlog.agent"` da lista `MODULES`.
2. Em `test_ac02_import_todos_frio`, remover `import agenticlog.agent; ` da string de codigo.

**Where:** `tests/acceptance/test_rag_retrieval_fase4.py` (modificado, ln 33-39 e ln 62-68)
**Depends on:** Fase E (shims deletados) — so faz sentido apos agent.py deletado
**Requirement:** SHIMS-15

**Done when:**
- [ ] `MODULES` nao contem `"agenticlog.agent"`.
- [ ] `test_ac02_import_todos_frio` nao importa `agenticlog.agent`.
- [ ] `pytest tests/acceptance/test_rag_retrieval_fase4.py::TestAcyclicImport -v` verde.

**Tests:** unit
**Gate:** quick

---

### FASE E — DELETAR SHIMS (ULTIMO passo)

#### T27: Deletar `src/agenticlog/rag.py` (SHIMS-01)

**What:** Deletar o arquivo `src/agenticlog/rag.py` inteiro.

**Pre-condicao:** Nenhum consumer importa de `agenticlog.rag` (verificado por `rg "from agenticlog\.rag import|import agenticlog\.rag" src/ scripts/`).

**Where:** `src/agenticlog/rag.py` (deletado)
**Depends on:** Fases A, B, C, D completas
**Requirement:** SHIMS-01

**Done when:**
- [ ] `rag.py` nao existe mais.
- [ ] `python -c "import agenticlog.rag"` falha com `ModuleNotFoundError`.
- [ ] `rg "from agenticlog\.rag import|import agenticlog\.rag" src/ scripts/` — 0 matches (shims ja deletados nao contam).

**Tests:** none (arquivo deletado)
**Gate:** full (validado por T32)

---

#### T28: Deletar `src/agenticlog/agent.py` (SHIMS-02)

**What:** Deletar o arquivo `src/agenticlog/agent.py` inteiro.

**Pre-condicao:** Nenhum consumer importa de `agenticlog.agent`.

**Where:** `src/agenticlog/agent.py` (deletado)
**Depends on:** Fases A, B, C, D completas
**Requirement:** SHIMS-02

**Done when:**
- [ ] `agent.py` nao existe mais.
- [ ] `python -c "import agenticlog.agent"` falha com `ModuleNotFoundError`.
- [ ] `rg "from agenticlog\.agent import|import agenticlog\.agent" src/ scripts/` — 0 matches.

**Tests:** none (arquivo deletado)
**Gate:** full (validado por T32)

---

#### T29: Deletar `src/agenticlog/api.py` (SHIMS-03)

**What:** Deletar o arquivo `src/agenticlog/api.py` inteiro.

**Pre-condicao:** `main_api.py` atualizado (T11), `app.py:27` atualizado (T10).

**Where:** `src/agenticlog/api.py` (deletado)
**Depends on:** T10, T11
**Requirement:** SHIMS-03

**Done when:**
- [ ] `api.py` nao existe mais.
- [ ] `python -c "import agenticlog.api"` falha com `ModuleNotFoundError`.

**Tests:** none
**Gate:** full (validado por T32)

---

#### T30: Deletar `src/agenticlog/health.py` (SHIMS-04)

**What:** Deletar o arquivo `src/agenticlog/health.py` inteiro.

**Pre-condicao:** `src/agenticlog/__init__.py` atualizado (T12), tests/acceptance patches migrados (T21).

**Where:** `src/agenticlog/health.py` (deletado)
**Depends on:** T12, T21
**Requirement:** SHIMS-04

**Done when:**
- [ ] `health.py` nao existe mais.
- [ ] `python -c "from agenticlog.health import check_lmstudio_health"` falha com `ModuleNotFoundError`.

**Tests:** none
**Gate:** full (validado por T32)

---

#### T31: Deletar `src/agenticlog/history.py` (SHIMS-05)

**What:** Deletar o arquivo `src/agenticlog/history.py` inteiro.

**Pre-condicao:** `serving/api.py` atualizado (T13).

**Where:** `src/agenticlog/history.py` (deletado)
**Depends on:** T13
**Requirement:** SHIMS-05

**Done when:**
- [ ] `history.py` nao existe mais.
- [ ] `python -c "from agenticlog.history import HistoryStore"` falha com `ModuleNotFoundError`.

**Tests:** none
**Gate:** full (validado por T32)

---

### FASE F — Gate final

#### T32: Gate final — suite completa + lint + verificacao de import (SHIMS-18)

**What:** Rodar suite completa, lint, mypy, e varredura de imports remanescentes.

**Where:** todo o repo
**Depends on:** T27, T28, T29, T30, T31

**Done when:**
- [ ] `pytest --cov=agenticlog --cov-report=term-missing -v` verde no CI Linux; cobertura >= 80%.
- [ ] `ruff check --diff .` limpo.
- [ ] `mypy src/agenticlog/` limpo.
- [ ] `rg "# Re-export shim \(ADR-018" src/` retorna 0 matches.
- [ ] `rg "from agenticlog\.rag import|from agenticlog\.agent import|from agenticlog\.api import|from agenticlog\.health import|from agenticlog\.history import" src/ scripts/` — 0 matches.
- [ ] `rg "import agenticlog\.rag|import agenticlog\.agent|import agenticlog\.api|import agenticlog\.health|import agenticlog\.history" src/ scripts/` — 0 matches.
- [ ] `python -c "import agenticlog"` exit 0, sem "circular" ou "partially initialized".
- [ ] `python -m agenticlog.ingestion --help` exit 0.
- [ ] `code-review` executado antes do merge.

**Commit:** `refactor(rag): remove shims de compatibilidade ADR-018 Fase 6`

**Tests:** integration (full suite)
**Gate:** build

---

## Parallel Execution Map

```
Fase A (Sequencial — singletons devem ser realocados em ordem):
  T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9

Fase B (Paralelo apos T9):
  T9 completo, entao:
    ├── T10 (app.py)
    ├── T11 (main_api.py)
    ├── T12 (__init__.py)
    ├── T13 (serving/api.py)
    ├── T14 (config.py:166)
    ├── T15 (scripts/)
    └── T16 (pyproject.toml)

Fase C (Paralelo apos Fase B):
    ├── T17 (test_rag.py)
    ├── T18 (test_agentic_rag.py)
    ├── T19 (oracle monkeypatches)
    ├── T20 (adicionar_pdf tests)
    ├── T21 (acceptance tests)
    └── T22 (ingestion tests)

Fase D (Paralelo apos Fase C):
    ├── T23 (remove TestShimsIdentidade)
    ├── T24 (remove TestServingShimsIdentidade)
    ├── T25 (remove TestShimIdentity)
    └── T26 (update acicidade MODULES)

Fase E (Sequencial — shims deletados um a um):
  T27 → T28 → T29 → T30 → T31

Fase F (Sequencial):
  T31 → T32
```

**Paralelismo constraint:** Tasks marcadas `[P]` sao paralelizaveis. Fases B, C, D tem tasks paralelizaveis por serem independentes (modificam arquivos diferentes). Fase E e sequencial por seguranca (deletar shims e destrutivo).

---

## Task Granularity Check

| Task | Scope | Status |
|------|-------|--------|
| T1: Unificar _get_rag_embedding_model | 1 file, 1 function | Granular |
| T2: Mover CLI para cli.py | 1 file, 2 functions | Granular |
| T3: Garantir import fitz | 1 import line | Granular |
| T4: Realocar globals retriever.py | 1 file, 4 symbols | Granular |
| T5: Realocar _llm generation.py | 1 file, 2 symbols | Granular |
| T6: Realocar search graph.py | 1 file, 1 symbol | Granular |
| T7: Side-effects __init__.py | 1 file, 3 lines | Granular |
| T8: invalidar_vector_db local | 1 function change | Granular |
| T9: orchestrator lazy import | 1 line change | Granular |
| T10: app.py imports | 1 file, 3 import groups | Granular |
| T11: main_api.py string | 1 line change | Granular |
| T12: __init__.py imports | 1 file, 2 import groups | Granular |
| T13: serving/api.py imports | 1 file, 2 import changes | Granular |
| T14: config.py:166 remove | 1 line delete | Granular |
| T15: scripts imports | 2 files | -- 2 files, but each is simple import change |
| T16: pyproject.toml | 1 line change | Granular |
| T17: test_rag.py patches | 1 file, ~15 patches | -- many patches, but same file |
| T18: test_agentic_rag.py patches | 1 file, ~10 patches | Granular |
| T19: oracle monkeypatches | 1 file, 8 changes | Granular |
| T20: adicionar_pdf tests patches | 2 files, ~5 patches | Granular |
| T21: acceptance tests patches | ~6 files | Multiple files, but mechanical |
| T22: ingestion tests patches | 3 files | Granular |
| T23: remove TestShimsIdentidade | 1 file | Granular |
| T24: remove TestServingShimsIdentidade | 1 file | Granular |
| T25: remove TestShimIdentity | 1 file | Granular |
| T26: update acicidade MODULES | 1 file, 2 changes | Granular |
| T27: delete rag.py | 1 file delete | Granular |
| T28: delete agent.py | 1 file delete | Granular |
| T29: delete api.py | 1 file delete | Granular |
| T30: delete health.py | 1 file delete | Granular |
| T31: delete history.py | 1 file delete | Granular |
| T32: gate final | suite completa | Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram Shows | Status |
|------|-------------------|---------------|--------|
| T1 | None | T1 (inicio) | Match |
| T2 | None | T2 (independente de T1) | -- T2 depende de T1 via import fitz? Na verdade T2 e independente. Diagrama corrigido: T2 paralelo a T1. |
| T3 | None | T3 (independente) | Match |
| T4 | None | T4 (independente) | Match |
| T5 | None | T5 (independente) | Match |
| T6 | None | T6 (independente) | Match |
| T7 | None | T7 (independente) | Match |
| T8 | T4 | T8 -> T4 | Match |
| T9 | T8 | T9 -> T8 | Match |
| T10 | Fase A | T9 -> T10 | Match |
| T11 | None | T11 independente na Fase B | Match |
| T12 | Fase A | T9 -> T12 | Match |
| T13 | T1 | T9, T1 -> T13 | Match |
| T14 | T2 | T9 -> T14 | Match |
| T15 | Fase A | T9 -> T15 | Match |
| T16 | None | T16 independente | Match |
| T17..T22 | Fase B | Fase B -> Fase C | Match |
| T23..T26 | Fase C | Fase C -> Fase D | Match |
| T27..T31 | Fase D | Fase D -> Fase E | Match |
| T32 | T27..T31 | T31 -> T32 | Match |

**Correcao:** T1, T2, T3 sao independentes entre si (singletons de rag.py vs CLI vs import fitz). Ajustar diagrama para refletir: T1, T2, T3 podem rodar em paralelo. T4, T5, T6, T7 tambem independentes entre si. Mas para seguranca (minimizar erros humanos), manter Fase A sequencial melhora a rastreabilidade. O custo de paralelizar e baixo vs o risco de esquecer uma dependencia.

---

## Test Co-location Validation

| Task | Code Layer | Matrix Requires | Task Says | Status |
|------|-----------|-----------------|-----------|--------|
| T1 | ingestion/embeddings.py | unit | "validado por T17, T19" | -- Teste adiado, mas T17/T19 cobrem. OK (deferimento minimo). |
| T2 | ingestion/cli.py | unit | "validado por T17, T32" | OK (T32 valida via suite completa) |
| T3 | ingestion/extraction.py | unit | "none" | -- Nao requer teste proprio (so adiciona import) |
| T4 | retrieval/retriever.py | unit | "validado por T17, T18, T19" | OK |
| T5 | retrieval/generation.py | unit | "validado por T18, T19" | OK |
| T6 | retrieval/graph.py | unit | "validado por T19" | OK |
| T7 | retrieval/__init__.py | none | "none" | OK |
| T8 | retrieval/retriever.py | unit | "validado por T17, T20" | OK |
| T9 | ingestion/orchestrator.py | unit | "validado por T32" | OK |
| T10 | app.py | none | "none" | OK (Streamlit sem cobertura) |
| T11 | main_api.py | none | "none" | OK |
| T12 | __init__.py | none | "validado por T32" | OK |
| T13 | serving/api.py | unit | "validado por T32" | OK |
| T14 | config.py | none | "none" | OK |
| T15 | scripts/ | none | "none" | OK |
| T16 | pyproject.toml | none | "none" | OK |
| T17 | test_rag.py | unit | "unit" | OK |
| T18 | test_agentic_rag.py | unit | "unit" | OK |
| T19 | test_rag_caracterizacao.py | integration | "integration" | OK |
| T20 | adicionar_pdf tests | unit | "unit" | OK |
| T21 | acceptance tests | unit/integration | "unit/integration" | OK |
| T22 | ingestion tests | unit | "unit" | OK |
| T23..T26 | test deletes/updates | unit | "unit" | OK |
| T27..T31 | file deletes | none | "none" | OK (deletar arquivo nao requer teste) |
| T32 | full suite | build | "integration" | OK |

---

## Notas de teste / gate (de CLAUDE.md)
- Sempre mockar LLM; sempre testar retrieval vazio.
- Nomes de teste com prefixo `teste_N_` (dominio) / `test_` (erro/seguranca) / `test_acNN_` (acceptance).
- Gate autoritativo: CI Linux `pytest --cov=agenticlog --cov-report=term-missing -v`.
- Commits Conventional em PT (`refactor:`); branch ja criada (`feature/rag-fase6-shims-remocao`); code-review antes do merge.
- `ruff`/`black`/`isort` limpos; type hints em todas as assinaturas; docstrings PT (Entrada/Saida/Lanca); imutabilidade; funcoes <50 ln.
- Nao usar `git add -A` (evita incluir PDFs untracked de data/documents/). Usar `git add <caminhos explicitos>`.
- Atualizar CLAUDE.md com novos entrypoints e remover secoes sobre rag.py/agent.py.
