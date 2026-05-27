# Lazy LLM Initialization — Tasks

**Spec**: `.specs/features/lazy-llm-init/spec.md`
**Status**: Done

---

## Execution Plan

### Phase 1: Foundation (Sequential)

```
T1
```

### Phase 2: Error Messages (Sequential)

```
T1 → T2
```

---

## Task Breakdown

### T1: Refatorar inicialização lazy em `agent.py`

**What**: Substituir init de módulo de `llm` e `vector_db` (linhas 44-58) por getters `_get_llm()` e `_get_vector_db()`; atualizar todos os nodes que referenciam essas variáveis; atualizar mocks existentes e adicionar testes unitários para o comportamento lazy.
**Where**: `src/agenticlog/agent.py`, `tests/test_agentic_rag.py`
**Depends on**: None
**Reuses**: Padrão de mock existente com `@patch` + `MagicMock` em `tests/test_agentic_rag.py`
**Requirements**: LAZY-01, LAZY-04, LAZY-05, LAZY-07

**Approach**:
```python
# Antes (falha no import)
llm = ChatOpenAI(base_url=OPENAI_API_BASE, ...)
vector_db = Chroma(persist_directory=DIR_VECTORDB, ...)

# Depois (lazy)
_llm = None
_vector_db = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(base_url=OPENAI_API_BASE, ...)
    return _llm

def _get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = Chroma(persist_directory=str(DIR_VECTORDB), ...)
    return _vector_db
```
Cada node usa `_get_llm()` / `_get_vector_db()` em vez das variáveis de módulo.

**Tools**:
- MCP: None
- Skill: None

**Done when**:
- [ ] `_llm` e `_vector_db` são `None` no nível de módulo — nenhuma inicialização no import
- [ ] `_get_llm()` e `_get_vector_db()` existem e inicializam na primeira chamada
- [ ] Todos os nodes (`retrieve_info`, `gera_multiplas_respostas`, `usar_ferramenta_web`) usam getters
- [ ] `import agenticlog.agent` não falha com LMStudio offline
- [ ] `import agenticlog.agent` não falha sem `data/vectordb/`
- [ ] Testes existentes em `test_agentic_rag.py` continuam passando (mocks atualizados se necessário)
- [ ] Novos testes cobrem: import sem LMStudio, getter retorna mesma instância em chamadas subsequentes
- [ ] Gate check passa: `pytest --cov=agenticlog --cov-report=term-missing -v`
- [ ] Test count: >= número anterior de testes + novos testes adicionados (sem deleção silenciosa)

**Tests**: unit
**Gate**: full

**Verify**:
```bash
# Sem LMStudio rodando:
python -c "import agenticlog.agent; print('OK')"
# Esperado: OK (sem exception)

pytest --cov=agenticlog --cov-report=term-missing -v
# Esperado: todos os testes passam, cobertura >= 80%
```

**Commit**: `refactor(agent): inicialização lazy de LLM e ChromaDB`

---

### T2: Melhorar mensagens de erro em `app.py`

**What**: Diferenciar os erros de "LMStudio offline" e "vectordb ausente" no bloco `try/except` existente (linha 66), exibindo mensagens amigáveis com orientação de ação para o usuário.
**Where**: `app.py`
**Depends on**: T1
**Reuses**: Bloco `try/except` existente em `app.py` linha 66; padrão `st.error()` / `st.warning()` já usado no arquivo
**Requirements**: LAZY-02, LAZY-03, LAZY-06

**Approach**:
```python
# Capturar tipos específicos de exceção e exibir mensagem direcionada:
# httpx.ConnectError / ConnectionRefusedError → LMStudio offline
# Exception com "does not exist" / "no such file" → vectordb ausente
# Qualquer outro → mensagem genérica (comportamento atual)
```

**Tools**:
- MCP: None
- Skill: None

**Done when**:
- [ ] Query com LMStudio offline exibe mensagem: "LMStudio não está rodando. Inicie o LMStudio e carregue o modelo hermes-3-llama-3.2-3b."
- [ ] Query sem `data/vectordb/` exibe mensagem: "Base vetorial não encontrada. Execute: python -m agenticlog.rag"
- [ ] Nenhum stack trace é exibido ao usuário final em nenhum dos dois casos
- [ ] Comportamento com LMStudio online e vectordb presente é idêntico ao anterior
- [ ] Gate check passa: `pytest tests/ -v --tb=short`

**Tests**: none (app.py sem cobertura de testes — confirmado em TESTING.md)
**Gate**: quick

**Verify**:
```bash
# Teste manual — parar LMStudio, rodar app, submeter query
# Esperado: mensagem de erro amigável, sem stack trace

pytest tests/ -v --tb=short
# Esperado: todos os testes existentes passam
```

**Commit**: `fix(app): mensagens de erro amigáveis para LMStudio offline e vectordb ausente`

---

## Parallel Execution Map

```
Phase 1 (Sequential):
  T1

Phase 2 (Sequential — T2 depende de T1):
  T1 → T2
```

Nenhuma tarefa marcada `[P]` — T2 depende de T1 e o conjunto é pequeno (2 tarefas).

---

## Task Granularity Check

| Task | Scope | Status |
|---|---|---|
| T1: Lazy init em agent.py + testes | 1 arquivo de código + 1 arquivo de testes (coesas) | ✅ Granular |
| T2: Mensagens de erro em app.py | 1 arquivo, 1 bloco try/except | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagrama mostra | Status |
|---|---|---|---|
| T1 | None | T1 sem predecessores | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |

---

## Test Co-location Validation

| Task | Camada criada/modificada | Matrix requer | Task diz | Status |
|---|---|---|---|---|
| T1 | `agent.py` nodes (unit) | unit | unit | ✅ OK |
| T1 | `test_agentic_rag.py` | — | testes incluídos na task | ✅ OK |
| T2 | `app.py` | none (não testado per TESTING.md) | none | ✅ OK |

---

## Requirement Traceability (atualizada)

| Requirement ID | Story | Task | Status |
|---|---|---|---|
| LAZY-01 | P1: import sem LMStudio offline | T1 | Verified |
| LAZY-02 | P1: UI carrega sem LMStudio | T2 | Verified |
| LAZY-03 | P1: erro amigável ao submeter query offline | T2 | Verified |
| LAZY-04 | P1: comportamento inalterado quando online | T1 | Verified |
| LAZY-05 | P2: import sem vectordb | T1 | Verified |
| LAZY-06 | P2: erro orientando rebuild do vectordb | T2 | Verified |
| LAZY-07 | P3: pytest passa sem LMStudio | T1 | Verified |

**Coverage**: 7 requirements, 7 mapeados para tasks, 0 não mapeados ✅
