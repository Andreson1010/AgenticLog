# Lazy LLM Initialization Specification

## Problem Statement

`agent.py` inicializa o cliente LMStudio e o ChromaDB no nível de módulo (linhas 44-58). Se o LMStudio não estiver rodando ou `data/vectordb/` não existir, o `import` falha e o app inteiro crasha antes de o usuário ver qualquer UI. Isso transforma uma dependência de runtime em uma dependência de importação, tornando testes mais frágeis e o app menos resiliente.

## Goals

- [ ] Importar `agenticlog` sem LMStudio rodando e sem vectordb presente
- [ ] Inicializar LLM e ChromaDB apenas no primeiro uso (lazy)
- [ ] Exibir mensagem de erro clara no Streamlit quando LMStudio estiver indisponível
- [ ] Manter compatibilidade total com os mocks dos testes existentes

## Out of Scope

| Feature | Reason |
|---|---|
| Retry logic / circuit breaker | Feature separada no backlog |
| Health check endpoint | Fora do escopo atual (sem API REST) |
| Fallback para outro modelo LLM | Decisão de produto não tomada |
| Configuração de timeout | Melhoria incremental futura |

---

## User Stories

### P1: App inicia sem LMStudio ⭐ MVP

**User Story**: Como desenvolvedor, quero que `streamlit run app.py` suba mesmo com LMStudio desligado, para poder inspecionar a UI e depurar sem precisar do modelo carregado.

**Why P1**: Crash no import é o bloqueador mais crítico — impede qualquer uso do app sem setup completo.

**Acceptance Criteria**:

1. WHEN `import agenticlog` é executado com LMStudio offline THEN o módulo SHALL importar sem exceção
2. WHEN `streamlit run app.py` é executado com LMStudio offline THEN a UI SHALL carregar e exibir o campo de query
3. WHEN o usuário submete uma query com LMStudio offline THEN o app SHALL exibir mensagem de erro amigável (não stack trace)
4. WHEN o usuário submete uma query com LMStudio online THEN o workflow SHALL funcionar identicamente ao comportamento atual

**Independent Test**: Parar LMStudio → `streamlit run app.py` → página carrega → submeter query → ver mensagem de erro (não crash).

---

### P2: App inicia sem vectordb

**User Story**: Como desenvolvedor, quero que o app suba mesmo sem `data/vectordb/` presente, para que a mensagem de erro oriente a rodar `python -m agenticlog.rag` antes de usar o sistema.

**Why P2**: Complementa P1 — os dois recursos são inicializados no mesmo bloco problemático.

**Acceptance Criteria**:

1. WHEN `data/vectordb/` não existe THEN `import agenticlog` SHALL importar sem exceção
2. WHEN o usuário submete uma query sem vectordb THEN o app SHALL exibir mensagem orientando a rodar `python -m agenticlog.rag`
3. WHEN vectordb existe e LMStudio está online THEN o comportamento SHALL ser idêntico ao atual

**Independent Test**: Remover `data/vectordb/` → `streamlit run app.py` → submeter query → ver mensagem orientando rebuild.

---

### P3: Testes independentes de LMStudio

**User Story**: Como desenvolvedor, quero rodar `pytest` sem LMStudio instalado ou rodando, para que o CI passe sem dependências de runtime externas.

**Why P3**: Os testes já mocam tudo, mas o import do módulo ainda falha se LMStudio estiver offline — lazy init resolve isso como efeito colateral.

**Acceptance Criteria**:

1. WHEN `pytest tests/` é executado com LMStudio offline THEN todos os testes existentes SHALL passar
2. WHEN novos testes para lazy init são adicionados THEN eles SHALL usar `unittest.mock.patch` sem precisar de LMStudio real

**Independent Test**: Desligar LMStudio → `pytest tests/ -v` → todos os testes passam.

---

## Edge Cases

- WHEN LMStudio fica offline após inicialização bem-sucedida THEN o próximo invoke SHALL falhar com erro amigável (não crash do processo)
- WHEN `data/vectordb/` existe mas está corrompido THEN o app SHALL exibir erro claro (não silenciar a exceção)
- WHEN lazy init é chamado concorrentemente (múltiplos usuários no Streamlit) THEN a inicialização SHALL ocorrer apenas uma vez (thread-safe)

---

## Requirement Traceability

| Requirement ID | Story | Status |
|---|---|---|
| LAZY-01 | P1: import sem LMStudio offline | Pending |
| LAZY-02 | P1: UI carrega sem LMStudio | Pending |
| LAZY-03 | P1: erro amigável ao submeter query offline | Pending |
| LAZY-04 | P1: comportamento inalterado quando online | Pending |
| LAZY-05 | P2: import sem vectordb | Pending |
| LAZY-06 | P2: erro orientando rebuild do vectordb | Pending |
| LAZY-07 | P3: pytest passa sem LMStudio | Pending |

---

## Success Criteria

- [ ] `streamlit run app.py` sobe com LMStudio offline (P1)
- [ ] Query com LMStudio offline exibe mensagem de erro (não crash) (P1)
- [ ] `pytest tests/ -v` passa com LMStudio offline (P3)
- [ ] Todos os testes existentes continuam passando após refactor (P1+P3)
- [ ] Cobertura >= 80% mantida após mudanças (gate: `pytest --cov=agenticlog --cov-report=term-missing -v`)
