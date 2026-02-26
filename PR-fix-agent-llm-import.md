# PR: fix/agent-llm-import – Corrigir llm.invoke na importação (bloqueia CI)

## Resumo

Remove a chamada `llm.invoke()` executada em **tempo de importação** do módulo `agent.py`, que bloqueava o CI ao exigir conexão com a API do LLM (chave, rede, etc.) sempre que o pacote era importado.

## Problema

O bloco abaixo era executado ao importar `agenticlog`:

```python
try:
    response = llm.invoke("Just respond with the word: CONNECTED")
    print(f"Status do {LLM_MODEL}: {response.content}")
except Exception as e:
    print(f"Connection error: {e}")
```

**Impacto:**
- Testes (`from agenticlog.agent import ...`) falhavam no CI por falta de API key ou rede
- Qualquer import do pacote (ex.: `app.py`, testes) tentava conectar ao LLM
- CI bloqueado mesmo para testes unitários que usam mocks

## Solução

Remoção do bloco de teste de conexão em tempo de importação. O LLM continua sendo instanciado normalmente; apenas o teste de conectividade foi removido.

---

## Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `src/agenticlog/agent.py` | Removido bloco `try/except` com `llm.invoke()` (linhas 46–50) |

---

## Mensagem de commit sugerida (Conventional Commits)

```
fix: remove llm.invoke na importação para desbloquear CI

O teste de conexão com o LLM em tempo de importação exigia API key
e rede, bloqueando testes e CI. O LLM continua instanciado; apenas
o health check na importação foi removido.
```

---

## Checklist do PR

- [x] Branch criada: `fix/agent-llm-import`
- [x] Alteração aplicada em `src/agenticlog/agent.py`
- [ ] Testes executados localmente (requer ambiente com dependências)
- [ ] CI verde após merge
- [ ] `git pull origin main` após merge (conforme Protocolo de Git)

---

## Comandos para abrir o PR

```bash
git add src/agenticlog/agent.py
git commit -m "fix: remove llm.invoke na importação para desbloquear CI"
git push -u origin fix/agent-llm-import
```

Depois, abrir o PR no GitHub da branch `fix/agent-llm-import` para `main`.

---

## Observações

- **Nenhuma alteração** em `requirements.txt`, `app.py` ou testes
- O bloco `try/except` em `usar_ferramenta_web` (linha ~137) permanece para capturar erros de rede na busca web
- Se for necessário um health check do LLM no futuro, deve ser feito sob demanda (ex.: endpoint `/health` ou comando CLI), não na importação
