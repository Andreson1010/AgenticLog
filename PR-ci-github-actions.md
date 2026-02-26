# PR: feature/ci-github-actions – Workflow de testes no GitHub

## Resumo

Adiciona um workflow de CI (GitHub Actions) que executa os testes unitários automaticamente em cada push e pull request para as branches `main` e `master`.

## Objetivo

- Garantir que os testes passem antes de merge
- Detectar regressões em pull requests
- Padronizar o ambiente de execução dos testes (Ubuntu, Python 3.11)

## Solução

Workflow `.github/workflows/test.yml` que:

1. **Trigger**: Executa em `push` e `pull_request` para `main` e `master`
2. **Ambiente**: Ubuntu latest, Python 3.11
3. **Cache**: Cache de pacotes pip para acelerar builds
4. **Setup**: Instala dependências (`requirements.txt` + editable install)
5. **Diretórios**: Cria `data/documents` e `data/vectordb` (exigidos pelo Chroma)
6. **Testes**: Executa `python -m unittest discover -s tests -v`

---

## Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `.github/workflows/test.yml` | **Novo** – workflow de CI para testes |

---

## Mensagem de commit sugerida (Conventional Commits)

```
ci: add GitHub Actions workflow for tests

Runs unit tests on push and pull_request to main/master.
Uses Python 3.11, cache for pip, and unittest discover.
```

---

## Checklist do PR

- [ ] Branch criada: `feature/ci-github-actions`
- [ ] Arquivo `.github/workflows/test.yml` adicionado
- [ ] Testes executados localmente (requer ambiente com dependências)
- [ ] CI verde após push (verificar na aba Actions do GitHub)
- [ ] `git pull origin main` após merge (conforme Protocolo de Git)

---

## Comandos para abrir o PR

```bash
git checkout -b feature/ci-github-actions
git add .github/workflows/test.yml
git commit -m "ci: add GitHub Actions workflow for tests"
git push -u origin feature/ci-github-actions
```

Depois, abrir o PR no GitHub da branch `feature/ci-github-actions` para `main`.

---

## Observações

- **Dependência do fix/agent-llm-import**: O CI só funcionará se o `llm.invoke()` em tempo de importação tiver sido removido do `agent.py`. Caso contrário, os testes falharão por falta de API key ou rede.
- **Tempo de build**: A instalação pode levar vários minutos devido a `torch`, `sentence-transformers` e outras dependências pesadas. O cache de pip ajuda em builds subsequentes.
- **Matrix**: O workflow está configurado com `python-version: ["3.11"]` para compatibilidade com `pyproject.toml (requires-python = ">=3.11")`. Se precisar testar em múltiplas versões, basta ajustar o array.
