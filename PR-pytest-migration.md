# PR: feature/pytest-migration – Migração para pytest + cobertura

## Resumo

Migra o runner de testes de `unittest` para **pytest** e adiciona **cobertura de código** (coverage) ao CI. Os testes existentes permanecem compatíveis (pytest executa testes unittest nativamente).

## Objetivo

- Padronizar o uso de pytest como framework de testes
- Gerar relatório de cobertura no CI e localmente
- Preparar para integração futura (ex.: Codecov, badges)
- Manter compatibilidade com os testes atuais (sem reescrita)

## Solução

1. **Dependências**: `requirements-dev.txt` com pytest e pytest-cov (inclui `-r requirements.txt`)
2. **Configuração**: `pyproject.toml` com `[tool.pytest.ini_options]` e `[tool.coverage.run]`
3. **CI**: Workflow atualizado para `pytest --cov=agenticlog --cov-report=term-missing --cov-report=xml`
4. **conftest.py**: Garante `src` no path antes dos testes

---

## Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `requirements-dev.txt` | **Novo** – dependências de dev (pytest, pytest-cov) |
| `pyproject.toml` | Adicionados `[tool.pytest.ini_options]` e `[tool.coverage.run/report]` |
| `.github/workflows/test.yml` | Usa `requirements-dev.txt` e `pytest` com coverage |
| `tests/conftest.py` | **Novo** – configuração pytest (path para `src`) |
| `.gitignore` | Adicionados `.coverage`, `htmlcov/` para artefatos de coverage |

---

## Mensagem de commit sugerida (Conventional Commits)

```
test: migrate to pytest with coverage

- Add pytest and pytest-cov via requirements-dev.txt
- Configure pytest and coverage in pyproject.toml
- Update CI to run pytest with coverage report
- Add conftest.py for test path setup
```

---

## Checklist do PR

- [ ] Branch criada: `feature/pytest-migration`
- [ ] Arquivos adicionados/modificados conforme tabela acima
- [ ] Testes executados localmente: `pytest --cov=agenticlog --cov-report=term-missing -v`
- [ ] CI verde após push (verificar na aba Actions do GitHub)
- [ ] `git pull origin main` após merge (conforme Protocolo de Git)

---

## Comandos para abrir o PR

```bash
git checkout -b feature/pytest-migration
git add requirements-dev.txt pyproject.toml .github/workflows/test.yml tests/conftest.py
git commit -m "test: migrate to pytest with coverage"
git push -u origin feature/pytest-migration
```

Depois, abrir o PR no GitHub da branch `feature/pytest-migration` para `main`.

---

## Comandos úteis

```bash
# Executar testes
pytest -v

# Executar com cobertura (terminal)
pytest --cov=agenticlog --cov-report=term-missing -v

# Executar com cobertura (HTML)
pytest --cov=agenticlog --cov-report=html -v
# Abrir htmlcov/index.html no navegador
```

---

## Observações

- **Compatibilidade**: pytest executa testes unittest sem alterações. Os métodos `teste_*` e `test_*` são reconhecidos pela config `python_functions = ["test_*", "teste_*"]`.
- **Coverage**: O relatório cobre `src/agenticlog`. O XML (`--cov-report=xml`) permite integração futura com Codecov ou similar.
- **CI**: O cache de pip inclui `requirements-dev.txt` no hash para invalidar quando dependências de dev mudarem.
