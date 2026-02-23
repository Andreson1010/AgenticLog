# pyproject.toml — O que é e o que definir

## Objetivo

O `pyproject.toml` é o arquivo de configuração padrão para projetos Python (PEP 518, 621). Ele centraliza metadados do projeto, dependências, requisitos de Python e configurações de ferramentas em um único arquivo, substituindo `setup.py`, `setup.cfg` e partes de outros arquivos de configuração.

---

## O que deve ser definido

### 1. Metadados do projeto (`[project]`)

| Campo | Obrigatório | Descrição |
|-------|-------------|-----------|
| `name` | Sim | Nome do pacote (usado no PyPI, `pip install`) |
| `version` | Sim | Versão do projeto (ex.: `"0.1.0"`, `"1.2.3"`) |
| `description` | Recomendado | Breve descrição do projeto |
| `readme` | Recomendado | Arquivo README (ex.: `"LEIAME.md"`) |
| `requires-python` | Recomendado | Versão mínima do Python (ex.: `">=3.11"`) |
| `license` | Opcional | Licença (ex.: `"MIT"`) |
| `authors` | Opcional | Lista de autores |
| `keywords` | Opcional | Palavras-chave para descoberta |
| `classifiers` | Opcional | Classificadores PyPI (ex.: `"Programming Language :: Python :: 3.12"`) |

### 2. Dependências

```toml
[project]
dependencies = [
    "fastapi>=0.100.0",
    "streamlit>=1.28.0",
]
```

- Dependências de produção listadas aqui
- Pode usar `requirements.txt` em paralelo (ex.: para versões fixas)

### 3. Dependências de desenvolvimento (`[project.optional-dependencies]` ou `[tool.uv]`)

```toml
[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "ruff>=0.1.0",
]
```

### 4. Configurações de ferramentas

Cada ferramenta usa sua própria seção:

```toml
[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.11"
```

---

## Exemplo completo

```toml
[project]
name = "meu-projeto"
version = "0.1.0"
description = "Descrição do projeto"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Seu Nome", email = "email@exemplo.com" }]
dependencies = [
    "fastapi>=0.100.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
```

---

## Relação com outros arquivos

| Arquivo | Função |
|---------|--------|
| `pyproject.toml` | Metadados, versão do Python, dependências principais, configuração de ferramentas |
| `requirements.txt` | Lista de pacotes com versões fixas (gerada por `pip freeze` ou mantida manualmente) |
| `.python-version` | Versão exata do Python (pyenv, uv) |

---

## Benefícios

1. **Padrão único** — Um formato para metadados e configuração
2. **Versão do Python** — `requires-python` garante compatibilidade
3. **Integração** — uv, pip, Poetry, PDM e outras ferramentas o utilizam
4. **Publicação** — Necessário para publicar no PyPI
5. **Configuração centralizada** — Ruff, pytest, mypy, etc. em um só lugar
