# Convenções de Documentação

Este documento define o padrão para documentar instalação de bibliotecas, frameworks e ferramentas no projeto.

## Regra: Sempre especificar

Ao incluir comandos de instalação ou execução, **sempre informe**:

### 1. Shell / Terminal

Indique qual interpretador usar:

| Shell | Sistema |
|-------|---------|
| **PowerShell** | Windows |
| **CMD** | Windows |
| **Bash** | Linux / macOS |

Exemplo:
- **Windows (PowerShell):** `pip install numpy`
- **Windows (CMD):** `pip install numpy`
- **Linux/macOS (Bash):** `pip install numpy`

### 2. Ambiente virtual

Indique se o comando exige ambiente virtual ativo:

- **Com ambiente ativo** — ex.: `pip install` após `conda activate` ou `.venv\Scripts\Activate.ps1`
- **Sem ambiente** — ex.: `uv pip install` (uv usa o .venv automaticamente)
- **Instalação global** — ex.: instalar uv, conda, pyenv

Exemplo:
```markdown
Onde: Pasta raiz do projeto. Ambiente virtual: ATIVO (conda activate proj_8).

  pip install -r requirements.txt
```

### 3. Diretório / Local

Indique onde o comando deve ser executado:

- **Pasta raiz do projeto** — ex.: `proj_8/`
- **Qualquer diretório** — quando o local não importa
- **Dentro do diretório do ambiente** — quando aplicável

Exemplo:
```markdown
Onde: Pasta raiz do projeto (proj_8). Ambiente virtual: ainda não ativo.

  uv venv --python 3.12
```

## Template resumido

Para cada comando de instalação ou execução, inclua:

- [ ] **Shell:** PowerShell / CMD / Bash
- [ ] **Ambiente:** ativo / não ativo / não necessário
- [ ] **Diretório:** pasta raiz do projeto / qualquer / outro

## Exemplo completo

```markdown
--------------------------------------------------------------------------------
4. INSTALAR AS DEPENDÊNCIAS
--------------------------------------------------------------------------------

Onde: Pasta raiz do projeto (proj_8). Ambiente virtual: ATIVO.

Windows (PowerShell ou CMD):
  pip install -r requirements.txt

Linux/macOS (Bash):
  pip install -r requirements.txt
```
