# Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA

## Ferramentas utilizadas

| Ferramenta | Uso no projeto |
|------------|----------------|
| **LMStudio** | API local do modelo de linguagem Hermes (hermes-3-llama-3.2-3b). Deve estar rodando em `http://127.0.0.1:1234` antes de executar a aplicação. |
| **ChromaDB** | Banco de dados vetorial para armazenar e buscar embeddings dos documentos (RAG). Dados persistidos em `data/vectordb/`. |
| **HuggingFace** | Modelo BAAI/bge-base-en para gerar embeddings de texto na busca semântica. |
| **DuckDuckGo** | Busca web para o agente obter informações atualizadas quando necessário. |
| **LangGraph** | Framework para orquestrar o workflow do agente (grafos de estado). |
| **Streamlit** | Interface web da aplicação. |
| **LangChain** | Framework para chains, agents, prompts e integração com LLMs. |

> **Pré-requisito:** Inicie o LMStudio e carregue o modelo Hermes na porta 1234 antes de rodar a app.

## Para executar a app (com uv):

> **Local:** Todos os comandos abaixo devem ser executados na **pasta raiz do projeto** (proj_8), exceto quando indicado.

### 0. Instalar o uv

**Onde:** Qualquer diretório. **Ambiente virtual:** não necessário.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Windows (CMD):**
```cmd
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS (Bash):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Ou via pip** (com Python já instalado, ambiente virtual opcional): `pip install uv`

Verificar instalação: `uv --version`

### 1. Instalar Python 3.12 (se necessário)

**Onde:** Qualquer diretório. **Ambiente virtual:** não necessário.

O projeto requer Python 3.11 ou 3.12 (onnxruntime não suporta Python 3.14+).

**PowerShell / CMD / Bash:**
```bash
uv python install 3.12
```

Listar versões: `uv python list`

### 2. Criar ambiente virtual e instalar dependências

**Onde:** Pasta raiz do projeto (proj_8). **Ambiente virtual:** ainda não ativo (estamos criando).

**Criar o ambiente** (o uv baixa o Python automaticamente se não existir):

**PowerShell / CMD / Bash:**
```bash
uv venv --python 3.12
```

**Instalar as dependências** (sem ativar o env — uv usa o .venv automaticamente):

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

O `-e .` instala o pacote agenticlog em modo editável (necessário para a estrutura src/).

Ou, se já tiver Python 3.12 no sistema:
```bash
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

### 3. Ativar o ambiente

**Onde:** Pasta raiz do projeto (onde está o `.venv`).

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/macOS (Bash):**
```bash
source .venv/bin/activate
```

Após ativar, o prompt mostrará `(.venv)` indicando que o ambiente está ativo.

### 4. Executar a aplicação

**Onde:** Pasta raiz do projeto. **Ambiente virtual:** ativo (ou use `uv run`).

**Criar o VectorDB** (primeira vez, antes de rodar a app):

```bash
python -m agenticlog.rag
```

**Rodar a interface Streamlit:**

```bash
streamlit run app.py
```

**Executar os testes:**

```bash
python tests/test_agentic_rag.py -v
```

**Alternativa sem ativar o ambiente** (uv run usa o .venv automaticamente):

```bash
uv run python -m agenticlog.rag
uv run streamlit run app.py
uv run python tests/test_agentic_rag.py -v
```

## Exemplos de perguntas:

- Resuma o conceito de supply chain.
- Quais as fases da cadeia de suprimentos?
- Pesquise sobre as novidades recentes em logistica e supply chain.

## Desativar o ambiente

```bash
deactivate
```

(O ambiente fica na pasta `.venv` — para removê-lo, basta deletar essa pasta.)

