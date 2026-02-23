# Explicação Detalhada: agentic_rag_avk.py

## Visão Geral

O arquivo `agentic_rag_avk.py` implementa um sistema **Agentic RAG** (Retrieval-Augmented Generation com Agentes de IA) que combina recuperação de informações de uma base de conhecimento vetorial com busca web dinâmica, utilizando um workflow baseado em grafos de estado (LangGraph).

---

## Estrutura do Módulo

### 1. Imports e Configurações Iniciais

#### Bibliotecas Principais

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
```

**Explicação**:
- **LangGraph**: Framework para criar workflows de agentes com grafos de estado
- **ChatOpenAI**: Interface para modelos de linguagem (aqui usado com Ollama local)
- **Chroma**: Banco de dados vetorial para armazenar e buscar embeddings
- **HuggingFaceEmbeddings**: Modelo para gerar embeddings de texto

#### Configurações de Compatibilidade

```python
import torch
torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**Objetivo**: Resolver problemas de compatibilidade entre Streamlit, PyTorch e o sistema operacional.

---

### 2. Inicialização do LLM (Linhas 29-41)

```python
llm = ChatOpenAI(model_name = "qwen2.5:0.5b",
                 openai_api_base = "http://localhost:11434/v1",
                 openai_api_key = "ollama",
                 temperature = 0,
                 max_tokens = 2048)
```

**Função**: Configura o modelo de linguagem local (Ollama) para gerar respostas.

**Parâmetros**:
- `model_name`: Modelo Qwen 2.5 0.5B (pequeno e eficiente)
- `openai_api_base`: Endpoint local do Ollama
- `temperature = 0`: Respostas determinísticas (sem criatividade)
- `max_tokens = 2048`: Limite de tokens na resposta

**Teste de Conexão** (linhas 37-41):
- Verifica se o Ollama está rodando e acessível
- Imprime status de conexão ou erro

**Correlação com outros módulos**:
- Usado em `app_avk.py` indiretamente através do `agent_workflow`
- Não é usado diretamente em `rag_avk.py` (que apenas cria o VectorDB)

---

### 3. Configuração do Sistema RAG (Linhas 43-61)

#### Modelo de Embeddings

```python
embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en")
```

**Função**: Gera embeddings vetoriais para busca semântica.

**Correlação com `rag_avk.py`**:
- **MESMO MODELO**: `rag_avk.py` usa o mesmo modelo (`BAAI/bge-base-en`) para criar o VectorDB
- Garante compatibilidade: embeddings gerados na criação = embeddings usados na busca

#### Banco Vetorial

```python
vector_db = Chroma(persist_directory = "vectordb", embedding_function = embedding_model)
retriever = vector_db.as_retriever()
```

**Função**: Carrega o banco vetorial criado por `rag_avk.py` e cria um retriever.

**Correlação com `rag_avk.py`**:
- **DEPENDÊNCIA DIRETA**: `agentic_rag_avk.py` depende do VectorDB criado por `rag_avk.py`
- `rag_avk.py` cria o banco em `vectordb/` → `agentic_rag_avk.py` carrega esse banco
- **Fluxo**: `rag_avk.py` (criação) → `agentic_rag_avk.py` (uso)

#### Chain de Recuperação e Geração

```python
prompt = PromptTemplate.from_template(
    "Você é um assistente especializado em logística e supply chain..."
)
avk_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()
qa_chain = create_retrieval_chain(retriever, avk_chain)
```

**Função**: Cria uma chain que combina recuperação (RAG) com geração de resposta.

**Fluxo**:
1. `RunnablePassthrough()`: Passa a query adiante
2. `prompt`: Formata o prompt com contexto e pergunta
3. `llm`: Gera a resposta
4. `StrOutputParser()`: Converte para string

**Correlação**:
- Usado em `avk_gera_multiplas_respostas()` para gerar respostas baseadas no RAG

---

### 4. Configuração do Agente Web (Linhas 63-74)

```python
search = DuckDuckGoSearchAPIWrapper(region = "br-pt", max_results = 5)
web_search_tool = Tool(name = "WebSearch", func = search.run, description = "Busca web.")
avk_agent_executor = initialize_agent(tools = [web_search_tool], ...)
```

**Função**: Cria um agente que pode buscar informações na web quando necessário.

**Características**:
- Busca em português do Brasil (`region = "br-pt"`)
- Retorna até 5 resultados (`max_results = 5`)
- Usa `CHAT_ZERO_SHOT_REACT_DESCRIPTION`: agente que raciocina sobre quando usar a ferramenta

**Correlação**:
- Usado em `avk_usar_ferramenta_web()` quando a query requer informações atualizadas

---

### 5. Classe AgentState (Linhas 76-84)

```python
class AgentState(BaseModel):
    query: str
    next_step: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0
```

**Função**: Define o estado do agente durante a execução do workflow.

**Campos**:
- `query`: Pergunta do usuário
- `next_step`: Próximo passo no workflow ("retrieve", "gerar", "usar_web")
- `retrieved_info`: Documentos recuperados do VectorDB
- `possible_responses`: Múltiplas respostas geradas pelo LLM
- `similarity_scores`: Scores de similaridade entre respostas e documentos
- `ranked_response`: Melhor resposta selecionada
- `confidence_score`: Nível de confiança da resposta

**Correlação com `app_avk.py`**:
- `app_avk.py` cria `AgentState(query=query)` e passa para `agent_workflow.invoke()`
- O estado final é usado para exibir resposta e confiança na interface

---

## Funções do Workflow

### 6. avk_passo_decisao_agente() (Linhas 86-95)

```python
def avk_passo_decisao_agente(state: AgentState) -> AgentState:
    query = state.query.lower()
    if any(palavra in query for palavra in ["explique", "resuma", "defina", ...]):
        state.next_step = "gerar"
    elif any(palavra in query for palavra in ["busque na web", "notícias", ...]):
        state.next_step = "usar_web"
    else:
        state.next_step = "retrieve"
    return state
```

**Função**: **Nó de decisão** - analisa a query e decide qual caminho seguir no workflow.

**Lógica de Decisão**:
1. **"gerar"**: Queries conceituais (explique, resuma, defina) → vai direto para geração
2. **"usar_web"**: Queries sobre atualidades → usa busca web
3. **"retrieve"**: Queries específicas → busca no VectorDB primeiro

**Correlação**:
- **Nó de entrada** do workflow (linha 160)
- Testado em `testa_agentic_rag.py` (testes 1-3)

**Fluxo no Workflow**:
```
decision → [retrieve | gerar | usar_web]
```

---

### 7. avk_retrieve_info() (Linhas 104-108)

```python
def avk_retrieve_info(state: AgentState) -> AgentState:
    retrieved_docs = retriever.invoke(state.query)
    state.retrieved_info = retrieved_docs
    return state
```

**Função**: Recupera documentos relevantes do VectorDB usando busca semântica.

**Processo**:
1. Converte a query em embedding
2. Busca documentos similares no ChromaDB
3. Armazena no estado

**Correlação**:
- **Usa o VectorDB criado por `rag_avk.py`**
- Documentos recuperados são usados em `avk_gera_multiplas_respostas()` e `avk_avalia_similaridade()`
- Testado em `testa_agentic_rag.py` (teste 5)

**Fluxo no Workflow**:
```
retrieve → generate_multiple
```

---

### 8. avk_gera_multiplas_respostas() (Linhas 110-114)

```python
def avk_gera_multiplas_respostas(state: AgentState) -> AgentState:
    responses = [qa_chain.invoke({"input": state.query}) for _ in range(5)]
    state.possible_responses = responses
    return state
```

**Função**: Gera **5 respostas diferentes** para a mesma query usando o RAG.

**Por que múltiplas respostas?**
- LLMs podem gerar respostas variadas
- Permite avaliar qual resposta está mais alinhada com os documentos recuperados
- Aumenta a confiabilidade do sistema

**Correlação**:
- Usa `qa_chain` que combina `retriever` + `llm`
- Respostas são avaliadas em `avk_avalia_similaridade()`
- Testado em `testa_agentic_rag.py` (teste 6)

**Fluxo no Workflow**:
```
generate_multiple → evaluate_similarity
```

---

### 9. avk_avalia_similaridade() (Linhas 116-134)

```python
def avk_avalia_similaridade(state: AgentState) -> AgentState:
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses
    retrieved_embeddings = embedding_model.embed_documents(retrieved_texts)
    response_embeddings = embedding_model.embed_documents(response_texts)
    
    similarities = [
        np.mean([cosine_similarity([response_embedding], [doc_embedding])[0][0] 
                 for doc_embedding in retrieved_embeddings])
        for response_embedding in response_embeddings
    ]
    state.similarity_scores = similarities
    return state
```

**Função**: Calcula a similaridade entre cada resposta gerada e os documentos recuperados.

**Processo**:
1. Extrai textos dos documentos recuperados
2. Gera embeddings para documentos e respostas
3. Calcula similaridade de cosseno entre cada resposta e todos os documentos
4. Média das similaridades = score de cada resposta

**Métrica**: **Similaridade de Cosseno**
- Valores entre -1 e 1
- Quanto mais próximo de 1, mais similar
- Indica se a resposta está alinhada com o conhecimento recuperado

**Correlação**:
- Usa o mesmo `embedding_model` usado para criar o VectorDB
- Scores são usados em `avk_rank_respostas()` para selecionar a melhor resposta
- Testado em `testa_agentic_rag.py` (teste 7)

**Fluxo no Workflow**:
```
evaluate_similarity → rank_responses
```

---

### 10. avk_rank_respostas() (Linhas 136-146)

```python
def avk_rank_respostas(state: AgentState) -> AgentState:
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores))
    if response_with_scores:
        ranked_responses = sorted(response_with_scores, key=lambda x: x[1], reverse=True)
        state.ranked_response = ranked_responses[0][0]
        state.confidence_score = ranked_responses[0][1]
    else:
        state.ranked_response = "Desculpe, não encontrei informações relevantes."
        state.confidence_score = 0.0
    return state
```

**Função**: Seleciona a resposta com maior similaridade aos documentos recuperados.

**Processo**:
1. Combina respostas com seus scores
2. Ordena por score (maior primeiro)
3. Seleciona a melhor resposta
4. Define o confidence_score como o score da melhor resposta

**Correlação**:
- `ranked_response` e `confidence_score` são exibidos em `app_avk.py`
- Testado em `testa_agentic_rag.py` (teste 8)

**Fluxo no Workflow**:
```
rank_responses → [FIM]
```

---

### 11. avk_usar_ferramenta_web() (Linhas 97-102)

```python
def avk_usar_ferramenta_web(state: AgentState) -> AgentState:
    resultado = avk_agent_executor.invoke(state.query)
    state.ranked_response = resultado.get("output", "Nenhuma informação obtida pela busca web.")
    state.confidence_score = 0.0
    return state
```

**Função**: Executa busca web quando a query requer informações atualizadas.

**Processo**:
1. Usa o agente web (`avk_agent_executor`) para buscar na internet
2. Armazena a resposta diretamente (sem avaliação de similaridade)
3. Define confidence_score como 0.0 (não há documentos para comparar)

**Correlação**:
- Caminho alternativo no workflow (não passa por retrieve/generate/evaluate)
- Testado em `testa_agentic_rag.py` (teste 4)

**Fluxo no Workflow**:
```
usar_web → [FIM]
```

---

## Construção do Workflow (Linhas 148-178)

### 12. Criação do Grafo de Estado

```python
workflow = StateGraph(AgentState)
workflow.add_node("decision", avk_passo_decisao_agente)
workflow.add_node("retrieve", avk_retrieve_info)
workflow.add_node("generate_multiple", avk_gera_multiplas_respostas)
workflow.add_node("evaluate_similarity", avk_avalia_similaridade)
workflow.add_node("rank_responses", avk_rank_respostas)
workflow.add_node("usar_web", avk_usar_ferramenta_web)
```

**Função**: Define os nós (funções) do workflow.

### 13. Definição do Ponto de Entrada

```python
workflow.set_entry_point("decision")
```

**Função**: Define que o workflow sempre começa no nó de decisão.

### 14. Arestas Condicionais

```python
workflow.add_conditional_edges(
    "decision",
    lambda state: {
        "retrieve": "retrieve",
        "gerar": "generate_multiple",
        "usar_web": "usar_web"
    }[state.next_step]
)
```

**Função**: Define que após a decisão, o fluxo vai para um dos três caminhos baseado em `state.next_step`.

### 15. Arestas Sequenciais

```python
workflow.add_edge("retrieve", "generate_multiple")
workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")
```

**Função**: Define o fluxo sequencial para o caminho RAG (retrieve → generate → evaluate → rank).

### 16. Compilação

```python
agent_workflow = workflow.compile()
```

**Função**: Compila o workflow em um objeto executável.

**Correlação com `app_avk.py`**:
- `app_avk.py` importa `agent_workflow` e executa com `agent_workflow.invoke(AgentState(query=query))`

---

## Diagrama do Workflow

```
                    [INÍCIO]
                      |
                   decision
                      |
        +-------------+-------------+
        |             |             |
     retrieve      gerar        usar_web
        |             |             |
        |         generate_     [FIM - Resposta Web]
        |         multiple         |
        |             |             |
        |      evaluate_similarity  |
        |             |             |
        |        rank_responses     |
        |             |             |
        +-------------+-------------+
                      |
                   [FIM]
```

---

## Correlações com Outros Módulos

### Correlação com `rag_avk.py`

| Aspecto | rag_avk.py | agentic_rag_avk.py |
|---------|------------|-------------------|
| **Função** | Cria o VectorDB | Usa o VectorDB |
| **Modelo Embeddings** | `BAAI/bge-base-en` | `BAAI/bge-base-en` (mesmo) |
| **Diretório VectorDB** | Cria em `vectordb/` | Carrega de `vectordb/` |
| **Dependência** | Independente | **Depende** do VectorDB criado |
| **Ordem de Execução** | Deve rodar primeiro | Deve rodar depois |

**Fluxo de Dependência**:
```
rag_avk.py (cria VectorDB) → agentic_rag_avk.py (usa VectorDB)
```

### Correlação com `app_avk.py`

| Aspecto | app_avk.py | agentic_rag_avk.py |
|---------|------------|-------------------|
| **Função** | Interface web (Streamlit) | Lógica do agente |
| **Importação** | `from agentic_rag_avk import AgentState, agent_workflow` | Exporta `AgentState` e `agent_workflow` |
| **Uso** | `agent_workflow.invoke(AgentState(query=query))` | Define `agent_workflow` |
| **Dados Exibidos** | `ranked_response`, `confidence_score`, `retrieved_info` | Gera esses dados |

**Fluxo de Integração**:
```
app_avk.py (interface) → agent_workflow.invoke() → agentic_rag_avk.py (processamento) → app_avk.py (exibição)
```

### Correlação com `testa_agentic_rag.py`

| Aspecto | testa_agentic_rag.py | agentic_rag_avk.py |
|---------|---------------------|-------------------|
| **Função** | Testes unitários | Código testado |
| **Importação** | `from agentic_rag_dsa import ...` | Nota: O teste usa `dsa_*` mas o padrão é o mesmo |
| **Cobertura** | Testa todas as funções principais | Funções são testadas |

**Funções Testadas**:
1. `avk_passo_decisao_agente()` - 3 testes (retrieve, usar_web, gerar)
2. `avk_usar_ferramenta_web()` - 1 teste
3. `avk_retrieve_info()` - 1 teste
4. `avk_gera_multiplas_respostas()` - 1 teste
5. `avk_avalia_similaridade()` - 1 teste
6. `avk_rank_respostas()` - 1 teste

---

## Fluxo Completo de Execução

### Cenário 1: Query Específica (RAG)

1. **Usuário** digita query em `app_avk.py`
2. **app_avk.py** cria `AgentState(query=query)` e chama `agent_workflow.invoke()`
3. **decision** analisa query → `next_step = "retrieve"`
4. **retrieve** busca documentos no VectorDB (criado por `rag_avk.py`)
5. **generate_multiple** gera 5 respostas usando `qa_chain` (RAG + LLM)
6. **evaluate_similarity** calcula similaridade entre respostas e documentos
7. **rank_responses** seleciona a melhor resposta (maior similaridade)
8. **app_avk.py** exibe `ranked_response` e `confidence_score`

### Cenário 2: Query Conceitual (Geração Direta)

1. **Usuário** digita "Resuma o conceito de supply chain"
2. **decision** → `next_step = "gerar"`
3. **generate_multiple** gera 5 respostas (sem retrieve prévio)
4. **evaluate_similarity** avalia similaridade
5. **rank_responses** seleciona melhor
6. **app_avk.py** exibe resultado

### Cenário 3: Query de Atualidades (Web)

1. **Usuário** digita "Busque notícias recentes sobre logística"
2. **decision** → `next_step = "usar_web"`
3. **usar_web** busca na internet usando DuckDuckGo
4. **app_avk.py** exibe resultado (sem confidence_score)

---

## Métodos e Técnicas Utilizadas

### 1. **RAG (Retrieval-Augmented Generation)**
- Recupera documentos relevantes antes de gerar resposta
- Reduz alucinações do LLM
- Baseia respostas em conhecimento verificável

### 2. **Agentic Architecture**
- Agente toma decisões sobre estratégia de busca
- Combina múltiplas fontes (VectorDB + Web)
- Adapta-se dinamicamente à query

### 3. **Multi-Response Generation**
- Gera múltiplas respostas para mesma query
- Avalia qual está mais alinhada com documentos
- Aumenta confiabilidade

### 4. **Similarity-Based Ranking**
- Usa embeddings e similaridade de cosseno
- Rankeia respostas por alinhamento com conhecimento
- Fornece métrica de confiança

### 5. **State Graph Workflow**
- Workflow baseado em grafos de estado
- Roteamento condicional baseado em decisões
- Fluxo claro e testável

---

## Pontos Importantes

### 1. **Dependência do VectorDB**
- `agentic_rag_avk.py` **não funciona** sem o VectorDB criado por `rag_avk.py`
- Execute `rag_avk.py` primeiro para criar/atualizar o banco

### 2. **Modelo de Embeddings Consistente**
- Mesmo modelo (`BAAI/bge-base-en`) em criação e uso
- Garante compatibilidade entre embeddings

### 3. **Múltiplas Respostas para Confiabilidade**
- Gera 5 respostas e seleciona a melhor
- Aumenta custo computacional mas melhora qualidade

### 4. **Confidence Score**
- Baseado em similaridade com documentos recuperados
- Não aplicável para busca web (sempre 0.0)

### 5. **Workflow Condicional**
- Três caminhos diferentes baseados na query
- Otimiza uso de recursos (não busca web se não necessário)

---

## Melhorias Possíveis

1. **Top-K Configurável**: Permitir ajustar quantos documentos recuperar
2. **Threshold de Similaridade**: Filtrar documentos com baixa relevância
3. **Cache de Respostas**: Evitar reprocessar queries similares
4. **Métricas de Avaliação**: Adicionar mais métricas além de similaridade
5. **Logging**: Registrar decisões e caminhos do workflow
6. **Tratamento de Erros**: Melhorar tratamento de falhas em cada nó

---

*Documento criado para o Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA*

