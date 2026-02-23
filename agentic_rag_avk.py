# Projeto 8 - Pipeline de Automação de Testes Para Agentes de IA
"""
Módulo: agentic_rag_avk.py

OBJETIVO PRINCIPAL:
Implementa um sistema Agentic RAG (Retrieval-Augmented Generation com Agentes de IA) que:
1. Combina recuperação de informações de base de conhecimento vetorial (RAG)
2. Integra busca web dinâmica para informações atualizadas
3. Utiliza workflow baseado em grafos de estado (LangGraph) para orquestração
4. Gera múltiplas respostas e seleciona a melhor baseada em similaridade com documentos

CORRELAÇÃO COM OUTROS MÓDULOS:
- rag_avk.py: DEPENDÊNCIA - Este módulo usa o VectorDB criado por rag_avk.py
              O VectorDB deve ser criado primeiro executando rag_avk.py
              Usa o mesmo modelo de embeddings (BAAI/bge-base-en) para compatibilidade
              
- app_avk.py: INTEGRAÇÃO - app_avk.py importa AgentState e agent_workflow deste módulo
              Interface Streamlit chama agent_workflow.invoke() para processar queries
              Exibe ranked_response, confidence_score e retrieved_info retornados
              
- testa_agentic_rag.py: TESTES - Testa todas as funções principais deste módulo
                        Valida lógica de decisão, recuperação, geração e ranking
"""

# Imports
import os
import numpy as np  # type: ignore
from langgraph.graph import StateGraph  # type: ignore[reportMissingImports]  # Framework para criar workflows de agentes com grafos de estado
from pydantic import BaseModel  # Validação de dados e definição de modelos de estado
from langchain_openai import ChatOpenAI  # Interface para modelos de linguagem (usado com Ollama local)
from langchain_chroma import Chroma  # Banco de dados vetorial para armazenar e buscar embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Modelo para gerar embeddings de texto
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper  # Wrapper para busca web
from langchain.tools import Tool  # Ferramenta para agentes de IA
from langchain.agents import initialize_agent, AgentType  # Criação de agentes autônomos
from langchain.chains import create_retrieval_chain  # Chain que combina RAG com geração
from langchain_core.runnables import RunnablePassthrough  # Runnable que passa dados adiante
from langchain_core.prompts import PromptTemplate  # Template para formatação de prompts
from langchain_core.output_parsers import StrOutputParser  # Parser para converter saída em string
from sklearn.metrics.pairwise import cosine_similarity  # Métrica de similaridade entre vetores
import warnings
warnings.filterwarnings('ignore')

# Evita problema de compatibilidade entre Streamlit e PyTorch
# CORRELAÇÃO: Necessário para app_avk.py (Streamlit) funcionar corretamente com PyTorch
import torch
torch.classes.__path__ = []

# Evita problema de compatibilidade com o SO
# Previne warnings de paralelismo em sistemas Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# INICIALIZAÇÃO DO LLM (MODELO DE LINGUAGEM)
# ============================================================================
# OBJETIVO: Configura o modelo de linguagem local (Qwen 2.5 0.5B) para gerar respostas
# USO: Utilizado em qa_chain para gerar respostas baseadas em contexto RAG
#      Também usado em avk_agent_executor para decisões do agente web
# PARÂMETROS:
#   - model_name: Modelo Qwen 2.5 0.5B (pequeno e eficiente para uso local)
#   - openai_api_base: Endpoint local do Ollama (deve estar rodando na porta 11434)
#   - temperature = 0: Respostas determinísticas (sem criatividade/variação)
#   - max_tokens = 2048: Limite máximo de tokens na resposta gerada
llm = ChatOpenAI(model_name = "hermes-3-llama-3.2-3b",
                 openai_api_base = "http://127.0.0.1:1234/v1",
                 openai_api_key = "hermes",
                 temperature = 0,
                 max_tokens = 2048)

# Teste simples de conexão com o Ollama
# OBJETIVO: Valida se o Ollama está rodando e acessível antes de processar queries
# IMPORTÂNCIA: Evita erros em runtime se o Ollama não estiver disponível
try:
    response = llm.invoke("Just respond with the word: CONNECTED")
    print(f"Status do hermes-3-llama-3.2-3b: {response.content}")
except Exception as e:
    print(f"Connection error: {e}")

# ============================================================================
# CONFIGURAÇÃO DO SISTEMA RAG (RETRIEVAL-AUGMENTED GENERATION)
# ============================================================================

# Define o modelo de embeddings
# OBJETIVO: Gera embeddings vetoriais para busca semântica
# CORRELAÇÃO COM rag_avk.py: 
#   - DEVE SER O MESMO MODELO usado em rag_avk.py para criar o VectorDB
#   - Garante compatibilidade: embeddings gerados na criação = embeddings usados na busca
#   - Modelo: BAAI/bge-base-en (otimizado para busca semântica em inglês)
# USO: Utilizado em avk_avalia_similaridade() para calcular similaridade entre respostas e documentos
embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en")

# Carrega o banco vetorial do RAG
# OBJETIVO: Carrega o VectorDB criado por rag_avk.py
# CORRELAÇÃO COM rag_avk.py:
#   - DEPENDÊNCIA CRÍTICA: Este módulo NÃO funciona sem o VectorDB criado por rag_avk.py
#   - rag_avk.py cria o banco em "vectordb/" → Este módulo carrega esse banco
#   - Ordem de execução: rag_avk.py (criação) → agentic_rag_avk.py (uso)
#   - Se o diretório "vectordb/" não existir, ocorrerá erro ao executar
# PARÂMETROS:
#   - persist_directory: Diretório onde o VectorDB foi salvo (deve existir)
#   - embedding_function: Modelo usado para gerar embeddings (deve ser o mesmo da criação)
vector_db = Chroma(persist_directory = "data/vectordb", embedding_function = embedding_model)

# Cria o retriever para recuperar os dados do RAG
# OBJETIVO: Interface para buscar documentos similares no VectorDB
# USO: Utilizado em avk_retrieve_info() para recuperar documentos relevantes
#      Também usado em qa_chain para RAG automático
# FUNCIONAMENTO: Converte query em embedding → busca documentos similares → retorna top-K
retriever = vector_db.as_retriever()

# Cria o prompt template com placeholders
# OBJETIVO: Define o formato do prompt enviado ao LLM
# PLACEHOLDERS:
#   - {context}: Preenchido com documentos recuperados do VectorDB
#   - {input}: Preenchido com a query do usuário
# USO: Utilizado em avk_chain para formatar prompts antes de enviar ao LLM

# Para o caminho 'retrieve' (RAG)
# Mais rígido para evitar alucinação do Qwen 0.5B
# USO: Utilizado em avk_retrieve_info() para formatar o prompt antes de enviar ao LLM

prompt_rag_retrieve = PromptTemplate.from_template(
    """You are a truthful and precise assistant in logistics and supply chain.
    Your task is to answer the user's question based STRICTLY on the provided context below.

    REGRAS DE RESPOSTA:
    1. USE ONLY the information inside the context block.
    2. DO NOT use your internal knowledge or previous training.
    3. If the answer is not in the context, reply exactly: "Sorry, I did not find that information in the documents."
    4. Answer the user in Brazilian Portuguese based on the provided context.

    --- Context ---
    {context}
    --- End of Context ---

    User Question: {input}
    Answer:"""
)

# Para o caminho 'gerar' (Conceitual)
# Permite que o modelo use seu conhecimento interno
# OBSERVAÇÃO: Este prompt NÃO usa {context} porque é para respostas conceituais gerais
prompt_gerar = PromptTemplate.from_template(
    """You are a specialist in logistics and supply chain.
    Your task is to explain, summarize or define the requested concept.
    Answer using your general knowledge in a clear and professional way, in Brazilian Portuguese.

    User Question: {input}
    Answer:"""
)



# ============================================================================
# CONFIGURAÇÃO DO AGENTE WEB (BUSCA NA INTERNET)
# ============================================================================

# Define o mecanismo de busca
# OBJETIVO: Wrapper para busca web usando DuckDuckGo
# PARÂMETROS:
#   - region = "br-pt": Busca em português do Brasil
#   - max_results = 5: Retorna até 5 resultados por busca
# USO: Utilizado em web_search_tool para realizar buscas na internet
search = DuckDuckGoSearchAPIWrapper(region = "br-pt", max_results = 5)

# Cria a ferramenta do Agente de IA
# OBJETIVO: Define uma ferramenta que o agente pode usar para buscar na web
# FUNCIONAMENTO: O agente decide quando usar esta ferramenta baseado na query
# USO: Passada para avk_agent_executor como ferramenta disponível
web_search_tool = Tool(name = "WebSearch", func = search.run, description = "Busca web.")

# Inicializa o agente de pesquisa na web
# OBJETIVO: Cria um agente autônomo que pode decidir quando buscar na web
# CARACTERÍSTICAS:
#   - AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION: Agente que raciocina sobre quando usar ferramentas
#   - verbose = True: Exibe raciocínio do agente (útil para debug)
#   - handle_parsing_errors = True: Trata erros de parsing graciosamente
# USO: Utilizado em avk_usar_ferramenta_web() quando query requer informações atualizadas
# CORRELAÇÃO: Usa o mesmo llm configurado acima para raciocínio do agente
avk_agent_executor = initialize_agent(tools = [web_search_tool],
                                      llm = llm,
                                      agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose = True,
                                      handle_parsing_errors = True)

# ============================================================================
# CLASSE DE ESTADO DO AGENTE
# ============================================================================
# OBJETIVO: Define a estrutura de dados que representa o estado do agente durante a execução
# USO: Passado entre nós do workflow, mantendo contexto durante todo o processamento
# CORRELAÇÃO COM app_avk.py:
#   - app_avk.py cria AgentState(query=query) e passa para agent_workflow.invoke()
#   - O estado final é usado para exibir resposta, confiança e documentos na interface
# CAMPOS:
#   - query: Pergunta do usuário (entrada)
#   - next_step: Próximo passo no workflow ("retrieve", "gerar", "usar_web") - definido por avk_passo_decisao_agente()
#   - retrieved_info: Lista de documentos recuperados do VectorDB - preenchido por avk_retrieve_info()
#   - possible_responses: Lista de múltiplas respostas geradas pelo LLM - preenchido por avk_gera_multiplas_respostas()
#   - similarity_scores: Scores de similaridade entre respostas e documentos - calculado por avk_avalia_similaridade()
#   - ranked_response: Melhor resposta selecionada - definido por avk_rank_respostas() ou avk_usar_ferramenta_web()
#   - confidence_score: Nível de confiança da resposta (0.0 a 1.0) - calculado por avk_rank_respostas()
class AgentState(BaseModel):
    query: str
    next_step: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0

# ============================================================================
# FUNÇÕES DO WORKFLOW
# ============================================================================

# Função para o passo de decisão do agente
# OBJETIVO: Nó de decisão - analisa a query e decide qual caminho seguir no workflow
# ENTRADA: AgentState com query do usuário
# SAÍDA: AgentState com next_step definido
# LÓGICA DE DECISÃO com palavras chaves em inglês:
#   1. "to-generate": Queries conceituais (explique, resuma, defina) → vai direto para geração sem retrieve
#   2. "Use-web": Queries sobre atualidades → usa busca web (não passa por RAG)
#   3. "retrieve": Queries específicas → busca no VectorDB primeiro (caminho RAG completo)
# USO NO WORKFLOW: Nó de entrada (workflow sempre começa aqui)
# CORRELAÇÃO: Testado em testa_agentic_rag.py (testes 1-3 validam cada caminho)
def avk_passo_decisao_agente(state: AgentState) -> AgentState:
    query = state.query.lower()
    if any(palavra in query for palavra in ["explain", "summarize", "define", "concept", "general", "what is", "explique", "resuma", "defina", "conceito", "geral", "o que é"]):
        state.next_step = "gerar"
    elif any(palavra in query for palavra in ["search the web", "news", "updated", "recent", "latest information", "busque na web", "notícias", "atualizado", "recente", "últimas informações"]):
        state.next_step = "usar_web"
    else:
        state.next_step = "retrieve"
    return state

# Função para usar busca na web
# OBJETIVO: Executa busca web quando a query requer informações atualizadas
# ENTRADA: AgentState com query
# SAÍDA: AgentState com ranked_response preenchido (resposta da web)
# PROCESSO:
#   1. Usa avk_agent_executor (agente web) para buscar na internet
#   2. Armazena resposta diretamente (sem avaliação de similaridade, pois não há documentos para comparar)
#   3. Define confidence_score como 0.0 (não há documentos para comparar)
# USO NO WORKFLOW: Caminho alternativo (não passa por retrieve/generate/evaluate)
# CORRELAÇÃO: Testado em testa_agentic_rag.py (teste 4)
# OBSERVAÇÃO: Este caminho é independente do RAG - usa apenas busca web
def avk_usar_ferramenta_web(state: AgentState) -> AgentState:
    result = avk_agent_executor.invoke(state.query)
    state.ranked_response = result.get("output", "No information obtained by web search.")
    state.confidence_score = 0.0  
    return state

# Função para recuperar documentos do RAG
# OBJETIVO: Recupera documentos relevantes do VectorDB usando busca semântica
# ENTRADA: AgentState com query
# SAÍDA: AgentState com retrieved_info preenchido (lista de documentos)
# PROCESSO:
#   1. Converte query em embedding usando embedding_model
#   2. Busca documentos similares no ChromaDB (VectorDB)
#   3. Retorna top-K documentos mais relevantes (K padrão do retriever)
# CORRELAÇÃO COM rag_avk.py:
#   - Usa o VectorDB criado por rag_avk.py (carregado em vector_db)
#   - Documentos foram indexados por rag_avk.py usando mesmo embedding_model
# USO NO WORKFLOW: Primeiro passo do caminho RAG (retrieve → generate → evaluate → rank)
# USO POSTERIOR: Documentos são usados em:
#   - avk_gera_multiplas_respostas(): Contexto para geração (via qa_chain)
#   - avk_avalia_similaridade(): Comparação com respostas geradas
# CORRELAÇÃO: Testado em testa_agentic_rag.py (teste 5)
def avk_retrieve_info(state: AgentState) -> AgentState:
    retrieved_docs = retriever.invoke(state.query)
    state.retrieved_info = retrieved_docs
    return state

# Função para gerar múltiplas respostas do LLM
# OBJETIVO: Gera 5 respostas diferentes para a mesma query usando o RAG
# ENTRADA: AgentState com query (e opcionalmente retrieved_info se veio de retrieve)
# SAÍDA: AgentState com possible_responses preenchido (lista de 5 respostas)
# PROCESSO:
#   1. Invoca qa_chain 5 vezes com a mesma query
#   2. Cada invocação pode gerar resposta ligeiramente diferente (mesmo com temperature=0)
#   3. qa_chain automaticamente usa retrieved_info se disponível, ou busca novos documentos
# POR QUE MÚLTIPLAS RESPOSTAS?
#   - LLMs podem gerar respostas variadas mesmo com mesmos parâmetros
#   - Permite avaliar qual resposta está mais alinhada com documentos recuperados
#   - Aumenta confiabilidade: seleciona a melhor entre várias opções
# USO NO WORKFLOW: Segundo passo do caminho RAG (após retrieve ou diretamente após decision)
# USO POSTERIOR: Respostas são avaliadas em avk_avalia_similaridade()
# CORRELAÇÃO: 
#   - Usa qa_chain que combina retriever + llm (definidos acima)
#   - Testado em testa_agentic_rag.py (teste 6)

# Função para gerar múltiplas respostas do LLM 
def avk_gera_multiplas_respostas(state: AgentState) -> AgentState:
    # 1. Prepara o contexto formatado (se houver documentos recuperados)
    def format_docs(docs):
        """Formata lista de Document objects em string"""
        if not docs:
            return ""
        return "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Formata o contexto uma vez (se for caminho retrieve)
    if state.next_step == "retrieve":
        # RAG: Usar prompt restritivo. O retriever já rodou e preencheu state.retrieved_info.
        context_text = format_docs(state.retrieved_info)
        current_prompt = prompt_rag_retrieve
    else: # state.next_step == "gerar"
        # Conceitual: Não usar RAG. Usar prompt permissivo.
        context_text = ""  # Sem contexto para respostas conceituais
        current_prompt = prompt_gerar
        # Garante que o contexto não seja injetado
        state.retrieved_info = []

    # 3. Cria a chain dinâmica baseada no prompt escolhido
    if state.next_step == "retrieve":
        # Chain para RAG: inclui contexto dos documentos
        qa_chain_dynamic = (
            current_prompt
            | llm
            | StrOutputParser()
        )
    else:
        # Chain para geração conceitual: sem contexto
        qa_chain_dynamic = (
            current_prompt
            | llm
            | StrOutputParser()
        )

    # 4. Invoca a chain 5 vezes
    responses = []
    for _ in range(5):
        if state.next_step == "retrieve":
            # Para RAG: passa contexto e input
            response = qa_chain_dynamic.invoke({
                "context": context_text,
                "input": state.query
            })
        else:
            # Para conceitual: passa apenas input
            response = qa_chain_dynamic.invoke({
                "input": state.query
            })
        responses.append(response)
    
    # 5. Armazena as respostas no formato esperado
    state.possible_responses = [{"answer": r} for r in responses]
    return state


# Função que avalia a similaridade entre respostas do LLM e documentos recuperados do RAG
# OBJETIVO: Calcula a similaridade entre cada resposta gerada e os documentos recuperados
# ENTRADA: AgentState com possible_responses e retrieved_info
# SAÍDA: AgentState com similarity_scores preenchido (lista de scores, um para cada resposta)
# PROCESSO:
#   1. Extrai textos dos documentos recuperados (retrieved_info)
#   2. Extrai textos das respostas geradas (possible_responses)
#   3. Gera embeddings para documentos e respostas usando embedding_model
#   4. Calcula similaridade de cosseno entre cada resposta e todos os documentos
#   5. Média das similaridades = score de cada resposta
# MÉTRICA: Similaridade de Cosseno
#   - Valores entre -1 e 1 (geralmente 0 a 1 para embeddings normalizados)
#   - Quanto mais próximo de 1, mais similar (resposta alinhada com conhecimento)
#   - Indica se a resposta está baseada nos documentos recuperados
# CORRELAÇÃO:
#   - Usa o mesmo embedding_model usado para criar o VectorDB (garante compatibilidade)
#   - Scores são usados em avk_rank_respostas() para selecionar a melhor resposta
# USO NO WORKFLOW: Terceiro passo do caminho RAG (após generate_multiple)
# CORRELAÇÃO: Testado em testa_agentic_rag.py (teste 7)
def avk_avalia_similaridade(state: AgentState) -> AgentState:
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses
    retrieved_embeddings = embedding_model.embed_documents(retrieved_texts) if retrieved_texts else []
    response_texts = [response["answer"] if isinstance(response, dict) and "answer" in response else str(response) for response in responses]
    response_embeddings = embedding_model.embed_documents(response_texts) if response_texts else []

    if not retrieved_embeddings or not response_embeddings:
        state.similarity_scores = [0.0] * len(response_texts)
        return state

    similarities = [
        np.mean([cosine_similarity([response_embedding], [doc_embedding])[0][0] for doc_embedding in retrieved_embeddings])
        for response_embedding in response_embeddings
    ]

    state.similarity_scores = similarities
    return state

# Função para criar um rank das respostas (somente a melhor resposta será mostrada ao usuário final)
# OBJETIVO: Seleciona a resposta com maior similaridade aos documentos recuperados
# ENTRADA: AgentState com possible_responses e similarity_scores
# SAÍDA: AgentState com ranked_response (melhor resposta) e confidence_score (score da melhor)
# PROCESSO:
#   1. Combina respostas com seus scores de similaridade
#   2. Ordena por score (maior primeiro = mais similar aos documentos)
#   3. Seleciona a melhor resposta (primeira da lista ordenada)
#   4. Define confidence_score como o score da melhor resposta
# CONFIDENCE_SCORE:
#   - Representa o quão alinhada a resposta está com o conhecimento recuperado
#   - Valores altos (>0.7): Resposta muito alinhada com documentos
#   - Valores baixos (<0.5): Resposta pode ter informações não baseadas nos documentos
# USO NO WORKFLOW: Último passo do caminho RAG (após evaluate_similarity)
# CORRELAÇÃO COM app_avk.py:
#   - ranked_response é exibido na interface como resposta final
#   - confidence_score é exibido como métrica de confiança
# CORRELAÇÃO: Testado em testa_agentic_rag.py (teste 8)
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

# ============================================================================
# CONSTRUÇÃO DO WORKFLOW (GRAFO DE ESTADO)
# ============================================================================
# OBJETIVO: Define a orquestração do sistema Agentic RAG usando LangGraph
# ESTRUTURA: Grafo direcionado onde cada nó é uma função e arestas definem o fluxo

# Cria o workflow de execução do Agente de IA
# OBJETIVO: Instancia um grafo de estado usando AgentState como estrutura de dados
workflow = StateGraph(AgentState)

# Adiciona nós ao workflow (cada nó é uma função do sistema)
# OBJETIVO: Define as etapas de processamento do workflow
# NÓS:
#   - "decision": Nó de decisão (sempre executado primeiro)
#   - "retrieve": Recuperação de documentos do VectorDB
#   - "generate_multiple": Geração de múltiplas respostas
#   - "evaluate_similarity": Avaliação de similaridade
#   - "rank_responses": Seleção da melhor resposta
#   - "usar_web": Busca web (caminho alternativo)
workflow.add_node("decision", avk_passo_decisao_agente)
workflow.add_node("retrieve", avk_retrieve_info)
workflow.add_node("generate_multiple", avk_gera_multiplas_respostas)
workflow.add_node("evaluate_similarity", avk_avalia_similaridade)
workflow.add_node("rank_responses", avk_rank_respostas)
workflow.add_node("usar_web", avk_usar_ferramenta_web)

# Ponto de entrada (início) da execução
# OBJETIVO: Define que o workflow sempre começa no nó de decisão
# FLUXO: Todas as execuções começam em "decision" que analisa a query
workflow.set_entry_point("decision")

# Cria arestas condicionais
# OBJETIVO: Define roteamento dinâmico baseado na decisão do agente
# FUNCIONAMENTO: Após "decision", o fluxo vai para um dos três caminhos:
#   - "retrieve" → caminho RAG completo (retrieve → generate → evaluate → rank)
#   - "gerar" → caminho direto (generate → evaluate → rank, sem retrieve prévio)
#   - "usar_web" → caminho web (apenas usar_web, depois fim)
# LÓGICA: Usa state.next_step definido por avk_passo_decisao_agente()
workflow.add_conditional_edges(
    "decision",
    lambda state: {
        "retrieve": "retrieve",
        "gerar": "generate_multiple",
        "usar_web": "usar_web"
    }[state.next_step]
)

# Adiciona as demais arestas
# OBJETIVO: Define o fluxo sequencial para o caminho RAG
# FLUXO RAG COMPLETO:
#   retrieve → generate_multiple → evaluate_similarity → rank_responses → [FIM]
# OBSERVAÇÃO: 
#   - "gerar" também vai para generate_multiple (via aresta condicional acima)
#   - "usar_web" termina diretamente (não tem arestas adicionais)
workflow.add_edge("retrieve", "generate_multiple")
workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")

# Compila o workflow para execução
# OBJETIVO: Compila o grafo em um objeto executável
# RESULTADO: agent_workflow pode ser invocado com AgentState(query="...")
# CORRELAÇÃO COM app_avk.py:
#   - app_avk.py importa agent_workflow deste módulo
#   - Interface chama: agent_workflow.invoke(AgentState(query=query))
#   - Retorna AgentState final com ranked_response, confidence_score, etc.
agent_workflow = workflow.compile()





