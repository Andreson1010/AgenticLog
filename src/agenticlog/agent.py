# AgenticLog - Lógica Agentic RAG
"""
Sistema Agentic RAG com LangGraph: RAG vetorial, geração conceitual e busca web.

O grafo de estados roteia cada consulta para um de três caminhos:
- retrieve  → busca no ChromaDB, gera múltiplas respostas, rankeia por similaridade de cosseno.
- gerar     → gera resposta conceitual diretamente com o LLM (sem retrieval).
- usar_web  → delega a um agente DuckDuckGo para consultas que exigem informações recentes.
"""

import os
import logging
import numpy as np  # type: ignore
import httpx
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from langgraph.graph import StateGraph  # type: ignore[reportMissingImports]
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

import torch

torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agenticlog.config import (
    DIR_VECTORDB,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRY_ATTEMPTS,
    LLM_RETRY_WAIT_INITIAL_SECONDS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    ROUTING_KEYWORDS_GERAR,
    ROUTING_KEYWORDS_WEB,
)

logger = logging.getLogger(__name__)

_llm_retry = retry(
    stop=stop_after_attempt(LLM_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(min=LLM_RETRY_WAIT_INITIAL_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS),
    retry=retry_if_exception_type(
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            anthropic.APIConnectionError,
        )
    ),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# Singletons lazy — inicializados somente na primeira chamada, não na importação
_llm = None
_vector_db = None
_embedding_model = None


def _get_llm() -> ChatOpenAI:
    """Retorna o singleton do LLM, criando-o na primeira chamada.

    Saída: instância de ChatOpenAI configurada com as constantes do config.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model_name=LLM_MODEL,
            openai_api_base=LLM_API_BASE,
            openai_api_key=LLM_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            request_timeout=LLM_TIMEOUT_SECONDS,
        )
    return _llm


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna o singleton do modelo de embeddings, criando-o na primeira chamada.

    Saída: instância de HuggingFaceEmbeddings.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model


def _get_vector_db() -> Chroma:
    """Retorna o singleton do ChromaDB, criando-o na primeira chamada.

    Saída: instância de Chroma conectada ao diretório persistido.
    """
    global _vector_db
    if _vector_db is None:
        _vector_db = Chroma(
            persist_directory=str(DIR_VECTORDB),
            embedding_function=_get_embedding_model(),
        )
    return _vector_db


def _get_retriever():
    """Retorna um retriever de similaridade a partir do vector_db lazy.

    Saída: retriever configurado com search_type='similarity' e k=3.
    """
    return _get_vector_db().as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )


def inicializar_recursos() -> None:
    """Inicializa singletons do agente (LLM, ChromaDB, embeddings) na inicialização do servidor.

    Entrada: nenhuma
    Saída: nenhuma — efeito colateral: singletons globais inicializados

    Ordem de inicialização: embeddings → vector_db → llm.
    Chamada única a partir do lifespan do FastAPI; elimina race condition em requisições concorrentes.
    """
    _get_embedding_model()
    _get_vector_db()
    _get_llm()


def invalidar_vector_db() -> None:
    """Invalida o singleton _vector_db para que a próxima chamada a _get_vector_db() reconecte ao ChromaDB.

    Entrada: nenhuma.
    Saída: nenhuma.
    Efeito colateral: atribui None a _vector_db global.
    """
    global _vector_db
    _vector_db = None


# Prompts — inicializados na importação do módulo
prompt_rag_retrieve = PromptTemplate.from_template(
    """You are a truthful and precise assistant in logistics and supply chain.
    Your task is to answer the user's question based STRICTLY on the provided context below.

    # REGRAS DE RESPOSTA: restrições obrigatórias que impedem o LLM de alucinar ou usar conhecimento externo
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

prompt_gerar = PromptTemplate.from_template(
    """You are a specialist in logistics and supply chain.
    Your task is to explain, summarize or define the requested concept.
    Answer using your general knowledge in a clear and professional way, in Brazilian Portuguese.

    User Question: {input}
    Answer:"""
)

_prompt_web = PromptTemplate.from_template(
    "Use the following web search results to answer the question in Brazilian Portuguese.\n\n"
    "Search results:\n{context}\n\n"
    "Question: {input}\nAnswer:"
)

# Ferramentas de busca web — DuckDuckGo não requer LMStudio, inicializado na importação
search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)


class AgentState(BaseModel):
    """Carregador de estado imutável por convenção entre os nós do grafo LangGraph.

    Cada nó recebe uma cópia do estado e retorna um novo estado atualizado;
    nenhum nó deve modificar campos de outros nós diretamente.
    """

    query: str                          # pergunta original do usuário
    next_step: str = ""                 # rota decidida: "retrieve", "gerar" ou "usar_web"
    retrieved_info: list = []           # documentos retornados pelo retriever vetorial
    possible_responses: list = []       # 5 respostas geradas pelo LLM para ranqueamento
    similarity_scores: list = []        # scores de cosseno de cada resposta vs. contexto recuperado
    ranked_response: str = ""           # melhor resposta após ranqueamento por similaridade
    confidence_score: float = 0.0       # score de confiança da resposta ranqueada (0.0–1.0)


def passo_decisao_agente(state: AgentState) -> AgentState:
    """Nó de decisão do grafo: define state.next_step com base em palavras-chave da consulta.

    Entrada: state.query (pergunta do usuário).
    Saída:   state.next_step — "gerar", "usar_web" ou "retrieve" (padrão).

    # Listas de palavras-chave: controlam o roteamento — termos conceituais → "gerar",
    # termos de busca web → "usar_web", qualquer outro caso → "retrieve".
    """
    query = state.query.lower()
    if any(p in query for p in ROUTING_KEYWORDS_GERAR):
        return state.model_copy(update={"next_step": "gerar"})
    if any(p in query for p in ROUTING_KEYWORDS_WEB):
        return state.model_copy(update={"next_step": "usar_web"})
    return state.model_copy(update={"next_step": "retrieve"})


@_llm_retry
def _invoke_chain(chain, inputs: dict) -> str:
    """Invoca uma chain LangChain com retry exponential backoff em erros de conexão/timeout.

    Entrada: chain — pipeline prompt | llm | parser; inputs — dict de variáveis do prompt.
    Saída:   resposta gerada pelo LLM como string.
    """
    return chain.invoke(inputs)


def usar_ferramenta_web(state: AgentState) -> AgentState:
    """Nó de busca web: executa DuckDuckGo e chama LLM com os resultados.

    Entrada: state.query.
    Saída:   state.ranked_response (resultado da busca), state.confidence_score = 0.0
             (sem base vetorial para calcular similaridade).
    """
    try:
        resultados = search.run(state.query)
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return state.model_copy(update={
            "ranked_response": "Busca indisponível no momento.",
            "confidence_score": 0.0,
        })

    chain = _prompt_web | _get_llm() | StrOutputParser()
    return state.model_copy(update={
        "ranked_response": _invoke_chain(chain, {"context": resultados, "input": state.query}),
        "confidence_score": 0.0,
    })


def retrieve_info(state: AgentState) -> AgentState:
    """Nó de recuperação: busca documentos relevantes no ChromaDB via retriever vetorial.

    Entrada: state.query.
    Saída:   state.retrieved_info — lista de Document retornados pelo retriever.
    """
    retrieved_docs = _get_retriever().invoke(state.query)
    return state.model_copy(update={"retrieved_info": retrieved_docs})


def gera_multiplas_respostas(state: AgentState) -> AgentState:
    """Nó de geração: produz 5 respostas candidatas usando o LLM para posterior ranqueamento.

    Entrada: state.next_step (determina o prompt usado), state.retrieved_info (se "retrieve"),
             state.query.
    Saída:   state.possible_responses — lista de 5 dicts {"answer": str}.
    """
    def format_docs(docs):
        if not docs:
            return ""
        return "\n\n".join([doc.page_content for doc in docs])

    if state.next_step == "retrieve":
        context_text = format_docs(state.retrieved_info)
        current_prompt = prompt_rag_retrieve
        retrieved_info = state.retrieved_info
    else:
        context_text = ""
        current_prompt = prompt_gerar
        retrieved_info = []

    qa_chain_dynamic = current_prompt | _get_llm() | StrOutputParser()

    responses = []
    for _ in range(5):
        if state.next_step == "retrieve":
            response = _invoke_chain(qa_chain_dynamic, {"context": context_text, "input": state.query})
        else:
            response = _invoke_chain(qa_chain_dynamic, {"input": state.query})
        responses.append(response)

    return state.model_copy(update={
        "retrieved_info": retrieved_info,
        "possible_responses": [{"answer": r} for r in responses],
    })


def avalia_similaridade(state: AgentState) -> AgentState:
    """Nó de avaliação: calcula score de similaridade de cosseno entre cada resposta e o contexto recuperado.

    Entrada: state.retrieved_info, state.possible_responses.
    Saída:   state.similarity_scores — lista de floats, um por resposta candidata.

    Usa similaridade de cosseno porque respostas semanticamente próximas ao contexto recuperado
    têm maior fidelidade factual: quanto mais o espaço vetorial da resposta coincide com o do
    contexto, menos o LLM alucionou informações externas.
    """
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses
    retrieved_embeddings = (
        _get_embedding_model().embed_documents(retrieved_texts) if retrieved_texts else []
    )
    response_texts = [
        r["answer"] if isinstance(r, dict) and "answer" in r else str(r)
        for r in responses
    ]
    response_embeddings = (
        _get_embedding_model().embed_documents(response_texts) if response_texts else []
    )

    if not retrieved_embeddings or not response_embeddings:
        return state.model_copy(update={"similarity_scores": [0.0] * len(response_texts)})

    similarities = [
        np.mean(
            [
                cosine_similarity([re], [de])[0][0]
                for de in retrieved_embeddings
            ]
        )
        for re in response_embeddings
    ]
    return state.model_copy(update={"similarity_scores": similarities})


def rank_respostas(state: AgentState) -> AgentState:
    """Nó de ranqueamento: seleciona a resposta com maior score de similaridade como resposta final.

    Entrada: state.possible_responses, state.similarity_scores.
    Saída:   state.ranked_response — melhor resposta (str ou dict); state.confidence_score — score vencedor.
    """
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores))
    if response_with_scores:
        ranked = sorted(response_with_scores, key=lambda x: x[1], reverse=True)
        return state.model_copy(update={
            "ranked_response": ranked[0][0],
            "confidence_score": ranked[0][1],
        })
    return state.model_copy(update={
        "ranked_response": "Desculpe, não encontrei informações relevantes.",
        "confidence_score": 0.0,
    })


# Workflow
workflow = StateGraph(AgentState)
workflow.add_node("decision", passo_decisao_agente)
workflow.add_node("retrieve", retrieve_info)
workflow.add_node("generate_multiple", gera_multiplas_respostas)
workflow.add_node("evaluate_similarity", avalia_similaridade)
workflow.add_node("rank_responses", rank_respostas)
workflow.add_node("usar_web", usar_ferramenta_web)
workflow.set_entry_point("decision")
workflow.add_conditional_edges(
    "decision",
    lambda s: {"retrieve": "retrieve", "gerar": "generate_multiple", "usar_web": "usar_web"}[
        s.next_step
    ],
)
workflow.add_edge("retrieve", "generate_multiple")
workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")

agent_workflow = workflow.compile()
