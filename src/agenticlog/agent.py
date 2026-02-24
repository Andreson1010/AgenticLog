# AgenticLog - Lógica Agentic RAG
"""
Sistema Agentic RAG: RAG vetorial, geração conceitual e busca web.
"""

import os
import numpy as np  # type: ignore
from langgraph.graph import StateGraph  # type: ignore[reportMissingImports]
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
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
)

# LLM
llm = ChatOpenAI(
    model_name=LLM_MODEL,
    openai_api_base=LLM_API_BASE,
    openai_api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

try:
    response = llm.invoke("Just respond with the word: CONNECTED")
    print(f"Status do {LLM_MODEL}: {response.content}")
except Exception as e:
    print(f"Connection error: {e}")

# Embeddings e VectorDB
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(
    persist_directory=str(DIR_VECTORDB),
    embedding_function=embedding_model,
)
retriever = vector_db.as_retriever()

# Prompts
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

prompt_gerar = PromptTemplate.from_template(
    """You are a specialist in logistics and supply chain.
    Your task is to explain, summarize or define the requested concept.
    Answer using your general knowledge in a clear and professional way, in Brazilian Portuguese.

    User Question: {input}
    Answer:"""
)

# Agente web
search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)
web_search_tool = Tool(name="WebSearch", func=search.run, description="Busca web.")
avk_agent_executor = initialize_agent(
    tools=[web_search_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)


class AgentState(BaseModel):
    query: str
    next_step: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0


def passo_decisao_agente(state: AgentState) -> AgentState:
    query = state.query.lower()
    if any(
        p in query
        for p in [
            "explain", "summarize", "define", "concept", "general", "what is",
            "explique", "resuma", "defina", "conceito", "geral", "o que é",
        ]
    ):
        state.next_step = "gerar"
    elif any(
        p in query
        for p in [
            "search the web", "news", "updated", "recent", "latest information",
            "busque na web", "notícias", "atualizado", "recente", "últimas informações",
        ]
    ):
        state.next_step = "usar_web"
    else:
        state.next_step = "retrieve"
    return state


def usar_ferramenta_web(state: AgentState) -> AgentState:
    try:
        result = avk_agent_executor.invoke(state.query)
        state.ranked_response = result.get("output", "No information obtained by web search.")
    except Exception as e:
        state.ranked_response = f"Erro na busca web: {e}. Tente novamente mais tarde."
    state.confidence_score = 0.0
    return state


def retrieve_info(state: AgentState) -> AgentState:
    retrieved_docs = retriever.invoke(state.query)
    state.retrieved_info = retrieved_docs
    return state


def gera_multiplas_respostas(state: AgentState) -> AgentState:
    def format_docs(docs):
        if not docs:
            return ""
        return "\n\n".join([doc.page_content for doc in docs])

    if state.next_step == "retrieve":
        context_text = format_docs(state.retrieved_info)
        current_prompt = prompt_rag_retrieve
    else:
        context_text = ""
        current_prompt = prompt_gerar
        state.retrieved_info = []

    qa_chain_dynamic = current_prompt | llm | StrOutputParser()

    responses = []
    for _ in range(5):
        if state.next_step == "retrieve":
            response = qa_chain_dynamic.invoke(
                {"context": context_text, "input": state.query}
            )
        else:
            response = qa_chain_dynamic.invoke({"input": state.query})
        responses.append(response)

    state.possible_responses = [{"answer": r} for r in responses]
    return state


def avalia_similaridade(state: AgentState) -> AgentState:
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses
    retrieved_embeddings = (
        embedding_model.embed_documents(retrieved_texts) if retrieved_texts else []
    )
    response_texts = [
        r["answer"] if isinstance(r, dict) and "answer" in r else str(r)
        for r in responses
    ]
    response_embeddings = (
        embedding_model.embed_documents(response_texts) if response_texts else []
    )

    if not retrieved_embeddings or not response_embeddings:
        state.similarity_scores = [0.0] * len(response_texts)
        return state

    similarities = [
        np.mean(
            [
                cosine_similarity([re], [de])[0][0]
                for de in retrieved_embeddings
            ]
        )
        for re in response_embeddings
    ]
    state.similarity_scores = similarities
    return state


def rank_respostas(state: AgentState) -> AgentState:
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores))
    if response_with_scores:
        ranked = sorted(response_with_scores, key=lambda x: x[1], reverse=True)
        state.ranked_response = ranked[0][0]
        state.confidence_score = ranked[0][1]
    else:
        state.ranked_response = "Desculpe, não encontrei informações relevantes."
        state.confidence_score = 0.0
    return state


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
