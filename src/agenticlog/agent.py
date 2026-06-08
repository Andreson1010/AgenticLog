# AgenticLog - Lógica Agentic RAG
"""
Sistema Agentic RAG com LangGraph: RAG vetorial, geração conceitual e busca web.

O grafo de estados roteia cada consulta para um de três caminhos:
- retrieve  → busca no ChromaDB, gera múltiplas respostas, rankeia por similaridade de cosseno.
- gerar     → gera resposta conceitual diretamente com o LLM (sem retrieval).
- usar_web  → delega a um agente DuckDuckGo para consultas que exigem informações recentes.
"""

import hashlib
import logging
import os
import warnings

import anthropic
import httpx
import numpy as np  # type: ignore[import-untyped]
from langchain_chroma import Chroma
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph  # type: ignore[import-untyped]
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

warnings.filterwarnings("ignore")

import torch  # noqa: E402

torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agenticlog.config import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DIR_VECTORDB,
    EMBEDDING_MODEL,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MAX_RETRY_ATTEMPTS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_WAIT_INITIAL_SECONDS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
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
_vector_dbs: dict[str, "Chroma"] = {}
_embedding_model = None


def _get_llm() -> ChatOpenAI:
    """Retorna o singleton do LLM, criando-o na primeira chamada.

    Saída: instância de ChatOpenAI configurada com as constantes do config.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(  # type: ignore[call-arg]
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


def _get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME) -> Chroma:
    """Retorna o singleton do ChromaDB para a coleção indicada, criando-o na primeira chamada.

    Entrada: collection_name — nome da coleção ChromaDB.
    Saída: instância de Chroma conectada ao diretório persistido para essa coleção.
    """
    if collection_name not in _vector_dbs:
        _vector_dbs[collection_name] = Chroma(
            persist_directory=str(DIR_VECTORDB),
            collection_name=collection_name,
            embedding_function=_get_embedding_model(),
        )
    return _vector_dbs[collection_name]


def _listar_colecoes() -> list[str]:
    """Retorna os nomes de todas as coleções presentes no ChromaDB persistido em disco.

    Usa lazy import de chromadb para evitar efeitos colaterais na importação do módulo.
    Saída: lista de nomes de coleção (strings). Retorna [DEFAULT_COLLECTION_NAME] em caso de
    erro ou resultado vazio para garantir que o agente sempre consulte ao menos uma coleção.
    """
    try:
        import chromadb  # lazy — evita side-effects na importação do módulo
        client = chromadb.PersistentClient(path=str(DIR_VECTORDB))
        collections = client.list_collections()
        # chromadb 0.4.x returns Collection objects with .name attribute
        # chromadb >= 0.6 may return strings — handle both
        names = [
            col.name if hasattr(col, "name") else str(col)
            for col in collections
        ]
        if not names:
            return [DEFAULT_COLLECTION_NAME]
        return names
    except Exception as exc:
        logger.warning("Falha ao listar coleções ChromaDB; usando coleção padrão. Detalhe: %s", exc)
        return [DEFAULT_COLLECTION_NAME]


def _get_retriever(query: str) -> list[Document]:
    """Executa fan-out em todas as coleções ChromaDB e retorna até 3 documentos únicos.

    Coleção vazia contribui 0 documentos (skip silencioso).
    Erro ChromaDB em qualquer coleção propaga imediatamente (fail-fast).
    """
    collection_names = _listar_colecoes()
    all_docs: list[Document] = []

    for name in collection_names:
        db = _get_vector_db(name)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        all_docs.extend(docs)

    seen: set[str] = set()
    unique: list[Document] = []
    for doc in all_docs:
        key = hashlib.md5(doc.page_content.encode()).hexdigest()  # nosec B324
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique[:3]


def inicializar_recursos() -> None:
    """Inicializa singletons do agente (LLM, ChromaDB, embeddings) na inicialização do servidor.

    Entrada: nenhuma
    Saída: nenhuma — efeito colateral: singletons globais inicializados

    Ordem de inicialização: embeddings → vector_db → llm.
    Chamada única a partir do lifespan do FastAPI; elimina race condition em requisições concorrentes.
    """
    _get_embedding_model()
    _get_vector_db(DEFAULT_COLLECTION_NAME)
    _get_llm()


def invalidar_vector_db() -> None:
    """Invalida todos os singletons de ChromaDB para que a próxima chamada reconecte ao disco.

    Entrada: nenhuma.
    Saída: nenhuma.
    Efeito colateral: limpa o dicionário _vector_dbs global.
    """
    _vector_dbs.clear()


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
    """Nó de recuperação: busca documentos relevantes no ChromaDB via fan-out multi-coleção.

    Entrada: state.query.
    Saída:   state.retrieved_info — lista de Document retornados pelo retriever.
    """
    retrieved_docs = _get_retriever(state.query)
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
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores, strict=False))
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
