# AgenticLog - Geração e ranqueamento (ADR-018 Fase 4)
"""
Módulo de geração e ranqueamento de respostas LLM.

Extraído de `agent.py` (ADR-018 Fase 4; fachada deletada na Fase 6). Contém o
Protocol LLMClient, decorator _llm_retry, prompts, o singleton `_llm` + `_get_llm`,
_invoke_chain, gera_multiplas_respostas, avalia_similaridade e rank_respostas.

O singleton `_llm` vive neste módulo (realocado de `agent.py` na Fase 6). Os
demais singletons (`_get_embedding_model`, `_get_vector_db`) são acessados via
lazy imports de `agenticlog.retrieval.retriever` DENTRO de funções (DN-2), para
preservar o grafo de imports acíclico.
"""

import logging
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np  # type: ignore[import-untyped]
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agenticlog.config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MAX_RETRY_ATTEMPTS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_WAIT_INITIAL_SECONDS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    NUM_CANDIDATE_RESPONSES,
)

# Import de state é seguro: state.py não importa agent (mesmo pacote, DAG)
from agenticlog.retrieval.state import AgentState

logger = logging.getLogger(__name__)

# Singleton do LLM — inicializado na primeira chamada (local, não mais em agent.py).
# Anotação como string (forward-ref): LLMClient é definido mais abaixo neste módulo.
_llm: "LLMClient | None" = None

_llm_retry = retry(
    stop=stop_after_attempt(LLM_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(min=LLM_RETRY_WAIT_INITIAL_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS),
    retry=retry_if_exception_type(
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            APIConnectionError,
        )
    ),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


@runtime_checkable
class LLMClient(Protocol):
    """Interface estrutural mínima do cliente LLM usada pelo pipeline de geração/ranqueamento.

    Cobre apenas as operações utilizadas nas chains (`prompt | _get_llm() | parser`):
    composição via pipe (__or__/__ror__) e invocação (invoke). `ChatOpenAI` satisfaz
    este Protocol estruturalmente — nenhum wrapper/adapter/subclasse é necessário.
    """

    def __or__(self, other: Any) -> Any: ...

    def __ror__(self, other: Any) -> Any: ...

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


def _get_llm() -> LLMClient:
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


# Prompts — inicializados na importação do módulo
prompt_rag_retrieve = PromptTemplate.from_template(
    """You are a truthful and precise assistant in logistics and supply chain.
    Your task is to answer the user's question based STRICTLY on the provided context below.

    # REGRAS DE RESPOSTA: restrições obrigatórias que impedem o LLM de \
alucinar ou usar conhecimento externo
    REGRAS DE RESPOSTA:
    1. USE ONLY the information inside the context block.
    2. DO NOT use your internal knowledge or previous training.
    3. If the answer is not in the context, reply exactly: "Sorry, I did not \
find that information in the documents."
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


@_llm_retry
def _invoke_chain(chain, inputs: dict) -> str:
    """Invoca uma chain LangChain com retry exponential backoff em erros de conexão/timeout.

    Entrada: chain — pipeline prompt | llm | parser; inputs — dict de variáveis do prompt.
    Saída:   resposta gerada pelo LLM como string.
    """
    return chain.invoke(inputs)


def gera_multiplas_respostas(state: AgentState) -> AgentState:
    """Nó de geração: produz NUM_CANDIDATE_RESPONSES respostas candidatas via LLM para ranqueamento.

    Entrada: state.next_step (determina o prompt usado), state.retrieved_info (se "retrieve"),
             state.query.
    Saída:   state.possible_responses — lista de NUM_CANDIDATE_RESPONSES dicts {"answer": str}.
             Com LLM_TEMPERATURE=0 o default é 1 candidata (gerar N idênticas seria desperdício).
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
    for _ in range(NUM_CANDIDATE_RESPONSES):
        if state.next_step == "retrieve":
            response = _invoke_chain(qa_chain_dynamic,
                                       {"context": context_text, "input": state.query})
        else:
            response = _invoke_chain(qa_chain_dynamic, {"input": state.query})
        responses.append(response)

    return state.model_copy(update={
        "retrieved_info": retrieved_info,
        "possible_responses": [{"answer": r} for r in responses],
    })


def avalia_similaridade(state: AgentState) -> AgentState:
    """Nó de avaliação: calcula score de similaridade de cosseno entre cada \
resposta e o contexto recuperado.

    Entrada: state.retrieved_info, state.possible_responses.
    Saída:   state.similarity_scores — lista de floats, um por resposta candidata.

    Usa similaridade de cosseno porque respostas semanticamente próximas ao contexto recuperado
    têm maior fidelidade factual: quanto mais o espaço vetorial da resposta coincide com o do
    contexto, menos o LLM alucionou informações externas.
    """
    from agenticlog.retrieval.retriever import (
        _get_embedding_model as _get_emb_wrapper,  # lazy — getter local (Fase 6)
    )
    _emb_model = _get_emb_wrapper()

    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses
    retrieved_embeddings = (
        _emb_model.embed_documents(retrieved_texts) if retrieved_texts else []
    )
    response_texts = [
        r["answer"] if isinstance(r, dict) and "answer" in r else str(r)
        for r in responses
    ]
    response_embeddings = (
        _emb_model.embed_documents(response_texts) if response_texts else []
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
    """Nó de ranqueamento: seleciona a resposta com maior score de similaridade \
como resposta final.

    Entrada: state.possible_responses, state.similarity_scores.
    Saída:   state.ranked_response — melhor resposta (str ou dict); \
state.confidence_score — score vencedor.
    """
    response_with_scores = list(
        zip(state.possible_responses, state.similarity_scores, strict=False)
    )
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
