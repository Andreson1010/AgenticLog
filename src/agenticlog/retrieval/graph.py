# AgenticLog - Grafo LangGraph (ADR-018 Fase 4)
"""
Orquestração do grafo LangGraph: nós, FSM routing, workflow compilado.

Extraído de `agent.py` (ADR-018 Fase 4). Contém os 6 nós do grafo, a
compilação do StateGraph e a inicialização de recursos.

`search` permanece FISICAMENTE em `agent.py` (DN-1) — `usar_ferramenta_web`
o acessa via lazy import dentro do corpo da função.

`inicializar_recursos` acessa `_get_embedding_model`, `_get_llm` e
`_get_vector_db` via lazy imports de `agent.py` para garantir que os
singletons sejam criados no namespace de `agent`.

NADA importa `agent` em nível de módulo — todos os acessos são lazy
imports DENTRO de funções (DN-2, RETR-13).
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph  # type: ignore[import-untyped]

from agenticlog.config import (
    DEFAULT_COLLECTION_NAME,
    ROUTING_KEYWORDS_WEB,
)
from agenticlog.retrieval.generation import (
    _get_llm,
    _invoke_chain,
    _prompt_web,
    avalia_similaridade,
    gera_multiplas_respostas,
    rank_respostas,
)
from agenticlog.retrieval.retriever import _get_retriever
from agenticlog.retrieval.state import AgentState

logger = logging.getLogger(__name__)


def passo_decisao_agente(state: AgentState) -> AgentState:
    """Nó de decisão do grafo: define state.next_step com base em palavras-chave da consulta.

    Entrada: state.query (pergunta do usuário).
    Saída:   state.next_step — "usar_web" ou "retrieve" (padrão).

    Estratégia retrieve-first: termos de busca web → "usar_web"; todo o resto → "retrieve".
    A geração direta ("gerar") não é mais uma rota primária — vira fallback no nó retrieve
    quando a base vetorial não retorna nenhum documento (ver retrieve_info).
    """
    query = state.query.lower()
    if any(p in query for p in ROUTING_KEYWORDS_WEB):
        return state.model_copy(update={"next_step": "usar_web"})
    return state.model_copy(update={"next_step": "retrieve"})


def usar_ferramenta_web(state: AgentState) -> AgentState:
    """Nó de busca web: executa DuckDuckGo e chama LLM com os resultados.

    Entrada: state.query.
    Saída:   state.ranked_response (resultado da busca), state.confidence_score = 0.0
             (sem base vetorial para calcular similaridade).

    Acessa `search` via lazy import de `agenticlog.agent` (DN-1) — o global
    `search` permanece fisicamente em agent.py para preservar monkeypatch.
    """
    from agenticlog.agent import search  # lazy — resolvido a cada chamada (DN-1)

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
             Se a busca vier vazia, faz fallback para next_step="gerar" (geração
             direta sem contexto), seguindo a estratégia retrieve-first.
    """
    retrieved_docs = _get_retriever(state.query)
    if not retrieved_docs:
        # Fail-loud: retrieval vazio costuma indicar base não populada (coleção vazia/órfã),
        # não apenas ausência de match. Logamos WARNING para que a degradação não passe
        # silenciosa — o fallback para 'gerar' produz resposta SEM contexto da base.
        logger.warning(
            "Retrieval retornou 0 documentos para a query; caindo para geração direta "
            "(sem contexto). Verifique se o vector DB está populado."
        )
        return state.model_copy(update={"retrieved_info": [], "next_step": "gerar"})
    logger.debug("Retrieval retornou %d documentos.", len(retrieved_docs))
    return state.model_copy(update={"retrieved_info": retrieved_docs})


def inicializar_recursos() -> None:
    """Inicializa singletons do agente (LLM, ChromaDB, embeddings) na inicialização do servidor.

    Entrada: nenhuma
    Saída: nenhuma — efeito colateral: singletons globais inicializados

    Ordem de inicialização: embeddings → vector_db → llm.
    Chamada única a partir do lifespan do FastAPI; elimina race condition em requisições concorrentes.

    Acessa os 3 getters via lazy import de `agenticlog.agent` para garantir que
    os singletons sejam criados no namespace de `agent` (DN-2).
    """
    from agenticlog.agent import _get_embedding_model as _get_emb
    from agenticlog.agent import _get_llm as _get_llm_fn
    from agenticlog.agent import _get_vector_db as _get_vdb

    _get_emb()
    _get_vdb(DEFAULT_COLLECTION_NAME)
    _get_llm_fn()


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
    lambda s: {"retrieve": "retrieve", "usar_web": "usar_web"}[s.next_step],
)
workflow.add_edge("retrieve", "generate_multiple")
workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")

agent_workflow = workflow.compile()
