# AgenticLog - Pacote de recuperação (ADR-018 Fase 4)
"""Módulo de retrieval: estado, geração, retriever vetorial e grafo LangGraph.

Extraído de `agenticlog.agent` (Fase 4 da ADR-018). Contém as 4 camadas:
- state      — AgentState (Pydantic)
- generation  — LLMClient Protocol, prompts, geração e ranqueamento
- retriever   — conexão ChromaDB, fan-out multi-coleção, invalidação
- graph       — grafo LangGraph: nós, FSM, workflow compilado

Nenhum destes módulos importa `agent` em nível de módulo (top-level) —
todos os acessos a singletons de `agent.py` são lazy imports dentro de
corpos de função, garantindo acicidade.
"""

from agenticlog.retrieval.state import AgentState
from agenticlog.retrieval.generation import (
    LLMClient,
    _invoke_chain,
    _get_llm,
    _llm_retry,
    avalia_similaridade,
    gera_multiplas_respostas,
    prompt_gerar,
    prompt_rag_retrieve,
    rank_respostas,
    _prompt_web,
)
from agenticlog.retrieval.retriever import (
    _build_embedding_model,
    _get_retriever,
    invalidar_vector_db,
)
from agenticlog.retrieval.graph import (
    agent_workflow,
    inicializar_recursos,
    passo_decisao_agente,
    retrieve_info,
    usar_ferramenta_web,
)

__all__ = [
    "AgentState",
    "LLMClient",
    "_invoke_chain",
    "_get_llm",
    "_llm_retry",
    "avalia_similaridade",
    "gera_multiplas_respostas",
    "prompt_gerar",
    "prompt_rag_retrieve",
    "rank_respostas",
    "_prompt_web",
    "_build_embedding_model",
    "_get_retriever",
    "invalidar_vector_db",
    "agent_workflow",
    "inicializar_recursos",
    "passo_decisao_agente",
    "retrieve_info",
    "usar_ferramenta_web",
]
