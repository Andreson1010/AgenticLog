# AgenticLog - Pacote de recuperação (ADR-018 Fase 4 + Fase 6)
"""Módulo de retrieval: estado, geração, retriever vetorial e grafo LangGraph.

Extraído de `agenticlog.agent` (Fase 4 da ADR-018). Contém as 4 camadas:
- state      — AgentState (Pydantic)
- generation  — LLMClient Protocol, prompts, geração e ranqueamento
- retriever   — conexão ChromaDB, fan-out multi-coleção, invalidação
- graph       — grafo LangGraph: nós, FSM, workflow compilado
"""

import os
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402

torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agenticlog.retrieval.generation import (  # noqa: E402
    LLMClient,
    _get_llm,
    _invoke_chain,
    _llm_retry,
    _prompt_web,
    avalia_similaridade,
    gera_multiplas_respostas,
    prompt_gerar,
    prompt_rag_retrieve,
    rank_respostas,
)
from agenticlog.retrieval.graph import (  # noqa: E402
    agent_workflow,
    inicializar_recursos,
    passo_decisao_agente,
    retrieve_info,
    usar_ferramenta_web,
)
from agenticlog.retrieval.retriever import (  # noqa: E402
    _build_embedding_model,
    _get_retriever,
    invalidar_vector_db,
)
from agenticlog.retrieval.state import AgentState  # noqa: E402

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
