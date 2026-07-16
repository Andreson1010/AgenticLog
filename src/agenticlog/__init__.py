# AgenticLog - Agentic RAG para logística e supply chain
"""Pacote principal do AgenticLog.

Sistema de Agentic RAG para consultas de logística e supply chain, construído com LangGraph.
Roteia cada consulta entre recuperação vetorial (ChromaDB), geração conceitual (LLM) e busca web (DuckDuckGo).

Exportações públicas:
- AgentState                  — modelo Pydantic de estado imutável por convenção; carregado entre nós do grafo.
- agent_workflow              — grafo LangGraph compilado; invocar com AgentState(query=...) para obter resposta.
- check_lmstudio_health       — GET /v1/models antes do workflow; fast-fail sem retry; valida LLM_MODEL carregado.
- LMStudioUnavailableError    — LMStudio inacessível ou HTTP não-2xx no health check.
- ModeloNaoCarregadoError     — LMStudio responde mas LLM_MODEL não está na lista de /models.

Para detalhes do fluxo completo de nós e roteamento, consulte agent.py.
"""

from agenticlog.retrieval.graph import agent_workflow
from agenticlog.retrieval.state import AgentState
from agenticlog.serving.health import (
    LMStudioUnavailableError,
    ModeloNaoCarregadoError,
    check_lmstudio_health,
)

__all__ = [
    "AgentState",
    "agent_workflow",
    "check_lmstudio_health",
    "LMStudioUnavailableError",
    "ModeloNaoCarregadoError",
]
