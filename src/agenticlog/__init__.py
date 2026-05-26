# AgenticLog - Agentic RAG para logística e supply chain
"""Pacote principal do AgenticLog.

Sistema de Agentic RAG para consultas de logística e supply chain, construído com LangGraph.
Roteia cada consulta entre recuperação vetorial (ChromaDB), geração conceitual (LLM) e busca web (DuckDuckGo).

Exportações públicas:
- AgentState      — modelo Pydantic de estado imutável por convenção; carregado entre nós do grafo.
- agent_workflow  — grafo LangGraph compilado; invocar com AgentState(query=...) para obter resposta.

Para detalhes do fluxo completo de nós e roteamento, consulte agent.py.
"""

from agenticlog.agent import AgentState, agent_workflow

__all__ = ["AgentState", "agent_workflow"]
