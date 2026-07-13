# AgenticLog - Estado do agente (AgentState)
"""Modelo Pydantic de estado imutável por convenção entre nós do grafo LangGraph.

Extraído verbatim de `agent.py` (ADR-018 Fase 4). NÃO importa `agent`,
`config` ou qualquer outro módulo do projeto — apenas `pydantic`.
"""

from pydantic import BaseModel


class AgentState(BaseModel):
    """Carregador de estado imutável por convenção entre os nós do grafo LangGraph.

    Cada nó recebe uma cópia do estado e retorna um novo estado atualizado;
    nenhum nó deve modificar campos de outros nós diretamente.
    """

    query: str                          # pergunta original do usuário
    next_step: str = ""                 # rota decidida: "retrieve", "gerar" ou "usar_web"
    retrieved_info: list = []           # documentos retornados pelo retriever vetorial
    possible_responses: list = []       # NUM_CANDIDATE_RESPONSES respostas do LLM para ranqueamento
    similarity_scores: list = []        # scores de cosseno de cada resposta vs. contexto recuperado
    ranked_response: str = ""           # melhor resposta após ranqueamento por similaridade
    confidence_score: float = 0.0       # score de confiança da resposta ranqueada (0.0–1.0)
