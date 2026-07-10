# AgenticLog - Lógica Agentic RAG (fachada)
"""
Sistema Agentic RAG com LangGraph: RAG vetorial, geração conceitual e busca web.

Facade de compatibilidade (ADR-018 Fase 4): mantém singletons físicos + `search` +
3 WRAPPERS de ligação de seams + ~20 shims identity-preserving dos módulos
`agenticlog.retrieval.*`.

O comportamento real está em:
- `agenticlog.retrieval.state`      (AgentState)
- `agenticlog.retrieval.generation` (LLMClient, prompts, geração, ranqueamento)
- `agenticlog.retrieval.retriever`  (ChromaDB, fan-out, invalidação)
- `agenticlog.retrieval.graph`      (nós, FSM, workflow compilado)

Os singletons `_llm`/`_embedding_model`/`_vector_dbs` e o global `search`
PERMANECEM físicos em agent.py (não movidos) — preservando monkeypatch do oráculo.
"""

import logging
import os
import warnings

from langchain_chroma import Chroma
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

import torch  # noqa: E402

torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agenticlog.config import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DIR_VECTORDB,
)

logger = logging.getLogger(__name__)

# Singletons lazy — inicializados somente na primeira chamada, não na importação
_llm = None
_vector_dbs: dict[str, Chroma] = {}
_embedding_model = None

# Ferramentas de busca web — DuckDuckGo não requer LMStudio, inicializado na importação
search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=5)

import agenticlog.retrieval.retriever as _retr  # noqa: E402

# ── WRAPPERS (ADR-018 Fase 4) — remover na Fase 6 ─────────────────────────────
# NÃO são `is`-idênticos aos de `retrieval.*`: ligam seams de agent.py
# (DIR_VECTORDB, _embedding_model) NO MOMENTO DA CHAMADA, preservando o
# monkeypatch do oráculo (§4 do design; ADR-019 D4).

def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Retorna o singleton do modelo de embeddings, criando-o na primeira chamada.

    Wrapper (DN-2a): lê/seta o singleton _embedding_model global de agent.py
    diretamente e delega a construção a _retr._build_embedding_model().
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _retr._build_embedding_model()
    return _embedding_model


def _listar_colecoes() -> list[str]:
    """Wrapper: resolve agent.DIR_VECTORDB no call time (RETR-11, ADR-019 D4)."""
    return _retr._listar_colecoes(vectordb_dir=DIR_VECTORDB)


def _get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME) -> Chroma:
    """Wrapper: resolve agent.DIR_VECTORDB + singleton _vector_dbs no call time."""
    return _retr._get_vector_db(collection_name, vectordb_dir=DIR_VECTORDB)


# ── Re-export shims de state (ADR-018 Fase 4) — remover na Fase 6 ──────
# ── Re-export shims de generation (ADR-018 Fase 4) — remover na Fase 6 ─
from agenticlog.retrieval.generation import (  # noqa: E402,F401
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

# ── Re-export shims de graph (ADR-018 Fase 4) — remover na Fase 6 ──────
from agenticlog.retrieval.graph import (  # noqa: E402,F401
    agent_workflow,
    inicializar_recursos,
    passo_decisao_agente,
    retrieve_info,
    usar_ferramenta_web,
)

# ── Re-export shims de retriever (ADR-018 Fase 4) — remover na Fase 6 ──
from agenticlog.retrieval.retriever import (  # noqa: E402,F401
    _build_embedding_model,
    _get_retriever,
    invalidar_vector_db,
)
from agenticlog.retrieval.state import AgentState  # noqa: E402,F401
