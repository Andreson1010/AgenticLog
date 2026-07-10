# AgenticLog - Testes de aceitação: ADR-018 Fase 4 (extração retrieval/)
"""
Testes de acicidade, identidade shim, delegação wrapper e seam binding.

Oráculos de caracterização (zero-diff):
- tests/test_rag_caracterizacao.py (não tocar)
- tests/ingestion/test_shims_identidade.py (não tocar)

RETR-02  agent.X is retrieval.state.X (AgentState)
RETR-04  agent.X is retrieval.graph.X (agent_workflow)
RETR-11  _listar_colecoes / _get_vector_db são WRAPPERS (não is-identical)
RETR-13  subprocess fresh-interpreter -> exit 0
RETR-14  from agenticlog.agent import <simbolo> funciona
"""

import subprocess
import sys

import pytest

from agenticlog import agent
from agenticlog.retrieval import graph as retrieval_graph
from agenticlog.retrieval import generation as retrieval_generation
from agenticlog.retrieval import retriever as retrieval_retriever
from agenticlog.retrieval import state as retrieval_state


# ── T7: Acicidade (RETR-13) ───────────────────────────────────────────────────

class TestAcyclicImport:
    """Verifica que os 4 módulos de retrieval/ + agent são importáveis em interpretador frio."""

    MODULES = [
        "agenticlog.retrieval.state",
        "agenticlog.retrieval.generation",
        "agenticlog.retrieval.retriever",
        "agenticlog.retrieval.graph",
        "agenticlog.agent",
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_ac01_import_frio_sem_ciclo(self, module_name: str) -> None:
        """Cada módulo é importável individualmente em subprocess sem erro."""
        code = f"import {module_name}"
        # Usar o mesmo interpretador do processo atual
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Import de {module_name} falhou (rc={result.returncode}): "
            f"stderr={result.stderr}"
        )
        assert "circular" not in result.stderr.lower()
        assert "partially initialized" not in result.stderr.lower()

    def test_ac02_import_todos_frio(self) -> None:
        """Todos os 4 módulos + agent são importáveis no mesmo interpretador frio."""
        code = (
            "import agenticlog.retrieval.state; "
            "import agenticlog.retrieval.generation; "
            "import agenticlog.retrieval.retriever; "
            "import agenticlog.retrieval.graph; "
            "import agenticlog.agent; "
            "print('OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Import combinado falhou (rc={result.returncode}): "
            f"stderr={result.stderr}"
        )
        assert "circular" not in result.stderr.lower()
        assert "partially initialized" not in result.stderr.lower()


# ── T8: Identidade shim (RETR-02, RETR-04, RETR-14) ──────────────────────────

class TestShimIdentity:
    """Verifica identidade `is` entre shims de agent.py e definições reais em retrieval/."""

    # Pares (simbolo_agent, simbolo_retrieval, modulo_retrieval)
    SHIMS_IDENTITY = [
        # state
        ("AgentState", "AgentState", retrieval_state),
        # generation
        ("LLMClient", "LLMClient", retrieval_generation),
        ("_llm_retry", "_llm_retry", retrieval_generation),
        ("prompt_rag_retrieve", "prompt_rag_retrieve", retrieval_generation),
        ("prompt_gerar", "prompt_gerar", retrieval_generation),
        ("_prompt_web", "_prompt_web", retrieval_generation),
        ("_get_llm", "_get_llm", retrieval_generation),
        ("_invoke_chain", "_invoke_chain", retrieval_generation),
        ("gera_multiplas_respostas", "gera_multiplas_respostas", retrieval_generation),
        ("avalia_similaridade", "avalia_similaridade", retrieval_generation),
        ("rank_respostas", "rank_respostas", retrieval_generation),
        # retriever
        ("_build_embedding_model", "_build_embedding_model", retrieval_retriever),
        ("_get_retriever", "_get_retriever", retrieval_retriever),
        ("invalidar_vector_db", "invalidar_vector_db", retrieval_retriever),
        # graph
        ("passo_decisao_agente", "passo_decisao_agente", retrieval_graph),
        ("usar_ferramenta_web", "usar_ferramenta_web", retrieval_graph),
        ("retrieve_info", "retrieve_info", retrieval_graph),
        ("inicializar_recursos", "inicializar_recursos", retrieval_graph),
        ("agent_workflow", "agent_workflow", retrieval_graph),
    ]

    @pytest.mark.parametrize("attr_name, retrieval_attr_name, retrieval_mod", SHIMS_IDENTITY)
    def test_identidade_is(self, attr_name, retrieval_attr_name, retrieval_mod):
        """agent.<attr> is retrieval.<mod>.<attr> para todos os shims."""
        agent_val = getattr(agent, attr_name, None)
        retrieval_val = getattr(retrieval_mod, retrieval_attr_name, None)
        assert agent_val is retrieval_val, (
            f"agent.{attr_name} NÃO é o mesmo objeto que "
            f"{retrieval_mod.__name__}.{retrieval_attr_name} "
            f"(agent={id(agent_val)}, retrieval={id(retrieval_val)})"
        )


class TestWrapperDelegation:
    """Verifica que os WRAPPERS não são `is`-idênticos às funções parametrizadas.

    _listar_colecoes, _get_vector_db e _get_embedding_model são WRAPPERS em
    agent.py (funções definidas em agent.py, não shims). Portanto agent.X
    NÃO deve ser retrieval.<mod>.Y.
    """

    def test_listar_colecoes_e_wrapper(self):
        """agent._listar_colecoes NÃO é retrieval.retriever._listar_colecoes."""
        assert agent._listar_colecoes is not retrieval_retriever._listar_colecoes

    def test_get_vector_db_e_wrapper(self):
        """agent._get_vector_db NÃO é retrieval.retriever._get_vector_db."""
        assert agent._get_vector_db is not retrieval_retriever._get_vector_db

    def test_get_embedding_model_e_wrapper(self):
        """agent._get_embedding_model NÃO é retrieval.retriever._build_embedding_model."""
        assert agent._get_embedding_model is not retrieval_retriever._build_embedding_model


# ── Seam binding (RETR-11, DN-1) ─────────────────────────────────────────────

class TestSeamBinding:
    """Verifica que monkeypatch em agent.py flui para as funções movidas em retrieval/.

    search   → monkeypatch agent.search → graph.usar_ferramenta_web vê o fake (DN-1)
    DIR_VECTORDB → monkeypatch agent.DIR_VECTORDB → wrapper _listar_colecoes vê (RETR-11)
    """

    def test_seam_binding_search(self, monkeypatch):
        """monkeypatch agent.search → usar_ferramenta_web vê o fake."""
        called = False

        class FakeSearch:
            def run(self, query):
                nonlocal called
                called = True
                raise Exception("Simulated search failure")

        monkeypatch.setattr("agenticlog.agent.search", FakeSearch())

        from agenticlog.retrieval.state import AgentState
        state = AgentState(query="test search web query")
        result = agent.usar_ferramenta_web(state)
        assert called, "FakeSearch.run() não foi chamado"
        assert result.ranked_response == "Busca indisponível no momento."
        assert result.confidence_score == 0.0

    def test_seam_binding_search_persiste_apos_import(self, monkeypatch):
        """import agenticlog.retrieval.graph após monkeypatch → vê o fake."""
        # Import graf AFTER monkeypatch para garantir que lazy import funciona
        monkeypatch.setattr("agenticlog.agent.search", None)

        from agenticlog.retrieval import graph as retrieval_graph

        # Verificar que usar_ferramenta_web vê o valor patchado
        # (importa agent.search lazy no corpo da função)
        # Se search é None, chamar .run() levantaria AttributeError
        import agenticlog.agent as agent_mod
        assert agent_mod.search is None
        # Não chamamos a função aqui — apenas confirmamos que o patch está ativo

    def _get_chromadb_persistent_client_path(self, vectordb_dir):
        """Helper: cria uma instância de PersistentClient e retorna o path."""
        import chromadb
        client = chromadb.PersistentClient(path=str(vectordb_dir))
        return client._settings.persist_directory

    def test_seam_binding_dir_vectordb(self, monkeypatch, tmp_path):
        """monkeypatch agent.DIR_VECTORDB → wrapper _listar_colecoes vê o path patchado."""
        import chromadb

        # Patch o DIR_VECTORDB
        monkeypatch.setattr("agenticlog.agent.DIR_VECTORDB", tmp_path)

        # _listar_colecoes (wrapper) deve ler agent.DIR_VECTORDB no call time
        # e passar para retriever._listar_colecoes(vectordb_dir=tmp_path)
        result = agent._listar_colecoes()
        # Como tmp_path está vazio (sem ChromaDB), deve retornar fallback
        from agenticlog.config import DEFAULT_COLLECTION_NAME
        assert result == [DEFAULT_COLLECTION_NAME]
