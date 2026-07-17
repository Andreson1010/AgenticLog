# AgenticLog - Testes de aceitação: ADR-018 Fase 4 (extração retrieval/)
"""
Testes de acicidade, delegação wrapper e seam binding.

TestShimIdentity removido na Fase 6 (shim agent.py deletado).
TestAcyclicImport.MODULES atualizado (agenticlog.agent removido).
"""

import subprocess
import sys

import pytest

from agenticlog.retrieval import retriever as retrieval_retriever

# ── T7: Acicidade (RETR-13) ───────────────────────────────────────────────────

class TestAcyclicImport:
    """Verifica que os módulos de retrieval/ são importáveis em interpretador frio."""

    MODULES = [
        "agenticlog.retrieval.state",
        "agenticlog.retrieval.generation",
        "agenticlog.retrieval.retriever",
        "agenticlog.retrieval.graph",
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_ac01_import_frio_sem_ciclo(self, module_name: str) -> None:
        """Cada módulo é importável individualmente em subprocess sem erro."""
        code = f"import {module_name}"
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
        """Todos os módulos de retrieval são importáveis no mesmo interpretador frio."""
        code = (
            "import agenticlog.retrieval.state; "
            "import agenticlog.retrieval.generation; "
            "import agenticlog.retrieval.retriever; "
            "import agenticlog.retrieval.graph; "
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


# ── Seam binding ─────────────────────────────────────────────────────────────

class TestSeamBinding:
    """Verifica que monkeypatch nas definições canônicas flui corretamente."""

    def test_seam_binding_search(self, monkeypatch):
        """monkeypatch graph.search → usar_ferramenta_web vê o fake."""
        called = False

        class FakeSearch:
            def run(self, query):
                nonlocal called
                called = True
                raise Exception("Simulated search failure")

        monkeypatch.setattr("agenticlog.retrieval.graph.search", FakeSearch())

        from agenticlog.retrieval.graph import usar_ferramenta_web
        from agenticlog.retrieval.state import AgentState
        state = AgentState(query="test search web query")
        result = usar_ferramenta_web(state)
        assert called, "FakeSearch.run() não foi chamado"
        assert result.ranked_response == "Busca indisponível no momento."
        assert result.confidence_score == 0.0

    def test_seam_binding_search_persiste_apos_import(self, monkeypatch):
        """Import graph após monkeypatch → vê o patch."""
        monkeypatch.setattr("agenticlog.retrieval.graph.search", None)

        import agenticlog.retrieval.graph as retrieval_graph_imported
        assert retrieval_graph_imported.search is None

    def _get_chromadb_persistent_client_path(self, vectordb_dir):
        """Helper: cria uma instância de PersistentClient e retorna o path."""
        import chromadb
        client = chromadb.PersistentClient(path=str(vectordb_dir))
        return client._settings.persist_directory

    def test_seam_binding_dir_vectordb(self, monkeypatch, tmp_path):
        """monkeypatch config.DIR_VECTORDB → _listar_colecoes vê o path patchado."""
        monkeypatch.setattr("agenticlog.config.DIR_VECTORDB", tmp_path)

        result = retrieval_retriever._listar_colecoes()
        from agenticlog.config import DEFAULT_COLLECTION_NAME
        assert result == [DEFAULT_COLLECTION_NAME]
