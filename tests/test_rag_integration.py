# AgenticLog - Testes de integração para ingestão incremental no ChromaDB
"""
Testes de integração para adicionar_documento_incrementalmente.
Usam ChromaDB real em disco (tmp_path), sem chamadas LLM.
Marcar com @pytest.mark.integration.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import pytest

from agenticlog.rag import adicionar_documento_incrementalmente, RAGSecurityError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JQ_COMPATIBLE_DOC_A = json.dumps({"pedido": "P001", "status": "entregue"}).encode()
_JQ_COMPATIBLE_DOC_B = json.dumps({"pedido": "P002", "status": "pendente"}).encode()


def _ingerir(filename: str, conteudo: bytes, docs_dir: Path, vdb_dir: Path) -> dict:
    """Chama adicionar_documento_incrementalmente com dirs patchados."""
    with patch("agenticlog.rag.DIR_DOCUMENTS", new=docs_dir), \
         patch("agenticlog.rag.DIR_VECTORDB", new=vdb_dir), \
         patch("agenticlog.rag.invalidar_vector_db"):
        return adicionar_documento_incrementalmente(filename, conteudo)


# ---------------------------------------------------------------------------
# Integration test class
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIngestionIntegration:
    """Testes de integração — usam ChromaDB real em tmp_path, sem LLM."""

    def teste_1_dois_arquivos_chunks_acumulam(self, tmp_path: Path) -> None:
        """Dois uploads incrementais: chunk total = soma dos chunks de cada arquivo."""
        docs_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        docs_dir.mkdir()

        resultado_a = _ingerir("doc_a.json", _JQ_COMPATIBLE_DOC_A, docs_dir, vdb_dir)
        resultado_b = _ingerir("doc_b.json", _JQ_COMPATIBLE_DOC_B, docs_dir, vdb_dir)

        assert resultado_a["status"] == "adicionado"
        assert resultado_b["status"] == "adicionado"

        # Verificar que ambos os arquivos estão no vectordb
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vdb = Chroma(persist_directory=str(vdb_dir), embedding_function=emb)
        total = vdb.get()
        assert len(total["ids"]) >= 2, f"Esperava pelo menos 2 chunks, obteve {len(total['ids'])}"

    def teste_2_pre_existentes_intactos_apos_novo_upload(self, tmp_path: Path) -> None:
        """Chunks de arquivo A permanecem intactos após upload de arquivo B."""
        docs_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        docs_dir.mkdir()

        # Upload A — registrar IDs
        _ingerir("doc_a.json", _JQ_COMPATIBLE_DOC_A, docs_dir, vdb_dir)

        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vdb = Chroma(persist_directory=str(vdb_dir), embedding_function=emb)
        ids_a = set(vdb.get()["ids"])
        assert len(ids_a) > 0

        # Upload B
        _ingerir("doc_b.json", _JQ_COMPATIBLE_DOC_B, docs_dir, vdb_dir)

        # Verificar que todos os IDs de A ainda estão presentes
        vdb2 = Chroma(persist_directory=str(vdb_dir), embedding_function=emb)
        ids_apos = set(vdb2.get()["ids"])
        assert ids_a.issubset(ids_apos), f"IDs de A ausentes após upload de B: {ids_a - ids_apos}"

    def teste_3_primeira_ingestao_sem_colecao_existente(self, tmp_path: Path) -> None:
        """Sem vectordb existente: cria coleção e ingere sem erro."""
        docs_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        docs_dir.mkdir()
        # vdb_dir não existe ainda
        assert not vdb_dir.exists()

        resultado = _ingerir("primeiro.json", _JQ_COMPATIBLE_DOC_A, docs_dir, vdb_dir)

        assert resultado["status"] == "adicionado"
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vdb = Chroma(persist_directory=str(vdb_dir), embedding_function=emb)
        assert len(vdb.get()["ids"]) > 0

    def teste_4_rollback_nao_deixa_chunks_orfaos(self, tmp_path: Path) -> None:
        """Falha no add_documents: nenhum chunk órfão persistido após rollback."""
        docs_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        docs_dir.mkdir()

        # Upload A com sucesso
        _ingerir("doc_a.json", _JQ_COMPATIBLE_DOC_A, docs_dir, vdb_dir)

        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        vdb = Chroma(persist_directory=str(vdb_dir), embedding_function=emb)
        count_antes = len(vdb.get()["ids"])
        assert count_antes > 0

        # Upload B falha no add_documents — rollback deve limpar
        # Patchamos add_documents para sempre lançar exceção nesta chamada
        with patch("agenticlog.rag.DIR_DOCUMENTS", new=docs_dir), \
             patch("agenticlog.rag.DIR_VECTORDB", new=vdb_dir), \
             patch("agenticlog.rag.invalidar_vector_db"), \
             patch.object(Chroma, "add_documents", side_effect=RuntimeError("falha simulada")):
            with pytest.raises(RuntimeError, match="falha simulada"):
                adicionar_documento_incrementalmente("doc_b.json", _JQ_COMPATIBLE_DOC_B)

        # Verificar que o count não aumentou (rollback funcionou)
        vdb3 = Chroma(persist_directory=str(vdb_dir), embedding_function=emb)
        count_depois = len(vdb3.get()["ids"])
        assert count_depois == count_antes, (
            f"Esperava {count_antes} chunks após rollback, obteve {count_depois}"
        )
