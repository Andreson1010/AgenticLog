# AgenticLog — Integration tests: incremental ChromaDB ingestion
"""
Testes de integração para adicionar_documento_incrementalmente.
Usa ChromaDB real em disco (tmp_path), embeddings mockados.
Execute: pytest -m integration tests/test_rag_integration.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from langchain_core.documents import Document as LCDocument

from agenticlog.rag import adicionar_documento_incrementalmente, RAGSecurityError


def _mock_emb():
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.1] * 10] * len(texts)
    emb.embed_query.return_value = [0.1] * 10
    return emb


@pytest.mark.integration
class TestIngestionIntegration:
    """Integração com ChromaDB real em disco, embeddings mockados."""

    def teste_1_dois_arquivos_chunks_acumulam(self, tmp_path: Path) -> None:
        """Dois uploads incrementais: chunks totais = soma dos chunks de cada arquivo."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo_a = json.dumps({"produto": "cadeira", "cor": "azul"}).encode()
        conteudo_b = json.dumps({"produto": "mesa", "cor": "verde"}).encode()

        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.rag.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.rag.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.rag.HuggingFaceEmbeddings", return_value=mock_emb_instance),
            patch("agenticlog.agent.invalidar_vector_db"),
        ):
            result_a = adicionar_documento_incrementalmente("cadeira.json", conteudo_a)
            result_b = adicionar_documento_incrementalmente("mesa.json", conteudo_b)

        assert result_a["status"] == "adicionado"
        assert result_b["status"] == "adicionado"
        assert (doc_dir / "cadeira.json").exists()
        assert (doc_dir / "mesa.json").exists()

    def teste_2_pre_existentes_intactos_apos_novo_upload(self, tmp_path: Path) -> None:
        """Chunks de arquivo A permanecem intactos após upload de arquivo B."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo_a = json.dumps({"pedido": "P001", "status": "entregue"}).encode()
        conteudo_b = json.dumps({"pedido": "P002", "status": "pendente"}).encode()

        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.rag.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.rag.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.rag.HuggingFaceEmbeddings", return_value=mock_emb_instance),
            patch("agenticlog.agent.invalidar_vector_db"),
        ):
            result_a = adicionar_documento_incrementalmente("p001.json", conteudo_a)
            assert result_a["status"] == "adicionado"

            result_b = adicionar_documento_incrementalmente("p002.json", conteudo_b)
            assert result_b["status"] == "adicionado"

        assert (doc_dir / "p001.json").exists()
        assert (doc_dir / "p002.json").exists()

    def teste_3_primeira_ingestao_sem_colecao_existente(self, tmp_path: Path) -> None:
        """Sem vectordb existente: cria coleção e ingere sem erro."""
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        vdb_dir = tmp_path / "vectordb"
        # Intencionalmente NÃO criamos vdb_dir — ChromaDB deve criar automaticamente

        conteudo = json.dumps({"rota": "SP-RJ", "km": "450"}).encode()
        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.rag.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.rag.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.rag.HuggingFaceEmbeddings", return_value=mock_emb_instance),
            patch("agenticlog.agent.invalidar_vector_db"),
        ):
            result = adicionar_documento_incrementalmente("rota.json", conteudo)

        assert result["status"] == "adicionado"
        assert (doc_dir / "rota.json").exists()

    def teste_4_rollback_nao_deixa_chunks_orfaos(self, tmp_path: Path) -> None:
        """Falha no add_documents: arquivo removido do disco."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo = json.dumps({"carga": "fragil"}).encode()
        mock_emb_instance = _mock_emb()

        from langchain_chroma import Chroma as RealChroma

        fail_vdb = MagicMock()
        fail_vdb.get.return_value = {"ids": [], "metadatas": []}
        fail_vdb.add_documents.side_effect = RuntimeError("falha simulada")

        with (
            patch("agenticlog.rag.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.rag.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.rag.HuggingFaceEmbeddings", return_value=mock_emb_instance),
            patch("agenticlog.rag.Chroma", return_value=fail_vdb),
            patch("agenticlog.rag.JSONLoader") as mock_loader_cls,
            patch("agenticlog.rag.RecursiveCharacterTextSplitter") as mock_splitter_cls,
        ):
            mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
            mock_splitter_cls.return_value.split_documents.return_value = [
                LCDocument(page_content="chunk", metadata={})
            ]

            with pytest.raises(RuntimeError, match="falha simulada"):
                adicionar_documento_incrementalmente("carga.json", conteudo)

        assert not (doc_dir / "carga.json").exists()
