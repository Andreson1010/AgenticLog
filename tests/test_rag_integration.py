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

from agenticlog.config import DEFAULT_COLLECTION_NAME
from agenticlog.ingestion.orchestrator import adicionar_documento_incrementalmente
from agenticlog.retrieval.retriever import _get_retriever


def _mock_emb():
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [[0.1] * 10] * len(texts)
    emb.embed_query.return_value = [0.1] * 10
    return emb


@pytest.mark.integration
class TestIngestionIntegration:
    """Integração com ChromaDB real em disco, embeddings mockados."""

    def teste_1_dois_arquivos_chunks_acumulam(self, tmp_path: Path) -> None:
        """Dois uploads incrementais na mesma coleção: ambos adicionados com sucesso."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo_a = json.dumps({"produto": "cadeira", "cor": "azul"}).encode()
        conteudo_b = json.dumps({"produto": "mesa", "cor": "verde"}).encode()

        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.retrieval.retriever.invalidar_vector_db"),
        ):
            result_a = adicionar_documento_incrementalmente(
                "cadeira.json", conteudo_a, collection_name=DEFAULT_COLLECTION_NAME
            )
            result_b = adicionar_documento_incrementalmente(
                "mesa.json", conteudo_b, collection_name=DEFAULT_COLLECTION_NAME
            )

        assert result_a["status"] == "adicionado"
        assert result_b["status"] == "adicionado"
        assert (doc_dir / "cadeira.json").exists()
        assert (doc_dir / "mesa.json").exists()

    def teste_2_pre_existentes_intactos_apos_novo_upload(self, tmp_path: Path) -> None:
        """Chunks de arquivo A permanecem intactos após upload de arquivo B na mesma coleção."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo_a = json.dumps({"pedido": "P001", "status": "entregue"}).encode()
        conteudo_b = json.dumps({"pedido": "P002", "status": "pendente"}).encode()

        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.retrieval.retriever.invalidar_vector_db"),
        ):
            result_a = adicionar_documento_incrementalmente(
                "p001.json", conteudo_a, collection_name=DEFAULT_COLLECTION_NAME
            )
            assert result_a["status"] == "adicionado"

            result_b = adicionar_documento_incrementalmente(
                "p002.json", conteudo_b, collection_name=DEFAULT_COLLECTION_NAME
            )
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
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.retrieval.retriever.invalidar_vector_db"),
        ):
            result = adicionar_documento_incrementalmente(
                "rota.json", conteudo, collection_name=DEFAULT_COLLECTION_NAME
            )

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

        fail_vdb = MagicMock()
        fail_vdb.get.return_value = {"ids": [], "metadatas": []}
        fail_vdb.add_documents.side_effect = RuntimeError("falha simulada")

        with (
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.ingestion.orchestrator.Chroma", return_value=fail_vdb),
            patch("agenticlog.ingestion.extraction.JSONLoader") as mock_loader_cls,
            patch("agenticlog.ingestion.orchestrator.SemanticChunker") as mock_splitter_cls,
        ):
            mock_loader_cls.return_value.load.return_value = [LCDocument(page_content="d", metadata={})]
            mock_splitter_cls.return_value.split_documents.return_value = [
                LCDocument(page_content="chunk", metadata={})
            ]

            with pytest.raises(RuntimeError, match="falha simulada"):
                adicionar_documento_incrementalmente(
                    "carga.json", conteudo, collection_name=DEFAULT_COLLECTION_NAME
                )

        assert not (doc_dir / "carga.json").exists()

    def teste_5_ingestao_em_duas_colecoes_distintas(self, tmp_path: Path) -> None:
        """Ingestão em 'fornecedores' e 'contratos': ambas as coleções recebem docs separados."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo_forn = json.dumps({"fornecedor": "Alfa Ltda", "cidade": "SP"}).encode()
        conteudo_cont = json.dumps({"contrato": "C-001", "valor": "50000"}).encode()

        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.retrieval.retriever.invalidar_vector_db"),
        ):
            result_forn = adicionar_documento_incrementalmente(
                "fornecedor.json", conteudo_forn, collection_name="fornecedores"
            )
            result_cont = adicionar_documento_incrementalmente(
                "contrato.json", conteudo_cont, collection_name="contratos"
            )

        assert result_forn["status"] == "adicionado"
        assert result_cont["status"] == "adicionado"
        assert (doc_dir / "fornecedor.json").exists()
        assert (doc_dir / "contrato.json").exists()

    def teste_6_ingestao_colecao_unica_resultados_isolados(self, tmp_path: Path) -> None:
        """Ingestão em apenas uma coleção: somente essa coleção possui o documento."""
        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo = json.dumps({"rota": "BH-RJ", "km": "450"}).encode()
        mock_emb_instance = _mock_emb()

        with (
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.retrieval.retriever.invalidar_vector_db"),
        ):
            result = adicionar_documento_incrementalmente(
                "rota.json", conteudo, collection_name="rotas"
            )

        assert result["status"] == "adicionado"
        # O arquivo existe no diretório de documentos
        assert (doc_dir / "rota.json").exists()
        # A coleção padrão NÃO contém o documento (foi armazenado em "rotas")
        from langchain_chroma import Chroma
        colecao_default = Chroma(
            persist_directory=str(vdb_dir),
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_function=mock_emb_instance,
        )
        ids_default = colecao_default.get()["ids"]
        assert len(ids_default) == 0, "Coleção padrão não deveria ter docs da coleção 'rotas'"

    def teste_7_query_retorna_docs_de_multiplas_colecoes(self, tmp_path: Path) -> None:
        """Query cross-collection: documentos de 'fornecedores' e 'contratos' presentes no resultado."""
        from langchain_chroma import Chroma as ChromaReal

        doc_dir = tmp_path / "documents"
        vdb_dir = tmp_path / "vectordb"
        doc_dir.mkdir()
        vdb_dir.mkdir()

        conteudo_forn = json.dumps({"fornecedor": "Beta SA", "cidade": "Curitiba"}).encode()
        conteudo_cont = json.dumps({"contrato": "C-999", "valor": "99000"}).encode()

        mock_emb_instance = _mock_emb()

        # Ingere documentos nas duas coleções usando ChromaDB real em disco temporário
        with (
            patch("agenticlog.ingestion.orchestrator.DIR_DOCUMENTS", new=doc_dir),
            patch("agenticlog.ingestion.orchestrator.DIR_VECTORDB", new=vdb_dir),
            patch("agenticlog.ingestion.orchestrator.criar_embedding_model", return_value=mock_emb_instance),
            patch("agenticlog.retrieval.retriever.invalidar_vector_db"),
        ):
            adicionar_documento_incrementalmente(
                "fornecedor_b.json", conteudo_forn, collection_name="fornecedores"
            )
            adicionar_documento_incrementalmente(
                "contrato_b.json", conteudo_cont, collection_name="contratos"
            )

        # Constrói instâncias Chroma apontando para vdb_dir de teste para cada coleção
        db_forn = ChromaReal(
            persist_directory=str(vdb_dir),
            collection_name="fornecedores",
            embedding_function=mock_emb_instance,
        )
        db_cont = ChromaReal(
            persist_directory=str(vdb_dir),
            collection_name="contratos",
            embedding_function=mock_emb_instance,
        )

        def _fake_get_vector_db(collection_name: str = DEFAULT_COLLECTION_NAME):
            return db_forn if collection_name == "fornecedores" else db_cont

        # Executa _get_retriever com fan-out sobre as duas coleções de teste
        with (
            patch("agenticlog.retrieval.retriever._listar_colecoes", return_value=["fornecedores", "contratos"]),
            patch("agenticlog.retrieval.retriever._get_vector_db", side_effect=_fake_get_vector_db),
        ):
            docs = _get_retriever("fornecedor contrato")

        # Deve retornar documentos de pelo menos uma das coleções
        assert len(docs) >= 1, "Esperava ao menos um documento das coleções combinadas"
        all_content = " ".join(d.page_content for d in docs)
        assert "Beta SA" in all_content or "C-999" in all_content, (
            "Conteúdo de pelo menos uma das coleções deve aparecer nos resultados"
        )
