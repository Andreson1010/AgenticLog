# AgenticLog - Testes unitários para ingestion/store.py (ADR-018 Fase 3b)
"""Cobre a primitiva de rollback de escrita no Chroma e as primitivas de coleção
parametrizadas por ``vectordb_dir`` (ING3B-01)."""

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.ingestion.store as store  # noqa: E402


class TestAddDocumentsComRollback(unittest.TestCase):
    """add_documents_com_rollback: escrita atômica com rollback best-effort."""

    def teste_1_sucesso_chama_add_documents(self) -> None:
        """Caminho feliz: chama add_documents(chunks, ids=chunk_ids), sem delete."""
        vdb = MagicMock()
        chunks = ["c1", "c2"]
        chunk_ids = ["id1", "id2"]

        store.add_documents_com_rollback(vdb, chunks, chunk_ids)

        vdb.add_documents.assert_called_once_with(chunks, ids=chunk_ids)
        vdb.delete.assert_not_called()

    def teste_2_falha_no_add_dispara_delete_e_relevanta_original(self) -> None:
        """Falha no add: delete(ids=chunk_ids) é chamado e a exceção ORIGINAL re-levantada."""
        vdb = MagicMock()
        vdb.add_documents.side_effect = RuntimeError("embed fail")
        chunk_ids = ["id1"]

        with self.assertRaises(RuntimeError) as ctx:
            store.add_documents_com_rollback(vdb, ["c1"], chunk_ids)

        self.assertEqual(str(ctx.exception), "embed fail")
        vdb.delete.assert_called_once_with(ids=chunk_ids)

    def teste_3_rollback_falho_loga_orfaos_e_relevanta_original(self) -> None:
        """Delete também falha: loga IDs órfãos (WARNING) e ainda re-levanta a original."""
        vdb = MagicMock()
        vdb.add_documents.side_effect = RuntimeError("embed fail")
        vdb.delete.side_effect = RuntimeError("rollback fail")
        chunk_ids = ["id_orfao"]

        with self.assertLogs("agenticlog.rag", level="WARNING") as log_ctx:
            with self.assertRaises(RuntimeError) as ctx:
                store.add_documents_com_rollback(vdb, ["c1"], chunk_ids)

        self.assertEqual(str(ctx.exception), "embed fail")
        self.assertTrue(any("IDs órfãos" in m for m in log_ctx.output))


class TestOutrasColecoesExistem(unittest.TestCase):
    """_outras_colecoes_existem: seam ``vectordb_dir`` (explícito e via store.DIR_VECTORDB)."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._db = Path(self._tmp) / "chroma.sqlite3"

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _criar_db(self, nomes: list[str]) -> None:
        con = sqlite3.connect(self._db)
        try:
            con.execute("CREATE TABLE collections (name TEXT)")
            con.executemany("INSERT INTO collections (name) VALUES (?)", [(n,) for n in nomes])
            con.commit()
        finally:
            con.close()

    def teste_1_vectordb_dir_explicito_detecta_irma(self) -> None:
        """Com vectordb_dir explícito: coleção irmã presente → True."""
        self._criar_db(["logistica", "outra"])
        self.assertTrue(
            store._outras_colecoes_existem("logistica", vectordb_dir=Path(self._tmp))
        )

    def teste_2_vectordb_dir_explicito_apenas_alvo(self) -> None:
        """Com vectordb_dir explícito: só a coleção alvo → False."""
        self._criar_db(["logistica"])
        self.assertFalse(
            store._outras_colecoes_existem("logistica", vectordb_dir=Path(self._tmp))
        )

    def teste_3_fallback_para_store_dir_vectordb_quando_none(self) -> None:
        """vectordb_dir=None resolve para store.DIR_VECTORDB (patchado no corpo)."""
        self._criar_db(["logistica", "irma"])
        with patch("agenticlog.ingestion.store.DIR_VECTORDB", Path(self._tmp)):
            self.assertTrue(store._outras_colecoes_existem("logistica"))

    def teste_4_db_inexistente_retorna_false(self) -> None:
        """Sem arquivo SQLite: nenhuma irmã → False (wipe seguro)."""
        self.assertFalse(
            store._outras_colecoes_existem("logistica", vectordb_dir=Path(self._tmp))
        )


class TestResetarColecao(unittest.TestCase):
    """_resetar_colecao: propaga o seam ``vectordb_dir`` a _outras_colecoes_existem."""

    @patch("agenticlog.ingestion.store._outras_colecoes_existem", return_value=False)
    @patch("agenticlog.ingestion.store.shutil.rmtree")
    def teste_1_colecao_unica_remove_vectordb_dir(self, mock_rmtree, mock_outras) -> None:
        """Coleção única + diretório presente: rmtree do vectordb_dir explícito."""
        vdir = MagicMock()
        vdir.exists.return_value = True

        store._resetar_colecao("logistica", vectordb_dir=vdir)

        mock_rmtree.assert_called_once_with(vdir, ignore_errors=True)
        # o seam é propagado para _outras_colecoes_existem
        mock_outras.assert_called_once_with("logistica", vectordb_dir=vdir)

    @patch("chromadb.PersistentClient")
    @patch("agenticlog.ingestion.store._outras_colecoes_existem", return_value=True)
    @patch("agenticlog.ingestion.store.shutil.rmtree")
    def teste_2_multi_colecao_preserva_irmas(
        self, mock_rmtree, mock_outras, mock_client_cls
    ) -> None:
        """Multi-coleção: descarta só a coleção alvo (não remove o diretório)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        store._resetar_colecao("logistica", vectordb_dir=Path("/tmp/vdb"))

        mock_rmtree.assert_not_called()
        mock_client.delete_collection.assert_called_once_with("logistica")


if __name__ == "__main__":
    unittest.main(verbosity=2)
