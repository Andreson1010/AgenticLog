# AgenticLog - Testes unitários para agent.py
"""
Testes para o módulo agenticlog.agent.
Cobre: invalidar_vector_db, _get_vector_db pós-invalidação.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import unittest

import agenticlog.agent as agent_mod


class TestInvalidarVectorDb(unittest.TestCase):
    """Testes para invalidar_vector_db() — INCRM-02."""

    def teste_1_invalidar_seta_none(self):
        """invalidar_vector_db() deve atribuir None a _vector_db."""
        agent_mod._vector_db = MagicMock()  # simula singleton inicializado
        agent_mod.invalidar_vector_db()
        self.assertIsNone(agent_mod._vector_db)

    def teste_2_invalidar_quando_ja_none_nao_levanta(self):
        """invalidar_vector_db() com _vector_db já None não deve lançar exceção."""
        agent_mod._vector_db = None
        agent_mod.invalidar_vector_db()  # não deve levantar
        self.assertIsNone(agent_mod._vector_db)

    def teste_3_get_vector_db_apos_invalidacao_cria_nova_instancia(self):
        """_get_vector_db() após invalidação deve criar uma nova instância Chroma."""
        agent_mod._vector_db = None
        mock_chroma = MagicMock()
        with patch("agenticlog.agent.Chroma", return_value=mock_chroma) as MockChroma:
            with patch("agenticlog.agent._get_embedding_model", return_value=MagicMock()):
                resultado = agent_mod._get_vector_db()
        MockChroma.assert_called_once()
        self.assertIs(resultado, mock_chroma)


if __name__ == "__main__":
    unittest.main(verbosity=2)
