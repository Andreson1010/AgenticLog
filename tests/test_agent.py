# AgenticLog - Testes unitários para agent.py
"""Testes para funções públicas de agenticlog.agent."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

import agenticlog.agent as agent_mod
from agenticlog.agent import invalidar_vector_db


class TestInvalidarVectorDb(unittest.TestCase):
    """Testes para invalidar_vector_db()."""

    def teste_1_invalidar_seta_none(self) -> None:
        """invalidar_vector_db() deve atribuir None a _vector_db."""
        agent_mod._vector_db = MagicMock()
        invalidar_vector_db()
        self.assertIsNone(agent_mod._vector_db)

    def teste_2_invalidar_quando_ja_none_nao_levanta(self) -> None:
        """invalidar_vector_db() com _vector_db já None não deve lançar exceção."""
        agent_mod._vector_db = None
        invalidar_vector_db()
        self.assertIsNone(agent_mod._vector_db)


if __name__ == "__main__":
    unittest.main()
