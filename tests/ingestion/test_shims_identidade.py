# AgenticLog - Acicidade de ingestion
"""Acicidade de imports (ADR-018 Fase 3a).

TestShimsIdentidade removido na Fase 6 (shims deletados).
TestIngestionAcyclic mantido.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))


class TestIngestionAcyclic(unittest.TestCase):
    """Acicidade de imports do pacote ingestion (RAGING-10)."""

    def teste_1_import_ingestion_interpretador_frio_sai_zero(self) -> None:
        """`import agenticlog.ingestion` em subprocess frio sai 0 e sem 'circular'."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", "import agenticlog.ingestion"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Import frio falhou (exit {result.returncode}).\nstderr: {result.stderr!r}",
        )
        self.assertNotIn("circular", result.stderr.lower(), result.stderr)
        self.assertNotIn("partially initialized", result.stderr.lower(), result.stderr)
