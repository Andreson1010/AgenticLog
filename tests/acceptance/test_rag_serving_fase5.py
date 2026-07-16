# AgenticLog - Acicidade de serving/ (ADR-018 Fase 5 + Fase 6)
"""Acicidade de imports (ADR-018 Fase 5).

TestServingShimsIdentidade removido na Fase 6 (shims api.py e health.py deletados).
TestServingAcyclic mantido — verifica acicidade fresh-interpreter.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))


class TestServingAcyclic(unittest.TestCase):
    """Acicidade de imports do pacote serving/ (SERV-09, SERV-11)."""

    def _run_subprocess(self, code: str) -> subprocess.CompletedProcess:
        """Executa codigo em subprocesso com PYTHONPATH ajustado."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )

    def teste_1_import_serving_interpretador_frio_sai_zero(self) -> None:
        """import agenticlog.serving em subprocess frio sai 0 e sem circular."""
        result = self._run_subprocess("import agenticlog.serving")
        self.assertEqual(
            result.returncode,
            0,
            f"Import frio falhou (exit {result.returncode}).\nstderr: {result.stderr!r}",
        )
        self.assertNotIn("circular", result.stderr.lower(), result.stderr)
        self.assertNotIn("partially initialized", result.stderr.lower(), result.stderr)

    def teste_2_import_serving_api_sozinho(self) -> None:
        """import agenticlog.serving.api -> exit 0."""
        result = self._run_subprocess("import agenticlog.serving.api; print('OK')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def teste_3_import_serving_health_sozinho(self) -> None:
        """import agenticlog.serving.health -> exit 0."""
        result = self._run_subprocess("import agenticlog.serving.health; print('OK')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)
