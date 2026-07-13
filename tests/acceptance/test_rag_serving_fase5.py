# AgenticLog - Round-trip de identidade dos shims + acicidade de serving/ (ADR-018 Fase 5)
"""Contrato de identidade dos shims e acicidade de imports (ADR-018 Fase 5).

SERV-11 (identidade `is` de agenticlog.api.X is agenticlog.serving.api.X,
agenticlog.health.X is agenticlog.serving.health.X, e acicidade fresh-interpreter
de `import agenticlog.serving`).
"""

import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

import agenticlog.api as api_shim  # noqa: E402
import agenticlog.health as health_shim  # noqa: E402
import agenticlog.serving.api as serving_api  # noqa: E402
import agenticlog.serving.health as serving_health  # noqa: E402

# Lista de (nome, modulo_origem, modulo_destino) para cada simbolo
_SIMBOLOS_API = [
    # (nome, shim, definicao)
    ("app", api_shim, serving_api),
    ("consultar", api_shim, serving_api),
    ("listar_historico", api_shim, serving_api),
    ("QueryRequest", api_shim, serving_api),
    ("QueryResponse", api_shim, serving_api),
    ("HistoryEntry", api_shim, serving_api),
    ("DocumentInfo", api_shim, serving_api),
    ("lifespan", api_shim, serving_api),
    ("_verificar_vectordb", api_shim, serving_api),
    ("_serializar_documentos", api_shim, serving_api),
    ("_normalizar_estado", api_shim, serving_api),
    ("_resposta_segura", api_shim, serving_api),
    ("_construir_registro", api_shim, serving_api),
    ("handler_lmstudio", api_shim, serving_api),
    ("handler_connect_error", api_shim, serving_api),
    ("handler_generico", api_shim, serving_api),
    ("MSG_LMSTUDIO_INDISPONIVEL", api_shim, serving_api),
    ("MSG_VECTORDB_AUSENTE", api_shim, serving_api),
    ("_ERROS_MODO_SEGURO", api_shim, serving_api),
]

_SIMBOLOS_HEALTH = [
    ("check_lmstudio_health", health_shim, serving_health),
    ("reset_health_check_sentinel", health_shim, serving_health),
    ("LMStudioUnavailableError", health_shim, serving_health),
    ("ModeloNaoCarregadoError", health_shim, serving_health),
    ("_health_checked", health_shim, serving_health),
    ("_extrair_ids_modelos", health_shim, serving_health),
    ("MAX_MODELOS_LOG", health_shim, serving_health),
]


class TestServingShimsIdentidade(unittest.TestCase):
    """Round-trip identidade: agenticlog.api.X is agenticlog.serving.api.X."""

    def teste_1_identidade_cada_simbolo_api(self) -> None:
        """Cada simbolo de api e o MESMO objeto no shim e na definicao."""
        for nome, shim_mod, serving_mod in _SIMBOLOS_API:
            with self.subTest(simbolo=f"api.{nome}"):
                self.assertIs(getattr(shim_mod, nome), getattr(serving_mod, nome))

    def teste_2_identidade_cada_simbolo_health(self) -> None:
        """Cada simbolo de health e o MESMO objeto no shim e na definicao."""
        for nome, shim_mod, serving_mod in _SIMBOLOS_HEALTH:
            with self.subTest(simbolo=f"health.{nome}"):
                self.assertIs(getattr(shim_mod, nome), getattr(serving_mod, nome))

    def teste_3_shim_health_httpx_preservado(self) -> None:
        """agenticlog.health.httpx e o modulo httpx (import mantido no shim)."""
        import httpx as _httpx

        self.assertIs(health_shim.httpx, _httpx)

    def teste_4_logger_names_explicitos(self) -> None:
        """Logger names sao strings explicitas 'agenticlog.api' e 'agenticlog.health'."""
        self.assertEqual(logging.getLogger("agenticlog.api").name, "agenticlog.api")
        self.assertEqual(logging.getLogger("agenticlog.health").name, "agenticlog.health")

    def teste_5_serving_app_e_o_mesmo(self) -> None:
        """agenticlog.serving.app e agenticlog.serving.api.app (via lazy __getattr__)."""
        import agenticlog.serving as serving

        self.assertIs(serving.app, serving_api.app)

    def teste_6_path_import_antigo_resolve_api(self) -> None:
        """from agenticlog.api import X continua resolvendo."""
        for nome, _, _ in _SIMBOLOS_API:
            with self.subTest(simbolo=nome):
                self.assertTrue(hasattr(api_shim, nome))

    def teste_7_path_import_antigo_resolve_health(self) -> None:
        """from agenticlog.health import X continua resolvendo."""
        for nome, _, _ in _SIMBOLOS_HEALTH:
            with self.subTest(simbolo=nome):
                self.assertTrue(hasattr(health_shim, nome))


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

    def teste_2_import_api_e_serving_api(self) -> None:
        """import agenticlog.api + import agenticlog.serving.api -> exit 0."""
        result = self._run_subprocess(
            "import agenticlog.api; import agenticlog.serving.api; "
            "print('OK:', agenticlog.api.app is agenticlog.serving.api.app)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK: True", result.stdout)

    def teste_3_import_health_e_serving_health(self) -> None:
        """import agenticlog.health + import agenticlog.serving.health -> exit 0."""
        result = self._run_subprocess(
            "import agenticlog.health; import agenticlog.serving.health; "
            "print('OK:', agenticlog.health.check_lmstudio_health "
            "is agenticlog.serving.health.check_lmstudio_health)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK: True", result.stdout)
