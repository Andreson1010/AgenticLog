"""Acceptance tests for the load-env-credentials feature.

Each test maps to exactly one acceptance criterion from the approved spec:
  .specs/features/load-env-credentials/spec.md

Tests verify the feature from the outside (module import contract), never
reaching into implementation internals beyond what config.py exports publicly.
"""

import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

# Worktree root — used to locate .env and .env.example on disk for AC-07/AC-08.
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent.parent

_MODULE = "agenticlog.config"
_LOAD_DOTENV = "dotenv.load_dotenv"


def _drop_module() -> None:
    """Remove config from sys.modules so the next import re-executes module body."""
    sys.modules.pop(_MODULE, None)


class TestLoadEnvCredentials(TestCase):
    """Acceptance tests: load-env-credentials feature."""

    def setUp(self) -> None:
        self._saved = sys.modules.get(_MODULE)

    def tearDown(self) -> None:
        if self._saved is not None:
            sys.modules[_MODULE] = self._saved
        else:
            sys.modules.pop(_MODULE, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _import_config(self, env_overrides: dict, remove_keys: tuple = ()):
        """Import (reload) agenticlog.config under a controlled environment.

        load_dotenv is mocked to a no-op so the on-disk .env cannot inject
        values that would mask what this test explicitly provides.

        Args:
            env_overrides: variables to add/override in os.environ.
            remove_keys:   variables to forcibly delete before import.
        """
        _drop_module()
        with patch(_LOAD_DOTENV, return_value=None):
            with patch.dict("os.environ", env_overrides, clear=False):
                for key in remove_keys:
                    os.environ.pop(key, None)
                import agenticlog.config as cfg
        sys.modules[_MODULE] = cfg
        return cfg

    # ------------------------------------------------------------------
    # AC-01: .env default values are reflected in constants
    # ------------------------------------------------------------------

    def test_ac01_default_env_values_loaded_into_constants(self):
        """
        AC-01: WHEN .env has OPENAI_API_KEY=hermes and
        OPENAI_API_BASE=http://127.0.0.1:1234/v1 AND config.py is imported
        THEN LLM_API_KEY == "hermes" AND LLM_API_BASE == "http://127.0.0.1:1234/v1".
        """
        cfg = self._import_config({
            "OPENAI_API_KEY": "hermes",
            "OPENAI_API_BASE": "http://127.0.0.1:1234/v1",
        })
        self.assertEqual(cfg.LLM_API_KEY, "hermes")
        self.assertEqual(cfg.LLM_API_BASE, "http://127.0.0.1:1234/v1")

    # ------------------------------------------------------------------
    # AC-02: Custom values in environment are reflected in constants
    # ------------------------------------------------------------------

    def test_ac02_custom_api_key_reflected_in_constant(self):
        """
        AC-02: WHEN .env has a custom OPENAI_API_KEY value
        THEN LLM_API_KEY reflects that custom value.
        """
        cfg = self._import_config({
            "OPENAI_API_KEY": "my-custom-api-key",
            "OPENAI_API_BASE": "http://127.0.0.1:1234/v1",
        })
        self.assertEqual(cfg.LLM_API_KEY, "my-custom-api-key")

    def test_ac02_custom_api_base_reflected_in_constant(self):
        """
        AC-02: WHEN .env has a custom OPENAI_API_BASE value
        THEN LLM_API_BASE reflects that custom value.
        """
        cfg = self._import_config({
            "OPENAI_API_KEY": "hermes",
            "OPENAI_API_BASE": "http://10.0.0.99:9999/v1",
        })
        self.assertEqual(cfg.LLM_API_BASE, "http://10.0.0.99:9999/v1")

    # ------------------------------------------------------------------
    # AC-03: Shell env takes precedence over .env (override=False)
    # ------------------------------------------------------------------

    def test_ac03_shell_variable_takes_precedence_over_dotenv(self):
        """
        AC-03: WHEN a variable is set in the shell AND .env also defines it
        THEN the shell value wins (load_dotenv called with override=False by default).

        This test does NOT mock load_dotenv so that real dotenv semantics apply.
        The shell variable is pre-set before config is imported; load_dotenv must
        not overwrite it.
        """
        _drop_module()
        # Pre-set the shell variable; the on-disk .env has OPENAI_API_KEY=hermes.
        # If override=False is respected, the shell value "shell-override-key" must win.
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "shell-override-key",
            "OPENAI_API_BASE": "http://127.0.0.1:1234/v1",
        }, clear=False):
            import agenticlog.config as cfg
        sys.modules[_MODULE] = cfg
        self.assertEqual(cfg.LLM_API_KEY, "shell-override-key")

    # ------------------------------------------------------------------
    # AC-04: Missing OPENAI_API_KEY falls back to default "hermes"
    # ------------------------------------------------------------------

    def test_ac04_missing_api_key_uses_default(self):
        """
        AC-04: WHEN OPENAI_API_KEY is absent from both shell and .env
        THEN LLM_API_KEY defaults to "hermes" (LMStudio ignores the key value).
        """
        cfg = self._import_config(
            {"OPENAI_API_BASE": "http://127.0.0.1:1234/v1"},
            remove_keys=("OPENAI_API_KEY",),
        )
        self.assertEqual(cfg.LLM_API_KEY, "hermes")

    # ------------------------------------------------------------------
    # AC-05: Missing OPENAI_API_BASE falls back to local LMStudio endpoint
    # ------------------------------------------------------------------

    def test_ac05_missing_api_base_uses_default(self):
        """
        AC-05: WHEN OPENAI_API_BASE is absent from both shell and .env
        THEN LLM_API_BASE defaults to "http://127.0.0.1:1234/v1".
        """
        cfg = self._import_config(
            {"OPENAI_API_KEY": "hermes"},
            remove_keys=("OPENAI_API_BASE",),
        )
        self.assertEqual(cfg.LLM_API_BASE, "http://127.0.0.1:1234/v1")

    # ------------------------------------------------------------------
    # AC-06: load_dotenv() is called before os.environ reads for credentials
    # ------------------------------------------------------------------

    def test_ac06_load_dotenv_called_before_environ_reads(self):
        """
        AC-06: WHEN config.py module loads THEN load_dotenv() is invoked
        before any os.environ read for LLM credentials.

        Verified by recording the call order: load_dotenv mock must appear
        in the call log before the os.environ.__getitem__ calls for the
        credential keys.
        """
        _drop_module()
        call_order: list[str] = []

        def fake_load_dotenv(*args, **kwargs):
            call_order.append("load_dotenv")

        real_getitem = os.environ.__class__.__getitem__

        def tracking_getitem(self_env, key):
            if key in ("OPENAI_API_KEY", "OPENAI_API_BASE"):
                call_order.append(f"getitem:{key}")
            return real_getitem(self_env, key)

        with patch(_LOAD_DOTENV, side_effect=fake_load_dotenv):
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "hermes",
                "OPENAI_API_BASE": "http://127.0.0.1:1234/v1",
            }, clear=False):
                with patch.object(os.environ.__class__, "__getitem__", tracking_getitem):
                    import agenticlog.config as cfg
        sys.modules[_MODULE] = cfg

        self.assertIn("load_dotenv", call_order,
                      "load_dotenv was never called during config import")
        load_dotenv_pos = call_order.index("load_dotenv")
        credential_positions = [
            i for i, entry in enumerate(call_order)
            if entry.startswith("getitem:")
        ]
        self.assertTrue(
            len(credential_positions) > 0,
            "os.environ reads for credentials were not tracked",
        )
        self.assertTrue(
            all(load_dotenv_pos < pos for pos in credential_positions),
            f"load_dotenv ({load_dotenv_pos}) must precede all credential reads "
            f"({credential_positions}). Full call order: {call_order}",
        )

    # ------------------------------------------------------------------
    # AC-07: Real .env has OPENAI_API_KEY and OPENAI_API_BASE entries
    # ------------------------------------------------------------------

    def test_ac07_real_dotenv_file_contains_required_credentials(self):
        """
        AC-07: The .env file in the project root must contain
        OPENAI_API_KEY and OPENAI_API_BASE entries.
        """
        dotenv_path = _WORKTREE_ROOT / ".env"
        if not dotenv_path.exists():
            # .env é gitignored. No CI é provisionado via `cp .env.example .env`;
            # em dev local sem .env, pular em vez de falhar (não é regressão de código).
            self.skipTest(f".env ausente em {dotenv_path} — provisionado no CI, skip local")

        content = dotenv_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        key_lines = [ln for ln in lines if ln.startswith("OPENAI_API_KEY=")]
        base_lines = [ln for ln in lines if ln.startswith("OPENAI_API_BASE=")]

        self.assertTrue(
            len(key_lines) >= 1,
            "OPENAI_API_KEY= entry not found in .env",
        )
        self.assertTrue(
            len(base_lines) >= 1,
            "OPENAI_API_BASE= entry not found in .env",
        )
        # Values must be non-empty
        self.assertTrue(
            key_lines[0].split("=", 1)[1].strip() != "",
            "OPENAI_API_KEY has an empty value in .env",
        )
        self.assertTrue(
            base_lines[0].split("=", 1)[1].strip() != "",
            "OPENAI_API_BASE has an empty value in .env",
        )

    # ------------------------------------------------------------------
    # AC-08: .env.example is unchanged (already correct)
    # ------------------------------------------------------------------

    def test_ac08_dotenv_example_unchanged_and_contains_credential_entries(self):
        """
        AC-08: .env.example must still contain OPENAI_API_KEY and
        OPENAI_API_BASE entries and must not have been altered by this feature.
        """
        example_path = _WORKTREE_ROOT / ".env.example"
        self.assertTrue(
            example_path.exists(),
            f".env.example not found at {example_path}",
        )

        content = example_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        key_lines = [ln for ln in lines if ln.startswith("OPENAI_API_KEY=")]
        base_lines = [ln for ln in lines if ln.startswith("OPENAI_API_BASE=")]

        self.assertTrue(
            len(key_lines) >= 1,
            "OPENAI_API_KEY= entry missing from .env.example",
        )
        self.assertTrue(
            len(base_lines) >= 1,
            "OPENAI_API_BASE= entry missing from .env.example",
        )
        # Spec says the values in .env.example are the LMStudio defaults.
        self.assertEqual(
            key_lines[0], "OPENAI_API_KEY=hermes",
            ".env.example OPENAI_API_KEY value changed from expected 'hermes'",
        )
        self.assertEqual(
            base_lines[0], "OPENAI_API_BASE=http://127.0.0.1:1234/v1",
            ".env.example OPENAI_API_BASE value changed from expected default",
        )
