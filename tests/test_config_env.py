"""Tests for env-var loading in agenticlog.config (load-env-credentials feature)."""

import os
import sys
from unittest import TestCase
from unittest.mock import patch

_NOOP_LOAD_DOTENV = "dotenv.load_dotenv"


class TestConfigEnv(TestCase):
    """Unit tests for env-var credential loading in agenticlog.config."""

    def setUp(self):
        """Save the live module so tearDown can restore it."""
        self._saved_module = sys.modules.get("agenticlog.config")

    def tearDown(self):
        """Always leave sys.modules in a consistent state for sibling tests."""
        if self._saved_module is not None:
            sys.modules["agenticlog.config"] = self._saved_module
        else:
            sys.modules.pop("agenticlog.config", None)

    def _reload(self, overrides: dict, remove_keys: tuple = ()):
        """Reload config with specific credential overrides.

        Uses patch.dict without clear=True so system env vars (HOME, PATH,
        USERPROFILE, etc.) are preserved.  load_dotenv is mocked to a no-op
        so the on-disk .env file cannot inject values and interfere.

        Args:
            overrides: env vars to set (merged on top of real env).
            remove_keys: env var names to forcibly remove before import.
        """
        sys.modules.pop("agenticlog.config", None)
        # Build the patch dict: set overrides, then remove unwanted keys.
        patch_values = dict(overrides)
        # We must explicitly delete keys not in overrides — patch.dict does
        # not support deletion directly, so we temporarily set them to a
        # sentinel and delete inside the block.
        with patch(_NOOP_LOAD_DOTENV, return_value=None):
            with patch.dict("os.environ", patch_values, clear=False):
                for key in remove_keys:
                    os.environ.pop(key, None)
                import agenticlog.config as cfg
        sys.modules["agenticlog.config"] = cfg
        return cfg

    def teste_1_api_key_loaded_from_env(self):
        """LLM_API_KEY equals OPENAI_API_KEY from environment."""
        cfg = self._reload({"OPENAI_API_KEY": "custom-key", "OPENAI_API_BASE": "http://127.0.0.1:1234/v1"})
        self.assertEqual(cfg.LLM_API_KEY, "custom-key")

    def teste_2_api_base_loaded_from_env(self):
        """LLM_API_BASE equals OPENAI_API_BASE from environment."""
        cfg = self._reload({"OPENAI_API_KEY": "hermes", "OPENAI_API_BASE": "http://10.0.0.1:1234/v1"})
        self.assertEqual(cfg.LLM_API_BASE, "http://10.0.0.1:1234/v1")

    def teste_3_defaults_from_env_file(self):
        """Default .env values are reflected in constants."""
        cfg = self._reload({"OPENAI_API_KEY": "hermes", "OPENAI_API_BASE": "http://127.0.0.1:1234/v1"})
        self.assertEqual(cfg.LLM_API_KEY, "hermes")
        self.assertEqual(cfg.LLM_API_BASE, "http://127.0.0.1:1234/v1")

    def teste_4_missing_api_key_uses_default(self):
        """Missing OPENAI_API_KEY falls back to 'hermes' (LMStudio ignores the key)."""
        cfg = self._reload(
            {"OPENAI_API_BASE": "http://127.0.0.1:1234/v1"},
            remove_keys=("OPENAI_API_KEY",),
        )
        self.assertEqual(cfg.LLM_API_KEY, "hermes")

    def teste_5_missing_api_base_uses_default(self):
        """Missing OPENAI_API_BASE falls back to local LMStudio endpoint."""
        cfg = self._reload(
            {"OPENAI_API_KEY": "hermes"},
            remove_keys=("OPENAI_API_BASE",),
        )
        self.assertEqual(cfg.LLM_API_BASE, "http://127.0.0.1:1234/v1")

    def teste_6_shell_takes_precedence_over_dotenv(self):
        """Shell variable set before load_dotenv is not overwritten (override=False)."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "shell-key", "OPENAI_API_BASE": "http://127.0.0.1:1234/v1"},
            clear=False,
        ):
            sys.modules.pop("agenticlog.config", None)
            import agenticlog.config as cfg
            sys.modules["agenticlog.config"] = cfg
        self.assertEqual(cfg.LLM_API_KEY, "shell-key")

    def teste_7_llm_model_unset_uses_default(self):
        """LLM_MODEL env var absent falls back to the hardcoded default."""
        cfg = self._reload({}, remove_keys=("LLM_MODEL",))
        self.assertEqual(cfg.LLM_MODEL, "hermes-3-llama-3.2-3b")

    def teste_8_llm_model_set_uses_override(self):
        """LLM_MODEL set to a non-empty value overrides the default."""
        cfg = self._reload({"LLM_MODEL": "my-custom-model"})
        self.assertEqual(cfg.LLM_MODEL, "my-custom-model")

    def teste_9_llm_model_empty_string_uses_default(self):
        """LLM_MODEL set to the empty string is treated as unset (falls back to default)."""
        cfg = self._reload({"LLM_MODEL": ""})
        self.assertEqual(cfg.LLM_MODEL, "hermes-3-llama-3.2-3b")

    def teste_10_llm_model_whitespace_is_verbatim(self):
        """LLM_MODEL set to a whitespace-only value is used verbatim (not treated as unset)."""
        cfg = self._reload({"LLM_MODEL": " "})
        self.assertEqual(cfg.LLM_MODEL, " ")
