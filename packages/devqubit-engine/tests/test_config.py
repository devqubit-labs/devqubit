# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""Tests for devqubit_engine.config — E2E config loading, redaction, parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from devqubit_engine.config import (
    Config,
    RedactionConfig,
    _parse_bool,
    _parse_patterns,
    get_config,
    load_config,
    reset_config,
    set_config,
)


class TestRedactionConfig:
    """Tests for credential redaction."""

    def test_default_patterns_match_cloud_tokens(self):
        """Default patterns catch common cloud provider secrets."""
        rc = RedactionConfig()

        assert rc.should_redact("AWS_SECRET_ACCESS_KEY")
        assert rc.should_redact("AZURE_CLIENT_SECRET")
        assert rc.should_redact("GCP_SERVICE_ACCOUNT_KEY")
        assert rc.should_redact("GOOGLE_APPLICATION_CREDENTIALS")
        assert rc.should_redact("IBM_TOKEN")
        assert rc.should_redact("IONQ_API_KEY")
        assert rc.should_redact("BRAKET_SESSION_TOKEN")
        assert rc.should_redact("MY_SECRET_VALUE")
        assert rc.should_redact("DATABASE_PASSWORD")
        assert rc.should_redact("PRIVATE_KEY_PATH")

    def test_safe_variables_not_redacted(self):
        """Non-sensitive variables pass through unchanged."""
        rc = RedactionConfig()

        assert not rc.should_redact("PATH")
        assert not rc.should_redact("HOME")
        assert not rc.should_redact("PYTHONPATH")
        assert not rc.should_redact("DEVQUBIT_HOME")
        assert not rc.should_redact("SHOTS")

    def test_case_insensitive_matching(self):
        """Patterns match regardless of case."""
        rc = RedactionConfig()

        assert rc.should_redact("my_secret")
        assert rc.should_redact("My_Secret")
        assert rc.should_redact("MY_SECRET")
        assert rc.should_redact("api_key")
        assert rc.should_redact("API_KEY")

    def test_redact_env_replaces_secrets(self):
        """redact_env produces a new dict with secrets masked."""
        rc = RedactionConfig()
        env = {
            "PATH": "/usr/bin",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI",
            "MY_TOKEN": "tok_12345",
            "DEVQUBIT_HOME": "/home/user/.devqubit",
        }

        redacted = rc.redact_env(env)

        assert redacted["PATH"] == "/usr/bin"
        assert redacted["DEVQUBIT_HOME"] == "/home/user/.devqubit"
        assert redacted["AWS_SECRET_ACCESS_KEY"] == "[REDACTED]"
        assert redacted["MY_TOKEN"] == "[REDACTED]"

    def test_redact_env_does_not_mutate_input(self):
        """redact_env returns a new dict, input is untouched."""
        rc = RedactionConfig()
        original = {"SECRET_KEY": "hunter2"}

        redacted = rc.redact_env(original)

        assert original["SECRET_KEY"] == "hunter2"
        assert redacted["SECRET_KEY"] == "[REDACTED]"

    def test_disabled_redaction_passes_everything(self):
        """When enabled=False, nothing is redacted."""
        rc = RedactionConfig(enabled=False)

        assert not rc.should_redact("AWS_SECRET_ACCESS_KEY")
        env = {"SECRET_KEY": "hunter2"}
        assert rc.redact_env(env)["SECRET_KEY"] == "hunter2"

    def test_custom_replacement_string(self):
        """Custom replacement string is used."""
        rc = RedactionConfig(replacement="***")
        redacted = rc.redact_env({"MY_TOKEN": "secret"})
        assert redacted["MY_TOKEN"] == "***"

    def test_custom_patterns(self):
        """Custom patterns extend detection."""
        rc = RedactionConfig(patterns=[r"^CUSTOM_"])

        assert rc.should_redact("CUSTOM_VALUE")
        assert not rc.should_redact("AWS_SECRET_ACCESS_KEY")  # default patterns gone

    def test_apikey_pattern_matches_with_and_without_underscore(self):
        """The API_?KEY pattern matches both APIKEY and API_KEY."""
        rc = RedactionConfig()

        assert rc.should_redact("MY_APIKEY")
        assert rc.should_redact("MY_API_KEY")


class TestParseBool:
    """Edge cases for boolean parsing from env vars."""

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes", " true "])
    def test_truthy_values(self, val):
        assert _parse_bool(val) is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "random", "FALSE"])
    def test_falsy_values(self, val):
        assert _parse_bool(val) is False

    def test_none_returns_default_true(self):
        assert _parse_bool(None) is True
        assert _parse_bool(None, default=True) is True

    def test_none_returns_default_false(self):
        assert _parse_bool(None, default=False) is False

    def test_empty_string_returns_default(self):
        assert _parse_bool("") is True
        assert _parse_bool("", default=False) is False


class TestParsePatterns:
    def test_comma_separated(self):
        assert _parse_patterns("FOO,BAR,BAZ") == ["FOO", "BAR", "BAZ"]

    def test_strips_whitespace(self):
        assert _parse_patterns(" FOO , BAR ") == ["FOO", "BAR"]

    def test_none_returns_none(self):
        assert _parse_patterns(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_patterns("") is None

    def test_all_empty_items_returns_none(self):
        assert _parse_patterns(",,,") is None


class TestLoadConfig:
    """E2E: env vars => load_config => Config with correct values."""

    def test_defaults_with_clean_env(self, clean_env):
        """No env vars => sensible defaults."""
        cfg = load_config()

        assert cfg.root_dir == Path.home() / ".devqubit"
        assert cfg.capture_pip is True
        assert cfg.capture_git is True
        assert cfg.validate is True
        assert cfg.redaction.enabled is True

    def test_custom_home(self, clean_env, tmp_path):
        """DEVQUBIT_HOME overrides root_dir."""
        custom = tmp_path / "custom"
        clean_env.setenv("DEVQUBIT_HOME", str(custom))

        cfg = load_config()

        assert cfg.root_dir == custom.resolve()

    def test_disable_features(self, clean_env):
        """Boolean env vars disable features."""
        clean_env.setenv("DEVQUBIT_CAPTURE_PIP", "false")
        clean_env.setenv("DEVQUBIT_CAPTURE_GIT", "0")
        clean_env.setenv("DEVQUBIT_VALIDATE", "no")

        cfg = load_config()

        assert cfg.capture_pip is False
        assert cfg.capture_git is False
        assert cfg.validate is False

    def test_disable_redaction(self, clean_env):
        """DEVQUBIT_REDACT_DISABLE=1 turns off redaction."""
        clean_env.setenv("DEVQUBIT_REDACT_DISABLE", "1")

        cfg = load_config()

        assert cfg.redaction.enabled is False

    def test_extra_redaction_patterns(self, clean_env):
        """DEVQUBIT_REDACT_PATTERNS extends default patterns."""
        clean_env.setenv("DEVQUBIT_REDACT_PATTERNS", "^CUSTOM_,^MY_APP_")

        cfg = load_config()

        assert cfg.redaction.should_redact("CUSTOM_SECRET")
        assert cfg.redaction.should_redact("MY_APP_TOKEN")
        # Default patterns still work
        assert cfg.redaction.should_redact("AWS_SECRET_ACCESS_KEY")

    def test_custom_storage_url(self, clean_env):
        """DEVQUBIT_STORAGE_URL is respected."""
        clean_env.setenv("DEVQUBIT_STORAGE_URL", "s3://bucket/objects")

        cfg = load_config()

        assert cfg.storage_url == "s3://bucket/objects"


class TestConfigSingleton:
    """get_config caches, set_config overrides, reset_config clears."""

    def test_get_config_caches(self, clean_env):
        """Two calls to get_config return the same object."""
        a = get_config()
        b = get_config()
        assert a is b

    def test_set_config_overrides(self, tmp_path):
        """set_config replaces the cached singleton."""
        custom = Config(root_dir=tmp_path)
        set_config(custom)

        assert get_config() is custom

    def test_reset_then_get_reloads(self, clean_env, tmp_path):
        """After reset, get_config creates a fresh instance."""
        first = get_config()
        reset_config()
        second = get_config()

        assert first is not second
