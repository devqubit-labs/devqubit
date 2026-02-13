# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Configuration management for devqubit engine.

This module provides centralized configuration with defaults,
environment variable overrides, and sensitive data redaction
capabilities.

Examples
--------
>>> from devqubit_engine.config import get_config, load_config
>>> config = get_config()
>>> print(config.root_dir)
/home/user/.devqubit

>>> # Override via environment
>>> import os
>>> os.environ["DEVQUBIT_HOME"] = "/custom/path"
>>> from devqubit_engine.config import reset_config
>>> reset_config()
>>> config = get_config()
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


logger = logging.getLogger(__name__)


# Default patterns for sensitive environment variables to redact.
# Patterns are matched as substrings (using re.search) with IGNORECASE.
DEFAULT_REDACT_PATTERNS: tuple[str, ...] = (
    r"TOKEN",
    r"SECRET",
    r"PASSWORD",
    r"API_?KEY",
    r"CREDENTIAL",
    r"PRIVATE",
    r"^AWS_",
    r"^AZURE_",
    r"^GCP_",
    r"^GOOGLE_",
    r"^IBM_",
    r"^IONQ_",
    r"^BRAKET_",
)

# Module-level compiled patterns cache for default patterns
_compiled_default_patterns: tuple[re.Pattern[str], ...] | None = None


def _get_compiled_patterns(
    patterns: tuple[str, ...] | list[str],
) -> list[re.Pattern[str]]:
    """
    Compile regex patterns with caching for default patterns.

    Parameters
    ----------
    patterns : tuple or list of str
        Regex patterns to compile.

    Returns
    -------
    list of Pattern
        Compiled regex patterns with IGNORECASE flag.
    """
    global _compiled_default_patterns

    patterns_tuple = tuple(patterns)
    if patterns_tuple == DEFAULT_REDACT_PATTERNS:
        if _compiled_default_patterns is None:
            _compiled_default_patterns = tuple(
                re.compile(p, re.IGNORECASE) for p in DEFAULT_REDACT_PATTERNS
            )
            logger.debug("Compiled %d default redaction patterns", len(patterns_tuple))
        return list(_compiled_default_patterns)

    return [re.compile(p, re.IGNORECASE) for p in patterns]


@dataclass
class RedactionConfig:
    """
    Configuration for redacting sensitive information from environment.

    Patterns are evaluated using regex search (substring match) with
    case-insensitive matching. Use ``^`` for start-of-string anchoring.

    Parameters
    ----------
    enabled : bool, optional
        Whether redaction is enabled. Default is True.
    patterns : list of str, optional
        Regex patterns for matching sensitive variable names.
        Default includes common cloud provider and secret patterns.
    replacement : str, optional
        Replacement string for redacted values. Default is "[REDACTED]".
    """

    enabled: bool = True
    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_REDACT_PATTERNS))
    replacement: str = "[REDACTED]"

    # Compiled patterns, populated once in __post_init__
    _compiled: list[re.Pattern[str]] = field(
        default_factory=list,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Compile regex patterns eagerly at construction time."""
        self._compiled = _get_compiled_patterns(self.patterns)

    def should_redact(self, key: str) -> bool:
        """
        Check if an environment variable name should be redacted.

        Parameters
        ----------
        key : str
            Environment variable name to check.

        Returns
        -------
        bool
            True if the key matches any redaction pattern.
        """
        if not self.enabled:
            return False

        for pattern in self._compiled:
            if pattern.search(key):
                return True
        return False

    def redact_env(self, env: dict[str, str]) -> dict[str, str]:
        """
        Redact sensitive values from an environment dictionary.

        Parameters
        ----------
        env : dict
            Environment variable dictionary to redact.

        Returns
        -------
        dict
            New dictionary with sensitive values replaced.
        """
        if not self.enabled:
            return dict(env)

        result: dict[str, str] = {}
        redacted_count = 0

        for key, value in env.items():
            if self.should_redact(key):
                result[key] = self.replacement
                redacted_count += 1
            else:
                result[key] = value

        if redacted_count > 0:
            logger.debug("Redacted %d environment variables", redacted_count)

        return result


def _default_root_dir() -> Path:
    """
    Get the default root directory for devqubit data.

    Returns
    -------
    Path
        Default path at ``~/.devqubit``.
    """
    return Path.home() / ".devqubit"


@dataclass
class Config:
    """
    devqubit configuration container.

    Configuration is loaded from environment variables with sensible defaults.
    All paths are automatically expanded (``~`` to home directory).

    Parameters
    ----------
    root_dir : Path, optional
        Root directory for devqubit data. Default is ``~/.devqubit``.
    storage_url : str, optional
        Object store URL. Default is ``file://{root_dir}/objects``.
    registry_url : str, optional
        Registry URL. Default is ``file://{root_dir}``.
    capture_pip : bool, optional
        Whether to capture pip freeze in environment snapshots. Default is True.
    capture_git : bool, optional
        Whether to capture git provenance. Default is True.
    validate : bool, optional
        Whether to validate records against schema. Default is True.
    max_artifact_bytes : int, optional
        Maximum allowed artifact size in bytes. Default is 20 MB.
    redaction : RedactionConfig, optional
        Configuration for redacting sensitive information.

    Attributes
    ----------
    objects_dir : Path
        Path to the objects storage directory.
    registry_db : Path
        Path to the registry SQLite database.
    """

    # Environment variable prefix
    ENV_PREFIX: ClassVar[str] = "DEVQUBIT_"

    root_dir: Path = field(default_factory=_default_root_dir)
    storage_url: str = ""
    registry_url: str = ""
    capture_pip: bool = True
    capture_git: bool = True
    validate: bool = True
    max_artifact_bytes: int = 20 * 1024 * 1024  # 20 MB
    redaction: RedactionConfig = field(default_factory=RedactionConfig)

    def __post_init__(self) -> None:
        """Expand paths and set default URLs after initialization."""
        # Ensure root_dir is a resolved Path
        if isinstance(self.root_dir, str):
            self.root_dir = Path(self.root_dir)
        self.root_dir = self.root_dir.expanduser().resolve()

        # Set default URLs based on root_dir if not provided
        if not self.storage_url:
            self.storage_url = f"file://{self.root_dir}/objects"
        if not self.registry_url:
            self.registry_url = f"file://{self.root_dir}"

        logger.debug(
            "Config initialized: root_dir=%s, storage_url=%s",
            self.root_dir,
            self.storage_url,
        )

    @property
    def objects_dir(self) -> Path:
        """
        Get the objects storage directory path.

        Returns
        -------
        Path
            Path to ``{root_dir}/objects``.
        """
        return self.root_dir / "objects"

    @property
    def registry_db(self) -> Path:
        """
        Get the registry database file path.

        Returns
        -------
        Path
            Path to ``{root_dir}/registry.db``.
        """
        return self.root_dir / "registry.db"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Configuration as a dictionary with string paths.
        """
        return {
            "root_dir": str(self.root_dir),
            "storage_url": self.storage_url,
            "registry_url": self.registry_url,
            "capture_pip": self.capture_pip,
            "capture_git": self.capture_git,
            "validate": self.validate,
            "max_artifact_bytes": self.max_artifact_bytes,
            "redaction": {
                "enabled": self.redaction.enabled,
                "patterns": self.redaction.patterns,
                "replacement": self.redaction.replacement,
            },
        }


def _parse_bool(value: str | None, default: bool = True) -> bool:
    """
    Parse a boolean value from an environment variable string.

    Parameters
    ----------
    value : str or None
        Environment variable value to parse.
    default : bool, optional
        Default value if the input is None or empty. Default is True.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_patterns(value: str | None) -> list[str] | None:
    """
    Parse comma-separated regex patterns from an environment variable.

    Parameters
    ----------
    value : str or None
        Comma-separated patterns string.

    Returns
    -------
    list of str or None
        List of stripped, non-empty patterns, or None if input is empty.
    """
    if not value:
        return None
    patterns = [p.strip() for p in value.split(",") if p.strip()]
    return patterns if patterns else None


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Environment Variables
    ---------------------
    DEVQUBIT_HOME : str
        Root directory for devqubit data. Default is ``~/.devqubit``.
    DEVQUBIT_STORAGE_URL : str
        Object store URL. Default is ``file://{DEVQUBIT_HOME}/objects``.
    DEVQUBIT_REGISTRY_URL : str
        Registry URL. Default is ``file://{DEVQUBIT_HOME}``.
    DEVQUBIT_CAPTURE_PIP : str
        Capture pip freeze output. Values: "1", "true", "yes", "on".
        Default is true.
    DEVQUBIT_CAPTURE_GIT : str
        Capture git provenance. Values: "1", "true", "yes", "on".
        Default is true.
    DEVQUBIT_VALIDATE : str
        Validate run records against JSON schema.
        Default is true.
    DEVQUBIT_MAX_ARTIFACT_BYTES : str
        Maximum artifact size in bytes. Default is 20971520 (20 MB).
    DEVQUBIT_REDACT_DISABLE : str
        Disable credential redaction. Values: "1", "true", "yes", "on".
        Default is false (redaction enabled).
    DEVQUBIT_REDACT_PATTERNS : str
        Comma-separated additional redaction patterns (regex).
        Patterns are added to the default list.

    Returns
    -------
    Config
        Loaded configuration instance.

    Examples
    --------
    >>> import os
    >>> os.environ["DEVQUBIT_HOME"] = "/tmp/devqubit"
    >>> os.environ["DEVQUBIT_CAPTURE_PIP"] = "true"
    >>> config = load_config()
    >>> print(config.capture_pip)
    True
    """
    logger.debug("Loading configuration from environment")

    # Root directory
    home_env = os.environ.get("DEVQUBIT_HOME")
    root_dir = Path(home_env).expanduser() if home_env else _default_root_dir()

    # Redaction config
    redact_disabled = _parse_bool(
        os.environ.get("DEVQUBIT_REDACT_DISABLE"), default=False
    )
    extra_patterns = _parse_patterns(os.environ.get("DEVQUBIT_REDACT_PATTERNS"))

    patterns = list(DEFAULT_REDACT_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)
        logger.debug("Added %d custom redaction patterns", len(extra_patterns))

    redaction = RedactionConfig(
        enabled=not redact_disabled,
        patterns=patterns,
    )

    # Boolean flags with appropriate defaults
    capture_git = _parse_bool(os.environ.get("DEVQUBIT_CAPTURE_GIT"), default=True)
    capture_pip = _parse_bool(os.environ.get("DEVQUBIT_CAPTURE_PIP"), default=True)
    validate = _parse_bool(os.environ.get("DEVQUBIT_VALIDATE"), default=True)

    # Artifact size limit
    max_artifact_env = os.environ.get("DEVQUBIT_MAX_ARTIFACT_BYTES")
    max_artifact_bytes = 20 * 1024 * 1024  # 20 MB default
    if max_artifact_env:
        try:
            max_artifact_bytes = int(max_artifact_env)
        except ValueError:
            logger.warning(
                "Invalid DEVQUBIT_MAX_ARTIFACT_BYTES=%r, using default %d",
                max_artifact_env,
                max_artifact_bytes,
            )

    config = Config(
        root_dir=root_dir,
        storage_url=os.environ.get("DEVQUBIT_STORAGE_URL", ""),
        registry_url=os.environ.get("DEVQUBIT_REGISTRY_URL", ""),
        capture_pip=capture_pip,
        capture_git=capture_git,
        validate=validate,
        max_artifact_bytes=max_artifact_bytes,
        redaction=redaction,
    )

    logger.debug("Configuration loaded: %s", config.to_dict())
    return config


# Global cached configuration instance
_config: Config | None = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """
    Get the current configuration (cached singleton).

    The configuration is loaded once from environment variables and
    cached for subsequent calls. Use :func:`reset_config` to force
    reloading, or :func:`set_config` to set programmatically.

    Returns
    -------
    Config
        The current configuration instance.

    See Also
    --------
    reset_config : Clear the cached configuration.
    set_config : Set a custom configuration.
    load_config : Load configuration from environment.
    """
    global _config
    if _config is not None:
        return _config
    with _config_lock:
        # Double-checked locking
        if _config is None:
            _config = load_config()
        return _config


def reset_config() -> None:
    """
    Reset the cached configuration.

    Forces the next :func:`get_config` call to reload configuration
    from environment variables. Also clears the compiled patterns cache.

    See Also
    --------
    get_config : Get the current configuration.
    set_config : Set a custom configuration.
    """
    global _config, _compiled_default_patterns
    with _config_lock:
        _config = None
        _compiled_default_patterns = None
    logger.debug("Configuration cache reset")


def set_config(config: Config) -> None:
    """
    Set the global configuration programmatically.

    Allows setting configuration without environment variables, useful
    for testing or programmatic configuration.

    Parameters
    ----------
    config : Config
        Configuration instance to use.

    See Also
    --------
    get_config : Get the current configuration.
    reset_config : Reset to reload from environment.

    Examples
    --------
    >>> from pathlib import Path
    >>> custom_config = Config(root_dir=Path("/tmp/test"))
    >>> set_config(custom_config)
    >>> get_config().root_dir
    PosixPath('/tmp/test')
    """
    global _config
    with _config_lock:
        _config = config
    logger.debug("Configuration set programmatically: %s", config.root_dir)
