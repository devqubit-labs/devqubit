# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Producer information for UEC 2.0.

This module defines ProducerInfo which captures the complete SDK
stack that produced an execution envelope. This is critical for:

- Debug: Understanding which SDK versions were involved
- Compatibility: Detecting version mismatches
- Reproducibility: Recreating the exact environment

Examples
--------
Simple Qiskit setup:

>>> producer = ProducerInfo.create(
...     adapter="devqubit-qiskit",
...     adapter_version="0.3.0",
...     sdk="qiskit",
...     sdk_version="1.3.0",
...     frontends=["qiskit"],
... )

Multi-layer PennyLane → Braket stack:

>>> producer = ProducerInfo.create(
...     adapter="devqubit-pennylane",
...     adapter_version="0.2.0",
...     sdk="braket-sdk",
...     sdk_version="1.80.0",
...     frontends=["pennylane", "amazon-braket-pennylane-plugin", "braket-sdk"],
... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


def _get_engine_version() -> str:
    """
    Get devqubit-engine version from package metadata.

    Returns
    -------
    str
        Version string or "unknown" if not installed.
    """
    try:
        from importlib.metadata import version

        return version("devqubit-engine")
    except Exception:
        return "unknown"


@dataclass
class ProducerInfo:
    """
    SDK stack information for the envelope producer.

    Captures the complete toolchain that produced an execution envelope,
    from the high-level frontend down to the physical backend SDK.

    Parameters
    ----------
    name : str
        Producer name. Always "devqubit" for devqubit-engine.
    engine_version : str
        devqubit-engine version string.
    adapter : str
        Adapter identifier (e.g., "devqubit-qiskit", "devqubit-braket").
    adapter_version : str
        Adapter version string.
    sdk : str
        Primary/lowest SDK name (e.g., "qiskit", "braket-sdk", "cirq").
    sdk_version : str
        Primary SDK version string.
    frontends : list of str
        SDK stack from highest to lowest layer.
        E.g., ["pennylane", "amazon-braket-pennylane-plugin", "braket-sdk"]
        For simple setups: ["qiskit"]
    build : str, optional
        Build identifier (git commit, dirty flag).
    """

    name: str
    engine_version: str
    adapter: str
    adapter_version: str
    sdk: str
    sdk_version: str
    frontends: list[str] = field(default_factory=list)
    build: str | None = None

    def __post_init__(self) -> None:
        """Validate required fields."""
        if not self.frontends:
            raise ValueError(
                "frontends must be a non-empty list. "
                "For simple setups, use [sdk_name]. "
                "For multi-layer stacks, list from highest to lowest."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary with all producer fields.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "engine_version": self.engine_version,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "sdk": self.sdk,
            "sdk_version": self.sdk_version,
            "frontends": self.frontends,
        }
        if self.build:
            d["build"] = self.build
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProducerInfo:
        """
        Create ProducerInfo from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with producer fields.

        Returns
        -------
        ProducerInfo
            Parsed producer info.
        """
        return cls(
            name=str(d.get("name", "devqubit")),
            engine_version=str(d.get("engine_version", "unknown")),
            adapter=str(d.get("adapter", "")),
            adapter_version=str(d.get("adapter_version", "")),
            sdk=str(d.get("sdk", "")),
            sdk_version=str(d.get("sdk_version", "")),
            frontends=d.get("frontends", []),
            build=d.get("build"),
        )

    @classmethod
    def create(
        cls,
        *,
        adapter: str,
        adapter_version: str,
        sdk: str,
        sdk_version: str,
        frontends: list[str],
        build: str | None = None,
    ) -> ProducerInfo:
        """
        Create ProducerInfo with auto-detected engine version.

        This is the recommended factory method for adapters.

        Parameters
        ----------
        adapter : str
            Adapter identifier (e.g., "devqubit-qiskit").
        adapter_version : str
            Adapter version string.
        sdk : str
            Primary SDK name (e.g., "qiskit", "braket-sdk").
        sdk_version : str
            Primary SDK version string.
        frontends : list of str
            SDK stack from highest to lowest layer.
        build : str, optional
            Build identifier.

        Returns
        -------
        ProducerInfo
            Configured producer info with auto-detected engine version.
        """
        return cls(
            name="devqubit",
            engine_version=_get_engine_version(),
            adapter=adapter,
            adapter_version=adapter_version,
            sdk=sdk,
            sdk_version=sdk_version,
            frontends=frontends,
            build=build,
        )
