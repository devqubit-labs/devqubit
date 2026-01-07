# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Circuit extraction from run records.

This module provides functions for extracting circuit data from run
records stored in devqubit. It handles SDK detection, artifact discovery,
and format conversion.
"""

from __future__ import annotations

import logging
import re

from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat
from devqubit_engine.core.record import RunRecord
from devqubit_engine.core.types import ArtifactRef
from devqubit_engine.storage.protocols import ObjectStoreProtocol


logger = logging.getLogger(__name__)


def detect_sdk(record: RunRecord) -> SDK:
    """
    Detect SDK from a run record.

    Uses the adapter name as the primary indicator, falling back to
    artifact kinds if adapter is not recognized.

    Parameters
    ----------
    record : RunRecord
        Run record to analyze.

    Returns
    -------
    SDK
        Detected SDK, or SDK.UNKNOWN if detection fails.
    """
    adapter = (record.adapter or "").lower()

    # Primary detection: adapter name
    if "qiskit" in adapter:
        return SDK.QISKIT
    elif "braket" in adapter:
        return SDK.BRAKET
    elif "cirq" in adapter:
        return SDK.CIRQ
    elif "pennylane" in adapter:
        return SDK.PENNYLANE

    # Fallback: infer from artifact kinds
    for artifact in record.artifacts:
        kind = artifact.kind.lower()
        if "qpy" in kind or "qiskit" in kind:
            return SDK.QISKIT
        elif "jaqcd" in kind or "braket" in kind:
            return SDK.BRAKET
        elif "cirq" in kind:
            return SDK.CIRQ
        elif "tape" in kind or "pennylane" in kind:
            return SDK.PENNYLANE

    logger.debug("Could not detect SDK from record")
    return SDK.UNKNOWN


def find_artifact(
    record: RunRecord,
    *,
    role: str | None = None,
    kind_contains: str | None = None,
) -> ArtifactRef | None:
    """
    Find artifact matching criteria.

    Searches through run artifacts to find one matching the specified
    role and/or kind pattern.

    Parameters
    ----------
    record : RunRecord
        Run record to search.
    role : str, optional
        Required artifact role (exact match).
    kind_contains : str, optional
        Substring to match in artifact kind (case-insensitive).

    Returns
    -------
    ArtifactRef or None
        First matching artifact, or None if no match found.
    """
    for artifact in record.artifacts:
        # Check role if specified
        if role is not None and artifact.role != role:
            continue

        # Check kind if specified
        if kind_contains is not None:
            if kind_contains.lower() not in artifact.kind.lower():
                continue

        return artifact

    return None


# Format patterns: (kind_contains, CircuitFormat, SDK, is_binary)
_FORMAT_PATTERNS: list[tuple[str, CircuitFormat, SDK, bool]] = [
    ("qpy", CircuitFormat.QPY, SDK.QISKIT, True),
    ("jaqcd", CircuitFormat.JAQCD, SDK.BRAKET, False),
    ("cirq", CircuitFormat.CIRQ_JSON, SDK.CIRQ, False),
    ("tape", CircuitFormat.TAPE_JSON, SDK.PENNYLANE, False),
]


def extract_circuit(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    prefer_native: bool = True,
) -> CircuitData | None:
    """
    Extract circuit data from a run record.

    Searches for circuit artifacts in the run record and loads the
    circuit data from the object store.

    Parameters
    ----------
    record : RunRecord
        Run record to extract circuit from.
    store : ObjectStoreProtocol
        Object store to load artifact data from.
    prefer_native : bool, optional
        If True (default), try native SDK formats first before falling
        back to OpenQASM.

    Returns
    -------
    CircuitData or None
        Extracted circuit data, or None if no circuit found.

    Notes
    -----
    The extraction order is:

    1. Native format matching the detected SDK (if prefer_native=True)
    2. OpenQASM 3 artifacts
    3. OpenQASM 2 artifacts
    4. Generic QASM artifacts
    """
    sdk = detect_sdk(record)
    logger.debug("Extracting circuit from record, detected SDK: %s", sdk.value)

    if prefer_native:
        for kind_contains, fmt, fmt_sdk, is_binary in _FORMAT_PATTERNS:
            # Match SDK if known
            if sdk != SDK.UNKNOWN and fmt_sdk != sdk:
                continue

            artifact = find_artifact(
                record, role="program", kind_contains=kind_contains
            )
            if artifact:
                try:
                    data = store.get_bytes(artifact.digest)
                    logger.debug(
                        "Found native format artifact: %s (%s)",
                        artifact.kind,
                        fmt.value,
                    )
                    if is_binary:
                        return CircuitData(data=data, format=fmt, sdk=fmt_sdk)
                    else:
                        return CircuitData(
                            data=data.decode("utf-8"),
                            format=fmt,
                            sdk=fmt_sdk,
                        )
                except Exception as e:
                    logger.debug(
                        "Failed to load artifact %s: %s", artifact.digest[:24], e
                    )
                    continue

    # OpenQASM fallback
    for kind_contains in ("openqasm3", "openqasm", "qasm"):
        artifact = find_artifact(record, role="program", kind_contains=kind_contains)
        if artifact:
            try:
                data = store.get_bytes(artifact.digest)
                text = data.decode("utf-8")

                # Detect QASM version
                fmt = CircuitFormat.OPENQASM3
                if re.match(r"^\s*OPENQASM\s+2\.", text):
                    fmt = CircuitFormat.OPENQASM2

                logger.debug("Found OpenQASM artifact: %s", fmt.value)
                return CircuitData(data=text, format=fmt, sdk=sdk)
            except Exception as e:
                logger.debug("Failed to load OpenQASM artifact: %s", e)
                continue

    logger.debug("No circuit artifact found in record")
    return None


def extract_openqasm_source(
    record: RunRecord,
    store: ObjectStoreProtocol,
) -> str | None:
    """
    Extract OpenQASM source text from a run record.

    Searches for OpenQASM artifacts and returns the source as text.
    Does not attempt to load native format artifacts.

    Parameters
    ----------
    record : RunRecord
        Run record to search.
    store : ObjectStoreProtocol
        Object store to load artifact data from.

    Returns
    -------
    str or None
        OpenQASM source text, or None if not found.
    """
    for kind_contains in ("openqasm3", "openqasm", "qasm"):
        artifact = find_artifact(record, role="program", kind_contains=kind_contains)
        if artifact:
            try:
                data = store.get_bytes(artifact.digest)
                return data.decode("utf-8")
            except Exception as e:
                logger.debug("Failed to load OpenQASM artifact: %s", e)
                continue

    return None
