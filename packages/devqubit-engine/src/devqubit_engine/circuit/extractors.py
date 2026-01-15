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
from typing import TYPE_CHECKING

from devqubit_engine.circuit.models import SDK, CircuitData, CircuitFormat
from devqubit_engine.core.record import RunRecord
from devqubit_engine.storage.artifacts.io import load_artifact_bytes
from devqubit_engine.storage.artifacts.lookup import find_artifact
from devqubit_engine.storage.types import ArtifactRef, ObjectStoreProtocol


if TYPE_CHECKING:
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


# Format detection patterns: (kind_pattern, CircuitFormat, SDK, is_binary)
_FORMAT_PATTERNS: tuple[tuple[str, CircuitFormat, SDK, bool], ...] = (
    ("qpy", CircuitFormat.QPY, SDK.QISKIT, True),
    ("jaqcd", CircuitFormat.JAQCD, SDK.BRAKET, False),
    ("cirq", CircuitFormat.CIRQ_JSON, SDK.CIRQ, False),
    ("tape", CircuitFormat.TAPE_JSON, SDK.PENNYLANE, False),
)


def detect_sdk(record: RunRecord) -> SDK:
    """
    Detect SDK from a run record.

    Uses the adapter name as the primary indicator.

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

    sdk_patterns = (
        ("qiskit", SDK.QISKIT),
        ("braket", SDK.BRAKET),
        ("cirq", SDK.CIRQ),
        ("pennylane", SDK.PENNYLANE),
    )

    for pattern, sdk in sdk_patterns:
        if pattern in adapter:
            return sdk

    logger.debug("Could not detect SDK from record")
    return SDK.UNKNOWN


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

    # Try native formats first
    if prefer_native:
        circuit = _try_native_formats(record, store, sdk)
        if circuit:
            return circuit

    # OpenQASM fallback
    return _try_openqasm_formats(record, store, sdk)


def _try_native_formats(
    record: RunRecord,
    store: ObjectStoreProtocol,
    sdk: SDK,
) -> CircuitData | None:
    """Try loading circuit from native SDK format artifacts."""
    for kind_pattern, fmt, fmt_sdk, is_binary in _FORMAT_PATTERNS:
        # Skip if SDK is known and doesn't match
        if sdk != SDK.UNKNOWN and fmt_sdk != sdk:
            continue

        artifact = find_artifact(
            record,
            role="program",
            kind_contains=kind_pattern,
        )
        if not artifact:
            continue

        data = load_artifact_bytes(artifact, store)
        if data is None:
            continue

        logger.debug("Found native format artifact: %s (%s)", artifact.kind, fmt.value)

        if is_binary:
            return CircuitData(
                data=data,
                format=fmt,
                sdk=fmt_sdk,
            )
        return CircuitData(
            data=data.decode("utf-8"),
            format=fmt,
            sdk=fmt_sdk,
        )

    return None


def _try_openqasm_formats(
    record: RunRecord,
    store: ObjectStoreProtocol,
    sdk: SDK,
) -> CircuitData | None:
    """Try loading circuit from OpenQASM artifacts."""
    for kind_pattern in ("openqasm3", "openqasm", "qasm"):
        artifact = find_artifact(
            record,
            role="program",
            kind_contains=kind_pattern,
        )
        if not artifact:
            continue

        data = load_artifact_bytes(artifact, store)
        if data is None:
            continue

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("Failed to decode OpenQASM artifact as UTF-8")
            continue

        # Detect QASM version
        fmt = CircuitFormat.OPENQASM3
        if re.match(r"^\s*OPENQASM\s+2\.", text):
            fmt = CircuitFormat.OPENQASM2

        logger.debug("Found OpenQASM artifact: %s", fmt.value)
        return CircuitData(data=text, format=fmt, sdk=sdk)

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
    for kind_pattern in ("openqasm3", "openqasm", "qasm"):
        artifact = find_artifact(
            record,
            role="program",
            kind_contains=kind_pattern,
        )
        if not artifact:
            continue

        data = load_artifact_bytes(artifact, store)
        if data is None:
            continue

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("Failed to decode OpenQASM artifact as UTF-8")
            continue

    return None


def extract_circuit_from_refs(
    refs: list[ArtifactRef],
    store: ObjectStoreProtocol,
    *,
    prefer_formats: list[str] | None = None,
) -> CircuitData | None:
    """
    Extract circuit data from artifact references.

    Uses UEC program artifact references to load circuit data,
    ensuring we get exactly the circuit referenced by the envelope.

    Parameters
    ----------
    refs : list of ArtifactRef
        Artifact references from envelope.program.logical or .physical.
    store : ObjectStoreProtocol
        Object store to load artifact data from.
    prefer_formats : list of str, optional
        Preferred format order (e.g., ["openqasm3", "qpy"]).
        Defaults to OpenQASM3 first.

    Returns
    -------
    CircuitData or None
        Extracted circuit data, or None if not found.
    """
    if not refs:
        return None

    if prefer_formats is None:
        prefer_formats = ["openqasm3", "openqasm", "qasm", "qpy", "jaqcd", "cirq"]

    # Build lookup by kind
    ref_by_kind = {ref.kind.lower(): ref for ref in refs}

    # Try preferred formats in order
    for fmt_pattern in prefer_formats:
        for kind_lower, ref in ref_by_kind.items():
            if fmt_pattern not in kind_lower:
                continue

            circuit = _load_circuit_from_ref(ref, kind_lower, store)
            if circuit:
                return circuit

    # Fallback: try first available ref
    return _load_fallback_ref(refs[0], store)


def _load_circuit_from_ref(
    ref: ArtifactRef,
    kind_lower: str,
    store: ObjectStoreProtocol,
) -> CircuitData | None:
    """Load circuit data from a single artifact ref based on kind."""
    try:
        data = store.get_bytes(ref.digest)
    except Exception as e:
        logger.debug("Failed to load artifact %s: %s", ref.digest[:24], e)
        return None

    # Determine format and create CircuitData
    if "qpy" in kind_lower:
        return CircuitData(data=data, format=CircuitFormat.QPY, sdk=SDK.QISKIT)

    # Text-based formats
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if "openqasm3" in kind_lower or "qasm3" in kind_lower:
        return CircuitData(
            data=text,
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.UNKNOWN,
        )

    if "openqasm" in kind_lower or "qasm" in kind_lower:
        fmt = CircuitFormat.OPENQASM3
        if re.match(r"^\s*OPENQASM\s+2\.", text):
            fmt = CircuitFormat.OPENQASM2
        return CircuitData(
            data=text,
            format=fmt,
            sdk=SDK.UNKNOWN,
        )

    if "jaqcd" in kind_lower:
        return CircuitData(
            data=text,
            format=CircuitFormat.JAQCD,
            sdk=SDK.BRAKET,
        )

    if "cirq" in kind_lower:
        return CircuitData(
            data=text,
            format=CircuitFormat.CIRQ_JSON,
            sdk=SDK.CIRQ,
        )

    # Generic text
    return CircuitData(
        data=text,
        format=CircuitFormat.OPENQASM3,
        sdk=SDK.UNKNOWN,
    )


def _load_fallback_ref(
    ref: ArtifactRef,
    store: ObjectStoreProtocol,
) -> CircuitData | None:
    """Load circuit from ref with format auto-detection."""
    try:
        data = store.get_bytes(ref.digest)
    except Exception as e:
        logger.debug("Failed to load fallback artifact: %s", e)
        return None

    # Try as text first
    try:
        text = data.decode("utf-8")
        return CircuitData(
            data=text,
            format=CircuitFormat.OPENQASM3,
            sdk=SDK.UNKNOWN,
        )
    except UnicodeDecodeError:
        # Binary format - assume QPY
        return CircuitData(
            data=data,
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )


def extract_circuit_from_envelope(
    envelope: ExecutionEnvelope,
    store: ObjectStoreProtocol,
    *,
    which: str = "logical",
    prefer_formats: list[str] | None = None,
) -> CircuitData | None:
    """
    Extract circuit data from ExecutionEnvelope.

    This is the primary UEC-aware circuit extraction function. Use this
    when you have an envelope and need to extract circuit data.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope containing program snapshot.
    store : ObjectStoreProtocol
        Object store to load artifact data from.
    which : {"logical", "physical"}
        Which circuit to extract: "logical" (pre-transpilation) or
        "physical" (post-transpilation/executed).
    prefer_formats : list of str, optional
        Preferred format order.

    Returns
    -------
    CircuitData or None
        Extracted circuit data, or None if not found.

    Raises
    ------
    ValueError
        If `which` is not "logical" or "physical".

    Examples
    --------
    >>> from devqubit_engine.uec.resolver import resolve_envelope
    >>> from devqubit_engine.circuit.extractors import extract_circuit_from_envelope
    >>>
    >>> envelope = resolve_envelope(record, store)
    >>> circuit = extract_circuit_from_envelope(envelope, store, which="logical")
    """
    if not envelope.program:
        logger.debug("No program snapshot in envelope")
        return None

    if which == "logical":
        artifacts = envelope.program.logical
    elif which == "physical":
        artifacts = envelope.program.physical
    else:
        raise ValueError(f"which must be 'logical' or 'physical', got '{which}'")

    if not artifacts:
        logger.debug("No %s artifacts in program snapshot", which)
        return None

    refs = [a.ref for a in artifacts if a.ref]
    return extract_circuit_from_refs(
        refs,
        store,
        prefer_formats=prefer_formats,
    )
