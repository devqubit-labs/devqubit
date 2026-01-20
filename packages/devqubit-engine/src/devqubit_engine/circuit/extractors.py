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
from devqubit_engine.storage.artifacts.io import load_artifact_bytes
from devqubit_engine.storage.artifacts.lookup import find_artifact
from devqubit_engine.storage.types import ArtifactRef, ObjectStoreProtocol
from devqubit_engine.tracking.record import RunRecord


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


def _parse_sdk(sdk_string: str | None) -> SDK:
    """
    Parse SDK string to SDK enum, with fallback to UNKNOWN.

    Parameters
    ----------
    sdk_string : str or None
        SDK identifier string (e.g., "qiskit", "braket").

    Returns
    -------
    SDK
        Parsed SDK enum value, or SDK.UNKNOWN if parsing fails.
    """
    if not sdk_string:
        return SDK.UNKNOWN
    try:
        return SDK(sdk_string.lower())
    except ValueError:
        logger.debug("Unknown SDK string: %s", sdk_string)
        return SDK.UNKNOWN


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
    envelope: ExecutionEnvelope | None = None,
    which: str = "logical",
    prefer_formats: list[str] | None = None,
    prefer_native: bool = True,
    uec_first: bool = True,
) -> CircuitData | None:
    """
    Extract circuit data from a run record with UEC-first strategy.

    This is the canonical circuit extraction function. It implements
    UEC-first logic: if an envelope is available, circuit is extracted
    from envelope refs. Fallback to RunRecord scanning is only allowed
    for synthesized (manual) envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record to extract circuit from.
    store : ObjectStoreProtocol
        Object store to load artifact data from.
    envelope : ExecutionEnvelope, optional
        Pre-resolved envelope. If None and uec_first=True, attempts to
        load envelope from record artifacts.
    which : {"logical", "physical"}, default="logical"
        Which circuit to extract: "logical" (pre-transpilation) or
        "physical" (post-transpilation/executed).
    prefer_formats : list of str, optional
        Preferred format order for envelope extraction.
    prefer_native : bool, default=True
        If True, try native SDK formats first before falling back to
        OpenQASM when scanning RunRecord (non-UEC path).
    uec_first : bool, default=True
        If True, prefer envelope refs over RunRecord scanning.

    Returns
    -------
    CircuitData or None
        Extracted circuit data, or None if no circuit found.

    Notes
    -----
    The extraction strategy is:

    1. If ``uec_first`` and envelope available:
       a. Try ``extract_circuit_from_envelope()``
       b. If envelope is NOT synthesized and no refs found: return None
          (don't guess from RunRecord for adapter envelopes)
    2. For synthesized/manual envelopes or ``uec_first=False``:
       a. Native format matching the detected SDK (if prefer_native=True)
       b. OpenQASM 3/2 artifacts
    """
    # UEC-first path
    if uec_first:
        # Try to get or load envelope
        env = envelope
        if env is None:
            env = _try_load_envelope(record, store)

        if env is not None:
            # Extract from envelope
            circuit = extract_circuit_from_envelope(
                env,
                store,
                which=which,
                prefer_formats=prefer_formats,
            )
            if circuit is not None:
                return circuit

            # Check if we should fallback to RunRecord scanning
            is_synthesized = env.metadata.get("synthesized_from_run", False)
            if not is_synthesized:
                # Adapter envelope without circuit refs - don't guess
                logger.debug(
                    "Adapter envelope for run %s has no circuit refs for '%s'",
                    record.run_id,
                    which,
                )
                return None

    # Fallback: scan RunRecord artifacts (for synthesized/manual or uec_first=False)
    sdk = detect_sdk(record)
    logger.debug("Extracting circuit from record, detected SDK: %s", sdk.value)

    # Try native formats first
    if prefer_native:
        circuit = _try_native_formats(record, store, sdk)
        if circuit:
            return circuit

    # OpenQASM fallback
    return _try_openqasm_formats(record, store, sdk)


def _try_load_envelope(
    record: RunRecord,
    store: ObjectStoreProtocol,
) -> ExecutionEnvelope | None:
    """Try to load envelope from record artifacts without raising."""
    try:
        from devqubit_engine.uec.api.resolve import load_envelope

        return load_envelope(record, store, raise_on_error=False)
    except Exception:
        return None


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


def extract_circuit_from_refs(
    refs: list[ArtifactRef],
    store: ObjectStoreProtocol,
    *,
    prefer_formats: list[str] | None = None,
    sdk_hint: SDK | None = None,
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
        Defaults to native formats first, then interchange formats.
    sdk_hint : SDK, optional
        SDK hint for interchange formats (OpenQASM) that don't have
        inherent SDK information. Extracted from envelope.producer.sdk.

    Returns
    -------
    CircuitData or None
        Extracted circuit data, or None if not found.
    """
    if not refs:
        return None

    # Default: prefer native formats (which carry SDK info) over interchange
    if prefer_formats is None:
        prefer_formats = [
            "qpy",
            "jaqcd",
            "cirq",
            "tape",
            "openqasm3",
            "openqasm",
            "qasm",
        ]

    effective_sdk_hint = sdk_hint or SDK.UNKNOWN

    # Iterate by format preference, then by ref order (preserves duplicates)
    for fmt_pattern in prefer_formats:
        for ref in refs:
            kind_lower = ref.kind.lower()
            if fmt_pattern not in kind_lower:
                continue

            circuit = _load_circuit_from_ref(ref, kind_lower, store, effective_sdk_hint)
            if circuit:
                return circuit

    # Fallback: try first available ref
    return _load_fallback_ref(refs[0], store, effective_sdk_hint)


def _load_circuit_from_ref(
    ref: ArtifactRef,
    kind_lower: str,
    store: ObjectStoreProtocol,
    sdk_hint: SDK,
) -> CircuitData | None:
    """
    Load circuit data from a single artifact ref based on kind.

    Parameters
    ----------
    ref : ArtifactRef
        Reference to the artifact.
    kind_lower : str
        Lowercased kind string for pattern matching.
    store : ObjectStoreProtocol
        Object store to load from.
    sdk_hint : SDK
        SDK hint for interchange formats.

    Returns
    -------
    CircuitData or None
        Loaded circuit data, or None if loading fails.
    """
    try:
        data = store.get_bytes(ref.digest)
    except Exception as e:
        logger.debug("Failed to load artifact %s: %s", ref.digest[:24], e)
        return None

    # Native formats - SDK determined by format
    if "qpy" in kind_lower:
        return CircuitData(
            data=data,
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )

    if "jaqcd" in kind_lower:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return CircuitData(
            data=text,
            format=CircuitFormat.JAQCD,
            sdk=SDK.BRAKET,
        )

    if "cirq" in kind_lower and "circuit" not in kind_lower:
        # Match "cirq" but not generic "circuit"
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return CircuitData(
            data=text,
            format=CircuitFormat.CIRQ_JSON,
            sdk=SDK.CIRQ,
        )

    if "tape" in kind_lower:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return CircuitData(
            data=text,
            format=CircuitFormat.TAPE_JSON,
            sdk=SDK.PENNYLANE,
        )

    # Interchange formats - use sdk_hint
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Binary data that doesn't match known native formats
        logger.debug("Unknown binary format for kind: %s", kind_lower)
        return None

    if "openqasm3" in kind_lower or "qasm3" in kind_lower:
        return CircuitData(
            data=text,
            format=CircuitFormat.OPENQASM3,
            sdk=sdk_hint,
        )

    if "openqasm" in kind_lower or "qasm" in kind_lower:
        # Auto-detect QASM version from content
        fmt = CircuitFormat.OPENQASM3
        if re.match(r"^\s*OPENQASM\s+2\.", text):
            fmt = CircuitFormat.OPENQASM2
        return CircuitData(
            data=text,
            format=fmt,
            sdk=sdk_hint,
        )

    # Generic text - assume OpenQASM3 with sdk_hint
    logger.debug("Unknown text format for kind: %s, assuming OpenQASM3", kind_lower)
    return CircuitData(
        data=text,
        format=CircuitFormat.OPENQASM3,
        sdk=sdk_hint,
    )


def _load_fallback_ref(
    ref: ArtifactRef,
    store: ObjectStoreProtocol,
    sdk_hint: SDK,
) -> CircuitData | None:
    """
    Load circuit from ref with format auto-detection.

    Parameters
    ----------
    ref : ArtifactRef
        Reference to the artifact.
    store : ObjectStoreProtocol
        Object store to load from.
    sdk_hint : SDK
        SDK hint for text-based formats.

    Returns
    -------
    CircuitData or None
        Loaded circuit data, or None if loading fails.
    """
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
            sdk=sdk_hint,
        )
    except UnicodeDecodeError:
        # Binary format - assume QPY (Qiskit native)
        return CircuitData(
            data=data,
            format=CircuitFormat.QPY,
            sdk=SDK.QISKIT,
        )


def _extract_sdk_from_envelope(envelope: ExecutionEnvelope) -> SDK:
    """
    Extract SDK from envelope producer information.

    Parameters
    ----------
    envelope : ExecutionEnvelope
        Envelope to extract SDK from.

    Returns
    -------
    SDK
        Detected SDK, or SDK.UNKNOWN if not available.
    """
    if envelope.producer and envelope.producer.sdk:
        return _parse_sdk(envelope.producer.sdk)
    return SDK.UNKNOWN


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
        Preferred format order. Defaults to native formats first.

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

    # Extract SDK hint from envelope producer
    sdk_hint = _extract_sdk_from_envelope(envelope)
    logger.debug("Extracted SDK hint from envelope: %s", sdk_hint.value)

    return extract_circuit_from_refs(
        refs,
        store,
        prefer_formats=prefer_formats,
        sdk_hint=sdk_hint,
    )
