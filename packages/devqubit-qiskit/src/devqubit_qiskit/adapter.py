# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Qiskit adapter for devqubit tracking system.

Provides integration with Qiskit backends, enabling automatic
tracking of quantum circuit execution, results, and device configurations
following the devqubit Uniform Execution Contract (UEC).

The adapter produces an ExecutionEnvelope containing four canonical snapshots:
- DeviceSnapshot: Backend state and calibration
- ProgramSnapshot: Logical circuit artifacts
- ExecutionSnapshot: Submission and job metadata
- ResultSnapshot: Normalized measurement results

Supported Backends
------------------
This adapter supports Qiskit BackendV2 implementations including:
- qiskit-aer simulators (AerSimulator, etc.)
- qiskit-ibm-runtime backends (when used directly, not via primitives)
- Fake backends for testing

Note: Legacy BackendV1 is not supported. Use BackendV2-based backends.
For Runtime primitives (SamplerV2, EstimatorV2), use the qiskit-runtime adapter.

Example
-------
>>> from qiskit import QuantumCircuit
>>> from qiskit_aer import AerSimulator
>>> from devqubit_engine.core import track
>>>
>>> qc = QuantumCircuit(2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> qc.measure_all()
>>>
>>> with track(project="my_experiment") as run:
...     backend = run.wrap(AerSimulator())
...     job = backend.run(qc, shots=1000)
...     result = job.result()
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.core.run import Run
from devqubit_engine.uec.device import DeviceSnapshot
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.uec.execution import ExecutionSnapshot
from devqubit_engine.uec.producer import ProducerInfo
from devqubit_engine.uec.program import (
    ProgramArtifact,
    ProgramSnapshot,
    TranspilationInfo,
)
from devqubit_engine.uec.result import (
    CountsFormat,
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.uec.types import (
    ArtifactRef,
    ProgramRole,
    TranspilationMode,
)
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_engine.utils.time_utils import utc_now_iso
from devqubit_qiskit.results import (
    extract_result_metadata,
    normalize_result_counts,
)
from devqubit_qiskit.serialization import QiskitCircuitSerializer
from devqubit_qiskit.snapshot import create_device_snapshot
from devqubit_qiskit.utils import extract_job_id, get_backend_name, qiskit_version
from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2


logger = logging.getLogger(__name__)

# Adapter version for ProducerInfo
_ADAPTER_VERSION = "0.4.0"

# Module-level serializer instance
_serializer = QiskitCircuitSerializer()


def _materialize_circuits(circuits: Any) -> tuple[list[Any], bool]:
    """
    Materialize circuit inputs exactly once.

    Prevents consumption bugs when the user provides generators/iterators.

    Parameters
    ----------
    circuits : Any
        A QuantumCircuit, or an iterable of QuantumCircuit objects.

    Returns
    -------
    circuit_list : list
        List of circuit-like objects.
    was_single : bool
        True if the input was a single circuit-like object.
    """
    if circuits is None:
        return [], False

    # QuantumCircuit is iterable over instructions, so check explicitly
    if isinstance(circuits, QuantumCircuit):
        return [circuits], True

    if isinstance(circuits, (list, tuple)):
        return list(circuits), False

    # Generic iterables (generator, iterator, etc.)
    try:
        return list(circuits), False
    except TypeError:
        # Not iterable -> treat as a single circuit-like payload
        return [circuits], True


def _compute_circuit_hash(circuits: list[Any]) -> str | None:
    """
    Compute a structure-only hash for Qiskit QuantumCircuit objects.

    Captures circuit structure (gates, qubits, classical bits) while
    ignoring parameter values for deduplication purposes.

    Parameters
    ----------
    circuits : list[Any]
        List of Qiskit QuantumCircuit objects.

    Returns
    -------
    str or None
        Full SHA-256 digest in format ``sha256:<hex>``, or None if empty.

    Notes
    -----
    The hash captures:
    - Operation names (e.g., 'rx', 'cx', 'measure')
    - Ordered qubit indices
    - Ordered clbit indices (measurement wiring)
    - Parameter arity (count only, not values)
    - Classical condition presence
    """
    if not circuits:
        return None

    circuit_signatures: list[str] = []

    for circuit in circuits:
        try:
            # Precompute indices for speed and stability
            qubit_index = {
                q: i for i, q in enumerate(getattr(circuit, "qubits", ()) or ())
            }
            clbit_index = {
                c: i for i, c in enumerate(getattr(circuit, "clbits", ()) or ())
            }

            op_sigs: list[str] = []
            for instr in getattr(circuit, "data", []) or []:
                op = getattr(instr, "operation", None)
                name = getattr(op, "name", None)
                op_name = name if isinstance(name, str) and name else type(op).__name__

                # Qubits / clbits in order (control-target order matters)
                qs: list[int] = []
                for q in getattr(instr, "qubits", ()) or ():
                    if q in qubit_index:
                        qs.append(qubit_index[q])
                    else:
                        # Fallback if circuit has unusual bit containers
                        qs.append(getattr(circuit.find_bit(q), "index", -1))
                cs: list[int] = []
                for c in getattr(instr, "clbits", ()) or ():
                    if c in clbit_index:
                        cs.append(clbit_index[c])
                    else:
                        cs.append(getattr(circuit.find_bit(c), "index", -1))

                # Parameter arity (count only)
                params = getattr(op, "params", None)
                parity = len(params) if isinstance(params, (list, tuple)) else 0

                # Classical condition presence
                cond = getattr(op, "condition", None)
                has_cond = 1 if cond is not None else 0

                op_sigs.append(
                    f"{op_name}|p{parity}|q{tuple(qs)}|c{tuple(cs)}|if{has_cond}"
                )

            circuit_signatures.append("||".join(op_sigs))

        except Exception:
            # Conservative fallback: avoid breaking tracking
            circuit_signatures.append(str(circuit)[:500])

    payload = "\n".join(circuit_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _circuits_to_text(circuits: list[Any]) -> str:
    """
    Convert circuits to human-readable text diagrams.

    Parameters
    ----------
    circuits : list
        List of QuantumCircuit objects.

    Returns
    -------
    str
        Combined text diagram of all circuits.
    """
    parts: list[str] = []

    for i, circuit in enumerate(circuits):
        if i > 0:
            parts.append("")  # Blank line between circuits

        name = getattr(circuit, "name", None) or f"circuit_{i}"
        parts.append(f"[{i}] {name}")

        try:
            diagram = circuit.draw(output="text", fold=80)
            if hasattr(diagram, "single_string"):
                parts.append(diagram.single_string())
            else:
                parts.append(str(diagram))
        except Exception:
            parts.append(str(circuit))

    return "\n".join(parts)


def _serialize_and_log_circuits(
    tracker: Run,
    circuits: list[Any],
    backend_name: str,
    circuit_hash: str | None,
) -> list[ProgramArtifact]:
    """
    Serialize and log circuits in multiple formats.

    Creates ProgramArtifact references for each circuit in each format,
    properly handling multi-circuit batches.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    circuits : list
        List of QuantumCircuit objects.
    backend_name : str
        Backend name for metadata.
    circuit_hash : str or None
        Circuit structure hash.

    Returns
    -------
    list of ProgramArtifact
        References to logged program artifacts, one per format per circuit.
    """
    artifacts: list[ProgramArtifact] = []
    meta = {
        "backend_name": backend_name,
        "qiskit_version": qiskit_version(),
        "circuit_hash": circuit_hash,
        "num_circuits": len(circuits),
    }

    # Log circuits in QPY format (batch, lossless)
    try:
        qpy_data = _serializer.serialize(circuits, CircuitFormat.QPY)
        ref = tracker.log_bytes(
            kind="qiskit.qpy.circuits",
            data=qpy_data.as_bytes(),
            media_type="application/vnd.qiskit.qpy",
            role="program",
            meta={**meta, "security_note": "opaque_bytes_only"},
        )
        # QPY is a batch format - single artifact for all circuits
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="qpy",
                name="circuits_batch",
                index=0,
            )
        )
    except Exception as e:
        logger.debug("Failed to serialize circuits to QPY: %s", e)

    # Log circuits in QASM3 format (per circuit, portable)
    oq3_items: list[dict[str, Any]] = []
    for i, c in enumerate(circuits):
        try:
            qasm_data = _serializer.serialize(c, CircuitFormat.OPENQASM3, index=i)
            qc_name = getattr(c, "name", None) or f"circuit_{i}"
            oq3_items.append(
                {
                    "source": qasm_data.as_text(),
                    "name": f"circuit_{i}:{qc_name}",
                    "index": i,
                }
            )
        except Exception:
            continue

    if oq3_items:
        oq3_result = tracker.log_openqasm3(oq3_items, name="circuits", meta=meta)
        # Generate ProgramArtifact per circuit, not just the first one
        items = oq3_result.get("items", [])
        for item in items:
            ref = item.get("raw_ref")
            if ref:
                item_index = item.get("index", 0)
                item_name = item.get("name", f"circuit_{item_index}")
                artifacts.append(
                    ProgramArtifact(
                        ref=ref,
                        role=ProgramRole.LOGICAL,
                        format="openqasm3",
                        name=item_name,
                        index=item_index,
                    )
                )

    # Log circuit diagrams (human-readable text)
    try:
        diagram_text = _circuits_to_text(circuits)
        ref = tracker.log_bytes(
            kind="qiskit.circuits.diagram",
            data=diagram_text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
            meta={"num_circuits": len(circuits)},
        )
        artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="diagram",
                name="circuits",
                index=0,
            )
        )
    except Exception:
        pass  # Diagram logging is best-effort

    return artifacts


def _create_program_snapshot(
    program_artifacts: list[ProgramArtifact],
    circuit_hash: str | None,
    num_circuits: int,
) -> ProgramSnapshot:
    """
    Create a ProgramSnapshot from logged artifacts.

    Parameters
    ----------
    program_artifacts : list of ProgramArtifact
        References to logged circuit artifacts.
    circuit_hash : str or None
        Circuit structure hash.
    num_circuits : int
        Number of circuits in the program.

    Returns
    -------
    ProgramSnapshot
        Program snapshot with artifact references.
    """
    return ProgramSnapshot(
        logical=program_artifacts,
        physical=[],  # Base Qiskit adapter doesn't transpile
        program_hash=circuit_hash,
        num_circuits=num_circuits,
    )


def _create_execution_snapshot(
    submitted_at: str,
    shots: int | None,
    exec_count: int,
    job_ids: list[str] | None,
    options: dict[str, Any],
) -> ExecutionSnapshot:
    """
    Create an ExecutionSnapshot.

    Parameters
    ----------
    submitted_at : str
        ISO timestamp of submission.
    shots : int or None
        Number of shots requested.
    exec_count : int
        Execution count.
    job_ids : list of str or None
        Job IDs if available.
    options : dict
        Execution options (args, kwargs).

    Returns
    -------
    ExecutionSnapshot
        Execution metadata snapshot.
    """
    return ExecutionSnapshot(
        submitted_at=submitted_at,
        shots=shots,
        execution_count=exec_count,
        job_ids=job_ids or [],
        transpilation=TranspilationInfo(
            mode=TranspilationMode.MANUAL,
            transpiled_by="user",
        ),
        options=options,
        sdk="qiskit",
    )


def _detect_physical_provider(backend: Any) -> str:
    """
    Detect physical provider from backend (not SDK).

    UEC requires provider to be the physical backend provider,
    not the SDK name. SDK goes in producer.frontends[].

    Parameters
    ----------
    backend : Any
        Qiskit backend instance.

    Returns
    -------
    str
        Physical provider: "ibm_quantum", "aer", "fake", or "local".
    """
    module_name = type(backend).__module__.lower()
    backend_name = get_backend_name(backend).lower()

    if "ibm" in module_name or "ibm_" in backend_name:
        return "ibm_quantum"
    if "qiskit_aer" in module_name or "aer" in module_name:
        return "aer"
    if "fake" in module_name:
        return "fake"
    return "local"


def _create_result_snapshot(
    tracker: Run,
    backend_name: str,
    result: Any,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot from a Qiskit Result object.

    Uses UEC 1.0 structure with items[], CountsFormat for
    cross-SDK comparability.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    backend_name : str
        Backend name.
    result : Any
        Qiskit Result object.

    Returns
    -------
    ResultSnapshot
        Structured result snapshot with items[].
    """
    # Handle None result
    if result is None:
        return ResultSnapshot(
            success=False,
            status="failed",
            items=[],
            error=ResultError(type="NullResult", message="Result is None"),
            metadata={"backend_name": backend_name},
        )

    # Serialize full result as artifact
    raw_result_ref: ArtifactRef | None = None
    try:
        if hasattr(result, "to_dict") and callable(result.to_dict):
            result_dict = result.to_dict()
        else:
            result_dict = result
        payload = to_jsonable(result_dict)
        raw_result_ref = tracker.log_json(
            name="qiskit.result",
            obj=payload,
            role="results",
            kind="result.qiskit.result_json",
        )
    except Exception as e:
        logger.debug("Failed to serialize result to dict: %s", e)

    # Extract measurement counts
    counts_data = normalize_result_counts(result)
    experiments = counts_data.get("experiments", [])

    # Log counts as separate artifact
    if experiments:
        tracker.log_json(
            name="counts",
            obj=counts_data,
            role="results",
            kind="result.counts.json",
        )

    # Qiskit counts format metadata
    # Qiskit uses little-endian (cbit[0] on right) = UEC canonical
    counts_format = CountsFormat(
        source_sdk="qiskit",
        source_key_format="qiskit_little_endian",
        bit_order="cbit0_right",  # Qiskit native = UEC canonical
        transformed=False,  # No transformation needed
    )

    # Build ResultItem list
    items: list[ResultItem] = []
    for exp in experiments:
        counts = exp.get("counts", {})
        shots = exp.get("shots")
        item_index = exp.get("index", 0)

        # Ensure counts keys are strings and values are ints
        normalized_counts = {str(k): int(v) for k, v in counts.items()}

        items.append(
            ResultItem(
                item_index=item_index,
                success=True,
                counts={
                    "counts": normalized_counts,
                    "shots": shots,
                    "format": counts_format.to_dict(),
                },
            )
        )

    # Extract metadata for status
    meta = extract_result_metadata(result)
    success = meta.get("success", True)
    status = "completed" if success else "failed"

    return ResultSnapshot(
        success=success,
        status=status,
        items=items,
        raw_result_ref=raw_result_ref,
        metadata={
            "backend_name": backend_name,
            "num_experiments": len(experiments),
            **meta,
        },
    )


def _create_failure_result_snapshot(
    exception: BaseException,
    backend_name: str,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot for a failed execution.

    Used when job.result() raises an exception. Ensures envelope
    is always created even on failures (UEC P0 requirement).

    Parameters
    ----------
    exception : BaseException
        The exception that caused the failure.
    backend_name : str
        Backend name for metadata.

    Returns
    -------
    ResultSnapshot
        Failed result snapshot with error details.
    """
    return ResultSnapshot.create_failed(
        exception=exception,
        metadata={"backend_name": backend_name},
    )


def _finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result_snapshot: ResultSnapshot,
) -> None:
    """
    Finalize envelope with result and log as artifact.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    envelope : ExecutionEnvelope
        Envelope to finalize.
    result_snapshot : ResultSnapshot
        Result to add to envelope.

    Raises
    ------
    ValueError
        If envelope is None.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    if result_snapshot is None:
        logger.warning("Finalizing envelope with None result_snapshot")

    # Add result to envelope
    envelope.result = result_snapshot

    # Set completion time
    if envelope.execution is not None:
        envelope.execution.completed_at = utc_now_iso()

    # Validate and log envelope using tracker's canonical method
    tracker.log_envelope(envelope=envelope)


def _create_minimal_device_snapshot(
    backend: Any,
    captured_at: str,
    error_msg: str | None = None,
) -> DeviceSnapshot:
    """
    Create a minimal DeviceSnapshot when full snapshot creation fails.

    Ensures envelope can always be completed even if backend introspection
    fails due to network issues or unsupported backend types.

    Parameters
    ----------
    backend : Any
        Qiskit backend (may be partially functional).
    captured_at : str
        ISO timestamp.
    error_msg : str, optional
        Error message explaining why full snapshot failed.

    Returns
    -------
    DeviceSnapshot
        Minimal snapshot with available information.
    """
    backend_name = get_backend_name(backend)
    name_lower = backend_name.lower()
    type_lower = type(backend).__name__.lower()

    # Determine backend type - default to simulator (safer fallback)
    # Schema allows: "hardware", "simulator", "emulator"
    backend_type = "simulator"
    if any(s in name_lower for s in ("ibm_", "ionq", "rigetti", "oqc")):
        backend_type = "hardware"
    elif any(s in name_lower or s in type_lower for s in ("sim", "emulator", "fake")):
        backend_type = "simulator"

    # Detect physical provider (not SDK)
    provider = _detect_physical_provider(backend)

    # Try to get num_qubits
    num_qubits = None
    try:
        num_qubits = backend.num_qubits
    except Exception:
        pass

    snapshot = DeviceSnapshot(
        captured_at=captured_at,
        backend_name=backend_name,
        backend_type=backend_type,
        provider=provider,
        num_qubits=num_qubits,
        sdk_versions={"qiskit": qiskit_version()},
    )

    if error_msg:
        logger.warning(
            "Created minimal device snapshot for %s: %s",
            backend_name,
            error_msg,
        )

    return snapshot


def _log_device_snapshot(backend: Any, tracker: Run) -> DeviceSnapshot:
    """
    Log device snapshot with fallback to minimal snapshot on failure.

    Logs both the snapshot summary and raw properties as separate artifacts
    for complete backend state capture.

    Parameters
    ----------
    backend : Any
        Qiskit backend.
    tracker : Run
        Tracker instance.

    Returns
    -------
    DeviceSnapshot
        Created device snapshot (full or minimal).
    """
    backend_name = get_backend_name(backend)
    captured_at = utc_now_iso()

    try:
        # Create snapshot with tracker for raw_properties logging
        snapshot = create_device_snapshot(
            backend,
            refresh_properties=True,
            tracker=tracker,
        )
    except Exception as e:
        # Generate minimal snapshot on failure instead of propagating
        logger.warning(
            "Full device snapshot failed for %s: %s. Using minimal snapshot.",
            backend_name,
            e,
        )
        snapshot = _create_minimal_device_snapshot(
            backend, captured_at, error_msg=str(e)
        )

    # Update tracker record with summary (for querying and fingerprinting)
    tracker.record["device_snapshot"] = {
        "sdk": "qiskit",
        "backend_name": backend_name,
        "backend_type": snapshot.backend_type,
        "provider": snapshot.provider,
        "captured_at": snapshot.captured_at,
        "num_qubits": snapshot.num_qubits,
        "calibration_summary": snapshot.get_calibration_summary(),
    }

    logger.debug("Logged device snapshot for %s", backend_name)

    return snapshot


@dataclass
class TrackedJob:
    """
    Wrapper for Qiskit job that tracks result retrieval.

    This class wraps a Qiskit job and logs artifacts when
    results are retrieved, producing a ResultSnapshot and
    finalizing the ExecutionEnvelope.

    Parameters
    ----------
    job : Any
        Original Qiskit job instance.
    tracker : Run
        Tracker instance for logging.
    backend_name : str
        Name of the backend that created this job.
    should_log_results : bool
        Whether to log results for this job.
    envelope : ExecutionEnvelope or None
        Envelope to finalize when result() is called.

    Attributes
    ----------
    job : Any
        The wrapped Qiskit job.
    tracker : Run
        The active run tracker.
    backend_name : str
        Backend name for metadata.
    result_snapshot : ResultSnapshot or None
        Captured result snapshot after result() is called.
    """

    job: Any
    tracker: Run
    backend_name: str
    should_log_results: bool = True
    envelope: ExecutionEnvelope | None = None

    # Set after result() is called
    result_snapshot: ResultSnapshot | None = field(default=None, init=False, repr=False)
    _result_logged: bool = field(default=False, init=False, repr=False)

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve job result and log artifacts.

        Always creates an envelope - even when job.result() fails.
        This is a UEC requirement (P0): envelope must exist for
        failure cases to enable debugging and telemetry.

        Idempotent: calling result() multiple times will only log once.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to job.result().
        **kwargs : Any
            Keyword arguments passed to job.result().

        Returns
        -------
        Result
            Qiskit Result object.

        Raises
        ------
        Exception
            Re-raises any exception from job.result() after logging
            the failure envelope.
        """
        # Try to get result - may raise exception
        try:
            result = self.job.result(*args, **kwargs)
        except Exception as exc:
            # P0: ALWAYS create failure envelope before re-raising
            if self.should_log_results and not self._result_logged:
                self._result_logged = True
                self._log_failure(exc)
            raise  # Re-raise original exception with original traceback

        # Happy path - log successful result
        if self.should_log_results and not self._result_logged:
            self._result_logged = True

            try:
                # Create result snapshot
                self.result_snapshot = _create_result_snapshot(
                    self.tracker,
                    self.backend_name,
                    result,
                )

                # Finalize envelope with result
                if self.envelope is not None and self.result_snapshot is not None:
                    _finalize_envelope_with_result(
                        self.tracker,
                        self.envelope,
                        self.result_snapshot,
                    )

                # Update tracker record
                if self.result_snapshot is not None:
                    self.tracker.record["results"] = {
                        "completed_at": utc_now_iso(),
                        "backend_name": self.backend_name,
                        "success": self.result_snapshot.success,
                        "status": self.result_snapshot.status,
                        "num_items": len(self.result_snapshot.items),
                        **self.result_snapshot.metadata,
                    }

                logger.debug("Logged results on %s", self.backend_name)

            except Exception as e:
                # Log error but don't fail - result retrieval should succeed
                logger.warning(
                    "Failed to log results for %s: %s",
                    self.backend_name,
                    e,
                )
                self.tracker.record.setdefault("warnings", []).append(
                    {
                        "type": "result_logging_failed",
                        "message": str(e),
                        "backend_name": self.backend_name,
                    }
                )

        return result

    def _log_failure(self, exc: Exception) -> None:
        """
        Log failure envelope when job.result() raises an exception.

        Parameters
        ----------
        exc : Exception
            The exception that was raised.
        """
        try:
            # Create failure result snapshot
            self.result_snapshot = _create_failure_result_snapshot(
                exception=exc,
                backend_name=self.backend_name,
            )

            # Finalize envelope with failure result
            if self.envelope is not None:
                _finalize_envelope_with_result(
                    self.tracker,
                    self.envelope,
                    self.result_snapshot,
                )

            # Update tracker record with failure info
            self.tracker.record["results"] = {
                "completed_at": utc_now_iso(),
                "backend_name": self.backend_name,
                "success": False,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }

            logger.debug(
                "Logged failure envelope for %s: %s",
                self.backend_name,
                type(exc).__name__,
            )

        except Exception as log_error:
            # Last resort - don't let logging failure mask original error
            logger.error(
                "Failed to log failure envelope for %s: %s",
                self.backend_name,
                log_error,
            )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped job."""
        return getattr(self.job, name)

    def __repr__(self) -> str:
        """Return string representation."""
        job_id = extract_job_id(self.job) or "unknown"
        return f"TrackedJob(backend={self.backend_name!r}, job_id={job_id!r})"


@dataclass
class TrackedBackend:
    """
    Wrapper for Qiskit backend that tracks circuit execution.

    This class wraps a Qiskit backend and logs circuits,
    execution parameters, and device snapshots when circuits
    are submitted, following the UEC with minimal overhead.

    Parameters
    ----------
    backend : Any
        Original Qiskit backend instance (must be BackendV2-compatible).
    tracker : Run
        Tracker instance for logging.
    log_every_n : int
        Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
    log_new_circuits : bool
        Auto-log new circuit structures (default True).
    stats_update_interval : int
        Update stats every N executions (default 1000).

    Attributes
    ----------
    backend : Any
        The wrapped Qiskit backend.
    tracker : Run
        The active run tracker.
    device_snapshot : DeviceSnapshot or None
        Cached device snapshot (created once per run).

    Notes
    -----
    Logging Behavior for Parameter Sweeps
    -------------------------------------
    The default settings (log_every_n=0, log_new_circuits=True) log the
    first execution and any new circuit structures. For parameter sweeps
    where the same circuit is executed with different parameter values,
    only the first execution is logged since the circuit structure hash
    ignores parameter values.

    To log all parameter sweep points, use log_every_n=-1 or set
    log_every_n to a positive value for sampling.
    """

    backend: Any
    tracker: Run
    log_every_n: int = 0
    log_new_circuits: bool = True
    stats_update_interval: int = 1000

    # Internal state (not init params)
    _snapshot_logged: bool = field(default=False, init=False, repr=False)
    _execution_count: int = field(default=0, init=False, repr=False)
    _logged_execution_count: int = field(default=0, init=False, repr=False)
    _seen_circuit_hashes: set[str] = field(default_factory=set, init=False, repr=False)
    _logged_circuit_hashes: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _program_snapshot_cache: dict[str, ProgramSnapshot] = field(
        default_factory=dict, init=False, repr=False
    )

    # Cached device snapshot
    device_snapshot: DeviceSnapshot | None = field(default=None, init=False, repr=False)

    def run(self, circuits: Any, *args: Any, **kwargs: Any) -> TrackedJob:
        """
        Execute circuits and log artifacts based on sampling settings.

        Produces ExecutionEnvelope with DeviceSnapshot, ProgramSnapshot,
        and ExecutionSnapshot following the UEC.

        Parameters
        ----------
        circuits : QuantumCircuit or iterable
            Circuit(s) to execute.
        *args : Any
            Additional positional args passed to backend.run().
        **kwargs : Any
            Additional keyword args passed to backend.run() (e.g., shots).

        Returns
        -------
        TrackedJob
            Wrapped job that tracks result retrieval.
        """
        backend_name = get_backend_name(self.backend)
        submitted_at = utc_now_iso()

        # Materialize once to avoid consuming generators during logging
        circuit_list, was_single = _materialize_circuits(circuits)

        # Payload for backend.run(): single circuit if user gave single, else list
        run_payload: Any = (
            circuit_list[0] if was_single and circuit_list else circuit_list
        )

        # Increment execution counter
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute circuit hash for structure detection
        circuit_hash = _compute_circuit_hash(circuit_list)
        is_new_circuit = circuit_hash and circuit_hash not in self._seen_circuit_hashes
        if circuit_hash:
            self._seen_circuit_hashes.add(circuit_hash)

        # Determine what to log based on settings
        should_log_structure = False
        should_log_results = False

        if self.log_every_n == -1:
            # Log all: structure if not logged, results always
            should_log_structure = circuit_hash not in self._logged_circuit_hashes
            should_log_results = True
        elif exec_count == 1:
            # First execution: log everything
            should_log_structure = True
            should_log_results = True
        elif self.log_new_circuits and is_new_circuit:
            # New circuit structure: log structure + first result
            should_log_structure = True
            should_log_results = True
        elif self.log_every_n > 0 and exec_count % self.log_every_n == 0:
            # Sampling: log results only
            should_log_results = True

        # Fast path: nothing to log
        if not should_log_structure and not should_log_results:
            job = self.backend.run(run_payload, *args, **kwargs)

            if (
                self.stats_update_interval > 0
                and exec_count % self.stats_update_interval == 0
            ):
                self._update_stats()

            return TrackedJob(
                job=job,
                tracker=self.tracker,
                backend_name=backend_name,
                should_log_results=False,
                envelope=None,
            )

        # Detect physical provider (not SDK)
        detected_provider = _detect_physical_provider(self.backend)

        # Set tags
        self.tracker.set_tag("backend_name", backend_name)
        self.tracker.set_tag("sdk", "qiskit")
        self.tracker.set_tag("adapter", "devqubit-qiskit")
        self.tracker.set_tag("provider", detected_provider)

        # Log device snapshot (once per run)
        if not self._snapshot_logged:
            self.device_snapshot = _log_device_snapshot(self.backend, self.tracker)
            self._snapshot_logged = True

        # Build program snapshot
        program_snapshot: ProgramSnapshot | None = None
        if should_log_structure and circuit_list:
            # Log execution parameters
            shots = kwargs.get("shots")
            if shots is not None:
                self.tracker.log_param("shots", int(shots))
            self.tracker.log_param("num_circuits", int(len(circuit_list)))

            # Check for parameter_binds (parameter sweep indicator)
            if kwargs.get("parameter_binds"):
                self.tracker.log_param(
                    "parameter_binds_count",
                    len(kwargs["parameter_binds"]),
                )

            # Log circuits
            program_artifacts = _serialize_and_log_circuits(
                self.tracker,
                circuit_list,
                backend_name,
                circuit_hash,
            )

            if circuit_hash:
                self._logged_circuit_hashes.add(circuit_hash)

            # Create program snapshot
            program_snapshot = _create_program_snapshot(
                program_artifacts,
                circuit_hash,
                len(circuit_list),
            )

            # Cache program snapshot for reuse in results-only logging
            if circuit_hash:
                self._program_snapshot_cache[circuit_hash] = program_snapshot

            # Update tracker record (used by fingerprint computation)
            self.tracker.record["backend"] = {
                "name": backend_name,
                "type": self.backend.__class__.__name__,
                "provider": detected_provider,
                "sdk": "qiskit",
            }

            self._logged_execution_count += 1

        # Reuse cached program snapshot when only logging results
        elif should_log_results and circuit_hash in self._program_snapshot_cache:
            program_snapshot = self._program_snapshot_cache[circuit_hash]

        # Build ExecutionSnapshot
        shots = kwargs.get("shots")
        execution_snapshot = _create_execution_snapshot(
            submitted_at=submitted_at,
            shots=int(shots) if shots is not None else None,
            exec_count=exec_count,
            job_ids=[],  # Will be updated after job creation
            options={
                "args": to_jsonable(list(args)),
                "kwargs": to_jsonable(kwargs),
            },
        )

        # Update tracker record (used by fingerprint computation)
        self.tracker.record["execute"] = {
            "sdk": "qiskit",
            "submitted_at": submitted_at,
            "backend_name": backend_name,
            "num_circuits": len(circuit_list),
            "execution_count": exec_count,
            "program_hash": circuit_hash,
            "args": to_jsonable(list(args)),
            "kwargs": to_jsonable(kwargs),
        }

        logger.debug(
            "Submitting %d circuits to %s",
            len(circuit_list),
            backend_name,
        )

        # Execute on actual backend
        job = self.backend.run(run_payload, *args, **kwargs)

        # Log job ID if available
        job_id = extract_job_id(job)
        if job_id:
            self.tracker.record["execute"]["job_ids"] = [job_id]
            execution_snapshot.job_ids = [job_id]
            logger.debug("Job ID: %s", job_id)

        # Create envelope (will be finalized when result() is called)
        envelope: ExecutionEnvelope | None = None
        if should_log_results and self.device_snapshot is not None:
            # Use existing or create minimal program snapshot
            if program_snapshot is None:
                program_snapshot = ProgramSnapshot(
                    logical=[],
                    physical=[],
                    program_hash=circuit_hash,
                    num_circuits=len(circuit_list),
                )

            # Create ProducerInfo for SDK stack tracking
            producer = ProducerInfo.create(
                adapter="devqubit-qiskit",
                adapter_version=_ADAPTER_VERSION,
                sdk="qiskit",
                sdk_version=qiskit_version(),
                frontends=["qiskit"],
            )

            # Create envelope with pending result (will be updated in result())
            # Using ResultSnapshot with status="pending" until result() is called
            pending_result = ResultSnapshot(
                success=False,
                status="failed",  # Will be updated when result() completes
                items=[],
                metadata={"state": "pending"},
            )

            envelope = ExecutionEnvelope(
                envelope_id=f"{utc_now_iso().replace(':', '').replace('-', '')[:20]}",
                created_at=utc_now_iso(),
                producer=producer,
                result=pending_result,
                device=self.device_snapshot,
                program=program_snapshot,
                execution=execution_snapshot,
            )

        # Update stats
        self._update_stats()

        return TrackedJob(
            job=job,
            tracker=self.tracker,
            backend_name=backend_name,
            should_log_results=should_log_results,
            envelope=envelope,
        )

    def _update_stats(self) -> None:
        """Update execution statistics in tracker record."""
        self.tracker.record["execution_stats"] = {
            "total_executions": self._execution_count,
            "logged_executions": self._logged_execution_count,
            "unique_circuits": len(self._seen_circuit_hashes),
            "logged_circuits": len(self._logged_circuit_hashes),
            "last_execution_at": utc_now_iso(),
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped backend."""
        return getattr(self.backend, name)

    def __repr__(self) -> str:
        """Return string representation."""
        backend_name = get_backend_name(self.backend)
        run_id = getattr(self.tracker, "run_id", "unknown")
        return f"TrackedBackend(backend={backend_name!r}, run_id={run_id!r})"


class QiskitAdapter:
    """
    Adapter for integrating Qiskit backends with devqubit tracking.

    This adapter wraps Qiskit backends to automatically log circuits,
    execution parameters, device configurations, and results following
    the devqubit Uniform Execution Contract (UEC).

    Attributes
    ----------
    name : str
        Adapter identifier ("qiskit").

    Notes
    -----
    This adapter only supports BackendV2-based backends. Legacy BackendV1
    backends are not supported and will return False from ``supports_executor()``.

    For Runtime primitives (SamplerV2, EstimatorV2), use the ``qiskit-runtime``
    adapter instead.

    Example
    -------
    >>> from qiskit_aer import AerSimulator
    >>> adapter = QiskitAdapter()
    >>> assert adapter.supports_executor(AerSimulator())
    >>> desc = adapter.describe_executor(AerSimulator())
    >>> print(desc["name"])
    'aer_simulator'
    """

    name: str = "qiskit"

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if executor is a supported Qiskit backend.

        Parameters
        ----------
        executor : Any
            Potential executor instance.

        Returns
        -------
        bool
            True if executor is a Qiskit BackendV2.
        """
        return isinstance(executor, BackendV2)

    def describe_executor(self, executor: Any) -> dict[str, Any]:
        """
        Create a description of the backend.

        Parameters
        ----------
        executor : Any
            Qiskit backend instance.

        Returns
        -------
        dict
            Backend description with keys: name, type, provider, sdk.
        """
        return {
            "name": get_backend_name(executor),
            "type": executor.__class__.__name__,
            "provider": _detect_physical_provider(executor),
            "sdk": "qiskit",
        }

    def wrap_executor(
        self,
        executor: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> TrackedBackend:
        """
        Wrap a backend with tracking capabilities.

        Parameters
        ----------
        executor : Any
            Qiskit backend to wrap.
        tracker : Run
            Tracker instance for logging.
        log_every_n : int
            Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
        log_new_circuits : bool
            Auto-log new circuit structures (default True).
        stats_update_interval : int
            Update stats every N executions (default 1000).

        Returns
        -------
        TrackedBackend
            Wrapped backend that logs execution artifacts.

        Notes
        -----
        For parameter sweeps (same circuit with different parameter values),
        the default settings will only log the first execution since circuit
        structure hashing ignores parameter values. Use ``log_every_n=-1``
        to log all executions, or ``log_every_n=N`` for sampling.
        """
        return TrackedBackend(
            backend=executor,
            tracker=tracker,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
