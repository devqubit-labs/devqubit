# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Braket adapter for devqubit tracking system.

Provides integration with Amazon Braket devices, enabling automatic tracking
of quantum circuit execution, results, and device configurations using the
Uniform Execution Contract (UEC).

Example
-------
>>> from braket.circuits import Circuit
>>> from braket.devices import LocalSimulator
>>> from devqubit_engine.core.tracker import track
>>>
>>> circuit = Circuit().h(0).cnot(0, 1)
>>>
>>> with track(project="my_experiment") as run:
...     device = run.wrap(LocalSimulator())
...     task = device.run(circuit, shots=1000)
...     result = task.result()
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from devqubit_braket.results import extract_counts_payload
from devqubit_braket.serialization import (
    BraketCircuitSerializer,
    circuits_to_text,
    is_braket_circuit,
    serialize_openqasm,
)
from devqubit_braket.snapshot import create_device_snapshot
from devqubit_braket.utils import braket_version, extract_task_id, get_backend_name
from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.core.run import Run
from devqubit_engine.core.snapshot import (
    DeviceSnapshot,
    ExecutionEnvelope,
    ExecutionSnapshot,
    NormalizedCounts,
    ProgramArtifact,
    ProgramSnapshot,
    ResultSnapshot,
    TranspilationInfo,
)
from devqubit_engine.core.types import (
    ArtifactRef,
    ProgramRole,
    ResultType,
    TranspilationMode,
)
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_engine.utils.time_utils import utc_now_iso


logger = logging.getLogger(__name__)

# Module-level serializer instance
_serializer = BraketCircuitSerializer()


def _is_program_set(obj: Any) -> bool:
    """
    Check if object is a Braket ProgramSet.

    ProgramSet is a composite task specification that contains multiple
    programs/circuits to be executed together.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if object appears to be a ProgramSet.
    """
    if obj is None:
        return False

    # Check for ProgramSet-specific attributes
    has_entries = hasattr(obj, "entries")
    has_to_ir = hasattr(obj, "to_ir")
    has_total_executables = hasattr(obj, "total_executables")

    # Must have entries and at least one other characteristic
    if has_entries and (has_to_ir or has_total_executables):
        return True

    # Check type name as fallback
    return "programset" in type(obj).__name__.lower()


def _extract_circuits_from_program_set(program_set: Any) -> list[Any]:
    """
    Extract individual circuits from a ProgramSet for logging purposes.

    Parameters
    ----------
    program_set : Any
        Braket ProgramSet instance.

    Returns
    -------
    list
        List of extracted circuit objects for logging.
    """
    circuits: list[Any] = []
    try:
        entries = getattr(program_set, "entries", None)
        if entries is None:
            return circuits

        for entry in entries:
            # Each entry may have a circuit/program attribute
            for attr in ("circuit", "program", "task_specification"):
                circ = getattr(entry, attr, None)
                if circ is not None and is_braket_circuit(circ):
                    circuits.append(circ)
                    break
            else:
                # Entry itself might be a circuit
                if is_braket_circuit(entry):
                    circuits.append(entry)
    except Exception as e:
        logger.debug("Failed to extract circuits from ProgramSet: %s", e)

    return circuits


def _get_program_set_metadata(program_set: Any) -> dict[str, Any]:
    """
    Extract metadata from a ProgramSet for logging.

    Parameters
    ----------
    program_set : Any
        Braket ProgramSet instance.

    Returns
    -------
    dict
        Metadata dict with ProgramSet-specific fields.
    """
    meta: dict[str, Any] = {"is_program_set": True}

    for attr in ("total_executables", "shots_per_executable", "total_shots"):
        try:
            val = getattr(program_set, attr, None)
            if val is not None:
                meta[attr] = int(val)
        except Exception:
            pass

    return meta


def _materialize_task_spec(
    task_specification: Any,
) -> tuple[Any, list[Any], bool, dict[str, Any] | None]:
    """
    Materialize task specification into run payload and circuits for logging.

    Separates what to send to Braket (run_payload) from what to log
    (circuits_for_logging), handling ProgramSet and other composite types.

    Parameters
    ----------
    task_specification : Any
        A Circuit, ProgramSet, list of circuits, or other task spec.

    Returns
    -------
    run_payload : Any
        What to actually send to device.run().
    circuits_for_logging : list
        List of circuit objects for artifact logging and hashing.
    was_single : bool
        True if input was a single circuit.
    extra_meta : dict or None
        Additional metadata (e.g., ProgramSet fields).
    """
    if task_specification is None:
        return None, [], False, None

    # Handle ProgramSet: send as-is, but extract circuits for logging
    if _is_program_set(task_specification):
        circuits = _extract_circuits_from_program_set(task_specification)
        meta = _get_program_set_metadata(task_specification)
        return task_specification, circuits, False, meta

    # Single circuit
    if is_braket_circuit(task_specification):
        return task_specification, [task_specification], True, None

    # List/tuple of circuits
    if isinstance(task_specification, (list, tuple)):
        circuit_list = list(task_specification)
        return circuit_list, circuit_list, False, None

    # Unknown iterable - try to materialize
    try:
        circuit_list = list(task_specification)
        return circuit_list, circuit_list, False, None
    except TypeError:
        # Not iterable, treat as single item
        return task_specification, [task_specification], True, None


def _compute_circuit_hash(circuits: list[Any]) -> str | None:
    """
    Compute a content hash for circuits.

    Parameters
    ----------
    circuits : list[Any]
        List of Braket Circuit objects.

    Returns
    -------
    str | None
        SHA256 hash with prefix, or None if circuits is empty.
    """
    if not circuits:
        return None

    circuit_signatures: list[str] = []

    for circuit in circuits:
        try:
            instrs = getattr(circuit, "instructions", None)
            if instrs is None:
                circuit_signatures.append(str(circuit)[:500])
                continue

            op_sigs: list[str] = []
            for instr in instrs:
                op = getattr(instr, "operator", None)
                # Gate name
                if op is not None:
                    op_name = getattr(op, "name", None)
                    op_name = (
                        op_name
                        if isinstance(op_name, str) and op_name
                        else type(op).__name__
                    )
                else:
                    op_name = type(instr).__name__

                # Parameter arity
                arity = 0
                if op is not None:
                    for attr in ("parameters", "params", "angles", "probabilities"):
                        val = getattr(op, attr, None)
                        if isinstance(val, (list, tuple)):
                            arity = len(val)
                            break

                # Target qubits
                tgt = getattr(instr, "target", None)
                if tgt is not None:
                    try:
                        targets = tuple(
                            str(getattr(q, "index", None) or q) for q in tgt
                        )
                    except Exception:
                        targets = (str(tgt),)
                else:
                    targets = ()

                op_sigs.append(f"{op_name}|p{arity}|t{targets}")

            circuit_signatures.append("||".join(op_sigs))

        except Exception:
            circuit_signatures.append(str(circuit)[:500])

    payload = "\n".join(circuit_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _serialize_and_log_circuits(
    tracker: Run,
    circuits: list[Any],
    device_name: str,
) -> list[ArtifactRef]:
    """
    Serialize circuits and log as artifacts.

    Logs both JAQCD and OpenQASM formats for comprehensive coverage.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    circuits : list
        List of Braket circuits.
    device_name : str
        Backend name for metadata.

    Returns
    -------
    list of ArtifactRef
        References to logged circuit artifacts.
    """
    artifact_refs: list[ArtifactRef] = []
    meta = {
        "backend_name": device_name,
        "braket_version": braket_version(),
    }

    for i, circuit in enumerate(circuits):
        # Serialize JAQCD (native format)
        try:
            jaqcd_data = _serializer.serialize(circuit, CircuitFormat.JAQCD, index=i)
            ref = tracker.log_bytes(
                kind="braket.ir.jaqcd",
                data=jaqcd_data.as_bytes(),
                media_type="application/json",
                role="program",
                meta={**meta, "index": i},
            )
            if ref:
                artifact_refs.append(ref)
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to JAQCD: %s", i, e)

        # Serialize OpenQASM (canonical format, better for diffing)
        try:
            qasm_data = serialize_openqasm(circuit, index=i)
            tracker.log_bytes(
                kind="braket.ir.openqasm",
                data=qasm_data.as_bytes(),
                media_type="text/x-qasm; charset=utf-8",
                role="program",
                meta={**meta, "index": i, "format": "openqasm3"},
            )
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to OpenQASM: %s", i, e)

    # Log circuit diagrams (human-readable)
    try:
        diagram_text = circuits_to_text(circuits)
        tracker.log_bytes(
            kind="braket.circuits.diagram",
            data=diagram_text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            role="program",
            meta={"num_circuits": len(circuits)},
        )
    except Exception as e:
        logger.debug("Failed to generate circuit diagrams: %s", e)

    return artifact_refs


def _create_program_snapshot(
    circuits: list[Any],
    artifact_refs: list[ArtifactRef],
    circuit_hash: str | None,
) -> ProgramSnapshot:
    """
    Create a ProgramSnapshot from circuits and their artifact refs.

    Parameters
    ----------
    circuits : list
        List of Braket circuits.
    artifact_refs : list of ArtifactRef
        References to logged circuit artifacts.
    circuit_hash : str or None
        Circuit structure hash.

    Returns
    -------
    ProgramSnapshot
        Program snapshot with logical artifacts.
    """
    logical_artifacts: list[ProgramArtifact] = []

    for i, ref in enumerate(artifact_refs):
        circuit_name = None
        if i < len(circuits):
            circuit_name = getattr(circuits[i], "name", None)

        logical_artifacts.append(
            ProgramArtifact(
                ref=ref,
                role=ProgramRole.LOGICAL,
                format="jaqcd",
                name=circuit_name or f"circuit_{i}",
                index=i,
            )
        )

    return ProgramSnapshot(
        logical=logical_artifacts,
        physical=[],  # Braket doesn't expose transpiled circuits
        program_hash=circuit_hash,
        num_circuits=len(circuits),
    )


def _create_execution_snapshot(
    shots: int | None,
    task_ids: list[str],
    submitted_at: str,
    options: dict[str, Any] | None = None,
) -> ExecutionSnapshot:
    """
    Create an ExecutionSnapshot for a Braket task submission.

    Parameters
    ----------
    shots : int or None
        Number of shots (None means provider default).
    task_ids : list of str
        Task identifiers.
    submitted_at : str
        ISO 8601 submission timestamp.
    options : dict, optional
        Additional execution options.

    Returns
    -------
    ExecutionSnapshot
        Execution metadata snapshot.
    """
    return ExecutionSnapshot(
        submitted_at=submitted_at,
        shots=shots,
        task_ids=task_ids,
        execution_count=len(task_ids) if task_ids else 1,
        transpilation=TranspilationInfo(
            mode=TranspilationMode.MANAGED,
            transpiled_by="provider",
        ),
        options=options or {},
        sdk="braket",
    )


def _create_result_snapshot(
    result: Any,
    raw_result_ref: ArtifactRef | None,
    shots: int | None,
    error_message: str | None = None,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot from Braket result.

    Parameters
    ----------
    result : Any
        Braket result object (may be None on failure).
    raw_result_ref : ArtifactRef or None
        Reference to raw result artifact.
    shots : int or None
        Number of shots used.
    error_message : str or None
        Error message if execution failed.

    Returns
    -------
    ResultSnapshot
        Result snapshot with normalized counts and success status.
    """
    normalized_counts: list[NormalizedCounts] = []
    success = False
    result_type = ResultType.COUNTS

    if result is not None and error_message is None:
        # Check if result is already a combined payload dict (from batch)
        if isinstance(result, dict) and "experiments" in result:
            counts_payload = result
        else:
            counts_payload = extract_counts_payload(result)

        if counts_payload and counts_payload.get("experiments"):
            for exp in counts_payload["experiments"]:
                counts = exp.get("counts", {})
                normalized_counts.append(
                    NormalizedCounts(
                        circuit_index=exp.get("index", 0),
                        counts=counts if counts else {},
                        shots=shots,
                        name=exp.get("name"),
                    )
                )

        # Fallback: if we have a result but no experiments extracted
        if not normalized_counts:
            batch_size = result.get("batch_size", 1) if isinstance(result, dict) else 1
            for i in range(batch_size):
                normalized_counts.append(
                    NormalizedCounts(circuit_index=i, counts={}, shots=shots)
                )

        # Success = we have actual non-empty counts
        success = any(nc.counts for nc in normalized_counts)

        # For shots=0 (analytical), may get statevector/other instead of counts
        if not success and shots == 0:
            if hasattr(result, "values") or hasattr(result, "result_types"):
                result_type = ResultType.OTHER
                success = True

    # Build ResultSnapshot - handle error_message field defensively
    snapshot_kwargs: dict[str, Any] = {
        "result_type": result_type,
        "raw_result_ref": raw_result_ref,
        "counts": normalized_counts,
        "num_experiments": len(normalized_counts),
        "success": success,
    }

    # Only add error_message if the field exists in ResultSnapshot
    try:
        sig = inspect.signature(ResultSnapshot)
        if "error_message" in sig.parameters:
            snapshot_kwargs["error_message"] = error_message
    except Exception:
        pass

    return ResultSnapshot(**snapshot_kwargs)


def _create_envelope(
    tracker: Run,
    device: Any,
    circuits: list[Any],
    shots: int | None,
    task_ids: list[str],
    submitted_at: str,
    circuit_hash: str | None,
    options: dict[str, Any] | None = None,
) -> ExecutionEnvelope:
    """
    Create and log a complete ExecutionEnvelope (pre-result).

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    device : Any
        Braket device.
    circuits : list
        List of circuits.
    shots : int or None
        Number of shots.
    task_ids : list of str
        Task identifiers.
    submitted_at : str
        Submission timestamp.
    circuit_hash : str or None
        Circuit hash.
    options : dict, optional
        Execution options.

    Returns
    -------
    ExecutionEnvelope
        Envelope with device, program, and execution snapshots.
    """
    device_name = get_backend_name(device=device)

    # Create device snapshot with tracker for raw_properties logging
    try:
        device_snapshot = create_device_snapshot(device=device, tracker=tracker)
    except Exception as e:
        logger.warning(
            "Failed to create device snapshot: %s. Using minimal snapshot.", e
        )
        # Create minimal snapshot on failure
        device_snapshot = DeviceSnapshot(
            captured_at=utc_now_iso(),
            backend_name=device_name,
            backend_type="unknown",
            provider="braket",
            sdk_versions={"braket": braket_version()},
        )

    # Update tracker record
    tracker.record["device_snapshot"] = {
        "sdk": "braket",
        "backend_name": device_name,
        "backend_type": device_snapshot.backend_type,
        "provider": device_snapshot.provider,
        "captured_at": device_snapshot.captured_at,
        "num_qubits": device_snapshot.num_qubits,
        "calibration_summary": device_snapshot.get_calibration_summary(),
    }

    # Log circuits and get artifact refs
    artifact_refs = _serialize_and_log_circuits(
        tracker=tracker,
        circuits=circuits,
        device_name=device_name,
    )

    # Create program snapshot
    program_snapshot = _create_program_snapshot(
        circuits=circuits,
        artifact_refs=artifact_refs,
        circuit_hash=circuit_hash,
    )

    # Create execution snapshot
    execution_snapshot = _create_execution_snapshot(
        shots=shots,
        task_ids=task_ids,
        submitted_at=submitted_at,
        options=options,
    )

    return ExecutionEnvelope(
        schema_version="devqubit.envelope/0.1",
        adapter="braket",
        created_at=utc_now_iso(),
        device=device_snapshot,
        program=program_snapshot,
        execution=execution_snapshot,
        result=None,  # Will be filled when result() is called
    )


def _finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result: Any,
    device_name: str,
    shots: int | None,
    error_message: str | None = None,
) -> ExecutionEnvelope:
    """
    Finalize envelope with result and log it.

    This function never raises exceptions - tracking should never crash
    user experiments. Validation errors are logged but execution continues.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    envelope : ExecutionEnvelope
        Envelope to finalize.
    result : Any
        Braket result object (may be None on failure).
    device_name : str
        Device name.
    shots : int or None
        Number of shots.
    error_message : str or None
        Error message if execution failed.

    Returns
    -------
    ExecutionEnvelope
        Finalized envelope.

    Raises
    ------
    ValueError
        If envelope is None.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    # Log raw result and get ref
    raw_result_ref: ArtifactRef | None = None
    if result is not None:
        try:
            result_payload = to_jsonable(result)
        except Exception:
            result_payload = {"repr": repr(result)[:2000]}

        try:
            raw_result_ref = tracker.log_json(
                name="braket.result",
                obj=result_payload,
                role="results",
                kind="result.braket.raw.json",
            )
        except Exception as e:
            logger.warning("Failed to log raw result: %s", e)
    elif error_message:
        try:
            tracker.log_json(
                name="braket.error",
                obj={"error": error_message, "timestamp": utc_now_iso()},
                role="results",
                kind="result.braket.error.json",
            )
        except Exception as e:
            logger.warning("Failed to log error: %s", e)

    # Create result snapshot
    result_snapshot = _create_result_snapshot(
        result, raw_result_ref, shots, error_message
    )

    # Update execution snapshot with completion time
    if envelope.execution:
        envelope.execution.completed_at = utc_now_iso()

    # Add result to envelope
    envelope.result = result_snapshot

    # Extract counts for separate logging
    counts_payload = None
    if result is not None:
        try:
            counts_payload = extract_counts_payload(result)
        except Exception as e:
            logger.debug("Failed to extract counts payload: %s", e)

    # Validate and log envelope
    try:
        tracker.log_envelope(envelope=envelope)
    except Exception as e:
        logger.warning("Failed to log envelope: %s", e)

    # Log normalized counts
    if counts_payload is not None:
        try:
            tracker.log_json(
                name="counts",
                obj=counts_payload,
                role="results",
                kind="result.counts.json",
            )
        except Exception as e:
            logger.debug("Failed to log counts: %s", e)

    # Update tracker record
    tracker.record["results"] = {
        "completed_at": utc_now_iso(),
        "backend_name": device_name,
        "num_experiments": result_snapshot.num_experiments,
        "result_type": result_snapshot.result_type.value,
        "success": result_snapshot.success,
    }
    if error_message:
        tracker.record["results"]["error"] = error_message

    logger.debug("Logged execution envelope for %s", device_name)

    return envelope


def _log_submission_failure(
    tracker: Run,
    device_name: str,
    error: Exception,
    circuits: list[Any],
    shots: int | None,
    submitted_at: str,
) -> None:
    """
    Log a task submission failure.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    device_name : str
        Device name.
    error : Exception
        The exception that occurred.
    circuits : list
        Circuits that were being submitted.
    shots : int or None
        Requested shots.
    submitted_at : str
        Submission timestamp.
    """
    error_info = {
        "type": "submission_failure",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "device_name": device_name,
        "num_circuits": len(circuits),
        "shots": shots,
        "submitted_at": submitted_at,
        "failed_at": utc_now_iso(),
    }

    tracker.log_json(
        name="submission_failure",
        obj=error_info,
        role="error",
        kind="devqubit.submission_failure.json",
    )

    tracker.record["submission_failure"] = error_info
    logger.warning("Task submission failed on %s: %s", device_name, error)


def _combine_batch_results(results_list: list[Any]) -> dict[str, Any]:
    """
    Combine batch results into a single structure for logging.

    Parameters
    ----------
    results_list : list
        List of individual result objects.

    Returns
    -------
    dict
        Combined result structure.
    """
    experiments: list[dict[str, Any]] = []

    for i, result in enumerate(results_list):
        if result is None:
            experiments.append({"index": i, "status": "failed", "counts": {}})
            continue

        counts_payload = extract_counts_payload(result)
        if counts_payload and counts_payload.get("experiments"):
            for exp in counts_payload["experiments"]:
                exp_copy = dict(exp)
                exp_copy["batch_index"] = i
                experiments.append(exp_copy)
        else:
            experiments.append({"index": i, "batch_index": i, "counts": {}})

    return {"experiments": experiments, "batch_size": len(results_list)}


@dataclass
class TrackedTask:
    """
    Wrapper for Braket task that tracks result retrieval.

    Intercepts `result()` calls to finalize the execution envelope
    with result data. Handles exceptions gracefully with failure logging.

    Parameters
    ----------
    task : Any
        Original Braket task instance.
    tracker : Run
        Tracker instance for logging.
    device_name : str
        Name of the device that created this task.
    envelope : ExecutionEnvelope or None
        Envelope to finalize with results.
    shots : int or None
        Number of shots for this execution.
    should_log_results : bool
        Whether to log results for this task.
    """

    task: Any
    tracker: Run
    device_name: str
    envelope: ExecutionEnvelope | None = None
    shots: int | None = None
    should_log_results: bool = True
    _result_logged: bool = field(default=False, init=False, repr=False)

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve task result and finalize envelope.

        Handles exceptions gracefully, logging failure information
        before re-raising.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to underlying result().
        **kwargs : Any
            Keyword arguments passed to underlying result().

        Returns
        -------
        Any
            Braket result object.

        Raises
        ------
        Exception
            Re-raises any exception from underlying result() after logging.
        """
        result = None
        error_message: str | None = None

        try:
            result = self.task.result(*args, **kwargs)
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.warning("Task result() failed on %s: %s", self.device_name, e)

            # Log failure even if we re-raise
            if self.should_log_results and self.envelope and not self._result_logged:
                self._result_logged = True
                try:
                    _finalize_envelope_with_result(
                        tracker=self.tracker,
                        envelope=self.envelope,
                        result=None,
                        device_name=self.device_name,
                        shots=self.shots,
                        error_message=error_message,
                    )
                except Exception as log_err:
                    logger.warning(
                        "Failed to log error envelope for %s: %s",
                        self.device_name,
                        log_err,
                    )
            raise

        # Log successful result
        if self.should_log_results and self.envelope and not self._result_logged:
            self._result_logged = True
            try:
                _finalize_envelope_with_result(
                    tracker=self.tracker,
                    envelope=self.envelope,
                    result=result,
                    device_name=self.device_name,
                    shots=self.shots,
                )
                logger.debug("Finalized envelope for task on %s", self.device_name)
            except Exception as log_err:
                logger.warning(
                    "Failed to finalize envelope for %s: %s",
                    self.device_name,
                    log_err,
                )
                # Record error in tracker for visibility
                self.tracker.record.setdefault("warnings", []).append(
                    {
                        "type": "result_logging_failed",
                        "message": str(log_err),
                        "device_name": self.device_name,
                    }
                )

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped task."""
        return getattr(self.task, name)

    def __repr__(self) -> str:
        """Return string representation."""
        task_id = extract_task_id(self.task) or "unknown"
        return f"TrackedTask(device={self.device_name!r}, task_id={task_id!r})"


@dataclass
class TrackedTaskBatch:
    """
    Wrapper for Braket task batch that tracks result retrieval.

    Wraps AwsQuantumTaskBatch to intercept `results()` calls and log
    all results with proper handling of partial failures.

    Parameters
    ----------
    batch : Any
        Original Braket task batch instance.
    tracker : Run
        Tracker instance for logging.
    device_name : str
        Name of the device that created this batch.
    envelope : ExecutionEnvelope or None
        Envelope to finalize with results.
    shots : int or None
        Number of shots for this execution.
    should_log_results : bool
        Whether to log results for this batch.
    """

    batch: Any
    tracker: Run
    device_name: str
    envelope: ExecutionEnvelope | None = None
    shots: int | None = None
    should_log_results: bool = True
    _results_logged: bool = field(default=False, init=False, repr=False)

    def results(self, *args: Any, **kwargs: Any) -> list[Any]:
        """
        Retrieve batch results and finalize envelope.

        Handles partial failures (None results for failed tasks).

        Parameters
        ----------
        *args : Any
            Positional arguments passed to underlying results().
        **kwargs : Any
            Keyword arguments passed to underlying results().

        Returns
        -------
        list
            List of Braket result objects (may contain None for failed tasks).
        """
        results_list: list[Any] = []
        error_message: str | None = None

        try:
            results_list = self.batch.results(*args, **kwargs)
        except Exception as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.warning("Batch results() failed on %s: %s", self.device_name, e)

            if self.should_log_results and self.envelope and not self._results_logged:
                self._results_logged = True
                try:
                    _finalize_envelope_with_result(
                        self.tracker,
                        self.envelope,
                        None,
                        self.device_name,
                        self.shots,
                        error_message=error_message,
                    )
                except Exception as log_err:
                    logger.warning(
                        "Failed to log error envelope for batch %s: %s",
                        self.device_name,
                        log_err,
                    )
            raise

        # Check for partial failures (None in results)
        failed_count = sum(1 for r in results_list if r is None)
        if failed_count > 0:
            logger.warning(
                "Batch on %s: %d/%d tasks failed",
                self.device_name,
                failed_count,
                len(results_list),
            )
            error_message = (
                f"Partial failure: {failed_count}/{len(results_list)} tasks failed"
            )

        # Aggregate successful results for logging
        if self.should_log_results and self.envelope and not self._results_logged:
            self._results_logged = True
            try:
                combined_result = _combine_batch_results(results_list)

                _finalize_envelope_with_result(
                    self.tracker,
                    self.envelope,
                    combined_result,
                    self.device_name,
                    self.shots,
                    error_message=error_message if failed_count > 0 else None,
                )
                logger.debug("Finalized envelope for batch on %s", self.device_name)
            except Exception as log_err:
                logger.warning(
                    "Failed to finalize envelope for batch %s: %s",
                    self.device_name,
                    log_err,
                )
                # Record error in tracker for visibility
                self.tracker.record.setdefault("warnings", []).append(
                    {
                        "type": "batch_result_logging_failed",
                        "message": str(log_err),
                        "device_name": self.device_name,
                    }
                )

        return results_list

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped batch."""
        return getattr(self.batch, name)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TrackedTaskBatch(device={self.device_name!r})"


@dataclass
class TrackedDevice:
    """
    Wrapper for Braket device that tracks circuit execution.

    Intercepts `run()` and `run_batch()` calls to automatically create
    execution envelopes with device, program, and execution snapshots.

    Parameters
    ----------
    device : Any
        Original Braket device instance.
    tracker : Run
        Tracker instance for logging artifacts.
    log_every_n : int
        Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
    log_new_circuits : bool
        Auto-log new circuit structures (default True).
    stats_update_interval : int
        Update stats every N executions (default 1000).
    """

    device: Any
    tracker: Run
    log_every_n: int = 0
    log_new_circuits: bool = True
    stats_update_interval: int = 1000

    # Internal state (explicitly typed)
    _snapshot_logged: bool = field(default=False, init=False, repr=False)
    _execution_count: int = field(default=0, init=False, repr=False)
    _logged_execution_count: int = field(default=0, init=False, repr=False)
    _seen_circuit_hashes: set[str] = field(default_factory=set, init=False, repr=False)
    _logged_circuit_hashes: set[str] = field(
        default_factory=set, init=False, repr=False
    )

    def run(
        self,
        task_specification: Any,
        shots: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> TrackedTask:
        """
        Execute circuit and create execution envelope.

        Parameters
        ----------
        task_specification : Circuit, ProgramSet, or Program
            Circuit or program to execute.
        shots : int or None, optional
            Number of shots. None lets Braket use its default (1000 for QPU).
        *args : Any
            Additional positional arguments passed to device.
        **kwargs : Any
            Additional keyword arguments passed to device.

        Returns
        -------
        TrackedTask
            Wrapped task that tracks result retrieval.
        """
        device_name = get_backend_name(self.device)
        submitted_at = utc_now_iso()

        # Separate run payload from circuits for logging
        run_payload, circuits_for_logging, was_single, extra_meta = (
            _materialize_task_spec(task_specification)
        )

        # For single circuit wrapped in list, unwrap for Braket
        if was_single and isinstance(run_payload, list) and len(run_payload) == 1:
            run_payload = run_payload[0]

        # Increment execution counter
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute circuit hash
        circuit_hash = _compute_circuit_hash(circuits_for_logging)
        is_new_circuit = circuit_hash and circuit_hash not in self._seen_circuit_hashes
        if circuit_hash:
            self._seen_circuit_hashes.add(circuit_hash)

        # Determine logging behavior
        should_log = self._should_log(exec_count, circuit_hash, is_new_circuit)

        # Build execution options
        options: dict[str, Any] = {}
        if args:
            options["args"] = to_jsonable(list(args))
        if kwargs:
            options["kwargs"] = to_jsonable(kwargs)
        if extra_meta:
            options.update(extra_meta)

        # Execute on actual device
        task: Any = None
        try:
            if shots is None:
                task = self.device.run(run_payload, *args, **kwargs)
            else:
                task = self.device.run(run_payload, shots=shots, *args, **kwargs)
        except Exception as e:
            if should_log and circuits_for_logging:
                _log_submission_failure(
                    self.tracker,
                    device_name,
                    e,
                    circuits_for_logging,
                    shots,
                    submitted_at,
                )
            raise

        # Extract task ID
        task_id = extract_task_id(task)
        task_ids = [task_id] if task_id else []

        # Create envelope if logging
        envelope: ExecutionEnvelope | None = None
        if should_log and circuits_for_logging:
            envelope = _create_envelope(
                tracker=self.tracker,
                device=self.device,
                circuits=circuits_for_logging,
                shots=shots,
                task_ids=task_ids,
                submitted_at=submitted_at,
                circuit_hash=circuit_hash,
                options=options if options else None,
            )

            if circuit_hash:
                self._logged_circuit_hashes.add(circuit_hash)

            self._logged_execution_count += 1

            # Set tracker tags/params
            self.tracker.set_tag("backend_name", device_name)
            self.tracker.set_tag("provider", "braket")
            self.tracker.set_tag("adapter", "braket")

            if shots is not None:
                self.tracker.log_param("shots", int(shots))
            self.tracker.log_param("num_circuits", len(circuits_for_logging))

            # Update tracker record
            self.tracker.record["backend"] = {
                "name": device_name,
                "type": self.device.__class__.__name__,
                "provider": "braket",
            }

            self.tracker.record["execute"] = {
                "submitted_at": submitted_at,
                "backend_name": device_name,
                "sdk": "braket",
                "num_circuits": len(circuits_for_logging),
                "execution_count": exec_count,
                "program_hash": circuit_hash,
                "shots": shots,
                "task_ids": task_ids,
            }

            logger.debug("Created envelope for task %s on %s", task_id, device_name)

        # Update stats periodically
        if (
            self.stats_update_interval > 0
            and exec_count % self.stats_update_interval == 0
        ):
            self._update_stats()

        return TrackedTask(
            task=task,
            tracker=self.tracker,
            device_name=device_name,
            envelope=envelope,
            shots=shots,
            should_log_results=should_log,
        )

    def run_batch(
        self,
        task_specifications: list[Any] | tuple[Any, ...],
        shots: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> TrackedTaskBatch:
        """
        Execute a batch of circuits using device.run_batch().

        This is the recommended way to run multiple circuits on Braket
        for better efficiency.

        Parameters
        ----------
        task_specifications : list or tuple
            List of circuits or programs to execute.
        shots : int or None, optional
            Number of shots per circuit. None uses provider default.
        *args : Any
            Additional positional arguments passed to device.run_batch().
        **kwargs : Any
            Additional keyword arguments passed to device.run_batch().

        Returns
        -------
        TrackedTaskBatch
            Wrapped batch that tracks result retrieval.
        """
        device_name = get_backend_name(self.device)
        submitted_at = utc_now_iso()

        # Flatten for logging (batch is always multiple)
        circuits_for_logging = list(task_specifications)

        # Increment execution counter
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute circuit hash
        circuit_hash = _compute_circuit_hash(circuits_for_logging)
        is_new_circuit = circuit_hash and circuit_hash not in self._seen_circuit_hashes
        if circuit_hash:
            self._seen_circuit_hashes.add(circuit_hash)

        # Determine logging behavior
        should_log = self._should_log(exec_count, circuit_hash, is_new_circuit)

        # Build execution options
        options: dict[str, Any] = {
            "batch": True,
            "batch_size": len(circuits_for_logging),
        }
        if args:
            options["args"] = to_jsonable(list(args))
        if kwargs:
            options["kwargs"] = to_jsonable(kwargs)

        # Execute batch
        batch: Any = None
        try:
            if shots is None:
                batch = self.device.run_batch(task_specifications, *args, **kwargs)
            else:
                batch = self.device.run_batch(
                    task_specifications, shots=shots, *args, **kwargs
                )
        except Exception as e:
            if should_log and circuits_for_logging:
                _log_submission_failure(
                    self.tracker,
                    device_name,
                    e,
                    circuits_for_logging,
                    shots,
                    submitted_at,
                )
            raise

        # Create envelope if logging
        envelope: ExecutionEnvelope | None = None
        if should_log and circuits_for_logging:
            envelope = _create_envelope(
                tracker=self.tracker,
                device=self.device,
                circuits=circuits_for_logging,
                shots=shots,
                task_ids=[],  # Batch doesn't have a single ID upfront
                submitted_at=submitted_at,
                circuit_hash=circuit_hash,
                options=options,
            )

            if circuit_hash:
                self._logged_circuit_hashes.add(circuit_hash)

            self._logged_execution_count += 1

            # Set tracker tags/params
            self.tracker.set_tag("backend_name", device_name)
            self.tracker.set_tag("provider", "braket")
            self.tracker.set_tag("adapter", "braket")
            self.tracker.set_tag("batch_execution", "true")

            if shots is not None:
                self.tracker.log_param("shots", int(shots))
            self.tracker.log_param("num_circuits", len(circuits_for_logging))
            self.tracker.log_param("batch_size", len(circuits_for_logging))

            # Update tracker record
            self.tracker.record["backend"] = {
                "name": device_name,
                "type": self.device.__class__.__name__,
                "provider": "braket",
            }

            self.tracker.record["execute"] = {
                "submitted_at": submitted_at,
                "backend_name": device_name,
                "sdk": "braket",
                "num_circuits": len(circuits_for_logging),
                "execution_count": exec_count,
                "program_hash": circuit_hash,
                "shots": shots,
                "batch": True,
            }

            logger.debug("Created envelope for batch on %s", device_name)

        # Update stats periodically
        if (
            self.stats_update_interval > 0
            and exec_count % self.stats_update_interval == 0
        ):
            self._update_stats()

        return TrackedTaskBatch(
            batch=batch,
            tracker=self.tracker,
            device_name=device_name,
            envelope=envelope,
            shots=shots,
            should_log_results=should_log,
        )

    def _should_log(
        self,
        exec_count: int,
        circuit_hash: str | None,
        is_new_circuit: bool,
    ) -> bool:
        """Determine if this execution should be logged."""
        if self.log_every_n == -1:
            return True
        if exec_count == 1:
            return True
        if self.log_new_circuits and is_new_circuit:
            return True
        if self.log_every_n > 0 and exec_count % self.log_every_n == 0:
            return True
        return False

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
        """Delegate attribute access to wrapped device."""
        return getattr(self.device, name)

    def __repr__(self) -> str:
        """Return string representation."""
        device_name = get_backend_name(self.device)
        return f"TrackedDevice(device={device_name!r}, run_id={self.tracker.run_id!r})"


class BraketAdapter:
    """
    Adapter for integrating Braket devices with devqubit tracking.

    This adapter wraps Braket devices to automatically create UEC-compliant
    execution envelopes containing device, program, execution, and result
    snapshots.

    Attributes
    ----------
    name : str
        Adapter identifier ("braket").
    """

    name: str = "braket"

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if executor is a supported Braket device.

        Parameters
        ----------
        executor : Any
            Potential executor instance.

        Returns
        -------
        bool
            True if executor is a Braket device with a `run` method.
        """
        if executor is None:
            return False

        module = getattr(executor, "__module__", "") or ""
        if "braket" not in module:
            return False

        return hasattr(executor, "run")

    def describe_executor(self, device: Any) -> dict[str, Any]:
        """
        Create a description of the device.

        Parameters
        ----------
        device : Any
            Braket device instance.

        Returns
        -------
        dict
            Device description with name, type, and provider.
        """
        return {
            "name": get_backend_name(device),
            "type": device.__class__.__name__,
            "provider": "braket",
        }

    def wrap_executor(
        self,
        device: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> TrackedDevice:
        """
        Wrap a device with tracking capabilities.

        Parameters
        ----------
        device : Any
            Braket device to wrap.
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
        TrackedDevice
            Wrapped device that logs execution artifacts.
        """
        return TrackedDevice(
            device=device,
            tracker=tracker,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
