# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Cirq adapter for devqubit tracking system.

Provides integration with Google Cirq simulators and processors, enabling
automatic tracking of quantum circuit execution, results, and configurations
using the Uniform Execution Contract (UEC).

Example
-------
>>> import cirq
>>> from devqubit_engine.tracking import track
>>>
>>> q0, q1 = cirq.LineQubit.range(2)
>>> circuit = cirq.Circuit([
...     cirq.H(q0),
...     cirq.CNOT(q0, q1),
...     cirq.measure(q0, q1, key='m'),
... ])
>>>
>>> with track(project="my_experiment") as run:
...     simulator = run.wrap(cirq.Simulator())
...     result = simulator.run(circuit, repetitions=1000)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from devqubit_cirq.circuits import (
    compute_circuit_hashes,
)
from devqubit_cirq.results import normalize_counts_payload
from devqubit_cirq.serialization import (
    CirqCircuitSerializer,
    circuits_to_text,
    is_cirq_circuit,
)
from devqubit_cirq.snapshot import create_device_snapshot
from devqubit_cirq.utils import cirq_version, get_adapter_version, get_backend_name
from devqubit_engine.circuit.models import CircuitFormat
from devqubit_engine.storage.types import ArtifactRef
from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.device import DeviceSnapshot
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.execution import ExecutionSnapshot, ProducerInfo
from devqubit_engine.uec.models.program import (
    ProgramArtifact,
    ProgramRole,
    ProgramSnapshot,
    TranspilationInfo,
    TranspilationMode,
)
from devqubit_engine.uec.models.result import (
    ResultError,
    ResultItem,
    ResultSnapshot,
)
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable


logger = logging.getLogger(__name__)


# Module-level serializer instance
_serializer = CirqCircuitSerializer()


def _materialize_circuits(circuits: Any) -> tuple[list[Any], bool]:
    """
    Materialize circuit inputs exactly once.

    Parameters
    ----------
    circuits : Any
        A Circuit, or an iterable of Circuit objects.

    Returns
    -------
    circuit_list : list
        List of circuit objects.
    was_single : bool
        True if input was a single circuit.
    """
    if circuits is None:
        return [], False

    if is_cirq_circuit(circuits):
        return [circuits], True

    if isinstance(circuits, (list, tuple)):
        return list(circuits), False

    try:
        return list(circuits), False
    except TypeError:
        return [circuits], True


def _serialize_and_log_circuits(
    tracker: Run,
    circuits: list[Any],
    simulator_name: str,
) -> list[ArtifactRef]:
    """
    Serialize circuits and log as artifacts.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    circuits : list
        List of Cirq circuits.
    simulator_name : str
        Backend name for metadata.

    Returns
    -------
    list of ArtifactRef
        References to logged circuit artifacts.
    """
    artifact_refs: list[ArtifactRef] = []
    meta = {"backend_name": simulator_name, "cirq_version": cirq_version()}

    # Serialize Cirq JSON (native format)
    for i, circuit in enumerate(circuits):
        try:
            json_data = _serializer.serialize(circuit, CircuitFormat.CIRQ_JSON, index=i)
            ref = tracker.log_bytes(
                kind="cirq.circuit.json",
                data=json_data.as_bytes(),
                media_type="application/json",
                role="program",
                meta={**meta, "index": i},
            )
            if ref:
                artifact_refs.append(ref)
        except Exception as e:
            logger.debug("Failed to serialize circuit %d to JSON: %s", i, e)

    # Log circuit diagrams (human-readable)
    try:
        tracker.log_bytes(
            kind="cirq.circuits.txt",
            data=circuits_to_text(circuits).encode("utf-8"),
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
    structural_hash: str | None,
    parametric_hash: str | None = None,
) -> ProgramSnapshot:
    """
    Create a ProgramSnapshot from circuits and their artifact refs.

    Parameters
    ----------
    circuits : list
        List of Cirq circuits.
    artifact_refs : list of ArtifactRef
        References to logged circuit artifacts.
    structural_hash : str or None
        Structural hash (ignores parameter values).
    parametric_hash : str or None
        Parametric hash (includes parameter values).

    Returns
    -------
    ProgramSnapshot
        Program snapshot with logical artifacts.
    """
    logical_artifacts: list[ProgramArtifact] = [
        ProgramArtifact(
            ref=ref,
            role=ProgramRole.LOGICAL,
            format="cirq_json",
            name=(
                getattr(circuits[i], "name", None) or f"circuit_{i}"
                if i < len(circuits)
                else f"circuit_{i}"
            ),
            index=i,
        )
        for i, ref in enumerate(artifact_refs)
    ]

    # If parametric_hash not provided, use structural_hash
    effective_parametric_hash = parametric_hash or structural_hash

    return ProgramSnapshot(
        logical=logical_artifacts,
        physical=[],  # Cirq doesn't expose transpiled circuits
        structural_hash=structural_hash,
        parametric_hash=effective_parametric_hash,
        # For Cirq without transpilation, executed hashes equal logical
        executed_structural_hash=structural_hash,
        executed_parametric_hash=effective_parametric_hash,
        num_circuits=len(circuits),
    )


def _create_execution_snapshot(
    repetitions: int,
    submitted_at: str,
    is_sweep: bool = False,
    params: Any = None,
    options: dict[str, Any] | None = None,
) -> ExecutionSnapshot:
    """
    Create an ExecutionSnapshot for a Cirq execution.

    Parameters
    ----------
    repetitions : int
        Number of repetitions (shots).
    submitted_at : str
        ISO 8601 submission timestamp.
    is_sweep : bool
        Whether this is a parameter sweep.
    params : Any, optional
        Parameter sweep or resolver.
    options : dict, optional
        Additional execution options.

    Returns
    -------
    ExecutionSnapshot
        Execution metadata snapshot.
    """
    exec_options = options.copy() if options else {}
    if is_sweep and params is not None:
        exec_options["sweep"] = True
        exec_options["params"] = to_jsonable(params)

    return ExecutionSnapshot(
        submitted_at=submitted_at,
        shots=repetitions,
        execution_count=1,
        transpilation=TranspilationInfo(
            mode=TranspilationMode.MANUAL,  # Cirq doesn't auto-transpile
            transpiled_by="user",
        ),
        options=exec_options,
        sdk="cirq",
    )


def _create_result_snapshot(
    result: Any,
    raw_result_ref: ArtifactRef | None,
    repetitions: int | None,
    is_sweep: bool = False,
    error_info: dict[str, Any] | None = None,
    counts_payload: dict[str, Any] | None = None,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot from Cirq result(s).

    Parameters
    ----------
    result : Any
        Cirq result object or list of results (None if execution failed).
    raw_result_ref : ArtifactRef or None
        Reference to raw result artifact.
    repetitions : int or None
        Number of repetitions used.
    is_sweep : bool
        Whether this is from a parameter sweep.
    error_info : dict, optional
        Error information if execution failed.
        Contains "type" and "message" keys.
    counts_payload : dict, optional
        Pre-computed normalized counts payload.  When provided the expensive
        ``normalize_counts_payload`` call is skipped.

    Returns
    -------
    ResultSnapshot
        Result snapshot with normalized counts.
    """
    # Handle failure case (error_info provided)
    if error_info is not None:
        error = ResultError(
            type=error_info.get("type", "UnknownError"),
            message=error_info.get("message", "Unknown error"),
        )
        metadata: dict[str, Any] = {"sweep": is_sweep} if is_sweep else {}
        return ResultSnapshot(
            success=False,
            status="failed",
            items=[],
            error=error,
            raw_result_ref=raw_result_ref,
            metadata=metadata,
        )

    # Handle None result (without explicit error)
    if result is None:
        return ResultSnapshot(
            success=False,
            status="failed",
            items=[],
            error=ResultError(type="NullResult", message="Result is None"),
            raw_result_ref=raw_result_ref,
            metadata={"sweep": is_sweep} if is_sweep else {},
        )

    # Process successful result
    if counts_payload is None:
        try:
            counts_payload = normalize_counts_payload(result)
        except Exception as e:
            logger.debug("Failed to normalize counts payload: %s", e)
            counts_payload = {"experiments": [], "format": {}}

    # Build items list
    items: list[ResultItem] = []

    # Use format dict directly from payload (already a CountsFormat.to_dict())
    format_dict = counts_payload.get("format", {})

    for exp in counts_payload.get("experiments", []):
        circuit_index = exp.get("index", 0)
        counts = exp.get("counts", {})

        items.append(
            ResultItem(
                item_index=circuit_index,
                success=True,
                counts={
                    "counts": counts,
                    "shots": repetitions,
                    "format": format_dict,
                },
            )
        )

    metadata = {"sweep": is_sweep} if is_sweep else {}
    metadata["num_experiments"] = len(items)

    return ResultSnapshot(
        success=len(items) > 0,
        status="completed" if len(items) > 0 else "failed",
        items=items,
        error=None,
        raw_result_ref=raw_result_ref,
        metadata=metadata,
    )


def _create_and_log_envelope(
    tracker: Run,
    simulator: Any,
    circuits: list[Any],
    repetitions: int,
    submitted_at: str,
    structural_hash: str | None,
    parametric_hash: str | None = None,
    is_sweep: bool = False,
    params: Any = None,
    options: dict[str, Any] | None = None,
) -> ExecutionEnvelope:
    """
    Create and prepare an ExecutionEnvelope (pre-result).

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    simulator : Any
        Cirq simulator.
    circuits : list
        List of circuits.
    repetitions : int
        Number of repetitions.
    submitted_at : str
        Submission timestamp.
    structural_hash : str or None
        Structural hash (ignores parameter values).
    parametric_hash : str or None
        Parametric hash (includes parameter values).
    is_sweep : bool
        Whether this is a parameter sweep.
    params : Any, optional
        Parameter sweep or resolver.
    options : dict, optional
        Execution options.

    Returns
    -------
    ExecutionEnvelope
        Envelope with device, program, and execution snapshots.
    """
    simulator_name = get_backend_name(simulator)

    # Create device snapshot with tracker for raw_properties logging
    try:
        device_snapshot = create_device_snapshot(simulator, tracker=tracker)
    except Exception as e:
        logger.warning(
            "Failed to create device snapshot: %s. Using minimal snapshot.", e
        )
        device_snapshot = DeviceSnapshot(
            captured_at=utc_now_iso(),
            backend_name=simulator_name,
            backend_type="simulator",
            provider="local",  # Physical provider, not SDK
            sdk_versions={"cirq": cirq_version()},
        )

    # Update tracker record with device snapshot
    tracker.record["device_snapshot"] = {
        "sdk": "cirq",  # SDK frontend (always cirq for this adapter)
        "backend_name": simulator_name,
        "backend_type": device_snapshot.backend_type,
        "provider": device_snapshot.provider,  # Physical provider from snapshot
        "captured_at": device_snapshot.captured_at,
        "num_qubits": device_snapshot.num_qubits,
    }

    # Serialize and log circuits
    artifact_refs = _serialize_and_log_circuits(tracker, circuits, simulator_name)

    # Create ProducerInfo – reuse version from device snapshot
    producer = ProducerInfo.create(
        adapter="devqubit-cirq",
        adapter_version=get_adapter_version(),
        sdk="cirq",
        sdk_version=device_snapshot.sdk_versions.get("cirq", cirq_version()),
        frontends=["cirq"],
    )

    return ExecutionEnvelope(
        envelope_id=uuid.uuid4().hex[:26],
        created_at=utc_now_iso(),
        producer=producer,
        # Placeholder – overwritten by _finalize_envelope_with_result
        result=ResultSnapshot(
            success=False,
            status="failed",
            items=[],
        ),
        device=device_snapshot,
        program=_create_program_snapshot(
            circuits, artifact_refs, structural_hash, parametric_hash
        ),
        execution=_create_execution_snapshot(
            repetitions, submitted_at, is_sweep, params, options
        ),
    )


def _finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result: Any,
    simulator_name: str,
    repetitions: int | None,
    is_sweep: bool = False,
    error_info: dict[str, Any] | None = None,
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
        Cirq result object or list of results (None if execution failed).
    simulator_name : str
        Simulator name.
    repetitions : int or None
        Number of repetitions.
    is_sweep : bool
        Whether this is from a parameter sweep.
    error_info : dict, optional
        Error information if execution failed.
        Contains "type" and "message" keys.

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

    # Log raw result (if we have one)
    raw_result_ref = None
    if result is not None:
        try:
            try:
                result_payload = to_jsonable(result)
            except (TypeError, ValueError) as e:
                logger.debug("Failed to serialize result, falling back to repr: %s", e)
                result_payload = {"repr": repr(result)[:2000]}

            raw_result_ref = tracker.log_json(
                name="cirq.result",
                obj=result_payload,
                role="results",
                kind="result.cirq.raw.json",
            )
        except Exception as e:
            logger.warning("Failed to log raw result: %s", e)

    # Compute counts payload once (used by both result snapshot and counts logging)
    counts_payload: dict[str, Any] = {"experiments": []}
    if result is not None and error_info is None:
        try:
            counts_payload = normalize_counts_payload(result)
        except Exception as e:
            logger.debug("Failed to normalize counts payload: %s", e)

    # Update envelope with result snapshot (handles both success and failure)
    envelope.result = _create_result_snapshot(
        result=result,
        raw_result_ref=raw_result_ref,
        repetitions=repetitions,
        is_sweep=is_sweep,
        error_info=error_info,
        counts_payload=counts_payload,
    )

    if envelope.execution:
        envelope.execution.completed_at = utc_now_iso()

    # Validate and log envelope - EnvelopeValidationError must propagate
    # to enforce strict validation for adapter runs
    tracker.log_envelope(envelope=envelope)

    # Log normalized counts
    if counts_payload.get("experiments"):
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
        "backend_name": simulator_name,
        "num_experiments": len(counts_payload.get("experiments", [])),
        "result_type": "counts",
        "sweep": is_sweep,
    }

    logger.debug("Logged execution envelope for %s", simulator_name)
    return envelope


@dataclass
class TrackedSimulator:
    """
    Wrapper for Cirq simulator that tracks circuit execution.

    Intercepts `run`, `run_sweep`, and `run_batch` calls to automatically
    create UEC-compliant execution envelopes.

    Parameters
    ----------
    simulator : Any
        Original Cirq simulator instance.
    tracker : Run
        Tracker instance for logging artifacts.
    log_every_n : int
        Logging frequency: 0=first only (default), N>0=every Nth, -1=all.
    log_new_circuits : bool
        Auto-log new circuit structures (default True).
    stats_update_interval : int
        Update stats every N executions (default 1000).
    """

    simulator: Any
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

    def _should_log(
        self,
        exec_count: int,
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

    def _track_execution(
        self,
        circuit_list: list[Any],
        result: Any,
        repetitions: int,
        submitted_at: str,
        is_sweep: bool = False,
        is_batch: bool = False,
        params: Any = None,
        extra_options: dict[str, Any] | None = None,
        error_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Common execution tracking logic for run, run_sweep, and run_batch.

        Parameters
        ----------
        circuit_list : list
            List of executed circuits.
        result : Any
            Execution result (None if execution failed).
        repetitions : int
            Number of repetitions.
        submitted_at : str
            Submission timestamp.
        is_sweep : bool
            Whether this is a parameter sweep.
        is_batch : bool
            Whether this is a batch execution.
        params : Any, optional
            Parameter sweep or resolver.
        extra_options : dict, optional
            Additional options to include.
        error_info : dict, optional
            Error information if execution failed.
            Contains "type" and "message" keys.
        """
        simulator_name = get_backend_name(self.simulator)

        # Increment and track circuit hash
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute both structural and parametric hashes in one pass
        # Extract resolver from params for parametric hash
        # params can be: ParamResolver, Sweep, list of resolvers, or None
        resolver = None
        if params is not None:
            # If it's a ParamResolver directly
            if hasattr(params, "param_dict"):
                resolver = params
            # If it's a Sweep, try to get first resolver
            elif hasattr(params, "__iter__"):
                try:
                    first_param = next(iter(params), None)
                    if first_param is not None and hasattr(first_param, "param_dict"):
                        resolver = first_param
                except (TypeError, StopIteration, AttributeError) as e:
                    logger.debug("Failed to extract resolver from params: %s", e)

        structural_hash, parametric_hash = compute_circuit_hashes(
            circuit_list, resolver
        )

        is_new_circuit = (
            structural_hash and structural_hash not in self._seen_circuit_hashes
        )
        if structural_hash:
            self._seen_circuit_hashes.add(structural_hash)

        # Check if we should log this execution
        if not (self._should_log(exec_count, is_new_circuit) and circuit_list):
            self._maybe_update_stats(exec_count)
            return

        # Build options
        options = extra_options.copy() if extra_options else {}
        if is_batch:
            options["batch"] = True

        # Create and finalize envelope
        try:
            envelope = _create_and_log_envelope(
                tracker=self.tracker,
                simulator=self.simulator,
                circuits=circuit_list,
                repetitions=repetitions,
                submitted_at=submitted_at,
                structural_hash=structural_hash,
                parametric_hash=parametric_hash,
                is_sweep=is_sweep,
                params=params,
                options=options if options else None,
            )

            _finalize_envelope_with_result(
                tracker=self.tracker,
                envelope=envelope,
                result=result,
                simulator_name=simulator_name,
                repetitions=repetitions,
                is_sweep=is_sweep,
                error_info=error_info,
            )
        except Exception as e:
            logger.warning(
                "Failed to create/finalize envelope for %s: %s",
                simulator_name,
                e,
            )
            self.tracker.record.setdefault("warnings", []).append(
                {
                    "type": "envelope_creation_failed",
                    "message": str(e),
                    "simulator_name": simulator_name,
                }
            )

        if structural_hash:
            self._logged_circuit_hashes.add(structural_hash)
        self._logged_execution_count += 1

        # Get physical provider from device snapshot (if available)
        # The device snapshot is created inside the envelope
        physical_provider = "local"  # Default for Cirq simulators
        try:
            device_snapshot = self.tracker.record.get("device_snapshot", {})
            physical_provider = device_snapshot.get("provider", "local")
        except (AttributeError, TypeError) as e:
            logger.debug("Failed to get provider from device_snapshot: %s", e)

        # Set tracker tags and params
        self.tracker.set_tag("backend_name", simulator_name)
        self.tracker.set_tag("provider", physical_provider)  # Physical provider
        self.tracker.set_tag("sdk", "cirq")  # SDK frontend
        self.tracker.set_tag("adapter", "devqubit-cirq")
        self.tracker.log_param("repetitions", repetitions)
        self.tracker.log_param("num_circuits", len(circuit_list))

        if is_sweep:
            self.tracker.log_param("sweep", True)
        if is_batch:
            self.tracker.log_param("batch", True)

        # Update tracker record
        self.tracker.record["backend"] = {
            "name": simulator_name,
            "type": self.simulator.__class__.__name__,
            "provider": physical_provider,  # Physical provider
            "sdk": "cirq",  # SDK frontend
        }

        self.tracker.record["execute"] = {
            "submitted_at": submitted_at,
            "backend_name": simulator_name,
            "sdk": "cirq",
            "num_circuits": len(circuit_list),
            "execution_count": exec_count,
            "structural_hash": structural_hash,
            "parametric_hash": parametric_hash,
            "repetitions": repetitions,
            "sweep": is_sweep,
            "batch": is_batch,
        }

        self._maybe_update_stats(exec_count)

    def _maybe_update_stats(self, exec_count: int) -> None:
        """Update stats if interval reached."""
        if (
            self.stats_update_interval > 0
            and exec_count % self.stats_update_interval == 0
        ):
            self._update_stats()

    def run(
        self,
        program: Any,
        *args: Any,
        repetitions: int = 1,
        **kwargs: Any,
    ) -> Any:
        """
        Execute circuit and create execution envelope.

        Parameters
        ----------
        program : Circuit
            Cirq circuit to execute.
        repetitions : int, optional
            Number of measurement repetitions. Default is 1.
        *args : Any
            Additional positional arguments passed to simulator.
        **kwargs : Any
            Additional keyword arguments passed to simulator.

        Returns
        -------
        cirq.Result
            Cirq Result object containing measurement outcomes.

        Raises
        ------
        Exception
            Re-raises any exception from the simulator after logging
            a failure envelope.
        """
        circuit_list, _ = _materialize_circuits(program)
        submitted_at = utc_now_iso()

        extra_options: dict[str, Any] = {}
        if args:
            extra_options["args"] = to_jsonable(list(args))
        if kwargs:
            extra_options["kwargs"] = to_jsonable(kwargs)

        # Capture exception and log failure envelope before re-raising
        result: Any = None
        original_exception: BaseException | None = None
        execution_succeeded = False

        try:
            result = self.simulator.run(
                program, *args, repetitions=repetitions, **kwargs
            )
            execution_succeeded = True
        except Exception as e:
            original_exception = e
            # Log failure envelope
            self._track_execution(
                circuit_list,
                None,  # No result
                repetitions,
                submitted_at,
                extra_options=extra_options if extra_options else None,
                error_info={
                    "type": type(e).__name__,
                    "message": str(e),
                },
            )

        if execution_succeeded:
            self._track_execution(
                circuit_list,
                result,
                repetitions,
                submitted_at,
                extra_options=extra_options if extra_options else None,
            )

        # Re-raise original exception preserving type and traceback
        if original_exception is not None:
            raise original_exception

        return result

    def run_sweep(
        self,
        program: Any,
        params: Any,
        *args: Any,
        repetitions: int = 1,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Execute circuit sweep and create execution envelope.

        Parameters
        ----------
        program : Circuit
            Cirq circuit to execute.
        params : Sweep or Resolver
            Parameter sweep or resolver.
        repetitions : int, optional
            Number of measurement repetitions per parameter set. Default is 1.
        *args : Any
            Additional positional arguments passed to simulator.
        **kwargs : Any
            Additional keyword arguments passed to simulator.

        Returns
        -------
        list of cirq.Result
            List of Result objects, one per parameter set.

        Raises
        ------
        Exception
            Re-raises any exception from the simulator after logging
            a failure envelope.
        """
        circuit_list, _ = _materialize_circuits(program)
        submitted_at = utc_now_iso()

        extra_options: dict[str, Any] = {}
        if args:
            extra_options["args"] = to_jsonable(list(args))
        if kwargs:
            extra_options["kwargs"] = to_jsonable(kwargs)

        # Capture exception and log failure envelope before re-raising
        results: Any = None
        original_exception: BaseException | None = None
        execution_succeeded = False

        try:
            results = self.simulator.run_sweep(
                program, params, *args, repetitions=repetitions, **kwargs
            )
            execution_succeeded = True
        except Exception as e:
            original_exception = e
            # Log failure envelope
            self._track_execution(
                circuit_list,
                None,  # No result
                repetitions,
                submitted_at,
                is_sweep=True,
                params=params,
                extra_options=extra_options if extra_options else None,
                error_info={
                    "type": type(e).__name__,
                    "message": str(e),
                },
            )

        if execution_succeeded:
            self._track_execution(
                circuit_list,
                results,
                repetitions,
                submitted_at,
                is_sweep=True,
                params=params,
                extra_options=extra_options if extra_options else None,
            )

        # Re-raise original exception preserving type and traceback
        if original_exception is not None:
            raise original_exception

        return results

    def run_batch(
        self,
        programs: Any,
        params_list: Any = None,
        *args: Any,
        repetitions: int | list[int] = 1,
        **kwargs: Any,
    ) -> list[list[Any]]:
        """
        Execute batch of circuits and create execution envelope.

        Wraps Cirq's run_batch which returns a list of lists:
        outer list corresponds to circuits, inner list to parameter sweeps.

        Parameters
        ----------
        programs : list of Circuit
            List of Cirq circuits to execute.
        params_list : list of Sweep or Resolver, optional
            Parameter sweeps/resolvers for each circuit. If None, uses
            empty resolver for each circuit.
        repetitions : int or list of int
            Number of repetitions. Can be a single int (applied to all)
            or a list with one value per circuit.
        *args : Any
            Additional positional arguments passed to simulator.
        **kwargs : Any
            Additional keyword arguments passed to simulator.

        Returns
        -------
        list of list of cirq.Result
            Nested list where results[i][j] is the result for circuit i
            with parameter set j.

        Raises
        ------
        Exception
            Re-raises any exception from the simulator after logging
            a failure envelope.
        """
        circuit_list, _ = _materialize_circuits(programs)
        submitted_at = utc_now_iso()

        # Determine effective repetitions for logging
        if isinstance(repetitions, (list, tuple)):
            total_reps = repetitions[0] if repetitions else 1
        else:
            total_reps = repetitions

        extra_options: dict[str, Any] = {}
        if isinstance(repetitions, (list, tuple)):
            extra_options["repetitions_per_circuit"] = list(repetitions)
        if params_list is not None:
            extra_options["params_list"] = to_jsonable(params_list)
        if args:
            extra_options["args"] = to_jsonable(list(args))
        if kwargs:
            extra_options["kwargs"] = to_jsonable(kwargs)

        # Capture exception and log failure envelope before re-raising
        results: Any = None
        original_exception: BaseException | None = None
        execution_succeeded = False

        try:
            results = self.simulator.run_batch(
                programs, params_list, *args, repetitions=repetitions, **kwargs
            )
            execution_succeeded = True
        except Exception as e:
            original_exception = e
            # Log failure envelope
            self._track_execution(
                circuit_list,
                None,  # No result
                total_reps,
                submitted_at,
                is_sweep=True,
                is_batch=True,
                params=params_list,
                extra_options=extra_options if extra_options else None,
                error_info={
                    "type": type(e).__name__,
                    "message": str(e),
                },
            )

        if execution_succeeded:
            self._track_execution(
                circuit_list,
                results,
                total_reps,
                submitted_at,
                is_sweep=True,
                is_batch=True,
                params=params_list,
                extra_options=extra_options if extra_options else None,
            )

        # Re-raise original exception preserving type and traceback
        if original_exception is not None:
            raise original_exception

        return results

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped simulator."""
        return getattr(self.simulator, name)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TrackedSimulator(simulator={self.simulator.__class__.__name__}, "
            f"run_id={self.tracker.run_id!r})"
        )


class CirqAdapter:
    """
    Adapter for integrating Cirq simulators with devqubit tracking.

    This adapter wraps Cirq simulators to automatically create UEC-compliant
    execution envelopes containing device, program, execution, and result
    snapshots.

    Attributes
    ----------
    name : str
        Adapter identifier ("cirq").
    """

    name: str = "cirq"

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if executor is a supported Cirq sampler.

        Uses isinstance check against cirq.Sampler as preferred method,
        with duck-typing fallback for third-party Cirq-compatible samplers.

        Parameters
        ----------
        executor : Any
            Potential executor instance.

        Returns
        -------
        bool
            True if executor is a Cirq Sampler or compatible object.
        """
        if executor is None:
            return False

        # Preferred: isinstance check against cirq.Sampler
        try:
            import cirq

            if isinstance(executor, cirq.Sampler):
                return True
        except ImportError as e:
            logger.debug("Optional import unavailable: %s", e)

        # Fallback: duck-typing for 3rd-party Cirq-compatible samplers
        if not hasattr(executor, "run"):
            return False

        # Verify it's Cirq-like (has run_sweep or module contains cirq)
        if hasattr(executor, "run_sweep"):
            return True

        module = getattr(executor, "__module__", "") or ""
        return "cirq" in module.lower()

    def describe_executor(self, simulator: Any) -> dict[str, Any]:
        """
        Create a description of the simulator.

        Parameters
        ----------
        simulator : Any
            Cirq simulator instance.

        Returns
        -------
        dict
            Simulator description with name, type, provider, and SDK.
        """
        # Import provider detection from snapshot module
        from devqubit_cirq.snapshot import _detect_execution_provider

        try:
            physical_provider = _detect_execution_provider(simulator)
        except (AttributeError, TypeError) as e:
            logger.debug("Failed to detect execution provider: %s", e)
            physical_provider = "local"

        return {
            "name": get_backend_name(simulator),
            "type": simulator.__class__.__name__,
            "provider": physical_provider,  # Physical provider
            "sdk": "cirq",  # SDK frontend
        }

    def wrap_executor(
        self,
        simulator: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> TrackedSimulator:
        """
        Wrap a simulator with tracking capabilities.

        Parameters
        ----------
        simulator : Any
            Cirq simulator to wrap.
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
        TrackedSimulator
            Wrapped simulator that logs execution artifacts.
        """
        return TrackedSimulator(
            simulator=simulator,
            tracker=tracker,
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
