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
>>> from devqubit_engine.tracking.run import track
>>>
>>> circuit = Circuit().h(0).cnot(0, 1)
>>>
>>> with track(project="my_experiment") as run:
...     device = run.wrap(LocalSimulator())
...     task = device.run(circuit, shots=1000)
...     result = task.result()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from devqubit_braket.circuits import compute_circuit_hashes
from devqubit_braket.envelope import create_envelope, log_submission_failure
from devqubit_braket.execution import TrackedTask, TrackedTaskBatch
from devqubit_braket.serialization import is_braket_circuit
from devqubit_braket.utils import extract_task_id, get_backend_name
from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.utils.common import utc_now_iso
from devqubit_engine.utils.serialization import to_jsonable


logger = logging.getLogger(__name__)


# =============================================================================
# ProgramSet Utilities
# =============================================================================


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

    has_entries = hasattr(obj, "entries")
    has_to_ir = hasattr(obj, "to_ir")
    has_total_executables = hasattr(obj, "total_executables")

    if has_entries and (has_to_ir or has_total_executables):
        return True

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
            for attr in ("circuit", "program", "task_specification"):
                circ = getattr(entry, attr, None)
                if circ is not None and is_braket_circuit(circ):
                    circuits.append(circ)
                    break
            else:
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

    if _is_program_set(task_specification):
        circuits = _extract_circuits_from_program_set(task_specification)
        meta = _get_program_set_metadata(task_specification)
        return task_specification, circuits, False, meta

    if is_braket_circuit(task_specification):
        return task_specification, [task_specification], True, None

    if isinstance(task_specification, list):
        return task_specification, task_specification, False, None

    if isinstance(task_specification, tuple):
        circuit_list = list(task_specification)
        return circuit_list, circuit_list, False, None

    try:
        circuit_list = list(task_specification)
        return circuit_list, circuit_list, False, None
    except TypeError:
        return task_specification, [task_specification], True, None


# =============================================================================
# TrackedDevice
# =============================================================================


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

        # Prepare execution context
        ctx = self._prepare_execution_context(
            circuits_for_logging=circuits_for_logging,
            kwargs=kwargs,
            extra_meta=extra_meta,
        )

        # Execute on actual device
        task: Any = None
        try:
            if shots is None:
                task = self.device.run(run_payload, *args, **kwargs)
            else:
                task = self.device.run(run_payload, shots=shots, *args, **kwargs)
        except Exception as e:
            if ctx["should_log"] and circuits_for_logging:
                log_submission_failure(
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
        if ctx["should_log"] and circuits_for_logging:
            envelope = self._create_and_log_envelope(
                device_name=device_name,
                circuits=circuits_for_logging,
                shots=shots,
                task_ids=task_ids,
                submitted_at=submitted_at,
                structural_hash=ctx["structural_hash"],
                parametric_hash=ctx["parametric_hash"],
                exec_count=ctx["exec_count"],
                options=ctx["options"],
                is_batch=False,
            )

        # Periodic stats update
        self._maybe_update_stats(ctx["exec_count"])

        return TrackedTask(
            task=task,
            tracker=self.tracker,
            device_name=device_name,
            envelope=envelope,
            shots=shots,
            should_log_results=ctx["should_log"],
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
        circuits_for_logging = (
            task_specifications
            if isinstance(task_specifications, list)
            else list(task_specifications)
        )

        # Prepare execution context
        ctx = self._prepare_execution_context(
            circuits_for_logging=circuits_for_logging,
            kwargs=kwargs,
            extra_meta=None,
        )
        ctx["options"]["batch"] = True
        ctx["options"]["batch_size"] = len(circuits_for_logging)

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
            if ctx["should_log"] and circuits_for_logging:
                log_submission_failure(
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
        if ctx["should_log"] and circuits_for_logging:
            envelope = self._create_and_log_envelope(
                device_name=device_name,
                circuits=circuits_for_logging,
                shots=shots,
                task_ids=[],  # Batch doesn't have a single ID upfront
                submitted_at=submitted_at,
                structural_hash=ctx["structural_hash"],
                parametric_hash=ctx["parametric_hash"],
                exec_count=ctx["exec_count"],
                options=ctx["options"],
                is_batch=True,
            )

        # Periodic stats update
        self._maybe_update_stats(ctx["exec_count"])

        return TrackedTaskBatch(
            batch=batch,
            tracker=self.tracker,
            device_name=device_name,
            envelope=envelope,
            shots=shots,
            should_log_results=ctx["should_log"],
        )

    def _prepare_execution_context(
        self,
        circuits_for_logging: list[Any],
        kwargs: dict[str, Any],
        extra_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Prepare common execution context for run() and run_batch().

        Returns a dict with: exec_count, structural_hash, parametric_hash,
        should_log, options.
        """
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute both hashes in one pass (avoids 2× circuit_to_op_stream)
        inputs = kwargs.get("inputs")
        structural_hash, parametric_hash = compute_circuit_hashes(
            circuits_for_logging, inputs
        )

        is_new_circuit = (
            structural_hash and structural_hash not in self._seen_circuit_hashes
        )
        if structural_hash:
            self._seen_circuit_hashes.add(structural_hash)

        should_log = self._should_log(exec_count, structural_hash, is_new_circuit)

        # Build execution options
        options: dict[str, Any] = {}
        if kwargs:
            options["kwargs"] = to_jsonable(kwargs)
        if extra_meta:
            options.update(extra_meta)

        return {
            "exec_count": exec_count,
            "structural_hash": structural_hash,
            "parametric_hash": parametric_hash,
            "should_log": should_log,
            "options": options,
        }

    def _create_and_log_envelope(
        self,
        device_name: str,
        circuits: list[Any],
        shots: int | None,
        task_ids: list[str],
        submitted_at: str,
        structural_hash: str | None,
        parametric_hash: str | None,
        exec_count: int,
        options: dict[str, Any],
        is_batch: bool,
    ) -> ExecutionEnvelope:
        """Create envelope and update tracker state."""
        envelope = create_envelope(
            tracker=self.tracker,
            device=self.device,
            circuits=circuits,
            shots=shots,
            task_ids=task_ids,
            submitted_at=submitted_at,
            structural_hash=structural_hash,
            parametric_hash=parametric_hash,
            execution_index=exec_count,
            options=options if options else None,
        )

        if structural_hash:
            self._logged_circuit_hashes.add(structural_hash)
        self._logged_execution_count += 1

        # Get actual provider from envelope device snapshot
        provider = "aws_braket"
        if envelope.device and envelope.device.provider:
            provider = envelope.device.provider

        # Set tracker tags/params
        self.tracker.set_tag("backend_name", device_name)
        self.tracker.set_tag("provider", provider)
        self.tracker.set_tag("adapter", "devqubit-braket")

        if is_batch:
            self.tracker.set_tag("batch_execution", "true")

        if shots is not None:
            self.tracker.log_param("shots", int(shots))
        self.tracker.log_param("num_circuits", len(circuits))

        if is_batch:
            self.tracker.log_param("batch_size", len(circuits))

        # Update tracker record
        self.tracker.record["backend"] = {
            "name": device_name,
            "type": self.device.__class__.__name__,
            "provider": provider,
        }

        self.tracker.record["execute"] = {
            "submitted_at": submitted_at,
            "backend_name": device_name,
            "sdk": "braket",
            "num_circuits": len(circuits),
            "execution_count": exec_count,
            "structural_hash": structural_hash,
            "parametric_hash": parametric_hash,
            "shots": shots,
            "batch": is_batch,
        }

        if not is_batch and task_ids:
            self.tracker.record["execute"]["task_ids"] = task_ids

        logger.debug(
            "Created envelope for %s on %s",
            "batch" if is_batch else f"task {task_ids}",
            device_name,
        )

        return envelope

    def _should_log(
        self,
        exec_count: int,
        structural_hash: str | None,
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

    def _maybe_update_stats(self, exec_count: int) -> None:
        """Update stats if interval has passed."""
        if (
            self.stats_update_interval > 0
            and exec_count % self.stats_update_interval == 0
        ):
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


# =============================================================================
# BraketAdapter
# =============================================================================


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
        from devqubit_braket.device import _detect_physical_provider

        return {
            "name": get_backend_name(device),
            "type": device.__class__.__name__,
            "provider": _detect_physical_provider(device),
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
