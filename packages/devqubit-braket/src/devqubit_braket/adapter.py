# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Braket adapter for devqubit tracking system.

Provides integration with Amazon Braket devices, enabling automatic tracking
of quantum circuit execution, results, and device configurations using the
Uniform Execution Contract (UEC) 1.0.

Example
-------
>>> from braket.circuits import Circuit
>>> from braket.devices import LocalSimulator
>>> from devqubit_engine.core.run import track
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
import logging
from dataclasses import dataclass, field
from typing import Any

from devqubit_braket.envelope import (
    create_envelope,
    log_submission_failure,
)
from devqubit_braket.serialization import is_braket_circuit
from devqubit_braket.tracked import TrackedTask, TrackedTaskBatch
from devqubit_braket.utils import extract_task_id, get_backend_name
from devqubit_engine.core.run import Run
from devqubit_engine.uec.envelope import ExecutionEnvelope
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_engine.utils.time_utils import utc_now_iso


logger = logging.getLogger(__name__)

# ============================================================================
# ProgramSet handling utilities
# ============================================================================


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


# ============================================================================
# Circuit hashing
# ============================================================================


def _compute_structural_hash(circuits: list[Any]) -> str | None:
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


def _compute_parametric_hash(
    circuits: list[Any],
    inputs: dict[str, float] | None = None,
) -> str | None:
    """
    Compute a parametric hash for Braket circuits.

    Unlike structural hash, this includes actual parameter values,
    making it suitable for identifying identical circuit executions.

    Parameters
    ----------
    circuits : list[Any]
        List of Braket Circuit objects.
    inputs : dict[str, float] or None
        Parameter bindings for FreeParameters.

    Returns
    -------
    str | None
        SHA256 hash with prefix, or None if circuits is empty.

    Notes
    -----
    Includes:
    - All structural information (gate types, qubit topology)
    - Resolved parameter values from inputs dict
    - Unresolved FreeParameter names
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

                # Get actual parameter values
                param_strs: list[str] = []
                if op is not None:
                    for attr in ("parameters", "params", "angles"):
                        val = getattr(op, attr, None)
                        if isinstance(val, (list, tuple)):
                            for p in val:
                                try:
                                    # Check if it's a FreeParameter
                                    if hasattr(p, "name"):
                                        if inputs and p.name in inputs:
                                            param_strs.append(
                                                f"{float(inputs[p.name]):.10f}"
                                            )
                                        else:
                                            param_strs.append(f"<param:{p.name}>")
                                    else:
                                        param_strs.append(f"{float(p):.10f}")
                                except (TypeError, ValueError):
                                    param_strs.append(str(p)[:50])
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

                params_suffix = (
                    f"|params=[{','.join(param_strs)}]" if param_strs else ""
                )
                op_sigs.append(f"{op_name}{params_suffix}|t{targets}")

            circuit_signatures.append("||".join(op_sigs))

        except Exception:
            circuit_signatures.append(str(circuit)[:500])

    payload = "\n".join(circuit_signatures).encode("utf-8", errors="replace")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


# ============================================================================
# TrackedDevice - wraps Braket devices with tracking
# ============================================================================


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
        circuit_hash = _compute_structural_hash(circuits_for_logging)
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
        if should_log and circuits_for_logging:
            envelope = create_envelope(
                tracker=self.tracker,
                device=self.device,
                circuits=circuits_for_logging,
                shots=shots,
                task_ids=task_ids,
                submitted_at=submitted_at,
                circuit_hash=circuit_hash,
                execution_index=exec_count,
                options=options if options else None,
            )

            if circuit_hash:
                self._logged_circuit_hashes.add(circuit_hash)

            self._logged_execution_count += 1

            # Set tracker tags/params (P1 fix: provider is platform, not SDK)
            self.tracker.set_tag("backend_name", device_name)
            self.tracker.set_tag("provider", "aws_braket")
            self.tracker.set_tag("adapter", "devqubit-braket")

            if shots is not None:
                self.tracker.log_param("shots", int(shots))
            self.tracker.log_param("num_circuits", len(circuits_for_logging))

            # Update tracker record
            self.tracker.record["backend"] = {
                "name": device_name,
                "type": self.device.__class__.__name__,
                "provider": "aws_braket",
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
        circuit_hash = _compute_structural_hash(circuits_for_logging)
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
        if should_log and circuits_for_logging:
            envelope = create_envelope(
                tracker=self.tracker,
                device=self.device,
                circuits=circuits_for_logging,
                shots=shots,
                task_ids=[],  # Batch doesn't have a single ID upfront
                submitted_at=submitted_at,
                circuit_hash=circuit_hash,
                execution_index=exec_count,
                options=options,
            )

            if circuit_hash:
                self._logged_circuit_hashes.add(circuit_hash)

            self._logged_execution_count += 1

            # Set tracker tags/params (P1 fix: provider is platform, not SDK)
            self.tracker.set_tag("backend_name", device_name)
            self.tracker.set_tag("provider", "aws_braket")
            self.tracker.set_tag("adapter", "devqubit-braket")
            self.tracker.set_tag("batch_execution", "true")

            if shots is not None:
                self.tracker.log_param("shots", int(shots))
            self.tracker.log_param("num_circuits", len(circuits_for_logging))
            self.tracker.log_param("batch_size", len(circuits_for_logging))

            # Update tracker record
            self.tracker.record["backend"] = {
                "name": device_name,
                "type": self.device.__class__.__name__,
                "provider": "aws_braket",
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


# ============================================================================
# BraketAdapter - main adapter class
# ============================================================================


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
            "provider": "aws_braket",
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
