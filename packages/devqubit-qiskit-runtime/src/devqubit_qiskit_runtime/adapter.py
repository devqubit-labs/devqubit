# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Qiskit IBM Runtime adapter for devqubit tracking system.

This module provides integration with Qiskit IBM Runtime primitives
(Sampler/Estimator), enabling automatic tracking of quantum circuit
execution, results, and primitive configurations following the devqubit
Uniform Execution Contract (UEC).

The adapter produces an ExecutionEnvelope containing four canonical snapshots:
- DeviceSnapshot: Backend state, calibration, and primitive frontend config
- ProgramSnapshot: Logical and physical circuit artifacts with transpilation tracking
- ExecutionSnapshot: Submission metadata with transpilation details
- ResultSnapshot: Normalized measurement results or expectation values

Supported Primitives
--------------------
This adapter supports Qiskit IBM Runtime V2 primitives:
- SamplerV2: For sampling measurement outcomes
- EstimatorV2: For computing expectation values

Note: For direct backend usage (backend.run()), use the ``qiskit`` adapter instead.

Performance Note
----------------
For training loops or repeated primitive executions, use the `log_every_n`
parameter to sample executions rather than logging all:

    tracked_sampler = run.wrap(sampler)  # log_every_n=0 (logs first run only)
    tracked_sampler = run.wrap(sampler, log_every_n=100)  # Log every 100th
    tracked_sampler = run.wrap(sampler, log_every_n=-1)  # Log all (slow!)

Note: Even with log_every_n=0, the first execution is always logged with full
snapshots and artifacts.

Example
-------
>>> from qiskit import QuantumCircuit
>>> from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
>>> from devqubit_engine.core import track
>>>
>>> service = QiskitRuntimeService()
>>> backend = service.backend("ibm_brisbane")
>>>
>>> qc = QuantumCircuit(2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> qc.measure_all()
>>>
>>> with track(project="my_experiment") as run:
...     sampler = run.wrap(SamplerV2(backend))
...     job = sampler.run([qc])
...     result = job.result()
"""

from __future__ import annotations

import logging
import uuid
import warnings
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
from devqubit_engine.uec.result import ResultSnapshot
from devqubit_engine.uec.types import (
    ProgramRole,
    TranspilationMode,
)
from devqubit_engine.utils.serialization import to_jsonable
from devqubit_engine.utils.time_utils import utc_now_iso
from devqubit_qiskit.serialization import QiskitCircuitSerializer
from devqubit_qiskit_runtime.circuits import (
    circuits_to_text,
    compute_circuit_hash,
)
from devqubit_qiskit_runtime.envelope import (
    create_failure_result_snapshot,
    finalize_envelope_with_result,
)
from devqubit_qiskit_runtime.pubs import (
    extract_circuits_from_pubs,
    extract_pubs_structure,
    iter_pubs,
)
from devqubit_qiskit_runtime.results import (
    build_estimator_result_snapshot,
    build_sampler_result_snapshot,
)
from devqubit_qiskit_runtime.snapshot import create_device_snapshot
from devqubit_qiskit_runtime.transpilation import (
    TranspilationConfig,
    TranspilationOptions,
    prepare_pubs_for_primitive,
)
from devqubit_qiskit_runtime.utils import (
    collect_sdk_versions,
    extract_job_id,
    get_backend_name,
    get_primitive_type,
    is_runtime_primitive,
)


# Adapter version for ProducerInfo
_ADAPTER_VERSION = "1.0.0"


logger = logging.getLogger(__name__)

# Module-level serializer instance
_serializer = QiskitCircuitSerializer()


def _map_transpilation_mode(mode: str) -> TranspilationMode:
    """Map string mode to UEC TranspilationMode enum."""
    mode_map = {
        "auto": TranspilationMode.AUTO,
        "managed": TranspilationMode.MANAGED,
        "manual": TranspilationMode.MANUAL,
    }
    return mode_map.get(mode.lower(), TranspilationMode.AUTO)


@dataclass
class TrackedRuntimeJob:
    """
    Wrapper for Runtime job that tracks result retrieval.

    Parameters
    ----------
    job : Any
        Original Runtime job instance.
    tracker : Run
        Tracker instance for logging.
    executor_name : str
        Name of the backend/primitive that created this job.
    primitive_type : str
        Type of primitive ('sampler' or 'estimator').
    should_log_results : bool
        Whether to log results for this job.
    envelope : ExecutionEnvelope or None
        Envelope to finalize when result() is called.

    Attributes
    ----------
    result_snapshot : ResultSnapshot or None
        Captured result snapshot after result() is called.
    """

    job: Any
    tracker: Run
    executor_name: str
    primitive_type: str
    should_log_results: bool = True
    envelope: ExecutionEnvelope | None = None

    result_snapshot: ResultSnapshot | None = field(default=None, init=False, repr=False)
    _cached_result: Any = field(default=None, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)

    def __repr__(self) -> str:
        job_id = extract_job_id(self.job) or "unknown"
        return (
            f"TrackedRuntimeJob(executor={self.executor_name!r}, "
            f"primitive_type={self.primitive_type!r}, job_id={job_id!r})"
        )

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve job result and log artifacts.

        This method is idempotent - calling it multiple times returns the same
        cached result and does not re-log artifacts.

        P0 Compliance: If job.result() raises an exception, an envelope with
        error details is still created and logged before re-raising.

        Returns
        -------
        Any
            PrimitiveResult object.

        Raises
        ------
        Exception
            Re-raises any exception from the underlying job.result().
        """
        if self._cached_result is not None:
            return self._cached_result

        # P0: Wrap job.result() to ensure envelope is created even on failure
        try:
            result = self.job.result(*args, **kwargs)
        except Exception as exc:
            # P0: Log failure envelope before re-raising
            self._log_failure(exc)
            raise

        self._cached_result = result

        if not self.should_log_results or self._finalized:
            return result

        self._finalized = True

        try:
            # Log raw result
            try:
                result_payload = to_jsonable(result)
            except Exception:
                result_payload = {"repr": repr(result)[:2000]}

            raw_result_ref = self.tracker.log_json(
                name="runtime.result",
                obj={
                    "sdk": "qiskit-ibm-runtime",
                    "primitive_type": self.primitive_type,
                    "result": result_payload,
                },
                role="results",
                kind="result.qiskit_runtime.output.json",
            )

            # Extract and log results based on primitive type
            if self.primitive_type == "sampler":
                self.result_snapshot = self._log_sampler_results(result, raw_result_ref)
            else:
                self.result_snapshot = self._log_estimator_results(
                    result, raw_result_ref
                )

            # Finalize envelope with result
            if self.envelope is not None and self.result_snapshot is not None:
                finalize_envelope_with_result(
                    self.tracker,
                    self.envelope,
                    self.result_snapshot,
                )

        except Exception as e:
            # Log error but don't fail - result retrieval should always succeed
            logger.warning(
                "Failed to log results for %s: %s",
                self.executor_name,
                e,
            )
            # Record error in tracker for visibility
            self.tracker.record.setdefault("warnings", []).append(
                {
                    "type": "result_logging_failed",
                    "message": str(e),
                    "executor_name": self.executor_name,
                    "primitive_type": self.primitive_type,
                }
            )

        return result

    def _log_failure(self, exc: Exception) -> None:
        """
        Log failure envelope when job.result() raises an exception.

        P0 Compliance: Ensures envelope is always created even on failures.

        Parameters
        ----------
        exc : Exception
            The exception that caused the failure.
        """
        if self._finalized or not self.should_log_results:
            return

        self._finalized = True

        try:
            # Create failure result snapshot
            self.result_snapshot = create_failure_result_snapshot(
                exception=exc,
                backend_name=self.executor_name,
                primitive_type=self.primitive_type,
            )

            # Finalize envelope with failure result
            if self.envelope is not None:
                finalize_envelope_with_result(
                    self.tracker,
                    self.envelope,
                    self.result_snapshot,
                )

            logger.debug(
                "Logged failure envelope for %s: %s",
                self.executor_name,
                type(exc).__name__,
            )
        except Exception as log_exc:
            logger.warning(
                "Failed to log failure envelope for %s: %s",
                self.executor_name,
                log_exc,
            )

    def _log_sampler_results(
        self,
        result: Any,
        raw_result_ref: Any,
    ) -> ResultSnapshot:
        """Log sampler results and return ResultSnapshot."""
        snapshot = build_sampler_result_snapshot(
            result,
            backend_name=self.executor_name,
            raw_result_ref=raw_result_ref,
        )

        if snapshot.items:
            counts_payload = {
                "experiments": [
                    {
                        "index": item.item_index,
                        "counts": item.counts.get("counts", {}) if item.counts else {},
                        "shots": item.counts.get("shots") if item.counts else None,
                    }
                    for item in snapshot.items
                    if item.counts
                ]
            }
            self.tracker.log_json(
                name="counts",
                obj=counts_payload,
                role="results",
                kind="result.counts.json",
            )

            num_experiments = len(snapshot.items)
            self.tracker.record["results"] = {
                "completed_at": utc_now_iso(),
                "backend_name": self.executor_name,
                "num_experiments": num_experiments,
                "primitive_type": self.primitive_type,
                "result_type": "counts",
            }
            logger.debug(
                "Logged sampler counts for %d experiments on %s",
                num_experiments,
                self.executor_name,
            )

        return snapshot

    def _log_estimator_results(
        self,
        result: Any,
        raw_result_ref: Any,
    ) -> ResultSnapshot:
        """Log estimator results and return ResultSnapshot."""
        snapshot = build_estimator_result_snapshot(
            result,
            backend_name=self.executor_name,
            raw_result_ref=raw_result_ref,
        )

        # Get experiments from snapshot metadata (estimator stores there)
        experiments_data = snapshot.metadata.get("experiments", [])
        if experiments_data:
            # Build payload for artifact
            experiments = []
            for exp_data in experiments_data:
                expectations = exp_data.get("expectations", [])
                if expectations:
                    evs = [e.get("value", 0.0) for e in expectations]
                    stds = [
                        e.get("std_error")
                        for e in expectations
                        if e.get("std_error") is not None
                    ]
                    exp_entry = {
                        "index": exp_data.get("index", 0),
                        "expectation_values": evs,
                    }
                    if stds:
                        exp_entry["standard_deviations"] = stds
                    experiments.append(exp_entry)

            if experiments:
                est_payload = {"experiments": experiments}
                self.tracker.log_json(
                    name="estimator_values",
                    obj=est_payload,
                    role="results",
                    kind="result.qiskit_runtime.estimator.json",
                )

                num_experiments = len(experiments)
                self.tracker.record["results"] = {
                    "completed_at": utc_now_iso(),
                    "backend_name": self.executor_name,
                    "num_experiments": num_experiments,
                    "primitive_type": self.primitive_type,
                    "result_type": "expectation",
                }
                logger.debug(
                    "Logged estimator values for %d experiments on %s",
                    num_experiments,
                    self.executor_name,
                )

        return snapshot

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped job."""
        return getattr(self.job, name)


@dataclass
class TrackedRuntimePrimitive:
    """
    Wrapper for Runtime primitive that tracks execution.

    This class wraps a Runtime Sampler/Estimator and logs circuits,
    execution parameters, and device snapshots following the UEC.

    Parameters
    ----------
    primitive : Any
        Original Runtime primitive instance.
    tracker : Run
        Tracker instance for logging.
    primitive_type : str
        Type of primitive ('sampler' or 'estimator').
    log_every_n : int
        Logging frequency:
        - 0 (default): Log first execution only.
        - N > 0: Log every Nth execution (e.g., 100 = log runs 1, 100, 200...).
        - -1: Log all executions (slowest, most complete).
    log_new_circuits : bool
        Auto-log new circuit structures (default True).
    stats_update_interval : int
        Update execution statistics every N executions (default 1000).

    Attributes
    ----------
    device_snapshot : DeviceSnapshot or None
        Cached device snapshot (created once per run).
    program_snapshot : ProgramSnapshot or None
        Current program snapshot (updated on circuit logging).
    execution_snapshot : ExecutionSnapshot or None
        Current execution snapshot (updated on run).
    """

    primitive: Any
    tracker: Run
    primitive_type: str
    log_every_n: int = 0
    log_new_circuits: bool = True
    stats_update_interval: int = 1000

    # Internal state (explicitly typed)
    _warned: bool = field(default=False, init=False, repr=False)
    _snapshot_logged: bool = field(default=False, init=False, repr=False)
    _execution_count: int = field(default=0, init=False, repr=False)
    _logged_execution_count: int = field(default=0, init=False, repr=False)
    _seen_circuit_hashes: set[str] = field(default_factory=set, init=False, repr=False)
    _logged_circuit_hashes: set[str] = field(
        default_factory=set, init=False, repr=False
    )

    # Cached device snapshot
    device_snapshot: DeviceSnapshot | None = field(default=None, init=False, repr=False)

    # UEC snapshots
    program_snapshot: ProgramSnapshot | None = field(
        default=None, init=False, repr=False
    )
    execution_snapshot: ExecutionSnapshot | None = field(
        default=None, init=False, repr=False
    )

    def __repr__(self) -> str:
        exec_name = get_backend_name(self.primitive)
        run_id = getattr(self.tracker, "run_id", "unknown")
        return (
            f"TrackedRuntimePrimitive(primitive_type={self.primitive_type!r}, "
            f"backend={exec_name!r}, run_id={run_id!r})"
        )

    def _log_device_snapshot(self, exec_name: str) -> DeviceSnapshot:
        """Log device snapshot (once per run) and return it."""
        if self.device_snapshot is not None:
            return self.device_snapshot

        # Create snapshot with tracker for raw_properties logging
        snapshot = create_device_snapshot(self.primitive, tracker=self.tracker)

        # Update tracker record with summary
        self.tracker.record["device_snapshot"] = {
            "sdk": "qiskit-ibm-runtime",
            "backend_name": exec_name,
            "backend_type": snapshot.backend_type,
            "provider": snapshot.provider,
            "captured_at": snapshot.captured_at,
            "num_qubits": snapshot.num_qubits,
            "calibration_summary": snapshot.get_calibration_summary(),
            "frontend": snapshot.frontend.to_dict() if snapshot.frontend else None,
        }

        self._snapshot_logged = True
        self.device_snapshot = snapshot
        logger.debug("Logged device snapshot for %s", exec_name)

        return snapshot

    def _log_circuits(
        self,
        circuits: list[Any],
        backend_name: str,
        circuit_hash: str | None,
    ) -> list[ProgramArtifact]:
        """Log circuits and return program artifacts."""
        artifacts: list[ProgramArtifact] = []
        sdk_versions = collect_sdk_versions()
        meta = {
            "backend_name": backend_name,
            "qiskit_version": sdk_versions.get("qiskit", "unknown"),
            "circuit_hash": circuit_hash,
        }

        # Serialize QPY
        try:
            qpy_data = _serializer.serialize(
                circuits if len(circuits) > 1 else circuits[0],
                CircuitFormat.QPY,
            )
            ref = self.tracker.log_bytes(
                kind="qiskit.qpy.circuits",
                data=qpy_data.as_bytes(),
                media_type="application/vnd.qiskit.qpy",
                role="program",
                meta=meta,
            )
            artifacts.append(
                ProgramArtifact(
                    ref=ref,
                    role=ProgramRole.LOGICAL,
                    format="qpy",
                    name="circuits",
                    index=0,
                )
            )
        except Exception:
            pass

        # Serialize QASM3 for each circuit
        for i, circuit in enumerate(circuits):
            try:
                qasm_data = _serializer.serialize(
                    circuit, CircuitFormat.OPENQASM3, index=i
                )
                qc_name = getattr(circuit, "name", None) or f"circuit_{i}"
                qasm3_item = {
                    "source": qasm_data.as_text(),
                    "name": f"circuit_{i}:{qc_name}",
                    "index": i,
                }
                oq3_result = self.tracker.log_openqasm3(
                    [qasm3_item],
                    name=f"circuit_{i}",
                    meta={**meta, "circuit_index": i, "circuit_name": qc_name},
                )
                items = oq3_result.get("items", [])
                if items:
                    ref = items[0].get("raw_ref")
                    if ref:
                        artifacts.append(
                            ProgramArtifact(
                                ref=ref,
                                role=ProgramRole.LOGICAL,
                                format="openqasm3",
                                name=f"circuit_{i}:{qc_name}",
                                index=i,
                            )
                        )
            except Exception:
                continue

        # Log circuit diagrams
        try:
            diagram_text = circuits_to_text(circuits)
            ref = self.tracker.log_bytes(
                kind="qiskit_runtime.circuits.diagram",
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
            pass

        return artifacts

    def _log_pubs_structure(self, pubs: Any) -> None:
        """Log PUB structure metadata."""
        pubs_struct = extract_pubs_structure(pubs, primitive_type=self.primitive_type)

        self.tracker.log_json(
            name="pubs",
            obj={
                "sdk": "qiskit-ibm-runtime",
                "primitive_type": self.primitive_type,
                "num_pubs": len(pubs_struct),
                "pubs": pubs_struct,
            },
            role="program",
            kind="qiskit_runtime.pubs.json",
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

    def run(self, pubs: Any, *args: Any, **kwargs: Any) -> TrackedRuntimeJob:
        """
        Execute PUBs and log artifacts following the UEC.

        devqubit control kwargs (stripped before calling primitive.run)
        --------------------------------------------------------------
        devqubit_transpilation_mode : {'auto','managed','manual'}, optional
            auto    : transpile only if needed (default)
            managed : always transpile
            manual  : never transpile (user is responsible)
        devqubit_transpilation_options : dict, optional
            Options forwarded to generate_preset_pass_manager.
        devqubit_transpilation_options_strict : bool, optional
            If True, unknown keys raise ValueError.
        devqubit_pass_manager : Any, optional
            User-supplied pass manager (advanced).
        devqubit_map_observables : bool, optional
            If True (default), map observables to circuit layout for estimator.
        devqubit_warn_on_auto_transpile : bool, optional
            If True (default), emit warning when auto transpilation is applied.
        devqubit_strict_isa_check : bool, optional
            If True (default), use strict ISA checking that trusts IBM's rejection.

        Parameters
        ----------
        pubs : Any
            Primitive Unified Blocs (circuits with parameters/observables).
        *args : Any
            Additional positional arguments passed to primitive.run().
        **kwargs : Any
            Additional keyword arguments passed to primitive.run().

        Returns
        -------
        TrackedRuntimeJob
            Wrapped job that tracks result retrieval.
        """
        exec_name = get_backend_name(self.primitive)
        submitted_at = utc_now_iso()

        # Materialize once
        pubs_list = iter_pubs(pubs)

        # Extract devqubit-only kwargs
        mode = kwargs.pop("devqubit_transpilation_mode", "auto")
        pm_options_dict = kwargs.pop("devqubit_transpilation_options", None)
        strict_opts = bool(kwargs.pop("devqubit_transpilation_options_strict", False))
        pass_manager = kwargs.pop("devqubit_pass_manager", None)
        map_obs = kwargs.pop("devqubit_map_observables", True)
        warn_on_auto = kwargs.pop("devqubit_warn_on_auto_transpile", True)
        strict_isa = kwargs.pop("devqubit_strict_isa_check", True)

        # Extract circuits from PUBs
        circuits = extract_circuits_from_pubs(pubs_list)

        # Increment execution counter
        self._execution_count += 1
        exec_count = self._execution_count

        # Compute circuit hash
        circuit_hash = compute_circuit_hash(circuits)
        is_new_circuit = circuit_hash and circuit_hash not in self._seen_circuit_hashes
        if circuit_hash:
            self._seen_circuit_hashes.add(circuit_hash)

        # Determine what to log
        should_log_structure = False
        should_log_results = False

        if self.log_every_n == -1:
            should_log_structure = circuit_hash not in self._logged_circuit_hashes
            should_log_results = True
        elif exec_count == 1:
            should_log_structure = True
            should_log_results = True
        elif self.log_new_circuits and is_new_circuit:
            should_log_structure = True
            should_log_results = True
        elif self.log_every_n > 0 and exec_count % self.log_every_n == 0:
            should_log_results = True

        # Transpilation policy
        options = TranspilationOptions.from_dict(
            pm_options_dict,
            strict=strict_opts,
            warn_unknown=True,
        )
        cfg = TranspilationConfig(
            mode=mode,
            options=options,
            pass_manager=pass_manager,
            map_observables=map_obs,
            strict_isa_check=strict_isa,
        )

        pubs_to_run, tmeta = prepare_pubs_for_primitive(
            pubs_list,
            primitive=self.primitive,
            primitive_type=self.primitive_type,
            config=cfg,
        )

        # Fast path: nothing to log
        if not should_log_structure and not should_log_results:
            job = self.primitive.run(pubs_to_run, *args, **kwargs)

            if (
                self.stats_update_interval > 0
                and exec_count % self.stats_update_interval == 0
            ):
                self._update_stats()

            return TrackedRuntimeJob(
                job=job,
                tracker=self.tracker,
                executor_name=exec_name,
                primitive_type=self.primitive_type,
                should_log_results=False,
                envelope=None,
            )

        # Tags
        self.tracker.set_tag("provider", "qiskit-ibm-runtime")
        self.tracker.set_tag("adapter", "qiskit-runtime")
        self.tracker.set_tag("backend_name", exec_name)
        self.tracker.set_tag("primitive_type", self.primitive_type)

        # Device snapshot (once per run)
        device_snapshot = self._log_device_snapshot(exec_name)

        # Log structure (produces ProgramSnapshot)
        program_artifacts: list[ProgramArtifact] = []
        physical_artifacts: list[ProgramArtifact] = []

        # Compute transpiled circuits once if needed
        transpiled_circuits: list[Any] = []
        if tmeta.get("transpiled_by_devqubit"):
            transpiled_circuits = extract_circuits_from_pubs(pubs_to_run)

        if should_log_structure:
            # Log original circuits/pubs
            if circuits:
                program_artifacts = self._log_circuits(
                    circuits, exec_name, circuit_hash
                )
            self._log_pubs_structure(pubs_list)

            # If devqubit transpiled, log transpiled circuits as physical
            if transpiled_circuits:
                transpiled_hash = compute_circuit_hash(transpiled_circuits)
                try:
                    qpy_data = _serializer.serialize(
                        (
                            transpiled_circuits
                            if len(transpiled_circuits) > 1
                            else transpiled_circuits[0]
                        ),
                        CircuitFormat.QPY,
                    )
                    ref = self.tracker.log_bytes(
                        kind="qiskit.qpy.circuits.transpiled",
                        data=qpy_data.as_bytes(),
                        media_type="application/vnd.qiskit.qpy",
                        role="program",
                        meta={
                            "backend_name": exec_name,
                            "transpiled_by": "devqubit",
                            "circuit_hash": transpiled_hash,
                        },
                    )
                    physical_artifacts.append(
                        ProgramArtifact(
                            ref=ref,
                            role=ProgramRole.PHYSICAL,
                            format="qpy",
                            name="circuits_transpiled",
                            index=0,
                        )
                    )
                except Exception:
                    pass

            if circuit_hash:
                self._logged_circuit_hashes.add(circuit_hash)

            self.tracker.record["backend"] = {
                "name": exec_name,
                "type": self.primitive.__class__.__name__,
                "provider": "qiskit-ibm-runtime",
                "primitive_type": self.primitive_type,
            }

            self._logged_execution_count += 1

        # Build ProgramSnapshot
        executed_hash = (
            compute_circuit_hash(transpiled_circuits) if transpiled_circuits else None
        )

        self.program_snapshot = ProgramSnapshot(
            logical=program_artifacts,
            physical=physical_artifacts,
            program_hash=circuit_hash,
            executed_hash=executed_hash,
            num_circuits=len(circuits),
        )

        # Build TranspilationInfo
        transpilation_mode = _map_transpilation_mode(mode)
        transpiled_by = "devqubit" if tmeta.get("transpiled_by_devqubit") else "user"
        if transpilation_mode == TranspilationMode.MANUAL:
            transpiled_by = "user"

        transpilation_info = TranspilationInfo(
            mode=transpilation_mode,
            transpiled_by=transpiled_by,
        )

        # Build ExecutionSnapshot
        self.execution_snapshot = ExecutionSnapshot(
            submitted_at=submitted_at,
            shots=None,  # Runtime primitives handle shots internally
            job_ids=[],  # Filled after submission
            execution_count=exec_count,
            transpilation=transpilation_info,
            options={
                "args": to_jsonable(list(args)),
                "kwargs": to_jsonable(kwargs),
                "transpilation_needed": tmeta.get("transpilation_needed"),
                "transpilation_reason": tmeta.get("transpilation_reason"),
                "observables_layout_mapped": tmeta.get("observables_layout_mapped"),
                # Store transpilation details here instead
                "optimization_level": options.optimization_level,
                "layout_method": options.layout_method,
                "routing_method": options.routing_method,
                "seed_transpiler": options.seed_transpiler,
            },
            sdk="qiskit-ibm-runtime",
        )

        # Update tracker record
        self.tracker.record["execute"] = {
            "submitted_at": submitted_at,
            "backend_name": exec_name,
            "sdk": "qiskit-ibm-runtime",
            "primitive_type": self.primitive_type,
            "num_pubs": len(pubs_list),
            "execution_count": exec_count,
            "program_hash": circuit_hash,
            "executed_hash": executed_hash,
            "args": to_jsonable(list(args)),
            "kwargs": to_jsonable(kwargs),
        }
        self.tracker.record["execute"].update(to_jsonable(tmeta))

        self.tracker.record["transpilation_mode"] = tmeta.get(
            "transpilation_mode", mode
        )
        self.tracker.record["transpilation"] = transpilation_info.to_dict()

        # Store transpilation options
        transpilation_opts = tmeta.get("transpilation_options") or {}
        if transpilation_opts:
            self.tracker.record["transpilation_options"] = transpilation_opts

        # Warnings
        if (
            warn_on_auto
            and not self._warned
            and mode == "auto"
            and tmeta.get("transpiled_by_devqubit") is True
            and tmeta.get("transpilation_needed") is True
            and pass_manager is None
            and not strict_opts
        ):
            warnings.warn(
                "devqubit: circuits were not ISA-compatible with the backend target, so "
                "default transpilation (preset pass manager) was applied. "
                "If you want full control, transpile manually and set "
                "devqubit_transpilation_mode='manual', or pass "
                "devqubit_transpilation_options / devqubit_pass_manager.",
                UserWarning,
                stacklevel=2,
            )
            self._warned = True

        if (
            warn_on_auto
            and not self._warned
            and mode == "manual"
            and tmeta.get("transpilation_needed") is True
        ):
            warnings.warn(
                "devqubit: devqubit_transpilation_mode='manual' was requested, but circuits "
                "do not look ISA-compatible with the backend target. The primitive may fail. "
                "Consider manual transpilation or use mode='auto'.",
                UserWarning,
                stacklevel=2,
            )
            self._warned = True

        logger.debug(
            "Submitting %d PUBs to %s (transpilation_mode=%s)",
            len(pubs_to_run),
            exec_name,
            mode,
        )

        # Execute on primitive
        job = self.primitive.run(pubs_to_run, *args, **kwargs)

        # Log job ID
        jid = extract_job_id(job)
        if jid:
            self.tracker.record["execute"]["job_ids"] = [jid]
            self.execution_snapshot.job_ids = [jid]
            logger.debug("Job ID: %s", jid)

        # Create envelope (will be finalized when result() is called)
        envelope: ExecutionEnvelope | None = None
        if should_log_results and device_snapshot is not None:
            sdk_versions = collect_sdk_versions()

            # Create ProducerInfo for SDK stack tracking
            producer = ProducerInfo.create(
                adapter="devqubit-qiskit-runtime",
                adapter_version=_ADAPTER_VERSION,
                sdk="qiskit-ibm-runtime",
                sdk_version=sdk_versions.get("qiskit_ibm_runtime", "unknown"),
                frontends=["qiskit-ibm-runtime"],
            )

            # Create pending result (will be updated when result() completes)
            pending_result = ResultSnapshot(
                success=False,
                status="failed",  # Will be updated when result() completes
                items=[],
                metadata={"state": "pending"},
            )

            envelope = ExecutionEnvelope(
                envelope_id=uuid.uuid4().hex[:26],
                created_at=utc_now_iso(),
                producer=producer,
                result=pending_result,
                device=device_snapshot,
                program=self.program_snapshot,
                execution=self.execution_snapshot,
            )

        # Update stats
        self._update_stats()

        return TrackedRuntimeJob(
            job=job,
            tracker=self.tracker,
            executor_name=exec_name,
            primitive_type=self.primitive_type,
            should_log_results=should_log_results,
            envelope=envelope,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped primitive."""
        return getattr(self.primitive, name)


class QiskitRuntimeAdapter:
    """
    Adapter for integrating Qiskit IBM Runtime primitives with devqubit.

    This adapter wraps Runtime Sampler/Estimator primitives to automatically
    log circuits, execution parameters, configurations, and results following
    the devqubit Uniform Execution Contract (UEC).

    Attributes
    ----------
    name : str
        Adapter identifier ("qiskit-runtime").

    Notes
    -----
    This adapter only supports Runtime V2 primitives (SamplerV2, EstimatorV2).
    For direct backend usage (backend.run()), use the ``qiskit`` adapter instead.
    """

    name: str = "qiskit-runtime"

    def supports_executor(self, executor: Any) -> bool:
        """
        Check if executor is a supported Runtime primitive.

        Parameters
        ----------
        executor : Any
            Potential executor instance.

        Returns
        -------
        bool
            True if executor is a Runtime Sampler or Estimator.
        """
        return is_runtime_primitive(executor)

    def describe_executor(self, executor: Any) -> dict[str, Any]:
        """
        Create a description of the primitive.

        Parameters
        ----------
        executor : Any
            Runtime primitive instance.

        Returns
        -------
        dict
            Primitive description.
        """
        return {
            "name": get_backend_name(executor),
            "type": executor.__class__.__name__,
            "provider": "qiskit-ibm-runtime",
            "primitive_type": get_primitive_type(executor),
        }

    def wrap_executor(
        self,
        executor: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
    ) -> TrackedRuntimePrimitive:
        """
        Wrap a primitive with tracking capabilities.

        Parameters
        ----------
        executor : Any
            Runtime primitive to wrap.
        tracker : Run
            Tracker instance for logging.
        log_every_n : int
            Logging frequency:
            - 0 (default): Log first execution only.
            - N > 0: Log every Nth execution (e.g., 100 = log runs 1, 100, 200...).
            - -1: Log all executions (slowest, most complete).
        log_new_circuits : bool
            Auto-log new circuit structures (default True).
        stats_update_interval : int
            Update stats every N executions (default 1000).

        Returns
        -------
        TrackedRuntimePrimitive
            Wrapped primitive with tracking.
        """
        return TrackedRuntimePrimitive(
            primitive=executor,
            tracker=tracker,
            primitive_type=get_primitive_type(executor),
            log_every_n=log_every_n,
            log_new_circuits=log_new_circuits,
            stats_update_interval=stats_update_interval,
        )
