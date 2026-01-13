# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Replay quantum experiments from bundles or stored runs.

This module provides functionality to re-execute quantum circuits from
previously tracked runs. Replay is restricted to same-SDK execution using
native serialization formats only.

.. warning::

    Replay is EXPERIMENTAL and requires explicit acknowledgement via
    ``ack_experimental=True``. Results may not be fully reproducible
    due to SDK-specific randomness and transpilation differences.

Supported formats (native SDK artifacts only):
    - Qiskit: QPY
    - Braket: JAQCD
    - Cirq: Cirq JSON
    - PennyLane: Tape JSON

OpenQASM is NOT supported - replay requires native SDK artifacts to ensure
the exact same program representation is used.

Examples
--------
>>> from devqubit_engine.bundle.replay import replay, list_available_backends

>>> # Replay from a bundle file (must acknowledge experimental status)
>>> result = replay("my_run.zip", ack_experimental=True)
>>> if result.ok:
...     print(f"Counts: {result.counts}")

>>> # Replay with seed for best-effort reproducibility
>>> result = replay("01HX...", ack_experimental=True, seed=42, shots=2048)

>>> # List available backends
>>> backends = list_available_backends()
>>> print(backends)
{'qiskit': ['aer_simulator'], 'cirq': ['simulator', 'density_matrix'], ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devqubit_engine.bundle.reader import Bundle, is_bundle_path
from devqubit_engine.circuit.extractors import detect_sdk, extract_circuit
from devqubit_engine.circuit.models import (
    SDK,
    CircuitData,
    CircuitFormat,
    LoadedCircuit,
)
from devqubit_engine.circuit.registry import get_loader, list_available
from devqubit_engine.core.record import RunRecord
from devqubit_engine.core.types import ArtifactRef
from devqubit_engine.storage.factory import create_registry, create_store
from devqubit_engine.storage.protocols import ObjectStoreProtocol, RegistryProtocol


logger = logging.getLogger(__name__)


# Supported native formats for replay (no OpenQASM)
_SUPPORTED_REPLAY_FORMATS = frozenset(
    {
        CircuitFormat.QPY,
        CircuitFormat.JAQCD,
        CircuitFormat.CIRQ_JSON,
        CircuitFormat.TAPE_JSON,
    }
)


@dataclass
class ReplayResult:
    """
    Result of a replay operation.

    Attributes
    ----------
    ok : bool
        True if replay executed successfully.
    original_run_id : str
        ID of the original run that was replayed.
    replay_run_id : str or None
        ID of the saved replay run (if save_run=True).
    counts : dict
        Measurement counts from replay execution.
    circuit_source : str
        Format used for loading (e.g., "qpy", "jaqcd").
    backend_used : str
        Simulator backend that executed the replay.
    original_backend : str
        Backend from the original run.
    original_adapter : str
        Adapter from the original run.
    shots : int
        Number of shots executed.
    seed : int or None
        Seed used for replay (if provided).
    message : str
        Human-readable status message.
    errors : list of str
        Errors or warnings encountered during replay.
    """

    ok: bool
    original_run_id: str
    replay_run_id: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    circuit_source: str = ""
    backend_used: str = ""
    original_backend: str = ""
    original_adapter: str = ""
    shots: int = 0
    seed: int | None = None
    message: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the replay result.
        """
        return {
            "ok": self.ok,
            "original_run_id": self.original_run_id,
            "replay_run_id": self.replay_run_id,
            "counts": self.counts,
            "circuit_source": self.circuit_source,
            "backend_used": self.backend_used,
            "original_backend": self.original_backend,
            "original_adapter": self.original_adapter,
            "shots": self.shots,
            "seed": self.seed,
            "message": self.message,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        status = "ok" if self.ok else "failed"
        return (
            f"ReplayResult({status}, run={self.original_run_id!r}, shots={self.shots})"
        )


class _BundleStore:
    """Read-only store wrapper for bundle files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get_bytes(self, digest: str) -> bytes:
        """Get bytes from bundle store."""
        with Bundle(self.path) as bundle:
            return bundle.store.get_bytes(digest)

    def exists(self, digest: str) -> bool:
        """Check if digest exists in bundle."""
        with Bundle(self.path) as bundle:
            return bundle.store.exists(digest)


_DEFAULT_SIMULATORS: dict[SDK, str] = {
    SDK.QISKIT: "aer_simulator",
    SDK.BRAKET: "local",
    SDK.CIRQ: "simulator",
    SDK.PENNYLANE: "default.qubit",
}


def replay(
    ref: str | Path,
    *,
    backend: str | None = None,
    root: Path | None = None,
    registry: RegistryProtocol | None = None,
    store: ObjectStoreProtocol | None = None,
    shots: int | None = None,
    seed: int | None = None,
    save_run: bool = False,
    project: str | None = None,
    ack_experimental: bool = False,
) -> ReplayResult:
    """
    Replay a quantum experiment from a bundle or run ID.

    Re-executes the circuit on a simulator backend. The circuit is loaded
    from native format artifacts only (QPY, JAQCD, Cirq JSON, Tape JSON).

    .. warning::

        Replay is EXPERIMENTAL. You must pass ``ack_experimental=True``
        to acknowledge this. Results may not be fully reproducible.

    Parameters
    ----------
    ref : str or Path
        Bundle path (.zip file) or run ID.
    backend : str, optional
        Simulator backend name. Uses SDK default if not specified.
    root : Path, optional
        Workspace root directory for loading runs.
    registry : RegistryProtocol, optional
        Registry for loading runs. Uses default if not provided.
    store : ObjectStoreProtocol, optional
        Object store for loading artifacts. Uses default if not provided.
    shots : int, optional
        Override shot count from original run.
    seed : int, optional
        Seed for best-effort reproducibility. Support varies by SDK:
        - Qiskit: seed_transpiler + seed_simulator (best-effort)
        - Cirq: fully supported
        - PennyLane: supported for default.qubit
        - Braket: not supported (warning logged)
    save_run : bool, optional
        If True, save the replay as a new tracked run. Default is False.
    project : str, optional
        Project name for saved replay run. Uses original project if not specified.
    ack_experimental : bool, optional
        Must be True to acknowledge experimental status. Default is False.

    Returns
    -------
    ReplayResult
        Result containing counts, metadata, and any errors.

    Raises
    ------
    None
        All errors are captured in ReplayResult.errors.

    Notes
    -----
    Replay does NOT support:
    - OpenQASM formats (native SDK artifacts required)
    - Cross-SDK replay (must use original SDK)
    - Runs with unknown SDK
    - Automatic program modification (e.g., adding measurements)

    The seed parameter provides best-effort reproducibility but is not
    guaranteed due to SDK-specific behavior and transpilation randomness.
    """
    errors: list[str] = []

    # Gate: require explicit acknowledgement
    if not ack_experimental:
        return ReplayResult(
            ok=False,
            original_run_id=str(ref),
            message=(
                "Replay is EXPERIMENTAL and requires explicit acknowledgement. "
                "Pass ack_experimental=True to proceed."
            ),
            errors=[
                "Replay is experimental: results may not be fully reproducible. "
                "Set ack_experimental=True to acknowledge and proceed."
            ],
        )

    logger.warning(
        "EXPERIMENTAL: devqubit replay is best-effort and may not be fully reproducible."
    )
    logger.info("Replaying from %s", ref)

    # Load run record and store
    try:
        record, orig_store = _load_from_ref(ref, root, registry, store)
    except Exception as e:
        logger.error("Failed to load run: %s", e)
        return ReplayResult(
            ok=False,
            original_run_id=str(ref),
            message=f"Failed to load run: {e}",
            errors=[str(e)],
        )

    original_run_id = record.run_id
    original_backend = record.backend_name or "unknown"
    original_adapter = record.adapter or "unknown"

    # SDK inferred from run metadata
    sdk_hint = detect_sdk(record)

    # Gate: forbid unknown SDK
    if sdk_hint == SDK.UNKNOWN:
        return ReplayResult(
            ok=False,
            original_run_id=original_run_id,
            original_backend=original_backend,
            original_adapter=original_adapter,
            message="Cannot replay: original SDK is unknown.",
            errors=[
                "SDK.UNKNOWN: replay requires the original SDK to be identifiable. "
                "Ensure the run was tracked with a known adapter (qiskit/braket/cirq/pennylane)."
            ],
        )

    # Extract execution params and clamp shots (including user override)
    exec_params = _get_execution_params(record)
    run_shots_raw = shots if shots is not None else exec_params.get("shots", 1024)
    try:
        run_shots = max(1, int(run_shots_raw))
    except (ValueError, TypeError):
        run_shots = 1024

    logger.debug(
        "Original run: id=%s, adapter=%s, backend=%s, sdk=%s",
        original_run_id,
        original_adapter,
        original_backend,
        sdk_hint.value,
    )

    # Extract circuit from artifacts
    circuit_data = extract_circuit(record, orig_store)
    if circuit_data is None:
        return ReplayResult(
            ok=False,
            original_run_id=original_run_id,
            original_backend=original_backend,
            original_adapter=original_adapter,
            shots=run_shots,
            seed=seed,
            message=f"No circuit artifact found for SDK '{sdk_hint.value}'.",
            errors=[
                "No native circuit artifact found in run. "
                "Replay requires QPY (Qiskit), JAQCD (Braket), Cirq JSON, or Tape JSON."
            ],
        )

    # Gate: forbid OpenQASM formats
    if circuit_data.format in (CircuitFormat.OPENQASM2, CircuitFormat.OPENQASM3):
        return ReplayResult(
            ok=False,
            original_run_id=original_run_id,
            original_backend=original_backend,
            original_adapter=original_adapter,
            circuit_source=circuit_data.format.value,
            shots=run_shots,
            seed=seed,
            message="Replay does not support OpenQASM formats.",
            errors=[
                f"Unsupported circuit format: {circuit_data.format.value}. "
                "Replay requires native SDK artifacts (QPY/JAQCD/Cirq JSON/Tape JSON). "
                "OpenQASM is not supported to ensure exact program representation."
            ],
        )

    # Gate: check format is in supported list
    if circuit_data.format not in _SUPPORTED_REPLAY_FORMATS:
        return ReplayResult(
            ok=False,
            original_run_id=original_run_id,
            original_backend=original_backend,
            original_adapter=original_adapter,
            circuit_source=circuit_data.format.value,
            shots=run_shots,
            seed=seed,
            message=f"Unsupported circuit format: {circuit_data.format.value}.",
            errors=[
                f"Format {circuit_data.format.value} is not supported for replay. "
                f"Supported formats: {[f.value for f in _SUPPORTED_REPLAY_FORMATS]}."
            ],
        )

    # Force SDK consistency
    target_sdk = sdk_hint
    if circuit_data.sdk != target_sdk:
        circuit_data = CircuitData(
            data=circuit_data.data,
            format=circuit_data.format,
            sdk=target_sdk,
            name=circuit_data.name,
            index=circuit_data.index,
            metadata=dict(circuit_data.metadata),
        )

    # Load circuit (strict - no fallback)
    loaded = _load_circuit_strict(circuit_data, target_sdk)
    if loaded is None:
        available = list_available()
        return ReplayResult(
            ok=False,
            original_run_id=original_run_id,
            original_backend=original_backend,
            original_adapter=original_adapter,
            circuit_source=circuit_data.format.value,
            shots=run_shots,
            seed=seed,
            message=f"Failed to load circuit for SDK '{target_sdk.value}'.",
            errors=[
                f"No loader for format={circuit_data.format.value} under sdk={target_sdk.value}. "
                f"Installed loaders: {available.get('loaders', [])}. "
                f"Install the matching adapter package (e.g., devqubit-{target_sdk.value})."
            ],
        )

    # Determine backend
    backend_name = backend or _DEFAULT_SIMULATORS.get(loaded.sdk, "aer_simulator")

    logger.info(
        "Executing replay: sdk=%s, backend=%s, shots=%d, seed=%s",
        loaded.sdk.value,
        backend_name,
        run_shots,
        seed,
    )

    # Execute on simulator
    try:
        replay_counts = _run_circuit(loaded, run_shots, backend_name, seed, errors)
    except Exception as e:
        logger.error("Execution failed: %s", e)
        return ReplayResult(
            ok=False,
            original_run_id=original_run_id,
            original_backend=original_backend,
            original_adapter=original_adapter,
            backend_used=backend_name,
            circuit_source=circuit_data.format.value,
            shots=run_shots,
            seed=seed,
            message=f"Execution failed: {e}",
            errors=[str(e)],
        )

    # Save replay run if requested
    replay_run_id = None
    if save_run:
        replay_run_id = _save_replay_run(
            record=record,
            circuit_data=circuit_data,
            replay_counts=replay_counts,
            backend_name=backend_name,
            run_shots=run_shots,
            seed=seed,
            project=project,
            root=root,
            registry=registry,
            store=store,
            errors=errors,
        )

    logger.info(
        "Replay complete: %d unique outcomes from %d shots",
        len(replay_counts),
        run_shots,
    )

    return ReplayResult(
        ok=True,
        original_run_id=original_run_id,
        replay_run_id=replay_run_id,
        counts=replay_counts,
        circuit_source=circuit_data.format.value,
        backend_used=backend_name,
        original_backend=original_backend,
        original_adapter=original_adapter,
        shots=run_shots,
        seed=seed,
        message=(
            f"EXPERIMENTAL: Replayed {original_run_id} on {backend_name} "
            f"({run_shots} shots, seed={seed})"
        ),
        errors=errors,
    )


def list_available_backends() -> dict[str, list[str]]:
    """
    List available simulator backends by SDK.

    Checks which SDK packages are installed and returns the available
    simulator backends for each.

    Returns
    -------
    dict
        Mapping of SDK name to list of available simulator backend names.
    """
    backends: dict[str, list[str]] = {}

    # Qiskit
    try:
        import qiskit  # noqa: F401

        try:
            from qiskit_aer import AerSimulator  # noqa: F401

            backends["qiskit"] = ["aer_simulator"]
        except ImportError:
            try:
                from qiskit.providers.aer import AerSimulator  # noqa: F401

                backends["qiskit"] = ["aer_simulator"]
            except ImportError:
                pass
    except ImportError:
        pass

    # Braket
    try:
        from braket.devices import LocalSimulator  # noqa: F401

        backends["braket"] = ["local", "density_matrix"]
    except ImportError:
        pass

    # Cirq
    try:
        import cirq  # noqa: F401

        backends["cirq"] = ["simulator", "density_matrix"]
    except ImportError:
        pass

    # PennyLane
    try:
        import pennylane  # noqa: F401

        pl_backends = ["default.qubit"]
        try:
            import pennylane_lightning  # noqa: F401

            pl_backends.append("lightning.qubit")
        except ImportError:
            pass
        backends["pennylane"] = pl_backends
    except ImportError:
        pass

    return backends


def _load_from_ref(
    ref: str | Path,
    root: Path | None = None,
    registry: RegistryProtocol | None = None,
    store: ObjectStoreProtocol | None = None,
) -> tuple[RunRecord, ObjectStoreProtocol]:
    """Load run record and store from bundle or registry."""
    if is_bundle_path(ref):
        bundle_path = Path(ref)
        with Bundle(bundle_path) as bundle:
            record_dict = bundle.run_record
            artifacts = [
                ArtifactRef.from_dict(a)
                for a in record_dict.get("artifacts", [])
                if isinstance(a, dict)
            ]
            record = RunRecord(record=record_dict, artifacts=artifacts)
        return record, _BundleStore(bundle_path)

    # Load from registry
    reg = registry or create_registry(f"file://{root}" if root else None)
    record = reg.load(str(ref))
    st = store or create_store(f"file://{root}/objects" if root else None)
    return record, st


def _get_execution_params(record: RunRecord) -> dict[str, Any]:
    """Extract execution parameters from run record."""
    params: dict[str, Any] = {}

    # Try data.params.shots
    data = record.record.get("data", {}) or {}
    if isinstance(data, dict):
        run_params = data.get("params", {}) or {}
        if isinstance(run_params, dict) and "shots" in run_params:
            try:
                params["shots"] = max(1, int(run_params["shots"]))
            except (ValueError, TypeError):
                pass

    # Try execute.kwargs.shots
    execute = record.record.get("execute", {}) or {}
    if isinstance(execute, dict):
        kwargs = execute.get("kwargs", {}) or {}
        if isinstance(kwargs, dict) and "shots" in kwargs:
            try:
                params["shots"] = max(1, int(kwargs["shots"]))
            except (ValueError, TypeError):
                pass

    params.setdefault("shots", 1024)
    return params


def _load_circuit_strict(
    circuit_data: CircuitData,
    target_sdk: SDK,
) -> LoadedCircuit | None:
    """
    Load circuit strictly for the target SDK (no fallback).

    Parameters
    ----------
    circuit_data : CircuitData
        Circuit data to load.
    target_sdk : SDK
        Target SDK (must not be UNKNOWN).

    Returns
    -------
    LoadedCircuit or None
        Loaded circuit, or None if loading failed.
    """
    if target_sdk == SDK.UNKNOWN:
        return None

    forced = circuit_data
    if circuit_data.sdk != target_sdk:
        forced = CircuitData(
            data=circuit_data.data,
            format=circuit_data.format,
            sdk=target_sdk,
            name=circuit_data.name,
            index=circuit_data.index,
            metadata=dict(circuit_data.metadata),
        )

    return _try_load_circuit(forced)


def _try_load_circuit(data: CircuitData) -> LoadedCircuit | None:
    """Attempt to load circuit data, returning None on failure."""
    try:
        loader = get_loader(data.sdk)
        if (
            hasattr(loader, "supported_formats")
            and data.format not in loader.supported_formats
        ):
            return None
        return loader.load(data)
    except Exception as e:
        logger.debug("Failed to load circuit: %s", e)
        return None


def _run_circuit(
    loaded: LoadedCircuit,
    shots: int,
    backend_name: str,
    seed: int | None,
    errors: list[str],
) -> dict[str, int]:
    """
    Run loaded circuit on simulator.

    Parameters
    ----------
    loaded : LoadedCircuit
        Loaded circuit with SDK information.
    shots : int
        Number of shots to execute.
    backend_name : str
        Simulator backend name.
    seed : int or None
        Seed for reproducibility (support varies by SDK).
    errors : list of str
        List to append warnings to.

    Returns
    -------
    dict
        Measurement counts.
    """
    if loaded.sdk == SDK.QISKIT:
        return _run_qiskit(loaded.circuit, shots, backend_name, seed)
    if loaded.sdk == SDK.BRAKET:
        return _run_braket(loaded.circuit, shots, backend_name, seed, errors)
    if loaded.sdk == SDK.CIRQ:
        return _run_cirq(loaded.circuit, shots, backend_name, seed)
    if loaded.sdk == SDK.PENNYLANE:
        return _run_pennylane(loaded.circuit, shots, backend_name, seed)
    raise ValueError(f"No runner for SDK: {loaded.sdk}")


def _run_qiskit(
    circuit: Any,
    shots: int,
    backend_name: str,
    seed: int | None,
) -> dict[str, int]:
    """
    Run Qiskit circuit on Aer simulator.

    Seed support is best-effort via seed_transpiler and seed_simulator.
    """
    from qiskit import transpile

    if backend_name not in ("aer_simulator", "aer", "qasm_simulator"):
        raise ValueError(
            f"Unsupported Qiskit backend: {backend_name}. "
            "Only simulators are supported for replay. Use 'aer_simulator'."
        )

    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
    except ImportError:
        from qiskit.providers.aer import AerSimulator  # type: ignore[import-not-found]

        backend = AerSimulator()

    # Best-effort seed for simulator
    if seed is not None:
        try:
            backend.set_options(seed_simulator=int(seed))
        except Exception:
            logger.debug("Could not set seed_simulator on AerSimulator")

    # Transpile with seed for reproducibility
    transpiled = transpile(
        circuit,
        backend=backend,
        seed_transpiler=seed,
    )
    job = backend.run(transpiled, shots=shots)
    return dict(job.result().get_counts())


def _run_braket(
    circuit: Any,
    shots: int,
    backend_name: str,
    seed: int | None,
    errors: list[str],
) -> dict[str, int]:
    """
    Run Braket circuit on local simulator.

    Notes
    -----
    - For Braket/JAQCD we do NOT rely on explicit `Measure` instructions.
      JAQCD does not support serializing `Measure`; counts are obtained
      from the simulator's `measurement_counts` for shots>0.
    - Program modification is NOT performed.
    """
    from braket.devices import LocalSimulator

    if seed is not None:
        msg = "Braket LocalSimulator does not support seed; results may vary."
        logger.warning(msg)
        errors.append(msg)

    if backend_name in ("local", "sv1", "state_vector"):
        device = LocalSimulator(backend="braket_sv")
    elif backend_name in ("dm", "density_matrix"):
        device = LocalSimulator(backend="braket_dm")
    else:
        raise ValueError(
            f"Unsupported Braket backend: {backend_name}. "
            "Only simulators are supported for replay. Use 'local' or 'density_matrix'."
        )

    task = device.run(circuit, shots=shots)
    return dict(task.result().measurement_counts)


def _run_cirq(
    circuit: Any,
    shots: int,
    backend_name: str,
    seed: int | None,
) -> dict[str, int]:
    """
    Run Cirq circuit on simulator.

    Cirq fully supports seed for reproducibility.
    """
    import cirq

    if backend_name == "simulator":
        sim = cirq.Simulator(seed=seed)
    elif backend_name == "density_matrix":
        sim = cirq.DensityMatrixSimulator(seed=seed)
    else:
        raise ValueError(
            f"Unsupported Cirq backend: {backend_name}. "
            "Only simulators are supported for replay. Use 'simulator' or 'density_matrix'."
        )

    result = sim.run(circuit, repetitions=shots)

    meas = getattr(result, "measurements", {}) or {}
    keys = sorted(meas.keys())
    if not keys:
        raise ValueError(
            "Cirq circuit has no measurements; cannot produce counts for replay."
        )

    # Prefer Cirq's histogram helper when present; fallback to manual.
    if hasattr(result, "multi_measurement_histogram"):
        hist = result.multi_measurement_histogram(keys=keys)

        widths = [int(meas[k].shape[1]) for k in keys]

        def _to_bitstring(val: int, width: int) -> str:
            v = int(val)
            return format(v, f"0{width}b") if width > 0 else ""

        counts: dict[str, int] = {}
        for outcome, count in hist.items():
            if not isinstance(outcome, tuple):
                outcome = (outcome,)
            bitstr = "".join(_to_bitstring(v, w) for v, w in zip(outcome, widths))
            counts[bitstr] = counts.get(bitstr, 0) + int(count)
        return counts

    # Manual fallback (older Cirq)
    counts = {}
    for i in range(shots):
        bits: list[int] = []
        for k in keys:
            row = meas[k][i]
            bits.extend(int(b) for b in row)
        key = "".join(str(b) for b in bits)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _run_pennylane(
    circuit: Any,
    shots: int,
    backend_name: str,
    seed: int | None,
) -> dict[str, int]:
    """
    Run PennyLane circuit on simulator.

    default.qubit supports seed; lightning.qubit support varies.
    """
    import pennylane as qml

    if backend_name not in ("default.qubit", "default", "lightning.qubit"):
        raise ValueError(
            f"Unsupported PennyLane backend: {backend_name}. "
            "Only simulators are supported for replay. Use 'default.qubit' or 'lightning.qubit'."
        )

    # Get tape from circuit
    tape = None
    if hasattr(circuit, "tape"):
        tape = circuit.tape
    elif hasattr(circuit, "operations"):
        tape = circuit

    if tape is None:
        raise ValueError("Expected a QuantumTape-like object for PennyLane replay")

    num_wires = len(tape.wires)

    if backend_name in ("default.qubit", "default"):
        dev = qml.device("default.qubit", wires=num_wires, shots=shots, seed=seed)
    else:
        # lightning.qubit: try passing seed, fallback if unsupported
        try:
            dev = qml.device("lightning.qubit", wires=num_wires, shots=shots, seed=seed)
        except TypeError:
            logger.debug("lightning.qubit does not support seed parameter")
            dev = qml.device("lightning.qubit", wires=num_wires, shots=shots)

    @qml.qnode(dev)
    def run_tape():
        for op in tape.operations:
            qml.apply(op)
        return qml.counts()

    result = run_tape()
    if isinstance(result, dict):
        return {str(k): int(v) for k, v in result.items()}

    raise ValueError(f"Unexpected result type: {type(result)}")


def _log_program_for_replay(run: Any, data: CircuitData) -> None:
    """Log program artifact for replay run so fingerprints are meaningful."""
    fmt = data.format
    # Only native formats - no OpenQASM
    kind_map = {
        CircuitFormat.QPY: ("qiskit.qpy", "application/octet-stream"),
        CircuitFormat.JAQCD: ("braket.jaqcd.json", "application/json"),
        CircuitFormat.CIRQ_JSON: ("cirq.circuit.json", "application/json"),
        CircuitFormat.TAPE_JSON: ("pennylane.tape.json", "application/json"),
    }

    kind, media = kind_map.get(fmt, ("program.unknown", "application/octet-stream"))

    run.log_bytes(
        kind=kind,
        data=data.as_bytes(),
        media_type=media,
        role="program",
        meta={"format": fmt.value, "replay_source": True},
    )


def _save_replay_run(
    record: RunRecord,
    circuit_data: CircuitData,
    replay_counts: dict[str, int],
    backend_name: str,
    run_shots: int,
    seed: int | None,
    project: str | None,
    root: Path | None,
    registry: RegistryProtocol | None,
    store: ObjectStoreProtocol | None,
    errors: list[str],
) -> str | None:
    """Save replay as a new tracked run."""
    try:
        from devqubit_engine.core.run import track

        dest_registry = registry or create_registry(f"file://{root}" if root else None)
        dest_store = store or create_store(f"file://{root}/objects" if root else None)

        proj = project or record.project or "replay"

        with track(
            project=proj,
            adapter="replay",
            registry=dest_registry,
            store=dest_store,
        ) as run:
            _log_program_for_replay(run, circuit_data)

            run.log_execute({"backend": backend_name, "shots": run_shots})
            run.log_params(
                {
                    "replayed_from": record.run_id,
                    "original_backend": record.backend_name or "unknown",
                    "replay_backend": backend_name,
                    "circuit_source": circuit_data.format.value,
                    "shots": run_shots,
                    "seed": seed,
                }
            )
            run.log_json(
                name="counts",
                obj={"experiments": [{"index": 0, "counts": replay_counts}]},
                role="results",
                kind="result.counts.json",
            )
            run.set_tags(
                {
                    "replay": "true",
                    "replayed_from": record.run_id,
                    "experimental": "true",
                }
            )

            logger.info("Saved replay run: %s", run.run_id)
            return run.run_id

    except Exception as e:
        logger.error("Failed to save replay run: %s", e)
        errors.append(f"Failed to save replay run: {e}")
        return None
