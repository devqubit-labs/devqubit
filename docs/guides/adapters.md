# Adapters

Adapters provide SDK-specific integration for **automatic circuit/program + result + device snapshot capture**.

If you haven't yet, read {doc}`../concepts/uec` to understand the **ExecutionEnvelope** produced by adapters.

Adapters provide SDK-specific integration for automatic circuit and result capture. Each adapter wraps your quantum backend/device/primitive to automatically log circuits, results, and device snapshots.

## Installation

Install the adapter for your quantum SDK:

```bash
pip install 'devqubit[qiskit]'          # IBM Qiskit (local backends, Aer)
pip install 'devqubit[qiskit-runtime]'  # IBM Qiskit Runtime (cloud primitives)
pip install 'devqubit[braket]'          # Amazon Braket
pip install 'devqubit[cirq]'            # Google Cirq
pip install 'devqubit[pennylane]'       # Xanadu PennyLane

# Or install all adapters at once
pip install 'devqubit[all]'
```

## Quick Start

All adapters follow the same pattern: wrap your executor with `run.wrap()`:

```python
from devqubit import track

with track(project="my-experiment") as run:
    backend = run.wrap(your_backend)  # Wrap once
    job = backend.run(circuit)        # Use normally
    result = job.result()             # Results auto-logged
```

---

## Uniform Execution Contract (UEC)

All adapters produce a standardized **ExecutionEnvelope** (`devqubit.envelope/1.0` schema) containing four canonical snapshots:

| Snapshot | Schema | Description |
|----------|--------|-------------|
| `producer` | — | Adapter and SDK version info (name, adapter, sdk_version, frontends) |
| `program` | `devqubit.program_snapshot/1.0` | Logical and physical circuit artifacts with structural/parametric hashes |
| `device` | `devqubit.device_snapshot/1.0` | Backend state, calibration, topology, and provider properties |
| `execution` | `devqubit.execution_snapshot/1.0` | Submission metadata, transpilation info, shots, job IDs |
| `result` | `devqubit.result_snapshot/1.0` | Normalized measurement counts, quasi-probabilities, or expectation values |

The envelope is automatically logged as `devqubit.envelope.json` with role `envelope`. This provides a complete, self-contained record of each execution that can be used for reproducibility, comparison (`diff`), and verification (`verify_baseline`).

**Adapter contract:**
1. Emit a schema-valid envelope for every execution (success or failure).
2. On failure, set `result.success=false` with structured `result.error`.
3. Store large payloads (raw results, backend properties) as artifacts via `ArtifactRef`.
4. Normalize bitstrings to canonical `cbit0_right` format for cross-SDK comparison.

```python
# Access envelope data after execution
run.record["execute"]           # Execution metadata
run.record["device_snapshot"]   # Device info summary
run.record["execution_stats"]   # Aggregate statistics
```

---

## Qiskit

For local backends and Aer simulators.

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from devqubit import track

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

with track(project="bell-state") as run:
    backend = run.wrap(AerSimulator())

    transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=1000)
    result = job.result()

    # All artifacts captured automatically:
    # - QPY binary, OpenQASM 3, circuit diagrams
    # - Measurement counts
    # - Backend snapshot
```

### Captured Artifacts

| Artifact | Kind | Role |
|----------|------|------|
| QPY binary | `qiskit.qpy.circuits` | `program` |
| OpenQASM 3 | `source.openqasm3` | `program` |
| Circuit diagram | `qiskit.circuits.diagram` | `program` |
| Counts | `result.counts.json` | `result` |
| Full result | `result.qiskit.result_json` | `result_raw` |
| Raw backend properties | `device.qiskit.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

---

## Qiskit Runtime

For IBM Quantum cloud primitives (Sampler, Estimator).

```python
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from devqubit import track

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

with track(project="runtime-experiment") as run:
    sampler = run.wrap(SamplerV2(backend))
    job = sampler.run([qc])
    result = job.result()
```

### Transpilation Modes

The Runtime adapter supports automatic ISA compatibility checking:

```python
# Auto mode (default): transpile only if needed
sampler = run.wrap(SamplerV2(backend))
job = sampler.run([qc], devqubit_transpilation_mode="auto")

# Manual mode: you handle transpilation
job = sampler.run([isa_circuit], devqubit_transpilation_mode="manual")

# With custom transpilation options
job = sampler.run([qc],
    devqubit_transpilation_mode="auto",
    devqubit_transpilation_options={"optimization_level": 2}
)
```

### Captured Artifacts

| Artifact | Kind | Role |
|----------|------|------|
| QPY binary | `qiskit.qpy.circuits` | `program` |
| Transpiled QPY | `qiskit.qpy.circuits.transpiled` | `program` |
| OpenQASM 3 | `source.openqasm3` | `program` |
| PUB structure | `qiskit_runtime.pubs.json` | `program` |
| Sampler counts | `result.counts.json` | `result` |
| Estimator values | `result.qiskit_runtime.estimator.json` | `result` |
| Raw runtime properties | `device.qiskit_runtime.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

---

## Amazon Braket

```python
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from devqubit import track

circuit = Circuit().h(0).cnot(0, 1)

with track(project="braket-experiment") as run:
    device = run.wrap(LocalSimulator())
    task = device.run(circuit, shots=1000)
    result = task.result()
```

### Captured Artifacts

| Artifact | Kind | Role |
|----------|------|------|
| OpenQASM 3 | `source.openqasm3` | `program` |
| Circuit diagram | `braket.circuits.diagram` | `program` |
| Counts | `result.counts.json` | `result` |
| Raw result | `result.braket.raw.json` | `result_raw` |
| Raw device properties | `device.braket.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

---

## Google Cirq

```python
import cirq
from devqubit import track

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="m"),
])

with track(project="cirq-experiment") as run:
    simulator = run.wrap(cirq.Simulator())
    result = simulator.run(circuit, repetitions=1000)
```

### Parameter Sweeps

```python
import sympy

theta = sympy.Symbol("theta")
circuit = cirq.Circuit([
    cirq.Ry(theta).on(q0),
    cirq.measure(q0, key="m"),
])

with track(project="sweep") as run:
    simulator = run.wrap(cirq.Simulator())
    sweep = cirq.Linspace("theta", 0, 2 * 3.14159, 10)
    results = simulator.run_sweep(circuit, sweep, repetitions=100)
```

### Captured Artifacts

| Artifact | Kind | Role |
|----------|------|------|
| Cirq JSON | `cirq.circuit.json` | `program` |
| Circuit diagram | `cirq.circuits.txt` | `program` |
| Counts | `result.counts.json` | `result` |
| Raw device properties | `device.cirq.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

---

## PennyLane

PennyLane uses in-place device patching for QNode compatibility.

```python
import pennylane as qml
from devqubit import track

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

with track(project="vqe") as run:
    tracked_dev = run.wrap(dev)

    # QNodes using this device are automatically tracked
    for step in range(100):
        result = circuit([step * 0.1])
```

### Captured Artifacts

| Artifact | Kind | Role |
|----------|------|------|
| Tape JSON | `pennylane.tapes.json` | `program` |
| Tape diagram | `pennylane.tapes.txt` | `program` |
| Results | `result.pennylane.output.json` | `result` |
| Raw device properties | `device.pennylane.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

### Multi-Layer Stack

PennyLane acts as a frontend to multiple execution providers. When using external backends, the adapter captures the full stack:

```python
# Braket backend through PennyLane
dev = qml.device("braket.aws.qubit", wires=2, device_arn="...")

# Qiskit backend through PennyLane
dev = qml.device("qiskit.remote", wires=2, backend="ibm_brisbane")
```

The device snapshot includes:
- **Frontend config**: PennyLane device settings (shots, diff_method, interface)
- **Resolved backend**: Underlying Braket/Qiskit device topology and calibration

---

## Performance Optimization

For training loops with thousands of executions, use sampling to reduce logging overhead:

```python
with track(project="qml-training") as run:
    # Default: log first execution only (fastest)
    backend = run.wrap(device)

    # Log every 100th execution
    backend = run.wrap(device, log_every_n=100)

    # Log all executions (slowest)
    backend = run.wrap(device, log_every_n=-1)

    # Disable new circuit detection
    backend = run.wrap(device, log_new_circuits=False)

    # Control stats update frequency
    backend = run.wrap(device, stats_update_interval=500)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_every_n` | `0` | `0`=first only, `N`=every Nth, `-1`=all |
| `log_new_circuits` | `True` | Auto-log when circuit structure changes |
| `stats_update_interval` | `1000` | Update execution stats every N runs |

### Execution Statistics

When using sampling, the tracker records aggregate statistics:

```python
run.record["execution_stats"]
# {
#     "total_executions": 10000,
#     "logged_executions": 15,
#     "unique_circuits": 3,
#     "logged_circuits": 3,
#     "last_execution_at": "2024-01-15T10:30:00Z"
# }
```

---

## Common Patterns

### Logging Compile Options

```python
with track(project="test") as run:
    compile_options = {
        "optimization_level": 3,
        "routing_method": "sabre",
        "seed_transpiler": 42,
    }

    for key, value in compile_options.items():
        run.log_param(key, value)

    transpiled = transpile(qc, backend, **compile_options)
```

### Multi-Circuit Batches

```python
circuits = [circuit_1, circuit_2, circuit_3]

with track(project="batch") as run:
    backend = run.wrap(AerSimulator())
    job = backend.run(circuits, shots=1000)
    result = job.result()
    # All circuits and results captured automatically
```

### Tagging Experiments

```python
with track(project="vqe") as run:
    run.set_tag("algorithm", "VQE")
    run.set_tag("ansatz", "EfficientSU2")
    run.set_tag("optimizer", "COBYLA")

    backend = run.wrap(device)
    # ... run experiment
```

---

## Using Base Engine

If no adapter exists for your SDK, use the base engine directly:

```python
from devqubit import track

with track(project="custom-sdk") as run:
    run.log_param("shots", 1000)
    run.set_tag("sdk", "custom")

    # Log circuit as bytes
    circuit_ref = run.log_bytes(
        kind="custom.circuit",
        data=circuit_bytes,
        media_type="application/octet-stream",
        role="program",
    )

    # Run your experiment
    result = custom_sdk.run(circuit)

    # Log results as JSON
    run.log_json(
        name="counts",
        obj={"00": 500, "11": 500},
        role="result",
        kind="result.counts.json",
    )
```

### Creating Custom Envelopes

For full UEC compliance, create an ExecutionEnvelope manually. The envelope will be validated against `devqubit.envelope/1.0` schema:

```python
from devqubit.uec import (
    ExecutionEnvelope,
    ExecutionSnapshot,
    DeviceSnapshot,
    ProgramSnapshot,
    ResultSnapshot,
    ResultItem,
    ProducerInfo,
)
from devqubit.utils import utc_now_iso

# Producer identifies your adapter
producer = ProducerInfo.create(
    adapter="devqubit-custom",
    adapter_version="0.1.0",
    sdk="custom-sdk",
    sdk_version="1.0.0",
    frontends=["custom-sdk"],
)

# Device snapshot (devqubit.device_snapshot/1.0)
device = DeviceSnapshot(
    captured_at=utc_now_iso(),
    backend_name="custom_device",
    backend_type="simulator",
    provider="local",
)

# Program snapshot (devqubit.program_snapshot/1.0)
# For adapter runs, structural_hash and parametric_hash are required
program = ProgramSnapshot(
    logical=[],  # Add ProgramArtifact refs here
    physical=[],
    structural_hash="sha256:...",
    parametric_hash="sha256:...",
    num_circuits=1,
)

# Execution snapshot (devqubit.execution_snapshot/1.0)
execution = ExecutionSnapshot(
    submitted_at=utc_now_iso(),
    shots=1000,
)

# Result snapshot (devqubit.result_snapshot/1.0)
result = ResultSnapshot.create_success(
    items=[
        ResultItem(
            item_index=0,
            success=True,
            counts={
                "counts": {"00": 500, "11": 500},
                "shots": 1000,
                "format": {
                    "source_sdk": "custom-sdk",
                    "source_key_format": "little_endian",
                    "bit_order": "cbit0_right",
                    "transformed": False,
                },
            },
        )
    ],
)

# Create envelope using factory method (auto-generates envelope_id and created_at)
envelope = ExecutionEnvelope.create(
    producer=producer,
    device=device,
    program=program,
    execution=execution,
    result=result,
)

# Validate before logging (optional but recommended)
validation = envelope.validate_schema()
if not validation.ok:
    print(f"Validation errors: {validation.errors}")

# Log the envelope
run.log_json(
    name="execution_envelope",
    obj=envelope.to_dict(),
    role="envelope",
    kind="devqubit.envelope.json",
)
```

**Schema requirements for adapter runs** (`producer.adapter != "manual"`):
- `program` and `execution` must exist
- `program.structural_hash` and `program.parametric_hash` are required
- If `program.physical` exists, `executed_structural_hash` and `executed_parametric_hash` are also required

---

## Adapter API Reference

All adapters implement the same interface:

```python
class Adapter:
    name: str  # Adapter identifier

    def supports_executor(self, executor: Any) -> bool:
        """Check if executor is supported."""

    def describe_executor(self, executor: Any) -> dict:
        """Get executor description."""

    def wrap_executor(
        self,
        executor: Any,
        tracker: Run,
        *,
        log_every_n: int = 0,
        log_new_circuits: bool = True,
        stats_update_interval: int = 1000,
        **kwargs,
    ) -> TrackedExecutor:
        """Wrap executor with tracking."""
```
