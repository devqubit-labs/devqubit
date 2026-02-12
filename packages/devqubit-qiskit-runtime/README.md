# devqubit-qiskit-runtime

[![PyPI](https://img.shields.io/pypi/v/devqubit-qiskit-runtime)](https://pypi.org/project/devqubit-qiskit-runtime/)

IBM Qiskit Runtime adapter for [devqubit](https://github.com/devqubit-labs/devqubit) — automatic circuit capture, transpilation management, calibration snapshots, and result logging for Runtime V2 primitives (`SamplerV2`, `EstimatorV2`).

> [!IMPORTANT]
> **This is an internal adapter package.** Install via `pip install "devqubit[qiskit-runtime]"` and use the `devqubit` public API.

## Installation

```bash
pip install "devqubit[qiskit-runtime]"
```

## Usage

```python
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from devqubit import track

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

with track(project="hardware-run") as run:
    sampler = run.wrap(SamplerV2(backend))
    job = sampler.run([(qc,)])
    result = job.result()
```

### Transpilation Modes

```python
# Auto (default): transpile to ISA only if needed
sampler = run.wrap(SamplerV2(backend))
job = sampler.run([qc], devqubit_transpilation_mode="auto")

# Manual: you handle transpilation yourself
job = sampler.run([isa_circuit], devqubit_transpilation_mode="manual")
```

## What's Captured

| Artifact | Kind | Role |
|---|---|---|
| QPY binary | `qiskit.qpy.circuits` | `program` |
| Transpiled QPY | `qiskit.qpy.circuits.transpiled` | `program` |
| OpenQASM 3 | `source.openqasm3` | `program` |
| PUB structure | `qiskit_runtime.pubs.json` | `program` |
| Sampler counts | `result.counts.json` | `result` |
| Estimator values | `result.qiskit_runtime.estimator.json` | `result` |
| Backend properties | `device.qiskit_runtime.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

## Documentation

See the [Adapters guide](https://devqubit.readthedocs.io/en/latest/guides/adapters.html) for transpilation modes, performance tuning, and batch execution.

## License

Apache 2.0
