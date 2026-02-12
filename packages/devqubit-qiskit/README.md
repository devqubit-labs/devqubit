# devqubit-qiskit

[![PyPI](https://img.shields.io/pypi/v/devqubit-qiskit)](https://pypi.org/project/devqubit-qiskit/)

Qiskit adapter for [devqubit](https://github.com/devqubit-labs/devqubit) — automatic circuit capture, backend snapshots, and result logging for Qiskit and Aer backends.

> [!IMPORTANT]
> **This is an internal adapter package.** Install via `pip install "devqubit[qiskit]"` and use the `devqubit` public API.

## Installation

```bash
pip install "devqubit[qiskit]"
```

## Usage

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from devqubit import track

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

with track(project="bell-state") as run:
    backend = run.wrap(AerSimulator())
    job = backend.run(qc, shots=1000)
    counts = job.result().get_counts()
```

## What's Captured

| Artifact | Kind | Role |
|---|---|---|
| QPY binary | `qiskit.qpy.circuits` | `program` |
| OpenQASM 3 | `source.openqasm3` | `program` |
| Circuit diagram | `qiskit.circuits.diagram` | `program` |
| Measurement counts | `result.counts.json` | `result` |
| Full SDK result | `result.qiskit.result_json` | `result_raw` |
| Backend properties | `device.qiskit.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

## Documentation

See the [Adapters guide](https://devqubit.readthedocs.io/en/latest/guides/adapters.html) for details on wrapping options, batch execution, and performance tuning.

## License

Apache 2.0
