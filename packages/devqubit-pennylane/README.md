# devqubit-pennylane

[![PyPI](https://img.shields.io/pypi/v/devqubit-pennylane)](https://pypi.org/project/devqubit-pennylane/)

PennyLane adapter for [devqubit](https://github.com/devqubit-labs/devqubit) — automatic tape capture, device snapshots, and result logging for PennyLane devices and QNode workflows.

> [!IMPORTANT]
> **This is an internal adapter package.** Install via `pip install "devqubit[pennylane]"` and use the `devqubit` public API.

## Installation

```bash
pip install "devqubit[pennylane]"
```

## Usage

```python
import pennylane as qml
from devqubit import track

dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts()

with track(project="pennylane-exp") as run:
    run.wrap(dev)       # in-place device patching
    counts = circuit()  # QNodes using this device are tracked automatically
```

### Multi-Layer Stack

PennyLane can act as a frontend to other execution providers. The adapter captures the full stack:

```python
# Braket backend through PennyLane
dev = qml.device("braket.aws.qubit", wires=2, device_arn="...")

# Qiskit backend through PennyLane
dev = qml.device("qiskit.remote", wires=2, backend="ibm_brisbane")
```

## What's Captured

| Artifact | Kind | Role |
|---|---|---|
| Tape JSON | `pennylane.tapes.json` | `program` |
| Tape diagram | `pennylane.tapes.txt` | `program` |
| Results | `result.pennylane.output.json` | `result` |
| Device properties | `device.pennylane.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

## Documentation

See the [Adapters guide](https://devqubit.readthedocs.io/en/latest/guides/adapters.html) for performance tuning (`log_every_n`), parameter sweeps, and multi-layer stack details.

## License

Apache 2.0
