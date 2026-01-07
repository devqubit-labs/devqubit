# devqubit-pennylane

PennyLane adapter for devqubit. Automatically captures circuits and results from PennyLane devices.

## Installation

```bash
pip install devqubit[pennylane]
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
    run.wrap(dev)
    counts = circuit()
```

## What's Captured

- **Circuits** — PennyLane tape, OpenQASM 3
- **Results** — Counts, expectation values, samples
- **Device info** — Device name, wires, shots

## License

Apache 2.0
