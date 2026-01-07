# devqubit-braket

Amazon Braket adapter for devqubit. Automatically captures circuits, results, and device information from Braket simulators and QPUs.

## Installation

```bash
pip install devqubit[braket]
```

## Usage

```python
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from devqubit import track

circuit = Circuit().h(0).cnot(0, 1)

with track(project="braket-exp") as run:
    device = run.wrap(LocalSimulator())
    task = device.run(circuit, shots=1000)
    result = task.result()
```

## What's Captured

- **Circuits** — OpenQASM 3, Braket IR
- **Results** — Measurement counts, result types
- **Device info** — Device ARN, properties, topology

## License

Apache 2.0
