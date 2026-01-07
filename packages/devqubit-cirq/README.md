# devqubit-cirq

Google Cirq adapter for devqubit. Automatically captures circuits and results from Cirq simulators.

## Installation

```bash
pip install devqubit[cirq]
```

## Usage

```python
import cirq
from devqubit import track

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="result"),
])

with track(project="cirq-exp") as run:
    simulator = run.wrap(cirq.Simulator())
    result = simulator.run(circuit, repetitions=1000)
```

## What's Captured

- **Circuits** — Cirq JSON, OpenQASM 3
- **Results** — Measurement counts, histograms
- **Simulator info** — Simulator type, configuration

## License

Apache 2.0
