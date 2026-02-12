# devqubit-cirq

[![PyPI](https://img.shields.io/pypi/v/devqubit-cirq)](https://pypi.org/project/devqubit-cirq/)

Google Cirq adapter for [devqubit](https://github.com/devqubit-labs/devqubit) — automatic circuit capture, simulator snapshots, and result logging for Cirq samplers and simulators.

> [!IMPORTANT]
> **This is an internal adapter package.** Install via `pip install "devqubit[cirq]"` and use the `devqubit` public API.

## Installation

```bash
pip install "devqubit[cirq]"
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

## What's Captured

| Artifact | Kind | Role |
|---|---|---|
| Cirq JSON | `cirq.circuit.json` | `program` |
| Circuit diagram | `cirq.circuits.txt` | `program` |
| Measurement counts | `result.counts.json` | `result` |
| Device properties | `device.cirq.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

## Documentation

See the [Adapters guide](https://devqubit.readthedocs.io/en/latest/guides/adapters.html) for parameter sweeps, performance tuning, and batch execution.

## License

Apache 2.0
