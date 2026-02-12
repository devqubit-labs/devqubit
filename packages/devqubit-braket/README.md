# devqubit-braket

[![PyPI](https://img.shields.io/pypi/v/devqubit-braket)](https://pypi.org/project/devqubit-braket/)

Amazon Braket adapter for [devqubit](https://github.com/devqubit-labs/devqubit) — automatic circuit capture, device property snapshots, and result logging for Braket local simulators and managed QPUs.

> [!IMPORTANT]
> **This is an internal adapter package.** Install via `pip install "devqubit[braket]"` and use the `devqubit` public API.

## Installation

```bash
pip install "devqubit[braket]"
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

| Artifact | Kind | Role |
|---|---|---|
| OpenQASM 3 | `source.openqasm3` | `program` |
| Circuit diagram | `braket.circuits.diagram` | `program` |
| Measurement counts | `result.counts.json` | `result` |
| Raw result | `result.braket.raw.json` | `result_raw` |
| Device properties | `device.braket.raw_properties.json` | `device_raw` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

## Documentation

See the [Adapters guide](https://devqubit.readthedocs.io/en/latest/guides/adapters.html) for details on wrapping options, batch execution, and performance tuning.

## License

Apache 2.0
