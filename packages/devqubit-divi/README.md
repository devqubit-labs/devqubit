# devqubit-divi

Qoro Divi adapter for [devqubit](https://github.com/devqubit-labs/devqubit).

> [!IMPORTANT]
> **This is an internal adapter package.** Install via `pip install "devqubit[cudaq]"` and use the `devqubit` public API.

## Installation

```bash
pip install "devqubit[cudaq]"
```

## Usage

### Sampling

```python
import cudaq
from devqubit import track

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)

with track(project="cudaq-experiment") as run:
    executor = run.wrap(cudaq)
    result = executor.sample(bell, shots_count=1000)
```

### Observe (Expectation Values)

```python
from cudaq import spin

hamiltonian = spin.z(0)

with track(project="cudaq-vqe") as run:
    executor = run.wrap(cudaq)
    result = executor.observe(bell, hamiltonian)
    print(result.expectation())
```

## What's Captured

| Artifact | Kind | Role |
|---|---|---|
| Kernel JSON | `cudaq.kernel.json` | `program` |
| Kernel diagram | `cudaq.kernel.diagram` | `program` |
| MLIR (Quake) | `cudaq.kernel.mlir` | `program` |
| QIR | `cudaq.kernel.qir` | `program` |
| Counts / expectation | `result.cudaq.output.json` | `result` |
| Execution envelope | `devqubit.envelope.json` | `envelope` |

## Documentation

See the [Adapters guide](https://devqubit.readthedocs.io/en/latest/guides/adapters.html) for details on `observe` workflows, performance tuning, and spin operator capture.

## License

Apache 2.0
