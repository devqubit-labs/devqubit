# devqubit-cudaq

CUDA-Q adapter for the devqubit quantum experiment tracking system.

## Installation

```bash
pip install devqubit[cudaq]
```

## Usage

```python
import cudaq
from devqubit import track

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    cudaq.h(q[0])
    cudaq.cx(q[0], q[1])
    cudaq.mz(q)

with track(project="cudaq-experiment") as run:
    executor = run.wrap(cudaq)
    result = executor.sample(bell, shots_count=1000)
    print(result)
```

## What's Captured

- **Circuits** — Cirq JSON, OpenQASM 3
- **Results** — Measurement counts, histograms
- **Simulator info** — Simulator type, configuration

## License

Apache 2.0
