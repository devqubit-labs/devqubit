# devqubit-engine

Core engine for devqubit experiment tracking. Provides storage, registry, run management, comparison, and CLI.

> [!IMPORTANT]
> Users should install `devqubit` (the meta-package) instead, which includes this package plus convenient extras.

## Installation

```bash
pip install devqubit-engine
```

## What's Included

- **Run tracking** — `track()` context manager, parameter/metric/artifact logging
- **Storage** — Content-addressable object store with SHA-256 deduplication
- **Registry** — Run indexing, search, baselines, groups
- **Comparison** — `diff()`, `diff_runs()`, TVD analysis, calibration drift detection
- **Verification** — `verify()` for CI/CD baseline checks
- **Bundles** — Portable run export/import
- **CLI** — `devqubit list`, `show`, `diff`, `verify`, `pack`, `unpack`

## Usage

```python
from devqubit import track

with track(project="my-experiment") as run:
    run.log_param("shots", 1000)
    run.log_metric("fidelity", 0.95)
    run.log_json(name="counts", obj={"00": 502, "11": 498}, role="results")
```

## Adapters

Circuit capture requires SDK-specific adapters (separate packages):

```bash
pip install devqubit[qiskit]         # Qiskit + Aer
pip install devqubit[qiskit-runtime] # IBM Quantum Runtime
pip install devqubit[braket]         # Amazon Braket
pip install devqubit[cirq]           # Google Cirq
pip install devqubit[pennylane]      # PennyLane
```

## Configuration

```bash
export DEVQUBIT_HOME=~/.devqubit
export DEVQUBIT_CAPTURE_GIT=true
export DEVQUBIT_CAPTURE_PIP=true
```

## License

Apache 2.0
