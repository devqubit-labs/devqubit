# devqubit

Local-first experiment tracking for quantum computing. Capture **code + circuits + backend/device context + configuration**
so runs are reproducible, comparable, and easy to share.

> **Status:** Alpha. APIs and bundle formats may evolve in `0.x` releases.

## Documentation

📚 Read the Docs: **https://devqubit.readthedocs.io**

The documentation source lives in this repository under `docs/`, but the canonical, rendered docs are on Read the Docs.

## Features

- **Automatic circuit capture** — QPY, OpenQASM 3, and native SDK formats
- **SDK adapters** — Qiskit, Qiskit Runtime, Amazon Braket, Cirq, PennyLane
- **Content-addressable storage** — deduplicated artifacts with SHA-256 digests
- **Reproducibility fingerprints** — detect changes in program, device, or configuration
- **Run comparison** — TVD analysis, drift detection
- **CI/CD verification** — verify runs against baselines in pipelines
- **Portable bundles** — export/import runs for collaboration

## Installation

### Requirements

- Python **3.11+** (tested on 3.11–3.13)

### Install from PyPI

```bash
pip install devqubit

# With SDK adapters
pip install "devqubit[qiskit]"          # Qiskit + Aer
pip install "devqubit[qiskit-runtime]"  # IBM Quantum Runtime
pip install "devqubit[braket]"          # Amazon Braket
pip install "devqubit[cirq]"            # Google Cirq
pip install "devqubit[pennylane]"       # PennyLane
pip install "devqubit[all]"             # All adapters

# With local web UI
pip install "devqubit[ui]"
```

### Install from source (recommended for development)

This repo is a **uv workspace** (monorepo). For a complete local dev environment:

```bash
git clone <REPO_URL>
cd <REPO_DIR>

# Core dev (fast)
uv sync --all-packages

# Full dev (adapters + UI extras)
uv sync --all-packages --all-extras
```

## Quick start

```python
from devqubit import track

with track(project="my-experiment") as run:
    # Parameters
    run.log_param("shots", 1000)
    run.log_param("backend", "xyz_backend")

    # ... execute your circuit ...

    # Metrics
    run.log_metric("fidelity", 0.95)
    run.log_metric("execution_time_ms", 1234.5)

    # Tags
    run.set_tag("backend_type", "simulator")
    run.set_tag("experiment_phase", "calibration")

    # JSON artifacts
    run.log_json(name="counts", obj=counts, role="results")

print(f"Run ID: {run.run_id}")
```

## SDK adapters

Adapters automatically capture circuits, results, and device information. Use `run.wrap()` to wrap your backend.

### Qiskit example

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
    run.log_json(name="counts", obj=counts, role="results")
```

Adapters for Qiskit Runtime, Amazon Braket, Cirq, and PennyLane follow the same pattern.

## CLI

```bash
devqubit list                              # List recent runs
devqubit show <run_id>                     # Show run details
devqubit diff <run_a> <run_b>              # Compare two runs
devqubit verify <run_id> --project myproj  # Verify against baseline
devqubit pack <run_id> -o experiment.zip   # Export portable bundle
```

## Web UI

```bash
devqubit ui
```

Starts a local web interface (default: http://127.0.0.1:8080) for browsing runs, viewing artifacts, and comparing experiments.

## Configuration

```bash
export DEVQUBIT_HOME=~/.devqubit          # Workspace directory
export DEVQUBIT_CAPTURE_GIT=true          # Capture git info
export DEVQUBIT_CAPTURE_PIP=true          # Capture installed packages
```

See the configuration docs in `docs/guides/configuration.md`.

## Development and contribution

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Create / sync the dev environment:
   ```bash
   uv sync --all-packages --all-extras
   ```
3. Install git hooks:
   ```bash
   uv run pre-commit install
   ```
4. Run checks and tests:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```

For full contributor guidance, see `CONTRIBUTING.md`.

## License

Apache 2.0
