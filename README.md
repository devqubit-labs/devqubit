[![CI](https://github.com/devqubit-labs/devqubit/actions/workflows/ci.yaml/badge.svg?branch=main)](...)
[![PyPI](https://img.shields.io/pypi/v/devqubit)](https://pypi.org/project/devqubit/)
[![Python](https://img.shields.io/pypi/pyversions/devqubit)](https://pypi.org/project/devqubit/)
[![Docs](https://readthedocs.org/projects/devqubit/badge/?version=latest)](https://devqubit.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

# devqubit

**Local-first experiment tracking for quantum computing.** Capture circuits, backend state, and configuration — runs are reproducible, comparable, and easy to share. Access your data via Python API, CLI, or Web UI.

> **Status:** Alpha – APIs may evolve in `0.x` releases.

## Why devqubit?

General-purpose experiment trackers (MLflow, Weights & Biases, DVC) are great for logging parameters, metrics, and artifacts. But quantum workloads often need *extra structure* that isn't first-class there by default: capturing what actually executed (program + compilation), where it executed (backend/device), and how it executed (runtime options).

| Challenge | MLflow / W&B / DVC | devqubit |
|-----------|-------------------|----------|
| **Circuit artifacts** | manual file logging | OpenQASM 3 + SDK-native formats (automatic) |
| **Device context** | manual | backend snapshots, calibration/noise context (automatic) |
| **Reproducibility** | depends on what you log | "program + device + config fingerprints (automatic)" |
| **Result comparison** | metric/table-oriented comparisons | distribution/structural/drift-aware diffs |
| **Noise-aware verification** | requires custom logic | configurable policies with noise tolerance |
| **Portable sharing** | artifact/version workflows exist | self-contained run bundles (manifest + SHA-256 digests) |

**devqubit is quantum-first:** same circuit, same backend, different day — different results. **devqubit** helps you track *why*.

## Features

- **Automatic circuit capture** – QPY, OpenQASM 3, SDK-native formats
- **Multi-SDK support** – Qiskit, Qiskit Runtime, Braket, Cirq, PennyLane
- **Content-addressable storage** – deduplicated artifacts with SHA-256 digests
- **Reproducibility fingerprints** – detect changes in program, device, or config
- **Run comparison** – TVD analysis, structural diff, calibration drift
- **CI/CD verification** – baselines with configurable noise-aware policies
- **Portable bundles** – export/import runs as self-contained ZIPs

## Documentation

📚 **https://devqubit.readthedocs.io**

## Installation

**Requirements:** Python 3.11+ (tested on 3.11–3.13)

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

## Quick start

### Track an experiment

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from devqubit import track

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

with track(project="bell-state", name="baseline-v1") as run:
    backend = run.wrap(AerSimulator())
    job = backend.run(qc, shots=1000)
    counts = job.result().get_counts()

    run.log_param("shots", 1000)
    run.log_metric("p00", counts.get("00", 0) / 1000)

print(f"Run saved: {run.run_id}")  # or use run.name
```

The adapter captures: circuit (QPY + QASM3), backend config, job metadata, and results.

## Comparing runs

```python
from devqubit.compare import diff

# By run name (with project context)
result = diff("baseline-v1", "experiment-v2", project="bell-state")

# Or by run ID
result = diff("01JD7X...", "01JD8Y...")

print(result.identical)           # False
print(result.program.match_mode)  # "structural"
print(result.tvd)                 # 0.023
```

Or via CLI:

```bash
devqubit diff baseline-v1 experiment-v2 --project bell-state
```

## CI/CD verification

Verify that a run matches an established baseline:

```python
from devqubit.compare import verify_baseline, VerifyPolicy

policy = VerifyPolicy(
    tvd_threshold=0.05,
    require_same_device=False,
)

result = verify_baseline(
    "nightly-run",  # run name or ID
    project="vqe-hydrogen",
    policy=policy,
)

assert result.ok, result.reason
```

In CI pipelines, use the CLI with JUnit output:

```bash
devqubit verify nightly-run --project vqe-hydrogen --junit results.xml
```

## CLI reference

```bash
devqubit list                                       # List recent runs
devqubit show <run>                                 # Show run details
devqubit diff <run_a> <run_b> --project myproj      # Compare two runs
devqubit verify <run> --project myproj              # Verify against baseline
devqubit pack <run> -o bundle.zip --project myproj  # Export portable bundle
devqubit unpack bundle.zip                          # Import bundle
devqubit ui                                         # Start web UI
```

> **Note:** `<run>` can be a run ID or run name. When using names, provide `--project` for disambiguation.

See [CLI reference](https://devqubit.readthedocs.io/en/latest/reference/cli.html) for full interface information.

## Web UI

```bash
devqubit ui
# → http://127.0.0.1:8080
```

Browse runs, view artifacts, compare experiments, and set baselines.

## Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `DEVQUBIT_HOME` | `~/.devqubit` | Workspace directory |
| `DEVQUBIT_CAPTURE_GIT` | `true` | Capture git commit/branch/remote |
| `DEVQUBIT_CAPTURE_PIP` | `true` | Capture installed packages |
| `DEVQUBIT_VALIDATE` | `true` | Validate records against schema |

See [configuration guide](https://devqubit.readthedocs.io/en/latest/guides/configuration.html) for advanced options.

## Project structure

```
devqubit/                    # Metapackage (re-exports from engine)
packages/
├── devqubit-engine/         # Core: tracking, storage, comparison, CLI
├── devqubit-ui/             # FastAPI web interface
├── devqubit-qiskit/         # Qiskit adapter
├── devqubit-qiskit-runtime/ # IBM Runtime adapter
├── devqubit-braket/         # Amazon Braket adapter
├── devqubit-cirq/           # Google Cirq adapter
└── devqubit-pennylane/      # PennyLane adapter
```

## Contributing

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone and sync:
   ```bash
   git clone https://github.com/devqubit-labs/devqubit.git
   cd devqubit
   uv sync --all-packages --all-extras
   ```
3. Install hooks and run checks:
   ```bash
   uv run pre-commit install
   uv run pre-commit run --all-files
   uv run pytest
   ```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## License

Apache 2.0 – see [LICENSE](LICENSE).
