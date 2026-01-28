[![CI](https://github.com/devqubit-labs/devqubit/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/devqubit-labs/devqubit/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/devqubit)](https://pypi.org/project/devqubit/)
[![Python](https://img.shields.io/pypi/pyversions/devqubit)](https://pypi.org/project/devqubit/)
[![Docs](https://readthedocs.org/projects/devqubit/badge/?version=latest)](https://devqubit.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

# devqubit

**Local-first experiment tracking for quantum computing.** Capture circuits, backend state, and configuration — runs are reproducible, comparable, and easy to share. Access your data via Python API, CLI, or Web UI.

> **Status:** Alpha — APIs may evolve in `0.x` releases.

## Why devqubit?

General-purpose experiment trackers (MLflow, Weights & Biases, DVC) are great for logging parameters, metrics, and artifacts. But quantum workloads need *extra structure* that isn't first-class there by default: capturing what actually executed (program + compilation), where it executed (backend/device), and how it executed (runtime options).

| Challenge | MLflow / W&B / DVC | devqubit |
|-----------|-------------------|----------|
| **Circuit artifacts** | manual file logging | OpenQASM 3 + SDK-native formats (automatic) |
| **Device context** | manual | backend snapshots, calibration/noise context (automatic) |
| **Reproducibility** | depends on what you log | program + device + config fingerprints (automatic) |
| **Result comparison** | metric/table-oriented | distribution-aware, structural diff, drift detection |
| **Noise-aware verification** | requires custom logic | configurable policies with noise tolerance |
| **Portable sharing** | artifact/version workflows | self-contained bundles (manifest + SHA-256 digests) |

**devqubit is quantum-first:** same circuit, same backend, different day — different results. devqubit helps you track *why*.

## Features

- **Automatic circuit capture** — QPY, OpenQASM 3, SDK-native formats
- **Multi-SDK support** — Qiskit, Qiskit Runtime, Braket, Cirq, PennyLane
- **Content-addressable storage** — deduplicated artifacts with SHA-256 digests
- **Reproducibility fingerprints** — detect changes in program, device, or config
- **Run comparison** — TVD analysis, structural diff, calibration drift
- **CI/CD verification** — baselines with configurable noise-aware policies
- **Portable bundles** — export/import runs as self-contained ZIPs

## Documentation

📚 **<https://devqubit.readthedocs.io>**

## Installation

**Requirements:** Python 3.11+

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

with track(project="bell-state", run_name="baseline-v1") as run:
    backend = run.wrap(AerSimulator())
    job = backend.run(qc, shots=1000)
    counts = job.result().get_counts()

    run.log_param("shots", 1000)
    run.log_metric("p00", counts.get("00", 0) / 1000)

print(f"Run saved: {run.run_id}")
```

The adapter automatically captures: circuit (QPY + QASM3), backend config, job metadata, and results.

### Compare runs

```python
from devqubit.compare import diff

result = diff("baseline-v1", "experiment-v2", project="bell-state")

print(result.identical)           # False
print(result.program.match_mode)  # "structural"
print(result.tvd)                 # 0.023
```

Or via CLI:

```bash
devqubit diff baseline-v1 experiment-v2 --project bell-state
```

### CI/CD verification

```python
from devqubit.compare import verify_baseline, VerifyPolicy

result = verify_baseline(
    "nightly-run",
    project="vqe-hydrogen",
    policy=VerifyPolicy(tvd_threshold=0.05),
)

assert result.ok, result.reason
```

```bash
# With JUnit output for CI pipelines
devqubit verify nightly-run --project vqe-hydrogen --junit results.xml
```

## CLI

```bash
devqubit list                          # List runs
devqubit show <run> --project myproj   # Run details
devqubit diff <a> <b> --project myproj # Compare runs
devqubit ui                            # Web interface
```

See [CLI reference](https://devqubit.readthedocs.io/en/latest/reference/cli.html) for all commands.

## Web UI

```bash
devqubit ui
# → http://127.0.0.1:8080
```

<p align="center">
  <a href="docs/assets/ui_runs.png">
    <img src="docs/assets/ui_runs.png" alt="Runs list" width="45%"/>
  </a>
  &nbsp;&nbsp;
  <a href="docs/assets/ui_run_view.png">
    <img src="docs/assets/ui_run_view.png" alt="Run comparison" width="45%"/>
  </a>
</p>

Browse runs, view artifacts, compare experiments, and manage baselines.

## Contributing

We welcome contributions of all kinds — bug fixes, docs, new adapters, or feature ideas.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines
2. Check [open issues](https://github.com/devqubit-labs/devqubit/issues) or start a [discussion](https://github.com/devqubit-labs/devqubit/discussions)
3. Fork, branch, and submit a PR

```bash
git clone https://github.com/devqubit-labs/devqubit.git
cd devqubit
uv sync --all-packages
uv run pre-commit install
uv run pytest
```

Early project = high impact contributions. Jump in!

## Community

- 💬 [Discussions](https://github.com/devqubit-labs/devqubit/discussions) — questions, ideas, feedback
- 🐛 [Issues](https://github.com/devqubit-labs/devqubit/issues) — bug reports, feature requests
- 📚 [Docs](https://devqubit.readthedocs.io) — guides and API reference

## License

Apache 2.0 — see [LICENSE](LICENSE).
