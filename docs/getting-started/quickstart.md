# Quickstart

This guide shows a complete workflow: track an experiment, inspect results, compare runs, and verify against a baseline.

## Installation

```bash
pip install devqubit

# With your SDK adapter
pip install "devqubit[qiskit]"          # Qiskit + Aer
pip install "devqubit[qiskit-runtime]"  # Qiskit + Runtime
pip install "devqubit[braket]"          # Amazon Braket
pip install "devqubit[cirq]"            # Google Cirq
pip install "devqubit[pennylane]"       # PennyLane
pip install "devqubit[cudaq]"           # NVIDIA CUDA-Q
```

## Track an Experiment

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from devqubit import track

# Create a Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

with track(project="bell-state", name="baseline-v1") as run:
    # Wrap the backend - this enables automatic capture
    backend = run.wrap(AerSimulator())

    # Execute as usual
    job = backend.run(qc, shots=1000)
    counts = job.result().get_counts()

    # Log parameters and metrics
    run.log_param("shots", 1000)
    run.log_param("optimization_level", 1)
    run.log_metric("p00", counts.get("00", 0) / 1000)
    run.log_metric("p11", counts.get("11", 0) / 1000)

print(f"Run saved: {run.run_id} (name: {run.name})")
```

The adapter automatically captures:
- Circuit artifacts (QPY + OpenQASM 3)
- Backend configuration and calibration
- Job metadata and execution options
- Measurement results with bit-order metadata

## Inspect with CLI

```bash
# List runs in a project
devqubit list --project bell-state

# Show run details (by name or ID)
devqubit show baseline-v1 --project bell-state

# List captured artifacts
devqubit artifacts list baseline-v1 --project bell-state

# Export a portable bundle
devqubit pack baseline-v1 -o bell-run.zip --project bell-state
```

> **Tip:** Most CLI commands accept run name or run ID. Use `--project` when referencing by name.

## Compare Runs

```python
from devqubit.compare import diff

# Compare by name (recommended)
result = diff("baseline-v1", "experiment-v2", project="bell-state")

# Or by run ID
result = diff("01JD7X...", "01JD8Y...")

print(result.identical)              # False
print(result.program.structural_match)  # True (same circuit structure)
print(result.tvd)                    # 0.023 (distribution distance)
print(result.device_drift)           # DriftResult with calibration changes
```

Or via CLI:

```bash
devqubit diff baseline-v1 experiment-v2 --project bell-state
```

## Baseline Verification

Set a known-good run as baseline and verify new runs against it:

```bash
# Set baseline (by name or ID)
devqubit baseline set bell-state baseline-v1

# Verify a candidate run
devqubit verify nightly-run --project bell-state

# With noise-aware threshold (recommended for hardware)
devqubit verify nightly-run --project bell-state --noise-factor 1.2

# Export JUnit report for CI
devqubit verify nightly-run --project bell-state --junit results.xml
```

Programmatic verification:

```python
from devqubit.compare import verify_baseline, VerifyPolicy

result = verify_baseline(
    "nightly-run",  # run name or ID
    project="bell-state",
    policy=VerifyPolicy(
        noise_factor=1.2,                 # 1.2x bootstrap noise threshold
        tvd_max=0.1,                      # hard limit
        program_match_mode="structural",  # allow parameter changes
    ),
)

if not result.ok:
    print(result.failures)
```

## Web UI

```bash
devqubit ui
# → http://127.0.0.1:8080
```

Browse runs, view artifacts, compare experiments, and manage baselines.

## Next Steps

- {doc}`../concepts/overview` — Core concepts and architecture
- {doc}`../concepts/workspace` — How to configure the workspace
- {doc}`../concepts/uec` — Uniform Execution Contract (what gets captured)
- {doc}`../guides/tracking` — Advanced tracking, groups, and DataFrame analysis
