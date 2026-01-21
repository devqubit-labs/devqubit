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

with track(project="bell-state") as run:
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

print(f"Run saved: {run.run_id}")
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

# Show run details
devqubit show <run_id>

# List captured artifacts
devqubit artifacts list <run_id>

# Export a portable bundle
devqubit pack <run_id> -o bell-run.zip
```

## Compare Runs

```python
from devqubit import diff

result = diff("01JD7X...", "01JD8Y...")

print(result.identical)              # False
print(result.program.structural_match)  # True (same circuit structure)
print(result.tvd)                    # 0.023 (distribution distance)
print(result.device_drift)           # DriftResult with calibration changes
```

Or via CLI:

```bash
devqubit diff 01JD7X... 01JD8Y...
```

## Baseline Verification

Set a known-good run as baseline and verify new runs against it:

```bash
# Set baseline
devqubit baseline set bell-state <baseline_run_id>

# Verify a candidate run
devqubit verify <candidate_run_id> --project bell-state

# With noise-aware threshold (recommended for hardware)
devqubit verify <candidate_run_id> --project bell-state --noise-factor 1.2

# Export JUnit report for CI
devqubit verify <candidate_run_id> --project bell-state --junit results.xml
```

Programmatic verification:

```python
from devqubit import verify_baseline
from devqubit.compare import VerifyPolicy

result = verify_baseline(
    candidate_run,
    project="bell-state",
    policy=VerifyPolicy(
        noise_factor=1.2,                 # 1.2x bootstrap noise threshold
        tvd_max=0.1,                      # hard limit
        program_match_mode="structural",  # allow parameter changes
    ),
)

if not result.ok:
    print(result.failures)
    print(result.verdict.summary)  # e.g., "Device drift: T1 degraded 15%"
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
