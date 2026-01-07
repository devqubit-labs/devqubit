# Quickstart

This page shows a minimal end-to-end workflow:

1. Track a run
2. Inspect it with the CLI
3. Compare against a baseline

## 1) Track an execution

```python
from devqubit import track

# Wrap your backend/device/executor with run.wrap(...)
with track(project="bell-state") as run:
    backend = run.wrap(your_backend)

    job = backend.run(your_circuit, shots=1000)
    result = job.result()

    run.log_metric("fidelity", 0.95)
    print("Run ID:", run.run_id)
```

## 2) Inspect with the CLI

```bash
devqubit list --project bell-state
devqubit show <run_id>
devqubit artifacts list <run_id>
```

## 3) Set a baseline and verify

```bash
# Pick a known-good run as your baseline
devqubit baseline set bell-state <baseline_run_id>

# Verify a new candidate run against the baseline
devqubit verify <candidate_run_id> --project bell-state --noise-factor 1.0
```

If you're running this in CI, use `--junit` to export a test report.

Next steps:
- Learn what gets captured in {doc}`../guides/adapters`
- Understand run comparison in {doc}`../guides/comparison`
