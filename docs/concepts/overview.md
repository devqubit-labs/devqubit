# Core concepts

devqubit treats each execution as a **run** with a complete context.

## Run

A run is a single tracked experiment execution. A run has:

- **metadata**: project, timestamps, status, run name
- **parameters**: experiment configuration values (`log_param`)
- **metrics**: numeric results (`log_metric`), including time series with steps
- **tags**: string key-value pairs for categorization (`set_tag`)
- **artifacts**: program/circuit, results, device snapshot, config, etc.
- **fingerprints**: hashes for reproducibility checks
- **environment**: Python version, installed packages (optional)
- **provenance**: git commit, branch, dirty state (optional)

## Run lifecycle

A run transitions through these statuses:

- `RUNNING` — run is active
- `FINISHED` — run completed successfully
- `FAILED` — run encountered an error (exception captured)
- `KILLED` — run was interrupted (e.g., KeyboardInterrupt)

## Artifact and role

Artifacts are any captured files/blobs (QPY, OpenQASM, counts JSON, device snapshot, …). Each artifact is content-addressed using SHA-256 digests, enabling deduplication and integrity verification.

Each artifact has a **role** that tells devqubit how it should be interpreted:

| Role | Description |
|------|-------------|
| `program` | Circuit/program artifacts (QPY, QASM) — used for fingerprinting |
| `results` | Measurement counts, expectation values |
| `device_raw` | Raw backend properties from provider (lossless capture) |
| `envelope` | ExecutionEnvelope — the Uniform Execution Contract record |
| `config` | Compile/execute options, environment capture |
| `documentation` | Notes, attachments |

## Fingerprints

Fingerprints are SHA-256 hashes computed from run contents:

| Fingerprint | Based on |
|-------------|----------|
| `program` | All program artifacts (role="program") |
| `canonical_program` | Canonical OpenQASM3 artifacts (cross-SDK comparable) |
| `device` | Backend identity + device snapshot artifacts |
| `intent` | Adapter + compile + execute configuration |
| `run` | Combined hash of program + device + intent |

Use fingerprints to detect drift/regressions and to enforce reproducibility in CI.

## Metric series

For iterative experiments (e.g., VQE optimization), metrics can be logged with a `step` parameter to create time series:

```python
for step in range(100):
    loss = optimizer.step()
    run.log_metric("loss", loss, step=step)
```

The final metric value (without step) is stored in `data.metrics`, while the full series is stored in `data.metric_series`.

## Run grouping and lineage

Runs can be organized using:

- **group_id** / **group_name** — group related runs together (parameter sweeps, benchmark suites, nightly calibration checks)
- **parent_run_id** — track run lineage (rerun from baseline, experiment iterations)

```python
# Parameter sweep with grouping
for shots in [100, 1000, 10000]:
    with track(project="bell", group_id="shots_sweep_001") as run:
        run.log_param("shots", shots)
        # ...

# Rerun from baseline
with track(project="bell", parent_run_id=baseline_run_id) as run:
    # ...
```

## Comparison and baseline verification

- **diff** compares two runs (or bundles) and reports what changed:
  - parameter/metric differences
  - program artifacts (exact and structural match)
  - device calibration drift
  - result distribution distance (TVD)
- **baseline** is a reference run per project
- **verify** checks a candidate run against the baseline using policy thresholds
