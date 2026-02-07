# Tracking

This guide covers **manual** tracking primitives. If you're using an SDK adapter, most capture happens automatically — see {doc}`adapters`.

## Basic Tracking

```python
from devqubit import track

with track(project="my-experiment") as run:
    run.log_param("shots", 1000)
    run.log_metric("fidelity", 0.95)
    run.set_tag("backend", "simulator")
    print(run.run_id)
```

The context manager handles run initialization, status tracking (RUNNING → FINISHED/FAILED), error capture, and fingerprint computation.

## Parameters, Metrics, Tags

```python
# Parameters (configuration values)
run.log_param("shots", 1000)
run.log_params({"shots": 1000, "seed": 42})

# Metrics (numeric results, optional time series)
run.log_metric("fidelity", 0.95)
for step, loss in enumerate(losses):
    run.log_metric("loss", loss, step=step)

# Tags (string key-value pairs)
run.set_tag("experiment", "bell-state")
run.set_tags({"device": "ibm_kyoto", "version": "1.0"})
```

## Artifacts

Artifacts are blobs stored in the workspace object store.

```python
# JSON
run.log_json(
    name="counts",
    obj={"00": 500, "11": 500},
    role="results",
)

# Binary
ref = run.log_bytes(
    kind="qiskit.qpy.circuits",
    data=qpy_bytes,
    media_type="application/x-qiskit-qpy",
    role="program",
)
print(ref.digest)

# Files
run.log_file("circuit.qasm", kind="source.openqasm3", role="program")
```

## Run Groups and Lineage

```python
# Parameter sweeps
for shots in [100, 1000, 10000]:
    with track(project="shot-sweep", group_id="sweep_20240115") as run:
        run.log_param("shots", shots)

# Parent-child relationships
with track(project="opt") as parent:
    parent_id = parent.run_id

with track(project="opt", parent_run_id=parent_id) as child:
    ...
```

Exceptions are captured automatically; failed runs are persisted and queryable.

## Analyzing Runs

Export runs to a pandas DataFrame for analysis and plotting:

```python
from devqubit.runs import runs_to_dataframe

# All runs in a project
df = runs_to_dataframe(project="vqe")

# Columns: run_id, created_at, status, param.shots, metric.fidelity, tag.backend, ...
df.sort_values("metric.fidelity", ascending=False).head(10)
```

Filter by status, time range, or group:

```python
# Only finished runs from last month
df = runs_to_dataframe(
    project="vqe",
    status="FINISHED",
    since="2026-01-01",
    until="2026-01-31",
)

# All runs in a parameter sweep
df = runs_to_dataframe(group_id="sweep_20260115")
```

Requires `pandas` package installed.

## Inspecting Runs

```bash
devqubit list
devqubit show <run_id>
devqubit artifacts list <run_id>
```
