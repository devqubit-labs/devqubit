# Tracking

This guide focuses on **manual** tracking primitives: parameters, metrics, tags, and artifacts.

If you are using an SDK adapter, most program/result/device capture happens automatically. See {doc}`adapters`.

## Basic tracking

```python
from devqubit import track

with track(project="my-experiment") as run:
    run.log_param("shots", 1000)
    run.log_metric("fidelity", 0.95)
    run.set_tag("backend", "simulator")

    print(run.run_id)
```

The context manager handles:

- run initialization with a unique ID
- status tracking (RUNNING → FINISHED/FAILED)
- error capture on exceptions
- fingerprint computation on exit

## Parameters

```python
run.log_param("shots", 1000)
run.log_params({"shots": 1000, "seed": 42})
```

## Metrics

```python
run.log_metric("fidelity", 0.95)

for step, loss in enumerate(losses):
    run.log_metric("loss", loss, step=step)
```

## Tags

Tags are key-value pairs (values are strings):

```python
run.set_tag("experiment", "bell-state")
run.set_tags({"device": "ibm_kyoto", "version": "1.0"})
```

## Artifacts

Artifacts are blobs/files stored in the workspace object store.

### JSON

```python
run.log_json(
    name="counts",
    obj={"00": 500, "11": 500},
    role="results",
    kind="result.counts.json",
)
```

### Bytes / binary payloads

```python
ref = run.log_bytes(
    kind="qiskit.qpy.circuits",
    data=qpy_bytes,
    media_type="application/x-qiskit-qpy",
    role="program",
)
print(ref.digest)
```

### Files

```python
run.log_file("circuit.qasm", kind="source.openqasm3", role="program")
```

## Run groups and lineage

Use groups to organize sweeps:

```python
for shots in [100, 1000, 10000]:
    with track(project="shot-sweep", group_id="sweep_20240115") as run:
        run.log_param("shots", shots)
```

Track parent-child relationships:

```python
with track(project="opt") as parent:
    parent_id = parent.run_id

with track(project="opt", parent_run_id=parent_id) as child:
    ...
```

## Error handling

Exceptions are captured automatically; failed runs are still persisted and queryable.

## Inspecting runs

```bash
devqubit list
devqubit show <run_id>
devqubit artifacts list <run_id>
```
