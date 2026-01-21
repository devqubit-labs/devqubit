# Core Concepts

devqubit treats each execution as a **run** — a complete, tracked experiment with enough context to:

- reproduce what happened,
- compare results across time / devices / SDKs,
- debug failures (even partial ones),
- gate changes in CI via verification policies.

> **Terminology**
> - **Run Record**: lightweight metadata + user logs + pointers to artifacts (`devqubit.run/1.0` schema).
> - **Artifacts**: content-addressed blobs (SHA-256) stored in an object store.
> - **UEC / ExecutionEnvelope**: canonical, structured execution context produced by adapters (`devqubit.envelope/1.0` schema).

## How It Works

When you wrap a backend with `run.wrap()`, devqubit intercepts executions and automatically captures circuits, device state, and results. Manual logging (`log_param`, `log_metric`) is stored alongside. Everything flows into a content-addressed store for deduplication and integrity, with queryable metadata in a registry.

```mermaid
flowchart TB
  subgraph USER["<b>User Code</b>"]
    direction TB
    TRACK["with track(project) as run"]
    LOG["log_param()<br/>log_metric()<br/>set_tag()"]
    WRAP["backend = run.wrap(device)"]
    EXEC["backend.run(circuit, shots=1000)"]
    TRACK --> LOG
    TRACK --> WRAP
    WRAP --> EXEC
  end

  subgraph CAPTURE["<b>Adapter</b> automatic capture"]
    direction LR
    CAP_PRG["📄 Circuit"]
    CAP_DEV["⚙️ Device"]
    CAP_RES["📊 Results"]
  end

  subgraph PERSIST["<b>Storage</b>"]
    direction TB
    ENV["ExecutionEnvelope"]
    RR["Run Record"]
    STORE[("Object Store<br/>content-addressed")]
    REG[("Registry<br/>queryable index")]
    ENV -->|artifact| STORE
    RR --> REG
    STORE -.->|refs| RR
  end

  subgraph TOOLS["<b>Analysis</b>"]
    direction LR
    DIFF["diff()"]
    VERIFY["verify()"]
    DIFF ~~~ VERIFY
  end

  EXEC --> CAPTURE
  CAP_PRG --> ENV
  CAP_DEV --> ENV
  CAP_RES --> ENV
  LOG --> RR
  REG --> TOOLS
  STORE --> TOOLS

  classDef userNode fill:#e8f4fd,stroke:#1e88e5,color:#0d47a1
  classDef captureNode fill:#fff3e0,stroke:#fb8c00,color:#e65100
  classDef storageNode fill:#e8f5e9,stroke:#43a047,color:#1b5e20
  classDef toolNode fill:#f3e5f5,stroke:#8e24aa,color:#4a148c

  class TRACK,LOG,WRAP,EXEC userNode
  class CAP_PRG,CAP_DEV,CAP_RES captureNode
  class ENV,RR,STORE,REG storageNode
  class DIFF,VERIFY toolNode
```

## What Is Persisted Where?

| Store | Content | Purpose |
|-------|---------|---------|
| **Object store** | Immutable blobs by SHA-256 digest | Deduplication, integrity, offline bundles |
| **Registry** | Run records (run_id, project, timestamps, fingerprints, artifact pointers) | Queries, listing, baseline management |

## Run

A run captures everything about a single experiment execution. Run records follow the `devqubit.run/1.0` schema:

| Category | Description |
|----------|-------------|
| **Metadata** | Project, timestamps, status, run name, adapter |
| **Parameters** | Configuration values via `log_param()` |
| **Metrics** | Numeric results via `log_metric()` (scalar or time series) |
| **Tags** | String key-value pairs via `set_tag()` |
| **Artifacts** | Programs (QASM/QPY), results, device snapshots, envelopes, notes |
| **Fingerprints** | Stable hashes for reproducibility and comparison |
| **Environment** | Python + packages (optional capture) |
| **Provenance** | Git commit/branch/dirty state (optional capture) |

## Run Lifecycle

```mermaid
stateDiagram-v2
  direction LR
  [*] --> RUNNING
  RUNNING --> FINISHED: success
  RUNNING --> FAILED: exception
  RUNNING --> KILLED: interrupt
  FINISHED --> [*]
  FAILED --> [*]
  KILLED --> [*]
```

**Robustness guarantees:**
- Best-effort finalization — failures during finalization are recorded; the system still attempts to persist.
- Content integrity — artifacts are addressed by digest; corruption is detectable.
- Schema evolution — explicit `schema` field enables backwards-compatible readers.

## Artifacts

Artifacts are immutable blobs stored by digest, enabling deduplication, caching, and integrity verification.

| Role | Description |
|------|-------------|
| `program` | Circuit/program artifacts (QPY, QASM) — used for fingerprinting |
| `results_raw` | Raw SDK result payloads |
| `device_raw` | Raw backend properties (lossless capture) |
| `envelope` | ExecutionEnvelope (UEC JSON) |
| `config` | Compile/execute options, environment snapshots |
| `documentation` | Notes, attachments |

Artifact ingestion enforces a maximum size (~20 MB default). For larger blobs, store a URI pointer or truncate with `meta.original_digest` marker.

## Fingerprints

Fingerprints are stable hashes computed from run contents, excluding volatile fields (timestamps, job IDs).

| Fingerprint | Based on |
|-------------|----------|
| `program` | Program hashes (from UEC program snapshot) |
| `device` | Device identity + stable snapshots |
| `intent` | Adapter + SDK + compile/execute config |
| `run` | Combined fingerprint of program + device + intent |

Use fingerprints to detect what changed between runs — same `program` fingerprint means same circuit structure, even if run at different times.

## Comparison and Verification

**diff** compares two runs across multiple dimensions: parameter/metric changes, program match (digest/structural/parametric), device drift (calibration deltas), and result distribution distance (TVD with optional bootstrap noise context).

**verify** checks a candidate run against a baseline with a policy: required equality constraints (params/program), TVD thresholds (hard limit or noise-calibrated), and produces a human-readable verdict with root-cause analysis.

```python
from devqubit import diff, verify_baseline

# Compare two runs
result = diff("run_a", "run_b")
print(result.tvd, result.program.structural_match)

# CI verification
result = verify_baseline("candidate", project="bell")
assert result.ok, result.verdict.summary
```
