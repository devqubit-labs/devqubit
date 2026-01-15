# Uniform Execution Contract (UEC)

Quantum experiments depend on more than code: they depend on *circuits/programs*, the *device/backend*, compilation/execution settings, and hardware calibration.

To keep runs reproducible and comparable, adapters produce a standardized **ExecutionEnvelope** containing four snapshots that capture the complete execution context.

## ExecutionEnvelope structure

```
ExecutionEnvelope
├── schema: "devqubit.envelope/1.0"
├── envelope_id, created_at       # Envelope metadata
├── producer: ProducerInfo        # SDK stack + versions
├── device: DeviceSnapshot        # Backend state and calibration
├── program: ProgramSnapshot      # Circuit artifacts and hashes
├── execution: ExecutionSnapshot  # Job metadata and settings
└── result: ResultSnapshot        # Normalized results (items[])
```

The envelope is stored as an artifact with role `envelope` (typically kind `devqubit.envelope.json`).

The envelope schema is `devqubit.envelope/1.0` and requires `envelope_id`, `created_at`, `producer`, and `result`.


## Snapshots

### ProducerInfo

Captures the complete SDK/toolchain stack that produced the envelope:

| Field | Description |
|-------|-------------|
| `name` | Producer name (always `"devqubit"`) |
| `engine_version` | devqubit-engine version |
| `adapter` | Adapter package name (e.g., `devqubit-qiskit`) |
| `adapter_version` | Adapter version |
| `sdk` | Lowest/primary SDK (e.g., `qiskit`, `braket-sdk`, `cirq`) |
| `sdk_version` | Primary SDK version |
| `frontends` | Ordered SDK stack from highest to lowest layer |
| `build` | Optional build identifier (commit/dirty flag) |


### DeviceSnapshot

Captures backend state at execution time:

| Field | Description |
|-------|-------------|
| `backend_name` | Backend identifier (e.g., "ibm_brisbane", "aer_simulator") |
| `backend_type` | Backend type (e.g., `"hardware"`, `"simulator"`, `"emulator"`, `"unknown"`) |
| `provider` | Physical provider (e.g., `"ibm_quantum"`, `"aws_braket"`, `"local"`) |
| `num_qubits` | Number of qubits |
| `connectivity` | Qubit coupling map as edge list |
| `native_gates` | Supported gate set |
| `calibration` | Extracted calibration metrics (T1, T2, gate errors, readout errors) |
| `sdk_versions` | SDK version information |
| `raw_properties_ref` | Reference to full raw properties artifact (for lossless capture) |

The `calibration` field contains aggregated statistics (median T1/T2, median gate errors) useful for drift detection without needing the full calibration data.

### ProgramSnapshot

Captures circuit/program artifacts:

| Field | Description |
|-------|-------------|
| `logical` | Logical program artifacts (before transpilation) |
| `physical` | Physical program artifacts (after transpilation, if captured) |
| `program_hash` | Structural hash for deduplication |
| `num_circuits` | Number of circuits in this execution |
| `transpilation` | Transpilation metadata (mode, settings) |

Each artifact in `logical`/`physical` includes:
- `format`: Circuit format (QPY, QASM3, etc.)
- `ref`: Reference to stored artifact
- `index`: Index in multi-circuit batch
- `name`: Circuit name (if available)

### ExecutionSnapshot

Captures submission and job metadata:

| Field | Description |
|-------|-------------|
| `submitted_at` | ISO timestamp of submission |
| `shots` | Number of shots requested |
| `job_ids` | Provider job IDs |
| `execution_count` | Execution counter within run |
| `transpilation` | Transpilation info (mode, transpiled_by) |
| `options` | Raw execution options (args, kwargs) |
| `sdk` | Optional legacy field (prefer `producer.sdk` / `producer.frontends`) |

### ResultSnapshot

Captures normalized execution results (always as a list of per-item results):

| Field | Description |
|-------|-------------|
| `success` | Overall execution success |
| `status` | `"completed"`, `"failed"`, `"cancelled"`, or `"partial"` |
| `items` | List of `ResultItem` (one per circuit/parameter-set) |
| `error` | Structured error info when failed |
| `raw_result_ref` | Reference to the full serialized SDK result (optional) |
| `metadata` | Additional result metadata |

Each `ResultItem` may contain one primary payload, e.g. `counts`, `quasi_probability`, or `expectation`. If `counts` are present, they include `format` metadata (source SDK + bit ordering) to make results comparable across SDKs.

## Why UEC matters

The Uniform Execution Contract makes it easier to:

- **Compare runs across devices and SDKs** — normalized structure enables apples-to-apples comparison
- **Detect device drift** — calibration data in DeviceSnapshot enables drift analysis between runs
- **Share self-contained bundles** — envelope contains everything needed to analyze results
- **Debug failures** — complete context captured even for failed runs

## Data flow

```
+---------------------+     +---------------------+     +---------------------+
|       Adapter       |---->|   Envelope (UEC)    |---->|   Artifact Store    |
| (qiskit, pennylane) |     |                     |     |                     |
+---------------------+     +---------------------+     +---------------------+
          |                           |
          |                           v
          |                 +---------------------+
          +---------------->|     Run Record      |
                            |     (summary)       |
                            +---------------------+
```

The adapter creates the ExecutionEnvelope and logs it as an artifact. Summary information is also written to the run record for efficient querying without loading full artifacts.

## Accessing envelope data

```python
import json
from devqubit.storage import create_store, create_registry

store = create_store()
registry = create_registry()

# Load run record
record = registry.load(run_id)

# Find envelope artifact
envelope_artifact = next(
    (a for a in record.artifacts if a.role == "envelope"),
    None
)

if envelope_artifact:
    # Load envelope JSON
    envelope_bytes = store.get_bytes(envelope_artifact.digest)
    envelope = json.loads(envelope_bytes)

    # Access device snapshot
    device = envelope["device"]
    print(f"Backend: {device['backend_name']}")
    print(f"Qubits: {device['num_qubits']}")
    # Access results (per-item)
    for item in envelope["result"]["items"]:
        if "counts" in item:
            counts = item["counts"]["counts"]
            print(f"Item {item['item_index']}: {counts}")
        elif "quasi_probability" in item:
            dist = item["quasi_probability"]["distribution"]
            print(f"Item {item['item_index']} quasi: {dist}")
        elif "expectation" in item:
            value = item["expectation"]["value"]
            print(f"Item {item['item_index']} expval: {value}")
```

See `../guides/adapters` for what each SDK adapter captures.
