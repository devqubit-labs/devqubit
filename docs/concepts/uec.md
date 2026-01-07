# Uniform Execution Contract (UEC)

Quantum experiments depend on more than code: they depend on *circuits/programs*, the *device/backend*, compilation/execution settings, and hardware calibration.

To keep runs reproducible and comparable, adapters produce a standardized **ExecutionEnvelope** containing four snapshots that capture the complete execution context.

## ExecutionEnvelope structure

```
ExecutionEnvelope
├── device: DeviceSnapshot       # Backend state and calibration
├── program: ProgramSnapshot     # Circuit artifacts and hashes
├── execution: ExecutionSnapshot # Job metadata and settings
└── result: ResultSnapshot       # Normalized measurement results
```

The envelope is stored as an artifact with role `envelope` (typically kind `devqubit.envelope.json`).

## Snapshots

### DeviceSnapshot

Captures backend state at execution time:

| Field | Description |
|-------|-------------|
| `backend_name` | Backend identifier (e.g., "ibm_brisbane", "aer_simulator") |
| `backend_type` | "simulator" or "hardware" |
| `provider` | Provider name (e.g., "ibm_quantum", "aer") |
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
- `artifact_ref`: Reference to stored artifact
- `circuit_index`: Index in multi-circuit batch
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
| `sdk` | SDK used for execution |

### ResultSnapshot

Captures normalized execution results:

| Field | Description |
|-------|-------------|
| `result_type` | Type of result (counts, quasi_dist, expectation, etc.) |
| `raw_result_ref` | Reference to full serialized result artifact |
| `counts` | Normalized measurement counts per circuit |
| `num_experiments` | Number of experiments in result |
| `success` | Whether execution succeeded |
| `error_message` | Error message if failed |
| `metadata` | Additional result metadata |

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
from devqubit import create_store, create_registry

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

    # Access results
    for counts in envelope["result"]["counts"]:
        print(f"Circuit {counts['circuit_index']}: {counts['counts']}")
```

See {doc}`../guides/adapters` for what each SDK adapter captures.
