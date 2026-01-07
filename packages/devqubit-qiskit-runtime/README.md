# devqubit-qiskit-runtime

IBM Qiskit Runtime adapter for devqubit. Automatically captures circuits, results, and device calibration data from IBM Quantum hardware.

## Installation

```bash
pip install devqubit[qiskit-runtime]
```

## Usage

```python
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from devqubit import track

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

with track(project="hardware-run") as run:
    sampler = run.wrap(SamplerV2(backend))
    job = sampler.run([(qc,)])
    result = job.result()
```

## What's Captured

- **Circuits** — QPY format, transpiled circuits
- **Results** — Counts, quasi-distributions, pub results
- **Device data** — Calibration snapshot, coupling map, basis gates
- **Job metadata** — Job ID, timestamps, execution mode

## License

Apache 2.0
