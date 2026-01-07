# devqubit-qiskit

Qiskit adapter for devqubit. Automatically captures circuits, results, and backend information from Qiskit and Aer.

## Installation

```bash
pip install devqubit[qiskit]
```

## Usage

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from devqubit import track

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

with track(project="bell-state") as run:
    backend = run.wrap(AerSimulator())
    job = backend.run(qc, shots=1000)
    counts = job.result().get_counts()
```

## What's Captured

- **Circuits** — QPY format, OpenQASM 3
- **Results** — Counts, quasi-distributions
- **Backend info** — Name, configuration, coupling map

## License

Apache 2.0
