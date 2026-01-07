# Installation

devqubit is designed to run **locally** (local-first) and to be easy to adopt in existing experiments.

## Install the core

```bash
pip install devqubit
```

If you prefer `uv`:

```bash
uv pip install devqubit
```

Verify:

```bash
devqubit --version
```

## Install an SDK adapter (optional but recommended)

Pick your SDK:

```bash
pip install devqubit-qiskit          # IBM Qiskit (local, Aer)
pip install devqubit-qiskit-runtime  # IBM Qiskit Runtime (cloud primitives)
pip install devqubit-braket          # Amazon Braket
pip install devqubit-cirq            # Google Cirq
pip install devqubit-pennylane       # Xanadu PennyLane
```

## Where data is stored

By default devqubit stores all runs in `~/.devqubit`. You can override it using `DEVQUBIT_HOME`.

See {doc}`../concepts/workspace` and {doc}`../guides/configuration`.
