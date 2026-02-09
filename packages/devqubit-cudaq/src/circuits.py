# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Kernel hashing for the CUDA-Q adapter.

Primary path: ``kernel.to_json()`` → op_stream → ``hash_circuit_pair()``.
Fallback: hash the raw JSON string.
Diagram parsing exists only for gate-level summarization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from devqubit_cudaq.utils import get_kernel_name
from devqubit_engine.circuit.hashing import hash_circuit_pair


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Native JSON → op_stream
# ---------------------------------------------------------------------------


def _get_native_json(kernel: Any) -> str | None:
    """Get JSON string from ``kernel.to_json()``, or ``None``."""
    try:
        fn = getattr(kernel, "to_json", None)
        if fn is not None and callable(fn):
            result = fn()
            if isinstance(result, str) and result:
                return result
    except Exception as exc:
        logger.debug("kernel.to_json() failed: %s", exc)
    return None


def _native_json_to_op_stream(
    native_json: str,
) -> tuple[list[dict[str, Any]], int] | None:
    """
    Parse CUDA-Q native JSON into canonical op_stream.

    Probes the JSON for a list of gate instructions under common
    schema keys (``instructions``, ``operations``, ``gates``).
    Returns ``None`` if the schema is not recognised — the caller
    then falls back to hashing the raw JSON string.
    """
    try:
        data = json.loads(native_json)
    except json.JSONDecodeError:
        return None

    instructions = (
        data.get("instructions") or data.get("operations") or data.get("gates")
    )
    if not isinstance(instructions, list) or not instructions:
        return None

    ops: list[dict[str, Any]] = []
    num_qubits = 0

    for instr in instructions:
        if not isinstance(instr, dict):
            continue

        gate = instr.get("gate") or instr.get("name") or instr.get("op") or "unknown"
        qubits = instr.get("qubits", instr.get("targets", []))
        if isinstance(qubits, int):
            qubits = [qubits]
        controls = instr.get("controls", [])
        if isinstance(controls, int):
            controls = [controls]
        all_qubits = list(controls) + list(qubits)

        op: dict[str, Any] = {
            "gate": str(gate).lower(),
            "qubits": all_qubits,
            "clbits": [],
        }

        raw_params = instr.get("params", instr.get("parameters"))
        if raw_params is not None:
            if isinstance(raw_params, (list, tuple)):
                op["params"] = {
                    f"p{i}": float(p) if isinstance(p, (int, float)) else None
                    for i, p in enumerate(raw_params)
                }
            elif isinstance(raw_params, (int, float)):
                op["params"] = {"p0": float(raw_params)}

        ops.append(op)
        if all_qubits:
            num_qubits = max(num_qubits, max(all_qubits) + 1)

    return (ops, num_qubits) if ops else None


# ---------------------------------------------------------------------------
# Hashing — public API
# ---------------------------------------------------------------------------


def _sha256_tag(payload: str) -> str:
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def compute_circuit_hashes(
    kernel: Any,
    args: tuple[Any, ...] = (),
) -> tuple[str | None, str | None]:
    """
    Compute structural and parametric hashes for a CUDA-Q kernel.

    1. ``to_json()`` → op_stream → ``hash_circuit_pair()``
    2. If schema unknown → hash raw JSON string
    3. Last resort → hash kernel name

    Parameters
    ----------
    kernel : Any
        CUDA-Q kernel.
    args : tuple
        Concrete kernel arguments.

    Returns
    -------
    (structural_hash, parametric_hash)
        Both ``sha256:<hex>`` or ``None``.
    """
    native_json = _get_native_json(kernel)

    if native_json is not None:
        # Try structured op_stream
        try:
            parsed = _native_json_to_op_stream(native_json)
            if parsed is not None:
                ops, nq = parsed
                all_ops: list[dict[str, Any]] = [
                    {"gate": "__kernel__", "qubits": [], "meta": {"nq": nq, "nc": 0}},
                ]
                all_ops.extend(ops)
                return hash_circuit_pair(all_ops, nq, 0)
        except Exception as exc:
            logger.debug("op_stream hashing failed: %s", exc)

        # Fallback: hash raw JSON
        h = _sha256_tag(native_json)
        return h, h

    # Last resort: kernel name
    try:
        h = _sha256_tag(get_kernel_name(kernel))
        return h, h
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Diagram parsing (for summarization ONLY)
# ---------------------------------------------------------------------------

_GATE_RE = re.compile(
    r"[┤|]\s*([a-zA-Z_]\w*)(?:\(([^)]*)\))?\s*[├|]",
)
_CTRL_RE = re.compile(r"●")
_WIRE_LABEL_RE = re.compile(r"^(q\d+)\s*:")


def _parse_diagram_to_ops(diagram: str) -> tuple[list[dict[str, Any]], int]:
    """
    Parse ``cudaq.draw()`` output into operation dicts.

    Used **only** by ``summarize_cudaq_kernel()`` — not for hashing.
    """
    if not diagram:
        return [], 0

    lines = diagram.strip().splitlines()
    wire_lines: dict[int, str] = {}

    for line in lines:
        m = _WIRE_LABEL_RE.match(line.strip())
        if m:
            idx = int(m.group(1).replace("q", ""))
            wire_lines[idx] = line

    num_qubits = max(wire_lines.keys(), default=-1) + 1 if wire_lines else 0
    ops: list[dict[str, Any]] = []

    for wire_idx, wire_content in sorted(wire_lines.items()):
        for gate_match in _GATE_RE.finditer(wire_content):
            gate_name = gate_match.group(1).lower()
            params_str = gate_match.group(2)

            params: dict[str, Any] = {}
            if params_str:
                for i, p in enumerate(params_str.split(",")):
                    p = p.strip()
                    try:
                        params[f"p{i}"] = float(p)
                    except ValueError:
                        params[f"p{i}"] = None

            op: dict[str, Any] = {
                "gate": gate_name,
                "qubits": [wire_idx],
                "clbits": [],
            }
            if params:
                op["params"] = params
            ops.append(op)

    for line in lines:
        for _ in _CTRL_RE.finditer(line):
            ops.append({"gate": "__ctrl__", "qubits": [], "clbits": []})

    return ops, num_qubits
