# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Transpilation utilities for Qiskit Runtime V2 primitives.

Qiskit Runtime V2 primitives only accept circuits (and for Estimator, observables)
that conform to the backend Target ISA, and V2 primitives do not perform layout,
routing, or translation.

This module implements three transpilation modes:
- 'auto'    : transpile only if circuits are not ISA-compatible (default)
- 'managed' : always transpile with devqubit
- 'manual'  : never transpile (user is responsible)

We build a preset pass manager using `qiskit.transpiler.generate_preset_pass_manager`,
which mirrors what `transpile()` internally uses.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

from devqubit_qiskit_runtime.pubs import extract_circuit_from_pub
from devqubit_qiskit_runtime.utils import get_backend_obj
from qiskit.circuit import QuantumCircuit


try:
    from qiskit.transpiler import generate_preset_pass_manager
except Exception:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


logger = logging.getLogger(__name__)


_OPTION_ALIASES: dict[str, str] = {
    "opt_level": "optimization_level",
    "seed": "seed_transpiler",
}

# Known kwargs for generate_preset_pass_manager
_KNOWN_KEYS = {
    "optimization_level",
    "target",
    "basis_gates",
    "coupling_map",
    "initial_layout",
    "layout_method",
    "routing_method",
    "translation_method",
    "scheduling_method",
    "approximation_degree",
    "seed_transpiler",
    "unitary_synthesis_method",
    "unitary_synthesis_plugin_config",
    "hls_config",
    "init_method",
    "optimization_method",
    "dt",
    "qubits_initially_zero",
}


@dataclass(frozen=True)
class TranspilationOptions:
    """
    Typed, sanitized options for Qiskit's preset pass manager.

    Users pass a dict; devqubit converts it into this dataclass once and then
    uses `.to_kwargs()` to build the kwargs for `generate_preset_pass_manager`.

    Attributes
    ----------
    optimization_level : int or None
        Qiskit preset optimization level in {0,1,2,3}.
    seed_transpiler : int or None
        Seed for randomized passes.
    layout_method : str or None
        Layout selection strategy.
    routing_method : str or None
        Routing strategy.
    translation_method : str or None
        Translation strategy.
    scheduling_method : str or None
        Scheduling strategy.
    initial_layout : Any or None
        User-provided initial layout.
    coupling_map : Any or None
        Custom coupling map.
    basis_gates : Any or None
        Custom basis gates.
    approximation_degree : float or None
        Approximation degree in [0.0, 1.0].
    target : Any or None
        Override target compilation object (advanced).
    dt : Any or None
        Backend time resolution.
    qubits_initially_zero : bool or None
        Whether qubits are assumed initially zeroed.
    extra : dict
        Unknown keys stored for metadata/tracking only. NOT passed to Qiskit.
    """

    optimization_level: int | None = None
    seed_transpiler: int | None = None
    layout_method: str | None = None
    routing_method: str | None = None
    translation_method: str | None = None
    scheduling_method: str | None = None

    initial_layout: Any = None
    coupling_map: Any = None
    basis_gates: Any = None

    approximation_degree: float | None = None
    target: Any = None
    dt: Any = None
    qubits_initially_zero: bool | None = None

    # Advanced knobs
    unitary_synthesis_method: Any = None
    unitary_synthesis_plugin_config: Any = None
    hls_config: Any = None
    init_method: Any = None
    optimization_method: Any = None

    # Unknown keys stored for tracking only - NOT passed to Qiskit
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        options: dict[str, Any] | None,
        *,
        strict: bool = False,
        warn_unknown: bool = True,
    ) -> TranspilationOptions:
        """
        Build TranspilationOptions from a user dictionary.

        Parameters
        ----------
        options : dict or None
            User-supplied options.
        strict : bool
            If True, unknown keys raise ValueError.
        warn_unknown : bool
            If True, emit warning for unknown keys when strict=False.

        Returns
        -------
        TranspilationOptions
            Parsed and validated options.

        Notes
        -----
        Unknown keys are stored in `extra` for metadata/tracking purposes,
        but they are NOT passed to Qiskit's `generate_preset_pass_manager`
        to avoid TypeError from unexpected arguments.
        """
        if not options:
            return cls()

        if not isinstance(options, dict):
            raise TypeError("devqubit_transpilation_options must be a dict (or None).")

        # Normalize aliases and drop None values
        normalized: dict[str, Any] = {}
        for k, v in options.items():
            if v is None:
                continue
            k2 = _OPTION_ALIASES.get(str(k), str(k))
            normalized[k2] = v

        # Disallow 'backend' here: backend should come from primitive
        if "backend" in normalized:
            normalized.pop("backend", None)
            warnings.warn(
                "devqubit: ignoring transpilation option 'backend' because the backend "
                "is taken from the Runtime primitive. Create the primitive with the desired "
                "backend/mode instead.",
                UserWarning,
                stacklevel=2,
            )

        # Split known vs extra
        known: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in normalized.items():
            if k in _KNOWN_KEYS:
                known[k] = v
            else:
                extra[k] = v

        if extra:
            if strict:
                raise ValueError(
                    f"Unknown transpilation option key(s): {sorted(extra.keys())}"
                )
            if warn_unknown:
                warnings.warn(
                    "devqubit: unknown transpilation option key(s) will be stored "
                    f"in metadata but NOT passed to Qiskit: {sorted(extra.keys())}. "
                    "This prevents TypeError from generate_preset_pass_manager.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate critical fields
        if "optimization_level" in known:
            lvl = int(known["optimization_level"])
            if lvl not in (0, 1, 2, 3):
                raise ValueError("optimization_level must be one of {0, 1, 2, 3}.")
            known["optimization_level"] = lvl

        if "seed_transpiler" in known:
            known["seed_transpiler"] = int(known["seed_transpiler"])

        if "approximation_degree" in known:
            a = float(known["approximation_degree"])
            if not (0.0 <= a <= 1.0):
                raise ValueError("approximation_degree must be in [0.0, 1.0].")
            known["approximation_degree"] = a

        if "qubits_initially_zero" in known:
            known["qubits_initially_zero"] = bool(known["qubits_initially_zero"])

        return cls(
            optimization_level=known.get("optimization_level"),
            seed_transpiler=known.get("seed_transpiler"),
            layout_method=known.get("layout_method"),
            routing_method=known.get("routing_method"),
            translation_method=known.get("translation_method"),
            scheduling_method=known.get("scheduling_method"),
            initial_layout=known.get("initial_layout"),
            coupling_map=known.get("coupling_map"),
            basis_gates=known.get("basis_gates"),
            approximation_degree=known.get("approximation_degree"),
            target=known.get("target"),
            dt=known.get("dt"),
            qubits_initially_zero=known.get("qubits_initially_zero"),
            unitary_synthesis_method=known.get("unitary_synthesis_method"),
            unitary_synthesis_plugin_config=known.get(
                "unitary_synthesis_plugin_config"
            ),
            hls_config=known.get("hls_config"),
            init_method=known.get("init_method"),
            optimization_method=known.get("optimization_method"),
            extra=dict(extra),
        )

    def to_kwargs(self, *, include_extra: bool = False) -> dict[str, Any]:
        """
        Convert options to kwargs for `generate_preset_pass_manager`.

        Parameters
        ----------
        include_extra : bool, optional
            If True, include unknown keys from `extra`. Default is False
            to prevent TypeError from Qiskit.

        Returns
        -------
        dict
            Keyword arguments with None values removed.
        """
        out: dict[str, Any] = {}

        # All known fields
        fields = [
            ("optimization_level", self.optimization_level),
            ("seed_transpiler", self.seed_transpiler),
            ("layout_method", self.layout_method),
            ("routing_method", self.routing_method),
            ("translation_method", self.translation_method),
            ("scheduling_method", self.scheduling_method),
            ("initial_layout", self.initial_layout),
            ("coupling_map", self.coupling_map),
            ("basis_gates", self.basis_gates),
            ("approximation_degree", self.approximation_degree),
            ("target", self.target),
            ("dt", self.dt),
            ("qubits_initially_zero", self.qubits_initially_zero),
            ("unitary_synthesis_method", self.unitary_synthesis_method),
            ("unitary_synthesis_plugin_config", self.unitary_synthesis_plugin_config),
            ("hls_config", self.hls_config),
            ("init_method", self.init_method),
            ("optimization_method", self.optimization_method),
        ]

        for key, val in fields:
            if val is not None:
                out[key] = val

        if include_extra:
            for k, v in (self.extra or {}).items():
                if v is not None:
                    out[k] = v

        return out

    def to_metadata_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-safe metadata dict for logging.

        Returns
        -------
        dict
            All options including extra, with non-serializable objects
            converted to string representations.
        """
        out: dict[str, Any] = {}

        # Simple scalar fields
        for attr in [
            "optimization_level",
            "seed_transpiler",
            "layout_method",
            "routing_method",
            "translation_method",
            "scheduling_method",
            "approximation_degree",
            "qubits_initially_zero",
            "dt",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                out[attr] = val

        # Complex objects: convert to string repr
        if self.initial_layout is not None:
            out["initial_layout"] = repr(self.initial_layout)[:500]
        if self.coupling_map is not None:
            out["coupling_map"] = repr(self.coupling_map)[:500]
        if self.basis_gates is not None:
            out["basis_gates"] = (
                list(self.basis_gates)
                if hasattr(self.basis_gates, "__iter__")
                else repr(self.basis_gates)[:500]
            )
        if self.target is not None:
            out["target"] = repr(self.target)[:200]

        # String conversion for methods
        for attr in ["unitary_synthesis_method", "init_method", "optimization_method"]:
            val = getattr(self, attr, None)
            if val is not None:
                out[attr] = str(val)

        # Extra keys (stored but not passed to Qiskit)
        if self.extra:
            out["_ignored_keys"] = list(self.extra.keys())

        return out


def apply_layout_to_observables(observables: Any, layout: Any) -> Any:
    """
    Apply transpiled circuit layout to Estimator observables when supported.

    Parameters
    ----------
    observables : Any
        Observable or collection of observables.
    layout : Any
        Circuit layout (typically isa_circuit.layout).

    Returns
    -------
    Any
        Mapped observables when possible; otherwise input unchanged.
    """
    if observables is None or layout is None:
        return observables

    if isinstance(observables, (list, tuple)):
        mapped = [apply_layout_to_observables(o, layout) for o in observables]
        return tuple(mapped) if isinstance(observables, tuple) else mapped

    fn = getattr(observables, "apply_layout", None)
    if callable(fn):
        try:
            return fn(layout)
        except Exception:
            return observables

    return observables


def _try_ibm_is_isa_circuit(circuit: QuantumCircuit, target: Any) -> bool | None:
    """
    Try IBM Runtime's ISA checker if importable.

    Returns
    -------
    bool or None
        True if IBM confirms ISA-compatible.
        False if IBM explicitly says not ISA-compatible.
        None if IBM checker is not available or errored.
    """
    try:
        from qiskit_ibm_runtime.utils.validations import is_isa_circuit
    except Exception:
        return None

    try:
        msg = is_isa_circuit(circuit, target)
        if msg is None:
            return True
        logger.debug("IBM ISA check failed: %s", msg[:200] if msg else "unknown")
        return False
    except Exception as e:
        logger.debug("IBM ISA checker raised exception: %s", e)
        return None


def _fallback_gate_check(circuit: QuantumCircuit, target: Any) -> bool:
    """
    Fallback gate-name compatibility check (heuristic).

    Only checks if gates are in the target's basis set. Does NOT validate
    connectivity, qubit coupling, or other constraints.

    This is a heuristic check that may produce false positives.
    """
    op_names = getattr(target, "operation_names", None)
    if not op_names:
        return True

    supported = set(op_names)
    for inst in circuit.data:
        name = getattr(inst.operation, "name", "") or ""
        if name in ("barrier", "delay", "measure", "reset"):
            continue
        if name not in supported:
            logger.debug("Gate '%s' not in target basis set", name)
            return False

    return True


def circuit_looks_isa(
    circuit: QuantumCircuit,
    target: Any,
    *,
    strict: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Check if a circuit appears to be ISA-compatible with a backend target.

    This is a heuristic check that cannot guarantee full ISA compatibility.
    The check examines gate names and may use IBM's is_isa_circuit when
    available, but does not verify all constraints (e.g., coupling map).

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to check.
    target : Any
        Backend target.
    strict : bool, optional
        If True (default), trust IBM's is_isa_circuit rejection.
        If False, only trust True results and fall back to gate-name check.

    Returns
    -------
    is_isa : bool
        True if circuit appears ISA-compatible; False otherwise.
    check_info : dict
        Metadata about the check performed:
        - method: "trivial", "ibm_is_isa_circuit", "has_layout", or "heuristic_gate_check"
        - passed: Whether the check passed
        - heuristic: True (always, since full validation is not performed)
    """
    check_info: dict[str, Any] = {
        "heuristic": True,
    }

    if target is None:
        check_info["method"] = "no_target"
        check_info["passed"] = True
        return True, check_info

    # Empty circuits are trivially ISA-compatible
    if not circuit.data:
        check_info["method"] = "trivial_empty"
        check_info["passed"] = True
        return True, check_info

    # Check if circuit only has trivial instructions
    has_non_trivial = any(
        getattr(inst.operation, "name", "") not in ("barrier", "delay")
        for inst in circuit.data
    )
    if not has_non_trivial:
        check_info["method"] = "trivial_no_ops"
        check_info["passed"] = True
        return True, check_info

    # If circuit has layout, it was already transpiled by a pass manager.
    # Verify gates are in basis set - if so, trust it as ISA-compatible.
    # This handles FakeBackends where IBM's is_isa_circuit can be overly strict.
    circuit_layout = getattr(circuit, "layout", None)
    if circuit_layout is not None:
        if _fallback_gate_check(circuit, target):
            check_info["method"] = "has_layout_gate_verified"
            check_info["passed"] = True
            return True, check_info
        # Layout present but gates don't match target - unusual, continue to IBM check
        logger.debug(
            "Circuit has layout but gates don't match target basis set, "
            "continuing to IBM ISA check"
        )

    # Try IBM's checker
    ibm_result = _try_ibm_is_isa_circuit(circuit, target)

    if ibm_result is True:
        check_info["method"] = "ibm_is_isa_circuit"
        check_info["passed"] = True
        return True, check_info
    if ibm_result is False and strict:
        # In strict mode, trust IBM's rejection UNLESS circuit has layout
        # (layout implies user already transpiled, IBM checker may be wrong for FakeBackends)
        if circuit_layout is not None:
            logger.debug(
                "IBM ISA check failed but circuit has layout - "
                "falling back to gate check"
            )
        else:
            check_info["method"] = "ibm_is_isa_circuit"
            check_info["passed"] = False
            return False, check_info

    # Fallback: simple gate-name check
    fallback_result = _fallback_gate_check(circuit, target)
    check_info["method"] = "heuristic_gate_check"
    check_info["passed"] = fallback_result
    return fallback_result, check_info


def circuit_looks_isa_simple(
    circuit: QuantumCircuit,
    target: Any,
    *,
    strict: bool = True,
) -> bool:
    """
    Check if a circuit appears to be ISA-compatible with a backend target.

    Simplified wrapper that returns only bool result. Use circuit_looks_isa()
    when metadata about the check method is needed.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to check.
    target : Any
        Backend target.
    strict : bool, optional
        If True (default), trust IBM's is_isa_circuit rejection.

    Returns
    -------
    bool
        True if circuit appears ISA-compatible; False otherwise.
    """
    is_isa, _ = circuit_looks_isa(circuit, target, strict=strict)
    return is_isa


@dataclass(frozen=True)
class TranspilationConfig:
    """
    Configuration for devqubit-managed transpilation.

    Attributes
    ----------
    mode : str
        'auto' | 'managed' | 'manual'
    options : TranspilationOptions
        Typed/sanitized transpilation options.
    pass_manager : Any or None
        User-supplied pass manager (advanced).
    map_observables : bool
        If True and primitive_type='estimator', map observables with apply_layout.
    strict_isa_check : bool
        If True (default), use strict ISA checking that trusts IBM's rejection.
    """

    mode: str = "auto"
    options: TranspilationOptions = field(default_factory=TranspilationOptions)
    pass_manager: Any = None
    map_observables: bool = True
    strict_isa_check: bool = True


def prepare_pubs_for_primitive(
    pubs_list: list[Any],
    primitive: Any,
    primitive_type: str,
    config: TranspilationConfig,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Prepare PUBs for submission to a Runtime primitive.

    Parameters
    ----------
    pubs_list : list
        Materialized PUB list.
    primitive : Any
        Runtime primitive (Sampler/Estimator V2).
    primitive_type : str
        'sampler' or 'estimator'.
    config : TranspilationConfig
        Transpilation config.

    Returns
    -------
    pubs_to_run : list
        Pubs suitable for primitive.run(...).
    meta : dict
        Metadata describing actions taken.
    """
    mode = config.mode
    opts = config.options.to_kwargs(include_extra=False)
    opts_meta = config.options.to_metadata_dict()

    meta: dict[str, Any] = {
        "transpilation_mode": mode,
        "transpiled_by_devqubit": False,
        "transpilation_needed": False,
        "transpilation_reason": None,
        "transpilation_options": opts_meta,
        "observables_layout_mapped": False,
    }

    backend = get_backend_obj(primitive)
    target = getattr(backend, "target", None) if backend is not None else None
    if target is None:
        meta["transpilation_reason"] = "no_target"
        return pubs_list, meta

    circuits = [
        c for c in (extract_circuit_from_pub(p) for p in pubs_list) if c is not None
    ]
    if not circuits:
        meta["transpilation_reason"] = "no_circuits"
        return pubs_list, meta

    # Check ISA compatibility (heuristic)
    isa_check_results: list[dict[str, Any]] = []
    all_isa = True
    for c in circuits:
        is_isa, check_info = circuit_looks_isa(
            c, target, strict=config.strict_isa_check
        )
        isa_check_results.append(check_info)
        if not is_isa:
            all_isa = False

    needs = not all_isa
    meta["transpilation_needed"] = needs
    meta["isa_check"] = {
        "all_passed": all_isa,
        "num_circuits": len(circuits),
        "checks": isa_check_results,
    }

    if mode == "manual":
        meta["transpilation_reason"] = "manual_mode"
        if needs:
            logger.warning(
                "devqubit: manual mode but circuits don't look ISA-compatible. "
                "Runtime may reject them."
            )
        return pubs_list, meta

    if mode == "auto" and not needs:
        meta["transpilation_reason"] = "already_isa"
        return pubs_list, meta

    # Build / use pass manager
    pm = config.pass_manager
    if pm is None:
        if "target" in opts:
            pm = generate_preset_pass_manager(**opts)
        else:
            pm = generate_preset_pass_manager(backend=backend, **opts)

    out: list[Any] = []

    for pub in pubs_list:
        # A) Bare circuit
        if isinstance(pub, QuantumCircuit):
            out.append(pm.run(pub))
            continue

        # B) Tuple/list pub (circuit first)
        if (
            isinstance(pub, (tuple, list))
            and len(pub) >= 1
            and isinstance(pub[0], QuantumCircuit)
        ):
            isa_circ = pm.run(pub[0])
            layout = getattr(isa_circ, "layout", None)

            if primitive_type == "estimator":
                obs = pub[1] if len(pub) >= 2 else None
                if config.map_observables:
                    obs = apply_layout_to_observables(obs, layout)
                out.append((isa_circ, obs, *pub[2:]))
            else:
                out.append((isa_circ, *pub[1:]))
            continue

        # C) Dict pub with 'circuit'
        if isinstance(pub, dict) and isinstance(pub.get("circuit"), QuantumCircuit):
            isa_circ = pm.run(pub["circuit"])
            layout = getattr(isa_circ, "layout", None)

            if primitive_type == "estimator":
                obs = pub.get("observables", pub.get("observable"))
                if config.map_observables:
                    obs = apply_layout_to_observables(obs, layout)
                pvals = pub.get("parameter_values")
                precision = pub.get("precision")
                tup: tuple[Any, ...] = (isa_circ, obs)
                if pvals is not None:
                    tup = (*tup, pvals)
                if precision is not None:
                    tup = (*tup, precision)
                out.append(tup)
            else:
                pvals = pub.get("parameter_values")
                shots = pub.get("shots")
                tup = (isa_circ,)
                if pvals is not None:
                    tup = (*tup, pvals)
                if shots is not None:
                    tup = (*tup, shots)
                out.append(tup)
            continue

        # D) Object pub with .circuit
        c = getattr(pub, "circuit", None)
        if isinstance(c, QuantumCircuit):
            isa_circ = pm.run(c)
            layout = getattr(isa_circ, "layout", None)

            if primitive_type == "estimator":
                obs = getattr(pub, "observables", None)
                if config.map_observables:
                    obs = apply_layout_to_observables(obs, layout)
                pvals = getattr(pub, "parameter_values", None)
                precision = getattr(pub, "precision", None)
                tup = (isa_circ, obs)
                if pvals is not None:
                    tup = (*tup, pvals)
                if precision is not None:
                    tup = (*tup, precision)
                out.append(tup)
            else:
                pvals = getattr(pub, "parameter_values", None)
                shots = getattr(pub, "shots", None)
                tup = (isa_circ,)
                if pvals is not None:
                    tup = (*tup, pvals)
                if shots is not None:
                    tup = (*tup, shots)
                out.append(tup)
            continue

        out.append(pub)

    meta["transpiled_by_devqubit"] = True
    meta["transpilation_reason"] = "transpiled"
    if primitive_type == "estimator" and config.map_observables:
        meta["observables_layout_mapped"] = True

    return out, meta
