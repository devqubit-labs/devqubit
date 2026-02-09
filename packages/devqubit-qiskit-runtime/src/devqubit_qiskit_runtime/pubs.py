# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
PUB (Primitive Unified Bloc) handling for Qiskit Runtime adapter.

Provides functions for materializing PUB inputs, extracting circuits, and
generating metadata summaries suitable for logging following the devqubit
Uniform Execution Contract (UEC).

PUBs are the standard input format for Runtime V2 primitives (SamplerV2,
EstimatorV2). They combine circuits with optional parameters, shots, and
observables.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.circuit import QuantumCircuit


def is_v2_pub_tuple(obj: Any) -> bool:
    """
    Check whether an object looks like a V2 PUB tuple.

    A V2 PUB tuple has a QuantumCircuit as the first element, and may include
    additional broadcasted data (parameter values, shots, observables, precision).

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True if `obj` resembles a V2 PUB tuple, otherwise False.

    Examples
    --------
    >>> is_v2_pub_tuple((circuit, params))
    True
    >>> is_v2_pub_tuple((circuit1, circuit2))  # Container of circuits
    False
    """
    if not isinstance(obj, tuple) or len(obj) == 0:
        return False
    if not isinstance(obj[0], QuantumCircuit):
        return False

    # If this tuple is actually a container of circuits, treat it as a container
    if len(obj) > 1 and all(isinstance(x, QuantumCircuit) for x in obj):
        return False

    return True


def materialize_pubs(pubs: Any) -> list[Any]:
    """
    Materialize PUB inputs into a concrete list.

    This function is called once in the adapter so that the same list can be
    used for both logging and execution (avoiding generator consumption).

    Parameters
    ----------
    pubs : Any
        PUB input in any supported form:
        - None
        - QuantumCircuit
        - dict-like pub with "circuit"
        - pub object with .circuit / .observables / .parameter_values
        - V2 PUB tuple starting with a circuit
        - list/tuple/generator/iterable of any of the above

    Returns
    -------
    list
        Materialized list of PUB items.

    Examples
    --------
    >>> materialize_pubs(circuit)
    [circuit]
    >>> materialize_pubs([circuit1, circuit2])
    [circuit1, circuit2]
    >>> materialize_pubs((circuit, params))
    [(circuit, params)]
    """
    if pubs is None:
        return []

    # Single circuit (QuantumCircuit is iterable, so must special-case)
    if isinstance(pubs, QuantumCircuit):
        return [pubs]

    # Single PUB tuple (circuit, ...)
    if is_v2_pub_tuple(pubs):
        return [pubs]

    # Dict-like single pub
    if isinstance(pubs, dict) and "circuit" in pubs:
        return [pubs]

    # Object-like single pub
    if (
        hasattr(pubs, "circuit")
        or hasattr(pubs, "observables")
        or hasattr(pubs, "parameter_values")
    ):
        return [pubs]

    # Concrete containers
    if isinstance(pubs, (list, tuple)):
        return list(pubs)

    # Generic iterables (e.g., generators)
    if isinstance(pubs, Iterable):
        return list(pubs)

    # Fallback: single pub
    return [pubs]


# Alias for API consistency
iter_pubs = materialize_pubs


def extract_circuit_from_pub(pub: Any) -> QuantumCircuit | None:
    """
    Extract a QuantumCircuit from a pub-like object.

    Supports common patterns:
    - QuantumCircuit itself
    - pub.circuit attribute (SamplerPub/EstimatorPub)
    - dict with "circuit" key
    - tuple/list where first element is a circuit (V2 PUB tuples)

    Parameters
    ----------
    pub : Any
        Pub-like object.

    Returns
    -------
    QuantumCircuit or None
        Extracted circuit if found.

    Examples
    --------
    >>> extract_circuit_from_pub(circuit)
    circuit
    >>> extract_circuit_from_pub((circuit, params))
    circuit
    >>> extract_circuit_from_pub({"circuit": circuit})
    circuit
    """
    if pub is None:
        return None

    if isinstance(pub, QuantumCircuit):
        return pub

    c = getattr(pub, "circuit", None)
    if isinstance(c, QuantumCircuit):
        return c

    if isinstance(pub, dict):
        c2 = pub.get("circuit", None)
        if isinstance(c2, QuantumCircuit):
            return c2

    if (
        isinstance(pub, (list, tuple))
        and len(pub) >= 1
        and isinstance(pub[0], QuantumCircuit)
    ):
        return pub[0]

    return None


def extract_circuits_from_pubs(pubs: Any) -> list[QuantumCircuit]:
    """
    Extract all circuits from a collection of pubs.

    Parameters
    ----------
    pubs : Any
        PUB collection in any supported form.

    Returns
    -------
    list of QuantumCircuit
        List of extracted circuits.

    Examples
    --------
    >>> extract_circuits_from_pubs([circuit1, circuit2])
    [circuit1, circuit2]
    >>> extract_circuits_from_pubs([(circuit, params)])
    [circuit]
    """
    circuits: list[QuantumCircuit] = []
    items = pubs if isinstance(pubs, list) else materialize_pubs(pubs)
    for pub in items:
        c = extract_circuit_from_pub(pub)
        if c is not None:
            circuits.append(c)
    return circuits


def _count_observables(obs: Any) -> int | None:
    """
    Count the number of observables in an observable-like object.

    Parameters
    ----------
    obs : Any
        Observable or collection of observables.

    Returns
    -------
    int or None
        Number of observables, or None if cannot determine.

    Notes
    -----
    Operator objects (SparsePauliOp, Pauli, PauliSumOp) have ``__len__``
    returning the number of internal Pauli terms, NOT the number of
    observables.  A single operator is always 1 observable.
    """
    if obs is None:
        return None

    # List/tuple of observables
    if isinstance(obs, (list, tuple)):
        return len(obs)

    # Operator-like objects: SparsePauliOp, Pauli, PauliSumOp, etc.
    # Their __len__ returns Pauli term count, not observable count.
    if hasattr(obs, "paulis") or hasattr(obs, "to_matrix"):
        return 1

    # Has __len__ (other observable collections, e.g. ObservablesArray)
    if hasattr(obs, "__len__"):
        try:
            return len(obs)
        except Exception:
            pass

    # Single observable
    return 1


def extract_pubs_structure(
    pubs: Any,
    *,
    primitive_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract metadata structure from pubs for logging.

    This function creates a summary of each PUB's structure suitable
    for the UEC program snapshot without including the actual circuit data.

    Parameters
    ----------
    pubs : Any
        PUB collection.
    primitive_type : str, optional
        Type of primitive ('sampler' or 'estimator'). If 'estimator',
        attempts to extract observable count from tuple PUBs.

    Returns
    -------
    list of dict
        List of pub metadata dictionaries with keys:
        - type: Python type name
        - format: PUB format identifier
        - has_circuit: Whether a circuit was found
        - num_observables: Number of observables (Estimator only)
        - has_parameters: Whether parameter values are present

    Examples
    --------
    >>> pubs_structure = extract_pubs_structure([(circuit, params)])
    >>> pubs_structure[0]["format"]
    'v2_pub_tuple'
    """
    pubs_struct: list[dict[str, Any]] = []

    items = pubs if isinstance(pubs, list) else materialize_pubs(pubs)
    for pub in items:
        info: dict[str, Any] = {
            "type": type(pub).__name__,
            "format": "unknown",
            "has_circuit": extract_circuit_from_pub(pub) is not None,
        }

        if isinstance(pub, QuantumCircuit):
            info["format"] = "circuit"
        elif is_v2_pub_tuple(pub):
            info["format"] = "v2_pub_tuple"
            info["tuple_len"] = len(pub)

            # For estimator tuple PUBs, try to extract observables count
            if primitive_type == "estimator" and len(pub) >= 2:
                num_obs = _count_observables(pub[1])
                if num_obs is not None:
                    info["num_observables"] = num_obs

        elif isinstance(pub, dict) and "circuit" in pub:
            info["format"] = "dict_pub"
            obs = pub.get("observables", pub.get("observable", None))
            if obs is not None:
                num_obs = _count_observables(obs)
                if num_obs is not None:
                    info["num_observables"] = num_obs

        elif hasattr(pub, "circuit"):
            info["format"] = "object_pub"

        # Observables (Estimator-style pub objects with .observables)
        if hasattr(pub, "observables") and "num_observables" not in info:
            try:
                num_obs = _count_observables(pub.observables)
                if num_obs is not None:
                    info["num_observables"] = num_obs
            except Exception:
                pass

        # Parameter values
        if hasattr(pub, "parameter_values"):
            try:
                pv = pub.parameter_values
                if pv is not None:
                    info["has_parameters"] = True
                    if hasattr(pv, "shape"):
                        info["parameter_shape"] = list(pv.shape)
                    elif hasattr(pv, "__len__"):
                        info["num_parameter_sets"] = len(pv)
            except Exception:
                pass

        # Check for parameters in tuple PUBs
        if is_v2_pub_tuple(pub) and "has_parameters" not in info:
            # For sampler: (circuit, params, shots)
            # For estimator: (circuit, obs, params, precision)
            param_idx = 2 if primitive_type == "estimator" else 1
            if len(pub) > param_idx:
                candidate = pub[param_idx]
                # Heuristic: if it's array-like, it's likely parameters
                if candidate is not None and hasattr(candidate, "__len__"):
                    if not isinstance(candidate, (str, dict)):
                        info["has_parameters"] = True

        pubs_struct.append(info)

    return pubs_struct


def extract_parameter_values_from_pubs(
    pubs: Any,
    *,
    primitive_type: str | None = None,
) -> list[Any]:
    """
    Extract parameter values from pubs for parametric hash computation.

    PUBs can carry parameter values separately from circuit data. For accurate
    fingerprinting, we need to include these values in the parametric hash.

    Parameters
    ----------
    pubs : Any
        PUB collection in any supported form.
    primitive_type : str, optional
        Type of primitive ('sampler' or 'estimator'). Determines the position
        of parameter values in tuple PUBs.

    Returns
    -------
    list
        List of parameter value arrays/objects, one per pub. Each element is:
        - None if pub has no parameter values
        - ndarray-like if pub has broadcasted parameters
        - dict if pub uses named parameters

    Notes
    -----
    Parameter value positions in tuple PUBs:
    - Sampler:   (circuit, params, shots)       -> index 1
    - Estimator: (circuit, obs, params, prec)   -> index 2
    """
    param_values: list[Any] = []

    items = pubs if isinstance(pubs, list) else materialize_pubs(pubs)
    for pub in items:
        pv: Any = None

        # Object-style pub with .parameter_values
        if hasattr(pub, "parameter_values"):
            try:
                pv = pub.parameter_values
            except Exception:
                pass

        # Dict-style pub
        elif isinstance(pub, dict):
            pv = pub.get("parameter_values")

        # Tuple-style V2 pub
        elif is_v2_pub_tuple(pub):
            param_idx = 2 if primitive_type == "estimator" else 1
            if len(pub) > param_idx:
                candidate = pub[param_idx]
                # Distinguish parameters from observables/shots
                if candidate is not None and not isinstance(candidate, (str, dict)):
                    if hasattr(candidate, "__len__") or hasattr(candidate, "__iter__"):
                        pv = candidate

        param_values.append(pv)

    return param_values
