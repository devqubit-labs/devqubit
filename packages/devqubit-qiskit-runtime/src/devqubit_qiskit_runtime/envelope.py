# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Envelope and snapshot utilities for Qiskit Runtime adapter.

This module provides functions for creating UEC snapshots and
managing ExecutionEnvelope lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from devqubit_engine.tracking.run import Run
from devqubit_engine.uec.models.envelope import ExecutionEnvelope
from devqubit_engine.uec.models.result import ResultSnapshot
from devqubit_engine.utils.common import utc_now_iso
from devqubit_qiskit_runtime.utils import get_backend_name, get_backend_obj


logger = logging.getLogger(__name__)


def detect_physical_provider(primitive: Any) -> str:
    """
    Detect physical provider from Runtime primitive (not SDK).

    UEC requires provider to be the physical backend provider,
    not the SDK name. SDK goes in producer.frontends[].

    Parameters
    ----------
    primitive : Any
        Runtime primitive instance.

    Returns
    -------
    str
        Physical provider: "ibm_quantum", "fake", "aer", or "local".
    """
    backend = get_backend_obj(primitive)
    if backend is None:
        # No backend resolved - check primitive module
        module_name = getattr(primitive, "__module__", "").lower()
        if "ibm" in module_name:
            return "ibm_quantum"
        return "local"

    module_name = type(backend).__module__.lower()
    backend_name = get_backend_name(primitive).lower()

    # IBM quantum hardware
    if "ibm" in module_name or "ibm_" in backend_name:
        # Check for fake backends
        if "fake" in module_name or "fake" in backend_name:
            return "fake"
        return "ibm_quantum"

    # Aer simulator (local)
    if "aer" in module_name:
        return "aer"

    return "local"


def create_failure_result_snapshot(
    exception: BaseException,
    backend_name: str,
    primitive_type: str,
) -> ResultSnapshot:
    """
    Create a ResultSnapshot for a failed execution.

    Used when job.result() raises an exception. Ensures envelope
    is always created even on failures (UEC requirement).

    Parameters
    ----------
    exception : BaseException
        The exception that caused the failure.
    backend_name : str
        Backend name for metadata.
    primitive_type : str
        Type of primitive ('sampler' or 'estimator').

    Returns
    -------
    ResultSnapshot
        Failed result snapshot with error details.
    """
    return ResultSnapshot.create_failed(
        exception=exception,
        metadata={
            "backend_name": backend_name,
            "primitive_type": primitive_type,
        },
    )


def finalize_envelope_with_result(
    tracker: Run,
    envelope: ExecutionEnvelope,
    result_snapshot: ResultSnapshot,
) -> None:
    """
    Finalize envelope with result and log as artifact.

    Parameters
    ----------
    tracker : Run
        Tracker instance.
    envelope : ExecutionEnvelope
        Envelope to finalize.
    result_snapshot : ResultSnapshot
        Result to add to envelope.

    Raises
    ------
    ValueError
        If envelope is None.
    """
    if envelope is None:
        raise ValueError("Cannot finalize None envelope")

    if result_snapshot is None:
        logger.warning("Finalizing envelope with None result_snapshot")

    # Add result to envelope
    envelope.result = result_snapshot

    # Set completion time
    if envelope.execution is not None:
        envelope.execution.completed_at = utc_now_iso()

    # Validate and log envelope using tracker's canonical method
    tracker.log_envelope(envelope=envelope)
