# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC envelope resolution - single entry point for obtaining envelopes.

This module provides the canonical interface for obtaining ExecutionEnvelope
from any run record. It implements the "UEC-first" strategy with strict
contract enforcement:

1. **Adapter runs**: Envelope MUST exist (created by adapter). Missing envelope
   is an integration error and raises MissingEnvelopeError.
2. **Manual runs**: Envelope is synthesized from RunRecord if not present.
   This is best-effort with limited semantics (no program hashes).

Examples
--------
>>> from devqubit_engine.uec import resolve_envelope
>>> envelope = resolve_envelope(record, store)
>>> print(envelope.envelope_id)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from devqubit_engine.uec.api.synthesize import synthesize_envelope
from devqubit_engine.uec.errors import MissingEnvelopeError
from devqubit_engine.utils.common import is_manual_run_record


if TYPE_CHECKING:
    from devqubit_engine.storage.types import ObjectStoreProtocol
    from devqubit_engine.tracking.record import RunRecord
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


def load_envelope(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
    *,
    include_invalid: bool = False,
) -> "ExecutionEnvelope | None":
    """
    Load ExecutionEnvelope from stored artifact.

    This function attempts to load an existing envelope artifact from
    the run record. It prefers valid envelopes but can optionally
    return invalid ones for debugging purposes.

    Parameters
    ----------
    record : RunRecord
        Run record to load envelope from.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.
    include_invalid : bool, default=False
        If True, also return invalid envelopes (kind contains "invalid").

    Returns
    -------
    ExecutionEnvelope or None
        Loaded envelope if found, None otherwise.

    Notes
    -----
    Selection priority:
    1. role="envelope", kind="devqubit.envelope.json" (valid)
    2. role="envelope", kind="devqubit.envelope.invalid.json" (if include_invalid)
    """
    from devqubit_engine.storage.artifacts.io import load_artifact_json
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope

    valid_artifact = None
    invalid_artifact = None

    for artifact in record.artifacts:
        if artifact.role != "envelope":
            continue

        if artifact.kind == "devqubit.envelope.json":
            valid_artifact = artifact
            break

        if artifact.kind == "devqubit.envelope.invalid.json":
            invalid_artifact = artifact

    if valid_artifact is not None:
        target_artifact = valid_artifact
    elif include_invalid and invalid_artifact is not None:
        target_artifact = invalid_artifact
    else:
        logger.debug("No envelope artifact found for run %s", record.run_id)
        return None

    try:
        envelope_data = load_artifact_json(target_artifact, store)
        if not isinstance(envelope_data, dict):
            logger.warning(
                "Envelope artifact is not a dict for run %s",
                record.run_id,
            )
            return None

        envelope = ExecutionEnvelope.from_dict(envelope_data)
        logger.debug(
            "Loaded envelope from artifact: run=%s, envelope_id=%s",
            record.run_id,
            envelope.envelope_id,
        )
        return envelope

    except Exception as e:
        logger.warning("Failed to parse envelope for run %s: %s", record.run_id, e)
        return None


def resolve_envelope(
    record: "RunRecord",
    store: "ObjectStoreProtocol",
    *,
    include_invalid: bool = False,
) -> "ExecutionEnvelope":
    """
    Resolve ExecutionEnvelope for a run (UEC-first with strict contract).

    This is the **primary entry point** for obtaining envelope data.
    All compare/diff/verify operations should use this function rather
    than directly accessing RunRecord fields or artifacts.

    Parameters
    ----------
    record : RunRecord
        Run record to resolve envelope for.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.
    include_invalid : bool, default=False
        If True, include invalid envelopes in search.

    Returns
    -------
    ExecutionEnvelope
        Resolved envelope.

    Raises
    ------
    MissingEnvelopeError
        If adapter run is missing envelope (adapter integration error).

    Notes
    -----
    Strategy:
    1. Try to load existing envelope artifact (UEC-first)
    2. If not found:
       - **Adapter run**: Raise MissingEnvelopeError (adapters MUST create envelope)
       - **Manual run**: Synthesize from RunRecord and artifacts

    Examples
    --------
    >>> envelope = resolve_envelope(record, store)
    >>> if envelope.metadata.get("synthesized_from_run"):
    ...     print("This is a synthesized envelope from manual run")
    """
    envelope = load_envelope(record, store, include_invalid=include_invalid)

    if envelope is not None:
        return envelope

    is_manual = is_manual_run_record(record.record)
    adapter = record.record.get("adapter", "manual")

    if not is_manual:
        raise MissingEnvelopeError(record.run_id, str(adapter))

    return synthesize_envelope(record, store)
