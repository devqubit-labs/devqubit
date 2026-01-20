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

Functions
---------
- :func:`resolve_envelope` - Primary entry point (UEC-first with synthesis fallback)
- :func:`load_envelope` - Load first envelope from artifacts
- :func:`load_all_envelopes` - Load all envelopes (for multi-circuit runs)

Examples
--------
>>> from devqubit_engine.uec.api.resolve import resolve_envelope
>>> envelope = resolve_envelope(record, store)
>>> print(envelope.envelope_id)

>>> # For multi-circuit runs with multiple envelopes
>>> from devqubit_engine.uec.api.resolve import load_all_envelopes
>>> envelopes = load_all_envelopes(record, store)
>>> for env in envelopes:
...     print(env.envelope_id, env.program.num_circuits)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from devqubit_engine.uec.api.synthesize import synthesize_envelope
from devqubit_engine.uec.errors import EnvelopeValidationError, MissingEnvelopeError
from devqubit_engine.utils.common import is_manual_run_record


if TYPE_CHECKING:
    from devqubit_engine.storage.types import ObjectStoreProtocol
    from devqubit_engine.tracking.record import RunRecord
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


def load_envelope(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    include_invalid: bool = False,
    raise_on_error: bool = False,
) -> ExecutionEnvelope | None:
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
    raise_on_error : bool, default=False
        If True, raise EnvelopeValidationError when envelope artifact
        exists but cannot be parsed. If False, return None on parse error.

    Returns
    -------
    ExecutionEnvelope or None
        Loaded envelope if found, None otherwise.

    Raises
    ------
    EnvelopeValidationError
        If ``raise_on_error=True`` and envelope artifact exists but
        cannot be parsed.

    Notes
    -----
    Selection priority when multiple envelopes exist:

    1. Valid envelopes (kind="devqubit.envelope.json") are preferred
    2. Among valid envelopes, prefer the one with latest ``execution.completed_at``
    3. If no ``completed_at`` available, prefer the last artifact (stable ordering)
    4. Invalid envelopes only returned if ``include_invalid=True`` and no valid found
    """
    from devqubit_engine.storage.artifacts.io import load_artifact_json
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope

    valid_artifacts: list = []
    invalid_artifact = None

    for artifact in record.artifacts:
        if artifact.role != "envelope":
            continue

        if artifact.kind == "devqubit.envelope.json":
            valid_artifacts.append(artifact)

        if artifact.kind == "devqubit.envelope.invalid.json":
            invalid_artifact = artifact

    if not valid_artifacts:
        if include_invalid and invalid_artifact is not None:
            target_artifact = invalid_artifact
        else:
            logger.debug("No envelope artifact found for run %s", record.run_id)
            return None

        try:
            envelope_data = load_artifact_json(target_artifact, store)
            if not isinstance(envelope_data, dict):
                error_msg = "Envelope artifact is not a dict"
                logger.warning("%s for run %s", error_msg, record.run_id)
                if raise_on_error:
                    adapter = record.record.get("adapter", "unknown")
                    raise EnvelopeValidationError(str(adapter), [error_msg])
                return None

            envelope = ExecutionEnvelope.from_dict(envelope_data)
            logger.debug(
                "Loaded invalid envelope from artifact: run=%s, envelope_id=%s",
                record.run_id,
                envelope.envelope_id,
            )
            return envelope
        except EnvelopeValidationError:
            raise
        except Exception as e:
            logger.warning("Failed to parse envelope for run %s: %s", record.run_id, e)
            if raise_on_error:
                adapter = record.record.get("adapter", "unknown")
                raise EnvelopeValidationError(str(adapter), [str(e)]) from e
            return None

    # Single envelope - fast path
    if len(valid_artifacts) == 1:
        target_artifact = valid_artifacts[0]
    else:
        # Multiple envelopes: load all and select by completed_at
        logger.debug(
            "Found %d envelope artifacts for run %s. Selecting by completed_at.",
            len(valid_artifacts),
            record.run_id,
        )
        target_artifact = _select_best_envelope_artifact(
            artifacts=valid_artifacts,
            store=store,
        )
        if target_artifact is None:
            logger.debug("No valid envelope could be loaded for run %s", record.run_id)
            return None

    try:
        envelope_data = load_artifact_json(target_artifact, store)
        if not isinstance(envelope_data, dict):
            error_msg = "Envelope artifact is not a dict"
            logger.warning(
                "%s for run %s",
                error_msg,
                record.run_id,
            )
            if raise_on_error:
                adapter = record.record.get("adapter", "unknown")
                raise EnvelopeValidationError(str(adapter), [error_msg])
            return None

        envelope = ExecutionEnvelope.from_dict(envelope_data)
        logger.debug(
            "Loaded envelope from artifact: run=%s, envelope_id=%s",
            record.run_id,
            envelope.envelope_id,
        )
        return envelope

    except EnvelopeValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.warning("Failed to parse envelope for run %s: %s", record.run_id, e)
        if raise_on_error:
            adapter = record.record.get("adapter", "unknown")
            raise EnvelopeValidationError(str(adapter), [str(e)]) from e
        return None


def _select_best_envelope_artifact(
    artifacts: list,
    store: ObjectStoreProtocol,
) -> object | None:
    """
    Select the best envelope artifact from multiple candidates.

    Selection criteria (in order):
    1. Envelope with latest ``execution.completed_at``
    2. If no ``completed_at``, use last artifact in list (stable ordering)

    Parameters
    ----------
    artifacts : list
        List of envelope artifacts.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.

    Returns
    -------
    object or None
        Best artifact, or None if all failed to parse.
    """
    from devqubit_engine.storage.artifacts.io import load_artifact_json

    candidates: list[tuple[object, str | None]] = []

    for artifact in artifacts:
        try:
            envelope_data = load_artifact_json(artifact, store)
            if not isinstance(envelope_data, dict):
                continue

            # Extract completed_at from execution snapshot
            execution = envelope_data.get("execution") or {}
            completed_at = (
                execution.get("completed_at") if isinstance(execution, dict) else None
            )

            candidates.append((artifact, completed_at))
        except Exception as e:
            logger.debug("Failed to parse envelope artifact: %s", e)
            continue

    if not candidates:
        return None

    # Sort by completed_at (None values last), then by position (last wins)
    def sort_key(item: tuple[object, str | None]) -> tuple[int, str, int]:
        artifact, completed_at = item
        idx = artifacts.index(artifact)
        if completed_at is None:
            return (1, "", idx)
        return (0, completed_at, idx)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0][0]


def load_all_envelopes(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    include_invalid: bool = False,
) -> list[ExecutionEnvelope]:
    """
    Load all ExecutionEnvelopes from stored artifacts.

    For runs with multiple circuit batches, each batch may have its own
    envelope. This function returns all valid envelopes.

    Parameters
    ----------
    record : RunRecord
        Run record to load envelopes from.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.
    include_invalid : bool, default=False
        If True, also include invalid envelopes.

    Returns
    -------
    list of ExecutionEnvelope
        All loaded envelopes (may be empty).
    """
    from devqubit_engine.storage.artifacts.io import load_artifact_json
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope

    envelopes: list[ExecutionEnvelope] = []
    valid_kinds = {"devqubit.envelope.json"}
    if include_invalid:
        valid_kinds.add("devqubit.envelope.invalid.json")

    for artifact in record.artifacts:
        if artifact.role != "envelope" or artifact.kind not in valid_kinds:
            continue

        try:
            envelope_data = load_artifact_json(artifact, store)
            if isinstance(envelope_data, dict):
                envelope = ExecutionEnvelope.from_dict(envelope_data)
                envelopes.append(envelope)
        except Exception as e:
            logger.warning(
                "Failed to parse envelope artifact %s for run %s: %s",
                artifact.digest[:16],
                record.run_id,
                e,
            )

    logger.debug(
        "Loaded %d envelope(s) for run %s",
        len(envelopes),
        record.run_id,
    )
    return envelopes


def resolve_envelope(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    include_invalid: bool = False,
) -> ExecutionEnvelope:
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
    EnvelopeValidationError
        If adapter run has invalid/unparseable envelope (adapter bug).

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
    is_manual = is_manual_run_record(record.record)
    adapter = record.record.get("adapter", "manual")

    # For adapter runs, raise on parse errors (integration bug)
    # For manual runs, silently return None on parse errors (fallback to synthesis)
    envelope = load_envelope(
        record,
        store,
        include_invalid=include_invalid,
        raise_on_error=not is_manual,
    )

    if envelope is not None:
        return envelope

    if not is_manual:
        raise MissingEnvelopeError(record.run_id, str(adapter))

    return synthesize_envelope(record, store)
