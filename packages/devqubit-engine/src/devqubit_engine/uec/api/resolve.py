# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC envelope resolution - single entry point for obtaining envelopes.

This module provides the canonical interface for obtaining ExecutionEnvelope
from any run record. It implements the "UEC-first" strategy:

1. **Adapter runs**: Envelope MUST exist (created by adapter). Missing envelope
   is an integration error and raises MissingEnvelopeError.
2. **Manual runs**: Envelope is synthesized from RunRecord if not present.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from devqubit_engine.storage.types import ArtifactRef
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

    Parameters
    ----------
    record : RunRecord
        Run record to load envelope from.
    store : ObjectStoreProtocol
        Object store for artifact retrieval.
    include_invalid : bool, default=False
        If True, also return invalid envelopes.
    raise_on_error : bool, default=False
        If True, raise EnvelopeValidationError on parse errors.

    Returns
    -------
    ExecutionEnvelope or None
        Loaded envelope if found, None otherwise.

    Raises
    ------
    EnvelopeValidationError
        If ``raise_on_error=True`` and envelope cannot be parsed.

    Notes
    -----
    Selection priority when multiple envelopes exist:

    1. Valid envelopes (kind="devqubit.envelope.json") preferred
    2. Among valid, prefer latest ``execution.completed_at``
    3. If no ``completed_at``, prefer last artifact
    4. Invalid envelopes only if ``include_invalid=True`` and no valid found
    """

    valid_artifacts: list = []
    invalid_artifact = None

    for artifact in record.artifacts:
        if artifact.role != "envelope":
            continue

        if artifact.kind == "devqubit.envelope.json":
            valid_artifacts.append(artifact)
        elif artifact.kind == "devqubit.envelope.invalid.json":
            invalid_artifact = artifact

    # No valid envelopes found
    if not valid_artifacts:
        if include_invalid and invalid_artifact is not None:
            return _try_load_envelope(invalid_artifact, record, store, raise_on_error)
        logger.debug("No envelope artifact found for run %s", record.run_id)
        return None

    # Single envelope - fast path
    if len(valid_artifacts) == 1:
        return _try_load_envelope(valid_artifacts[0], record, store, raise_on_error)

    # Multiple envelopes: select best by completed_at
    logger.debug(
        "Found %d envelope artifacts for run %s. Selecting by completed_at.",
        len(valid_artifacts),
        record.run_id,
    )
    target_artifact = _select_best_envelope_artifact(valid_artifacts, store)
    if target_artifact is None:
        logger.debug("No valid envelope could be loaded for run %s", record.run_id)
        return None

    return _try_load_envelope(target_artifact, record, store, raise_on_error)


def _try_load_envelope(
    artifact: ArtifactRef,
    record: RunRecord,
    store: ObjectStoreProtocol,
    raise_on_error: bool,
) -> ExecutionEnvelope | None:
    """Try to load and parse envelope from artifact."""
    from devqubit_engine.storage.artifacts.io import load_artifact_json
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope

    try:
        envelope_data = load_artifact_json(artifact, store)
        if not isinstance(envelope_data, dict):
            error_msg = "Envelope artifact is not a dict"
            logger.warning("%s for run %s", error_msg, record.run_id)
            if raise_on_error:
                adapter = record.record.get("adapter", "unknown")
                raise EnvelopeValidationError(str(adapter), [error_msg])
            return None

        envelope = ExecutionEnvelope.from_dict(envelope_data)
        logger.debug(
            "Loaded envelope: run=%s, envelope_id=%s",
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


def _select_best_envelope_artifact(
    artifacts: list[ArtifactRef],
    store: ObjectStoreProtocol,
) -> ArtifactRef | None:
    """
    Select the best envelope artifact from multiple candidates.

    Selection criteria (in priority order):

    1. Prefer envelopes with ``execution.completed_at`` over those without.
    2. Among those with timestamps, prefer the latest ``completed_at``.
    3. If no ``completed_at``, prefer the last artifact in the list.
    """
    from devqubit_engine.storage.artifacts.io import load_artifact_json

    candidates: list[tuple[ArtifactRef, str | None]] = []

    for artifact in artifacts:
        try:
            envelope_data = load_artifact_json(artifact, store)
            if not isinstance(envelope_data, dict):
                continue

            execution = envelope_data.get("execution") or {}
            completed_at = (
                execution.get("completed_at") if isinstance(execution, dict) else None
            )
            candidates.append((artifact, completed_at))
        except Exception as e:
            logger.debug("Failed to parse envelope artifact: %s", e)

    if not candidates:
        return None

    # Sort: envelopes WITH completed_at first (descending by date),
    # then envelopes WITHOUT completed_at (descending by list position).
    def sort_key(item: tuple[ArtifactRef, str | None]) -> tuple[int, str, int]:
        artifact, completed_at = item
        idx = artifacts.index(artifact)
        if completed_at is None:
            # Lower priority (0) — sorted after entries with timestamps
            return (0, "", idx)
        # Higher priority (1) — latest completed_at wins
        return (1, completed_at, idx)

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
    envelope.

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

    logger.debug("Loaded %d envelope(s) for run %s", len(envelopes), record.run_id)
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
    All compare/diff/verify operations should use this function.

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
        If adapter run is missing envelope.
    EnvelopeValidationError
        If adapter run has invalid/unparseable envelope.

    Notes
    -----
    Strategy:

    1. Try to load existing envelope artifact
    2. If not found:
       - **Adapter run**: Raise MissingEnvelopeError
       - **Manual run**: Synthesize from RunRecord
    """
    is_manual = is_manual_run_record(record.record)
    adapter = record.record.get("adapter", "manual")

    # For adapter runs, raise on parse errors
    envelope = load_envelope(
        record,
        store,
        include_invalid=include_invalid,
        raise_on_error=not is_manual,
    )

    if envelope is not None:
        return envelope

    # Adapter run without envelope is an error
    if not is_manual:
        raise MissingEnvelopeError(record.run_id, str(adapter))

    # Manual run - synthesize
    return synthesize_envelope(record, store)
