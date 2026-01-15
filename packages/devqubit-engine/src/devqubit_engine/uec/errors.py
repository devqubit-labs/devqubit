# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
UEC constants and exceptions.

This module defines constants and exceptions used across the UEC subsystem.
"""

from __future__ import annotations


# Keys in execute section that are volatile (change between runs)
# and should be stripped when computing fingerprints or comparing
VOLATILE_EXECUTE_KEYS: frozenset[str] = frozenset(
    {
        "submitted_at",
        "job_id",
        "job_ids",
        "completed_at",
        "session_id",
        "task_id",
        "task_ids",
    }
)

# Schema version for envelope
ENVELOPE_SCHEMA_VERSION = "devqubit.envelope/1.0"

# Schema version for result snapshot
RESULT_SCHEMA_VERSION = "devqubit.result_snapshot/1.0"


class MissingEnvelopeError(Exception):
    """
    Raised when adapter run is missing required envelope.

    Adapter runs MUST have an envelope artifact. If missing, this indicates
    an adapter integration error that should be fixed in the adapter.

    Parameters
    ----------
    run_id : str
        Run identifier.
    adapter : str
        Adapter name that should have created the envelope.
    """

    def __init__(self, run_id: str, adapter: str):
        self.run_id = run_id
        self.adapter = adapter
        super().__init__(
            f"Adapter run '{run_id}' (adapter={adapter}) is missing envelope. "
            f"This is an adapter integration error - adapters must create envelope."
        )


class EnvelopeValidationError(Exception):
    """
    Raised when adapter run produces invalid envelope.

    Adapter runs MUST produce valid envelopes. Invalid envelopes indicate
    an adapter integration error that must be fixed in the adapter code.

    Parameters
    ----------
    adapter : str
        Adapter name that produced invalid envelope.
    errors : list of str
        Validation error messages.
    """

    def __init__(self, adapter: str, errors: list[str]):
        self.adapter = adapter
        self.errors = errors
        error_summary = "; ".join(errors[:3])
        if len(errors) > 3:
            error_summary += f" ... and {len(errors) - 3} more"
        super().__init__(
            f"Adapter '{adapter}' produced invalid envelope: {error_summary}. "
            f"This is an adapter bug - adapters must produce valid envelopes."
        )
