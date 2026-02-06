# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""UEC exceptions."""

from __future__ import annotations

from devqubit_engine.errors import DevQubitError


class MissingEnvelopeError(DevQubitError):
    """
    Raised when an adapter run is missing its required envelope.

    Adapter runs MUST have an envelope artifact.  If missing, this
    indicates an adapter integration error that should be fixed in
    the adapter.

    Parameters
    ----------
    run_id : str
        Run identifier.
    adapter : str
        Adapter name that should have created the envelope.
    """

    def __init__(self, run_id: str, adapter: str) -> None:
        self.run_id = run_id
        self.adapter = adapter
        super().__init__(
            f"Adapter run '{run_id}' (adapter={adapter}) is missing envelope. "
            f"This is an adapter integration error - adapters must create envelope."
        )


class EnvelopeValidationError(DevQubitError):
    """
    Raised when an adapter run produces an invalid envelope.

    Adapter runs MUST produce valid envelopes.  Invalid envelopes
    indicate an adapter integration error that must be fixed in the
    adapter code.

    Parameters
    ----------
    adapter : str
        Adapter name that produced the invalid envelope.
    errors : list of str
        Validation error messages.
    """

    def __init__(self, adapter: str, errors: list[str]) -> None:
        self.adapter = adapter
        self.errors = errors
        error_summary = "; ".join(errors[:3])
        if len(errors) > 3:
            error_summary += f" ... and {len(errors) - 3} more"
        super().__init__(
            f"Adapter '{adapter}' produced invalid envelope: {error_summary}. "
            f"This is an adapter bug - adapters must produce valid envelopes."
        )
