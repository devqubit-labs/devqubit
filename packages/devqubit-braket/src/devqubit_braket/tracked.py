# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Tracked task wrappers for Braket adapter.

Provides TrackedTask and TrackedTaskBatch classes that wrap Braket
tasks to intercept result retrieval and finalize execution envelopes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from devqubit_braket.envelope import finalize_envelope
from devqubit_braket.results import extract_counts_payload
from devqubit_braket.utils import extract_task_id


if TYPE_CHECKING:
    from devqubit_engine.tracking.run import Run
    from devqubit_engine.uec.models.envelope import ExecutionEnvelope


logger = logging.getLogger(__name__)


def combine_batch_results(results_list: list[Any]) -> dict[str, Any]:
    """
    Combine batch results into a single structure for logging.

    Parameters
    ----------
    results_list : list
        List of individual result objects.

    Returns
    -------
    dict
        Combined result structure.
    """
    experiments: list[dict[str, Any]] = []

    for i, result in enumerate(results_list):
        if result is None:
            experiments.append({"index": i, "status": "failed", "counts": {}})
            continue

        counts_payload = extract_counts_payload(result)
        if counts_payload and counts_payload.get("experiments"):
            for exp in counts_payload["experiments"]:
                exp_copy = dict(exp)
                exp_copy["batch_index"] = i
                experiments.append(exp_copy)
        else:
            experiments.append({"index": i, "batch_index": i, "counts": {}})

    return {"experiments": experiments, "batch_size": len(results_list)}


@dataclass
class TrackedTask:
    """
    Wrapper for Braket task that tracks result retrieval.

    Intercepts `result()` calls to finalize the execution envelope
    with result data. Handles exceptions gracefully with failure logging.

    Parameters
    ----------
    task : Any
        Original Braket task instance.
    tracker : Run
        Tracker instance for logging.
    device_name : str
        Name of the device that created this task.
    envelope : ExecutionEnvelope or None
        Envelope to finalize with results.
    shots : int or None
        Number of shots for this execution.
    should_log_results : bool
        Whether to log results for this task.
    """

    task: Any
    tracker: Run
    device_name: str
    envelope: ExecutionEnvelope | None = None
    shots: int | None = None
    should_log_results: bool = True
    _result_logged: bool = field(default=False, init=False, repr=False)

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieve task result and finalize envelope.

        Handles exceptions gracefully, logging failure information
        before re-raising the ORIGINAL exception (P0 fix: preserve exception type).

        Parameters
        ----------
        *args : Any
            Positional arguments passed to underlying result().
        **kwargs : Any
            Keyword arguments passed to underlying result().

        Returns
        -------
        Any
            Braket result object.

        Raises
        ------
        Exception
            Re-raises original exception from underlying result() after logging.
        """
        result = None

        try:
            result = self.task.result(*args, **kwargs)
        except Exception as e:
            logger.warning("Task result() failed on %s: %s", self.device_name, e)

            # Log failure envelope before re-raising original exception
            if self.should_log_results and self.envelope and not self._result_logged:
                self._result_logged = True
                try:
                    finalize_envelope(
                        tracker=self.tracker,
                        envelope=self.envelope,
                        result=None,
                        device_name=self.device_name,
                        shots=self.shots,
                        error=e,
                    )
                except Exception as log_err:
                    logger.warning(
                        "Failed to log error envelope for %s: %s",
                        self.device_name,
                        log_err,
                    )

            raise

        # Log successful result
        if self.should_log_results and self.envelope and not self._result_logged:
            self._result_logged = True
            try:
                finalize_envelope(
                    tracker=self.tracker,
                    envelope=self.envelope,
                    result=result,
                    device_name=self.device_name,
                    shots=self.shots,
                )
                logger.debug("Finalized envelope for task on %s", self.device_name)
            except Exception as log_err:
                logger.warning(
                    "Failed to finalize envelope for %s: %s",
                    self.device_name,
                    log_err,
                )
                # Record error in tracker for visibility
                self.tracker.record.setdefault("warnings", []).append(
                    {
                        "type": "result_logging_failed",
                        "message": str(log_err),
                        "device_name": self.device_name,
                    }
                )

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped task."""
        return getattr(self.task, name)

    def __repr__(self) -> str:
        """Return string representation."""
        task_id = extract_task_id(self.task) or "unknown"
        return f"TrackedTask(device={self.device_name!r}, task_id={task_id!r})"


@dataclass
class TrackedTaskBatch:
    """
    Wrapper for Braket task batch that tracks result retrieval.

    Wraps AwsQuantumTaskBatch to intercept `results()` calls and log
    all results with proper handling of partial failures.

    Parameters
    ----------
    batch : Any
        Original Braket task batch instance.
    tracker : Run
        Tracker instance for logging.
    device_name : str
        Name of the device that created this batch.
    envelope : ExecutionEnvelope or None
        Envelope to finalize with results.
    shots : int or None
        Number of shots for this execution.
    should_log_results : bool
        Whether to log results for this batch.
    """

    batch: Any
    tracker: Run
    device_name: str
    envelope: ExecutionEnvelope | None = None
    shots: int | None = None
    should_log_results: bool = True
    _results_logged: bool = field(default=False, init=False, repr=False)

    def results(self, *args: Any, **kwargs: Any) -> list[Any]:
        """
        Retrieve batch results and finalize envelope.

        Handles partial failures (None results for failed tasks).

        Parameters
        ----------
        *args : Any
            Positional arguments passed to underlying results().
        **kwargs : Any
            Keyword arguments passed to underlying results().

        Returns
        -------
        list
            List of Braket result objects (may contain None for failed tasks).

        Raises
        ------
        Exception
            Re-raises original exception from underlying results() after logging.
        """
        results_list: list[Any] = []
        partial_error: Exception | None = None

        try:
            results_list = self.batch.results(*args, **kwargs)
        except Exception as e:
            logger.warning("Batch results() failed on %s: %s", self.device_name, e)

            if self.should_log_results and self.envelope and not self._results_logged:
                self._results_logged = True
                try:
                    finalize_envelope(
                        self.tracker,
                        self.envelope,
                        None,
                        self.device_name,
                        self.shots,
                        error=e,
                    )
                except Exception as log_err:
                    logger.warning(
                        "Failed to log error envelope for batch %s: %s",
                        self.device_name,
                        log_err,
                    )

            raise

        # Check for partial failures (None in results)
        failed_count = sum(1 for r in results_list if r is None)
        if failed_count > 0:
            logger.warning(
                "Batch on %s: %d/%d tasks failed",
                self.device_name,
                failed_count,
                len(results_list),
            )
            partial_error = RuntimeError(
                f"Partial failure: {failed_count}/{len(results_list)} tasks failed"
            )

        # Aggregate successful results for logging
        if self.should_log_results and self.envelope and not self._results_logged:
            self._results_logged = True
            try:
                combined_result = combine_batch_results(results_list)

                finalize_envelope(
                    self.tracker,
                    self.envelope,
                    combined_result,
                    self.device_name,
                    self.shots,
                    error=partial_error,
                )
                logger.debug("Finalized envelope for batch on %s", self.device_name)
            except Exception as log_err:
                logger.warning(
                    "Failed to finalize envelope for batch %s: %s",
                    self.device_name,
                    log_err,
                )
                # Record error in tracker for visibility
                self.tracker.record.setdefault("warnings", []).append(
                    {
                        "type": "batch_result_logging_failed",
                        "message": str(log_err),
                        "device_name": self.device_name,
                    }
                )

        return results_list

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped batch."""
        return getattr(self.batch, name)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TrackedTaskBatch(device={self.device_name!r})"
