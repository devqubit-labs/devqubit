# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Run record type for devqubit.

This module defines the data structure used for tracking and
persisting quantum experiment runs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from devqubit_engine.storage.types import ArtifactRef


@dataclass
class RunRecord:
    """
    Wrapper for run record dictionary with convenience accessors.

    Provides typed property access to common run record fields while
    maintaining the underlying dictionary for full flexibility and
    schema evolution.

    Parameters
    ----------
    record : dict
        Raw run record dictionary conforming to the devqubit.run schema.
    artifacts : list of ArtifactRef, optional
        Artifact references managed separately for efficiency.
        Default is an empty list.

    Attributes
    ----------
    run_id : str
        Unique run identifier (typically ULID).
    project : str
        Project name.
    adapter : str
        SDK adapter name (e.g., "qiskit", "pennylane").
    status : str
        Run status: "RUNNING", "FINISHED", "FAILED", or "KILLED".
    created_at : str
        ISO 8601 creation timestamp.
    fingerprints : dict
        Dictionary of computed fingerprints.
    params : dict
        Experiment parameters.
    metrics : dict
        Logged numeric metrics.
    tags : dict
        String tags.
    """

    record: dict[str, Any]
    artifacts: list[ArtifactRef] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def run_id(self) -> str:
        """
        Get the unique run identifier.

        Returns
        -------
        str
            Run ID, typically a ULID. Empty string if not set.
        """
        with self._lock:
            return self.record.get("run_id", "")

    @property
    def project(self) -> str:
        """
        Get the project name.

        Returns
        -------
        str
            Project name. Empty string if not set.
        """
        with self._lock:
            proj = self.record.get("project", {})
            if isinstance(proj, dict):
                return proj.get("name", "")
            return str(proj) if proj else ""

    @property
    def adapter(self) -> str:
        """
        Get the SDK adapter name.

        Returns
        -------
        str
            Adapter name (e.g., "qiskit", "pennylane", "cirq").
        """
        with self._lock:
            return self.record.get("adapter", "")

    @property
    def status(self) -> str:
        """
        Get the current run status.

        Returns
        -------
        str
            One of: "RUNNING", "FINISHED", "FAILED", "KILLED".
            Defaults to "RUNNING" if not explicitly set.
        """
        with self._lock:
            info = self.record.get("info", {})
            if isinstance(info, dict):
                return info.get("status", "RUNNING")
            return "RUNNING"

    @property
    def created_at(self) -> str:
        """
        Get the run creation timestamp.

        Returns
        -------
        str
            ISO 8601 formatted timestamp. Empty string if not set.
        """
        with self._lock:
            return self.record.get("created_at", "")

    @property
    def fingerprints(self) -> dict[str, str]:
        """
        Get the fingerprints dictionary.

        Returns
        -------
        dict
            Dictionary of fingerprint type to SHA-256 digest.
            Common keys: "program", "device", "intent", "run".
        """
        with self._lock:
            fps = self.record.get("fingerprints", {})
            return dict(fps) if isinstance(fps, dict) else {}

    @property
    def run_fingerprint(self) -> str | None:
        """
        Get the combined run fingerprint.

        The run fingerprint combines program, device, and intent
        fingerprints to uniquely identify the experimental setup.

        Returns
        -------
        str or None
            Combined fingerprint digest, or None if not computed.
        """
        return self.fingerprints.get("run")

    @property
    def program_fingerprint(self) -> str | None:
        """
        Get the program fingerprint.

        Computed from ExecutionEnvelope program hashes (structural_hash
        and parametric_hash) when available. Falls back to artifact
        digests for runs without envelope.

        Returns
        -------
        str or None
            Program fingerprint digest, or None if not computed.
        """
        return self.fingerprints.get("program")

    @property
    def backend_name(self) -> str | None:
        """
        Get the backend/device name.

        Returns
        -------
        str or None
            Backend name (e.g., "ibm_brisbane", "lightning.qubit"),
            or None if not set.
        """
        with self._lock:
            backend = self.record.get("backend", {})
            if isinstance(backend, dict):
                return backend.get("name")
            return None

    @property
    def group_id(self) -> str | None:
        """
        Get the group/experiment identifier.

        Used for grouping related runs together, such as:
        - Parameter sweeps
        - Nightly calibration checks
        - Benchmark suites
        - A/B testing experiments

        Returns
        -------
        str or None
            Group identifier, or None if not part of a group.
        """
        with self._lock:
            return self.record.get("group_id")

    @property
    def group_name(self) -> str | None:
        """
        Get the human-readable group name.

        Returns
        -------
        str or None
            Group name, or None if not set.
        """
        with self._lock:
            return self.record.get("group_name")

    @property
    def parent_run_id(self) -> str | None:
        """
        Get the parent run identifier for lineage tracking.

        Used for tracking run relationships such as:
        - Rerun from baseline
        - Rerun with new device
        - Experiment iterations

        Returns
        -------
        str or None
            Parent run ID, or None if this is a root run.
        """
        with self._lock:
            return self.record.get("parent_run_id")

    @property
    def run_name(self) -> str | None:
        """
        Get the human-readable run name.

        Returns
        -------
        str or None
            Run name, or None if not set.
        """
        with self._lock:
            info = self.record.get("info", {})
            if isinstance(info, dict):
                return info.get("run_name")
            return None

    @property
    def ended_at(self) -> str | None:
        """
        Get the run end timestamp.

        Returns
        -------
        str or None
            ISO 8601 formatted end timestamp, or None if still running.
        """
        with self._lock:
            info = self.record.get("info", {})
            if isinstance(info, dict):
                return info.get("ended_at")
            return None

    @property
    def params(self) -> dict[str, Any]:
        """
        Get the experiment parameters dictionary.

        Parameters are user-logged via ``log_param()`` and represent
        experimental configuration values.

        Returns
        -------
        dict
            Parameter name to value mapping.
        """
        with self._lock:
            data = self.record.get("data", {})
            if isinstance(data, dict):
                params = data.get("params", {})
                return dict(params) if isinstance(params, dict) else {}
            return {}

    @property
    def metrics(self) -> dict[str, float]:
        """
        Get the logged metrics dictionary.

        Metrics are numeric values logged via ``log_metric()``.

        Returns
        -------
        dict
            Metric name to numeric value mapping.
        """
        with self._lock:
            data = self.record.get("data", {})
            if isinstance(data, dict):
                metrics = data.get("metrics", {})
                return dict(metrics) if isinstance(metrics, dict) else {}
            return {}

    @property
    def tags(self) -> dict[str, str]:
        """
        Get the string tags dictionary.

        Tags are string key-value pairs set via ``set_tag()``.

        Returns
        -------
        dict
            Tag name to string value mapping.
        """
        with self._lock:
            data = self.record.get("data", {})
            if isinstance(data, dict):
                tags = data.get("tags", {})
                return dict(tags) if isinstance(tags, dict) else {}
            return {}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a complete dictionary with artifacts.

        Returns
        -------
        dict
            Complete run record with artifacts serialized.
        """
        with self._lock:
            result = dict(self.record)
            result["artifacts"] = [a.to_dict() for a in self.artifacts]
        return result

    def __repr__(self) -> str:
        """Return a concise representation of the run record."""
        return (
            f"RunRecord(run_id={self.run_id!r}, project={self.project!r}, "
            f"adapter={self.adapter!r}, status={self.status!r}, "
            f"artifacts={len(self.artifacts)})"
        )
