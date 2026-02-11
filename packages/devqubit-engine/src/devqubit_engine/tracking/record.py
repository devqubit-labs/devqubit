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


def resolve_run_id(
    run_id_or_name: str,
    project: str | None,
    registry: Any,
) -> str:
    """
    Resolve a run reference (ID or name) to a run ID.

    Parameters
    ----------
    run_id_or_name : str
        Run identifier (ULID) or run name.
    project : str or None
        Project name. Required when resolving by name.
    registry : RegistryProtocol
        Registry to query.

    Returns
    -------
    str
        Resolved run ID.

    Notes
    -----
    Resolution order:
    1. Try as run ID (fast path, exact match)
    2. If project provided and ID not found, try as name within project
    3. Return as-is if not found (let caller handle the error)
    """
    # Fast path: try as ID first
    if registry.exists(run_id_or_name):
        return run_id_or_name

    # If project provided, try as name
    if project is not None:
        runs = registry.list_runs(
            project=project,
            name=run_id_or_name,
            limit=1,
        )
        if runs:
            return runs[0]["run_id"]

    # Return as-is - caller will get proper error from registry.load()
    return run_id_or_name


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
    artifacts: list[ArtifactRef] | None = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        """Cache immutable fields and prepare finalization support."""
        # These fields are set once at construction and never modified.
        self._cached_run_id: str = self.record.get("run_id", "")
        self._cached_created_at: str = self.record.get("created_at", "")

        # Snapshot populated by mark_finalized(); once set, all property
        # reads bypass the lock entirely.
        self._finalized: bool = False
        self._snapshot: dict[str, Any] = {}

    def mark_finalized(self) -> None:
        """
        Freeze a snapshot of all mutable fields.

        After this call every property read returns cached values without
        acquiring the lock. Intended to be called once the run record is
        fully persisted and will no longer be mutated.
        """
        with self._lock:
            info = self.record.get("info", {})
            data = self.record.get("data", {})
            proj = self.record.get("project", {})
            backend = self.record.get("backend", {})
            fps = self.record.get("fingerprints", {})

            self._snapshot = {
                "project": (
                    proj.get("name", "")
                    if isinstance(proj, dict)
                    else (str(proj) if proj else "")
                ),
                "adapter": self.record.get("adapter", ""),
                "status": (
                    info.get("status", "RUNNING")
                    if isinstance(info, dict)
                    else "RUNNING"
                ),
                "ended_at": (info.get("ended_at") if isinstance(info, dict) else None),
                "run_name": (info.get("run_name") if isinstance(info, dict) else None),
                "backend_name": (
                    backend.get("name") if isinstance(backend, dict) else None
                ),
                "group_id": self.record.get("group_id"),
                "group_name": self.record.get("group_name"),
                "parent_run_id": self.record.get("parent_run_id"),
                "fingerprints": dict(fps) if isinstance(fps, dict) else {},
                "params": (
                    dict(data.get("params", {}))
                    if isinstance(data, dict) and isinstance(data.get("params"), dict)
                    else {}
                ),
                "metrics": (
                    dict(data.get("metrics", {}))
                    if isinstance(data, dict) and isinstance(data.get("metrics"), dict)
                    else {}
                ),
                "tags": (
                    dict(data.get("tags", {}))
                    if isinstance(data, dict) and isinstance(data.get("tags"), dict)
                    else {}
                ),
            }
            self._finalized = True

    def _get_artifacts(self) -> list[ArtifactRef]:
        """
        Safely get artifacts list.

        Returns
        -------
        list of ArtifactRef
            Artifacts list (empty if None).
        """
        return self.artifacts if self.artifacts is not None else []

    @property
    def run_id(self) -> str:
        """
        Get the unique run identifier.

        Returns
        -------
        str
            Run ID, typically a ULID. Empty string if not set.
        """
        return self._cached_run_id

    @property
    def project(self) -> str:
        """
        Get the project name.

        Returns
        -------
        str
            Project name. Empty string if not set.
        """
        if self._finalized:
            return self._snapshot["project"]
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
        if self._finalized:
            return self._snapshot["adapter"]
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
        if self._finalized:
            return self._snapshot["status"]
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
        return self._cached_created_at

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
        if self._finalized:
            return dict(self._snapshot["fingerprints"])
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
        if self._finalized:
            return self._snapshot["backend_name"]
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
        if self._finalized:
            return self._snapshot["group_id"]
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
        if self._finalized:
            return self._snapshot["group_name"]
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
        if self._finalized:
            return self._snapshot["parent_run_id"]
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
        if self._finalized:
            return self._snapshot["run_name"]
        with self._lock:
            info = self.record.get("info", {})
            if isinstance(info, dict):
                return info.get("run_name")
            return None

    @property
    def name(self) -> str | None:
        """
        Alias for run_name.

        Returns
        -------
        str or None
            Run name, or None if not set.
        """
        return self.run_name

    @property
    def ended_at(self) -> str | None:
        """
        Get the run end timestamp.

        Returns
        -------
        str or None
            ISO 8601 formatted end timestamp, or None if still running.
        """
        if self._finalized:
            return self._snapshot["ended_at"]
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
        if self._finalized:
            return dict(self._snapshot["params"])
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
        if self._finalized:
            return dict(self._snapshot["metrics"])
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
        if self._finalized:
            return dict(self._snapshot["tags"])
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
            artifacts = self._get_artifacts()
            result["artifacts"] = [a.to_dict() for a in artifacts]
        return result

    def __repr__(self) -> str:
        """Return a concise representation of the run record."""
        artifacts = self._get_artifacts()
        return (
            f"RunRecord(run_id={self.run_id!r}, project={self.project!r}, "
            f"adapter={self.adapter!r}, status={self.status!r}, "
            f"artifacts={len(artifacts)})"
        )
