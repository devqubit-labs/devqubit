# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Artifact utilities and browsing.

This module provides tools for finding, extracting, loading, listing,
viewing, and comparing artifacts from run records. Artifacts are the
stored outputs of quantum experiments, including circuits, results,
device snapshots, and other data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from devqubit_engine.core.record import RunRecord
from devqubit_engine.core.types import ArtifactRef
from devqubit_engine.storage.protocols import ObjectStoreProtocol


logger = logging.getLogger(__name__)


def find_artifact(
    record: RunRecord,
    *,
    role: str | None = None,
    kind_contains: str | None = None,
) -> ArtifactRef | None:
    """
    Find first artifact matching criteria.

    Parameters
    ----------
    record : RunRecord
        Run record containing artifacts.
    role : str, optional
        Required artifact role (e.g., "program", "results", "device_snapshot").
    kind_contains : str, optional
        Substring required in artifact kind (case-insensitive).

    Returns
    -------
    ArtifactRef or None
        First matching artifact reference, or None if not found.
    """
    for artifact in record.artifacts:
        if role and artifact.role != role:
            continue
        if kind_contains:
            kind = artifact.kind.lower()
            if kind_contains.lower() not in kind:
                continue
        logger.debug("Found artifact: role=%s, kind=%s", artifact.role, artifact.kind)
        return artifact
    return None


def load_json_artifact(
    artifact: ArtifactRef,
    store: ObjectStoreProtocol,
) -> Any | None:
    """
    Load and parse JSON artifact payload from object store.

    Parameters
    ----------
    artifact : ArtifactRef
        Artifact reference containing digest.
    store : ObjectStoreProtocol
        Object store to retrieve data from.

    Returns
    -------
    Any or None
        Parsed JSON payload, or None on failure.

    Notes
    -----
    Returns None on any failure (missing object, decode error, parse error).
    Check logs for details on failures.
    """
    try:
        data = store.get_bytes(artifact.digest)
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        logger.debug(
            "Failed to load JSON artifact %s: %s",
            artifact.digest[:16],
            e,
        )
        return None


def get_artifact_digests(
    record: RunRecord | dict[str, Any],
    role: str,
    *,
    kind_contains: str | None = None,
) -> list[str]:
    """
    Extract sorted artifact digests from a run record.

    Parameters
    ----------
    record : RunRecord or dict
        Run record containing artifacts. Accepts dict for
        backward compatibility with raw record dictionaries.
    role : str
        Filter by artifact role (e.g., "program", "results").
    kind_contains : str, optional
        Filter by substring in artifact kind.

    Returns
    -------
    list of str
        Sorted list of artifact digests matching filters.
    """
    # Handle both RunRecord and dict for backward compatibility
    if isinstance(record, RunRecord):
        artifacts = record.artifacts
    else:
        artifacts = []
        for artifact_dict in record.get("artifacts", []) or []:
            if isinstance(artifact_dict, dict):
                try:
                    artifacts.append(ArtifactRef.from_dict(artifact_dict))
                except (KeyError, ValueError):
                    continue

    digests: list[str] = []
    for artifact in artifacts:
        if artifact.role != role:
            continue

        if kind_contains:
            kind = artifact.kind.lower()
            if kind_contains.lower() not in kind:
                continue

        digests.append(artifact.digest)

    return sorted(digests)


@dataclass
class ArtifactInfo:
    """
    Extended artifact information for display.

    Attributes
    ----------
    ref : ArtifactRef
        Underlying artifact reference.
    index : int
        Position in artifacts list.
    name : str
        Artifact name from metadata.
    size : int or None
        Size in bytes if available.
    """

    ref: ArtifactRef
    index: int
    name: str
    size: int | None = None

    @property
    def kind(self) -> str:
        """Get artifact kind."""
        return self.ref.kind

    @property
    def digest(self) -> str:
        """Get content digest."""
        return self.ref.digest

    @property
    def digest_short(self) -> str:
        """Get shortened digest for display."""
        return self.ref.digest[:20] + "..."

    @property
    def role(self) -> str:
        """Get artifact role."""
        return self.ref.role

    @property
    def media_type(self) -> str:
        """Get MIME type."""
        return self.ref.media_type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display."""
        d: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "media_type": self.media_type,
            "digest": self.digest,
        }
        if self.size is not None:
            d["size_bytes"] = self.size
        if self.ref.meta:
            d["meta"] = self.ref.meta
        return d

    def __repr__(self) -> str:
        """Return string representation."""
        size_str = f", {self.size}B" if self.size else ""
        return f"ArtifactInfo({self.index}: {self.role}/{self.kind}{size_str})"


def list_artifacts(
    record: RunRecord,
    *,
    role: str | None = None,
    kind_contains: str | None = None,
    store: ObjectStoreProtocol | None = None,
) -> list[ArtifactInfo]:
    """
    List artifacts from a run record.

    Parameters
    ----------
    record : RunRecord
        Run record to list artifacts from.
    role : str, optional
        Filter by role (e.g., "program", "results", "device_snapshot").
    kind_contains : str, optional
        Filter by kind substring.
    store : ObjectStoreProtocol, optional
        If provided, include size information.

    Returns
    -------
    list of ArtifactInfo
        Artifact information sorted by role then kind.
    """
    results: list[ArtifactInfo] = []

    for i, art in enumerate(record.artifacts):
        # Apply filters
        if role and art.role != role:
            continue
        if kind_contains and kind_contains.lower() not in art.kind.lower():
            continue

        # Extract name from metadata
        meta = art.meta or {}
        name = (
            meta.get("name") or meta.get("filename") or meta.get("program_name") or ""
        )

        # Get size if store provided
        size = None
        if store:
            try:
                size = store.get_size(art.digest)
            except Exception:
                pass

        results.append(ArtifactInfo(ref=art, index=i, name=name, size=size))

    # Sort by role, then kind
    results.sort(key=lambda a: (a.role, a.kind, a.index))

    logger.debug("Listed %d artifacts (filtered by role=%s)", len(results), role)
    return results


def get_artifact(
    record: RunRecord,
    selector: str | int,
) -> ArtifactRef | None:
    """
    Get artifact by index or selector.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Either:
        - int: artifact index
        - str: digest prefix, kind, or "role:kind" pattern

    Returns
    -------
    ArtifactRef or None
        Matching artifact or None if not found.
    """
    if isinstance(selector, int):
        if 0 <= selector < len(record.artifacts):
            return record.artifacts[selector]
        return None

    selector = str(selector)

    # Try digest prefix
    if selector.startswith("sha256:"):
        for art in record.artifacts:
            if art.digest.startswith(selector):
                return art
        return None

    # Try role:kind pattern
    if ":" in selector:
        role, kind = selector.split(":", 1)
        for art in record.artifacts:
            if art.role == role and kind in art.kind:
                return art
        return None

    # Try kind match
    for art in record.artifacts:
        if selector in art.kind:
            return art

    return None


def get_artifact_bytes(
    record: RunRecord,
    selector: str | int,
    store: ObjectStoreProtocol,
) -> bytes | None:
    """
    Get artifact content bytes.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Artifact selector (see get_artifact).
    store : ObjectStoreProtocol
        Object store.

    Returns
    -------
    bytes or None
        Artifact content or None if not found.
    """
    art = get_artifact(record, selector)
    if not art:
        return None

    try:
        return store.get_bytes(art.digest)
    except Exception as e:
        logger.debug("Failed to get artifact bytes: %s", e)
        return None


def get_artifact_text(
    record: RunRecord,
    selector: str | int,
    store: ObjectStoreProtocol,
    *,
    encoding: str = "utf-8",
) -> str | None:
    """
    Get artifact content as text.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Artifact selector.
    store : ObjectStoreProtocol
        Object store.
    encoding : str, default="utf-8"
        Text encoding.

    Returns
    -------
    str or None
        Artifact text content or None if not found or decode fails.
    """
    data = get_artifact_bytes(record, selector, store)
    if data is None:
        return None

    try:
        return data.decode(encoding)
    except UnicodeDecodeError as e:
        logger.debug("Failed to decode artifact as %s: %s", encoding, e)
        return None


def get_artifact_json(
    record: RunRecord,
    selector: str | int,
    store: ObjectStoreProtocol,
) -> Any | None:
    """
    Get artifact content as parsed JSON.

    Parameters
    ----------
    record : RunRecord
        Run record.
    selector : str or int
        Artifact selector.
    store : ObjectStoreProtocol
        Object store.

    Returns
    -------
    Any or None
        Parsed JSON or None if not found/invalid.
    """
    text = get_artifact_text(record, selector, store)
    if text is None:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse artifact as JSON: %s", e)
        return None


def get_program_qasm(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    canonical: bool = True,
    index: int = 0,
) -> str | None:
    """
    Get OpenQASM3 program from a run.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    canonical : bool, default=True
        If True, prefer canonical form; if False, prefer raw.
    index : int, default=0
        Program index for multi-circuit runs.

    Returns
    -------
    str or None
        OpenQASM3 source or None if not found.

    Notes
    -----
    First checks program anchors in the run record, then falls back
    to searching artifacts directly. If canonical form is requested
    but not found, automatically falls back to raw form.
    """
    # Check anchors first
    program_section = record.record.get("program", {})
    oq3_anchors = program_section.get("openqasm3", [])

    if isinstance(oq3_anchors, list) and index < len(oq3_anchors):
        anchor = oq3_anchors[index]
        ptr_key = "canonical" if canonical else "raw"
        ptr = anchor.get(ptr_key)

        if isinstance(ptr, dict) and "digest" in ptr:
            try:
                data = store.get_bytes(ptr["digest"])
                logger.debug("Loaded QASM from anchor: %s", ptr_key)
                return data.decode("utf-8")
            except Exception as e:
                logger.debug("Failed to load QASM from anchor: %s", e)

    # Fallback: search artifacts
    kind_pattern = "openqasm3.canonical" if canonical else "openqasm3"
    for art in record.artifacts:
        if art.role == "program" and kind_pattern in art.kind:
            meta = art.meta or {}
            prog_idx = meta.get("program_index", 0)
            if prog_idx == index:
                try:
                    data = store.get_bytes(art.digest)
                    logger.debug("Loaded QASM from artifact: %s", art.kind)
                    return data.decode("utf-8")
                except Exception as e:
                    logger.debug("Failed to load QASM from artifact: %s", e)

    # If canonical not found and we wanted it, try raw
    if canonical:
        return get_program_qasm(record, store, canonical=False, index=index)

    return None


def list_programs(record: RunRecord) -> list[dict[str, Any]]:
    """
    List programs in a run.

    Parameters
    ----------
    record : RunRecord
        Run record.

    Returns
    -------
    list of dict
        Program information with keys:
        - index: Program index
        - name: Program name
        - has_raw: Whether raw QASM is available
        - has_canonical: Whether canonical QASM is available
    """
    programs: dict[int, dict[str, Any]] = {}

    # From anchors
    program_section = record.record.get("program", {})
    oq3_anchors = program_section.get("openqasm3", [])

    if isinstance(oq3_anchors, list):
        for anchor in oq3_anchors:
            if not isinstance(anchor, dict):
                continue
            idx = anchor.get("index", 0)
            programs[idx] = {
                "index": idx,
                "name": anchor.get("name", f"circuit_{idx}"),
                "has_raw": "raw" in anchor,
                "has_canonical": "canonical" in anchor,
            }

    # From artifacts (fill gaps)
    for art in record.artifacts:
        if art.role != "program":
            continue
        if "openqasm3" not in art.kind:
            continue

        meta = art.meta or {}
        idx = meta.get("program_index", 0)

        if idx not in programs:
            programs[idx] = {
                "index": idx,
                "name": meta.get("program_name", f"circuit_{idx}"),
                "has_raw": False,
                "has_canonical": False,
            }

        if ".canonical" in art.kind:
            programs[idx]["has_canonical"] = True
        else:
            programs[idx]["has_raw"] = True

    return sorted(programs.values(), key=lambda p: p["index"])


@dataclass
class CountsInfo:
    """
    Measurement counts information.

    Attributes
    ----------
    counts : dict
        Raw counts as {bitstring: count}.
    total_shots : int
        Total number of shots.
    num_outcomes : int
        Number of unique outcomes.
    """

    counts: dict[str, int]
    total_shots: int
    num_outcomes: int

    @property
    def probabilities(self) -> dict[str, float]:
        """Get normalized probabilities."""
        if self.total_shots == 0:
            return {}
        return {k: v / self.total_shots for k, v in self.counts.items()}

    def top_k(self, k: int = 10) -> list[tuple[str, int, float]]:
        """
        Get top-k outcomes by count.

        Parameters
        ----------
        k : int, default=10
            Number of outcomes to return.

        Returns
        -------
        list of tuple
            List of (bitstring, count, probability) tuples.
        """
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        result: list[tuple[str, int, float]] = []
        for bitstring, count in sorted_counts[:k]:
            prob = count / self.total_shots if self.total_shots > 0 else 0.0
            result.append((bitstring, count, prob))
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CountsInfo(shots={self.total_shots}, " f"outcomes={self.num_outcomes})"


def get_counts(
    record: RunRecord,
    store: ObjectStoreProtocol,
    *,
    experiment_index: int | None = None,
) -> CountsInfo | None:
    """
    Get measurement counts from a run.

    Parameters
    ----------
    record : RunRecord
        Run record.
    store : ObjectStoreProtocol
        Object store.
    experiment_index : int, optional
        If provided, get counts for specific experiment in batch.
        If None, aggregates counts from all experiments.

    Returns
    -------
    CountsInfo or None
        Counts information or None if not found.
    """
    # Find results artifact with counts
    artifact = find_artifact(record, role="results", kind_contains="counts")
    if not artifact:
        logger.debug("No counts artifact found in run %s", record.run_id)
        return None

    payload = load_json_artifact(artifact, store)
    if not isinstance(payload, dict):
        return None

    # Handle batch format
    experiments = payload.get("experiments")
    if isinstance(experiments, list) and experiments:
        if experiment_index is not None:
            # Specific experiment
            if experiment_index < len(experiments):
                exp = experiments[experiment_index]
                raw_counts = exp.get("counts", {})
            else:
                logger.debug(
                    "Experiment index %d out of range (max: %d)",
                    experiment_index,
                    len(experiments) - 1,
                )
                return None
        else:
            # Aggregate all experiments
            raw_counts: dict[str, int] = {}
            for exp in experiments:
                if isinstance(exp, dict):
                    for k, v in exp.get("counts", {}).items():
                        raw_counts[str(k)] = raw_counts.get(str(k), 0) + int(v)
    else:
        # Simple format
        raw_counts = payload.get("counts", {})

    if not isinstance(raw_counts, dict) or not raw_counts:
        return None

    # Build CountsInfo
    counts = {str(k): int(v) for k, v in raw_counts.items()}
    total = sum(counts.values())

    logger.debug("Loaded counts: %d shots, %d outcomes", total, len(counts))
    return CountsInfo(counts=counts, total_shots=total, num_outcomes=len(counts))


def diff_counts(
    counts_a: CountsInfo,
    counts_b: CountsInfo,
    *,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Compare two counts distributions.

    Parameters
    ----------
    counts_a : CountsInfo
        First counts (baseline).
    counts_b : CountsInfo
        Second counts (candidate).
    top_k : int, default=20
        Number of top outcomes to show.

    Returns
    -------
    dict
        Comparison result with keys:
        - tvd: Total Variation Distance
        - shots_a, shots_b: Shot counts
        - outcomes_a, outcomes_b: Outcome counts
        - aligned: Top-k aligned outcomes with deltas
        - total_outcomes: Total unique outcomes
    """
    probs_a = counts_a.probabilities
    probs_b = counts_b.probabilities

    all_keys = set(probs_a.keys()) | set(probs_b.keys())

    # Compute TVD
    tvd = 0.5 * sum(abs(probs_a.get(k, 0.0) - probs_b.get(k, 0.0)) for k in all_keys)

    # Build aligned comparison
    aligned: list[dict[str, Any]] = []
    for key in all_keys:
        p_a = probs_a.get(key, 0.0)
        p_b = probs_b.get(key, 0.0)
        aligned.append(
            {
                "outcome": key,
                "prob_a": p_a,
                "prob_b": p_b,
                "delta": p_b - p_a,
                "count_a": counts_a.counts.get(key, 0),
                "count_b": counts_b.counts.get(key, 0),
            }
        )

    # Sort by max probability
    aligned.sort(key=lambda x: max(x["prob_a"], x["prob_b"]), reverse=True)

    return {
        "tvd": tvd,
        "shots_a": counts_a.total_shots,
        "shots_b": counts_b.total_shots,
        "outcomes_a": counts_a.num_outcomes,
        "outcomes_b": counts_b.num_outcomes,
        "aligned": aligned[:top_k],
        "total_outcomes": len(all_keys),
    }


def format_counts_table(counts: CountsInfo, top_k: int = 10) -> str:
    """
    Format counts as ASCII table.

    Parameters
    ----------
    counts : CountsInfo
        Counts to format.
    top_k : int, default=10
        Number of outcomes to show.

    Returns
    -------
    str
        Formatted table.
    """
    lines = [
        f"Total shots: {counts.total_shots:,}",
        f"Unique outcomes: {counts.num_outcomes}",
        "",
        f"{'Outcome':<20} {'Count':>10} {'Prob':>10}",
        "-" * 42,
    ]

    for bitstring, count, prob in counts.top_k(top_k):
        lines.append(f"{bitstring:<20} {count:>10,} {prob:>10.4f}")

    if counts.num_outcomes > top_k:
        lines.append(f"... and {counts.num_outcomes - top_k} more outcomes")

    return "\n".join(lines)


def format_artifacts_table(artifacts: list[ArtifactInfo]) -> str:
    """
    Format artifacts as ASCII table.

    Parameters
    ----------
    artifacts : list of ArtifactInfo
        Artifacts to format.

    Returns
    -------
    str
        Formatted table.
    """
    if not artifacts:
        return "No artifacts found."

    lines = [
        f"{'#':<4} {'Role':<15} {'Kind':<30} {'Name':<20} {'Digest':<25}",
        "-" * 95,
    ]

    for art in artifacts:
        digest_short = art.digest[7:19] + "..."  # sha256:XXXX...
        name = art.name[:18] + ".." if len(art.name) > 20 else art.name
        kind = art.kind[:28] + ".." if len(art.kind) > 30 else art.kind
        lines.append(
            f"{art.index:<4} {art.role:<15} {kind:<30} {name:<20} {digest_short:<25}"
        )

    return "\n".join(lines)
