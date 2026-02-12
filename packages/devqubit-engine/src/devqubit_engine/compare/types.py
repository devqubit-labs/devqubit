# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Type definitions for comparison operations.

This module contains enumerations and simple types used across the
comparison subsystem. Keeping these separate avoids circular imports
and provides a clean type vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProgramMatchMode(str, Enum):
    """
    Program matching mode for verification.

    Attributes
    ----------
    EXACT : str
        Require artifact digest equality (strict reproducibility).
        Use when you need byte-for-byte identical circuits.
    STRUCTURAL : str
        Require circuit_hash (structure) equality (variational-friendly).
        Use for variational circuits where parameter values differ but
        structure is same.
    EITHER : str
        Pass if exact OR structural match (recommended default).
        Detects true program changes without breaking variational workflows.
    """

    EXACT = "exact"
    STRUCTURAL = "structural"
    EITHER = "either"


class ProgramMatchStatus(str, Enum):
    """
    Detailed program match status.

    Based on structural (structural_hash) and parametric (parametric_hash)
    comparison.

    Attributes
    ----------
    FULL_MATCH : str
        Both structural and parametric hashes match (identical execution).
    STRUCTURAL_MATCH_PARAM_MISMATCH : str
        Same circuit structure but different parameter values (VQE iteration).
    STRUCTURAL_MISMATCH : str
        Different circuit structure (different program).
    HASH_UNAVAILABLE : str
        Cannot determine - hashes not available (manual run).
    """

    FULL_MATCH = "FULL_MATCH"
    STRUCTURAL_MATCH_PARAM_MISMATCH = "STRUCTURAL_MATCH_PARAM_MISMATCH"
    STRUCTURAL_MISMATCH = "STRUCTURAL_MISMATCH"
    HASH_UNAVAILABLE = "HASH_UNAVAILABLE"


@dataclass
class FormatOptions:
    """
    Formatting options for text reports.

    Attributes
    ----------
    max_drifts : int
        Maximum drift metrics to display. Default is 5.
    max_circuit_changes : int
        Maximum circuit changes to display. Default is 10.
    max_param_changes : int
        Maximum parameter changes to display. Default is 10.
    max_metric_changes : int
        Maximum metric changes to display. Default is 10.
    width : int
        Line width for text output. Default is 70.
    """

    max_drifts: int = 5
    max_circuit_changes: int = 10
    max_param_changes: int = 10
    max_metric_changes: int = 10
    width: int = 70
