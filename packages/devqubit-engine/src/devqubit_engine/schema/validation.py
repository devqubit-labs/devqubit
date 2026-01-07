# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
JSON Schema validation for run records and execution envelopes.

This module validates run records and execution envelopes against JSON Schema
Draft 2020-12. The schemas are bundled inside the devqubit-engine package to
ensure validation is reproducible across environments.

Supported Schemas
-----------------
- ``devqubit.run/0.1``: Quantum experiment run records
- ``devqubit.envelope/0.1``: Execution envelope snapshots (device, program,
  execution, result)

Notes
-----
Validation is optional and controlled by the ``DEVQUBIT_VALIDATE``
environment variable or ``Config.validate`` setting.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from importlib import resources
from typing import Any

import jsonschema


logger = logging.getLogger(__name__)


# Mapping of schema IDs to bundled schema filenames
_SCHEMA_MAP: dict[str, str] = {
    "devqubit.run/0.1": "devqubit.run.0.1.schema.json",
    "devqubit.envelope/0.1": "devqubit.envelope.0.1.schema.json",
}


@lru_cache(maxsize=8)
def _load_schema(schema_id: str) -> dict[str, Any]:
    """
    Load and cache a JSON schema by its identifier.

    Parameters
    ----------
    schema_id : str
        Schema identifier (e.g., "devqubit.run/0.1", "devqubit.envelope/0.1").

    Returns
    -------
    dict
        Parsed JSON schema.

    Raises
    ------
    ValueError
        If the schema_id is not supported.
    FileNotFoundError
        If the schema file is not found in the package.
    jsonschema.exceptions.SchemaError
        If the schema itself is invalid.

    Notes
    -----
    Results are cached using ``functools.lru_cache`` for performance.
    The cache can be cleared with ``_load_schema.cache_clear()``.
    """
    filename = _SCHEMA_MAP.get(schema_id)
    if not filename:
        supported = ", ".join(sorted(_SCHEMA_MAP.keys()))
        raise ValueError(
            f"Unsupported schema: {schema_id!r}. Supported schemas: {supported}"
        )

    logger.debug("Loading schema: %s from %s", schema_id, filename)

    try:
        schema_file = resources.files("devqubit_engine.schema") / filename
        with schema_file.open("r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Schema file not found: {filename}. "
            "Ensure devqubit-engine is properly installed."
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file {filename}: {e}") from None

    # Validate the schema itself (catches schema authoring errors early)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as e:
        raise ValueError(f"Invalid schema {schema_id}: {e.message}") from e

    logger.debug("Schema %s loaded and validated successfully", schema_id)
    return schema


@lru_cache(maxsize=4)
def _get_validator(schema_id: str) -> jsonschema.Draft202012Validator:
    """
    Get a cached Draft 2020-12 validator for a schema.

    Parameters
    ----------
    schema_id : str
        Schema identifier.

    Returns
    -------
    jsonschema.Draft202012Validator
        Validator instance with format checking enabled.
    """
    schema = _load_schema(schema_id)
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )


def _format_validation_error(error: jsonschema.ValidationError) -> str:
    """
    Format a validation error into a human-readable string.

    Parameters
    ----------
    error : jsonschema.ValidationError
        Validation error from jsonschema.

    Returns
    -------
    str
        Formatted error message including:
        - JSON path to the invalid value
        - Error message
        - Schema path where the constraint is defined
    """
    # Format instance path (e.g., ['data']['params'] -> "['data']['params']")
    path_parts = list(error.absolute_path)
    if path_parts:
        instance_path = "".join(
            f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in path_parts
        )
    else:
        instance_path = "$"

    # Format schema path for debugging
    schema_path = "/".join(str(p) for p in error.absolute_schema_path)

    return f"{instance_path}: {error.message} (schema: {schema_path})"


def _format_all_validation_errors(
    errors: list[jsonschema.ValidationError],
    max_errors: int = 5,
) -> str:
    """
    Format multiple validation errors into a summary string.

    Parameters
    ----------
    errors : list of ValidationError
        List of validation errors.
    max_errors : int, optional
        Maximum number of errors to include in detail. Default is 5.

    Returns
    -------
    str
        Formatted multi-line error summary.
    """
    lines = [f"Validation failed with {len(errors)} error(s):"]

    for i, error in enumerate(errors[:max_errors]):
        lines.append(f"  {i + 1}. {_format_validation_error(error)}")

    if len(errors) > max_errors:
        lines.append(f"  ... and {len(errors) - max_errors} more error(s)")

    return "\n".join(lines)


def _validate_record(
    record: dict[str, Any],
    *,
    raise_on_error: bool = True,
    record_type: str = "record",
) -> list[jsonschema.ValidationError]:
    """
    Internal validation helper for any schema-identified record.

    Parameters
    ----------
    record : dict
        Record dictionary to validate. Must contain a "schema" field
        identifying the schema version.
    raise_on_error : bool, optional
        If True (default), raise ValueError on validation failure.
        If False, return the list of validation errors.
    record_type : str, optional
        Human-readable record type for error messages (e.g., "run record",
        "envelope").

    Returns
    -------
    list of ValidationError
        Empty list if valid, or list of errors if ``raise_on_error=False``.

    Raises
    ------
    ValueError
        If validation fails and ``raise_on_error=True``, or if the
        record is missing the "schema" field.
    """
    schema_id = record.get("schema")
    if not schema_id:
        if raise_on_error:
            raise ValueError(
                f"{record_type.capitalize()} missing 'schema' field. "
                "All records must specify their schema version."
            )
        # Create a synthetic error for missing schema
        return [
            jsonschema.ValidationError(
                message="'schema' is a required property",
                path=[],
                schema_path=["required"],
            )
        ]

    logger.debug("Validating %s against schema: %s", record_type, schema_id)

    try:
        validator = _get_validator(schema_id)
    except (ValueError, FileNotFoundError) as e:
        if raise_on_error:
            raise ValueError(f"Cannot validate {record_type}: {e}") from e
        return [
            jsonschema.ValidationError(
                message=str(e),
                path=[],
                schema_path=[],
            )
        ]

    # Collect all errors, sorted by path for consistent ordering
    errors = sorted(
        validator.iter_errors(record),
        key=lambda e: (list(e.absolute_path), e.message),
    )

    # Determine record identifier for logging
    record_id = record.get("run_id") or record.get("envelope_id") or "<unknown>"

    if errors:
        logger.warning(
            "Validation failed for %s %s: %d error(s)",
            record_type,
            record_id,
            len(errors),
        )

        if raise_on_error:
            # Raise with first error for simple cases, full summary for multiple
            if len(errors) == 1:
                raise ValueError(
                    f"Validation failed: {_format_validation_error(errors[0])}"
                )
            raise ValueError(_format_all_validation_errors(errors))

    else:
        logger.debug("Validation passed for %s %s", record_type, record_id)

    return errors


def validate_run_record(
    record: dict[str, Any],
    *,
    raise_on_error: bool = True,
) -> list[jsonschema.ValidationError]:
    """
    Validate a run record against its schema.

    Parameters
    ----------
    record : dict
        Run record dictionary to validate. Must contain a "schema" field
        identifying the schema version (e.g., "devqubit.run/0.1").
    raise_on_error : bool, optional
        If True (default), raise ValueError on validation failure.
        If False, return the list of validation errors.

    Returns
    -------
    list of ValidationError
        Empty list if valid, or list of errors if ``raise_on_error=False``.

    Raises
    ------
    ValueError
        If validation fails and ``raise_on_error=True``, or if the
        record is missing the "schema" field.
    """
    return _validate_record(
        record,
        raise_on_error=raise_on_error,
        record_type="run record",
    )


def validate_envelope(
    envelope: dict[str, Any],
    *,
    raise_on_error: bool = True,
) -> list[jsonschema.ValidationError]:
    """
    Validate an execution envelope against its schema.

    Parameters
    ----------
    envelope : dict
        Envelope dictionary to validate. Must contain a "schema" field
        identifying the schema version (e.g., "devqubit.envelope/0.1").
    raise_on_error : bool, optional
        If True (default), raise ValueError on validation failure.
        If False, return the list of validation errors.

    Returns
    -------
    list of ValidationError
        Empty list if valid, or list of errors if ``raise_on_error=False``.

    Raises
    ------
    ValueError
        If validation fails and ``raise_on_error=True``, or if the
        record is missing the "schema" field.
    """
    return _validate_record(
        envelope,
        raise_on_error=raise_on_error,
        record_type="envelope",
    )


def clear_cache() -> None:
    """
    Clear the schema and validator caches.

    Useful for testing or when schema files may have changed.
    """
    _load_schema.cache_clear()
    _get_validator.cache_clear()
    logger.debug("Schema caches cleared")
