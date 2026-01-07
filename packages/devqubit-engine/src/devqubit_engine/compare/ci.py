# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Baseline verification for CI testing.

This module provides integration and supports JUnit XML output
and GitHub Actions annotations.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from devqubit_engine.compare.results import VerifyResult


logger = logging.getLogger(__name__)

# Pattern matching XML 1.0 illegal characters
# Valid: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
_XML_ILLEGAL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff\ufffe\uffff]"
)


def _sanitize_xml_text(text: str) -> str:
    """
    Remove characters illegal in XML 1.0.

    Parameters
    ----------
    text : str
        Input text that may contain illegal characters.

    Returns
    -------
    str
        Sanitized text safe for XML 1.0.
    """
    return _XML_ILLEGAL_CHARS_RE.sub("", text)


def result_to_junit(
    result: VerifyResult,
    *,
    testsuite_name: str = "devqubit.verify",
    testcase_name: str | None = None,
) -> str:
    """
    Convert verification result to JUnit XML format.

    Parameters
    ----------
    result : VerifyResult
        Verification result.
    testsuite_name : str, default="devqubit.verify"
        Name for the test suite element.
    testcase_name : str, optional
        Name for the test case. Uses candidate run ID if not provided.

    Returns
    -------
    str
        JUnit XML string.
    """
    testsuite = ET.Element("testsuite")
    testsuite.set("name", testsuite_name)
    testsuite.set("tests", "1")
    testsuite.set("failures", "0" if result.ok else "1")
    testsuite.set("errors", "0")
    testsuite.set("time", f"{result.duration_ms / 1000:.3f}")

    testcase = ET.SubElement(testsuite, "testcase")
    testcase.set("name", testcase_name or result.candidate_run_id or "verify")
    testcase.set("classname", testsuite_name)
    testcase.set("time", f"{result.duration_ms / 1000:.3f}")

    if not result.ok:
        failure = ET.SubElement(testcase, "failure")
        failure.set("type", "VerificationFailure")

        # Sanitize failure messages for XML
        sanitized_failures = [_sanitize_xml_text(f) for f in result.failures]
        failure.set("message", "; ".join(sanitized_failures))

        failure_text = "\n".join(
            [
                f"Baseline: {result.baseline_run_id}",
                f"Candidate: {result.candidate_run_id}",
                "",
                "Failures:",
                *[f"  - {f}" for f in sanitized_failures],
            ]
        )
        failure.text = _sanitize_xml_text(failure_text)

    properties = ET.SubElement(testsuite, "properties")
    for name, value in [
        ("baseline_run_id", result.baseline_run_id or ""),
        ("candidate_run_id", result.candidate_run_id or ""),
        ("ok", str(result.ok)),
    ]:
        prop = ET.SubElement(properties, "property")
        prop.set("name", name)
        prop.set("value", value)

    return ET.tostring(testsuite, encoding="unicode")


def write_junit(
    result: VerifyResult,
    output_path: Path | str,
    *,
    testsuite_name: str = "devqubit.verify",
    testcase_name: str | None = None,
) -> None:
    """
    Write verification result to JUnit XML file.

    Parameters
    ----------
    result : VerifyResult
        Verification result.
    output_path : Path or str
        Output file path.
    testsuite_name : str, default="devqubit.verify"
        Name for the test suite element.
    testcase_name : str, optional
        Name for the test case.
    """
    xml = result_to_junit(
        result, testsuite_name=testsuite_name, testcase_name=testcase_name
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")
    logger.debug("Wrote JUnit report to %s", output_path)


def result_to_github_annotations(result: VerifyResult) -> str:
    """
    Convert verification result to GitHub Actions annotation format.

    Parameters
    ----------
    result : VerifyResult
        Verification result.

    Returns
    -------
    str
        GitHub Actions workflow commands.

    Examples
    --------
    >>> annotations = result_to_github_annotations(result)
    >>> print(annotations)  # Print in workflow to create annotations
    """
    lines: list[str] = []

    if result.ok:
        lines.append(
            f"::notice title=Verification Passed::"
            f"Candidate {result.candidate_run_id} matches baseline {result.baseline_run_id}"
        )
    else:
        for failure in result.failures:
            # Escape special characters for GitHub Actions
            msg = failure.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
            lines.append(f"::error title=Verification Failed::{msg}")

        lines.append(
            f"::error title=Summary::"
            f"Candidate {result.candidate_run_id} failed verification "
            f"against baseline {result.baseline_run_id}"
        )

    return "\n".join(lines)
