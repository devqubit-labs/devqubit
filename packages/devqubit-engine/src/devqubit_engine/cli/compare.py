# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Comparison and verification CLI commands.

This module provides commands for comparing runs, verifying against
baselines, and replaying quantum experiments. Uses the compare module's
formatters for consistent output across CLI and Python API.

Commands
--------
diff
    Compare two runs or bundles comprehensively.
verify
    Verify a run against a baseline with policy checks.
replay
    Replay a quantum experiment on a simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from devqubit_engine.cli._utils import echo, root_from_ctx


def register(cli: click.Group) -> None:
    """Register compare commands with CLI."""
    cli.add_command(diff_cmd)
    cli.add_command(verify_cmd)
    cli.add_command(replay_cmd)


@click.command("diff")
@click.argument("ref_a")
@click.argument("ref_b")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Save report to file."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "summary"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--no-circuit-diff",
    is_flag=True,
    help="Skip circuit semantic comparison.",
)
@click.pass_context
def diff_cmd(
    ctx: click.Context,
    ref_a: str,
    ref_b: str,
    output: Path | None,
    fmt: str,
    no_circuit_diff: bool,
) -> None:
    """
    Compare two runs or bundles comprehensively.

    Shows complete comparison including parameters, program, device drift,
    TVD with bootstrap-calibrated noise context analysis.

    Examples:
        devqubit diff abc123 def456
        devqubit diff experiment1.zip experiment2.zip
        devqubit diff abc123 def456 --format json -o report.json
    """
    from devqubit_engine.compare.diff import diff
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    try:
        result = diff(
            ref_a,
            ref_b,
            registry=registry,
            store=store,
            include_circuit_diff=not no_circuit_diff,
        )
    except Exception as e:
        raise click.ClickException(f"Comparison failed: {e}") from e

    # Format output
    if fmt == "json":
        formatted = result.format_json()
    elif fmt == "summary":
        formatted = result.format_summary()
    else:
        formatted = result.format()

    # Write to file or stdout
    if output:
        output.write_text(formatted, encoding="utf-8")
        echo(f"Report saved to {output}")
    else:
        echo(formatted)


@click.command("verify")
@click.argument("candidate_id")
@click.option("--baseline", "-b", help="Baseline run ID (default: project baseline).")
@click.option("--project", "-p", help="Project for baseline lookup.")
@click.option("--tvd-max", type=float, help="Maximum allowed TVD.")
@click.option(
    "--noise-factor",
    type=float,
    help="Fail if TVD > noise_factor x noise_p95 (bootstrap-calibrated threshold). Recommended: 1.0-1.5.",
)
@click.option(
    "--program-match-mode",
    type=click.Choice(["exact", "structural", "either"]),
    default="either",
    help="Program matching mode: exact (digest), structural, either (default).",
)
@click.option(
    "--no-params-match",
    is_flag=True,
    help="Don't require parameters to match.",
)
@click.option(
    "--no-program-match",
    is_flag=True,
    help="Don't require program to match.",
)
@click.option("--strict", is_flag=True, help="Require fingerprint match.")
@click.option("--promote", is_flag=True, help="Promote to baseline on pass.")
@click.option("--allow-missing", is_flag=True, help="Pass if no baseline exists.")
@click.option(
    "--junit", type=click.Path(path_type=Path), help="Write JUnit XML report."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "github", "summary"]),
    default="text",
    help="Output format.",
)
@click.pass_context
def verify_cmd(
    ctx: click.Context,
    candidate_id: str,
    baseline: str | None,
    project: str | None,
    tvd_max: float | None,
    noise_factor: float | None,
    program_match_mode: str,
    no_params_match: bool,
    no_program_match: bool,
    strict: bool,
    promote: bool,
    allow_missing: bool,
    junit: Path | None,
    fmt: str,
) -> None:
    """
    Verify a run against baseline with full root-cause analysis.

    Shows comprehensive verification results including what failed,
    why it failed, and suggested actions.

    The --noise-factor option provides a shot-count-aware threshold using
    bootstrap-calibrated noise estimation. Verification fails if TVD exceeds
    noise_factor x noise_p95 (the 95th percentile of expected shot noise).
    Recommended values: 1.0 (strict) to 1.5 (lenient).

    The --program-match-mode option controls how programs are compared:
    - exact: require identical artifact digests (strict reproducibility)
    - structural: require same circuit structure (VQE/QAOA friendly)
    - either: pass if exact OR structural matches (default)

    Examples:
        devqubit verify abc123 --baseline def456
        devqubit verify abc123 --project myproject --promote
        devqubit verify abc123 --tvd-max 0.05 --format json
        devqubit verify abc123 --noise-factor 1.0
        devqubit verify abc123 --program-match-mode structural
    """
    from devqubit_engine.compare.ci import result_to_github_annotations, write_junit
    from devqubit_engine.compare.results import ProgramMatchMode
    from devqubit_engine.compare.verify import (
        VerifyPolicy,
        verify,
        verify_against_baseline,
    )
    from devqubit_engine.core.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    try:
        candidate = registry.load(candidate_id)
    except RunNotFoundError as e:
        raise click.ClickException(f"Candidate not found: {candidate_id}") from e

    # Convert string to enum
    match_mode = ProgramMatchMode(program_match_mode)

    policy = VerifyPolicy(
        params_must_match=not no_params_match,
        program_must_match=not no_program_match,
        program_match_mode=match_mode,
        fingerprint_must_match=strict,
        tvd_max=tvd_max,
        noise_factor=noise_factor,
        allow_missing_baseline=allow_missing,
    )

    if baseline:
        try:
            baseline_record = registry.load(baseline)
        except RunNotFoundError as e:
            raise click.ClickException(f"Baseline not found: {baseline}") from e

        result = verify(
            baseline_record,
            candidate,
            store_baseline=store,
            store_candidate=store,
            policy=policy,
        )
    else:
        proj = project or candidate.project
        if not proj:
            raise click.ClickException(
                "No project specified and candidate has no project"
            )

        result = verify_against_baseline(
            candidate,
            project=proj,
            store=store,
            registry=registry,
            policy=policy,
            promote_on_pass=promote,
        )

    # Write JUnit if requested
    if junit:
        write_junit(result, junit)
        if not ctx.obj.get("quiet"):
            echo(f"JUnit report written to {junit}")

    # Format output
    if fmt == "json":
        formatted = result.format_json()
    elif fmt == "github":
        formatted = result_to_github_annotations(result)
    elif fmt == "summary":
        formatted = result.format_summary()
    else:
        formatted = result.format()

    echo(formatted)

    # Add promotion notice if applicable
    if result.ok and promote and not baseline:
        echo(f"\n✓ Promoted {candidate_id} to baseline for project")

    ctx.exit(0 if result.ok else 1)


@click.command("replay")
@click.argument("ref", required=False)
@click.option("--backend", "-b", default=None, help="Simulator backend name.")
@click.option("--shots", "-s", type=int, help="Override shot count.")
@click.option("--seed", type=int, help="Random seed for reproducibility (best-effort).")
@click.option("--save", is_flag=True, help="Save replay as a new tracked run.")
@click.option("--project", "-p", help="Project name for saved run.")
@click.option(
    "--experimental",
    is_flag=True,
    help="Acknowledge experimental status (required).",
)
@click.option(
    "--list-backends", is_flag=True, help="List available simulator backends."
)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def replay_cmd(
    ctx: click.Context,
    ref: str | None,
    backend: str | None,
    shots: int | None,
    seed: int | None,
    save: bool,
    project: str | None,
    experimental: bool,
    list_backends: bool,
    fmt: str,
) -> None:
    """
    Replay a quantum experiment from a bundle or run.

    Reconstructs the circuit and executes it on a simulator backend.
    Use 'devqubit diff' to compare replay results with the original.

    EXPERIMENTAL: Replay is best-effort and may not be fully reproducible.
    Use --experimental flag to acknowledge this.

    Note: Currently only simulator backends are supported.
    Note: Only native SDK formats (QPY, JAQCD, Cirq JSON, Tape JSON) are supported.
          OpenQASM is NOT supported to ensure exact program representation.

    Examples:
        devqubit replay experiment.zip --experimental
        devqubit replay abc123 --backend aer_simulator --experimental --seed 42
        devqubit replay abc123 --experimental --save && devqubit diff abc123 <replay_id>
        devqubit replay --list-backends
    """
    from devqubit_engine.bundle.replay import list_available_backends, replay

    if list_backends:
        backends = list_available_backends()
        if backends:
            echo("Available simulator backends:")
            echo("(Note: only simulators are currently supported)")
            echo("")
            for sdk in sorted(backends.keys()):
                echo(f"  {sdk}:")
                for b in backends[sdk]:
                    echo(f"    • {b}")
        else:
            echo("No backends available.")
            echo("Install: qiskit-aer, amazon-braket-sdk, cirq, or pennylane")
        return

    if not ref:
        raise click.ClickException("REF argument required (bundle path or run ID)")

    if not experimental:
        raise click.ClickException(
            "Replay is EXPERIMENTAL and may not be fully reproducible.\n"
            "Use --experimental flag to acknowledge this."
        )

    root = root_from_ctx(ctx)

    result = replay(
        ref,
        backend=backend,
        root=root,
        shots=shots,
        seed=seed,
        save_run=save,
        project=project,
        ack_experimental=True,
    )

    if fmt == "json":
        echo(json.dumps(result.to_dict(), indent=2, default=str))
        ctx.exit(0 if result.ok else 1)
        return

    # Text format
    lines = [
        "=" * 70,
        "REPLAY RESULT (EXPERIMENTAL)",
        "=" * 70,
        f"Original run:     {result.original_run_id}",
        f"Original adapter: {result.original_adapter}",
        f"Original backend: {result.original_backend}",
        f"Replay backend:   {result.backend_used} (simulator)",
        f"Circuit source:   {result.circuit_source}",
        f"Shots:            {result.shots}",
    ]

    if result.seed is not None:
        lines.append(f"Seed:             {result.seed} (best-effort)")

    lines.extend(
        [
            "",
            f"Result: {'✓ OK' if result.ok else '✗ FAILED'}",
            f"  {result.message}",
        ]
    )

    if result.replay_run_id:
        lines.append(f"\nReplay saved as: {result.replay_run_id}")
        lines.append(
            f"Compare with: devqubit diff {result.original_run_id} {result.replay_run_id}"
        )

    if result.errors:
        lines.extend(["", "Warnings:"])
        for err in result.errors:
            lines.append(f"  ⚠ {err}")

    lines.extend(["", "=" * 70])

    echo("\n".join(lines))
    ctx.exit(0 if result.ok else 1)
