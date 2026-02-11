# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 devqubit

"""
Comparison and verification CLI commands.

This module provides commands for comparing runs, and verifying against
baselines. Uses the compare module's formatters for consistent output
across CLI and Python API.

Commands
--------
diff
    Compare two runs or bundles comprehensively.
verify
    Verify a run against baseline with policy checks.
"""

from __future__ import annotations

from pathlib import Path

import click
from devqubit_engine.cli._utils import (
    echo,
    is_quiet,
    resolve_run,
    root_from_ctx,
)


def register(cli: click.Group) -> None:
    """Register compare commands with CLI."""
    cli.add_command(diff_cmd)
    cli.add_command(verify_cmd)


@click.command("diff")
@click.argument("ref_a")
@click.argument("ref_b")
@click.option(
    "--project",
    "-p",
    default=None,
    help="Project name (required when using run names).",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Save report to file."
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json", "summary"]),
    default="pretty",
    help="Output format.",
)
@click.option(
    "--no-circuit-diff",
    is_flag=True,
    help="Skip circuit semantic comparison.",
)
@click.option(
    "--no-noise-context",
    is_flag=True,
    help="Skip bootstrap noise estimation (faster).",
)
@click.option(
    "--item-index",
    type=int,
    default=0,
    help="Result item index for TVD (default: 0, use -1 for all).",
)
@click.pass_context
def diff_cmd(
    ctx: click.Context,
    ref_a: str,
    ref_b: str,
    project: str | None,
    output: Path | None,
    fmt: str,
    no_circuit_diff: bool,
    no_noise_context: bool,
    item_index: int,
) -> None:
    """
    Compare two runs or bundles comprehensively.

    Shows complete comparison including parameters, program, device drift,
    TVD with bootstrap-calibrated noise context analysis.

    REF_A and REF_B can be run IDs, run names (with --project), or bundle
    file paths (.zip).

    Examples:
        devqubit diff abc123 def456
        devqubit diff baseline-v1 experiment-v2 --project bell_state
        devqubit diff experiment1.zip experiment2.zip
        devqubit diff abc123 def456 --format json -o report.json
        devqubit diff abc123 def456 --no-noise-context
    """
    from devqubit_engine.compare.diff import diff
    from devqubit_engine.config import Config
    from devqubit_engine.storage.errors import RunNotFoundError
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    # Convert item_index=-1 to "all"
    item_idx: int | str = item_index if item_index >= 0 else "all"

    try:
        result = diff(
            ref_a,
            ref_b,
            registry=registry,
            store=store,
            project=project,
            include_circuit_diff=not no_circuit_diff,
            include_noise_context=not no_noise_context,
            item_index=item_idx,
        )
    except RunNotFoundError as e:
        if project:
            raise click.ClickException(
                f"Run not found: '{e.run_id}' (looked up as ID and as name in project '{project}')"
            ) from e
        raise click.ClickException(
            f"Run not found: '{e.run_id}'. Use --project to look up by name."
        ) from e
    except FileNotFoundError as e:
        raise click.ClickException(f"Bundle not found: {e}") from e
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
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(formatted, encoding="utf-8")
        echo(f"Report saved to {output}")
    else:
        echo(formatted)


@click.command("verify")
@click.argument("candidate_id_or_name")
@click.option("--baseline", "-b", help="Baseline run ID or name.")
@click.option(
    "--project", "-p", help="Project for baseline lookup and name resolution."
)
@click.option("--tvd-max", type=float, help="Maximum allowed TVD.")
@click.option(
    "--noise-factor",
    type=float,
    help="Fail if TVD > noise_factor x noise_p95. Recommended: 1.0-1.5.",
)
@click.option(
    "--program-match-mode",
    type=click.Choice(["exact", "structural", "either"]),
    default="either",
    help="Program matching mode.",
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
    "--junit",
    type=click.Path(path_type=Path),
    help="Write JUnit XML report.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["pretty", "json", "github", "summary"]),
    default="pretty",
    help="Output format.",
)
@click.pass_context
def verify_cmd(
    ctx: click.Context,
    candidate_id_or_name: str,
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

    CANDIDATE_ID_OR_NAME can be a run ID or run name. When using run name,
    --project is required.

    The --noise-factor option provides a shot-count-aware threshold using
    bootstrap-calibrated noise estimation.

    The --program-match-mode option controls how programs are compared:
    - exact: require identical artifact digests
    - structural: require same circuit structure (VQE/QAOA friendly)
    - either: pass if exact OR structural matches (default)

    Examples:
        devqubit verify abc123 --baseline def456
        devqubit verify my-experiment --project myproject --promote
        devqubit verify abc123 --tvd-max 0.05 --format json
        devqubit verify abc123 --noise-factor 1.0
    """
    from devqubit_engine.compare.ci import result_to_github_annotations, write_junit
    from devqubit_engine.compare.types import ProgramMatchMode
    from devqubit_engine.compare.verify import (
        VerifyPolicy,
        verify,
        verify_against_baseline,
    )
    from devqubit_engine.config import Config
    from devqubit_engine.storage.factory import create_registry, create_store

    root = root_from_ctx(ctx)
    config = Config(root_dir=root)
    registry = create_registry(config=config)
    store = create_store(config=config)

    # Resolve candidate run (supports ID or name with project)
    candidate = resolve_run(candidate_id_or_name, registry, project)

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
        # Resolve baseline run (supports ID or name with project)
        baseline_record = resolve_run(baseline, registry, project)

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
                "No project specified and candidate has no project. "
                "Use --project or --baseline."
            )

        try:
            result = verify_against_baseline(
                candidate,
                project=proj,
                store=store,
                registry=registry,
                policy=policy,
                promote_on_pass=promote,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    # Write JUnit if requested
    if junit:
        junit.parent.mkdir(parents=True, exist_ok=True)
        write_junit(result, junit)
        if not is_quiet(ctx):
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
        echo(f"\n[OK] Promoted {candidate.run_id} to baseline for project")

    ctx.exit(0 if result.ok else 1)
