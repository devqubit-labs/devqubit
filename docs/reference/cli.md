# CLI Reference

The `devqubit` CLI helps you inspect runs, compare results, manage baselines, and export portable bundles.

If you're new, start with {doc}`../getting-started/quickstart`.

> **Run identifiers:** Most commands accept `<run>` which can be either a **run ID** (ULID like `01JD7X...`) or a **run name** (like `baseline-v1`). When using names, provide `--project` for disambiguation.

## Quick Start

```bash
# List recent runs
devqubit list

# Show run details (by name or ID)
devqubit show baseline-v1 --project myproject

# Compare two runs
devqubit diff baseline-v1 candidate-v2 --project myproject

# Verify against baseline
devqubit verify nightly-run --project myproject

# Launch web UI
devqubit ui
```

---

## Run Management

### list

List recent runs with optional filters.

```bash
devqubit list [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Number of runs to show (default: 20) |
| `--project` | `-p` | Filter by project name |
| `--adapter` | `-a` | Filter by adapter (qiskit, braket, cirq, pennylane) |
| `--status` | `-s` | Filter by status (FINISHED, FAILED, RUNNING, KILLED) |
| `--backend` | `-b` | Filter by backend name |
| `--group` | `-g` | Filter by group ID |
| `--tag` | `-t` | Filter by tag (repeatable) |
| `--format` | | Output format: `table` (default) or `json` |

**Examples:**

```bash
# List last 50 runs
devqubit list --limit 50

# Filter by project and status
devqubit list --project bell-state --status FINISHED

# Filter by backend
devqubit list --backend ibm_brisbane

# Filter by tags (can combine multiple)
devqubit list --tag experiment=bell --tag validated

# Output as JSON for scripting
devqubit list --format json
```

---

### search

Search runs using query expressions with field operators.

```bash
devqubit search QUERY [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Max results (default: 20) |
| `--project` | `-p` | Filter by project first |
| `--sort` | `-s` | Sort by field (e.g., `metric.fidelity`) |
| `--asc` | | Sort ascending (default: descending) |
| `--format` | | Output format: `table` or `json` |

**Query Syntax:**

```
field operator value [and field operator value ...]
```

**Operators:**

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equals | `status = FINISHED` |
| `!=` | Not equals | `status != FAILED` |
| `>` | Greater than | `metric.fidelity > 0.9` |
| `>=` | Greater or equal | `params.shots >= 1000` |
| `<` | Less than | `metric.tvd < 0.05` |
| `<=` | Less or equal | `params.depth <= 10` |
| `~` | Contains (string) | `backend ~ ibm` |

**Queryable Fields:**

| Field | Description |
|-------|-------------|
| `params.<n>` | Run parameters |
| `metric.<n>` | Logged metrics |
| `tags.<n>` | Run tags |
| `status` | Run status |
| `project` | Project name |
| `adapter` | SDK adapter |
| `backend` | Backend name |

**Examples:**

```bash
# Find high-fidelity runs
devqubit search "metric.fidelity > 0.95"

# Combined conditions (AND only)
devqubit search "params.shots >= 1000 and metric.fidelity > 0.9"

# Sort by metric
devqubit search "status = FINISHED" --sort metric.fidelity

# Find runs on IBM backends
devqubit search "backend ~ ibm and status = FINISHED"
```

---

### show

Display detailed information about a run.

```bash
devqubit show <run> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--project` | Project name (required when using run name) |
| `--format` | Output format: `pretty` (default) or `json` |

**Output includes:**
- Run ID, project, adapter, status
- Created/ended timestamps
- Group and parent run (if applicable)
- Backend and provider info
- Fingerprint for reproducibility
- Git provenance (branch, commit, dirty state)
- Parameter and metric summaries
- Artifact count

**Examples:**

```bash
# By run name
devqubit show baseline-v1 --project bell-state

# By run ID
devqubit show 01JD7X...

# Full JSON for programmatic access
devqubit show baseline-v1 --project bell-state --format json
```

---

### delete

Delete a run from the workspace.

```bash
devqubit delete <run> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |
| `--yes` | `-y` | Skip confirmation prompt |

**Example:**

```bash
# Interactive confirmation
devqubit delete old-experiment --project myproj

# Non-interactive (for scripts)
devqubit delete 01JD7X... --yes
```

---

### projects

List all projects in the workspace.

```bash
devqubit projects [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `table` (default) or `json` |

Shows project name, run count, and baseline status.

---

## Run Groups

Groups organize related runs (parameter sweeps, experiments).

### groups list

List run groups.

```bash
devqubit groups list [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Filter by project |
| `--limit` | `-n` | Max results (default: 20) |
| `--format` | | Output format: `table` or `json` |

---

### groups show

Show runs within a group.

```bash
devqubit groups show GROUP_ID [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Max results (default: 50) |
| `--format` | | Output format: `table` or `json` |

---

## Artifacts

### artifacts list

List artifacts in a run.

```bash
devqubit artifacts list <run> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |
| `--role` | `-r` | Filter by role (program, result, device_raw, envelope) |
| `--kind` | `-k` | Filter by kind substring |
| `--format` | | Output format: `table` (default) or `json` |

**Example:**

```bash
# List all artifacts
devqubit artifacts list baseline-v1 --project bell-state

# Filter by role
devqubit artifacts list baseline-v1 --project bell-state --role program

# Filter by kind
devqubit artifacts list baseline-v1 --project bell-state --kind openqasm
```

---

### artifacts show

Display artifact content.

```bash
devqubit artifacts show <run> SELECTOR [OPTIONS]
```

**Selector formats:**
- Index number: `0`, `1`, `2`
- Kind substring: `counts`, `openqasm3`
- Role:kind pattern: `program:openqasm3`, `result:counts`

**Options:**

| Option | Description |
|--------|-------------|
| `--project` | Project name (required when using run name) |
| `--raw` | Output raw bytes to stdout (for piping) |
| `--format` | Output format: `pretty` (default) or `json` |

**Examples:**

```bash
# Show by index
devqubit artifacts show baseline-v1 0 --project bell-state

# Show by kind
devqubit artifacts show baseline-v1 counts --project bell-state

# Show by role:kind
devqubit artifacts show baseline-v1 program:openqasm3 --project bell-state

# Export raw content
devqubit artifacts show baseline-v1 results --project bell-state --raw > output.json
```

---

### artifacts counts

Display measurement counts from a run.

```bash
devqubit artifacts counts <run> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |
| `--top` | `-k` | Show top K outcomes (default: 10) |
| `--experiment` | `-e` | Experiment index for batch jobs |
| `--format` | | Output format: `table` (default) or `json` |

**Example:**

```bash
# Show top 5 outcomes
devqubit artifacts counts baseline-v1 --project bell-state --top 5

# Show counts for second circuit in batch
devqubit artifacts counts baseline-v1 --project bell-state --experiment 1
```

---

## Tags

Tags are key-value pairs for organizing and filtering runs.

### tag add

Add tags to a run.

```bash
devqubit tag add <run> TAG [TAG ...] [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |

Tags can be `key=value` pairs or just `key` (value defaults to "true").

**Examples:**

```bash
# Add key=value tag
devqubit tag add baseline-v1 experiment=bell --project bell-state

# Add multiple tags
devqubit tag add baseline-v1 validated production device=ibm_brisbane --project bell-state
```

---

### tag remove

Remove tags from a run.

```bash
devqubit tag remove <run> KEY [KEY ...] [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |

**Example:**

```bash
devqubit tag remove baseline-v1 temp debug --project bell-state
```

---

### tag list

List all tags on a run.

```bash
devqubit tag list <run> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--project` | Project name (required when using run name) |
| `--format` | Output format: `pretty` (default) or `json` |

---

## Comparison & Verification

### diff

Compare two runs or bundles comprehensively.

```bash
devqubit diff <run_a> <run_b> [OPTIONS]
```

`<run_a>` and `<run_b>` can be run names, run IDs, or bundle file paths.

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run names) |
| `--output` | `-o` | Save report to file |
| `--format` | | Output: `pretty` (default), `json`, or `summary` |
| `--no-circuit-diff` | | Skip circuit semantic comparison |
| `--no-noise-context` | | Skip bootstrap noise estimation (faster) |
| `--item-index` | | Result item index for TVD (default: 0, use -1 for all) |

**Comparison includes:**
- Parameter differences
- Circuit/program changes (structural and parametric hashes)
- Device drift analysis (calibration deltas)
- Total Variation Distance (TVD) with bootstrap-calibrated noise context

**Examples:**

```bash
# Compare two runs by name
devqubit diff baseline-v1 candidate-v2 --project bell-state

# Compare two runs by ID
devqubit diff 01JD7X... 01JD8Y...

# Compare bundles
devqubit diff experiment1.zip experiment2.zip

# Save report as JSON
devqubit diff baseline-v1 candidate-v2 --project bell-state --format json -o report.json

# Quick comparison (skip noise estimation)
devqubit diff baseline-v1 candidate-v2 --project bell-state --no-noise-context

# Compare all result items in batch
devqubit diff baseline-v1 candidate-v2 --project bell-state --item-index -1
```

---

### verify

Verify a run against baseline with full root-cause analysis.

```bash
devqubit verify <run> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--baseline` | `-b` | Baseline run (name or ID; default: project baseline) |
| `--project` | `-p` | Project for baseline lookup (required when using run name) |
| `--tvd-max` | | Maximum allowed TVD |
| `--noise-factor` | | Fail if TVD > noise_factor × noise_p95 (recommended: 1.0-1.5) |
| `--program-match-mode` | | `exact`, `structural`, or `either` (default) |
| `--no-params-match` | | Don't require parameters to match |
| `--no-program-match` | | Don't require program to match |
| `--strict` | | Require fingerprint match |
| `--promote` | | Promote to baseline on pass |
| `--allow-missing` | | Pass if no baseline exists |
| `--junit` | | Write JUnit XML report |
| `--format` | | Output: `pretty` (default), `json`, `github`, or `summary` |

**Program Match Modes:**
- `exact`: require identical artifact digests
- `structural`: require same circuit structure (VQE/QAOA friendly)
- `either`: pass if exact OR structural matches (default)

**Examples:**

```bash
# Verify by run name against project baseline
devqubit verify nightly-run --project bell-state

# Verify against explicit baseline (by name)
devqubit verify candidate-v2 --baseline baseline-v1 --project bell-state

# With TVD threshold
devqubit verify nightly-run --project bell-state --tvd-max 0.05

# Bootstrap-calibrated threshold (recommended)
devqubit verify nightly-run --project bell-state --noise-factor 1.0

# VQE-friendly (ignore parameter values in circuit)
devqubit verify nightly-run --project vqe-h2 --program-match-mode structural

# Strict mode (fingerprint must match)
devqubit verify nightly-run --project bell-state --strict

# CI integration with JUnit output
devqubit verify nightly-run --project bell-state --junit results.xml

# GitHub Actions format
devqubit verify nightly-run --project bell-state --format github

# Promote on success
devqubit verify nightly-run --project bell-state --promote

# Relaxed verification (only check TVD)
devqubit verify nightly-run --project bell-state \
  --no-params-match --no-program-match --noise-factor 1.5
```

---

### replay

Re-execute a quantum circuit from a run or bundle on a simulator.

**⚠️ EXPERIMENTAL:** Replay is best-effort and may not be fully reproducible.

```bash
devqubit replay <run> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |
| `--backend` | `-b` | Simulator backend name |
| `--shots` | `-s` | Override shot count |
| `--seed` | | Random seed for reproducibility (best-effort) |
| `--save` | | Save replay as new tracked run |
| `--experimental` | | Acknowledge experimental status (required) |
| `--list-backends` | | List available simulator backends |
| `--format` | | Output format: `pretty` or `json` |

**Supported Formats:**

Only native SDK formats are supported to ensure exact program representation:
- QPY (Qiskit)
- JAQCD (Braket)
- Cirq JSON
- Tape JSON (PennyLane)

**Note:** OpenQASM is NOT supported for replay.

**Examples:**

```bash
# List available backends
devqubit replay --list-backends

# Replay on default simulator (requires --experimental)
devqubit replay experiment.zip --experimental

# Specify backend, shots, and seed
devqubit replay baseline-v1 --project bell-state --backend aer_simulator --shots 10000 --seed 42 --experimental

# Save and compare with original
devqubit replay baseline-v1 --project bell-state --experimental --save --project replay-test
devqubit diff baseline-v1 <replay_run_id> --project bell-state
```

---

## Bundles

Bundles are portable ZIP archives containing a run and all its artifacts.

### pack

Create a bundle from a run.

```bash
devqubit pack <run> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | Project name (required when using run name) |
| `--out` | `-o` | Output file path (default: `<run_id>.devqubit.zip`) |
| `--force` | `-f` | Overwrite existing file |
| `--format` | | Output format: `pretty` (default) or `json` |

**Examples:**

```bash
# Pack by name
devqubit pack baseline-v1 --project bell-state

# Specify output path
devqubit pack baseline-v1 --project bell-state -o experiment.zip

# Pack by ID
devqubit pack 01JD7X... -o experiment.zip

# Overwrite existing
devqubit pack baseline-v1 --project bell-state -o experiment.zip --force
```

---

### unpack

Extract a bundle into a workspace.

```bash
devqubit unpack BUNDLE [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--to` | `-t` | Destination workspace |
| `--force` | `-f` | Overwrite existing run |
| `--verify/--no-verify` | | Verify digests (default: verify) |
| `--format` | | Output format: `pretty` (default) or `json` |

**Examples:**

```bash
# Unpack to current workspace
devqubit unpack experiment.zip

# Unpack to specific workspace
devqubit unpack experiment.zip --to /path/to/workspace

# Skip verification (faster)
devqubit unpack experiment.zip --no-verify
```

---

### info

Show bundle metadata without extracting.

```bash
devqubit info BUNDLE [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `pretty` (default) or `json` |
| `--objects` | List all object digests |
| `--artifacts` | List artifact details |

**Examples:**

```bash
# Show basic info
devqubit info experiment.zip

# Show with artifact list
devqubit info experiment.zip --artifacts

# Full details as JSON
devqubit info experiment.zip --format json --objects --artifacts
```

---

## Baselines

Baselines are reference runs for verification.

### baseline set

Set the baseline run for a project.

```bash
devqubit baseline set PROJECT <run> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--project` | Project context (required when using run name and it differs from PROJECT) |

**Examples:**

```bash
# Set baseline by run name
devqubit baseline set bell-state baseline-v1

# Set baseline by run ID
devqubit baseline set bell-state 01JD7X...
```

---

### baseline get

Get the current baseline for a project.

```bash
devqubit baseline get PROJECT [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `pretty` (default) or `json` |

---

### baseline clear

Clear the baseline for a project.

```bash
devqubit baseline clear PROJECT [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--yes` | `-y` | Skip confirmation |

---

### baseline list

List all project baselines.

```bash
devqubit baseline list [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `pretty` (default) or `json` |

---

## Storage Management

### storage gc

Garbage collect unreferenced objects.

```bash
devqubit storage gc [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--dry-run` | `-n` | Preview without deleting |
| `--yes` | `-y` | Skip confirmation |
| `--project` | `-p` | Limit to specific project |
| `--format` | | Output format: `pretty` (default) or `json` |

**Examples:**

```bash
# Preview what would be deleted
devqubit storage gc --dry-run

# Delete orphaned objects
devqubit storage gc --yes

# Limit to specific project
devqubit storage gc --project myproject --dry-run
```

---

### storage prune

Delete old runs by status.

```bash
devqubit storage prune [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--status` | `-s` | FAILED | Status to prune |
| `--older-than` | | 30 | Days old threshold |
| `--keep-latest` | | 5 | Keep N most recent matching runs |
| `--project` | `-p` | | Limit to specific project |
| `--dry-run` | `-n` | | Preview without deleting |
| `--yes` | `-y` | | Skip confirmation |
| `--format` | | | Output format: `pretty` (default) or `json` |

**Examples:**

```bash
# Preview pruning failed runs
devqubit storage prune --status FAILED --dry-run

# Prune runs older than 7 days, keep latest 3
devqubit storage prune --older-than 7 --keep-latest 3 --yes

# Prune only in specific project
devqubit storage prune --project old-experiments --yes
```

---

### storage health

Check workspace health and integrity.

```bash
devqubit storage health [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `pretty` (default) or `json` |

Reports:
- Total runs and objects
- Referenced vs orphaned objects
- Missing objects (integrity issues)

---

## Configuration

### config

Display current configuration.

```bash
devqubit config [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `pretty` (default) or `json` |

Shows:
- Workspace path
- Storage and registry URLs
- Capture settings (pip, git)
- Validation and redaction settings

---

## Web UI

### ui

Launch the local web interface.

```bash
devqubit ui [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | | 127.0.0.1 | Host to bind to |
| `--port` | `-p` | 8080 | Port to listen on |
| `--workspace` | `-w` | | Workspace directory |
| `--debug` | | | Enable debug mode |

**Note:** Requires the `devqubit-ui` package (`pip install devqubit[ui]`).

**Examples:**

```bash
# Start on default port
devqubit ui

# Custom port
devqubit ui --port 9000

# Bind to all interfaces (for remote access)
devqubit ui --host 0.0.0.0

# Specify workspace
devqubit ui --workspace /path/to/.devqubit
```

---

## Global Options

These options apply to all commands:

```bash
devqubit --root /path/to/.devqubit <command>
```

| Option | Short | Description |
|--------|-------|-------------|
| `--root` | `-r` | Workspace root directory (default: ~/.devqubit) |
| `--quiet` | `-q` | Less output |
| `--version` | | Show version |
| `--help` | | Show help |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEVQUBIT_HOME` | Default workspace root path |
| `DEVQUBIT_PROJECT` | Default project name |
| `DEVQUBIT_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (or verification passed) |
| 1 | Failure (verification failed, run not found, etc.) |
| 2 | Error (invalid arguments, I/O error) |

---

## Common Workflows

### CI/CD Verification

```bash
# Run experiment (in your code)
# ...

# Verify against baseline with bootstrap-calibrated threshold
devqubit verify nightly-run \
  --project $PROJECT \
  --program-match-mode either \
  --noise-factor 1.0 \
  --junit results.xml

# Exit code determines CI pass/fail
```

### Sharing Experiments

```bash
# Pack run for sharing (by name)
devqubit pack baseline-v1 --project bell-state -o experiment.zip

# Recipient unpacks
devqubit unpack experiment.zip

# Recipient can view, replay, or compare
devqubit show 01JD7X...  # use run ID from bundle
devqubit replay 01JD7X... --backend aer_simulator --experimental
```

### Workspace Maintenance

```bash
# Check health
devqubit storage health

# Clean up orphaned objects
devqubit storage gc

# Prune old failed runs
devqubit storage prune --status FAILED --older-than 30
```
