# Configuration

This page explains how to configure devqubit via **environment variables** and programmatic config objects.

For workspace layout and storage concepts, see {doc}`../concepts/workspace`.


devqubit uses environment variables for configuration with sensible defaults.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVQUBIT_HOME` | Workspace root directory | `~/.devqubit` |
| `DEVQUBIT_STORAGE_URL` | Object store URL | `file://{DEVQUBIT_HOME}/objects` |
| `DEVQUBIT_REGISTRY_URL` | Registry URL | `file://{DEVQUBIT_HOME}` |
| `DEVQUBIT_CAPTURE_GIT` | Capture git provenance | `true` |
| `DEVQUBIT_CAPTURE_PIP` | Capture pip freeze output | `true` |
| `DEVQUBIT_VALIDATE` | Validate run records against schema | `true` |
| `DEVQUBIT_REDACT_DISABLE` | Disable credential redaction | `false` |
| `DEVQUBIT_REDACT_PATTERNS` | Additional redaction patterns (comma-separated) | — |

Boolean values accept: `1`, `true`, `yes`, `on` (case-insensitive).

## Credential Redaction

By default, devqubit redacts sensitive environment variables when capturing the environment. Variables matching these patterns are replaced with `[REDACTED]`:

- `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, `CREDENTIAL`, `PRIVATE`
- Variables starting with: `AWS_`, `AZURE_`, `GCP_`, `GOOGLE_`, `IBM_`, `IONQ_`, `BRAKET_`

Add custom patterns:

```bash
export DEVQUBIT_REDACT_PATTERNS="MY_SECRET_.*,CUSTOM_TOKEN"
```

Disable redaction entirely (not recommended):

```bash
export DEVQUBIT_REDACT_DISABLE=true
```

## Programmatic Configuration

```python
from pathlib import Path
from devqubit import track, Config, set_config

# Create custom config
config = Config(
    root_dir=Path("/path/to/workspace"),
    capture_git=True,
    capture_pip=False,
)

# Option 1: Set globally
set_config(config)
with track(project="my_project") as run:
    ...

# Option 2: Pass directly to track()
with track(project="my_project", config=config) as run:
    ...
```

## Storage URLs

devqubit supports file-based storage:

```bash
# Absolute path
export DEVQUBIT_STORAGE_URL="file:///data/devqubit/objects"
export DEVQUBIT_REGISTRY_URL="file:///data/devqubit"

# Home directory expansion
export DEVQUBIT_HOME="~/projects/devqubit-data"
```

## Multiple Workspaces

Use separate workspaces for different environments:

```bash
# Development
export DEVQUBIT_HOME=~/.devqubit-dev

# Production
export DEVQUBIT_HOME=~/.devqubit-prod

# Per-command override
DEVQUBIT_HOME=/tmp/test-workspace python experiment.py
```

## CI/CD Configuration

Example GitHub Actions setup:

```yaml
env:
  DEVQUBIT_HOME: ${{ github.workspace }}/.devqubit
  DEVQUBIT_CAPTURE_GIT: "true"
  DEVQUBIT_CAPTURE_PIP: "true"

jobs:
  experiment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run experiment
        run: python run_experiment.py

      - name: Verify against baseline
        run: devqubit verify --project $PROJECT_NAME $RUN_ID

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: devqubit-workspace
          path: ${{ env.DEVQUBIT_HOME }}
```

## Logging

Configure logging via standard Python logging:

```python
import logging

# Enable debug logging for devqubit
logging.getLogger("devqubit_engine").setLevel(logging.DEBUG)

# Or for specific modules
logging.getLogger("devqubit_engine.tracking.run").setLevel(logging.DEBUG)
```

Log levels:
- `DEBUG` — Detailed operation logs, fingerprint computation
- `INFO` — Run lifecycle events (default)
- `WARNING` — Missing optional data, fallbacks used
- `ERROR` — Operation failures
