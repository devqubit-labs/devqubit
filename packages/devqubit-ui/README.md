# devqubit-ui

[![PyPI](https://img.shields.io/pypi/v/devqubit-ui)](https://pypi.org/project/devqubit-ui/)

Web UI for [devqubit](https://github.com/devqubit-labs/devqubit) — browse runs, view artifacts, compare experiments, and manage baselines in a local web interface.

> [!IMPORTANT]
> **This is an internal UI package.** Install via `pip install "devqubit[ui]"` and launch with `devqubit ui`.

## Installation

```bash
pip install "devqubit[ui]"
```

## Usage

### CLI

```bash
devqubit ui                           # http://127.0.0.1:8080
devqubit ui --port 9000               # custom port
devqubit ui --workspace /path/to/.devqubit  # custom workspace
```

### Python

```python
from devqubit.ui import run_server

run_server(port=8080)
```

### Production

```bash
uvicorn devqubit_ui.app:create_app --factory --host 0.0.0.0 --port 8080
```

## Features

- **Run browser** — list, filter, and search runs across projects
- **Run details** — parameters, metrics, tags, artifacts, and fingerprints
- **Artifact viewer** — inline JSON/text viewing; download for binary formats
- **Diff view** — side-by-side run comparison with TVD analysis
- **Projects & groups** — navigate by project, group, or sweep
- **Baseline management** — set, view, and clear project baselines
- **REST API** — JSON endpoints at `/api/*` for programmatic access

## Frontend Development

```bash
cd frontend
npm install
npm run dev       # dev server with HMR (proxies /api => :8000)
npm run build     # production build => auto-copied to src/devqubit_ui/static/
```

## License

Apache 2.0
