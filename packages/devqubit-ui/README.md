# devqubit-ui

Web UI for devqubit experiment tracking. Browse runs, view artifacts, and compare experiments in a local web interface.

## Installation

```bash
pip install devqubit-ui
```

Or via the meta-package:

```bash
pip install devqubit[ui]
```

## Usage

### From CLI

```bash
devqubit ui
devqubit ui --port 9000
devqubit ui --workspace /path/to/.devqubit
```

### From Python

```python
from devqubit import run_server

run_server(port=8080)
```

## Features

- **Run browser** — List, filter, and search runs
- **Run details** — Parameters, metrics, tags, artifacts
- **Artifact viewer** — View JSON/text; large files download-only
- **Diff view** — Compare runs side-by-side with TVD analysis
- **Projects & groups** — Organize and navigate experiments
- **REST API** — JSON endpoints at `/api/*` for programmatic access

## Development

### Frontend Development

```bash
cd frontend
npm install
npm run dev       # Start dev server with hot reload (proxies /api to :8000)
```

### Building

```bash
cd frontend
npm run build     # Build app → automatically copies to src/devqubit_ui/static/
npm run build:lib # Build library for npm publishing
```

The `npm run build` command:
1. Compiles TypeScript
2. Bundles the React app with Vite
3. **Automatically copies** the production build to `src/devqubit_ui/static/`

This means after running `npm run build`, the Python package is ready to serve the latest frontend.

### Type Checking & Linting

```bash
npm run typecheck  # TypeScript type checking
npm run lint       # ESLint
```

## Production

```bash
uvicorn devqubit_ui.app:create_app --factory --host 0.0.0.0 --port 8080
```

## Architecture

```
devqubit-ui/
├── frontend/           # React/TypeScript frontend
│   ├── src/
│   │   ├── components/ # Reusable UI components
│   │   ├── pages/      # Page components
│   │   ├── hooks/      # React hooks (data fetching, state)
│   │   ├── api/        # API client
│   │   ├── styles/     # Global CSS + Tailwind
│   │   └── types/      # TypeScript definitions
│   └── package.json
├── src/devqubit_ui/    # Python backend
│   ├── app.py          # FastAPI application
│   ├── routers/        # API routes
│   ├── static/         # Built frontend (auto-generated)
│   └── _plugin.py      # devqubit plugin entry point
└── pyproject.toml
```

## License

Apache 2.0
