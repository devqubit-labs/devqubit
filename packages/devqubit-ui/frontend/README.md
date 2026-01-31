# @devqubit/ui

React frontend for devqubit experiment tracking UI (open-core).

## Tech Stack

- React 18 + TypeScript
- Vite (build & dev server)
- TailwindCSS (styling)
- React Router (client-side routing)

## Development

```bash
npm install
npm run dev      # Start dev server (localhost:5173)
npm run build    # Build for production (auto-copies to static/)
npm run build:lib # Build as npm library
npm run lint     # Run ESLint
npm run typecheck # TypeScript type check
```

Dev server proxies `/api` requests to `localhost:8000`.

## Build for Python Package

The build process automatically copies production files to `../src/devqubit_ui/static/`:

```bash
npm run build
```
