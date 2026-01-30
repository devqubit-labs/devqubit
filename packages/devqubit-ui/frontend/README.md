# @devqubit/ui

React frontend for devqubit experiment tracking UI.

## Tech Stack

- React 18 + TypeScript
- Vite (build & dev server)
- TailwindCSS (styling)
- React Router (client-side routing)

## Development

```bash
npm install
npm run dev      # Start dev server (localhost:5173)
npm run build    # Build for production
npm run lint     # Run ESLint
```

Dev server proxies `/api` requests to `localhost:8000`.

## Package Exports

```tsx
import {
  App,
  Layout,
  LayoutConfigProvider,
  ApiClient,
  coreRoutes,
  useRuns,
  RunsTable,
} from '@devqubit/ui';
import '@devqubit/ui/styles.css';
```

## Build for Python Package

```bash
npm run build
cp -r dist/* ../python/src/devqubit_ui/static/
```
