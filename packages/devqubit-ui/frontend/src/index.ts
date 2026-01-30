/**
 * @devqubit/ui - DevQubit Open-Core UI Package
 *
 * React components and utilities for the devqubit experiment tracking UI.
 * This package can be used standalone or extended by devqubit-hub.
 *
 * Usage
 * -----
 * ```tsx
 * import { App, Layout, RunsTable, api } from '@devqubit/ui';
 * import '@devqubit/ui/styles.css';
 * ```
 */

// Styles - consumers must import separately: import '@devqubit/ui/styles.css'
import './styles/globals.css';

// Main App
export { App } from './App';
export type { AppProps } from './App';

// Components
export * from './components';

// Hooks
export {
  AppProvider,
  useApp,
  useRuns,
  useRun,
  useProjects,
  useGroups,
  useGroup,
  useDiff,
  useArtifact,
  useMutation,
} from './hooks';
export type { AppProviderProps, AsyncState, AppContextValue } from './hooks';

// API
export { ApiClient, ApiError, api } from './api';
export type { ApiConfig } from './api';

// Router
export { coreRoutes, createRouter, router } from './router';

// Pages (for extension/override)
export * from './pages';

// Types
export * from './types';

// Utils
export * from './utils';
