/**
 * @devqubit/ui - React UI for DevQubit
 *
 * Main entry point for the npm package.
 */

import './styles/globals.css';

// App & Providers
export { App } from './App';
export type { AppProps } from './App';

export { AppProvider, useApp } from './hooks';
export type { AppProviderProps, AppContextValue, AsyncState, PollingConfig } from './hooks';

// Router
export { router, createRouter, coreRoutes } from './router';

// API Client
export { ApiClient, ApiError, api } from './api';
export type { ApiConfig, RequestOptions } from './api';

// Hooks
export {
  useRuns,
  useRun,
  useProjects,
  useGroups,
  useGroup,
  useDiff,
  useArtifact,
  useMutation,
  usePolling,
  POLL_INTERVAL,
  ThemeProvider,
  useTheme,
  useThemeOptional,
} from './hooks';
export type { Theme, ThemeContextValue, ThemeProviderProps } from './hooks';

// Layout
export { Layout, PageHeader, LayoutConfigProvider, useLayoutConfig } from './components/Layout';
export type { LayoutProps, PageHeaderProps } from './components/Layout';

// UI Primitives
export {
  Badge,
  Button,
  Card,
  CardHeader,
  CardTitle,
  Alert,
  Spinner,
  EmptyState,
  KVList,
  Modal,
  Toast,
  FormGroup,
  FormRow,
  Label,
  Input,
  Select,
} from './components/ui';

export type {
  BadgeVariant,
  BadgeProps,
  ButtonVariant,
  ButtonSize,
  ButtonProps,
  CardProps,
  CardHeaderProps,
  CardTitleProps,
  AlertVariant,
  AlertProps,
  SpinnerProps,
  EmptyStateProps,
  KVListProps,
  ModalProps,
  ToastVariant,
  ToastProps,
  FormGroupProps,
  FormRowProps,
  LabelProps,
  InputProps,
  SelectProps,
} from './components/ui';

// Table
export {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableHeader,
  TableCell,
} from './components/Table';

export type {
  TableProps,
  TableHeadProps,
  TableBodyProps,
  TableRowProps,
  TableHeaderProps,
  TableCellProps,
} from './components/Table';

// Domain Components
export { RunsTable, StatusBadge } from './components/RunsTable';
export type { RunsTableProps } from './components/RunsTable';

export { ProjectsTable } from './components/ProjectsTable';
export type { ProjectsTableProps } from './components/ProjectsTable';

export { GroupsTable } from './components/GroupsTable';
export type { GroupsTableProps } from './components/GroupsTable';

// Export Run
export { ExportRunButton } from './components/ExportRunButton';
export type { ExportRunButtonProps, ExportProgress } from './components/ExportRunButton';

// Pages
export {
  RunsPage,
  RunDetailPage,
  ProjectsPage,
  GroupsPage,
  GroupDetailPage,
  DiffPage,
  SearchPage,
  ArtifactPage,
} from './pages';

// Utilities
export {
  shortId,
  shortDigest,
  timeAgo,
  formatNumber,
  formatBytes,
  jsonPretty,
  truncate,
  buildUrl,
  cn,
  isTerminalStatus,
  elapsedSeconds,
  formatDuration,
} from './utils';

// Types
export type {
  RunStatus,
  Artifact,
  Fingerprints,
  RunSummary,
  RunRecord,
  Project,
  Group,
  RunFilters,
  GroupFilters,
  Capabilities,
  DiffReport,
  Workspace,
  NavLink,
  LayoutConfig,
} from './types';
