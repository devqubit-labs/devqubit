/**
 * DevQubit UI Type Definitions
 *
 * Core TypeScript interfaces for the devqubit UI package.
 * These types mirror the Python backend models.
 */

/** Run status enum values */
export type RunStatus = 'RUNNING' | 'FINISHED' | 'FAILED' | 'UNKNOWN';

/** Artifact metadata */
export interface Artifact {
  kind: string;
  role: string;
  media_type: string;
  digest: string;
}

/** Run fingerprints for content-based identification */
export interface Fingerprints {
  run?: string;
  program?: string;
}

/** Run summary for list views */
export interface RunSummary {
  run_id: string;
  run_name?: string;
  project: string;
  adapter?: string;
  status: RunStatus;
  created_at: string;
  fingerprints?: Fingerprints;
}

/** Full run record with all details */
export interface RunRecord extends RunSummary {
  data?: {
    params?: Record<string, unknown>;
    metrics?: Record<string, number>;
    tags?: Record<string, string>;
  };
  artifacts: Artifact[];
  group_id?: string;
  group_name?: string;
  backend?: {
    name?: string;
    [key: string]: unknown;
  };
  errors?: Array<{
    type: string;
    message: string;
    traceback?: string;
  }>;
}

/** Project summary with run count and baseline */
export interface Project {
  name: string;
  description?: string;
  run_count: number;
  baseline?: {
    run_id: string;
    run_name?: string;
  };
}

/** Group summary */
export interface Group {
  group_id: string;
  group_name?: string;
  project: string;
  run_count: number;
}

/** Filter state for runs list */
export interface RunFilters {
  project?: string;
  status?: RunStatus | '';
  q?: string;
  limit: number;
}

/** Filter state for groups list */
export interface GroupFilters {
  project?: string;
}

/** Server capabilities response */
export interface Capabilities {
  mode: 'local' | 'hub';
  version: string;
  features: {
    auth: boolean;
    workspaces: boolean;
    rbac: boolean;
    service_accounts: boolean;
  };
}

/** Diff report comparing two runs */
export interface DiffReport {
  identical: boolean;
  metadata: {
    project_match: boolean;
    backend_match: boolean;
    project_a?: string;
    project_b?: string;
    backend_a?: string;
    backend_b?: string;
  };
  fingerprints: {
    a?: string;
    b?: string;
  };
  program: {
    exact_match: boolean;
    structural_match: boolean;
    structural_only_match?: boolean;
    circuit_hash_a?: string;
    circuit_hash_b?: string;
  };
  device_drift?: {
    significant_drift: boolean;
    has_calibration_data: boolean;
    top_drifts?: Array<{
      metric: string;
      percent_change?: number;
    }>;
  };
  params: {
    match: boolean;
    changed?: Record<string, { a: unknown; b: unknown }>;
    added?: Record<string, unknown>;
    removed?: Record<string, unknown>;
  };
  metrics: {
    match: boolean;
    changed?: Record<string, { a: number; b: number }>;
    added?: Record<string, number>;
    removed?: Record<string, number>;
  };
  circuit_diff?: {
    match: boolean;
    changed?: Record<string, { a: unknown; b: unknown; label?: string; delta?: number; pct?: number }>;
    is_clifford_changed?: boolean;
    is_clifford_a?: boolean;
    is_clifford_b?: boolean;
    added_gates?: string[];
    removed_gates?: string[];
  };
  tvd?: number;
  shots?: {
    a: number;
    b: number;
  };
  noise_context?: {
    expected_noise?: number;
    noise_p95?: number;
    noise_ratio?: number;
    p_value?: number;
  };
  warnings?: string[];
}

/** Workspace context (for hub extension) */
export interface Workspace {
  id: string;
  name: string;
}

/** Navigation link configuration */
export interface NavLink {
  href: string;
  label: string;
  matchPaths?: string[];
}

/** Layout configuration for extensibility */
export interface LayoutConfig {
  /** Custom logo (replaces default) */
  logo?: {
    text: string;
    icon?: string;
  };
  /** Replace all nav links (overrides default) */
  navLinks?: NavLink[];
  /** Add links before default nav */
  prependNavLinks?: NavLink[];
  /** Add links after default nav */
  appendNavLinks?: NavLink[];
  /** Content for header right side (workspace selector, user menu, etc.) */
  headerRight?: React.ReactNode;
}
