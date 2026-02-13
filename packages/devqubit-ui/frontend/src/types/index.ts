/**
 * DevQubit UI Type Definitions
 */

export type RunStatus = 'RUNNING' | 'FINISHED' | 'FAILED' | 'KILLED' | 'UNKNOWN';

export interface Artifact {
  kind: string;
  role: string;
  media_type: string;
  digest: string;
}

export interface Fingerprints {
  run?: string;
  program?: string;
}

export interface RunSummary {
  run_id: string;
  run_name?: string;
  project: string;
  adapter?: string;
  status: RunStatus;
  created_at: string;
  ended_at?: string | null;
  fingerprints?: Fingerprints;
}

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

export interface Project {
  name: string;
  description?: string;
  run_count: number;
  baseline?: {
    run_id: string;
    run_name?: string;
  };
}

export interface Group {
  group_id: string;
  group_name?: string;
  project: string;
  run_count: number;
}

export interface RunFilters {
  project?: string;
  status?: RunStatus | '';
  q?: string;
  limit: number;
}

export interface GroupFilters {
  project?: string;
}

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

export interface Workspace {
  id: string;
  name: string;
}

export interface NavLink {
  href: string;
  label: string;
  matchPaths?: string[];
}

export interface LayoutConfig {
  logo?: {
    text: string;
    icon?: string;
  };
  navLinks?: NavLink[];
  prependNavLinks?: NavLink[];
  appendNavLinks?: NavLink[];
  headerRight?: React.ReactNode;
}

export interface MetricPoint {
  step: number;
  value: number;
  timestamp: string;
}

export type MetricSeries = Record<string, MetricPoint[]>;
