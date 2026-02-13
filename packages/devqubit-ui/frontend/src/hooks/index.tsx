/**
 * DevQubit UI React Hooks
 *
 * Custom hooks for data fetching and state management.
 */

import { useState, useEffect, useCallback, useRef, useContext, createContext } from 'react';
import { ApiClient, api as defaultApi, ApiError } from '../api';
import { isTerminalStatus } from '../utils';
import type { Capabilities, Workspace } from '../types';

/** Default polling interval for data fetching (ms). */
export const POLL_INTERVAL = 1_000;

/**
 * Configurable polling intervals.
 *
 * Providers can override these to tune polling frequency for their
 * deployment scale (e.g. longer intervals for multi-user servers).
 */
export interface PollingConfig {
  /** Interval while active (non-terminal) runs exist (ms). */
  runsActive?: number;
  /** Interval when all runs are terminal (ms). */
  runsIdle?: number;
  /** Interval for single-run detail while running (ms). */
  runDetail?: number;
}

const DEFAULT_POLLING: Required<PollingConfig> = {
  runsActive: POLL_INTERVAL,
  runsIdle: POLL_INTERVAL,
  runDetail: POLL_INTERVAL,
};

/** Async state for data fetching */
interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
}

/** App context for shared state */
interface AppContextValue {
  api: ApiClient;
  capabilities: Capabilities | null;
  currentWorkspace: Workspace | null;
  setCurrentWorkspace: (workspace: Workspace | null) => void;
  pollingConfig: Required<PollingConfig>;
}

const AppContext = createContext<AppContextValue | null>(null);

/**
 * App context provider props
 */
export interface AppProviderProps {
  children: React.ReactNode;
  api?: ApiClient;
  initialWorkspace?: Workspace | null;
  pollingConfig?: PollingConfig;
}

/**
 * App context provider component.
 *
 * Provides API client and shared state to child components.
 */
export function AppProvider({
  children,
  api = defaultApi,
  initialWorkspace = null,
  pollingConfig,
}: AppProviderProps) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [currentWorkspace, setCurrentWorkspace] = useState<Workspace | null>(initialWorkspace);

  const resolvedPolling: Required<PollingConfig> = {
    ...DEFAULT_POLLING,
    ...pollingConfig,
  };

  useEffect(() => {
    api.getCapabilities().then(setCapabilities).catch(console.error);
  }, [api]);

  return (
    <AppContext.Provider value={{ api, capabilities, currentWorkspace, setCurrentWorkspace, pollingConfig: resolvedPolling }}>
      {children}
    </AppContext.Provider>
  );
}

/**
 * Access app context.
 */
export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}

/**
 * Generic async data fetcher hook.
 *
 * Background refetches (from polling) are silent — they keep previous
 * data visible instead of flashing a loading indicator on every tick.
 */
function useAsync<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = []
): AsyncState<T> & { refetch: () => Promise<void> } {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null,
  });
  const mountedRef = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const doFetch = useCallback(async (silent: boolean) => {
    if (!silent) {
      setState(s => ({ ...s, loading: true, error: null }));
    }
    try {
      const data = await fetcherRef.current();
      if (mountedRef.current) {
        setState({ data, loading: false, error: null });
      }
    } catch (err) {
      if (mountedRef.current) {
        const apiError = err instanceof ApiError ? err : new ApiError(500, String(err));
        // On silent error keep previous data visible.
        setState(s => ({
          data: silent ? s.data : null,
          loading: false,
          error: apiError,
        }));
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    doFetch(false);
    return () => { mountedRef.current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  const refetch = useCallback(() => doFetch(true), [doFetch]);

  return { ...state, refetch };
}

/**
 * Poll with setTimeout chain + visibility awareness.
 *
 * Uses setTimeout (not setInterval) so the next tick only starts after
 * the previous fetch completes — prevents request pileup on slow connections.
 * Pauses while the tab is hidden; refetches immediately on focus.
 */
export function usePolling(
  refetch: () => void | Promise<void>,
  intervalMs: number,
  active: boolean,
) {
  useEffect(() => {
    if (!active) return;

    let timeoutId: ReturnType<typeof setTimeout>;
    let cancelled = false;

    const schedule = () => {
      if (cancelled) return;
      // ±10 % jitter to spread load across concurrent clients.
      const jitter = intervalMs * 0.1 * (Math.random() * 2 - 1);
      timeoutId = setTimeout(tick, Math.max(500, intervalMs + jitter));
    };

    const tick = async () => {
      if (cancelled) return;
      if (document.visibilityState === 'hidden') return;
      try { await refetch(); } catch { /* useAsync handles errors */ }
      schedule();
    };

    schedule();

    const onVisible = () => {
      if (cancelled || document.visibilityState !== 'visible') return;
      clearTimeout(timeoutId);
      Promise.resolve(refetch()).catch(() => {}).finally(schedule);
    };
    document.addEventListener('visibilitychange', onVisible);

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [refetch, intervalMs, active]);
}

/**
 * Fetch runs list with filters.
 *
 * Uses adaptive polling: faster while active runs exist, slower when idle.
 * Also refetches instantly when the browser tab regains focus.
 */
export function useRuns(params?: {
  project?: string;
  status?: string;
  q?: string;
  limit?: number;
}) {
  const { api, currentWorkspace, pollingConfig } = useApp();
  const result = useAsync(
    () => api.listRuns({ ...params, workspace: currentWorkspace?.id }),
    [api, currentWorkspace?.id, params?.project, params?.status, params?.q, params?.limit]
  );

  const hasActiveRuns = result.data?.runs.some(r => !isTerminalStatus(r.status)) ?? false;
  const interval = hasActiveRuns ? pollingConfig.runsActive : pollingConfig.runsIdle;

  usePolling(result.refetch, interval, true);

  return result;
}

/**
 * Fetch single run by ID.
 *
 * Polls while the run is in a non-terminal state.
 */
export function useRun(runId: string) {
  const { api, pollingConfig } = useApp();
  const result = useAsync(
    async () => {
      const { run } = await api.getRun(runId);
      return run;
    },
    [api, runId]
  );

  const isRunning = result.data ? !isTerminalStatus(result.data.status) : false;

  usePolling(result.refetch, pollingConfig.runDetail, isRunning);

  return result;
}

/**
 * Fetch projects list.
 */
export function useProjects() {
  const { api, currentWorkspace } = useApp();
  return useAsync(
    async () => {
      const { projects } = await api.listProjects({ workspace: currentWorkspace?.id });
      return projects;
    },
    [api, currentWorkspace?.id]
  );
}

/**
 * Fetch groups list.
 */
export function useGroups(params?: { project?: string }) {
  const { api, currentWorkspace } = useApp();
  return useAsync(
    async () => {
      const { groups } = await api.listGroups({ ...params, workspace: currentWorkspace?.id });
      return groups;
    },
    [api, currentWorkspace?.id, params?.project]
  );
}

/**
 * Fetch group by ID.
 */
export function useGroup(groupId: string) {
  const { api } = useApp();
  return useAsync(
    () => api.getGroup(groupId),
    [api, groupId]
  );
}

/**
 * Fetch diff report.
 */
export function useDiff(runIdA: string, runIdB: string) {
  const { api } = useApp();
  return useAsync(
    () => api.getDiff(runIdA, runIdB),
    [api, runIdA, runIdB]
  );
}

/**
 * Fetch artifact metadata.
 */
export function useArtifact(runId: string, index: number) {
  const { api } = useApp();
  return useAsync(
    () => api.getArtifact(runId, index),
    [api, runId, index]
  );
}

/**
 * Hook for mutation operations (delete, set baseline, etc.)
 */
export function useMutation<TArgs extends unknown[], TResult>(
  mutationFn: (...args: TArgs) => Promise<TResult>
): {
  mutate: (...args: TArgs) => Promise<TResult>;
  loading: boolean;
  error: ApiError | null;
} {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (...args: TArgs) => {
    setLoading(true);
    setError(null);
    try {
      const result = await mutationFn(...args);
      return result;
    } catch (err) {
      const apiError = err instanceof ApiError ? err : new ApiError(500, String(err));
      setError(apiError);
      throw apiError;
    } finally {
      setLoading(false);
    }
  }, [mutationFn]);

  return { mutate, loading, error };
}

/**
 * Fetch metric time-series for a run.
 *
 * Returns null when the run has no step-based metrics.
 * Polls while the run is active so charts update in near-real-time.
 */
export function useMetricSeries(runId: string, isRunning: boolean) {
  const { api, pollingConfig } = useApp();
  const result = useAsync(
    async () => {
      try {
        const { series } = await api.getMetricSeries(runId);
        // Only return if there's actual data
        const hasData = Object.values(series).some(pts => pts.length > 0);
        return hasData ? series : null;
      } catch {
        // 501 = backend doesn't support it, 404 = no data — both fine
        return null;
      }
    },
    [api, runId]
  );

  usePolling(result.refetch, pollingConfig.runDetail, isRunning);

  return result;
}

export type {
  AsyncState,
  AppContextValue,
};

// Theme
export { ThemeProvider, useTheme, useThemeOptional } from './useTheme';
export type { Theme, ThemeContextValue, ThemeProviderProps } from './useTheme';
