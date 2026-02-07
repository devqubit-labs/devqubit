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
}

const AppContext = createContext<AppContextValue | null>(null);

/**
 * App context provider props
 */
export interface AppProviderProps {
  children: React.ReactNode;
  api?: ApiClient;
  initialWorkspace?: Workspace | null;
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
}: AppProviderProps) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [currentWorkspace, setCurrentWorkspace] = useState<Workspace | null>(initialWorkspace);

  useEffect(() => {
    api.getCapabilities().then(setCapabilities).catch(console.error);
  }, [api]);

  return (
    <AppContext.Provider value={{ api, capabilities, currentWorkspace, setCurrentWorkspace }}>
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
 */
function useAsync<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = []
): AsyncState<T> & { refetch: () => void } {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null,
  });
  const mountedRef = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const fetch = useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const data = await fetcherRef.current();
      if (mountedRef.current) {
        setState({ data, loading: false, error: null });
      }
    } catch (err) {
      if (mountedRef.current) {
        setState({
          data: null,
          loading: false,
          error: err instanceof ApiError ? err : new ApiError(500, String(err)),
        });
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetch();
    return () => { mountedRef.current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ...state, refetch: fetch };
}

/**
 * Poll with setInterval + immediate refetch when the browser tab regains focus.
 *
 * The visibility listener covers the typical workflow: user starts a run in the
 * terminal, switches to the browser, and expects the UI to be up-to-date instantly.
 */
export function usePolling(refetch: () => void, intervalMs: number, active: boolean) {
  useEffect(() => {
    if (!active) return;

    const id = setInterval(refetch, intervalMs);

    const onVisible = () => {
      if (document.visibilityState === 'visible') refetch();
    };
    document.addEventListener('visibilitychange', onVisible);

    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [refetch, intervalMs, active]);
}

/**
 * Fetch runs list with filters.
 *
 * Polls every second so new runs and status changes appear immediately.
 * Also refetches instantly when the browser tab regains focus.
 */
export function useRuns(params?: {
  project?: string;
  status?: string;
  q?: string;
  limit?: number;
}) {
  const { api, currentWorkspace } = useApp();
  const result = useAsync(
    () => api.listRuns({ ...params, workspace: currentWorkspace?.id }),
    [api, currentWorkspace?.id, params?.project, params?.status, params?.q, params?.limit]
  );

  usePolling(result.refetch, POLL_INTERVAL, true);

  return result;
}

/**
 * Fetch single run by ID.
 *
 * Polls every second while the run is still in a non-terminal state.
 */
export function useRun(runId: string) {
  const { api } = useApp();
  const result = useAsync(
    async () => {
      const { run } = await api.getRun(runId);
      return run;
    },
    [api, runId]
  );

  const isRunning = result.data ? !isTerminalStatus(result.data.status) : false;

  usePolling(result.refetch, POLL_INTERVAL, isRunning);

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

export type {
  AsyncState,
  AppContextValue,
};

// Theme
export { ThemeProvider, useTheme, useThemeOptional } from './useTheme';
export type { Theme, ThemeContextValue, ThemeProviderProps } from './useTheme';
