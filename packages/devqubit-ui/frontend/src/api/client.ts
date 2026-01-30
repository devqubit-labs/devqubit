/**
 * DevQubit API Client
 *
 * HTTP client for communicating with the devqubit backend.
 * Provides typed methods for all API endpoints.
 */

import type {
  RunSummary,
  RunRecord,
  Project,
  Group,
  Capabilities,
  DiffReport,
  Artifact,
} from '../types';

/** API error with status and message */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** API client configuration */
export interface ApiConfig {
  baseUrl?: string;
  headers?: Record<string, string>;
}

/**
 * DevQubit API Client
 *
 * Provides methods for all backend API endpoints. Can be extended
 * by devqubit-hub for workspace-aware requests.
 */
export class ApiClient {
  protected baseUrl: string;
  protected headers: Record<string, string>;

  constructor(config: ApiConfig = {}) {
    this.baseUrl = config.baseUrl ?? '';
    this.headers = {
      'Content-Type': 'application/json',
      ...config.headers,
    };
  }

  /**
   * Make HTTP request with error handling.
   */
  protected async request<T>(
    method: string,
    path: string,
    options: { body?: unknown; params?: Record<string, unknown> } = {}
  ): Promise<T> {
    let url = `${this.baseUrl}${path}`;

    if (options.params) {
      const searchParams = new URLSearchParams();
      Object.entries(options.params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
          searchParams.set(key, String(value));
        }
      });
      const qs = searchParams.toString();
      if (qs) url += `?${qs}`;
    }

    const response = await fetch(url, {
      method,
      headers: this.headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new ApiError(response.status, errorData.detail || response.statusText);
    }

    if (response.status === 204 || response.headers.get('content-length') === '0') {
      return undefined as T;
    }

    return response.json();
  }

  /**
   * Get server capabilities.
   */
  async getCapabilities(): Promise<Capabilities> {
    return this.request<Capabilities>('GET', '/api/v1/capabilities');
  }

  /**
   * List runs with optional filters.
   */
  async listRuns(params?: {
    project?: string;
    status?: string;
    q?: string;
    limit?: number;
    workspace?: string;
  }): Promise<{ runs: RunSummary[]; count: number }> {
    return this.request('GET', '/api/runs', { params });
  }

  /**
   * Get run details by ID.
   */
  async getRun(runId: string): Promise<{ run: RunRecord }> {
    return this.request('GET', `/api/runs/${runId}`);
  }

  /**
   * Delete a run.
   */
  async deleteRun(runId: string): Promise<void> {
    await this.request('DELETE', `/api/runs/${runId}`);
  }

  /**
   * Set project baseline.
   */
  async setBaseline(project: string, runId: string): Promise<{ status: string }> {
    return this.request('POST', `/api/projects/${project}/baseline/${runId}`, {
      params: { redirect: 'false' },
    });
  }

  /**
   * List projects.
   */
  async listProjects(params?: { workspace?: string }): Promise<{ projects: Project[] }> {
    return this.request('GET', '/api/projects', { params });
  }

  /**
   * List groups.
   */
  async listGroups(params?: {
    project?: string;
    workspace?: string;
  }): Promise<{ groups: Group[] }> {
    return this.request('GET', '/api/groups', { params });
  }

  /**
   * Get group details with runs.
   */
  async getGroup(groupId: string): Promise<{ group_id: string; runs: RunSummary[] }> {
    return this.request('GET', `/api/groups/${groupId}`);
  }

  /**
   * Get diff report between two runs.
   */
  async getDiff(runIdA: string, runIdB: string): Promise<{
    run_a: RunSummary;
    run_b: RunSummary;
    report: DiffReport;
  }> {
    return this.request('GET', '/api/diff', {
      params: { a: runIdA, b: runIdB },
    });
  }

  /**
   * Get artifact metadata for a run.
   */
  async getArtifact(runId: string, index: number): Promise<{
    artifact: Artifact;
    size: number;
    content?: string;
    content_json?: unknown;
    preview_available: boolean;
    error?: string;
  }> {
    return this.request('GET', `/api/runs/${runId}/artifacts/${index}`);
  }

  /**
   * Get artifact download URL.
   */
  getArtifactDownloadUrl(runId: string, index: number): string {
    return `${this.baseUrl}/api/runs/${runId}/artifacts/${index}/raw`;
  }
}

/** Default API client instance */
export const api = new ApiClient();
