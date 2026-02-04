/**
 * DevQubit API Client
 *
 * HTTP client for communicating with the devqubit backend.
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

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface ApiConfig {
  baseUrl?: string;
  headers?: Record<string, string>;
}

/**
 * DevQubit API Client.
 *
 * Provides typed methods for all backend API endpoints.
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

  async getCapabilities(): Promise<Capabilities> {
    return this.request<Capabilities>('GET', '/api/v1/capabilities');
  }

  async listRuns(params?: {
    project?: string;
    status?: string;
    q?: string;
    limit?: number;
    workspace?: string;
  }): Promise<{ runs: RunSummary[]; count: number }> {
    return this.request('GET', '/api/runs', { params });
  }

  async getRun(runId: string): Promise<{ run: RunRecord }> {
    return this.request('GET', `/api/runs/${runId}`);
  }

  async deleteRun(runId: string): Promise<void> {
    await this.request('DELETE', `/api/runs/${runId}`);
  }

  async setBaseline(project: string, runId: string): Promise<{ status: string }> {
    return this.request('POST', `/api/projects/${project}/baseline/${runId}`, {
      params: { redirect: 'false' },
    });
  }

  async listProjects(params?: { workspace?: string }): Promise<{ projects: Project[] }> {
    return this.request('GET', '/api/projects', { params });
  }

  async listGroups(params?: {
    project?: string;
    workspace?: string;
  }): Promise<{ groups: Group[] }> {
    return this.request('GET', '/api/groups', { params });
  }

  async getGroup(groupId: string): Promise<{ group_id: string; runs: RunSummary[] }> {
    return this.request('GET', `/api/groups/${groupId}`);
  }

  async getDiff(runIdA: string, runIdB: string): Promise<{
    run_a: RunSummary;
    run_b: RunSummary;
    report: DiffReport;
  }> {
    return this.request('GET', '/api/diff', {
      params: { a: runIdA, b: runIdB },
    });
  }

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

  getArtifactDownloadUrl(runId: string, index: number): string {
    return `${this.baseUrl}/api/runs/${runId}/artifacts/${index}/raw`;
  }

  async createExport(runId: string): Promise<{
    status: string;
    run_id: string;
    artifact_count: number;
    object_count: number;
    missing_objects: string[];
  }> {
    return this.request('POST', `/api/runs/${runId}/export`);
  }

  async getExportInfo(runId: string): Promise<{
    run_id: string;
    run_name?: string;
    project?: string;
    artifact_count: number;
    object_count: number;
    available_objects: number;
    missing_objects: number;
  }> {
    return this.request('GET', `/api/runs/${runId}/export/info`);
  }

  getExportDownloadUrl(runId: string): string {
    return `${this.baseUrl}/api/runs/${runId}/export/download`;
  }
}

export const api = new ApiClient();
