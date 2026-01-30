/**
 * DevQubit UI Runs Page
 */

import { useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import { Card, Button, Spinner, FormGroup, FormRow, Label, Input, Select } from '../components/ui';
import { RunsTable } from '../components/RunsTable';
import { useRuns, useProjects, useApp, useMutation } from '../hooks';
import type { RunFilters, RunStatus } from '../types';

export function RunsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { api } = useApp();
  const { data: projectsData } = useProjects();

  const [filters, setFilters] = useState<RunFilters>({
    project: searchParams.get('project') || '',
    status: (searchParams.get('status') as RunStatus | '') || '',
    q: searchParams.get('q') || '',
    limit: parseInt(searchParams.get('limit') || '25', 10),
  });

  const { data, loading, refetch } = useRuns(filters);
  const deleteMutation = useMutation((runId: string) => api.deleteRun(runId));

  const updateFilter = useCallback((key: keyof RunFilters, value: string | number) => {
    setFilters((f) => ({ ...f, [key]: value }));
    const newParams = new URLSearchParams(searchParams);
    if (value) {
      newParams.set(key, String(value));
    } else {
      newParams.delete(key);
    }
    setSearchParams(newParams, { replace: true });
  }, [searchParams, setSearchParams]);

  const handleDelete = useCallback(async (runId: string) => {
    await deleteMutation.mutate(runId);
    refetch();
  }, [deleteMutation, refetch]);

  return (
    <Layout>
      <PageHeader title="Runs" />

      <Card className="mb-4">
        <FormRow>
          <FormGroup>
            <Label htmlFor="project">Project</Label>
            <Select
              id="project"
              value={filters.project}
              onChange={(e) => updateFilter('project', e.target.value)}
            >
              <option value="">All projects</option>
              {projectsData?.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </Select>
          </FormGroup>
          <FormGroup>
            <Label htmlFor="status">Status</Label>
            <Select
              id="status"
              value={filters.status}
              onChange={(e) => updateFilter('status', e.target.value)}
            >
              <option value="">All</option>
              <option value="FINISHED">Finished</option>
              <option value="FAILED">Failed</option>
              <option value="RUNNING">Running</option>
            </Select>
          </FormGroup>
          <FormGroup>
            <Label htmlFor="q">Query</Label>
            <Input
              id="q"
              value={filters.q}
              onChange={(e) => updateFilter('q', e.target.value)}
              placeholder="metric.fidelity > 0.9"
            />
          </FormGroup>
          <FormGroup>
            <Label htmlFor="limit">Limit</Label>
            <Select
              id="limit"
              value={filters.limit}
              onChange={(e) => updateFilter('limit', parseInt(e.target.value, 10))}
            >
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </Select>
          </FormGroup>
          <FormGroup className="flex items-end gap-2">
            <Button variant="primary" onClick={() => refetch()}>
              Filter
            </Button>
            {loading && <Spinner />}
          </FormGroup>
        </FormRow>
      </Card>

      <Card>
        {loading && !data ? (
          <div className="flex justify-center py-8"><Spinner /></div>
        ) : (
          <RunsTable
            runs={data?.runs ?? []}
            onDelete={handleDelete}
            loading={deleteMutation.loading}
          />
        )}
      </Card>
    </Layout>
  );
}
