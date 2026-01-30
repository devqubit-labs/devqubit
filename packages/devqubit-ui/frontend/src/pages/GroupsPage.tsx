/**
 * DevQubit UI Groups Page
 */

import { useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import { Card, Button, Spinner, FormGroup, FormRow, Label, Select } from '../components/ui';
import { GroupsTable } from '../components/GroupsTable';
import { useGroups, useProjects } from '../hooks';

export function GroupsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: projectsData } = useProjects();

  const [project, setProject] = useState(searchParams.get('project') || '');
  const { data: groups, loading, refetch } = useGroups({ project: project || undefined });

  const updateProject = useCallback((value: string) => {
    setProject(value);
    const newParams = new URLSearchParams(searchParams);
    if (value) {
      newParams.set('project', value);
    } else {
      newParams.delete('project');
    }
    setSearchParams(newParams, { replace: true });
  }, [searchParams, setSearchParams]);

  return (
    <Layout>
      <PageHeader title="Run Groups" />

      <Card className="mb-4">
        <FormRow>
          <FormGroup>
            <Label htmlFor="project">Project</Label>
            <Select
              id="project"
              value={project}
              onChange={(e) => updateProject(e.target.value)}
            >
              <option value="">All projects</option>
              {projectsData?.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </Select>
          </FormGroup>
          <FormGroup className="flex items-end gap-2">
            <Button variant="primary" onClick={() => refetch()}>Filter</Button>
            {loading && <Spinner />}
          </FormGroup>
        </FormRow>
      </Card>

      <Card>
        {loading && !groups ? (
          <div className="flex justify-center py-8"><Spinner /></div>
        ) : (
          <GroupsTable groups={groups ?? []} />
        )}
      </Card>
    </Layout>
  );
}
