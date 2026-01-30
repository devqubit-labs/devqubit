/**
 * DevQubit UI Projects Page
 */

import { Layout, PageHeader } from '../components/Layout';
import { Card, Spinner } from '../components/ui';
import { ProjectsTable } from '../components/ProjectsTable';
import { useProjects, useApp } from '../hooks';

export function ProjectsPage() {
  const { currentWorkspace } = useApp();
  const { data: projects, loading } = useProjects();

  return (
    <Layout>
      <PageHeader title="Projects" />
      <Card>
        {loading ? (
          <div className="flex justify-center py-8"><Spinner /></div>
        ) : (
          <ProjectsTable projects={projects ?? []} currentWorkspace={currentWorkspace} />
        )}
      </Card>
    </Layout>
  );
}
