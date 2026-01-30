/**
 * DevQubit UI Group Detail Page
 */

import { useParams } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import { Card, CardHeader, CardTitle, Spinner, EmptyState } from '../components/ui';
import { RunsTable } from '../components/RunsTable';
import { useGroup, useApp, useMutation } from '../hooks';
import { shortId } from '../utils';

export function GroupDetailPage() {
  const { groupId } = useParams<{ groupId: string }>();
  const { api } = useApp();
  const { data, loading, error, refetch } = useGroup(groupId!);

  const deleteMutation = useMutation((runId: string) => api.deleteRun(runId));

  const handleDelete = async (runId: string) => {
    await deleteMutation.mutate(runId);
    refetch();
  };

  if (loading) {
    return <Layout><div className="flex justify-center py-12"><Spinner /></div></Layout>;
  }

  if (error || !data) {
    return (
      <Layout>
        <Card><EmptyState message="Group not found" hint={error?.message} /></Card>
      </Layout>
    );
  }

  return (
    <Layout>
      <PageHeader
        title={<>Group <span className="font-mono">{shortId(groupId!)}</span></>}
        subtitle={<span className="font-mono text-muted">{groupId}</span>}
      />

      <Card>
        <CardHeader>
          <CardTitle>Runs in Group ({data.runs.length})</CardTitle>
        </CardHeader>
        <RunsTable
          runs={data.runs}
          onDelete={handleDelete}
          loading={deleteMutation.loading}
        />
      </Card>
    </Layout>
  );
}
