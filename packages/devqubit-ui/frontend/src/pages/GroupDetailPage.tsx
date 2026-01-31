/**
 * DevQubit UI Group Detail Page
 */

import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import { Card, CardHeader, CardTitle, Spinner, EmptyState, Toast } from '../components/ui';
import { RunsTable } from '../components/RunsTable';
import { useGroup, useApp, useMutation } from '../hooks';
import { shortId } from '../utils';

export function GroupDetailPage() {
  const { groupId } = useParams<{ groupId: string }>();
  const { api } = useApp();
  const { data, loading, error, refetch } = useGroup(groupId!);
  const [toast, setToast] = useState<{ message: string; variant: 'success' | 'error' } | null>(null);

  const deleteMutation = useMutation((runId: string) => api.deleteRun(runId));

  // Auto-hide toast
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  const handleDelete = async (runId: string) => {
    try {
      await deleteMutation.mutate(runId);
      setToast({ message: 'Run deleted', variant: 'success' });
      refetch();
    } catch {
      setToast({ message: 'Failed to delete run', variant: 'error' });
    }
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

      {toast && (
        <Toast
          message={toast.message}
          variant={toast.variant}
          visible={!!toast}
          onClose={() => setToast(null)}
        />
      )}
    </Layout>
  );
}
