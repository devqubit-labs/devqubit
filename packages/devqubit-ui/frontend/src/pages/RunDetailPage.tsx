/**
 * Run Detail Page
 */

import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Layout } from '../components/Layout';
import {
  Card, CardHeader, CardTitle, Badge, Button, Spinner, EmptyState, KVList,
  Table, TableHead, TableBody, TableRow, TableHeader, TableCell, Modal, Toast,
  ExportRunButton,
} from '../components';
import { StatusBadge } from '../components/RunsTable';
import { useRun, useProjects, useApp, useMutation } from '../hooks';
import { shortId, shortDigest, timeAgo, formatNumber } from '../utils';
import type { Artifact, Project } from '../types';

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { api } = useApp();
  const { data: run, loading, error, refetch } = useRun(runId!);
  const { data: projects, refetch: refetchProjects } = useProjects();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [toast, setToast] = useState<{ message: string; variant: 'success' | 'error' } | null>(null);

  const { mutate: doSetBaseline, loading: settingBaseline } = useMutation(
    () => api.setBaseline(run!.project, run!.run_id)
  );

  const { mutate: doDelete, loading: deleting } = useMutation(
    () => api.deleteRun(runId!)
  );

  // Check if this run is the baseline for its project
  const project = projects?.find((p: Project) => p.name === run?.project);
  const isBaseline = project?.baseline?.run_id === runId;

  // Auto-hide toast
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  if (loading) {
    return <Layout><div className="flex justify-center py-12"><Spinner /></div></Layout>;
  }

  if (error || !run) {
    return (
      <Layout>
        <Card>
          <EmptyState
            message="Run not found"
            hint={error?.message || `Run ${runId} does not exist`}
          />
        </Card>
      </Layout>
    );
  }

  const backend = run.backend || {};
  const fingerprints = run.fingerprints || {};
  const params = run.data?.params || {};
  const metrics = run.data?.metrics || {};
  const tags = run.data?.tags || {};
  const artifacts = run.artifacts || [];
  const errors = run.errors || [];

  const handleSetBaseline = async () => {
    try {
      await doSetBaseline();
      await refetchProjects();
      refetch();
      setToast({ message: 'Baseline updated', variant: 'success' });
    } catch {
      setToast({ message: 'Failed to set baseline', variant: 'error' });
    }
  };

  const handleDelete = async () => {
    try {
      await doDelete();
      setShowDeleteModal(false);
      setToast({ message: 'Run deleted', variant: 'success' });
      setTimeout(() => navigate('/runs'), 1000);
    } catch {
      setToast({ message: 'Failed to delete run', variant: 'error' });
      setShowDeleteModal(false);
    }
  };

  const handleDownload = async (idx: number) => {
    const artifact = artifacts[idx];
    try {
      const url = api.getArtifactDownloadUrl(run.run_id, idx);
      const response = await fetch(url);
      if (!response.ok) throw new Error('Download failed');

      const blob = await response.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = artifact?.kind || `artifact-${idx}`;
      a.click();
      URL.revokeObjectURL(a.href);

      setToast({ message: 'Download started', variant: 'success' });
    } catch {
      setToast({ message: 'Download failed', variant: 'error' });
    }
  };

  return (
    <Layout>
      {/* Baseline Banner */}
      {isBaseline && (
        <div className="mb-4 px-4 py-3 bg-[#EFF6FF] border border-[#2563EB]/20 rounded-md text-[#2563EB] text-sm font-medium">
          This run is the current baseline for project "{run.project}"
        </div>
      )}

      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">
            {run.run_name || 'Unnamed Run'}
          </h1>
          <p className="text-muted text-sm font-mono">{run.run_id}</p>
        </div>
        <div className="flex gap-2">
          <ExportRunButton runId={run.run_id} runName={run.run_name} />
          {!isBaseline && (
            <Button variant="secondary" size="sm" onClick={handleSetBaseline} disabled={settingBaseline}>
              {settingBaseline && <Spinner />}
              Set as Baseline
            </Button>
          )}
          <Button variant="ghost-danger" size="sm" onClick={() => setShowDeleteModal(true)}>
            Delete
          </Button>
        </div>
      </div>

      {/* Overview & Fingerprints */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Card>
          <CardHeader><CardTitle>Overview</CardTitle></CardHeader>
          <KVList items={[
            { label: 'Project', value: <Link to={`/runs?project=${run.project}`}>{run.project}</Link> },
            { label: 'Name', value: run.run_name || '—' },
            { label: 'Adapter', value: run.adapter || 'N/A' },
            { label: 'Status', value: <StatusBadge status={run.status} /> },
            { label: 'Created', value: `${run.created_at} (${timeAgo(run.created_at)})` },
            { label: 'Backend', value: backend.name || 'N/A' },
            ...(run.group_id ? [{
              label: 'Group',
              value: <Link to={`/groups/${run.group_id}`}>{run.group_name || shortId(run.group_id)}</Link>
            }] : []),
          ]} />
        </Card>

        <Card>
          <CardHeader><CardTitle>Fingerprints</CardTitle></CardHeader>
          <KVList items={[
            { label: 'Run', value: <span className="font-mono text-sm truncate">{fingerprints.run || 'N/A'}</span> },
            { label: 'Program', value: <span className="font-mono text-sm truncate">{fingerprints.program || 'N/A'}</span> },
          ]} />
        </Card>
      </div>

      {/* Params, Metrics, Tags */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <Card>
          <CardHeader><CardTitle>Parameters</CardTitle></CardHeader>
          {Object.keys(params).length ? (
            <Table>
              <TableBody>
                {Object.entries(params).map(([k, v]) => (
                  <TableRow key={k}>
                    <TableCell>{k}</TableCell>
                    <TableCell className="font-mono">{String(v)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-muted">No parameters</p>
          )}
        </Card>

        <Card>
          <CardHeader><CardTitle>Metrics</CardTitle></CardHeader>
          {Object.keys(metrics).length ? (
            <Table>
              <TableBody>
                {Object.entries(metrics).map(([k, v]) => (
                  <TableRow key={k}>
                    <TableCell>{k}</TableCell>
                    <TableCell className="font-mono">{typeof v === 'number' ? formatNumber(v) : String(v)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-muted">No metrics</p>
          )}
        </Card>

        <Card>
          <CardHeader><CardTitle>Tags</CardTitle></CardHeader>
          {Object.keys(tags).length ? (
            <div className="flex flex-wrap gap-2">
              {Object.entries(tags).map(([k, v]) => (
                <Badge key={k} variant="gray">{k}: {String(v)}</Badge>
              ))}
            </div>
          ) : (
            <p className="text-muted">No tags</p>
          )}
        </Card>
      </div>

      {/* Artifacts */}
      <Card className="mb-4">
        <CardHeader><CardTitle>Artifacts ({artifacts.length})</CardTitle></CardHeader>
        {artifacts.length ? (
          <Table>
            <TableHead>
              <TableRow>
                <TableHeader>#</TableHeader>
                <TableHeader>Kind</TableHeader>
                <TableHeader>Role</TableHeader>
                <TableHeader>Media Type</TableHeader>
                <TableHeader>Digest</TableHeader>
                <TableHeader>Actions</TableHeader>
              </TableRow>
            </TableHead>
            <TableBody>
              {artifacts.map((artifact: Artifact, idx: number) => (
                <TableRow key={idx}>
                  <TableCell>{idx}</TableCell>
                  <TableCell className="font-mono text-sm">{artifact.kind}</TableCell>
                  <TableCell><Badge variant="gray">{artifact.role}</Badge></TableCell>
                  <TableCell className="text-muted text-sm">{artifact.media_type}</TableCell>
                  <TableCell className="font-mono text-sm">{shortDigest(artifact.digest)}</TableCell>
                  <TableCell>
                    <div className="flex gap-2">
                      <Link to={`/runs/${run.run_id}/artifacts/${idx}`}>
                        <Button variant="secondary" size="sm">View</Button>
                      </Link>
                      <Button variant="secondary" size="sm" onClick={() => handleDownload(idx)}>
                        Download
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <p className="text-muted">No artifacts</p>
        )}
      </Card>

      {/* Errors */}
      {errors.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-danger">Errors</CardTitle></CardHeader>
          {errors.map((err: { type: string; message: string; traceback?: string }, idx: number) => (
            <div key={idx} className="mb-2">
              <strong>{err.type}</strong>: {err.message}
              {err.traceback && <pre className="mt-2">{err.traceback}</pre>}
            </div>
          ))}
        </Card>
      )}

      {/* Delete Modal */}
      <Modal
        open={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Delete Run"
        actions={
          <>
            <Button variant="secondary" onClick={() => setShowDeleteModal(false)}>Cancel</Button>
            <Button variant="danger" onClick={handleDelete} disabled={deleting}>
              {deleting && <Spinner />}
              Delete
            </Button>
          </>
        }
      >
        <p>Are you sure you want to delete this run?</p>
        <p className="font-mono text-sm mt-2">{shortId(run.run_id)}</p>
        <p className="text-sm text-danger mt-2">This action cannot be undone.</p>
      </Modal>

      {/* Toast */}
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
