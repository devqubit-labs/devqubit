/**
 * DevQubit UI Run Detail Page
 */

import { useState, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import {
  Card, CardHeader, CardTitle, Badge, Button, Spinner, KVList, Modal,
  Table, TableHead, TableBody, TableRow, TableHeader, TableCell, EmptyState,
} from '../components';
import { StatusBadge } from '../components/RunsTable';
import { useRun, useApp, useMutation } from '../hooks';
import { shortId, shortDigest, timeAgo, formatNumber } from '../utils';
import type { Artifact } from '../types';

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { api } = useApp();
  const { data: run, loading, error } = useRun(runId!);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);

  const deleteMutation = useMutation(() => api.deleteRun(runId!));
  const baselineMutation = useMutation(() => api.setBaseline(run!.project, runId!));

  const handleDelete = useCallback(async () => {
    await deleteMutation.mutate();
    navigate('/runs');
  }, [deleteMutation, navigate]);

  const handleSetBaseline = useCallback(async () => {
    await baselineMutation.mutate();
    window.location.reload();
  }, [baselineMutation]);

  if (loading) {
    return <Layout><div className="flex justify-center py-12"><Spinner /></div></Layout>;
  }

  if (error || !run) {
    return (
      <Layout>
        <Card>
          <EmptyState message="Run not found" hint={error?.message} />
        </Card>
      </Layout>
    );
  }

  const params = run.data?.params ?? {};
  const metrics = run.data?.metrics ?? {};
  const tags = run.data?.tags ?? {};
  const artifacts = run.artifacts ?? [];
  const backend = run.backend ?? {};
  const fingerprints = run.fingerprints ?? {};
  const errors = run.errors ?? [];

  return (
    <Layout>
      <PageHeader
        title={
          <>
            {run.run_name || 'Unnamed Run'}
            {/* is_baseline would need additional API support */}
          </>
        }
        subtitle={<span className="font-mono">{run.run_id}</span>}
        actions={
          <>
            <Button variant="secondary" size="sm" onClick={handleSetBaseline} loading={baselineMutation.loading}>
              Set as Baseline
            </Button>
            <Button variant="ghost-danger" size="sm" onClick={() => setDeleteModalOpen(true)}>
              Delete
            </Button>
          </>
        }
      />

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
            ...(run.group_id ? [{ label: 'Group', value: <Link to={`/groups/${run.group_id}`}>{run.group_name || shortId(run.group_id)}</Link> }] : []),
          ]} />
        </Card>

        <Card>
          <CardHeader><CardTitle>Fingerprints</CardTitle></CardHeader>
          <KVList items={[
            { label: 'Run', value: <span className="font-mono text-sm">{shortDigest(fingerprints.run)}</span> },
            { label: 'Program', value: <span className="font-mono text-sm">{shortDigest(fingerprints.program)}</span> },
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
                    <TableCell className="text-muted font-medium w-2/5">{k}</TableCell>
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
                    <TableCell className="text-muted font-medium w-2/5">{k}</TableCell>
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
                  <TableCell className="font-mono text-sm truncate-id">{shortDigest(artifact.digest)}</TableCell>
                  <TableCell>
                    <div className="flex gap-2">
                      <Link to={`/runs/${run.run_id}/artifacts/${idx}`}>
                        <Button variant="secondary" size="sm">View</Button>
                      </Link>
                      <a href={api.getArtifactDownloadUrl(run.run_id, idx)}>
                        <Button variant="secondary" size="sm">Download</Button>
                      </a>
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
            <div key={idx} className="mb-4 last:mb-0">
              <strong>{err.type}</strong>: {err.message}
              {err.traceback && <pre className="mt-2">{err.traceback}</pre>}
            </div>
          ))}
        </Card>
      )}

      {/* Delete Modal */}
      <Modal
        open={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Delete Run"
        actions={
          <>
            <Button variant="secondary" onClick={() => setDeleteModalOpen(false)}>Cancel</Button>
            <Button variant="danger" onClick={handleDelete} loading={deleteMutation.loading}>Delete</Button>
          </>
        }
      >
        <p>Are you sure you want to delete this run?</p>
        <p className="font-mono text-sm mt-2">{shortId(run.run_id)}</p>
        <p className="text-sm text-danger mt-2">This action cannot be undone.</p>
      </Modal>
    </Layout>
  );
}
