/**
 * DevQubit UI Artifact Page
 */

import { useParams, Link } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import { Card, CardHeader, CardTitle, Badge, Button, Spinner, KVList, EmptyState } from '../components';
import { useArtifact, useRun, useApp } from '../hooks';
import { shortId, formatBytes, jsonPretty } from '../utils';

export function ArtifactPage() {
  const { runId, index } = useParams<{ runId: string; index: string }>();
  const idx = parseInt(index!, 10);
  const { api } = useApp();
  const { data: run } = useRun(runId!);
  const { data, loading, error } = useArtifact(runId!, idx);

  if (loading) {
    return <Layout><div className="flex justify-center py-12"><Spinner /></div></Layout>;
  }

  if (error || !data) {
    return (
      <Layout>
        <Card><EmptyState message="Artifact not found" hint={error?.message} /></Card>
      </Layout>
    );
  }

  const { artifact, size, content, content_json, preview_available, error: artifactError } = data;
  const maxPreviewSize = 10 * 1024 * 1024; // 10MB

  return (
    <Layout>
      <PageHeader
        title={artifact.kind}
        subtitle={
          <>← Back to run <Link to={`/runs/${runId}`}>{shortId(runId!)}{run?.run_name && ` — ${run.run_name}`}</Link></>
        }
        actions={
          <a href={api.getArtifactDownloadUrl(runId!, idx)}>
            <Button variant="primary">Download</Button>
          </a>
        }
      />

      <Card className="mb-4">
        <KVList items={[
          { label: 'Kind', value: <span className="font-mono">{artifact.kind}</span> },
          { label: 'Role', value: <Badge variant="gray">{artifact.role}</Badge> },
          { label: 'Media Type', value: artifact.media_type },
          { label: 'Digest', value: <span className="font-mono text-sm">{artifact.digest}</span> },
          { label: 'Size', value: formatBytes(size) },
        ]} />
      </Card>

      <Card>
        <CardHeader><CardTitle>Content</CardTitle></CardHeader>

        {artifactError ? (
          <>
            <p className="text-sm text-danger">⚠ Error loading artifact: {artifactError}</p>
            <p className="text-muted mt-2">
              <a href={api.getArtifactDownloadUrl(runId!, idx)}>
                <Button variant="primary">Download to view</Button>
              </a>
            </p>
          </>
        ) : !preview_available ? (
          <>
            <p className="text-muted">
              <strong>Artifact too large for preview</strong> ({formatBytes(size)} exceeds {formatBytes(maxPreviewSize)} limit)
            </p>
            <p className="text-sm text-muted mt-2">Download the artifact to view its contents.</p>
            <p className="mt-4">
              <a href={api.getArtifactDownloadUrl(runId!, idx)}>
                <Button variant="primary">Download</Button>
              </a>
            </p>
          </>
        ) : content_json ? (
          <pre>{jsonPretty(content_json)}</pre>
        ) : content ? (
          <pre>{content}</pre>
        ) : (
          <p className="text-muted">Binary content ({formatBytes(size)}) — download to view</p>
        )}
      </Card>
    </Layout>
  );
}
