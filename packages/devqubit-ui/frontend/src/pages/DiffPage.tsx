/**
 * DevQubit UI Diff Page
 */

import { useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { Layout, PageHeader } from '../components/Layout';
import {
  Card, CardHeader, CardTitle, Badge, Button, Spinner, EmptyState,
  FormGroup, Label, Select, Table, TableHead, TableBody, TableRow, TableHeader, TableCell,
} from '../components';
import { useDiff, useRuns } from '../hooks';
import { shortId, shortDigest, timeAgo } from '../utils';
import type { RunSummary } from '../types';

function DiffSelect() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: runsData } = useRuns({ limit: 100 });
  const runs = runsData?.runs ?? [];

  const [runA, setRunA] = useState(searchParams.get('a') || '');
  const [runB, setRunB] = useState(searchParams.get('b') || '');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (runA && runB) {
      setSearchParams({ a: runA, b: runB });
    }
  };

  return (
    <>
      <Card>
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <FormGroup>
              <Label htmlFor="a">Run A (Baseline)</Label>
              <Select id="a" value={runA} onChange={(e) => setRunA(e.target.value)} required>
                <option value="">Select run...</option>
                {runs.map((run: RunSummary) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_name || 'Unnamed run'} ({shortId(run.run_id)}) — {run.project}
                  </option>
                ))}
              </Select>
            </FormGroup>
            <FormGroup>
              <Label htmlFor="b">Run B (Candidate)</Label>
              <Select id="b" value={runB} onChange={(e) => setRunB(e.target.value)} required>
                <option value="">Select run...</option>
                {runs.map((run: RunSummary) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_name || 'Unnamed run'} ({shortId(run.run_id)}) — {run.project}
                  </option>
                ))}
              </Select>
            </FormGroup>
          </div>
          <Button type="submit" variant="primary">Compare</Button>
        </form>
      </Card>

      {runs.length === 0 && (
        <Card className="mt-4">
          <EmptyState message="No runs available" hint="Create some runs first" />
        </Card>
      )}
    </>
  );
}

function DiffCell({ match }: { match: boolean }) {
  return (
    <span className={match ? 'diff-match' : 'diff-mismatch'}>
      {match ? '✓ Match' : '✗ Different'}
    </span>
  );
}

function DiffResult({ runIdA, runIdB }: { runIdA: string; runIdB: string }) {
  const { data, loading, error } = useDiff(runIdA, runIdB);

  if (loading) return <div className="flex justify-center py-12"><Spinner /></div>;
  if (error || !data) return <Card><EmptyState message="Failed to load diff" hint={error?.message} /></Card>;

  const { run_a, run_b, report } = data;

  return (
    <>
      {/* Run Headers */}
      <Card className="mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm text-muted mb-2">Run A (Baseline)</h3>
            <Link to={`/runs/${run_a.run_id}`} className="text-primary hover:underline font-medium">
              {run_a.run_name || 'Unnamed run'}
            </Link>
            <p className="text-xs text-muted mt-1">
              {shortId(run_a.run_id)} • {run_a.project} • {timeAgo(run_a.created_at)}
            </p>
          </div>
          <div>
            <h3 className="text-sm text-muted mb-2">Run B (Candidate)</h3>
            <Link to={`/runs/${run_b.run_id}`} className="text-primary hover:underline font-medium">
              {run_b.run_name || 'Unnamed run'}
            </Link>
            <p className="text-xs text-muted mt-1">
              {shortId(run_b.run_id)} • {run_b.project} • {timeAgo(run_b.created_at)}
            </p>
          </div>
        </div>
      </Card>

      {/* Metadata */}
      <Card className="mb-4">
        <CardHeader><CardTitle>Metadata</CardTitle></CardHeader>
        <Table>
          <TableBody>
            <TableRow>
              <TableCell className="text-muted font-medium w-2/5">Project</TableCell>
              <TableCell><DiffCell match={report.metadata.project_match} /></TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="text-muted font-medium">Backend</TableCell>
              <TableCell><DiffCell match={report.metadata.backend_match} /></TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Card>

      {/* Fingerprints */}
      <Card className="mb-4">
        <CardHeader><CardTitle>Fingerprints</CardTitle></CardHeader>
        <Table>
          <TableBody>
            <TableRow>
              <TableCell className="text-muted font-medium w-2/5">Run A</TableCell>
              <TableCell className="font-mono text-xs">{shortDigest(report.fingerprints.a)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="text-muted font-medium">Run B</TableCell>
              <TableCell className="font-mono text-xs">{shortDigest(report.fingerprints.b)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="text-muted font-medium">Match</TableCell>
              <TableCell><DiffCell match={report.fingerprints.a === report.fingerprints.b} /></TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Card>

      {/* Program */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>
            Program
            <Badge variant={report.program.exact_match ? 'success' : 'warning'}>
              {report.program.exact_match ? 'Exact Match' : 'Different'}
            </Badge>
          </CardTitle>
        </CardHeader>
        <Table>
          <TableBody>
            <TableRow>
              <TableCell className="text-muted font-medium w-2/5">Exact Match</TableCell>
              <TableCell><DiffCell match={report.program.exact_match} /></TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="text-muted font-medium">Structural Match</TableCell>
              <TableCell><DiffCell match={report.program.structural_match} /></TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Card>

      {/* Device Drift */}
      {report.device_drift && (
        <Card className="mb-4">
          <CardHeader>
            <CardTitle>
              Device Calibration
              <Badge variant={report.device_drift.significant_drift ? 'warning' : 'success'}>
                {report.device_drift.significant_drift ? 'Significant Drift' : 'Stable'}
              </Badge>
            </CardTitle>
          </CardHeader>
          {report.device_drift.top_drifts && report.device_drift.top_drifts.length > 0 && (
            <Table>
              <TableHead>
                <TableRow>
                  <TableHeader>Metric</TableHeader>
                  <TableHeader>Change</TableHeader>
                </TableRow>
              </TableHead>
              <TableBody>
                {report.device_drift.top_drifts.map((drift: { metric: string; percent_change?: number }) => (
                  <TableRow key={drift.metric}>
                    <TableCell>{drift.metric}</TableCell>
                    <TableCell className="font-mono">
                      {drift.percent_change !== undefined ? `${drift.percent_change.toFixed(1)}%` : 'N/A'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Card>
      )}

      {/* Params */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>
            Parameters
            <Badge variant={report.params.match ? 'success' : 'warning'}>
              {report.params.match ? 'Match' : 'Different'}
            </Badge>
          </CardTitle>
        </CardHeader>
        {report.params.match ? (
          <p className="text-muted">All parameters match</p>
        ) : (
          <>
            {report.params.changed && Object.keys(report.params.changed).length > 0 && (
              <>
                <h4 className="text-sm text-muted mb-2">Changed</h4>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeader>Parameter</TableHeader>
                      <TableHeader>Run A</TableHeader>
                      <TableHeader>Run B</TableHeader>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(report.params.changed).map(([key, values]) => (
                      <TableRow key={key}>
                        <TableCell>{key}</TableCell>
                        <TableCell className="font-mono">{String((values as { a: unknown; b: unknown }).a)}</TableCell>
                        <TableCell className="font-mono">{String((values as { a: unknown; b: unknown }).b)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            )}
          </>
        )}
      </Card>

      {/* Metrics */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>
            Metrics
            <Badge variant={report.metrics.match ? 'success' : 'warning'}>
              {report.metrics.match ? 'Match' : 'Different'}
            </Badge>
          </CardTitle>
        </CardHeader>
        {report.metrics.match ? (
          <p className="text-muted">All metrics match</p>
        ) : (
          <>
            {report.metrics.changed && Object.keys(report.metrics.changed).length > 0 && (
              <>
                <h4 className="text-sm text-muted mb-2">Changed</h4>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeader>Metric</TableHeader>
                      <TableHeader>Run A</TableHeader>
                      <TableHeader>Run B</TableHeader>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(report.metrics.changed).map(([key, values]) => (
                      <TableRow key={key}>
                        <TableCell>{key}</TableCell>
                        <TableCell className="font-mono">{(values as { a: number; b: number }).a}</TableCell>
                        <TableCell className="font-mono">{(values as { a: number; b: number }).b}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            )}
          </>
        )}
      </Card>

      {/* TVD */}
      {report.tvd !== undefined && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Total Variation Distance</CardTitle></CardHeader>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell className="text-muted font-medium">TVD</TableCell>
                <TableCell className="font-mono">{report.tvd.toFixed(6)}</TableCell>
              </TableRow>
              {report.shots && (
                <>
                  <TableRow>
                    <TableCell className="text-muted font-medium">Shots A</TableCell>
                    <TableCell className="font-mono">{report.shots.a}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="text-muted font-medium">Shots B</TableCell>
                    <TableCell className="font-mono">{report.shots.b}</TableCell>
                  </TableRow>
                </>
              )}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Warnings */}
      {report.warnings && report.warnings.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Warnings</CardTitle></CardHeader>
          <ul className="list-disc pl-5 space-y-1">
            {report.warnings.map((warning: string, idx: number) => (
              <li key={idx} className="text-sm text-warning">{warning}</li>
            ))}
          </ul>
        </Card>
      )}
    </>
  );
}

export function DiffPage() {
  const [searchParams] = useSearchParams();
  const runIdA = searchParams.get('a');
  const runIdB = searchParams.get('b');
  const hasBothRuns = runIdA && runIdB;

  return (
    <Layout>
      <PageHeader
        title={
          <>
            Compare Runs
            {hasBothRuns && <Badge variant="info">Comparing</Badge>}
          </>
        }
        subtitle={hasBothRuns && <Link to="/diff">← Select different runs</Link>}
      />
      {hasBothRuns ? <DiffResult runIdA={runIdA} runIdB={runIdB} /> : <DiffSelect />}
    </Layout>
  );
}
