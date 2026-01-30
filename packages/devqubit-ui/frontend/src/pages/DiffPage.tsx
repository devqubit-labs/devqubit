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
                {runs.map((run) => (
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
                {runs.map((run) => (
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

      <Card className="mt-4">
        <CardHeader><CardTitle>Tips</CardTitle></CardHeader>
        <ul className="text-muted text-sm list-disc pl-5 space-y-1">
          <li>Select two runs to compare their parameters, metrics, and artifacts</li>
          <li>The diff will show which parameters changed and compute TVD for result distributions</li>
          <li>You can also compare from the run detail page against the project baseline</li>
        </ul>
      </Card>
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
            <h3 className="text-xs text-muted uppercase tracking-wider mb-1">Run A (Baseline)</h3>
            <p><Link to={`/runs/${run_a.run_id}`}>{run_a.run_name || 'Unnamed Run'}</Link></p>
            <p className="font-mono text-sm text-muted">{shortId(run_a.run_id)}</p>
            <p className="text-muted text-sm">{run_a.project} · {timeAgo(run_a.created_at)}</p>
          </div>
          <div>
            <h3 className="text-xs text-muted uppercase tracking-wider mb-1">Run B (Candidate)</h3>
            <p><Link to={`/runs/${run_b.run_id}`}>{run_b.run_name || 'Unnamed Run'}</Link></p>
            <p className="font-mono text-sm text-muted">{shortId(run_b.run_id)}</p>
            <p className="text-muted text-sm">{run_b.project} · {timeAgo(run_b.created_at)}</p>
          </div>
        </div>
      </Card>

      {/* Metadata & Fingerprints */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Card>
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

        <Card>
          <CardHeader><CardTitle>Fingerprints</CardTitle></CardHeader>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell className="text-muted font-medium w-1/3">Run A</TableCell>
                <TableCell className="font-mono text-sm">{shortDigest(report.fingerprints.a)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="text-muted font-medium">Run B</TableCell>
                <TableCell className="font-mono text-sm">{shortDigest(report.fingerprints.b)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="text-muted font-medium">Match</TableCell>
                <TableCell><DiffCell match={report.fingerprints.a === report.fingerprints.b} /></TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </Card>
      </div>

      {/* Program & Device Calibration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Card>
          <CardHeader>
            <CardTitle>
              Program
              {report.program.exact_match ? (
                <Badge variant="success">Exact Match</Badge>
              ) : report.program.structural_match ? (
                <Badge variant="info">Structural Match</Badge>
              ) : (
                <Badge variant="warning">Different</Badge>
              )}
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

        <Card>
          <CardHeader>
            <CardTitle>
              Device Calibration
              {report.device_drift?.significant_drift ? (
                <Badge variant="warning">Drifted</Badge>
              ) : report.device_drift?.has_calibration_data ? (
                <Badge variant="success">Stable</Badge>
              ) : (
                <Badge variant="gray">N/A</Badge>
              )}
            </CardTitle>
          </CardHeader>
          {report.device_drift?.significant_drift ? (
            <p className="text-sm text-warning">⚠ Significant calibration drift detected.</p>
          ) : !report.device_drift?.has_calibration_data ? (
            <p className="text-muted">No calibration data available</p>
          ) : (
            <p className="text-muted">Calibration within acceptable thresholds</p>
          )}
        </Card>
      </div>

      {/* Parameters */}
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
                        <TableCell className="font-mono">{String(values.a)}</TableCell>
                        <TableCell className="font-mono">{String(values.b)}</TableCell>
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
                        <TableCell className="font-mono">{values.a}</TableCell>
                        <TableCell className="font-mono">{values.b}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            )}
          </>
        )}
      </Card>

      {/* TVD Results */}
      {report.tvd !== undefined && report.tvd !== null && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Results</CardTitle></CardHeader>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell className="text-muted font-medium w-2/5">Total Variation Distance (TVD)</TableCell>
                <TableCell className="font-mono">{report.tvd.toFixed(6)}</TableCell>
              </TableRow>
              {report.shots && (
                <TableRow>
                  <TableCell className="text-muted font-medium">Total Shots (A / B)</TableCell>
                  <TableCell className="font-mono">{report.shots.a} / {report.shots.b}</TableCell>
                </TableRow>
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
            {report.warnings.map((warning, idx) => (
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
