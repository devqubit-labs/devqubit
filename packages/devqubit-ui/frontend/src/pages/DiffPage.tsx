/**
 * Diff Page - Run comparison
 */

import { useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { Layout } from '../components/Layout';
import {
  Card, CardHeader, CardTitle, Badge, Button, Spinner, EmptyState,
  FormGroup, Label, Select, Table, TableHead, TableBody, TableRow, TableHeader, TableCell,
} from '../components';
import { useDiff, useRuns } from '../hooks';
import { shortId, shortDigest, timeAgo } from '../utils';
import type { RunSummary } from '../types';

function DiffCell({ match, yesText = '✓ Match', noText = '✗ Different' }: { match: boolean; yesText?: string; noText?: string }) {
  return <span className={match ? 'diff-match' : 'diff-mismatch'}>{match ? yesText : noText}</span>;
}

function DiffSelect() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: runsData } = useRuns({ limit: 100 });
  const runs = runsData?.runs ?? [];
  const [runA, setRunA] = useState(searchParams.get('a') || '');
  const [runB, setRunB] = useState(searchParams.get('b') || '');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (runA && runB) setSearchParams({ a: runA, b: runB });
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
                    {run.run_name || 'Unnamed'} ({shortId(run.run_id)}) — {run.project}
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
                    {run.run_name || 'Unnamed'} ({shortId(run.run_id)}) — {run.project}
                  </option>
                ))}
              </Select>
            </FormGroup>
          </div>
          <Button type="submit" variant="primary" disabled={!runA || !runB}>Compare</Button>
        </form>
      </Card>

      <Card className="mt-4">
        <CardHeader><CardTitle>Tips</CardTitle></CardHeader>
        <ul className="text-muted text-sm list-disc pl-6 space-y-1">
          <li>Select two runs to compare their parameters, metrics, and artifacts</li>
          <li>The diff will show changed values and compute TVD for result distributions</li>
          <li>You can also compare from the run detail page</li>
        </ul>
      </Card>
    </>
  );
}

interface CircuitDiff {
  match: boolean;
  changed?: Record<string, { label?: string; a: unknown; b: unknown; delta?: number; pct?: number }>;
  is_clifford_changed?: boolean;
  is_clifford_a?: boolean;
  is_clifford_b?: boolean;
  added_gates?: string[];
  removed_gates?: string[];
}

interface DiffReport {
  identical: boolean;
  metadata: { project_match: boolean; backend_match: boolean; project_a?: string; project_b?: string; backend_a?: string; backend_b?: string };
  fingerprints: { a: string; b: string };
  program: { exact_match: boolean; structural_match: boolean };
  device_drift?: { significant_drift: boolean; has_calibration_data: boolean };
  params: { match: boolean; changed?: Record<string, { a: unknown; b: unknown }> };
  metrics: { match: boolean; changed?: Record<string, { a: number; b: number }> };
  circuit_diff?: CircuitDiff;
  tvd?: number;
  shots?: { a: number; b: number };
  noise_context?: { noise_p95?: number; expected_noise?: number; p_value?: number; noise_ratio?: number };
  warnings?: string[];
}

function DiffResult({ runIdA, runIdB }: { runIdA: string; runIdB: string }) {
  const { data, loading, error } = useDiff(runIdA, runIdB);

  if (loading) return <div className="flex justify-center py-12"><Spinner /></div>;
  if (error || !data) return <Card><EmptyState message="Failed to load diff" hint={error?.message} /></Card>;

  const { run_a, run_b, report } = data as { run_a: RunSummary; run_b: RunSummary; report: DiffReport };

  return (
    <>
      {/* Warnings - show at top if any */}
      {report.warnings && report.warnings.length > 0 && (
        <div className="alert alert-warning mb-4">
          <strong>Warnings:</strong>
          <ul className="list-disc pl-6 mt-1">
            {report.warnings.map((warning, idx) => (
              <li key={idx}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Run Headers */}
      <Card className="mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm text-muted uppercase tracking-wider mb-1">Run A (Baseline)</h3>
            <p><Link to={`/runs/${run_a.run_id}`}>{run_a.run_name || 'Unnamed Run'}</Link></p>
            <p className="font-mono text-sm text-muted">{shortId(run_a.run_id)}</p>
            <p className="text-muted text-sm">{run_a.project} · {timeAgo(run_a.created_at)}</p>
          </div>
          <div>
            <h3 className="text-sm text-muted uppercase tracking-wider mb-1">Run B (Candidate)</h3>
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
              <TableRow><TableCell>Project</TableCell><TableCell><DiffCell match={report.metadata.project_match} /></TableCell></TableRow>
              <TableRow><TableCell>Backend</TableCell><TableCell><DiffCell match={report.metadata.backend_match} /></TableCell></TableRow>
              {!report.metadata.project_match && (
                <>
                  <TableRow><TableCell className="text-muted text-sm">Project A</TableCell><TableCell className="font-mono text-sm">{report.metadata.project_a || 'N/A'}</TableCell></TableRow>
                  <TableRow><TableCell className="text-muted text-sm">Project B</TableCell><TableCell className="font-mono text-sm">{report.metadata.project_b || 'N/A'}</TableCell></TableRow>
                </>
              )}
              {!report.metadata.backend_match && (
                <>
                  <TableRow><TableCell className="text-muted text-sm">Backend A</TableCell><TableCell className="font-mono text-sm">{report.metadata.backend_a || 'N/A'}</TableCell></TableRow>
                  <TableRow><TableCell className="text-muted text-sm">Backend B</TableCell><TableCell className="font-mono text-sm">{report.metadata.backend_b || 'N/A'}</TableCell></TableRow>
                </>
              )}
            </TableBody>
          </Table>
        </Card>

        <Card>
          <CardHeader><CardTitle>Fingerprints</CardTitle></CardHeader>
          <Table>
            <TableBody>
              <TableRow><TableCell>Run A</TableCell><TableCell className="font-mono text-sm">{shortDigest(report.fingerprints.a)}</TableCell></TableRow>
              <TableRow><TableCell>Run B</TableCell><TableCell className="font-mono text-sm">{shortDigest(report.fingerprints.b)}</TableCell></TableRow>
              <TableRow><TableCell>Match</TableCell><TableCell><DiffCell match={report.fingerprints.a === report.fingerprints.b} yesText="✓ Yes" noText="✗ No" /></TableCell></TableRow>
            </TableBody>
          </Table>
        </Card>
      </div>

      {/* Program & Device */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Card>
          <CardHeader>
            <CardTitle>
              Program
              {report.program.exact_match ? <Badge variant="success">Exact Match</Badge> :
               report.program.structural_match ? <Badge variant="info">Structural Match</Badge> :
               <Badge variant="warning">Different</Badge>}
            </CardTitle>
          </CardHeader>
          <Table>
            <TableBody>
              <TableRow><TableCell>Exact Match</TableCell><TableCell><DiffCell match={report.program.exact_match} yesText="✓ Yes" noText="✗ No" /></TableCell></TableRow>
              <TableRow><TableCell>Structural Match</TableCell><TableCell><DiffCell match={report.program.structural_match} yesText="✓ Yes" noText="✗ No" /></TableCell></TableRow>
            </TableBody>
          </Table>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>
              Device Calibration
              {report.device_drift?.significant_drift ? <Badge variant="warning">Drifted</Badge> :
               report.device_drift?.has_calibration_data ? <Badge variant="success">Stable</Badge> :
               <Badge variant="gray">N/A</Badge>}
            </CardTitle>
          </CardHeader>
          {report.device_drift?.significant_drift ? (
            <p className="text-sm text-warning">⚠ Significant calibration drift detected</p>
          ) : report.device_drift?.has_calibration_data ? (
            <p className="text-muted">Calibration within acceptable thresholds</p>
          ) : (
            <p className="text-muted">No calibration data available</p>
          )}
        </Card>
      </div>

      {/* Parameters */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Parameters <Badge variant={report.params.match ? 'success' : 'warning'}>{report.params.match ? 'Match' : 'Different'}</Badge></CardTitle>
        </CardHeader>
        {report.params.match ? <p className="text-muted">All parameters match</p> : (
          <>
            {report.params.changed && Object.keys(report.params.changed).length > 0 && (
              <Table>
                <TableHead><TableRow><TableHeader>Parameter</TableHeader><TableHeader>Run A</TableHeader><TableHeader>Run B</TableHeader></TableRow></TableHead>
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
            )}
          </>
        )}
      </Card>

      {/* Metrics */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Metrics <Badge variant={report.metrics.match ? 'success' : 'warning'}>{report.metrics.match ? 'Match' : 'Different'}</Badge></CardTitle>
        </CardHeader>
        {report.metrics.match ? <p className="text-muted">All metrics match</p> : (
          <>
            {report.metrics.changed && Object.keys(report.metrics.changed).length > 0 && (
              <Table>
                <TableHead><TableRow><TableHeader>Metric</TableHeader><TableHeader>Run A</TableHeader><TableHeader>Run B</TableHeader></TableRow></TableHead>
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
            )}
          </>
        )}
      </Card>

      {/* Circuit Diff */}
      {report.circuit_diff && (
        <Card className="mb-4">
          <CardHeader>
            <CardTitle>
              Circuit
              <Badge variant={report.circuit_diff.match ? 'success' : 'warning'}>
                {report.circuit_diff.match ? 'Match' : 'Different'}
              </Badge>
            </CardTitle>
          </CardHeader>

          {report.circuit_diff.match ? (
            <p className="text-muted">Circuit structure matches</p>
          ) : (
            <>
              {/* Changed properties */}
              {report.circuit_diff.changed && Object.keys(report.circuit_diff.changed).length > 0 && (
                <>
                  <h4 className="text-sm text-muted mb-2">Changed</h4>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableHeader>Property</TableHeader>
                        <TableHeader>Run A</TableHeader>
                        <TableHeader>Run B</TableHeader>
                        <TableHeader>Delta</TableHeader>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(report.circuit_diff.changed).map(([key, values]) => (
                        <TableRow key={key}>
                          <TableCell>{values.label || key}</TableCell>
                          <TableCell className="font-mono">{String(values.a)}</TableCell>
                          <TableCell className="font-mono">{String(values.b)}</TableCell>
                          <TableCell className="font-mono">
                            {values.delta != null && (
                              <>
                                {values.delta > 0 ? '+' : ''}{values.delta}
                                {values.pct != null && ` (${values.pct > 0 ? '+' : ''}${values.pct.toFixed(1)}%)`}
                              </>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </>
              )}

              {/* Clifford status */}
              {report.circuit_diff.is_clifford_changed && (
                <div className="mt-4">
                  <h4 className="text-sm text-muted mb-2">Clifford Status</h4>
                  <Table>
                    <TableBody>
                      <TableRow><TableCell>Run A</TableCell><TableCell className="font-mono">{report.circuit_diff.is_clifford_a != null ? String(report.circuit_diff.is_clifford_a) : 'unknown'}</TableCell></TableRow>
                      <TableRow><TableCell>Run B</TableCell><TableCell className="font-mono">{report.circuit_diff.is_clifford_b != null ? String(report.circuit_diff.is_clifford_b) : 'unknown'}</TableCell></TableRow>
                    </TableBody>
                  </Table>
                </div>
              )}

              {/* Added gates */}
              {report.circuit_diff.added_gates && report.circuit_diff.added_gates.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm text-muted mb-2">New Gate Types (in B)</h4>
                  <p className="font-mono text-sm">{report.circuit_diff.added_gates.join(', ')}</p>
                </div>
              )}

              {/* Removed gates */}
              {report.circuit_diff.removed_gates && report.circuit_diff.removed_gates.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm text-muted mb-2">Removed Gate Types (from A)</h4>
                  <p className="font-mono text-sm">{report.circuit_diff.removed_gates.join(', ')}</p>
                </div>
              )}
            </>
          )}
        </Card>
      )}

      {/* TVD / Results */}
      {report.tvd != null && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Results</CardTitle></CardHeader>
          <Table>
            <TableBody>
              <TableRow><TableCell>Total Variation Distance (TVD)</TableCell><TableCell className="font-mono">{report.tvd.toFixed(6)}</TableCell></TableRow>
              {report.shots && <TableRow><TableCell>Total Shots (A / B)</TableCell><TableCell className="font-mono">{report.shots.a} / {report.shots.b}</TableCell></TableRow>}
              {report.noise_context?.noise_p95 && (
                <TableRow><TableCell>Noise Threshold (p95)</TableCell><TableCell className="font-mono">{report.noise_context.noise_p95.toFixed(6)}</TableCell></TableRow>
              )}
              {report.noise_context?.p_value != null && (
                <TableRow><TableCell>p-value</TableCell><TableCell className="font-mono">{report.noise_context.p_value.toFixed(3)}</TableCell></TableRow>
              )}
            </TableBody>
          </Table>

          {/* Statistical interpretation */}
          {report.tvd > 0 && report.noise_context && (
            <p className="text-sm mt-4">
              {report.noise_context.p_value != null ? (
                report.noise_context.p_value >= 0.10 ? (
                  <span className="text-success">✓ Consistent with sampling noise — difference is not statistically significant.</span>
                ) : report.noise_context.p_value >= 0.05 ? (
                  <span className="text-warning">⚠ Borderline (p={report.noise_context.p_value.toFixed(2)}). Consider increasing shots.</span>
                ) : (
                  <span className="text-danger">✗ Statistically significant difference (p={report.noise_context.p_value.toFixed(2)}) — results show meaningful divergence.</span>
                )
              ) : report.noise_context.noise_ratio != null ? (
                report.noise_context.noise_ratio < 1.5 ? (
                  <span className="text-success">✓ TVD is within expected shot noise range.</span>
                ) : report.noise_context.noise_ratio < 3.0 ? (
                  <span className="text-warning">⚠ Ambiguous ({report.noise_context.noise_ratio.toFixed(1)}× expected noise). Consider increasing shots.</span>
                ) : (
                  <span className="text-danger">✗ TVD exceeds expected noise ({report.noise_context.noise_ratio.toFixed(1)}×) — results show meaningful differences.</span>
                )
              ) : null}
            </p>
          )}
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
      <div className="page-header">
        <div>
          <h1 className="page-title">
            Compare Runs
            {hasBothRuns && <Badge variant="info">Comparing</Badge>}
          </h1>
          {hasBothRuns && <p className="text-muted text-sm"><Link to="/diff">← Select different runs</Link></p>}
        </div>
      </div>
      {hasBothRuns ? <DiffResult runIdA={runIdA} runIdB={runIdB} /> : <DiffSelect />}
    </Layout>
  );
}
