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

/* =========================================================================
   Icons
   ========================================================================= */

interface IconProps {
  className?: string;
}

function CheckIcon({ className = '' }: IconProps) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function XIcon({ className = '' }: IconProps) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function AlertTriangleIcon({ className = '' }: IconProps) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function InfoIcon({ className = '' }: IconProps) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

/* =========================================================================
   Diff Components
   ========================================================================= */

function DiffIndicator({ match }: { match: boolean }) {
  if (match) {
    return (
      <span className="diff-match inline-flex items-center gap-1.5">
        <CheckIcon className="flex-shrink-0" />
        <span className="font-medium">Match</span>
      </span>
    );
  }
  return (
    <span className="diff-mismatch inline-flex items-center gap-1.5">
      <XIcon className="flex-shrink-0" />
      <span className="font-medium">Different</span>
    </span>
  );
}

/* =========================================================================
   Warnings Box
   ========================================================================= */

function WarningsBox({ warnings }: { warnings: string[] }) {
  if (!warnings || warnings.length === 0) return null;

  return (
    <div className="alert alert-warning mb-4">
      <div className="flex gap-3">
        <AlertTriangleIcon className="flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="font-semibold mb-1">
            {warnings.length === 1 ? 'Warning' : `${warnings.length} Warnings`}
          </p>
          <ul className="space-y-1">
            {warnings.map((warning, idx) => (
              <li key={idx} className="text-sm">{warning}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* =========================================================================
   Run Header Cards
   ========================================================================= */

function RunCard({
  label,
  run,
  variant,
}: {
  label: string;
  run: RunSummary;
  variant: 'a' | 'b';
}) {
  return (
    <div className="relative">
      <div className={`absolute top-0 left-0 w-1 h-full rounded-l ${variant === 'a' ? 'bg-info' : 'bg-primary'}`} />
      <div className="pl-4">
        <p className={`text-xs font-semibold uppercase tracking-wider mb-2 ${variant === 'a' ? 'text-info' : 'text-primary'}`}>
          {label}
        </p>
        <Link to={`/runs/${run.run_id}`} className="text-base font-medium hover:underline">
          {run.run_name || 'Unnamed Run'}
        </Link>
        <p className="font-mono text-xs mt-1 text-muted">{shortId(run.run_id)}</p>
        <p className="text-sm mt-1 text-muted">{run.project} · {timeAgo(run.created_at)}</p>
      </div>
    </div>
  );
}

/* =========================================================================
   Section Header
   ========================================================================= */

function SectionTitle({
  children,
  badge,
}: {
  children: React.ReactNode;
  badge?: { variant: 'success' | 'warning' | 'danger' | 'info' | 'gray'; label: string };
}) {
  return (
    <CardHeader>
      <CardTitle>
        <span className="flex items-center gap-2 flex-wrap">
          {children}
          {badge && <Badge variant={badge.variant}>{badge.label}</Badge>}
        </span>
      </CardTitle>
    </CardHeader>
  );
}

/* =========================================================================
   Statistical Interpretation
   ========================================================================= */

function StatisticalResult({
  tvd,
  noiseContext,
}: {
  tvd: number;
  noiseContext?: {
    p_value?: number;
    noise_ratio?: number;
    noise_p95?: number;
  };
}) {
  if (tvd === 0) {
    return (
      <p className="text-sm mt-4 text-success">
        ✓ Distributions are identical (TVD = 0)
      </p>
    );
  }

  if (!noiseContext) return null;

  const { p_value, noise_ratio } = noiseContext;

  if (p_value != null) {
    if (p_value >= 0.10) {
      return (
        <p className="text-sm mt-4 text-success">
          ✓ Consistent with sampling noise — difference is not statistically significant
        </p>
      );
    }
    if (p_value >= 0.05) {
      return (
        <p className="text-sm mt-4 text-warning">
          ⚠ Borderline (p={p_value.toFixed(2)}) — consider increasing shots
        </p>
      );
    }
    return (
      <p className="text-sm mt-4 text-danger">
        ✗ Statistically significant difference (p={p_value.toFixed(2)}) — results show meaningful divergence
      </p>
    );
  }

  if (noise_ratio != null) {
    if (noise_ratio < 1.5) {
      return (
        <p className="text-sm mt-4 text-success">
          ✓ TVD is within expected shot noise range
        </p>
      );
    }
    if (noise_ratio < 3.0) {
      return (
        <p className="text-sm mt-4 text-warning">
          ⚠ Ambiguous ({noise_ratio.toFixed(1)}× expected noise) — consider increasing shots
        </p>
      );
    }
    return (
      <p className="text-sm mt-4 text-danger">
        ✗ TVD exceeds expected noise ({noise_ratio.toFixed(1)}×) — results show meaningful differences
      </p>
    );
  }

  return null;
}

/* =========================================================================
   Diff Select Form
   ========================================================================= */

function DiffSelect() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: runsData } = useRuns({ limit: 100 });
  const runs = runsData?.runs ?? [];
  const [runA, setRunA] = useState(searchParams.get('a') || '');
  const [runB, setRunB] = useState(searchParams.get('b') || '');
  const [validationError, setValidationError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!runA || !runB) {
      setValidationError('Please select both runs to compare');
      return;
    }
    if (runA === runB) {
      setValidationError('Please select two different runs');
      return;
    }
    setValidationError('');
    setSearchParams({ a: runA, b: runB });
  };

  return (
    <>
      <Card>
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <FormGroup>
              <Label htmlFor="a">Run A (Baseline)</Label>
              <Select
                id="a"
                value={runA}
                onChange={(e) => { setRunA(e.target.value); setValidationError(''); }}
              >
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
              <Select
                id="b"
                value={runB}
                onChange={(e) => { setRunB(e.target.value); setValidationError(''); }}
              >
                <option value="">Select run...</option>
                {runs.map((run: RunSummary) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_name || 'Unnamed'} ({shortId(run.run_id)}) — {run.project}
                  </option>
                ))}
              </Select>
            </FormGroup>
          </div>
          {validationError && (
            <p className="text-sm mb-3 text-danger">{validationError}</p>
          )}
          <Button type="submit" variant="primary">Compare</Button>
        </form>
      </Card>

      <Card className="mt-4">
        <CardHeader><CardTitle>How it works</CardTitle></CardHeader>
        <div className="space-y-2 text-sm text-muted">
          <div className="flex items-start gap-2">
            <InfoIcon className="flex-shrink-0 mt-0.5" />
            <span>Select two runs to compare their parameters, metrics, and artifacts</span>
          </div>
          <div className="flex items-start gap-2">
            <InfoIcon className="flex-shrink-0 mt-0.5" />
            <span>The diff shows changed values and computes TVD for result distributions</span>
          </div>
          <div className="flex items-start gap-2">
            <InfoIcon className="flex-shrink-0 mt-0.5" />
            <span>You can also compare from the run detail page</span>
          </div>
        </div>
      </Card>
    </>
  );
}

/* =========================================================================
   Diff Result
   ========================================================================= */

function DiffResult({ runIdA, runIdB }: { runIdA: string; runIdB: string }) {
  const { data, loading, error } = useDiff(runIdA, runIdB);

  if (loading) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center py-12 gap-3">
          <Spinner />
          <p className="text-sm text-muted">Comparing runs...</p>
        </div>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <EmptyState message="Failed to load diff" hint={error?.message} />
      </Card>
    );
  }

  const { run_a, run_b, report } = data;

  // Device calibration badge
  const deviceBadge = report.device_drift?.significant_drift
    ? { variant: 'warning' as const, label: 'Drift Detected' }
    : report.device_drift?.has_calibration_data
      ? { variant: 'success' as const, label: 'Stable' }
      : { variant: 'gray' as const, label: 'No Data' };

  return (
    <>
      {/* Warnings */}
      <WarningsBox warnings={report.warnings || []} />

      {/* Run Headers */}
      <Card className="mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <RunCard label="Run A (Baseline)" run={run_a} variant="a" />
          <RunCard label="Run B (Candidate)" run={run_b} variant="b" />
        </div>
      </Card>

      {/* Metadata & Fingerprints */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Card>
          <SectionTitle>Metadata</SectionTitle>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell>Project</TableCell>
                <TableCell><DiffIndicator match={report.metadata.project_match} /></TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Backend</TableCell>
                <TableCell><DiffIndicator match={report.metadata.backend_match} /></TableCell>
              </TableRow>
              {!report.metadata.project_match && (
                <>
                  <TableRow>
                    <TableCell className="text-sm text-muted">Project A</TableCell>
                    <TableCell className="font-mono text-sm">{report.metadata.project_a || 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="text-sm text-muted">Project B</TableCell>
                    <TableCell className="font-mono text-sm">{report.metadata.project_b || 'N/A'}</TableCell>
                  </TableRow>
                </>
              )}
              {!report.metadata.backend_match && (
                <>
                  <TableRow>
                    <TableCell className="text-sm text-muted">Backend A</TableCell>
                    <TableCell className="font-mono text-sm">{report.metadata.backend_a || 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="text-sm text-muted">Backend B</TableCell>
                    <TableCell className="font-mono text-sm">{report.metadata.backend_b || 'N/A'}</TableCell>
                  </TableRow>
                </>
              )}
            </TableBody>
          </Table>
        </Card>

        <Card>
          <SectionTitle>Fingerprints</SectionTitle>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell>Run A</TableCell>
                <TableCell className="font-mono text-sm">{shortDigest(report.fingerprints.a)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Run B</TableCell>
                <TableCell className="font-mono text-sm">{shortDigest(report.fingerprints.b)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Match</TableCell>
                <TableCell>
                  <DiffIndicator match={report.fingerprints.a === report.fingerprints.b} />
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </Card>
      </div>

      {/* Program & Device */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <Card>
          <SectionTitle
            badge={{
              variant: report.program.exact_match ? 'success' : report.program.structural_match ? 'info' : 'warning',
              label: report.program.exact_match ? 'Exact Match' : report.program.structural_match ? 'Structural Match' : 'Different',
            }}
          >
            Program
          </SectionTitle>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell>Exact Match</TableCell>
                <TableCell><DiffIndicator match={report.program.exact_match} /></TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Structural Match</TableCell>
                <TableCell><DiffIndicator match={report.program.structural_match} /></TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </Card>

        <Card>
          <SectionTitle badge={deviceBadge}>Device Calibration</SectionTitle>
          {report.device_drift?.significant_drift ? (
            <p className="text-sm text-warning">Significant calibration drift detected between runs</p>
          ) : report.device_drift?.has_calibration_data ? (
            <p className="text-muted">Calibration within acceptable thresholds</p>
          ) : (
            <p className="text-muted">No calibration data available</p>
          )}
        </Card>
      </div>

      {/* Parameters */}
      <Card className="mb-4">
        <SectionTitle badge={{ variant: report.params.match ? 'success' : 'warning', label: report.params.match ? 'Match' : 'Different' }}>
          Parameters
        </SectionTitle>
        {report.params.match ? (
          <p className="text-muted">All parameters match</p>
        ) : (
          <>
            {report.params.changed && Object.keys(report.params.changed).length > 0 && (
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
            )}
          </>
        )}
      </Card>

      {/* Metrics */}
      <Card className="mb-4">
        <SectionTitle badge={{ variant: report.metrics.match ? 'success' : 'warning', label: report.metrics.match ? 'Match' : 'Different' }}>
          Metrics
        </SectionTitle>
        {report.metrics.match ? (
          <p className="text-muted">All metrics match</p>
        ) : (
          <>
            {report.metrics.changed && Object.keys(report.metrics.changed).length > 0 && (
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
            )}
          </>
        )}
      </Card>

      {/* Circuit Diff */}
      {report.circuit_diff && (
        <Card className="mb-4">
          <SectionTitle badge={{ variant: report.circuit_diff.match ? 'success' : 'warning', label: report.circuit_diff.match ? 'Match' : 'Different' }}>
            Circuit
          </SectionTitle>

          {report.circuit_diff.match ? (
            <p className="text-muted">Circuit structure matches</p>
          ) : (
            <div className="space-y-4">
              {report.circuit_diff.changed && Object.keys(report.circuit_diff.changed).length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 text-muted">Changed Properties</h4>
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
                              <span className={values.delta > 0 ? 'text-danger' : 'text-success'}>
                                {values.delta > 0 ? '+' : ''}{values.delta}
                                {values.pct != null && ` (${values.pct > 0 ? '+' : ''}${values.pct.toFixed(1)}%)`}
                              </span>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}

              {report.circuit_diff.is_clifford_changed && (
                <div>
                  <h4 className="text-sm font-medium mb-2 text-muted">Clifford Status</h4>
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell>Run A</TableCell>
                        <TableCell className="font-mono">
                          {report.circuit_diff.is_clifford_a != null ? String(report.circuit_diff.is_clifford_a) : 'unknown'}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Run B</TableCell>
                        <TableCell className="font-mono">
                          {report.circuit_diff.is_clifford_b != null ? String(report.circuit_diff.is_clifford_b) : 'unknown'}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              )}

              {report.circuit_diff.added_gates && report.circuit_diff.added_gates.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 text-muted">New Gate Types (in B)</h4>
                  <div className="flex flex-wrap gap-1">
                    {report.circuit_diff.added_gates.map((gate) => (
                      <Badge key={gate} variant="success">{gate}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {report.circuit_diff.removed_gates && report.circuit_diff.removed_gates.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 text-muted">Removed Gate Types (from A)</h4>
                  <div className="flex flex-wrap gap-1">
                    {report.circuit_diff.removed_gates.map((gate) => (
                      <Badge key={gate} variant="danger">{gate}</Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {/* TVD / Results */}
      {report.tvd != null && (
        <Card className="mb-4">
          <SectionTitle>Results Distribution</SectionTitle>
          <Table>
            <TableBody>
              <TableRow>
                <TableCell>Total Variation Distance (TVD)</TableCell>
                <TableCell className="font-mono font-medium">{report.tvd.toFixed(6)}</TableCell>
              </TableRow>
              {report.shots && (
                <TableRow>
                  <TableCell>Total Shots (A / B)</TableCell>
                  <TableCell className="font-mono">{report.shots.a} / {report.shots.b}</TableCell>
                </TableRow>
              )}
              {report.noise_context?.noise_p95 && (
                <TableRow>
                  <TableCell>Noise Threshold (p95)</TableCell>
                  <TableCell className="font-mono">{report.noise_context.noise_p95.toFixed(6)}</TableCell>
                </TableRow>
              )}
              {report.noise_context?.p_value != null && (
                <TableRow>
                  <TableCell>p-value</TableCell>
                  <TableCell className="font-mono">{report.noise_context.p_value.toFixed(4)}</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          <StatisticalResult tvd={report.tvd} noiseContext={report.noise_context} />
        </Card>
      )}
    </>
  );
}

/* =========================================================================
   Main Page Component
   ========================================================================= */

export function DiffPage() {
  const [searchParams] = useSearchParams();
  const runIdA = searchParams.get('a');
  const runIdB = searchParams.get('b');
  const hasBothRuns = runIdA && runIdB;

  return (
    <Layout>
      <div className="page-header">
        <div>
          <h1 className="page-title">Compare Runs</h1>
          {hasBothRuns && (
            <p className="text-sm mt-1 text-muted">
              <Link to="/diff">← Select different runs</Link>
            </p>
          )}
        </div>
      </div>
      {hasBothRuns ? <DiffResult runIdA={runIdA} runIdB={runIdB} /> : <DiffSelect />}
    </Layout>
  );
}
