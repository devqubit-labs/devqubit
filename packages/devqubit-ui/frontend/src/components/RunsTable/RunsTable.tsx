/**
 * DevQubit UI Runs Table Component
 *
 * Displays a list of runs with status, duration, and delete functionality.
 * Running runs show a live-updating duration timer.
 */

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Table, TableHead, TableBody, TableRow, TableHeader, TableCell } from '../Table';
import { Badge, Button, EmptyState, Modal, Spinner } from '../ui';
import { shortId, timeAgo, elapsedSeconds, formatDuration, isTerminalStatus } from '../../utils';
import type { RunSummary, RunStatus } from '../../types';

export interface RunsTableProps {
  runs: RunSummary[];
  onDelete?: (runId: string) => void;
  loading?: boolean;
  emptyHint?: string;
  baselineRunId?: string;
}

function StatusBadge({ status }: { status: RunStatus }) {
  const variant = {
    FINISHED: 'success',
    FAILED: 'danger',
    KILLED: 'gray',
    RUNNING: 'warning',
    UNKNOWN: 'gray',
  }[status] as 'success' | 'danger' | 'warning' | 'gray';
  return <Badge variant={variant}>{status}</Badge>;
}

/**
 * Live-updating duration display.
 *
 * For terminal runs, renders a static formatted duration.
 * For RUNNING runs, ticks every second to show elapsed time.
 */
function Duration({ createdAt, endedAt, status }: {
  createdAt: string;
  endedAt?: string | null;
  status: RunStatus;
}) {
  const isRunning = !isTerminalStatus(status);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [isRunning]);

  // For running: compute against current time; for terminal: use ended_at
  const end = isRunning ? undefined : endedAt;
  const seconds = elapsedSeconds(createdAt, end);

  // Override formatDuration for running to use fresh `now`
  const displaySeconds = isRunning
    ? Math.max(0, Math.floor((now - new Date(createdAt).getTime()) / 1000))
    : seconds;

  return (
    <span className={isRunning ? 'tabular-nums' : undefined}>
      {formatDuration(displaySeconds)}
    </span>
  );
}

export function RunsTable({ runs, onDelete, loading, emptyHint, baselineRunId }: RunsTableProps) {
  const [deleteTarget, setDeleteTarget] = useState<RunSummary | null>(null);

  const handleConfirmDelete = () => {
    if (deleteTarget && onDelete) {
      onDelete(deleteTarget.run_id);
      setDeleteTarget(null);
    }
  };

  if (!runs.length) {
    return (
      <EmptyState
        message="No runs found"
        hint={emptyHint ?? 'Try adjusting your filters'}
      />
    );
  }

  return (
    <>
      <Table>
        <TableHead>
          <TableRow>
            <TableHeader>Run ID</TableHeader>
            <TableHeader>Name</TableHeader>
            <TableHeader>Project</TableHeader>
            <TableHeader>Status</TableHeader>
            <TableHeader>Duration</TableHeader>
            <TableHeader>Created</TableHeader>
            <TableHeader>Actions</TableHeader>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.run_id}>
              <TableCell>
                <Link to={`/runs/${run.run_id}`} className="font-mono">
                  {shortId(run.run_id)}
                </Link>
                {run.run_id === baselineRunId && (
                  <Badge variant="info" className="ml-2">Baseline</Badge>
                )}
              </TableCell>
              <TableCell>{run.run_name || '—'}</TableCell>
              <TableCell>{run.project}</TableCell>
              <TableCell>
                <StatusBadge status={run.status} />
              </TableCell>
              <TableCell className="font-mono text-sm">
                <Duration
                  createdAt={run.created_at}
                  endedAt={run.ended_at}
                  status={run.status}
                />
              </TableCell>
              <TableCell className="text-muted">{timeAgo(run.created_at)}</TableCell>
              <TableCell>
                <div className="flex gap-2">
                  <Link to={`/runs/${run.run_id}`}>
                    <Button variant="secondary" size="sm">View</Button>
                  </Link>
                  {onDelete && (
                    <Button
                      variant="ghost-danger"
                      size="sm"
                      onClick={() => setDeleteTarget(run)}
                      disabled={loading}
                    >
                      Delete
                    </Button>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Modal
        open={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        title="Delete Run"
        actions={
          <>
            <Button variant="secondary" onClick={() => setDeleteTarget(null)}>Cancel</Button>
            <Button variant="danger" onClick={handleConfirmDelete} disabled={loading}>
              {loading && <Spinner />}
              Delete
            </Button>
          </>
        }
      >
        <p>Are you sure you want to delete this run?</p>
        {deleteTarget && (
          <p className="font-mono text-sm mt-2">{shortId(deleteTarget.run_id)}</p>
        )}
        <p className="text-sm text-danger mt-2">This action cannot be undone.</p>
      </Modal>
    </>
  );
}

export { StatusBadge };
