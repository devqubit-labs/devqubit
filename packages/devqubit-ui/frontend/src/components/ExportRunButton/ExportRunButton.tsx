/**
 * ExportRunButton Component
 *
 * Provides run export functionality (bundle creation) for devqubit-ui.
 * Uses the ApiClient from AppProvider so that Hub's CSRF / auth headers
 * are included automatically.
 */

import { useState } from 'react';
import { Button, Spinner, Modal, Toast, Badge } from '../ui';
import { useApp } from '../../hooks';

export interface ExportRunButtonProps {
  /** Run ID to export */
  runId: string;
  /** Optional run name for display */
  runName?: string;
  /** Button variant */
  variant?: 'primary' | 'secondary' | 'ghost';
  /** Button size */
  size?: 'default' | 'sm';
  /** Additional class name */
  className?: string;
}

export interface ExportProgress {
  status: 'idle' | 'preparing' | 'packing' | 'downloading' | 'complete' | 'error';
  message?: string;
  progress?: number;
  artifactCount?: number;
  objectCount?: number;
  bundleSize?: number;
}

/**
 * Button that triggers run export/bundle download.
 */
export function ExportRunButton({
  runId,
  runName,
  variant = 'secondary',
  size = 'sm',
  className = '',
}: ExportRunButtonProps) {
  const { api } = useApp();
  const [progress, setProgress] = useState<ExportProgress>({ status: 'idle' });
  const [showModal, setShowModal] = useState(false);
  const [toast, setToast] = useState<{ message: string; variant: 'success' | 'error' } | null>(null);

  const handleExport = async () => {
    setProgress({ status: 'preparing', message: 'Preparing bundle...' });
    setShowModal(true);

    try {
      // Create bundle via ApiClient (includes CSRF / auth headers)
      const result = await api.createExport(runId);

      setProgress({
        status: 'packing',
        message: 'Creating bundle...',
        artifactCount: result.artifact_count,
        objectCount: result.object_count,
      });

      // Download the ZIP blob
      setProgress(prev => ({
        ...prev,
        status: 'downloading',
        message: 'Downloading bundle...',
      }));

      const downloadUrl = api.getExportDownloadUrl(runId);
      const downloadResponse = await fetch(downloadUrl, {
        credentials: 'same-origin',
      });

      if (!downloadResponse.ok) {
        throw new Error('Download failed');
      }

      const blob = await downloadResponse.blob();
      const filename = runName
        ? `${runName.replace(/[^a-zA-Z0-9_-]/g, '_')}_${runId.slice(0, 8)}.zip`
        : `run_${runId.slice(0, 8)}.zip`;

      // Trigger browser download
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setProgress({
        status: 'complete',
        message: 'Export complete!',
        bundleSize: blob.size,
        artifactCount: result.artifact_count,
        objectCount: result.object_count,
      });

      setToast({ message: 'Bundle downloaded successfully', variant: 'success' });

      // Auto-close modal after success
      setTimeout(() => {
        setShowModal(false);
        setProgress({ status: 'idle' });
      }, 2000);

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Export failed';
      setProgress({ status: 'error', message });
      setToast({ message, variant: 'error' });
    }
  };

  const handleClose = () => {
    if (progress.status !== 'preparing' && progress.status !== 'packing' && progress.status !== 'downloading') {
      setShowModal(false);
      setProgress({ status: 'idle' });
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isLoading = ['preparing', 'packing', 'downloading'].includes(progress.status);

  return (
    <>
      <Button
        variant={variant}
        size={size}
        onClick={handleExport}
        disabled={isLoading}
        className={className}
      >
        {isLoading && <Spinner />}
        <ExportIcon />
        Export
      </Button>

      <Modal
        open={showModal}
        onClose={handleClose}
        title="Export Run"
        actions={
          progress.status === 'error' || progress.status === 'complete' ? (
            <Button variant="secondary" onClick={handleClose}>
              Close
            </Button>
          ) : undefined
        }
      >
        <div className="py-4">
          {/* Status indicator */}
          <div className="flex items-center gap-3 mb-4">
            {progress.status === 'error' ? (
              <div className="w-10 h-10 rounded-full bg-[var(--dq-danger-bg)] flex items-center justify-center">
                <ErrorIcon />
              </div>
            ) : progress.status === 'complete' ? (
              <div className="w-10 h-10 rounded-full bg-[var(--dq-success-bg)] flex items-center justify-center">
                <CheckIcon />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-full bg-[var(--dq-primary-bg)] flex items-center justify-center">
                <Spinner />
              </div>
            )}
            <div>
              <p className="font-medium">{progress.message}</p>
              {progress.status === 'error' && (
                <p className="text-sm text-[var(--dq-text-muted)]">
                  Please try again or contact support.
                </p>
              )}
            </div>
          </div>

          {/* Stats */}
          {(progress.artifactCount !== undefined || progress.objectCount !== undefined || progress.bundleSize !== undefined) && (
            <div className="flex flex-wrap gap-2 mt-4">
              {progress.artifactCount !== undefined && (
                <Badge variant="gray">
                  {progress.artifactCount} artifacts
                </Badge>
              )}
              {progress.objectCount !== undefined && (
                <Badge variant="gray">
                  {progress.objectCount} objects
                </Badge>
              )}
              {progress.bundleSize !== undefined && (
                <Badge variant="info">
                  {formatBytes(progress.bundleSize)}
                </Badge>
              )}
            </div>
          )}

          {/* Run info */}
          <div className="mt-4 pt-4 border-t border-[var(--dq-border-light)]">
            <p className="text-sm text-[var(--dq-text-muted)]">
              Run ID: <code className="text-xs">{runId}</code>
            </p>
            {runName && (
              <p className="text-sm text-[var(--dq-text-muted)]">
                Name: {runName}
              </p>
            )}
          </div>
        </div>
      </Modal>

      {/* Toast notification */}
      {toast && (
        <Toast
          message={toast.message}
          variant={toast.variant}
          visible={!!toast}
          onClose={() => setToast(null)}
        />
      )}
    </>
  );
}

// Icons
function ExportIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="var(--dq-success)"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function ErrorIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="var(--dq-danger)"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  );
}
