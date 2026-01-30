/**
 * DevQubit UI Projects Table Component
 */

import { Link } from 'react-router-dom';
import { Table, TableHead, TableBody, TableRow, TableHeader, TableCell } from '../Table';
import { Badge, Button, EmptyState } from '../ui';
import { shortId, truncate } from '../../utils';
import type { Project, Workspace } from '../../types';

export interface ProjectsTableProps {
  projects: Project[];
  currentWorkspace?: Workspace | null;
}

export function ProjectsTable({ projects, currentWorkspace }: ProjectsTableProps) {
  if (!projects.length) {
    return (
      <EmptyState
        message="No projects yet"
        hint="Projects are created automatically when you log runs"
      />
    );
  }

  const runsUrl = (project: string) => {
    const params = new URLSearchParams({ project });
    if (currentWorkspace) params.set('workspace', currentWorkspace.id);
    return `/runs?${params}`;
  };

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeader>Project</TableHeader>
          <TableHeader>Runs</TableHeader>
          <TableHeader>Baseline</TableHeader>
          <TableHeader></TableHeader>
        </TableRow>
      </TableHead>
      <TableBody>
        {projects.map((project) => (
          <TableRow key={project.name}>
            <TableCell>
              <span className="font-medium">{project.name}</span>
              {project.description && (
                <p className="text-muted text-sm">{truncate(project.description)}</p>
              )}
            </TableCell>
            <TableCell>
              <Badge variant="gray">{project.run_count ?? 0}</Badge>
            </TableCell>
            <TableCell>
              {project.baseline ? (
                <>
                  <Link to={`/runs/${project.baseline.run_id}`} className="font-mono text-sm">
                    {shortId(project.baseline.run_id)}
                  </Link>
                  {project.baseline.run_name && (
                    <p className="text-muted text-sm">{project.baseline.run_name}</p>
                  )}
                </>
              ) : (
                <span className="text-muted">—</span>
              )}
            </TableCell>
            <TableCell className="text-right">
              <Link to={runsUrl(project.name)}>
                <Button variant="secondary" size="sm">View Runs</Button>
              </Link>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
