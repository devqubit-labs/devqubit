/**
 * DevQubit UI Groups Table Component
 */

import { Link } from 'react-router-dom';
import { Table, TableHead, TableBody, TableRow, TableHeader, TableCell } from '../Table';
import { Badge, Button, EmptyState } from '../ui';
import { shortId } from '../../utils';
import type { Group } from '../../types';

export interface GroupsTableProps {
  groups: Group[];
}

export function GroupsTable({ groups }: GroupsTableProps) {
  if (!groups.length) {
    return (
      <EmptyState
        message="No groups found"
        hint="Groups are created when runs have a group_id set"
      />
    );
  }

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeader>Group ID</TableHeader>
          <TableHeader>Name</TableHeader>
          <TableHeader>Project</TableHeader>
          <TableHeader>Runs</TableHeader>
          <TableHeader>Actions</TableHeader>
        </TableRow>
      </TableHead>
      <TableBody>
        {groups.map((group) => (
          <TableRow key={group.group_id}>
            <TableCell className="font-mono">{shortId(group.group_id)}</TableCell>
            <TableCell>{group.group_name || '—'}</TableCell>
            <TableCell>{group.project}</TableCell>
            <TableCell>
              <Badge variant="gray">{group.run_count}</Badge>
            </TableCell>
            <TableCell>
              <Link to={`/groups/${group.group_id}`}>
                <Button variant="secondary" size="sm">View Runs</Button>
              </Link>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
