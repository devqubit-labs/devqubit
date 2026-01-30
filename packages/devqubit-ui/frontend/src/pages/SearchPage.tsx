/**
 * DevQubit UI Search Page
 */

import { useState, useCallback } from 'react';
import { Layout, PageHeader } from '../components/Layout';
import {
  Card, CardHeader, CardTitle, Button, FormGroup, Label, Input,
  Table, TableHead, TableBody, TableRow, TableHeader, TableCell,
} from '../components';
import { RunsTable } from '../components/RunsTable';
import { useApp, useMutation } from '../hooks';
import type { RunSummary } from '../types';

export function SearchPage() {
  const { api } = useApp();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RunSummary[] | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const searchMutation = useMutation(async (q: string) => {
    const data = await api.listRuns({ q, limit: 100 });
    return data.runs;
  });

  const handleSearch = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) return;
    const runs = await searchMutation.mutate(query);
    setResults(runs);
    setHasSearched(true);
  }, [query, searchMutation]);

  const deleteMutation = useMutation((runId: string) => api.deleteRun(runId));

  const handleDelete = useCallback(async (runId: string) => {
    await deleteMutation.mutate(runId);
    if (results) {
      setResults(results.filter((r) => r.run_id !== runId));
    }
  }, [deleteMutation, results]);

  return (
    <Layout>
      <PageHeader title="Search Runs" />

      <Card className="mb-4">
        <form onSubmit={handleSearch}>
          <FormGroup>
            <Label htmlFor="q">Query</Label>
            <Input
              id="q"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="metric.fidelity > 0.95 and params.shots = 1000"
              className="font-mono"
            />
          </FormGroup>
          <div className="flex gap-2 items-center">
            <Button type="submit" variant="primary" loading={searchMutation.loading}>
              Search
            </Button>
            {searchMutation.loading && <span className="text-muted text-sm">Searching...</span>}
          </div>
        </form>
      </Card>

      {hasSearched && results && (
        <Card className="mb-4">
          <RunsTable
            runs={results}
            onDelete={handleDelete}
            loading={deleteMutation.loading}
            emptyHint="No runs match your query"
          />
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>Query Syntax</CardTitle></CardHeader>
          <Table>
            <TableHead>
              <TableRow>
                <TableHeader>Field</TableHeader>
                <TableHeader>Description</TableHeader>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow><TableCell className="font-mono">params.X</TableCell><TableCell>Parameter value</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">metric.X</TableCell><TableCell>Metric value</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">tags.X</TableCell><TableCell>Tag value</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">status</TableCell><TableCell>Run status</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">project</TableCell><TableCell>Project name</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">backend</TableCell><TableCell>Backend name</TableCell></TableRow>
            </TableBody>
          </Table>
        </Card>

        <Card>
          <CardHeader><CardTitle>Operators</CardTitle></CardHeader>
          <Table>
            <TableHead>
              <TableRow>
                <TableHeader>Operator</TableHeader>
                <TableHeader>Description</TableHeader>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow><TableCell className="font-mono">=</TableCell><TableCell>Equals</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">!=</TableCell><TableCell>Not equals</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">&gt;</TableCell><TableCell>Greater than</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">&gt;=</TableCell><TableCell>Greater or equal</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">&lt;</TableCell><TableCell>Less than</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">&lt;=</TableCell><TableCell>Less or equal</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">~</TableCell><TableCell>Contains</TableCell></TableRow>
              <TableRow><TableCell className="font-mono">and</TableCell><TableCell>Combine conditions</TableCell></TableRow>
            </TableBody>
          </Table>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader><CardTitle>Examples</CardTitle></CardHeader>
        <Table>
          <TableBody>
            <TableRow>
              <TableCell className="font-mono">metric.fidelity &gt; 0.95</TableCell>
              <TableCell>High fidelity runs</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="font-mono">params.shots = 1000 and status = FINISHED</TableCell>
              <TableCell>Finished runs with 1000 shots</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="font-mono">tags.backend ~ ibm</TableCell>
              <TableCell>Runs with IBM backends</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="font-mono">metric.error &lt; 0.01</TableCell>
              <TableCell>Low error runs</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="font-mono">project = vqe and metric.energy &lt; -2.0</TableCell>
              <TableCell>VQE runs with low energy</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Card>
    </Layout>
  );
}
