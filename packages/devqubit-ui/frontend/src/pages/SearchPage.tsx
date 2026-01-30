/**
 * Search Page
 */

import { useState } from 'react';
import { Layout } from '../components/Layout';
import {
  Card, CardHeader, CardTitle, Button, Spinner,
  Table, TableHead, TableBody, TableRow, TableHeader, TableCell,
  FormGroup, Label, Input,
} from '../components';
import { RunsTable } from '../components/RunsTable';
import { useApp, useMutation } from '../hooks';
import type { RunSummary } from '../types';

export function SearchPage() {
  const { api } = useApp();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RunSummary[] | null>(null);
  const [searched, setSearched] = useState(false);

  const { mutate: search, loading } = useMutation(async () => {
    const data = await api.listRuns({ q: query, limit: 100 });
    setResults(data.runs);
    setSearched(true);
    return data;
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      search();
    }
  };

  return (
    <Layout>
      <div className="page-header">
        <h1 className="page-title">Search Runs</h1>
      </div>

      <Card className="mb-4">
        <form onSubmit={handleSubmit}>
          <FormGroup>
            <Label htmlFor="q">Query</Label>
            <Input
              id="q"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="metric.fidelity > 0.95 and params.shots = 1000"
              className="font-mono"
            />
          </FormGroup>
          <div className="flex gap-2 items-center">
            <Button type="submit" variant="primary" disabled={loading || !query.trim()}>
              {loading && <Spinner />}
              Search
            </Button>
            {loading && <span className="text-muted text-sm">Searching...</span>}
          </div>
        </form>
      </Card>

      {searched && results && (
        <Card className="mb-4">
          {results.length > 0 ? (
            <RunsTable runs={results} />
          ) : (
            <p className="text-muted text-center py-8">No runs match your query</p>
          )}
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
