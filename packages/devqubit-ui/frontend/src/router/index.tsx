/**
 * DevQubit UI Router Configuration
 */

import { createBrowserRouter, Navigate, type RouteObject } from 'react-router-dom';
import {
  RunsPage,
  RunDetailPage,
  ProjectsPage,
  GroupsPage,
  GroupDetailPage,
  DiffPage,
  SearchPage,
  ArtifactPage,
} from '../pages';

export const coreRoutes: RouteObject[] = [
  { path: '/', element: <Navigate to="/runs" replace /> },
  { path: '/runs', element: <RunsPage /> },
  { path: '/runs/:runId', element: <RunDetailPage /> },
  { path: '/runs/:runId/artifacts/:index', element: <ArtifactPage /> },
  { path: '/projects', element: <ProjectsPage /> },
  { path: '/groups', element: <GroupsPage /> },
  { path: '/groups/:groupId', element: <GroupDetailPage /> },
  { path: '/diff', element: <DiffPage /> },
  { path: '/search', element: <SearchPage /> },
];

export function createRouter(additionalRoutes: RouteObject[] = []) {
  return createBrowserRouter([...coreRoutes, ...additionalRoutes]);
}

export const router = createRouter();
