/**
 * DevQubit UI Router Configuration
 *
 * Defines all routes for the open-core UI.
 * Can be extended by devqubit-hub for additional routes.
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

/**
 * Core route definitions for devqubit-ui.
 */
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

/**
 * Create router with optional additional routes.
 *
 * Parameters
 * ----------
 * additionalRoutes : RouteObject[]
 *     Additional routes to append (for hub extension).
 *
 * Returns
 * -------
 * Router
 *     Configured browser router.
 */
export function createRouter(additionalRoutes: RouteObject[] = []) {
  return createBrowserRouter([...coreRoutes, ...additionalRoutes]);
}

export const router = createRouter();
