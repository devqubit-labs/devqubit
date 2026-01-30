/**
 * DevQubit UI App Component
 *
 * Main application entry point with router and providers.
 */

import { RouterProvider } from 'react-router-dom';
import { AppProvider, type AppProviderProps } from './hooks';
import { router, createRouter } from './router';
import type { RouteObject } from 'react-router-dom';

export interface AppProps extends Omit<AppProviderProps, 'children'> {
  additionalRoutes?: RouteObject[];
}

/**
 * Main application component.
 *
 * Parameters
 * ----------
 * api : ApiClient, optional
 *     Custom API client instance.
 * initialWorkspace : Workspace, optional
 *     Initial workspace context.
 * additionalRoutes : RouteObject[], optional
 *     Additional routes for hub extension.
 */
export function App({ additionalRoutes, ...providerProps }: AppProps) {
  const appRouter = additionalRoutes?.length ? createRouter(additionalRoutes) : router;

  return (
    <AppProvider {...providerProps}>
      <RouterProvider router={appRouter} />
    </AppProvider>
  );
}
