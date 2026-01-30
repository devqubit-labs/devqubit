/**
 * DevQubit UI App Component
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
 */
export function App({ additionalRoutes, ...providerProps }: AppProps) {
  const appRouter = additionalRoutes?.length ? createRouter(additionalRoutes) : router;

  return (
    <AppProvider {...providerProps}>
      <RouterProvider router={appRouter} />
    </AppProvider>
  );
}
