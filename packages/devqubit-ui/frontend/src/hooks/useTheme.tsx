/**
 * DevQubit UI Theme Hook
 *
 * Default theme provider that returns 'light' theme.
 * Downstream editions can override with a full implementation
 * (dark mode, system preference, localStorage persistence, etc.)
 * by re-exporting their own ThemeProvider under the same name.
 */

import { createContext, useContext, type ReactNode } from 'react';

export type Theme = 'light' | 'dark' | 'system';

export interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: Theme;
  storageKey?: string;
}

/**
 * Default Theme Provider — always resolves to 'light'.
 *
 * Override by providing your own ThemeProvider that satisfies
 * the same ThemeContextValue interface.
 */
export function ThemeProvider({ children }: ThemeProviderProps) {
  const value: ThemeContextValue = {
    theme: 'light',
    resolvedTheme: 'light',
    setTheme: () => {},
    toggleTheme: () => {},
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * Hook to access theme context.
 */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    return {
      theme: 'light',
      resolvedTheme: 'light',
      setTheme: () => {},
      toggleTheme: () => {},
    };
  }
  return ctx;
}

/**
 * Hook that returns theme context or defaults if not in ThemeProvider.
 */
export function useThemeOptional(): ThemeContextValue {
  return useTheme();
}
