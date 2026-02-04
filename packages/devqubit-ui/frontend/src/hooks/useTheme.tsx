/**
 * DevQubit UI Theme Hook (Placeholder)
 *
 * Provides theme interface for extensibility. This is a no-op implementation
 * that always returns 'light' theme. devqubit-hub provides the full implementation
 * with actual dark mode switching for Teams/Enterprise editions.
 *
 * Usage in hub: Import ThemeProvider from hub's own useTheme, not from @devqubit/ui
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
 * Theme Provider (No-op placeholder)
 *
 * Open-core version that provides light theme only.
 * For dark mode support, use devqubit-hub's ThemeProvider.
 */
export function ThemeProvider({ children }: ThemeProviderProps) {
  // No-op: always light theme in open-core
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
 * Hook to access theme context
 */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    // Return default light theme if not in provider
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
 * Hook that returns theme context or defaults if not in ThemeProvider
 */
export function useThemeOptional(): ThemeContextValue {
  return useTheme();
}
