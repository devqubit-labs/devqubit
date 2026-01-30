/**
 * DevQubit UI Layout Component
 *
 * Main layout wrapper with header, navigation, and content area.
 * Designed for extension by devqubit-hub.
 */

import { Link, useLocation } from 'react-router-dom';
import { createContext, useContext } from 'react';
import { cn } from '../../utils';
import type { LayoutConfig, NavLink } from '../../types';

/** Default navigation links */
const DEFAULT_NAV_LINKS: NavLink[] = [
  { href: '/runs', label: 'Runs', matchPaths: ['/runs'] },
  { href: '/projects', label: 'Projects', matchPaths: ['/projects'] },
  { href: '/groups', label: 'Groups', matchPaths: ['/groups'] },
  { href: '/diff', label: 'Compare', matchPaths: ['/diff'] },
  { href: '/search', label: 'Search', matchPaths: ['/search'] },
];

/** Layout config context for global configuration */
const LayoutConfigContext = createContext<LayoutConfig | null>(null);

/** Provider for global layout config (used by hub) */
export function LayoutConfigProvider({
  config,
  children
}: {
  config: LayoutConfig;
  children: React.ReactNode;
}) {
  return (
    <LayoutConfigContext.Provider value={config}>
      {children}
    </LayoutConfigContext.Provider>
  );
}

/** Hook to access layout config */
export function useLayoutConfig(): LayoutConfig | null {
  return useContext(LayoutConfigContext);
}

export interface LayoutProps {
  children: React.ReactNode;
  /** Local config overrides global layoutConfig */
  config?: LayoutConfig;
}

/**
 * Main layout component with header and navigation.
 *
 * Parameters
 * ----------
 * children : ReactNode
 *     Page content to render in main area.
 * config : LayoutConfig, optional
 *     Local config (merges with global layoutConfig from AppProvider).
 */
export function Layout({ children, config: localConfig }: LayoutProps) {
  const location = useLocation();
  const globalConfig = useLayoutConfig();

  // Merge configs: local overrides global
  const config = { ...globalConfig, ...localConfig };

  // Build nav links: prepend + (custom or default) + append
  const baseLinks = config?.navLinks ?? DEFAULT_NAV_LINKS;
  const navLinks = [
    ...(config?.prependNavLinks ?? []),
    ...baseLinks,
    ...(config?.appendNavLinks ?? []),
  ];

  const logo = config?.logo ?? { text: 'devqubit', icon: '⚛' };

  const isActive = (link: NavLink) => {
    if (link.matchPaths) {
      return link.matchPaths.some(p => location.pathname.startsWith(p));
    }
    return location.pathname === link.href;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gray-900 h-14 sticky top-0 z-50">
        <div className="max-w-container mx-auto px-6 h-full flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link to="/" className="text-lg font-bold text-primary hover:text-primary-light flex items-center gap-1.5">
              {logo.icon && <span>{logo.icon}</span>}
              {logo.text}
            </Link>
            <nav className="flex gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  to={link.href}
                  className={cn(
                    'px-3 py-2 rounded-md text-sm font-medium transition-all',
                    isActive(link)
                      ? 'text-white bg-primary/20'
                      : 'text-gray-400 hover:text-white hover:bg-white/10'
                  )}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-4">
            {config?.headerRight}
          </div>
        </div>
      </header>

      <main className="max-w-container mx-auto p-6">
        {children}
      </main>
    </div>
  );
}

export interface PageHeaderProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
}

/**
 * Page header with title and optional actions.
 */
export function PageHeader({ title, subtitle, actions }: PageHeaderProps) {
  return (
    <div className="flex items-start justify-between mb-6 gap-4 flex-wrap">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 flex items-center gap-2 flex-wrap">
          {title}
        </h1>
        {subtitle && <p className="text-sm text-muted mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}
