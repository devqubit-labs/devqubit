/**
 * DevQubit UI Layout Component
 */

import { Link, useLocation } from 'react-router-dom';
import { createContext, useContext } from 'react';
import { cn } from '../../utils';
import type { LayoutConfig, NavLink } from '../../types';

const DEFAULT_NAV_LINKS: NavLink[] = [
  { href: '/runs', label: 'Runs', matchPaths: ['/runs'] },
  { href: '/projects', label: 'Projects', matchPaths: ['/projects'] },
  { href: '/groups', label: 'Groups', matchPaths: ['/groups'] },
  { href: '/diff', label: 'Compare', matchPaths: ['/diff'] },
  { href: '/search', label: 'Search', matchPaths: ['/search'] },
];

const LayoutConfigContext = createContext<LayoutConfig | null>(null);

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

export function useLayoutConfig(): LayoutConfig | null {
  return useContext(LayoutConfigContext);
}

export interface LayoutProps {
  children: React.ReactNode;
  config?: LayoutConfig;
}

export function Layout({ children, config: localConfig }: LayoutProps) {
  const location = useLocation();
  const globalConfig = useLayoutConfig();
  const config = { ...globalConfig, ...localConfig };

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
      <header className="bg-gray-900 h-14 sticky top-0 z-50 border-b border-gray-800">
        <div className="max-w-container mx-auto px-6 h-full flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link to="/" className="text-lg font-semibold text-primary hover:text-primary-light flex items-center gap-2 transition-colors hover:no-underline">
              {logo.icon && <span className="text-xl">{logo.icon}</span>}
              {logo.text}
            </Link>
            <nav className="flex gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  to={link.href}
                  className={cn(
                    'px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-150 hover:no-underline',
                    isActive(link)
                      ? 'text-white bg-white/10'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
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

      <main className="max-w-container mx-auto p-6 fade-in">
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
