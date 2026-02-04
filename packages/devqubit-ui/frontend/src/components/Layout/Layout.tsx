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
    <div className="dq-layout">
      <header className="dq-header">
        <div className="dq-header-inner">
          <div className="dq-header-left">
            <Link to="/" className="dq-logo">
              {logo.icon && <span className="dq-logo-icon">{logo.icon}</span>}
              {logo.text}
            </Link>
            <nav className="dq-nav">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  to={link.href}
                  className={cn('dq-nav-link', isActive(link) && 'active')}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
          <div className="dq-header-right">
            {config?.headerRight}
          </div>
        </div>
      </header>

      <main className="dq-main fade-in">
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
    <div className="page-header">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="text-sm text-muted mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex gap-2">{actions}</div>}
    </div>
  );
}
