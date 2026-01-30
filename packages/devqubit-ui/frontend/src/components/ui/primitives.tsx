/**
 * DevQubit UI Primitive Components
 */

import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes } from 'react';
import { cn } from '../../utils';

/* Badge */
export type BadgeVariant = 'success' | 'danger' | 'warning' | 'info' | 'gray' | 'neutral';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ variant = 'gray', className, children, ...props }: BadgeProps) {
  return (
    <span className={cn('badge', `badge-${variant}`, className)} {...props}>
      {children}
    </span>
  );
}

/* Button */
export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'ghost-danger';
export type ButtonSize = 'default' | 'sm';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'secondary', size = 'default', loading, className, children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'btn',
          variant === 'ghost-danger' ? 'btn-ghost-danger' : `btn-${variant}`,
          size === 'sm' && 'btn-sm',
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading && <Spinner />}
        {children}
      </button>
    );
  }
);
Button.displayName = 'Button';

/* Card */
export type CardProps = HTMLAttributes<HTMLDivElement>;
export type CardHeaderProps = HTMLAttributes<HTMLDivElement>;
export type CardTitleProps = HTMLAttributes<HTMLHeadingElement>;

export function Card({ className, children, ...props }: CardProps) {
  return <div className={cn('card', className)} {...props}>{children}</div>;
}

export function CardHeader({ className, children, ...props }: CardHeaderProps) {
  return <div className={cn('card-header', className)} {...props}>{children}</div>;
}

export function CardTitle({ className, children, ...props }: CardTitleProps) {
  return <h3 className={cn('card-title', className)} {...props}>{children}</h3>;
}

/* Alert */
export type AlertVariant = 'success' | 'danger' | 'warning' | 'info';

export interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
}

export function Alert({ variant = 'info', className, children, ...props }: AlertProps) {
  return <div className={cn('alert', `alert-${variant}`, className)} {...props}>{children}</div>;
}

/* Spinner */
export type SpinnerProps = HTMLAttributes<HTMLSpanElement>;

export function Spinner({ className, ...props }: SpinnerProps) {
  return <span className={cn('spinner', className)} {...props} />;
}

/* Empty State */
export interface EmptyStateProps extends HTMLAttributes<HTMLDivElement> {
  message: string;
  hint?: string;
}

export function EmptyState({ message, hint, className, ...props }: EmptyStateProps) {
  return (
    <div className={cn('empty-state', className)} {...props}>
      <p>{message}</p>
      {hint && <p className="text-sm text-muted mt-2">{hint}</p>}
    </div>
  );
}

/* KVList */
export interface KVListProps extends HTMLAttributes<HTMLDListElement> {
  items: Array<{ label: string; value: React.ReactNode }>;
}

export function KVList({ items, className, ...props }: KVListProps) {
  return (
    <dl className={cn('kv-list', className)} {...props}>
      {items.map(({ label, value }) => (
        <div key={label} className="contents">
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/* Modal */
export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}

export function Modal({ open, onClose, title, children, actions }: ModalProps) {
  if (!open) return null;

  return (
    <div
      className={cn('modal-backdrop', open && 'active')}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal-title">{title}</h3>
        <div className="modal-body">{children}</div>
        {actions && <div className="modal-actions">{actions}</div>}
      </div>
    </div>
  );
}

/* Toast */
export type ToastVariant = 'success' | 'error' | 'info';

export interface ToastProps {
  message: string;
  variant?: ToastVariant;
  visible: boolean;
  onClose: () => void;
}

export function Toast({ message, variant = 'info', visible, onClose }: ToastProps) {
  if (!visible) return null;

  const variantClass = {
    success: 'bg-success-bg text-[#065f46] border-[#a7f3d0]',
    error: 'bg-danger-bg text-[#991b1b] border-[#fecaca]',
    info: 'bg-info-bg text-[#1e40af] border-[#bfdbfe]',
  }[variant];

  return (
    <div className={cn('toast', variantClass)} onClick={onClose}>
      {message}
    </div>
  );
}
