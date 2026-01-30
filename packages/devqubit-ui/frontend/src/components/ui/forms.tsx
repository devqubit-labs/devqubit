/**
 * DevQubit UI Form Components
 *
 * Form elements styled according to the design system.
 */

import { forwardRef, type InputHTMLAttributes, type SelectHTMLAttributes } from 'react';
import { cn } from '../../utils';

/* =========================================================================
   FormGroup
   ========================================================================= */

export interface FormGroupProps {
  children: React.ReactNode;
  className?: string;
}

export function FormGroup({ children, className }: FormGroupProps) {
  return <div className={cn('form-group', className)}>{children}</div>;
}

/* =========================================================================
   Label
   ========================================================================= */

export interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {}

export function Label({ className, children, ...props }: LabelProps) {
  return (
    <label className={cn('form-label', className)} {...props}>
      {children}
    </label>
  );
}

/* =========================================================================
   Input
   ========================================================================= */

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return <input ref={ref} className={cn('form-input', className)} {...props} />;
  }
);
Input.displayName = 'Input';

/* =========================================================================
   Select
   ========================================================================= */

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <select ref={ref} className={cn('form-input', className)} {...props}>
        {children}
      </select>
    );
  }
);
Select.displayName = 'Select';

/* =========================================================================
   FormRow
   ========================================================================= */

export interface FormRowProps {
  children: React.ReactNode;
  className?: string;
}

export function FormRow({ children, className }: FormRowProps) {
  return <div className={cn('form-row', className)}>{children}</div>;
}
