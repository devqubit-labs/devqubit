/**
 * DevQubit UI Form Components
 */

import { forwardRef, type InputHTMLAttributes, type SelectHTMLAttributes, type LabelHTMLAttributes } from 'react';
import { cn } from '../../utils';

type LabelProps = LabelHTMLAttributes<HTMLLabelElement>;
type InputProps = InputHTMLAttributes<HTMLInputElement>;
type SelectProps = SelectHTMLAttributes<HTMLSelectElement>;

export interface FormGroupProps {
  children: React.ReactNode;
  className?: string;
}

export function FormGroup({ children, className }: FormGroupProps) {
  return <div className={cn('form-group', className)}>{children}</div>;
}

export function Label({ className, children, ...props }: LabelProps) {
  return (
    <label className={cn('form-label', className)} {...props}>
      {children}
    </label>
  );
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return <input ref={ref} className={cn('form-input', className)} {...props} />;
  }
);
Input.displayName = 'Input';

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

export interface FormRowProps {
  children: React.ReactNode;
  className?: string;
}

export function FormRow({ children, className }: FormRowProps) {
  return <div className={cn('form-row', className)}>{children}</div>;
}
