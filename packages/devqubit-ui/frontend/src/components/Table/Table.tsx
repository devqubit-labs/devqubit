/**
 * DevQubit UI Table Component
 *
 * Styled table component for data display.
 */

import { cn } from '../../utils';

export interface TableProps extends React.TableHTMLAttributes<HTMLTableElement> {}

export function Table({ className, children, ...props }: TableProps) {
  return (
    <table className={cn('table', className)} {...props}>
      {children}
    </table>
  );
}

export interface TableHeadProps extends React.HTMLAttributes<HTMLTableSectionElement> {}

export function TableHead({ className, children, ...props }: TableHeadProps) {
  return <thead className={className} {...props}>{children}</thead>;
}

export interface TableBodyProps extends React.HTMLAttributes<HTMLTableSectionElement> {}

export function TableBody({ className, children, ...props }: TableBodyProps) {
  return <tbody className={className} {...props}>{children}</tbody>;
}

export interface TableRowProps extends React.HTMLAttributes<HTMLTableRowElement> {}

export function TableRow({ className, children, ...props }: TableRowProps) {
  return <tr className={className} {...props}>{children}</tr>;
}

export interface TableHeaderProps extends React.ThHTMLAttributes<HTMLTableCellElement> {}

export function TableHeader({ className, children, ...props }: TableHeaderProps) {
  return <th className={className} {...props}>{children}</th>;
}

export interface TableCellProps extends React.TdHTMLAttributes<HTMLTableCellElement> {}

export function TableCell({ className, children, ...props }: TableCellProps) {
  return <td className={className} {...props}>{children}</td>;
}
