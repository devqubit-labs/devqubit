/**
 * DevQubit UI Table Component
 */

import { cn } from '../../utils';

export type TableProps = React.TableHTMLAttributes<HTMLTableElement>;
export type TableHeadProps = React.HTMLAttributes<HTMLTableSectionElement>;
export type TableBodyProps = React.HTMLAttributes<HTMLTableSectionElement>;
export type TableRowProps = React.HTMLAttributes<HTMLTableRowElement>;
export type TableHeaderProps = React.ThHTMLAttributes<HTMLTableCellElement>;
export type TableCellProps = React.TdHTMLAttributes<HTMLTableCellElement>;

export function Table({ className, children, ...props }: TableProps) {
  return (
    <table className={cn('table', className)} {...props}>
      {children}
    </table>
  );
}

export function TableHead({ className, children, ...props }: TableHeadProps) {
  return <thead className={className} {...props}>{children}</thead>;
}

export function TableBody({ className, children, ...props }: TableBodyProps) {
  return <tbody className={className} {...props}>{children}</tbody>;
}

export function TableRow({ className, children, ...props }: TableRowProps) {
  return <tr className={className} {...props}>{children}</tr>;
}

export function TableHeader({ className, children, ...props }: TableHeaderProps) {
  return <th className={className} {...props}>{children}</th>;
}

export function TableCell({ className, children, ...props }: TableCellProps) {
  return <td className={className} {...props}>{children}</td>;
}
