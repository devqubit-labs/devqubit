/**
 * DevQubit UI Utility Functions
 *
 * Helper functions for formatting, date handling, and data transformation.
 */

/**
 * Truncate a UUID to its first 8 characters.
 *
 * Parameters
 * ----------
 * id : string
 *     Full UUID string.
 *
 * Returns
 * -------
 * string
 *     Truncated ID (first 8 characters).
 */
export function shortId(id: string): string {
  return id?.slice(0, 8) ?? '';
}

/**
 * Truncate a digest/hash to its first 12 characters.
 *
 * Parameters
 * ----------
 * digest : string
 *     Full digest string.
 *
 * Returns
 * -------
 * string
 *     Truncated digest (first 12 characters).
 */
export function shortDigest(digest: string | undefined | null): string {
  return digest?.slice(0, 12) ?? 'N/A';
}

/**
 * Format a date string as relative time (e.g., "2 hours ago").
 *
 * Parameters
 * ----------
 * dateStr : string
 *     ISO date string.
 *
 * Returns
 * -------
 * string
 *     Human-readable relative time.
 */
export function timeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (seconds < 60) return 'just now';
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    return `${mins} minute${mins !== 1 ? 's' : ''} ago`;
  }
  if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
  }
  if (seconds < 2592000) {
    const days = Math.floor(seconds / 86400);
    return `${days} day${days !== 1 ? 's' : ''} ago`;
  }
  if (seconds < 31536000) {
    const months = Math.floor(seconds / 2592000);
    return `${months} month${months !== 1 ? 's' : ''} ago`;
  }
  const years = Math.floor(seconds / 31536000);
  return `${years} year${years !== 1 ? 's' : ''} ago`;
}

/**
 * Format a number with appropriate precision.
 *
 * Parameters
 * ----------
 * value : number
 *     Numeric value to format.
 * precision : number, default=6
 *     Maximum significant digits.
 *
 * Returns
 * -------
 * string
 *     Formatted number string.
 */
export function formatNumber(value: number, precision: number = 6): string {
  if (typeof value !== 'number' || isNaN(value)) return String(value);
  return Number(value.toPrecision(precision)).toString();
}

/**
 * Format bytes to human-readable size.
 *
 * Parameters
 * ----------
 * bytes : number
 *     Size in bytes.
 *
 * Returns
 * -------
 * string
 *     Formatted size string (e.g., "1.5 MB").
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

/**
 * Pretty-print JSON with indentation.
 *
 * Parameters
 * ----------
 * obj : unknown
 *     Object to stringify.
 *
 * Returns
 * -------
 * string
 *     Formatted JSON string.
 */
export function jsonPretty(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

/**
 * Truncate text with ellipsis.
 *
 * Parameters
 * ----------
 * text : string
 *     Text to truncate.
 * maxLength : number, default=60
 *     Maximum length before truncation.
 *
 * Returns
 * -------
 * string
 *     Truncated text with ellipsis if needed.
 */
export function truncate(text: string | undefined | null, maxLength: number = 60): string {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

/**
 * Build URL with query parameters.
 *
 * Parameters
 * ----------
 * base : string
 *     Base URL path.
 * params : Record<string, unknown>
 *     Query parameters (falsy values excluded).
 *
 * Returns
 * -------
 * string
 *     URL with query string.
 */
export function buildUrl(base: string, params: Record<string, unknown>): string {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      searchParams.set(key, String(value));
    }
  });
  const qs = searchParams.toString();
  return qs ? `${base}?${qs}` : base;
}

/**
 * Classnames utility for conditional class joining.
 *
 * Parameters
 * ----------
 * classes : Array<string | false | null | undefined>
 *     Class names or falsy values.
 *
 * Returns
 * -------
 * string
 *     Space-separated class string.
 */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ');
}
