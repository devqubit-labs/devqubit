/**
 * Metric time-series chart — zero-dependency SVG line chart.
 *
 * Each chart independently measures its container width via
 * ResizeObserver, so it works correctly in any CSS layout.
 * Tooltip is rendered as SVG elements to avoid clipping issues.
 */

import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import type { MetricPoint, MetricSeries } from '../../types';
import { formatNumber } from '../../utils';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PADDING = { top: 12, right: 16, bottom: 28, left: 56 };
const CHART_HEIGHT = 210;
const TICK_COUNT = 5;

const COLORS = [
  'var(--dq-primary)',
  'var(--dq-info)',
  'var(--dq-warning)',
  'var(--dq-danger)',
  '#8B5CF6',
  '#EC4899',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function niceStep(range: number, ticks: number): number {
  const raw = range / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

function axisTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const step = niceStep(max - min, count);
  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.01; v += step) {
    ticks.push(v);
  }
  return ticks;
}

function formatAxisValue(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  if (abs >= 1) return v.toFixed(abs >= 100 ? 0 : abs >= 10 ? 1 : 2);
  return v.toPrecision(3);
}

/** Downsample to at most `maxPx` points using LTTB-like min/max buckets. */
function downsample(pts: MetricPoint[], maxPx: number): MetricPoint[] {
  if (pts.length <= maxPx) return pts;
  const bucketSize = pts.length / maxPx;
  const out: MetricPoint[] = [pts[0]];
  for (let i = 1; i < maxPx - 1; i++) {
    const start = Math.floor(i * bucketSize);
    const end = Math.min(Math.floor((i + 1) * bucketSize), pts.length);
    let minPt = pts[start], maxPt = pts[start];
    for (let j = start; j < end; j++) {
      if (pts[j].value < minPt.value) minPt = pts[j];
      if (pts[j].value > maxPt.value) maxPt = pts[j];
    }
    if (minPt.step <= maxPt.step) {
      out.push(minPt);
      if (minPt !== maxPt) out.push(maxPt);
    } else {
      out.push(maxPt);
      if (minPt !== maxPt) out.push(minPt);
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}

/** Hook: observe element width via ResizeObserver. */
function useContainerWidth(ref: React.RefObject<HTMLDivElement | null>): number {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setWidth(Math.floor(w));
    });
    obs.observe(el);
    setWidth(Math.floor(el.getBoundingClientRect().width) || 0);
    return () => obs.disconnect();
  }, [ref]);

  return width;
}

// ---------------------------------------------------------------------------
// SVG Tooltip (rendered inside SVG — never clips)
// ---------------------------------------------------------------------------

interface SvgTooltipProps {
  x: number;
  y: number;
  value: string;
  step: number;
  chartWidth: number;
}

function SvgTooltip({ x, y, value, step, chartWidth }: SvgTooltipProps) {
  const label = `${value}  step ${step}`;
  const estWidth = label.length * 6.2 + 16;
  const tipH = 22;
  const gap = 8;

  // Position above the dot; flip below if too close to top
  let tipY = y - tipH - gap;
  if (tipY < 2) tipY = y + gap + 4;

  // Clamp horizontally within chart bounds
  let tipX = x - estWidth / 2;
  tipX = Math.max(PADDING.left, Math.min(tipX, chartWidth - PADDING.right - estWidth));

  return (
    <g pointerEvents="none">
      <rect
        x={tipX}
        y={tipY}
        width={estWidth}
        height={tipH}
        rx={4}
        fill="var(--dq-bg-primary)"
        stroke="var(--dq-border-color)"
        strokeWidth={1}
      />
      <text
        x={tipX + 8}
        y={tipY + tipH / 2}
        dominantBaseline="central"
        fontSize={11}
        className="fill-[var(--dq-text-primary)]"
        fontFamily="ui-monospace, monospace"
      >
        <tspan>{value}</tspan>
        <tspan dx={8} className="fill-[var(--dq-text-muted)]">step {step}</tspan>
      </text>
    </g>
  );
}

// ---------------------------------------------------------------------------
// Single chart (self-measuring)
// ---------------------------------------------------------------------------

interface LineChartProps {
  metricKey: string;
  points: MetricPoint[];
  color: string;
}

function LineChart({ metricKey, points, color }: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const width = useContainerWidth(containerRef);
  const [hover, setHover] = useState<{ x: number; y: number; pt: MetricPoint } | null>(null);

  const sorted = useMemo(
    () => [...points].sort((a, b) => a.step - b.step),
    [points],
  );

  const plotW = width - PADDING.left - PADDING.right;
  const plotH = CHART_HEIGHT - PADDING.top - PADDING.bottom;

  const sampled = useMemo(
    () => (plotW > 0 ? downsample(sorted, plotW) : sorted),
    [sorted, plotW],
  );

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (const p of sorted) {
      if (p.value < lo) lo = p.value;
      if (p.value > hi) hi = p.value;
    }
    const pad = (hi - lo) * 0.05 || 0.1;
    return {
      xMin: sorted[0]?.step ?? 0,
      xMax: sorted[sorted.length - 1]?.step ?? 1,
      yMin: lo - pad,
      yMax: hi + pad,
    };
  }, [sorted]);

  const xScale = useCallback(
    (step: number) => PADDING.left + ((step - xMin) / (xMax - xMin || 1)) * plotW,
    [xMin, xMax, plotW],
  );
  const yScale = useCallback(
    (val: number) => PADDING.top + plotH - ((val - yMin) / (yMax - yMin || 1)) * plotH,
    [yMin, yMax, plotH],
  );

  const pathD = useMemo(
    () =>
      sampled
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.step).toFixed(1)},${yScale(p.value).toFixed(1)}`)
        .join(' '),
    [sampled, xScale, yScale],
  );

  const yTicks = useMemo(() => axisTicks(yMin, yMax, TICK_COUNT), [yMin, yMax]);
  const xTicks = useMemo(() => axisTicks(xMin, xMax, TICK_COUNT).map(Math.round), [xMin, xMax]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect || !sorted.length) return;
      const mouseX = e.clientX - rect.left;
      const step = xMin + ((mouseX - PADDING.left) / plotW) * (xMax - xMin);
      let lo = 0, hi = sorted.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid].step < step) lo = mid + 1; else hi = mid;
      }
      const idx =
        lo > 0 && Math.abs(sorted[lo - 1].step - step) < Math.abs(sorted[lo].step - step)
          ? lo - 1
          : lo;
      const pt = sorted[idx];
      setHover({ x: xScale(pt.step), y: yScale(pt.value), pt });
    },
    [sorted, xMin, xMax, plotW, xScale, yScale],
  );

  const lastPt = sorted[sorted.length - 1];

  // Render an empty sizer div until we know our width
  if (width < 100) {
    return <div ref={containerRef} className="metric-chart" />;
  }

  return (
    <div ref={containerRef} className="metric-chart">
      <div className="metric-chart-header">
        <span className="metric-chart-label" style={{ color }}>{metricKey}</span>
        {lastPt && (
          <span className="metric-chart-latest">
            latest: {formatNumber(lastPt.value)} @ step {lastPt.step}
          </span>
        )}
      </div>

      <svg
        ref={svgRef}
        width={width}
        height={CHART_HEIGHT}
        className="metric-chart-svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHover(null)}
      >
        {/* Horizontal grid lines (dashed, subtle) */}
        {yTicks.map(v => (
          <line
            key={`gy-${v}`}
            x1={PADDING.left}
            x2={width - PADDING.right}
            y1={yScale(v)}
            y2={yScale(v)}
            stroke="var(--dq-border-color)"
            strokeWidth={0.5}
            strokeDasharray="3,3"
          />
        ))}

        {/* Vertical grid lines (dashed, subtle) */}
        {xTicks.map(v => (
          <line
            key={`gx-${v}`}
            x1={xScale(v)}
            x2={xScale(v)}
            y1={PADDING.top}
            y2={PADDING.top + plotH}
            stroke="var(--dq-border-color)"
            strokeWidth={0.5}
            strokeDasharray="3,3"
          />
        ))}

        {/* Plot area border (left + bottom) */}
        <line
          x1={PADDING.left} y1={PADDING.top}
          x2={PADDING.left} y2={PADDING.top + plotH}
          stroke="var(--dq-border-color)" strokeWidth={1}
        />
        <line
          x1={PADDING.left} y1={PADDING.top + plotH}
          x2={width - PADDING.right} y2={PADDING.top + plotH}
          stroke="var(--dq-border-color)" strokeWidth={1}
        />

        {/* Y axis labels */}
        {yTicks.map(v => (
          <text
            key={v}
            x={PADDING.left - 6}
            y={yScale(v)}
            textAnchor="end"
            dominantBaseline="middle"
            className="fill-[var(--dq-text-muted)]"
            fontSize={10}
          >
            {formatAxisValue(v)}
          </text>
        ))}

        {/* X axis labels */}
        {xTicks.map(v => (
          <text
            key={v}
            x={xScale(v)}
            y={CHART_HEIGHT - 4}
            textAnchor="middle"
            className="fill-[var(--dq-text-muted)]"
            fontSize={10}
          >
            {v}
          </text>
        ))}

        {/* Data line */}
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Hover crosshair + dot + tooltip */}
        {hover && (
          <>
            <line
              x1={hover.x} x2={hover.x}
              y1={PADDING.top} y2={PADDING.top + plotH}
              stroke="var(--dq-text-muted)"
              strokeWidth={1}
              strokeDasharray="3,3"
              opacity={0.4}
            />
            <circle cx={hover.x} cy={hover.y} r={3.5} fill={color} />
            <SvgTooltip
              x={hover.x}
              y={hover.y}
              value={formatNumber(hover.pt.value)}
              step={hover.pt.step}
              chartWidth={width}
            />
          </>
        )}
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Public: renders one chart per metric key, paginated
// ---------------------------------------------------------------------------

const PAGE_SIZE = 6; // 2 rows × 3 columns max

export interface MetricChartsProps {
  series: MetricSeries;
}

export function MetricCharts({ series }: MetricChartsProps) {
  const keys = Object.keys(series).sort();
  const [page, setPage] = useState(0);

  if (!keys.length) return null;

  const totalPages = Math.ceil(keys.length / PAGE_SIZE);
  const pageKeys = keys.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div>
      <div className="metric-charts-grid">
        {pageKeys.map((key, i) => (
          <LineChart
            key={key}
            metricKey={key}
            points={series[key]}
            color={COLORS[(page * PAGE_SIZE + i) % COLORS.length]}
          />
        ))}
      </div>

      {totalPages > 1 && (
        <div className="metric-charts-pagination">
          <button
            className="metric-charts-page-btn"
            onClick={() => setPage(p => p - 1)}
            disabled={page === 0}
          >
            ‹ Prev
          </button>
          <span className="metric-charts-page-info">
            {page + 1} / {totalPages}
          </span>
          <button
            className="metric-charts-page-btn"
            onClick={() => setPage(p => p + 1)}
            disabled={page >= totalPages - 1}
          >
            Next ›
          </button>
        </div>
      )}
    </div>
  );
}
