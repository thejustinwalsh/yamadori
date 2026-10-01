// Small, hand-drawn SVG charts over time buckets, in the design tokens only
// (the same construction as KŌGŌSEI: a viewBox that stretches, strokes that
// do not, labels as HTML so they never stretch).
//
// Colour is categorical and FIXED by slot (api/stats.ts SLOTS): moss, cyan,
// crimson, rose, then OTHER in grey. Checked with the dataviz validator on the
// chart ground (#0a0e17): the tokens are the design system's neon and sit
// above its lightness band, and moss/rose is 7.4 ΔE for a deutan reader -- the
// 6-8 band that is legal only with secondary encoding. So every series ALSO
// has its own dash pattern and is named in a legend beside its mark, and each
// bucket's values are in a hover title (a table view of the same numbers is
// on the page).
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { binLabel, bucketLabel, ok } from '../api/stats';
import { niceMax } from '../api/powerSeries';
import { colors, space } from '../tokens/tokens.stylex';
import { text } from './text';

const W = 600;

const s = stylex.create({
  box: {
    position: 'relative',
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
    minWidth: 0,
  },
  svg: { display: 'block', width: '100%' },
  tag: {
    position: 'absolute',
    top: '2px',
    left: '4px',
    right: '4px',
    display: 'flex',
    justifyContent: 'space-between',
    gap: space.spaceXs,
    pointerEvents: 'none',
    color: colors.outline,
  },
  axis: { display: 'flex', justifyContent: 'space-between', gap: space.spaceXs, color: colors.outline, marginTop: '2px' },
  legend: { display: 'flex', flexWrap: 'wrap', gap: `2px ${space.spaceSm}`, alignItems: 'center', color: colors.onSurfaceVariant },
  key: { display: 'inline-flex', alignItems: 'center', gap: '4px', minWidth: 0 },
  swatch: { display: 'inline-block', width: '14px', height: '0', borderTopWidth: 2, borderTopStyle: 'solid' },
  grid: { stroke: colors.outlineVariant, strokeWidth: 1, opacity: 0.35 },
  hit: { fill: 'transparent', ':hover': { fill: `color-mix(in srgb, ${colors.onSurface} 7%, transparent)` } },
  empty: { color: colors.outline, padding: space.spaceSm, margin: 0 },
  wrap: { display: 'flex', flexDirection: 'column', gap: '3px', minWidth: 0 },
});

// stroke, fill and legend swatch per slot (index SLOTS = OTHER)
const tone = stylex.create({
  s0: { stroke: colors.primaryContainer, fill: colors.primaryContainer, borderTopColor: colors.primaryContainer },
  s1: { stroke: colors.tertiaryContainer, fill: colors.tertiaryContainer, borderTopColor: colors.tertiaryContainer },
  s2: { stroke: colors.secondaryContainer, fill: colors.secondaryContainer, borderTopColor: colors.secondaryContainer },
  s3: { stroke: colors.secondary, fill: colors.secondary, borderTopColor: colors.secondary },
  s4: { stroke: colors.outline, fill: colors.outline, borderTopColor: colors.outline },
});
const TONES = [tone.s0, tone.s1, tone.s2, tone.s3, tone.s4];
// secondary encoding: one dash pattern per slot (solid first)
const DASH = [undefined, '6 3', '2 3', '8 3 2 3', '1 3'];
const SWATCH_STYLE = ['solid', 'dashed', 'dotted', 'dashed', 'dotted'] as const;

export type Serie = { name: string; values: (number | null)[]; slot: number };

const clampSlot = (k: number) => Math.max(0, Math.min(TONES.length - 1, k));

export function Legend({ items }: { items: { name: string; slot: number; extra?: string }[] }) {
  if (items.length < 2) return null;
  return (
    <div {...stylex.props(text.labelXs, s.legend)}>
      {items.map((it) => (
        <span key={it.name} {...stylex.props(s.key)}>
          <i aria-hidden {...stylex.props(s.swatch, TONES[clampSlot(it.slot)])} style={{ borderTopStyle: SWATCH_STYLE[clampSlot(it.slot)] }} />
          {it.name}
          {it.extra ? ` · ${it.extra}` : ''}
        </span>
      ))}
    </div>
  );
}

function Axis({ t, bucketS }: { t: number[]; bucketS: number }) {
  if (!t.length) return null;
  const mid = t[Math.floor(t.length / 2)] as number;
  return (
    <div {...stylex.props(text.labelXs, s.axis)}>
      <span>{bucketLabel(t[0] as number, bucketS)}</span>
      <span>{bucketLabel(mid, bucketS)}</span>
      <span>{bucketLabel(t[t.length - 1] as number, bucketS)} →</span>
    </div>
  );
}

/** Title text for one bucket: its time and every series' value there. */
function hoverTitle(ts: number, bucketS: number, series: Serie[], i: number, fmt: (x: number) => string): string {
  const vals = series.map((x) => `${x.name} ${ok(x.values[i]) ? fmt(x.values[i] as number) : '—'}`);
  return `${bucketLabel(ts, bucketS)} · ${vals.join(' · ')}`;
}

const defaultFmt = (x: number) => (x >= 100 ? String(Math.round(x)) : x.toFixed(1));

/**
 * Lines over time buckets: one per series, a null breaks the line (an
 * unmeasured bucket is never drawn as zero). Points mark isolated values.
 */
export function LineChart({
  t,
  bucketS,
  series,
  height = 90,
  unit,
  label,
  max,
  fmt: fmtIn,
}: {
  t: number[];
  bucketS: number;
  series: Serie[];
  height?: number;
  unit: string;
  label: string;
  max?: number;
  fmt?: (x: number) => string;
}) {
  const fmt = fmtIn ?? defaultFmt;
  const all = series.flatMap((x) => x.values);
  const has = all.some(ok);
  const top = max ?? niceMax(all, 1);
  const n = Math.max(t.length, 1);
  const bw = W / n;
  const X = (i: number) => (i + 0.5) * bw;
  const Y = (v: number) => height - (Math.min(v, top) / top) * (height - 4) - 2;
  return (
    <div {...stylex.props(s.wrap)}>
      <div {...stylex.props(s.box)}>
        <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" height={height} {...stylex.props(s.svg)} role="img" aria-label={label}>
          <line x1={0} x2={W} y1={Y(top / 2)} y2={Y(top / 2)} vectorEffect="non-scaling-stroke" {...stylex.props(s.grid)} />
          {series.map((x) => {
            let d = '';
            let pen = false;
            const dots: number[] = [];
            x.values.forEach((v, i) => {
              if (!ok(v)) {
                pen = false;
                return;
              }
              d += `${pen ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`;
              const prev = x.values[i - 1];
              const next = x.values[i + 1];
              if (!ok(prev) && !ok(next)) dots.push(i);
              pen = true;
            });
            const k = clampSlot(x.slot);
            return (
              <g key={x.name}>
                <path d={d} fill="none" strokeWidth={2} strokeDasharray={DASH[k]} vectorEffect="non-scaling-stroke" {...stylex.props(TONES[k])} style={{ fill: 'none' }} />
                {dots.map((i) => (
                  <rect key={i} x={X(i) - 2} y={Y(x.values[i] as number) - 2} width={4} height={4} {...stylex.props(TONES[k])} />
                ))}
              </g>
            );
          })}
          {t.map((ts, i) => (
            <rect key={ts} x={i * bw} y={0} width={bw} height={height} {...stylex.props(s.hit)}>
              <title>{hoverTitle(ts, bucketS, series, i, fmt)}</title>
            </rect>
          ))}
        </svg>
        <div {...stylex.props(text.labelXs, s.tag)}>
          <span>{label.toUpperCase()}</span>
          <span>
            {fmt(top)} {unit}
          </span>
        </div>
        {!has && <p {...stylex.props(text.labelXs, s.empty)}>NO MEASUREMENT IN THIS WINDOW</p>}
      </div>
      <Axis t={t} bucketS={bucketS} />
      <Legend items={series.map((x) => ({ name: x.name, slot: x.slot }))} />
    </div>
  );
}

/** Counts per bucket, stacked by series, with a 1-unit gap between stacks. */
export function StackedBars({ t, bucketS, series, height = 80, label, unit = '' }: { t: number[]; bucketS: number; series: Serie[]; height?: number; label: string; unit?: string }) {
  const n = Math.max(t.length, 1);
  const sums = t.map((_, i) => series.reduce((a, x) => a + (ok(x.values[i]) ? (x.values[i] as number) : 0), 0));
  const top = niceMax(sums, 1);
  const bw = W / n;
  const gap = Math.min(2, bw * 0.2);
  const sum = sums.reduce((a, b) => a + b, 0);
  return (
    <div {...stylex.props(s.wrap)}>
      <div {...stylex.props(s.box)}>
        <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" height={height} {...stylex.props(s.svg)} role="img" aria-label={label}>
          {t.map((ts, i) => {
            let y = height;
            return (
              <g key={ts}>
                {series.map((x) => {
                  const v = ok(x.values[i]) ? (x.values[i] as number) : 0;
                  if (v <= 0) return null;
                  const h = Math.max(1, (v / top) * (height - 4));
                  y -= h;
                  return <rect key={x.name} x={i * bw + gap / 2} y={y} width={Math.max(bw - gap, 0.6)} height={Math.max(h - 0.5, 0.5)} {...stylex.props(TONES[clampSlot(x.slot)])} />;
                })}
                <rect x={i * bw} y={0} width={bw} height={height} {...stylex.props(s.hit)}>
                  <title>{hoverTitle(ts, bucketS, series, i, (v) => String(v))}</title>
                </rect>
              </g>
            );
          })}
        </svg>
        <div {...stylex.props(text.labelXs, s.tag)}>
          <span>{label.toUpperCase()}</span>
          <span>
            {top}
            {unit ? ` ${unit}` : ''} / BUCKET
          </span>
        </div>
        {!sum && <p {...stylex.props(text.labelXs, s.empty)}>NONE IN THIS WINDOW</p>}
      </div>
      <Axis t={t} bucketS={bucketS} />
      <Legend items={series.map((x) => ({ name: x.name, slot: x.slot, extra: String(x.values.reduce<number>((a, b) => a + (ok(b) ? b : 0), 0)) }))} />
    </div>
  );
}

/** A 10-bin histogram of values in [0, 1] (confidence, a noul, a need). */
export function Histogram({ bins, label, slot = 1, height = 46, note }: { bins: number[] | null | undefined; label: string; slot?: number; height?: number; note?: ReactNode }) {
  const b = bins ?? [];
  const top = Math.max(1, ...b);
  const n = Math.max(b.length, 1);
  const bw = W / n;
  const sum = b.reduce((a, x) => a + x, 0);
  return (
    <div {...stylex.props(s.wrap)}>
      <div {...stylex.props(s.box)}>
        <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" height={height} {...stylex.props(s.svg)} role="img" aria-label={`${label}: ${b.join(', ')}`}>
          {b.map((v, i) => {
            const h = v ? Math.max(1.5, (v / top) * (height - 14)) : 0;
            return (
              <g key={i}>
                {v ? <rect x={i * bw + 2} y={height - h} width={bw - 4} height={h} {...stylex.props(TONES[clampSlot(slot)])} /> : null}
                <rect x={i * bw} y={0} width={bw} height={height} {...stylex.props(s.hit)}>
                  <title>{`${binLabel(i, n)} · ${v}`}</title>
                </rect>
              </g>
            );
          })}
        </svg>
        <div {...stylex.props(text.labelXs, s.tag)}>
          <span>{label.toUpperCase()}</span>
          <span>n {sum}</span>
        </div>
      </div>
      <div {...stylex.props(text.labelXs, s.axis)}>
        <span>0</span>
        <span>{note ?? '0.5'}</span>
        <span>1</span>
      </div>
    </div>
  );
}
