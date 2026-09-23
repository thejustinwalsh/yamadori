// Small parts extracted from design/mockups/*.html. The mockups hardcode hex
// (zero CSS variables); every colour here is a frontmatter token, with the
// mockups' `/35`-style opacity suffixes reproduced by color-mix.
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import { colors, radius, space } from '../tokens/tokens.stylex';
import { text } from './text';

export type Tone = 'moss' | 'cyan' | 'crimson' | 'muted' | 'rose';

const toneFg = stylex.create({
  moss: { color: colors.primaryContainer },
  cyan: { color: colors.tertiaryContainer },
  crimson: { color: colors.secondary },
  rose: { color: colors.secondary },
  muted: { color: colors.outline },
});

const chip = stylex.create({
  base: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: space.spaceXs,
    paddingInline: space.spaceXs,
    paddingBlock: '1px',
    borderWidth: 1,
    borderStyle: 'solid',
    borderRadius: 0, // DESIGN.md: pills are banned; architectural rectangles
    whiteSpace: 'nowrap',
  },
  moss: {
    borderColor: `color-mix(in srgb, ${colors.primaryContainer} 40%, transparent)`,
    backgroundColor: `color-mix(in srgb, ${colors.primaryContainer} 10%, transparent)`,
  },
  cyan: {
    borderColor: `color-mix(in srgb, ${colors.tertiaryContainer} 40%, transparent)`,
    backgroundColor: `color-mix(in srgb, ${colors.tertiaryContainer} 10%, transparent)`,
  },
  crimson: {
    borderColor: `color-mix(in srgb, ${colors.secondary} 40%, transparent)`,
    backgroundColor: `color-mix(in srgb, ${colors.secondaryContainer} 20%, transparent)`,
  },
  rose: {
    borderColor: `color-mix(in srgb, ${colors.secondary} 40%, transparent)`,
    backgroundColor: `color-mix(in srgb, ${colors.secondaryContainer} 12%, transparent)`,
  },
  muted: {
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 60%, transparent)`,
    backgroundColor: colors.surfaceContainerLowest,
  },
});

export function Chip({ tone = 'moss', children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} {...stylex.props(text.labelXs, chip.base, chip[tone], toneFg[tone])}>
      {children}
    </span>
  );
}

const lbl = stylex.create({ base: { color: colors.outline } });
export function Label({ children }: { children: ReactNode }) {
  return <span {...stylex.props(text.labelXs, lbl.base)}>{children}</span>;
}

const stat = stylex.create({
  box: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    padding: space.spaceXs,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`,
    minWidth: 0,
  },
  value: { color: colors.primary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  sub: { color: colors.onSurfaceVariant },
});

export function Stat({ label, value, sub, tone }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: Tone }) {
  return (
    <div {...stylex.props(stat.box)}>
      <Label>{label}</Label>
      <span {...stylex.props(text.titleMd, text.num, stat.value, tone && toneFg[tone])}>{value}</span>
      {sub !== undefined && <span {...stylex.props(text.labelXs, stat.sub)}>{sub}</span>}
    </div>
  );
}

const meter = stylex.create({
  track: {
    position: 'relative',
    height: '6px',
    width: '100%',
    backgroundColor: colors.surfaceContainerHigh,
    overflow: 'hidden',
  },
  hatch: {
    backgroundImage: `repeating-linear-gradient(-45deg, ${colors.surfaceContainerLow}, ${colors.surfaceContainerLow} 4px, ${colors.surfaceContainerHigh} 4px, ${colors.surfaceContainerHigh} 8px)`,
  },
  fill: { position: 'absolute', insetBlock: 0, left: 0, transition: 'width 600ms cubic-bezier(.2,.8,.2,1)' },
  moss: { backgroundColor: colors.primaryContainer, boxShadow: `0 0 10px color-mix(in srgb, ${colors.primaryContainer} 45%, transparent)` },
  cyan: { backgroundColor: colors.tertiaryContainer, boxShadow: `0 0 10px color-mix(in srgb, ${colors.tertiaryContainer} 45%, transparent)` },
  crimson: { backgroundColor: colors.secondaryContainer, boxShadow: `0 0 10px color-mix(in srgb, ${colors.secondaryContainer} 55%, transparent)` },
  rose: { backgroundColor: colors.secondary },
  muted: { backgroundColor: colors.outline },
  width: (pct: number) => ({ width: `${pct}%` }),
});

/** A bar for a MEASURED fraction. `value` null draws the empty hatched track. */
export function Meter({ value, tone = 'moss', label }: { value: number | null; tone?: Tone; label: string }) {
  const pct = value === null ? 0 : Math.max(0, Math.min(100, value * 100));
  return (
    <div role="img" aria-label={label} {...stylex.props(meter.track, meter.hatch)}>
      {value !== null && <span {...stylex.props(meter.fill, meter[tone], meter.width(pct))} />}
    </div>
  );
}

const seg = stylex.create({
  row: { display: 'flex', height: '10px', gap: '2px', width: '100%' },
  part: { height: '100%', minWidth: '2px', transition: 'flex-grow 600ms ease' },
  grow: (n: number) => ({ flexGrow: n }),
});

/** Segmented split, e.g. the KV pool: main / deep thinking / reserve. */
export function SplitBar({ parts, label }: { parts: { value: number; tone: Tone | 'hatch' }[]; label: string }) {
  return (
    <div role="img" aria-label={label} {...stylex.props(seg.row)}>
      {parts.map((p, i) => (
        <span
          key={i}
          {...stylex.props(seg.part, seg.grow(Math.max(p.value, 0)), p.tone === 'hatch' ? meter.hatch : meter[p.tone])}
        />
      ))}
    </div>
  );
}

const dot = stylex.create({
  base: { display: 'inline-block', width: '7px', height: '7px', flexShrink: 0 },
  live: {
    backgroundColor: colors.primaryContainer,
    boxShadow: `0 0 8px ${colors.primaryContainer}`,
    animationName: stylex.keyframes({ '0%': { opacity: 1 }, '50%': { opacity: 0.35 }, '100%': { opacity: 1 } }),
    animationDuration: '2.4s',
    animationIterationCount: 'infinite',
  },
  stale: { backgroundColor: colors.secondary },
  dead: { backgroundColor: colors.outlineVariant },
  down: { backgroundColor: colors.secondaryContainer, boxShadow: `0 0 8px ${colors.secondaryContainer}` },
});
export function Dot({ state }: { state: 'live' | 'stale' | 'dead' | 'down' }) {
  return <span aria-hidden {...stylex.props(dot.base, dot[state])} />;
}

const row = stylex.create({
  base: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: space.spaceSm,
    padding: space.spaceXs,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 20%, transparent)`,
    minWidth: 0,
  },
  bad: {
    borderColor: `color-mix(in srgb, ${colors.secondaryContainer} 55%, transparent)`,
    boxShadow: `inset 2px 0 0 ${colors.secondaryContainer}`,
  },
});
export function Row({ children, bad }: { children: ReactNode; bad?: boolean }) {
  return <div {...stylex.props(text.labelXs, row.base, bad && row.bad)}>{children}</div>;
}

export const layout = stylex.create({
  stack: { display: 'flex', flexDirection: 'column', gap: space.spaceXs },
  stackSm: { display: 'flex', flexDirection: 'column', gap: space.spaceSm },
  rowWrap: { display: 'flex', flexWrap: 'wrap', gap: space.spaceXs, alignItems: 'center' },
  between: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: space.spaceSm },
  grid2: { display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: space.spaceXs },
  grid3: { display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: space.spaceXs },
  grid4: {
    display: 'grid',
    gridTemplateColumns: { default: 'repeat(4, minmax(0, 1fr))', '@media (max-width: 520px)': 'repeat(2, minmax(0, 1fr))' },
    gap: space.spaceXs,
  },
  truncate: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 },
  grow: { flexGrow: 1, minWidth: 0 },
  mono: { fontVariantNumeric: 'tabular-nums' },
  rounded: { borderRadius: radius.base },
});
