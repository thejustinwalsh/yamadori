// Shared pieces of the benchmark page: the arms legend, interval bars, the
// paired table and the mechanism-health table. Every one draws numbers the
// server already computed and prints the n beside them.
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import type { EmptySection, ErrorSection, LegendRow } from '../../api/types';
import { ago, pValue } from '../../format';
import { colors, space } from '../../tokens/tokens.stylex';
import { Chip, Label, layout, type Tone } from '../../ui/primitives';
import { RateBar } from '../../ui/RateBar';
import { StateView } from '../../ui/StateView';
import { Table } from '../../ui/Table';
import { text } from '../../ui/text';
import type { MechRow } from './model';

export const ps = stylex.create({
  what: { margin: 0, color: colors.onSurfaceVariant, maxWidth: '80ch' },
  h3: { margin: 0, marginTop: space.spaceSm, color: colors.onSurface },
  small: { margin: 0, color: colors.onSurfaceVariant },
  off: { color: colors.outline },
  on: { color: colors.onSurface },
  auto: { color: colors.tertiaryContainer },
  bars: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 14rem) minmax(90px, 1fr) max-content', '@media (max-width: 599px)': 'minmax(0, 1fr)' },
    columnGap: space.spaceSm,
    rowGap: '6px',
    alignItems: 'center',
    minWidth: 0,
  },
  barLabel: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: colors.onSurfaceVariant },
  barLabelOurs: { color: colors.onSurface, fontWeight: 600 },
  barValue: { fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', color: colors.onSurfaceVariant, textAlign: 'right' },
  barValueOurs: { color: colors.onSurface },
  groupHead: { gridColumn: '1 / -1', color: colors.outline, marginTop: space.spaceXs },
  // One bar. Desktop: its three cells join the parent grid's columns, so
  // every bar lines up. Phone: label and value on one line, the bar under it.
  barRow: {
    display: { default: 'contents', '@media (max-width: 599px)': 'grid' },
    gridTemplateColumns: { default: null, '@media (max-width: 599px)': 'minmax(0, 1fr) max-content' },
    gridTemplateAreas: { default: null, '@media (max-width: 599px)': '"l v" "b b"' },
    columnGap: space.spaceSm,
    rowGap: '2px',
  },
  areaL: { gridArea: { default: 'auto', '@media (max-width: 599px)': 'l' } },
  areaB: { gridArea: { default: 'auto', '@media (max-width: 599px)': 'b' } },
  areaV: { gridArea: { default: 'auto', '@media (max-width: 599px)': 'v' } },
  details: { display: 'flex', flexDirection: 'column', gap: space.spaceSm },
  summary: { cursor: 'pointer', color: colors.onSurfaceVariant, minHeight: '24px' },
  line: { margin: 0, color: colors.onSurfaceVariant, overflowWrap: 'anywhere' },
  warnLine: { margin: 0, color: colors.secondary, overflowWrap: 'anywhere' },
  link: { color: colors.onSurfaceVariant, overflowWrap: 'anywhere' },
});

export function What({ children }: { children: ReactNode }) {
  return <p {...stylex.props(text.bodySm, ps.what)}>{children}</p>;
}

export function H3({ children }: { children: ReactNode }) {
  return <h3 {...stylex.props(text.labelMd, ps.h3)}>{children}</h3>;
}

export function Chips({ children }: { children: ReactNode }) {
  return <div {...stylex.props(layout.rowWrap)}>{children}</div>;
}

export function Lines({ items, warn }: { items: ReactNode[]; warn?: boolean }) {
  if (!items.length) return null;
  return (
    <div {...stylex.props(layout.stack)}>
      {items.map((x, i) => (
        <p key={i} {...stylex.props(text.bodySm, warn ? ps.warnLine : ps.line)}>
          {x}
        </p>
      ))}
    </div>
  );
}

/** The non-ready states of a section, from the server's own words. */
export function NotReady({ sec, what }: { sec: EmptySection | ErrorSection | undefined; what: string }) {
  if (!sec) return <StateView kind="empty" title={`${what} · not in the payload (needs proxy restart)`} />;
  if (sec.state === 'error') {
    const e = sec as ErrorSection;
    return <StateView kind="error" title={`${e.component} raised`} detail={`${e.error} · check ${e.check}`} />;
  }
  const e = sec as EmptySection;
  return (
    <StateView
      kind="empty"
      title={`${e.file}${e.exists ? '' : ' · not found'}${e.note ? ` · ${e.note}` : ''}`}
      detail={e.how}
    />
  );
}

export function FileAge({ file, age, extra }: { file: string; age: number | null | undefined; extra?: string }) {
  return (
    <Label>
      {file}
      {age !== null && age !== undefined ? ` · written ${ago(age)} ago` : ''}
      {extra ? ` · ${extra}` : ''}
    </Label>
  );
}

// ---------------------------------------------------------------- legend --

function Flag({ v }: { v: string | undefined | null }) {
  if (v === undefined || v === null) return <span {...stylex.props(ps.off)}>—</span>;
  const st = v === 'off' || v === '1' ? ps.off : v.startsWith('auto') ? ps.auto : ps.on;
  return <span {...stylex.props(st)}>{v}</span>;
}

/** Each arm in README "Effort tiers" columns. "auto" = allowed; selection decides per request. */
export function ArmsLegend({ rows }: { rows: LegendRow[] | undefined }) {
  if (!rows?.length) return null;
  return (
    <Table
      rows={rows}
      rowKey={(r) => r.arm}
      caption="arms: what is switched on in each"
      columns={[
        { key: 'arm', head: 'arm', cell: (r) => <strong>{r.arm}</strong> },
        {
          key: 'what',
          head: 'what it is',
          cell: (r) => (
            <span>
              {r.note || (r.kind === 'tier' ? `tier ${r.tier}` : r.kind === 'features' ? 'forced by X-Yamadori-Features' : '')}
              {r.reasoning_cap ? ` · thinking cap ${r.reasoning_cap}` : ''}
            </span>
          ),
        },
        { key: 'eff', head: 'thinking sent', cell: (r) => <Flag v={r.thinking_sent ?? undefined} /> },
        { key: 'ret', head: 'retrieval', cell: (r) => <Flag v={r.retrieval} /> },
        { key: 'hin', head: 'hints', cell: (r) => <Flag v={r.hints} /> },
        { key: 'fan', head: 'fan-out', cell: (r) => <Flag v={r.fanout} /> },
        { key: 'dt', head: 'deep thinking', cell: (r) => <Flag v={r.deep_thinking} /> },
        {
          key: 'chk',
          head: 'check',
          cell: (r) => <Flag v={r.client_tool ? `${r.client_tool}` : r.check === 'on' && r.repair === 'on' ? 'check_code + repair' : r.check === 'on' ? 'check_code' : r.check} />,
        },
      ]}
    />
  );
}

// ------------------------------------------------------------------ bars --

export type Bar = {
  label: string;
  /** 0..1 on the shared axis */
  value: number | null;
  lo?: number | null;
  hi?: number | null;
  text: string;
  ours?: boolean;
  tone?: Tone;
  group?: string;
};

/**
 * Horizontal bars on one fixed axis, interval whiskers where there is an
 * interval. Ours in the live colour; published reference rows muted.
 */
export function CompareBars({ bars, label }: { bars: Bar[]; label: string }) {
  if (!bars.length) return null;
  let group: string | undefined;
  const cells: ReactNode[] = [];
  bars.forEach((b, i) => {
    if (b.group && b.group !== group) {
      group = b.group;
      cells.push(
        <span key={`g${i}`} {...stylex.props(text.labelXs, ps.groupHead)}>
          {b.group}
        </span>,
      );
    }
    cells.push(
      <div key={`r${i}`} {...stylex.props(ps.barRow)}>
        <span title={b.label} {...stylex.props(text.bodySm, ps.barLabel, b.ours && ps.barLabelOurs, ps.areaL)}>
          {b.label}
        </span>
        <div {...stylex.props(ps.areaB)}>
          <RateBar rate={b.value} lo={b.lo ?? undefined} hi={b.hi ?? undefined} tone={b.tone ?? (b.ours ? 'moss' : 'muted')} label={`${b.label}: ${b.text}`} />
        </div>
        <span {...stylex.props(text.bodySm, ps.barValue, b.ours && ps.barValueOurs, ps.areaV)}>{b.text}</span>
      </div>,
    );
  });
  return (
    <div role="group" aria-label={label} {...stylex.props(ps.bars)}>
      {cells}
    </div>
  );
}

// ---------------------------------------------------------------- paired --

export type PairRow = {
  key: string;
  a: string;
  b: string;
  scope?: string;
  n: number | null;
  aPass?: number | null;
  bPass?: number | null;
  bOnly: number | null;
  aOnly: number | null;
  p: number | null;
  pLabel?: string;
  floor?: number | null;
  diff?: string;
};

/** Bare against each augmented arm on the items both scored; the discordant pairs are what the exact test runs on. */
export function PairedTable({ rows, caption }: { rows: PairRow[]; caption: string }) {
  if (!rows.length) return null;
  const anyDiff = rows.some((r) => r.diff);
  const anyScope = rows.some((r) => r.scope);
  const anyPass = rows.some((r) => r.aPass !== undefined);
  return (
    <Table
      rows={rows}
      rowKey={(r) => r.key}
      caption={caption}
      columns={[
        { key: 'ab', head: 'comparison', cell: (r) => `${r.b} vs ${r.a}` },
        ...(anyScope ? [{ key: 'sc', head: 'on', cell: (r: PairRow) => r.scope ?? '' }] : []),
        { key: 'n', head: 'paired n', num: true, cell: (r) => r.n ?? '—' },
        ...(anyPass
          ? [{ key: 'pp', head: 'pass bare / arm', num: true, cell: (r: PairRow) => `${r.aPass ?? '—'} / ${r.bPass ?? '—'}` }]
          : []),
        ...(anyDiff ? [{ key: 'df', head: 'arm − bare [95% CI]', num: true, cell: (r: PairRow) => r.diff ?? '—' }] : []),
        { key: 'bo', head: 'arm only', num: true, cell: (r) => r.bOnly ?? '—' },
        { key: 'ao', head: 'bare only', num: true, cell: (r) => r.aOnly ?? '—' },
        {
          key: 'd',
          head: 'discordant',
          num: true,
          cell: (r) => (r.bOnly === null || r.aOnly === null ? '—' : r.bOnly + r.aOnly),
        },
        { key: 'p', head: 'McNemar exact p', num: true, cell: (r) => (r.p === null ? '—' : `${pValue(r.p)}${r.pLabel ? ` ${r.pLabel}` : ''}`) },
        {
          key: 'f',
          head: 'fewest discordant that can reach p<α',
          num: true,
          cell: (r) => (r.floor === undefined || r.floor === null ? '—' : r.floor),
        },
      ]}
    />
  );
}

// ------------------------------------------------------------- mechanisms --

function cell(v: number | null | undefined, n: number) {
  if (v === undefined) return <span {...stylex.props(ps.off)}>—</span>;
  if (v === null) return <span {...stylex.props(ps.off)}>not exposed</span>;
  if (!n) return <span {...stylex.props(ps.off)}>0/0</span>;
  return `${v}/${n}`;
}

/** Per arm, per mechanism: how often it was allowed, chosen, ran and produced data. */
export function MechTable({ rows, caption, stackErrors }: { rows: MechRow[]; caption: string; stackErrors?: Record<string, number> }) {
  if (!rows.length) return null;
  return (
    <>
      <Table
        rows={rows}
        rowKey={(r) => `${r.arm}|${r.mechanism}`}
        caption={caption}
        columns={[
          { key: 'arm', head: 'arm', cell: (r) => r.arm },
          { key: 'm', head: 'mechanism', cell: (r) => r.mechanism },
          { key: 'n', head: 'of', num: true, cell: (r) => `${r.n} ${r.unit}` },
          { key: 'al', head: 'allowed', num: true, cell: (r) => cell(r.allowed, r.n) },
          { key: 'de', head: 'chosen', num: true, cell: (r) => cell(r.decided, r.n) },
          { key: 'ra', head: 'ran', num: true, cell: (r) => cell(r.ran, r.base ?? r.n) },
          { key: 'pr', head: 'produced data', num: true, cell: (r) => cell(r.produced, r.base ?? r.n) },
          { key: 'x', head: 'detail', cell: (r) => r.extra ?? '' },
        ]}
      />
      {stackErrors && Object.values(stackErrors).some((v) => v > 0) ? (
        <Chips>
          {Object.entries(stackErrors)
            .filter(([, v]) => v > 0)
            .map(([a, v]) => (
              <Chip key={a} tone="crimson">
                {a} · {v} STACK ERROR{v === 1 ? '' : 'S'} · NOT SCORED
              </Chip>
            ))}
        </Chips>
      ) : null}
    </>
  );
}

/** "stop 21 · length 1" */
export const tally = (m: Record<string, number> | undefined | null): string =>
  m && Object.keys(m).length ? Object.entries(m).map(([k, v]) => `${k} ${v}`).join(' · ') : '—';

export function Collapse({ summary, children, open }: { summary: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} {...stylex.props(ps.details)}>
      <summary {...stylex.props(text.labelMd, ps.summary)}>{summary}</summary>
      <div {...stylex.props(layout.stackSm)}>{children}</div>
    </details>
  );
}
