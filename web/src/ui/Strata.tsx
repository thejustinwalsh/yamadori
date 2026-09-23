// 層 TIER STRATA: result-tier hits across find_by_meaning searches over a
// trailing window (vitals.strata). TAPROOT: a declaration with the searched
// name is here. BRANCH: both retrievers agree. SHOOT: one retriever only.
// Drawn as a soil cross-section, shallowest on top, each layer as thick as
// its share of the hits.
import * as stylex from '@stylexjs/stylex';
import type { Strata } from '../api/types';
import { n } from '../format';
import { color as token } from '../tokens/design';
import { colors, space } from '../tokens/tokens.stylex';
import { need } from './ErrorBoundary';
import { Panel } from './Panel';
import { Chip, layout, Stat } from './primitives';
import { StateView } from './StateView';

const SRC = '/dash/api/vitals · strata';
const isCount = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0;

export function windowLabel(seconds: number): string {
  if (!isCount(seconds) || seconds <= 0) return '—';
  if (seconds % 86400 === 0) return seconds === 86400 ? '24 h' : `${seconds / 86400} d`;
  if (seconds % 3600 === 0) return `${seconds / 3600} h`;
  return `${Math.round(seconds / 60)} min`;
}

const s = stylex.create({
  svg: { width: '100%', height: 'auto', display: 'block' },
  line: { margin: 0, color: colors.onSurfaceVariant },
  gap: { gap: space.spaceSm },
});

const LAYERS = [
  { key: 'shoot', label: 'SHOOT', fill: token.secondary, ink: token.onSecondary },
  { key: 'branch', label: 'BRANCH', fill: token.tertiaryContainer, ink: token.onTertiary },
  { key: 'taproot', label: 'TAPROOT', fill: token.primaryContainer, ink: token.onPrimary },
] as const;

const W = 520;

/** Layer geometry, top to bottom. Pure, outside the component: the compiler
 * will not compile a render that reassigns a local after it returns. */
function layers(st: Strata) {
  const total = st.taproot + st.branch + st.shoot;
  const MIN = 26;
  const SPAN = 150;
  const rows: { key: string; label: string; fill: string; ink: string; count: number; share: number; y: number; h: number }[] = [];
  let y = 12;
  for (const l of LAYERS) {
    const count = st[l.key];
    const share = total > 0 ? count / total : 0;
    const h = MIN + SPAN * share;
    rows.push({ ...l, count, share, y, h });
    y += h + 3;
  }
  return { rows, H: y + 6 };
}

function CrossSection({ st }: { st: Strata }) {
  const { rows, H } = layers(st);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} {...stylex.props(s.svg)} role="img" aria-label={rows.map((r) => `${r.label} ${r.count}`).join(', ')}>
      <defs>
        <pattern id="soil" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill={token.surfaceContainerLowest} />
          <line x1="0" y1="0" x2="0" y2="6" stroke={token.surfaceContainerHigh} strokeWidth="2" />
        </pattern>
      </defs>
      <line x1="0" y1="6" x2={W} y2="6" stroke={token.outline} strokeDasharray="3 5" />
      {rows.map((r) => (
        <g key={r.key}>
          <rect x="0" y={r.y} width={W} height={r.h} fill="url(#soil)" />
          <rect x="0" y={r.y} width={Math.max(2, W * r.share)} height={r.h} fill={r.fill} fillOpacity={0.85} />
          <text x="10" y={r.y + r.h / 2 + 4} fontFamily="ui-monospace, monospace" fontSize="12" fontWeight="700" fill={r.share > 0.18 ? r.ink : r.fill}>
            {r.label}
          </text>
          <text x={W - 10} y={r.y + r.h / 2 + 4} textAnchor="end" fontFamily="ui-monospace, monospace" fontSize="12" fill={token.onSurface}>
            {n(r.count)} · {(r.share * 100).toFixed(1)}%
          </text>
        </g>
      ))}
    </svg>
  );
}

export function StrataPanel({ strata, present, fill }: { strata: Strata | null | undefined; present: boolean; fill?: boolean }) {
  if (!present || strata === null || strata === undefined) {
    return (
      <Panel kanji="層" title="TIER STRATA" tag="NOT REPORTED" tagTone="muted" fill={fill}>
        <StateView kind="inert" title="not reported by this server" />
      </Panel>
    );
  }
  const searches = need(strata, 'searches', SRC, isCount);
  const win = windowLabel(need(strata, 'window_seconds', SRC, isCount));
  if (searches === 0) {
    return (
      <Panel kanji="層" title="TIER STRATA" tag={win} tagTone="muted" fill={fill}>
        <StateView kind="empty" title={`no searches in the last ${win}`} />
      </Panel>
    );
  }
  const st: Strata = {
    taproot: need(strata, 'taproot', SRC, isCount),
    branch: need(strata, 'branch', SRC, isCount),
    shoot: need(strata, 'shoot', SRC, isCount),
    searches,
    window_seconds: strata.window_seconds,
  };
  const hits = st.taproot + st.branch + st.shoot;
  return (
    <Panel kanji="層" title="TIER STRATA" tag={`${n(searches)} SEARCHES · ${win}`} tagTone="moss" edge="moss" fill={fill}>
      <CrossSection st={st} />
      <div {...stylex.props(layout.grid3)}>
        <Stat label="SEARCHES" value={n(searches)} sub={win} />
        <Stat label="HITS" value={n(hits)} sub={`${(hits / searches).toFixed(1)} per search`} />
        <Stat label="TAPROOT SHARE" value={hits ? `${((100 * st.taproot) / hits).toFixed(0)}%` : '—'} tone="moss" />
      </div>
      <div {...stylex.props(layout.rowWrap)}>
        <Chip tone="moss">TAPROOT · NAMED DECLARATION</Chip>
        <Chip tone="cyan">BRANCH · BOTH RETRIEVERS</Chip>
        <Chip tone="rose">SHOOT · ONE RETRIEVER</Chip>
      </div>
    </Panel>
  );
}
