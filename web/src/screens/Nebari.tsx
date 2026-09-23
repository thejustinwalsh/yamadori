// 根張り NEBARI — the root flare. What the model can draw on: the recipe
// corpus behind hints, and the source trees indexed for retrieval. Every
// root drawn here is a counted row in a real payload.
import * as stylex from '@stylexjs/stylex';
import { PATHS, useShared } from '../api/data';
import { KNOWN_VOID } from '../api/provenance';
import type { Results, RetrievalSection } from '../api/types';
import { usePoll } from '../api/usePoll';
import { n, pct } from '../format';
import { color as token } from '../tokens/design';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { StrataPanel } from '../ui/Strata';
import { Chip, Label, layout, Meter, Stat } from '../ui/primitives';
import { RateBar } from '../ui/RateBar';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';

type Stats = {
  total: number;
  by_state: Record<string, number>;
  by_domain: Record<string, number>;
  by_domain_tag: Record<string, number>;
  untagged_domains: number;
  over_40_words: number;
  with_trigger: number;
  without_trigger: number;
  files: string[];
};

const s = stylex.create({
  svg: { width: '100%', height: 'auto', display: 'block', backgroundColor: colors.surfaceContainerLowest },
  note: { margin: 0, color: colors.onSurfaceVariant },
  warn: { margin: 0, color: colors.secondary },
  tier: {
    display: 'grid',
    gridTemplateColumns: '72px minmax(0,1fr)',
    gap: space.spaceSm,
    padding: space.spaceSm,
    backgroundColor: colors.surfaceContainerLowest,
    borderLeftWidth: 2,
    borderLeftStyle: 'dashed',
    borderLeftColor: colors.outlineVariant,
  },
  bar: { display: 'grid', gridTemplateColumns: 'minmax(90px, 140px) minmax(0,1fr) 44px', gap: space.spaceSm, alignItems: 'center' },
});

/**
 * Root flare: one root per corpus domain tag, length by log(count), and one
 * heavy root per retrieval source, width by rows scored. Deterministic
 * layout: angle from the sorted index, never random.
 */
function RootFlare({ tags, sources }: { tags: [string, number][]; sources: { source: string; n: number; emb: number }[] }) {
  const W = 980;
  const H = 420;
  const cx = W / 2;
  const cy = 50;
  const maxTag = Math.max(1, ...tags.map(([, c]) => c));
  const maxSrc = Math.max(1, ...sources.map((x) => x.n));
  // Heavy source roots straight down the middle; tag roots fan out either
  // side, largest nearest the trunk, like a real nebari.
  const tagRoots = tags.map(([t, c]) => ({ kind: 'tag' as const, source: t, n: c, emb: 0 }));
  const left = tagRoots.filter((_, i) => i % 2 === 0).reverse();
  const right = tagRoots.filter((_, i) => i % 2 === 1);
  const all = [...right.reverse(), ...sources.map((x) => ({ kind: 'src' as const, ...x })), ...left.reverse()];
  const span = Math.PI * 0.92;
  const start = (Math.PI - span) / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} {...stylex.props(s.svg)} role="img" aria-label={`root flare: ${sources.length} retrieval sources, ${tags.length} corpus domain tags`}>
      <defs>
        <radialGradient id="flare" cx="50%" cy="15%" r="70%">
          <stop offset="0%" stopColor={token.primaryContainer} stopOpacity="0.16" />
          <stop offset="100%" stopColor={token.surfaceContainerLowest} stopOpacity="0" />
        </radialGradient>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect x="0" y="0" width={W} height={H} fill="url(#flare)" />
      <line x1="0" y1={cy} x2={W} y2={cy} stroke={token.outlineVariant} strokeDasharray="2 6" />
      {all.map((r, i) => {
        // Source roots get 2.5 slots of angle each, so their labels have room.
        const weight = (x: typeof r) => (x.kind === 'src' ? 2.5 : 1);
        const total = all.reduce((acc, x) => acc + weight(x), 0);
        const before = all.slice(0, i).reduce((acc, x) => acc + weight(x), 0);
        const a = start + (span * (before + weight(r) / 2)) / total;
        const isSrc = r.kind === 'src';
        const srcIdx = isSrc ? all.slice(0, i).filter((x) => x.kind === 'src').length : 0;
        const len = isSrc ? 230 + 60 * (r.n / maxSrc) : 70 + 150 * (Math.log1p(r.n) / Math.log1p(maxTag));
        const ex = cx + Math.cos(a) * len * 1.35;
        const ey = cy + Math.sin(a) * len;
        const mx = cx + Math.cos(a) * len * 0.45 + (i % 2 ? 14 : -14);
        const my = cy + Math.sin(a) * len * 0.35;
        const width = isSrc ? 3 + 10 * (r.n / maxSrc) : 1 + 3 * (r.n / maxTag);
        const stroke = isSrc ? token.tertiaryContainer : token.primaryContainer;
        return (
          <g key={`${r.kind}-${r.source}`}>
            <path
              d={`M ${cx} ${cy} Q ${mx} ${my} ${ex} ${ey}`}
              fill="none"
              stroke={stroke}
              strokeOpacity={isSrc ? 0.9 : 0.55}
              strokeWidth={width}
              strokeLinecap="round"
              filter={isSrc ? 'url(#glow)' : undefined}
            />
            <rect x={ex - 3} y={ey - 3} width="6" height="6" fill={stroke} />
            <text
              x={isSrc ? ex : ex + (Math.cos(a) >= 0 ? 8 : -8)}
              y={isSrc ? ey + 20 + (srcIdx % 2) * 15 : ey + 4}
              textAnchor={isSrc ? 'middle' : Math.cos(a) >= 0 ? 'start' : 'end'}
              fontFamily="ui-monospace, monospace"
              fontSize={isSrc ? 13 : 10}
              fontWeight={isSrc ? 700 : 400}
              fill={isSrc ? token.tertiaryContainer : token.onSurfaceVariant}
            >
              {isSrc ? `${r.source.toUpperCase()} · ${n(r.n)} · emb ${pct(r.emb, 0)}` : `${r.source} ${n(r.n)}`}
            </text>
          </g>
        );
      })}
      <rect x={cx - 70} y={cy - 16} width="140" height="14" fill={token.surfaceContainerHigh} />
      <text x={cx} y={cy - 5} textAnchor="middle" fontFamily="ui-monospace, monospace" fontSize="10" fill={token.outline}>
        TRUNK BASE
      </text>
    </svg>
  );
}

type Polled<T> = { data: T | null; failure: Parameters<typeof PollState>[0]['failure'] };

function retrievalOf(res: Polled<Results>): RetrievalSection | null {
  const ret = res.data?.sections?.retrieval as RetrievalSection | { state: string } | undefined;
  return ret && ret.state === 'ready' ? (ret as RetrievalSection) : null;
}

function FlarePanel({ st, res }: { st: Polled<Stats>; res: Polled<Results> }) {
  const r = retrievalOf(res);
  const emb = r?.arms.find((a) => a === 'embedding');
  const sources = r ? r.sources.map((x) => ({ source: x.source, n: x.n, emb: emb ? x.by_arm[emb] ?? 0 : 0 })) : [];
  const tags = st.data?.by_domain_tag ? Object.entries(st.data.by_domain_tag).sort((a, b) => b[1] - a[1]) : [];
  return (
    <Panel kanji="根張り" title="NEBARI · ROOT FLARE" tag={`${sources.length} SOURCES · ${tags.length} TAGS`} edge="moss" fill>
      {st.data || r ? <RootFlare tags={tags} sources={sources} /> : <PollState path="/dash/api/stats" failure={st.failure ?? res.failure} />}
    </Panel>
  );
}

function SourcesPanel({ res }: { res: Polled<Results> }) {
  const r = retrievalOf(res);
  return (
    <Panel kanji="源" title="RETRIEVAL SOURCES" tag="HIT@1 BY SOURCE" tagTone="cyan" fill>
      {!res.data ? (
        <PollState path={PATHS.results} failure={res.failure} />
      ) : !r ? (
        <StateView kind="empty" title="no retrieval results" />
      ) : (
        <Table
          rows={r.sources}
          rowKey={(x) => x.source}
          columns={[
            { key: 's', head: 'source', cell: (x) => x.source },
            { key: 'n', head: 'rows', num: true, cell: (x) => n(x.n) },
            ...r.arms.map((arm) => ({
              key: arm,
              head: KNOWN_VOID[arm] ? `${arm} (void)` : arm,
              cell: (x: RetrievalSection['sources'][number]) => (
                <div {...stylex.props(layout.stack)}>
                  <RateBar rate={x.by_arm[arm] ?? null} tone={KNOWN_VOID[arm] ? 'muted' : 'cyan'} label={`${x.source} ${arm} ${pct(x.by_arm[arm])}`} />
                  <Label>{pct(x.by_arm[arm], 0)}</Label>
                </div>
              ),
            })),
          ]}
        />
      )}
    </Panel>
  );
}

function CorpusPanel({ st }: { st: Polled<Stats> }) {
  const d = st.data;
  const tags = d?.by_domain_tag ? Object.entries(d.by_domain_tag).sort((a, b) => b[1] - a[1]) : [];
  return (
    <Panel kanji="譜" title="RECIPE CORPUS" tag={d ? `${n(d.total)} RECIPES` : '—'} fill>
      {!d ? (
        <PollState path="/dash/api/stats" failure={st.failure} />
      ) : (
        <>
          <div {...stylex.props(layout.grid2)}>
            <Stat label="STATE A TRIGGER" value={n(d.with_trigger)} sub={`${pct(d.with_trigger / Math.max(d.total, 1), 0)} of rows`} />
            <Stat label="UNTAGGED DOMAINS" value={n(d.untagged_domains)} tone={d.untagged_domains ? 'rose' : 'moss'} sub="eligible everywhere" />
            <Stat label="OVER 40 WORDS" value={n(d.over_40_words)} />
            <Stat label="FILES" value={n(d.files?.length ?? 0)} />
          </div>
          <div {...stylex.props(layout.rowWrap)}>
            {Object.entries(d.by_state ?? {}).map(([k, c]) => (
              <Chip key={k} tone={k === 'unreviewed' ? 'rose' : 'moss'}>
                {k} {n(c)}
              </Chip>
            ))}
          </div>
          <div {...stylex.props(layout.stack)}>
            {tags.map(([t, c]) => (
              <div key={t} {...stylex.props(s.bar)}>
                <Label>{t}</Label>
                <Meter value={c / Math.max(1, tags[0]?.[1] ?? 1)} label={`${t} ${c}`} />
                <span {...stylex.props(text.labelXs, text.num, text.primary)}>{n(c)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}

const nebariAreas = stylex.create({
  nebari: {
    gridTemplateAreas: {
      default: '"flare" "strata" "corpus" "src"',
      [MQ.tablet]: '"flare flare flare flare flare flare" "strata strata strata corpus corpus corpus" "src src src corpus corpus corpus"',
      [MQ.desktop]:
        '"flare flare flare flare flare flare flare corpus corpus corpus corpus corpus" "src src src src src src src strata strata strata strata strata"',
    },
  },
});

export function Nebari() {
  const { vitals } = useShared();
  const st = usePoll<Stats>('/dash/api/stats', 60000);
  const res = usePoll<Results>(PATHS.results, 60000);
  const v = vitals.data;
  return (
    <Bento areas={nebariAreas.nebari}>
      <Cell area="flare">
        <ErrorBoundary what="NEBARI · ROOT FLARE" source="/dash/api/stats" fill>
          <FlarePanel st={st} res={res} />
        </ErrorBoundary>
      </Cell>
      <Cell area="src">
        <ErrorBoundary what="RETRIEVAL SOURCES" source={`${PATHS.results} · sections.retrieval`} fill>
          <SourcesPanel res={res} />
        </ErrorBoundary>
      </Cell>
      <Cell area="corpus">
        <ErrorBoundary what="RECIPE CORPUS" source="/dash/api/stats" fill>
          <CorpusPanel st={st} />
        </ErrorBoundary>
      </Cell>
      <Cell area="strata">
        <ErrorBoundary what="TIER STRATA" source="/dash/api/vitals · strata" fill>
          {v ? (
            <StrataPanel strata={v.strata} present={'strata' in v} fill />
          ) : (
            <Panel kanji="層" title="TIER STRATA" fill>
              <PollState path="/dash/api/vitals" failure={vitals.failure} />
            </Panel>
          )}
        </ErrorBoundary>
      </Cell>
    </Bento>
  );
}
