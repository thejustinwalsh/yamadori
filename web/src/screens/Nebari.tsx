// 根張り NEBARI — the root flare. What the model can draw on: the skills
// (the one knowledge system since 2026-09-26: mcp/skills.py, served through
// skill selection), the held package indexes the code tools and deep
// thinking read, and how well retrieval found things when it was measured.
// Every root drawn here is a counted row in a real payload
// (/dash/api/nebari, mcp/dash_nebari.py).
//
// Until 2026-09-29 the roots were the recipe corpus's domain tags
// (/dash/api/stats). The corpus stopped being served when its rows were
// migrated into skills; bench/recipes stays only as provenance.
import * as stylex from '@stylexjs/stylex';
import { PATHS, useShared } from '../api/data';
import { errorOf, flareRoots, itemsOf, labelOf, NEBARI_PATH, packagesOf, skillsOf, type HeldPackage, type Nebari, type Root } from '../api/nebari';
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

const s = stylex.create({
  svg: { width: '100%', height: 'auto', display: 'block', backgroundColor: colors.surfaceContainerLowest },
  note: { margin: 0, color: colors.onSurfaceVariant },
  bar: { display: 'grid', gridTemplateColumns: 'minmax(90px, 140px) minmax(0,1fr) 44px', gap: space.spaceSm, alignItems: 'center' },
});

/**
 * Root flare: one heavy root per framework the served skills cover, one thin
 * root per language, length by log(count). Deterministic layout: angle from
 * the sorted index, never random.
 */
function RootFlare({ roots }: { roots: Root[] }) {
  const W = 980;
  const H = 420;
  const cx = W / 2;
  const cy = 50;
  const max = Math.max(1, ...roots.map((r) => r.n));
  // Heavy roots down the middle, thin roots fanned out either side, largest
  // nearest the trunk, like a real nebari.
  const heavy = roots.filter((r) => r.heavy);
  const thin = roots.filter((r) => !r.heavy);
  const left = thin.filter((_, i) => i % 2 === 0).reverse();
  const right = thin.filter((_, i) => i % 2 === 1);
  const all = [...right.reverse(), ...heavy, ...left.reverse()];
  const span = Math.PI * 0.92;
  const start = (Math.PI - span) / 2;
  const weight = (x: Root) => (x.heavy ? 2 : 1);
  const total = all.reduce((acc, x) => acc + weight(x), 0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} {...stylex.props(s.svg)} role="img" aria-label={`root flare: served skills across ${heavy.length} frameworks and ${thin.length} languages`}>
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
        const before = all.slice(0, i).reduce((acc, x) => acc + weight(x), 0);
        const a = start + (span * (before + weight(r) / 2)) / total;
        const hi = all.slice(0, i).filter((x) => x.heavy).length;
        const len = r.heavy ? 150 + 140 * (Math.log1p(r.n) / Math.log1p(max)) : 70 + 150 * (Math.log1p(r.n) / Math.log1p(max));
        const ex = cx + Math.cos(a) * len * 1.35;
        const ey = cy + Math.sin(a) * len;
        const mx = cx + Math.cos(a) * len * 0.45 + (i % 2 ? 14 : -14);
        const my = cy + Math.sin(a) * len * 0.35;
        const width = r.heavy ? 3 + 9 * (r.n / max) : 1 + 3 * (r.n / max);
        const stroke = r.heavy ? token.tertiaryContainer : token.primaryContainer;
        return (
          <g key={`${r.heavy ? 'h' : 't'}-${r.label}`}>
            <path
              d={`M ${cx} ${cy} Q ${mx} ${my} ${ex} ${ey}`}
              fill="none"
              stroke={stroke}
              strokeOpacity={r.heavy ? 0.9 : 0.55}
              strokeWidth={width}
              strokeLinecap="round"
              filter={r.heavy ? 'url(#glow)' : undefined}
            />
            <rect x={ex - 3} y={ey - 3} width="6" height="6" fill={stroke} />
            <text
              x={r.heavy ? ex : ex + (Math.cos(a) >= 0 ? 8 : -8)}
              y={r.heavy ? ey + 20 + (hi % 2) * 15 : ey + 4}
              textAnchor={r.heavy ? 'middle' : Math.cos(a) >= 0 ? 'start' : 'end'}
              fontFamily="ui-monospace, monospace"
              fontSize={r.heavy ? 12 : 10}
              fontWeight={r.heavy ? 700 : 400}
              fill={r.heavy ? token.tertiaryContainer : token.onSurfaceVariant}
            >
              {r.heavy ? `${r.label.toUpperCase()} · ${n(r.n)}` : `${r.label} ${n(r.n)}`}
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

function FlarePanel({ nb }: { nb: Polled<Nebari> }) {
  const sk = skillsOf(nb.data);
  const roots = flareRoots(sk);
  const fw = roots.filter((r) => r.heavy).length;
  return (
    <Panel kanji="根張り" title="NEBARI · ROOT FLARE" tag={sk ? `${n(sk.served)} SERVED SKILLS · ${fw} FRAMEWORKS` : '—'} edge="moss" fill>
      {!nb.data ? (
        <PollState path={NEBARI_PATH} failure={nb.failure} />
      ) : !sk ? (
        <StateView kind="error" title={`${NEBARI_PATH} · skills`} detail={errorOf(nb.data.skills) ?? undefined} />
      ) : !roots.length ? (
        <StateView kind="empty" title="no served skill is filed under a framework or language" />
      ) : (
        <RootFlare roots={roots} />
      )}
    </Panel>
  );
}

function SkillsPanel({ nb }: { nb: Polled<Nebari> }) {
  const sk = skillsOf(nb.data);
  const domains = sk ? Object.entries(sk.served_by?.domain ?? {}) : [];
  const top = domains[0]?.[1] ?? 1;
  return (
    <Panel kanji="技" title="SKILLS" tag={sk ? `${n(sk.served)} SERVED OF ${n(sk.total)}` : '—'} fill>
      {!nb.data ? (
        <PollState path={NEBARI_PATH} failure={nb.failure} />
      ) : !sk ? (
        <StateView kind="error" title={`${NEBARI_PATH} · skills`} detail={errorOf(nb.data.skills) ?? undefined} />
      ) : (
        <>
          <div {...stylex.props(layout.grid2)}>
            <Stat label="ARMED" value={n(sk.counts.armed ?? 0)} sub={`${n(sk.served)} served: enabled, with a line to serve`} tone="moss" />
            <Stat label="QUARANTINED" value={n(sk.counts.quarantined ?? 0)} tone={(sk.counts.quarantined ?? 0) ? 'rose' : undefined} sub="held back by a check" />
          </div>
          <div {...stylex.props(layout.rowWrap)}>
            {Object.entries(sk.counts ?? {})
              .filter(([k, c]) => c > 0 && k !== 'armed' && k !== 'quarantined')
              .map(([k, c]) => (
                <Chip key={k} tone={k === 'failed' ? 'crimson' : 'muted'}>
                  {k} {n(c)}
                </Chip>
              ))}
            {Object.entries(sk.served_by?.phase ?? {}).map(([k, c]) => (
              <Chip key={`p-${k}`} tone="cyan">
                phase {labelOf(sk, 'phase', k)} {n(c)}
              </Chip>
            ))}
          </div>
          <div {...stylex.props(layout.stack)}>
            {domains.map(([t, c]) => (
              <div key={t} {...stylex.props(s.bar)}>
                <Label>{t}</Label>
                <Meter value={c / Math.max(1, top)} label={`${t} ${c} served skills`} />
                <span {...stylex.props(text.labelXs, text.num, text.primary)}>{n(c)}</span>
              </div>
            ))}
          </div>
          <p {...stylex.props(text.labelXs, s.note)}>served skills by domain · the whole library, with its tests and provenance, is on SKILLS</p>
        </>
      )}
    </Panel>
  );
}

function PackagesPanel({ nb }: { nb: Polled<Nebari> }) {
  const pk = packagesOf(nb.data);
  const ix = nb.data && nb.data.indexes && !('error' in nb.data.indexes) ? nb.data.indexes : null;
  return (
    <Panel kanji="蔵" title="HELD PACKAGES" tag={pk ? `${pk.length} INDEXES` : '—'} tagTone="cyan" fill>
      {!nb.data ? (
        <PollState path={NEBARI_PATH} failure={nb.failure} />
      ) : !pk ? (
        <StateView kind="error" title={`${NEBARI_PATH} · packages`} detail={errorOf(nb.data.packages) ?? undefined} />
      ) : !pk.length ? (
        <StateView kind="empty" title="no package index held (index/packages)" />
      ) : (
        <>
          <Table
            rows={pk}
            rowKey={(p: HeldPackage) => `${p.package}@${p.version}`}
            columns={[
              {
                key: 'p',
                head: 'package',
                cell: (p: HeldPackage) => (
                  <span {...stylex.props(layout.rowWrap)}>
                    {p.package}@{p.version}
                    {p.unseen ? (
                      <Chip tone="rose" title={p.unseen}>
                        UNSEEN
                      </Chip>
                    ) : null}
                    {p.embedded === false ? <Chip tone="muted">NOT EMBEDDED</Chip> : null}
                  </span>
                ),
              },
              { key: 'i', head: 'items', num: true, cell: (p: HeldPackage) => n(itemsOf(p)) },
              { key: 'd', head: 'defs', num: true, cell: (p: HeldPackage) => n(p.defs) },
              { key: 'f', head: 'files', num: true, cell: (p: HeldPackage) => n(p.files) },
              { key: 'pub', head: 'published', cell: (p: HeldPackage) => p.published ?? '—' },
            ]}
          />
          <p {...stylex.props(text.labelXs, s.note)}>
            UNSEEN: deep thinking's known-hard area (deep.unseen: first published, or a new major, after the model's cutoff)
            {ix ? ` · bound code index ${n(ix.code ? ix.code.chunks + ix.code.defs : null)} items · ${n(ix.repos?.indexes ?? 0)} repository indexes` : ''}
          </p>
        </>
      )}
    </Panel>
  );
}

function SourcesPanel({ res }: { res: Polled<Results> }) {
  const r = retrievalOf(res);
  return (
    <Panel kanji="源" title="RETRIEVAL SOURCES" tag="HIT@1 BY SOURCE · MEASURED" tagTone="cyan" fill>
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

const nebariAreas = stylex.create({
  nebari: {
    gridTemplateAreas: {
      default: '"flare" "skills" "pkgs" "strata" "src"',
      [MQ.tablet]: '"flare flare flare flare flare flare" "skills skills skills pkgs pkgs pkgs" "strata strata strata src src src"',
      [MQ.desktop]:
        '"flare flare flare flare flare flare flare skills skills skills skills skills" "pkgs pkgs pkgs pkgs pkgs pkgs pkgs skills skills skills skills skills" "src src src src src src src strata strata strata strata strata"',
    },
  },
});

export function Nebari() {
  const { vitals } = useShared();
  const nb = usePoll<Nebari>(NEBARI_PATH, 60000);
  const res = usePoll<Results>(PATHS.results, 60000);
  const v = vitals.data;
  return (
    <Bento areas={nebariAreas.nebari}>
      <Cell area="flare">
        <ErrorBoundary what="NEBARI · ROOT FLARE" source={NEBARI_PATH} fill>
          <FlarePanel nb={nb} />
        </ErrorBoundary>
      </Cell>
      <Cell area="skills">
        <ErrorBoundary what="SKILLS" source={`${NEBARI_PATH} · skills`} fill>
          <SkillsPanel nb={nb} />
        </ErrorBoundary>
      </Cell>
      <Cell area="pkgs">
        <ErrorBoundary what="HELD PACKAGES" source={`${NEBARI_PATH} · packages`} fill>
          <PackagesPanel nb={nb} />
        </ErrorBoundary>
      </Cell>
      <Cell area="src">
        <ErrorBoundary what="RETRIEVAL SOURCES" source={`${PATHS.results} · sections.retrieval`} fill>
          <SourcesPanel res={res} />
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
