// 剪定 SENTEI — pruning. Every benchmark this stack has run, as evidence:
// each number with its n and interval, each arm with what it switches on,
// each mechanism with whether it actually ran. No verdicts: a comparison
// shows its discordant pairs and its exact p, and the reader decides
// (docs/PROTOCOL.md). The sections live in ./sentei/.
import * as stylex from '@stylexjs/stylex';
import { PATHS, useShared } from '../api/data';
import { KNOWN_VOID } from '../api/provenance';
import type { EmptySection, ErrorSection, LcbSection, Pair, Power, RecipeSection, Results, RetrievalSection } from '../api/types';
import { usePoll } from '../api/usePoll';
import { ago, n, pct, pValue } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Stat } from '../ui/primitives';
import { RateBar } from '../ui/RateBar';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';
import { Domain } from './sentei/Domain';
import { LiveBench } from './sentei/LiveBench';
import { summaryTiles } from './sentei/model';
import { Images, ModelCard, Speed, TierLadder } from './sentei/Other';
import { SweBench } from './sentei/SweBench';

const s = stylex.create({
  note: { margin: 0, color: colors.onSurfaceVariant },
  warn: { margin: 0, color: colors.secondary },
  void: {
    padding: space.spaceSm,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.secondaryContainer,
    backgroundImage: `repeating-linear-gradient(-45deg, color-mix(in srgb, ${colors.secondaryContainer} 10%, transparent), color-mix(in srgb, ${colors.secondaryContainer} 10%, transparent) 6px, transparent 6px, transparent 12px)`,
  },
});

type Sec<T> = T | EmptySection | ErrorSection | undefined;

function sectionState<T extends { state: 'ready' }>(sec: Sec<T>): { ready: T } | { node: React.ReactNode } {
  if (!sec) return { node: <StateView kind="empty" title="not in the payload" /> };
  if (sec.state === 'error') {
    const e = sec as ErrorSection;
    return { node: <StateView kind="error" title={`${e.component} raised`} detail={e.error} /> };
  }
  if (sec.state === 'empty') {
    const e = sec as EmptySection;
    return {
      node: (
        <StateView
          kind="empty"
          title={e.exists ? `${e.file} · no scored rows` : `${e.file} · not found`}
        />
      ),
    };
  }
  return { ready: sec as T };
}

function PowerLine({ p }: { p: Power }) {
  return (
    <div {...stylex.props(layout.rowWrap)}>
      <Chip tone="muted">N PAIRED {n(p.n_paired)}</Chip>
      <Chip tone="muted">α {p.alpha}</Chip>
      <Chip tone="muted">MIN DISCORDANT FOR SIG {p.min_discordant_for_sig ?? '—'}</Chip>
      {p.comparison_possible ? (
        <Chip tone="moss">COMPARISON POSSIBLE</Chip>
      ) : (
        <Chip tone="crimson">
          NO COMPARISON POSSIBLE · MAX {p.max_possible_discordant} DISCORDANT &lt; {p.min_discordant_for_sig ?? '?'}
        </Chip>
      )}
    </div>
  );
}

function Pairs({ pairs }: { pairs: Pair[] }) {
  return (
    <Table
      rows={pairs}
      rowKey={(p) => `${p.a}|${p.b}`}
      columns={[
        { key: 'ab', head: 'comparison', cell: (p) => `${p.a} vs ${p.b}` },
        { key: 'a', head: 'a only', num: true, cell: (p) => p.a_only },
        { key: 'b', head: 'b only', num: true, cell: (p) => p.b_only },
        { key: 'd', head: 'discordant', num: true, cell: (p) => p.discordant },
        { key: 'p', head: 'p (McNemar)', num: true, cell: (p) => pValue(p.p) },
        {
          key: 'v',
          head: 'status',
          cell: (p) =>
            KNOWN_VOID[p.a] || KNOWN_VOID[p.b] ? (
              <Chip tone="crimson">VOID ARM</Chip>
            ) : p.underpowered ? (
              <Chip tone="muted">UNDERPOWERED</Chip>
            ) : null,
        },
      ]}
    />
  );
}

function FileMeta({ file, age, bad }: { file: string; age: number | null; bad: number }) {
  return (
    <Label>
      {file} · written {ago(age)} ago{bad ? ` · ${bad} malformed line${bad === 1 ? '' : 's'}` : ''}
    </Label>
  );
}

function Lcb({ sec }: { sec: Sec<LcbSection> }) {
  const st = sectionState(sec);
  return (
    <Panel kanji="剪" title="LIVECODEBENCH · TIERS" tag="PASS@1" fill>
      {'node' in st ? (
        st.node
      ) : (
        <LcbBody l={st.ready} />
      )}
    </Panel>
  );
}

function LcbBody({ l }: { l: LcbSection }) {
  return (
    <>
      <FileMeta file={l.file} age={l.file_age_s} bad={l.bad_lines} />
      <div {...stylex.props(layout.grid4)}>
        <Stat label="ROWS" value={n(l.progress.rows)} />
        <Stat label="QUESTIONS ATTEMPTED" value={n(l.progress.attempted)} />
        <Stat label="COMPLETE, ALL ARMS" value={n(l.progress.complete_all_arms)} tone={l.progress.complete_all_arms < 10 ? 'rose' : 'moss'} />
        <Stat label="ERRORED ROWS" value={n(l.progress.errors)} tone={l.progress.errors ? 'crimson' : undefined} sub={`of ${n(l.progress.rows)}`} />
      </div>
      <PowerLine p={l.power} />
      {l.suspect.identical_across_arms && (
        <div {...stylex.props(layout.rowWrap)}>
          <Chip tone="crimson">SUSPECT · IDENTICAL ACROSS ARMS</Chip>
        </div>
      )}
      <Table
        rows={l.arm_table}
        rowKey={(a) => a.arm}
        columns={[
          { key: 'arm', head: 'arm', cell: (a) => a.arm },
          { key: 'k', head: 'solved', num: true, cell: (a) => `${a.k}/${a.n}` },
          { key: 'r', head: 'rate · 95% Wilson', cell: (a) => <RateBar rate={a.rate} lo={a.lo} hi={a.hi} label={`${a.arm} ${pct(a.rate)}`} /> },
          { key: 'rt', head: 'rate', num: true, cell: (a) => `${pct(a.rate, 0)} [${pct(a.lo, 0)}–${pct(a.hi, 0)}]` },
          { key: 's', head: 'mean s', num: true, cell: (a) => a.mean_s ?? '—' },
          { key: 'e', head: 'errors', num: true, cell: (a) => a.errors ?? '—' },
        ]}
      />
      <Pairs pairs={l.pairs} />
      <Label>
        ERRORS ·{' '}
        {Object.entries(l.progress.errors_by_arm ?? {})
          .map(([arm, e]) => `${arm} ${Object.entries(e).map(([k, v]) => `${k}×${v}`).join(' ')}`)
          .join(' · ')}{' '}
        · SCAFFOLDING FAILURES {l.scaffolding_total}
      </Label>
    </>
  );
}

function Retrieval({ sec }: { sec: Sec<RetrievalSection> }) {
  const st = sectionState(sec);
  return (
    <Panel kanji="根" title="RETRIEVAL · LOCATE" tag="HIT@1" tagTone="cyan" fill>
      {'node' in st ? (
        st.node
      ) : (
        <>
          <FileMeta file={st.ready.file} age={st.ready.file_age_s} bad={st.ready.bad_lines} />
          <PowerLine p={st.ready.power} />
          <Table
            rows={st.ready.arm_table}
            rowKey={(a) => a.arm}
            voided={(a) => !!KNOWN_VOID[a.arm]}
            columns={[
              { key: 'arm', head: 'arm', cell: (a) => (KNOWN_VOID[a.arm] ? <>{a.arm} <Chip tone="crimson">VOID</Chip></> : a.arm) },
              { key: 'h', head: 'hit@1 · 95% Wilson', cell: (a) => <RateBar rate={a.hit1_rate} lo={a.hit1_lo} hi={a.hit1_hi} tone={KNOWN_VOID[a.arm] ? 'muted' : a.extreme ? 'rose' : 'cyan'} label={`${a.arm} hit@1 ${pct(a.hit1_rate)}`} /> },
              { key: 'r', head: 'hit@1', num: true, cell: (a) => `${a.hit1}/${a.n}` },
              { key: 'h5', head: 'hit@5', num: true, cell: (a) => pct(a.hit5_rate, 0) },
              { key: 'm', head: 'MRR', num: true, cell: (a) => a.mrr },
              { key: 'ms', head: 'mean ms', num: true, cell: (a) => n(a.mean_ms) },
              { key: 'x', head: 'cost', num: true, cell: (a) => `${n(a.cost_x)}×` },
            ]}
          />
          <Pairs pairs={st.ready.pairs} />
        </>
      )}
    </Panel>
  );
}

function Recipe({ sec }: { sec: Sec<RecipeSection> }) {
  const st = sectionState(sec);
  return (
    <Panel kanji="苗" title="RECIPE ORACLE" tag="PASS" tagTone="rose">
      {'node' in st ? (
        st.node
      ) : (
        <>
          <FileMeta file={st.ready.file} age={st.ready.file_age_s} bad={st.ready.bad_lines} />
          <PowerLine p={st.ready.power} />
          <Table
            rows={st.ready.arm_table}
            rowKey={(a) => a.arm}
            columns={[
              { key: 'arm', head: 'arm', cell: (a) => a.arm },
              { key: 'k', head: 'passed', num: true, cell: (a) => `${a.k}/${a.n}` },
              { key: 'r', head: 'rate · 95% Wilson', cell: (a) => <RateBar rate={a.rate} lo={a.lo} hi={a.hi} tone="rose" label={`${a.arm} ${pct(a.rate)}`} /> },
            ]}
          />
          <Label>ORACLE {st.ready.oracle.k}/{st.ready.oracle.n} · {pct(st.ready.oracle.rate)}</Label>
        </>
      )}
    </Panel>
  );
}

function RetrievalNotes({ sec }: { sec: Sec<RetrievalSection> }) {
  const ready = sec && sec.state === 'ready' ? (sec as RetrievalSection) : null;
  const voids = ready ? Object.entries(KNOWN_VOID).filter(([arm]) => ready.arms.includes(arm)) : [];
  const caveats = Array.isArray(ready?.caveats) ? ready.caveats : [];
  return (
    <Panel kanji="注" title="RETRIEVAL · CAVEATS" tag={`${voids.length + caveats.length}`} tagTone={voids.length ? 'crimson' : 'muted'} fill>
      {!ready ? (
        <StateView kind="empty" title="no retrieval section" />
      ) : (
        <>
          {voids.map(([arm, v]) => (
            <div key={arm} {...stylex.props(s.void)}>
              <p {...stylex.props(text.labelMd, s.warn)}>[ ! ] {arm} · {v.why}</p>
              <Label>{v.source}</Label>
            </div>
          ))}
          {caveats.length ? (
            <details {...stylex.props(text.labelXs, s.note)}>
              <summary>SERVER CAVEATS · {caveats.length}</summary>
              {caveats.map((c, i) => (
                <p key={i} {...stylex.props(text.bodySm, s.note)}>
                  [ ~ ] {c}
                </p>
              ))}
            </details>
          ) : null}
        </>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- summary --

const sum = stylex.create({
  strip: {
    display: 'grid',
    gap: space.spaceXs,
    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 200px), 1fr))',
  },
  tile: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    padding: space.spaceSm,
    minHeight: '44px',
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.outlineVariant,
    textDecoration: 'none',
    color: 'inherit',
    minWidth: 0,
    ':hover': { borderColor: colors.outline },
    ':focus-visible': { outline: `2px solid ${colors.primaryContainer}`, outlineOffset: '2px' },
  },
  published: { borderStyle: 'dashed' },
  line: { display: 'flex', justifyContent: 'space-between', gap: space.spaceSm, alignItems: 'baseline', minWidth: 0 },
  lineLabel: { color: colors.onSurfaceVariant, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  value: { color: colors.primary, whiteSpace: 'nowrap' },
  sub: { color: colors.outline, margin: 0, overflowWrap: 'anywhere' },
  note: { color: colors.onSurfaceVariant, margin: 0, overflowWrap: 'anywhere' },
});

function SummaryStrip({ r }: { r: Results }) {
  const tiles = summaryTiles(r);
  return (
    <Panel kanji="総" title="BENCHMARKS" tag={`${tiles.length} SOURCES`} tagTone="muted" fill>
      <div {...stylex.props(sum.strip)}>
        {tiles.map((t) => (
          <a key={t.key} href={`#${t.anchor}`} {...stylex.props(sum.tile, t.published && sum.published)}>
            <Label>{t.title}</Label>
            {t.lines.length ? (
              t.lines.map((l) => (
                <div key={l.label} {...stylex.props(layout.stack)}>
                  <div {...stylex.props(sum.line)}>
                    <span {...stylex.props(text.bodySm, sum.lineLabel)}>{l.label}</span>
                    <span {...stylex.props(text.titleMd, text.num, sum.value)}>{l.value}</span>
                  </div>
                  {l.sub && <p {...stylex.props(text.labelXs, sum.sub)}>{l.sub}</p>}
                </div>
              ))
            ) : (
              <span {...stylex.props(text.bodySm, sum.note)}>{t.state === 'error' ? 'section error' : t.state === 'running' ? 'running · not yet scored' : 'no results yet'}</span>
            )}
            {t.note && <p {...stylex.props(text.labelXs, t.published ? sum.note : sum.sub)}>{t.note}</p>}
          </a>
        ))}
      </div>
    </Panel>
  );
}

const senteiAreas = stylex.create({
  sentei: {
    gridTemplateAreas: {
      default: '"sum" "tier" "card" "lb" "swe" "dom" "spd" "img" "lcb" "ret" "side"',
      [MQ.tablet]:
        '"sum sum sum sum sum sum" "tier tier tier tier tier tier" "card card card card card card" "lb lb lb lb lb lb" "swe swe swe swe swe swe" "dom dom dom dom dom dom" "spd spd spd spd spd spd" "img img img img img img" "lcb lcb lcb lcb lcb lcb" "ret ret ret ret ret ret" "side side side side side side"',
      [MQ.desktop]:
        '"sum sum sum sum sum sum sum sum sum sum sum sum" "tier tier tier tier tier tier tier card card card card card" "lb lb lb lb lb lb lb lb lb lb lb lb" "swe swe swe swe swe swe swe swe swe swe swe swe" "dom dom dom dom dom dom dom dom dom dom dom dom" "spd spd spd spd spd spd spd img img img img img" "lcb lcb lcb lcb lcb lcb lcb lcb lcb lcb lcb lcb" "ret ret ret ret ret ret ret side side side side side"',
    },
  },
});

export function Sentei() {
  const r = usePoll<Results>(PATHS.results, 30000);
  const { tiers } = useShared();
  if (!r.data) return <PollState path={PATHS.results} failure={r.failure} />;
  const sec = r.data.sections ?? {};
  const R = PATHS.results;
  return (
    <Bento areas={senteiAreas.sentei}>
      <Cell area="sum">
        <ErrorBoundary what="BENCHMARKS" source={R} fill>
          <SummaryStrip r={r.data} />
        </ErrorBoundary>
      </Cell>
      <Cell area="tier">
        <ErrorBoundary what="EFFORT TIERS" source={PATHS.tiers} fill>
          <TierLadder tiers={tiers} />
        </ErrorBoundary>
      </Cell>
      <Cell area="lb">
        <ErrorBoundary what="LIVEBENCH" source={`${R} · sections.livebench`} fill>
          <LiveBench sec={sec.livebench} />
        </ErrorBoundary>
      </Cell>
      <Cell area="card">
        <ErrorBoundary what="MODEL CARD" source={`${R} · sections.model_card`} fill>
          <ModelCard sec={sec.model_card} />
        </ErrorBoundary>
      </Cell>
      <Cell area="swe">
        <ErrorBoundary what="SWE-BENCH VERIFIED" source={`${R} · sections.swebench`} fill>
          <SweBench sec={sec.swebench} />
        </ErrorBoundary>
      </Cell>
      <Cell area="dom">
        <ErrorBoundary what="DOMAIN TASKS" source={`${R} · sections.domain`} fill>
          <Domain sec={sec.domain} />
        </ErrorBoundary>
      </Cell>
      <Cell area="spd">
        <ErrorBoundary what="SPEED" source={`${R} · sections.speed`} fill>
          <Speed sec={sec.speed} />
        </ErrorBoundary>
      </Cell>
      <Cell area="img">
        <ErrorBoundary what="IMAGES" source={`${R} · sections.imagegen`} fill>
          <Images sec={sec.imagegen} />
        </ErrorBoundary>
      </Cell>
      <Cell area="lcb">
        <ErrorBoundary what="LIVECODEBENCH · TIERS" source={`${R} · sections.lcb`} fill>
          <Lcb sec={sec.lcb} />
        </ErrorBoundary>
      </Cell>
      <Cell area="ret">
        <ErrorBoundary what="RETRIEVAL · LOCATE" source={`${R} · sections.retrieval`} fill>
          <Retrieval sec={sec.retrieval} />
        </ErrorBoundary>
      </Cell>
      <Cell area="side">
        <ErrorBoundary what="RECIPE ORACLE" source={`${R} · sections.recipe`}>
          <Recipe sec={sec.recipe} />
        </ErrorBoundary>
        <ErrorBoundary what="RETRIEVAL · CAVEATS" source={`${R} · sections.retrieval.caveats`} fill>
          <RetrievalNotes sec={sec.retrieval} />
        </ErrorBoundary>
      </Cell>
    </Bento>
  );
}
