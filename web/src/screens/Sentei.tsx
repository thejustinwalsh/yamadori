// 剪定 SENTEI — pruning. The benchmark results that decide what is cut, and
// the power each comparison actually has. Nothing here is cut on a number
// that could not have detected the effect (docs/PROTOCOL.md).
import * as stylex from '@stylexjs/stylex';
import { PATHS } from '../api/data';
import { KNOWN_VOID } from '../api/provenance';
import type { EmptySection, ErrorSection, LcbArm, LcbSection, Pair, Power, RecipeSection, Results, RetrievalSection } from '../api/types';
import { usePoll } from '../api/usePoll';
import { ago, n, pct, pValue } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, SplitBar, Stat } from '../ui/primitives';
import { RateBar } from '../ui/RateBar';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';

const s = stylex.create({
  grid: {
    display: 'grid',
    gap: space.gutter,
    gridTemplateColumns: { default: 'minmax(0, 1fr)', '@media (min-width: 1100px)': 'repeat(2, minmax(0, 1fr))' },
    alignItems: 'start',
  },
  note: { margin: 0, color: colors.onSurfaceVariant },
  warn: { margin: 0, color: colors.secondary },
  void: {
    padding: space.spaceSm,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.secondaryContainer,
    backgroundImage: `repeating-linear-gradient(-45deg, color-mix(in srgb, ${colors.secondaryContainer} 10%, transparent), color-mix(in srgb, ${colors.secondaryContainer} 10%, transparent) 6px, transparent 6px, transparent 12px)`,
  },
  wide: { gridColumn: { default: 'auto', '@media (min-width: 1100px)': '1 / -1' } },
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
          head: 'verdict',
          cell: (p) =>
            KNOWN_VOID[p.a] || KNOWN_VOID[p.b] ? (
              <Chip tone="crimson">VOID ARM</Chip>
            ) : p.verdict ? (
              <Chip tone="moss">{p.verdict}</Chip>
            ) : p.underpowered ? (
              <Chip tone="muted">UNDERPOWERED</Chip>
            ) : (
              <Chip tone="muted">NO DIFFERENCE SHOWN</Chip>
            ),
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

function ByDomain({ l }: { l: LcbSection }) {
  if (!l.difficulty?.length) return null;
  return (
    <Table
      rows={l.difficulty}
      rowKey={(d) => d.difficulty}
      columns={[
        { key: 'd', head: 'domain', cell: (d) => d.difficulty },
        { key: 'n', head: 'n', num: true, cell: (d) => d.n },
        ...l.arms.map((a) => ({
          key: a,
          head: a,
          num: true,
          cell: (d: LcbSection['difficulty'][number]) => {
            const c = d.by_arm[a];
            if (!c) return '—';
            const st = c.stages;
            return (
              <span
                title={
                  c.lo !== undefined && c.hi !== undefined
                    ? `${pct(c.rate, 0)} [${pct(c.lo, 0)}–${pct(c.hi, 0)}]` +
                      (st ? ` · extract ${st.extract} · compile ${st.compile} · test ${st.test}` : '')
                    : undefined
                }
              >
                {c.k}/{c.n}
              </span>
            );
          },
        })),
      ]}
    />
  );
}

const STAGE_TONE = { pass: 'moss', test: 'cyan', compile: 'crimson', extract: 'muted' } as const;

function StageSplit({ a }: { a: LcbArm }) {
  const st = a.stages;
  if (!st) return <>—</>;
  const parts = (['pass', 'test', 'compile', 'extract'] as const).map((k) => ({ value: st[k] ?? 0, tone: STAGE_TONE[k] }));
  return (
    <SplitBar
      parts={parts}
      label={`${a.arm}: pass ${st.pass ?? 0}, failed at test ${st.test}, compile ${st.compile}, extract ${st.extract}`}
    />
  );
}

const fixed = (x: number | null | undefined, d = 1): string => (x === null || x === undefined ? '—' : x.toFixed(d));

function DomainCost({ l }: { l: LcbSection }) {
  const anyCheck = l.arm_table.some((a) => a.self_check);
  const ex = l.excluded_contaminated;
  return (
    <>
      {ex && ex.tasks > 0 && (
        <div {...stylex.props(layout.rowWrap)}>
          <Chip tone="rose">CONTAMINATED EXCLUDED · {n(ex.tasks)} · {ex.domains.join(' · ')}</Chip>
        </div>
      )}
      <Table
        rows={l.arm_table}
        rowKey={(a) => a.arm}
        caption="stages, cost and checks per arm"
        columns={[
          { key: 'arm', head: 'arm', cell: (a) => a.arm },
          { key: 'split', head: 'pass · test · compile · extract', cell: (a) => <StageSplit a={a} /> },
          { key: 'p', head: 'pass', num: true, cell: (a) => a.stages?.pass ?? a.k },
          { key: 't', head: 'test', num: true, cell: (a) => a.stages?.test ?? '—' },
          { key: 'c', head: 'compile', num: true, cell: (a) => a.stages?.compile ?? '—' },
          { key: 'x', head: 'extract', num: true, cell: (a) => a.stages?.extract ?? '—' },
          { key: 'fc', head: 'compiles', num: true, cell: (a) => (a.final_compiles === undefined ? '—' : `${a.final_compiles}/${a.n}`) },
          { key: 'ms', head: 'mean s', num: true, cell: (a) => fixed(a.mean_s) },
          { key: 'md', head: 'median s', num: true, cell: (a) => fixed(a.median_s) },
          { key: 'ti', head: 'tok in', num: true, cell: (a) => n(a.tok_in) },
          { key: 'to', head: 'tok out', num: true, cell: (a) => n(a.tok_out) },
          { key: 'th', head: 'tool hops', num: true, cell: (a) => fixed(a.tool_hops, 2) },
          ...(anyCheck
            ? [
                {
                  key: 'cr',
                  head: 'checks mean / max',
                  num: true,
                  cell: (a: LcbArm) => (a.self_check ? `${fixed(a.check_rounds_mean)} / ${a.check_rounds_max ?? '—'}` : '—'),
                },
                {
                  key: 'ca',
                  head: 'checked',
                  num: true,
                  cell: (a: LcbArm) => (a.self_check ? `${a.checked_any ?? 0}/${a.n}` : '—'),
                },
                {
                  key: 'fp',
                  head: 'answer compiles (public check)',
                  num: true,
                  cell: (a: LcbArm) => (a.self_check ? `${a.final_public_ok ?? 0}/${a.final_public_n ?? 0}` : '—'),
                },
              ]
            : []),
        ]}
      />
    </>
  );
}

function Twins({ l }: { l: LcbSection }) {
  if (!l.twins?.length) return null;
  return (
    <Table
      rows={l.twins}
      rowKey={(t) => `${t.one_shot}|${t.s}`}
      caption="self-check arm against its one-shot twin"
      columns={[
        { key: 'ab', head: 'self-check vs one-shot', cell: (t) => `${t.s} vs ${t.one_shot}` },
        { key: 'n', head: 'paired', num: true, cell: (t) => t.n_paired },
        { key: 'pp', head: 'pass one-shot / S', num: true, cell: (t) => `${t.one_shot_pass} / ${t.s_pass}` },
        { key: 'so', head: 'S only', num: true, cell: (t) => t.s_only },
        { key: 'oo', head: 'one-shot only', num: true, cell: (t) => t.one_shot_only },
        { key: 'p', head: 'p (McNemar, family)', num: true, cell: (t) => (t.p_bonferroni === null ? pValue(t.p) : pValue(t.p_bonferroni)) },
        {
          key: 'fx',
          head: 'fixed by checking',
          num: true,
          cell: (t) => <Chip tone={t.fixed_by_checking ? 'moss' : 'muted'}>{t.fixed_by_checking}</Chip>,
        },
        { key: 'ct', head: 'compile → test', num: true, cell: (t) => t.compile_to_test },
        { key: 'tp', head: 'test → pass', num: true, cell: (t) => t.test_to_pass },
        { key: 'cf', head: 'compile fails one-shot / S', num: true, cell: (t) => `${t.one_shot_compile_fail} / ${t.s_compile_fail}` },
      ]}
    />
  );
}

function Domain({ sec }: { sec: Sec<LcbSection> }) {
  const st = sectionState(sec);
  return (
    <Panel kanji="盆" title="DOMAIN TASKS · SYSTEMS ON / OFF" tag="PASS@1" tagTone="cyan" fill>
      {'node' in st ? (
        st.node
      ) : (
        <>
          <LcbBody l={st.ready} />
          <DomainCost l={st.ready} />
          <Twins l={st.ready} />
          <ByDomain l={st.ready} />
        </>
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

const senteiAreas = stylex.create({
  sentei: {
    gridTemplateAreas: {
      default: '"dom" "lcb" "ret" "side"',
      [MQ.tablet]: '"dom dom dom dom dom dom" "lcb lcb lcb lcb lcb lcb" "ret ret ret ret ret ret" "side side side side side side"',
      [MQ.desktop]: '"dom dom dom dom dom dom dom dom dom dom dom dom" "lcb lcb lcb lcb lcb lcb lcb lcb lcb lcb lcb lcb" "ret ret ret ret ret ret ret side side side side side"',
    },
  },
});

export function Sentei() {
  const r = usePoll<Results>(PATHS.results, 30000);
  if (!r.data) return <PollState path={PATHS.results} failure={r.failure} />;
  const sec = r.data.sections ?? {};
  const R = PATHS.results;
  return (
    <Bento areas={senteiAreas.sentei}>
      <Cell area="dom">
        <ErrorBoundary what="DOMAIN TASKS" source={`${R} · sections.domain`} fill>
          <Domain sec={sec.domain} />
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
