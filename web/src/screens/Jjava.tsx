// 判断 JJAVA -- the decider (docs/JJAVA.md): when it is used and by whom, how
// fast, what it answers and how sure it is, the thresholds in force, the
// lane, and the Jev API's endpoints. Every number is from a record already
// written (/dash/api/jjava, mcp/dash_jjava.py); the route asks no model.
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import {
  jjavaPath,
  MODEL_SLOT,
  msText,
  part,
  ranked,
  spreadText,
  statusTone,
  topSeries,
  total,
  type Jjava,
  type QuestionSet,
} from '../api/stats';
import { usePoll } from '../api/usePoll';
import { clock, n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { Histogram, LineChart, StackedBars, type Serie } from '../ui/Charts';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Stat } from '../ui/primitives';
import { Counts, InjectorPanel, KV, Note, SectionError, tierTone, useWindow, WindowPicker } from '../ui/StatsParts';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';

const areas = stylex.create({
  page: {
    gridTemplateAreas: {
      default: '"head" "use" "lat" "sets" "jev" "thr" "lane" "prior" "inj"',
      [MQ.tablet]: '"head head head head head head" "use use use use use use" "lat lat lat lat lat lat" "sets sets sets sets sets sets" "jev jev jev jev jev jev" "thr thr thr lane lane lane" "prior prior prior prior prior prior" "inj inj inj inj inj inj"',
      [MQ.desktop]:
        '"head head head head head head head head head head head head" "use use use use use use use lat lat lat lat lat" "sets sets sets sets sets sets sets sets sets sets sets sets" "jev jev jev jev jev jev jev jev thr thr thr thr" "inj inj inj inj inj inj inj inj lane lane lane lane" "inj inj inj inj inj inj inj inj prior prior prior prior"',
    },
  },
});

const s = stylex.create({
  head: { display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: space.spaceSm },
  hists: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))' },
  set: { display: 'flex', flexDirection: 'column', gap: '3px', padding: space.spaceXs, backgroundColor: colors.surfaceContainerLowest, borderWidth: 1, borderStyle: 'solid', borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`, minWidth: 0 },
  asIs: { textTransform: 'none' },
});

type D = { data: Jjava | null; stale?: boolean };

function Head({ d, w, setW, failure }: { d: Jjava | null; w: ReturnType<typeof useWindow>[0]; setW: ReturnType<typeof useWindow>[1]; failure: Parameters<typeof PollState>[0]['failure'] }) {
  const src = d?.sources;
  return (
    <Panel kanji="判断" title="JJAVA · THE DECIDER" tag={d ? `${n(part(d.usage)?.decisions ?? 0)} DECISIONS · ${w.toUpperCase()}` : '—'} tagTone="cyan" edge="cyan"
      sub="our Jev: typed questions (noul / choice / score) read on the decider lane, answered in Jev's API shape (docs/JJAVA.md). Every figure is a record already written; this page asks no model.">
      <div {...stylex.props(s.head)}>
        <WindowPicker value={w} onChange={setW} />
        {d ? <Label>READ {clock(d.at)} · BUCKET {Math.round(d.window.bucket_s / 60)} MIN</Label> : null}
      </div>
      {!d ? (
        <PollState path={jjavaPath(w)} failure={failure} />
      ) : (
        <div {...stylex.props(layout.grid4)}>
          <Stat label="DECISION LOG" value={n(src?.decisions.rows ?? 0)} sub={src?.decisions.exists ? `${src.decisions.path}${src.decisions.cut ? ' · tail only' : ''} · ${n(src.decisions.with_latency)} timed` : 'no log yet'} />
          <Stat label="JEV CALLS" value={n(src?.jev_calls.rows ?? 0)} sub="corpus events, kind jev_call" />
          <Stat label="REQUESTS RECORDED" value={n(src?.requests ?? 0)} sub="stats store: decider Turn, injector" />
          <Stat label="STATS STORE" value={src?.stats.enabled ? 'RECORDING' : 'NOT HERE'} sub={`${src?.stats.enabled ? '' : 'this process does not record · '}${n(src?.stats.written ?? 0)} written · ${n(src?.stats.dropped ?? 0)} dropped · ${n(src?.stats.errors ?? 0)} errors`} tone={src?.stats.enabled ? 'moss' : 'rose'} />
        </div>
      )}
    </Panel>
  );
}

function seriesOf(by: Record<string, number[]>, slotOf?: (name: string, i: number) => number): Serie[] {
  return topSeries(by).map((x, i) => ({ name: x.name, values: x.values, slot: x.name.startsWith('other') ? 4 : slotOf ? slotOf(x.name, i) : i }));
}

function UsePanel({ data, stale }: D) {
  const u = part(data?.usage);
  const t = data?.window.buckets ?? [];
  const b = data?.window.bucket_s ?? 3600;
  return (
    <Panel kanji="用" title="USE · DECISIONS OVER TIME" tag={u ? `${n(u.decisions)} IN THE WINDOW` : '—'} tagTone="moss" stale={stale} fill
      sub="per bucket; a Jev call counts as its questions. By question set, by the model that read it, by caller.">
      {!data ? null : !u ? (
        <SectionError what="usage" x={data.usage} />
      ) : (
        <>
          <StackedBars t={t} bucketS={b} series={seriesOf(u.by_question_set)} label="by question set" />
          <StackedBars t={t} bucketS={b} series={seriesOf(u.by_model, (m, i) => MODEL_SLOT[m] ?? Math.min(i, 3))} label="by model" height={60} />
          <StackedBars t={t} bucketS={b} series={seriesOf(u.by_caller)} label="by caller" height={60} />
        </>
      )}
    </Panel>
  );
}

function LatencyPanel({ data, stale }: D) {
  const l = part(data?.latency);
  const t = data?.window.buckets ?? [];
  return (
    <Panel kanji="速" title="LATENCY" tag={l ? `READ P50 ${msText(l.per_read.p50)}` : '—'} tagTone="cyan" stale={stale} fill>
      {!data ? null : !l ? (
        <SectionError what="latency" x={data.latency} />
      ) : (
        <>
          <LineChart t={t} bucketS={data.window.bucket_s} unit="ms" label="per decision · p50 and p90" fmt={(x) => msText(x)}
            series={[{ name: 'p50', values: l.series.p50, slot: 1 }, { name: 'p90', values: l.series.p90, slot: 3 }]} />
          <div {...stylex.props(layout.stack)}>
            <KV k="PER READ" v={spreadText(l.per_read)} title="one forward pass of one option order" />
            <KV k="PER DECISION" v={spreadText(l.per_decision)} title="a question: both orders" />
            <KV k="PER BURST · DECISION LOG" v={spreadText(l.per_burst.decision_log)} title="one request's decisions of one kind" />
            <KV k="PER BURST · DECIDER TURN" v={spreadText(l.per_burst.decider_turns)} title="x_yamadori.skills.turn ms_questions" />
            <KV k="PER BURST · JEV CALL" v={spreadText(l.per_burst.jev_calls)} />
            <KV k="JEV · PER READ" v={spreadText(l.jev_per_read)} />
          </div>
          {Object.keys(l.per_read.by_model).length ? (
            <Table
              rows={Object.entries(l.per_read.by_model)}
              rowKey={([m]) => m}
              columns={[
                { key: 'm', head: 'model', cell: ([m]) => m },
                { key: 'p', head: 'per read', cell: ([, x]) => spreadText(x) },
              ]}
            />
          ) : null}
          <Note>{l.note}</Note>
        </>
      )}
    </Panel>
  );
}

function SetCard({ q }: { q: QuestionSet }) {
  const top = ranked(q.picks).slice(0, 4);
  return (
    <div {...stylex.props(s.set)}>
      <div {...stylex.props(layout.between)}>
        <span {...stylex.props(text.labelMd, text.primary, s.asIs)}>{q.name}</span>
        <Chip tone="muted">{q.type ?? '?'} · n {n(q.n)}</Chip>
      </div>
      <Label>{q.caller} · {Object.keys(q.models).join(', ')}</Label>
      {q.type === 'noul' ? (
        <Histogram bins={q.noul_hist} label="noul" slot={1} note={q.noul_middle != null ? `0.35–0.65: ${q.noul_middle}` : '0.5'} />
      ) : (
        <Histogram bins={q.confidence_hist} label="confidence" slot={0} />
      )}
      <KV k="PICKS" v={top.map(([k, c]) => `${k} ${c}`).join(' · ') || '—'} />
      <KV k="TIERS" v={<Counts m={q.tiers} toneOf={tierTone} />} />
      <KV k="TIES · ORDERS DISAGREE" v={`${q.ties} · ${q.orders_disagree}`} title="a tie is inside the model's tie band; the two option orders' argmaxes differ" />
    </div>
  );
}

function SetsPanel({ data, stale }: D) {
  const qs = part(data?.question_sets);
  return (
    <Panel kanji="問" title="QUESTION SETS · ANSWERS AND CONFIDENCE" tag={qs ? `${qs.length} SETS` : '—'} tagTone="cyan" stale={stale} fill
      sub="each set's answer distribution, its confidence (choice, score) or noul histogram in ten bins, the tiers it fell in. A noul near 0.5 means the state did not separate the cases.">
      {!data ? null : !qs ? (
        <SectionError what="question sets" x={data.question_sets} />
      ) : !qs.length ? (
        <StateView kind="empty" title="no decision in this window" detail="The decider runs where skills run (off at every tier since 2026-09-29) and behind the Jev API." />
      ) : (
        <div {...stylex.props(s.hists)}>
          {qs.map((q) => (
            <SetCard key={q.name} q={q} />
          ))}
        </div>
      )}
    </Panel>
  );
}

function ThresholdsPanel({ data, stale }: D) {
  const th = part(data?.thresholds);
  return (
    <Panel kanji="閾" title="THRESHOLDS · TIERS FIRED" tag={th ? th.state.toUpperCase() : '—'} tagTone={th?.state === 'untuned' ? 'muted' : 'moss'} stale={stale} fill>
      {!data ? null : !th ? (
        <SectionError what="thresholds" x={data.thresholds} />
      ) : (
        <>
          <KV k="DECIDE_TURN.THRESHOLDS" v={Object.keys(th.decide_turn).length ? JSON.stringify(th.decide_turn) : 'empty (untuned)'} />
          <KV k="SKILL_INJECT.THRESHOLDS" v={Object.keys(th.skill_inject).length ? JSON.stringify(th.skill_inject) : 'empty (untuned)'} />
          {Object.keys(th.fired).length ? (
            <Table
              rows={Object.entries(th.fired)}
              rowKey={([k]) => k}
              columns={[
                { key: 'q', head: 'question set', cell: ([k]) => <span {...stylex.props(s.asIs)}>{k}</span> },
                { key: 't', head: 'tiers', cell: ([, m]) => <Counts m={m} toneOf={tierTone} /> },
              ]}
            />
          ) : (
            <StateView kind="empty" title="no decision in this window" />
          )}
          <Note>{th.rule}</Note>
        </>
      )}
    </Panel>
  );
}

function LanePanel({ data, stale }: D) {
  const l = part(data?.lane);
  const t = data?.window.buckets ?? [];
  return (
    <Panel kanji="線" title="THE LANE · RELEASES" tag={l ? `${n(total(l.lane_burst_ended))} BURSTS ENDED` : '—'} tagTone="muted" stale={stale} fill
      sub={l?.lane ? `slot ${l.lane.slot ?? '—'} · ${l.lane.kept ? 'kept' : 'released after bursts'} · ${l.lane.ranked ? `rank ${l.lane.rank}` : 'not ranked'}` : undefined}>
      {!data ? null : !l ? (
        <SectionError what="lane" x={data.lane} />
      ) : (
        <>
          <StackedBars t={t} bucketS={data.window.bucket_s} height={50} label="lane burst ended · kept"
            series={[{ name: 'lane burst ended', values: l.lane_burst_ended, slot: 0 }, { name: 'kept (no release)', values: l.lane_kept, slot: 1 }]} />
          <KV k="EVERY RELEASE, BY WHY" v={<Counts m={l.by_why} />} />
          {l.recent.length ? (
            <Table
              tall
              rows={l.recent.slice(0, 12)}
              rowKey={(r, i) => `${r.ts}-${i}`}
              columns={[
                { key: 't', head: 'at', cell: (r) => clock(r.ts) },
                { key: 's', head: 'slot', num: true, cell: (r) => r.slot ?? '—' },
                { key: 'w', head: 'why', cell: (r) => <span {...stylex.props(s.asIs)}>{r.why}{r.by_why ? ` (${r.by_why})` : ''}</span> },
                { key: 'r', head: 'released', cell: (r) => (r.released ? <Chip tone="moss">YES</Chip> : <span {...stylex.props(s.asIs)}>{r.skipped ?? 'no'}</span>) },
                { key: 'c', head: 'cells', num: true, cell: (r) => n(r.cells_before) },
                { key: 'ms', head: 'ms', num: true, cell: (r) => n(r.ms) },
              ]}
            />
          ) : null}
          <Note>{l.note}</Note>
        </>
      )}
    </Panel>
  );
}

function JevPanel({ data, stale }: D) {
  const j = part(data?.jev);
  const t = data?.window.buckets ?? [];
  return (
    <Panel kanji="門" title="JEV API · ENDPOINTS" tag={j ? `${n(j.requests)} CALLS` : '—'} tagTone="cyan" stale={stale} fill
      sub="TypeSafe's public API over jjava (docs/JEV-CONFORMANCE.md): a TypeSafe SDK works with base_url <base>/jev">
      {!data ? null : !j ? (
        <SectionError what="jev" x={data.jev} />
      ) : (
        <>
          <Table
            rows={j.routes}
            rowKey={(r) => `${r.method} ${r.route}`}
            columns={[
              { key: 'm', head: 'method', cell: (r) => r.method },
              { key: 'r', head: 'route', cell: (r) => <span {...stylex.props(s.asIs)}>{r.route}</span> },
              { key: 'n', head: 'requests', num: true, cell: (r) => n(r.requests) },
              { key: 's', head: 'statuses', cell: (r) => <Counts m={r.statuses} toneOf={(k) => statusTone(k)} empty="none" /> },
            ]}
          />
          <StackedBars t={t} bucketS={data.window.bucket_s} height={50} label="calls per bucket by route"
            series={j.routes.map((r, i) => ({ name: r.route, values: r.series, slot: i }))} />
          <div {...stylex.props(layout.stack)}>
            <KV k="MODEL ASKED FOR" v={<Counts m={j.requested} toneOf={(k) => (k.startsWith('jev-') ? 'rose' : 'cyan')} />} title="jev-* ids are Jev's own, served as jjava-latest" />
            <KV k="MODEL THAT READ IT" v={<Counts m={j.served} tone="moss" />} />
            <KV k="TRAFFIC" v={<Counts m={j.traffic} />} />
            <KV k="REFUSALS" v={<Counts m={j.errors} tone="rose" empty="none" />} title="401 missing key, 422 validation, 429 lane busy, 529 model not on the card" />
            <KV k="USAGE" v={`${n(j.usage.input_tokens)} in · ${n(j.usage.output_tokens)} out (reads)`} />
            <KV k="LATENCY PER CALL" v={spreadText(j.latency)} />
          </div>
          {j.config?.models ? (
            <Table
              rows={Object.entries(j.config.models)}
              rowKey={([id]) => id}
              columns={[
                { key: 'id', head: 'id', cell: ([id]) => <span {...stylex.props(s.asIs)}>{id}</span> },
                { key: 'm', head: 'reads', cell: ([, f]) => f.model },
                { key: 'a', head: 'now', cell: ([, f]) => (f.available ? <Chip tone="moss">ON THE CARD</Chip> : <Chip tone="muted">529 NOW</Chip>) },
                { key: 'p', head: 'priors', cell: ([, f]) => (f.priors_measured?.letter_prior && f.priors_measured?.label_bias ? 'measured' : 'unmeasured') },
              ]}
            />
          ) : j.config?.error ? (
            <StateView kind="error" title="the model list could not be read" detail={j.config.error} />
          ) : null}
          {j.config?.aliases ? <KV k="JEV ALIASES → JJAVA-LATEST" v={j.config.aliases.join(', ')} /> : null}
          {j.recent.length ? (
            <Table
              tall
              rows={j.recent}
              rowKey={(r, i) => `${r.ts}-${i}`}
              bad={(r) => (r.status ?? 0) >= 400}
              columns={[
                { key: 't', head: 'at', cell: (r) => clock(r.ts) },
                { key: 'r', head: 'route', cell: (r) => <span {...stylex.props(s.asIs)}>{r.route}</span> },
                { key: 's', head: 'status', cell: (r) => <Chip tone={statusTone(r.status ?? 0)}>{r.status ?? '—'}</Chip> },
                { key: 'm', head: 'asked → read', cell: (r) => <span {...stylex.props(s.asIs)}>{r.requested ?? '—'} → {r.model ?? '—'}</span> },
                { key: 'q', head: 'questions', num: true, cell: (r) => r.questions },
                { key: 'ms', head: 'ms', num: true, cell: (r) => msText(r.ms) },
                { key: 'tr', head: 'traffic', cell: (r) => r.traffic ?? '—' },
              ]}
            />
          ) : (
            <StateView kind="empty" title="no Jev call in this window" />
          )}
          {j.unknown_routes.length ? <KV k="OTHER ROUTES SEEN" v={j.unknown_routes.join(', ')} /> : null}
          <Note>{j.note}</Note>
        </>
      )}
    </Panel>
  );
}

function PriorsPanel({ data, stale }: D) {
  const p = part(data?.priors);
  return (
    <Panel kanji="基" title="PER-MODEL PRIORS" tag={p ? `${Object.keys(p).length} MODELS` : '—'} tagTone="muted" stale={stale} fill
      sub="bench/decider/results/models/<model>.json (decider_bonsai profile_status): what was measured on the model that reads">
      {!data ? null : !p ? (
        <SectionError what="priors" x={data.priors} />
      ) : (
        <Table
          rows={Object.entries(p)}
          rowKey={([m]) => m}
          columns={[
            { key: 'm', head: 'model', cell: ([m, x]) => <>{m}{x.configured ? '' : <> <Chip tone="muted">NOT CONFIGURED</Chip></>}</> },
            { key: 'lp', head: 'letter prior', cell: ([, x]) => (x.priors?.letter_prior ? <Chip tone="moss">MEASURED</Chip> : <Chip tone="muted">NO</Chip>) },
            { key: 'lb', head: 'label bias', cell: ([, x]) => (x.priors?.label_bias ? <Chip tone="moss">MEASURED</Chip> : <Chip tone="muted">NO</Chip>) },
            { key: 'tb', head: 'tie band', cell: ([, x]) => (x.tie_band ? `${x.tie_band.value}${x.tie_band.measured ? '' : ' (bonsai’s)'}` : x.error ?? '—') },
            { key: 'me', head: 'measured', cell: ([, x]) => (x.measured ?? []).join(', ') || '—' },
          ]}
        />
      )}
    </Panel>
  );
}

export function JjavaScreen() {
  const [w, setW] = useWindow('yamadori_jjava_window');
  const poll = usePoll<Jjava>(jjavaPath(w), 30000);
  const d = poll.data;
  const P = jjavaPath(w);
  const guard = (what: string, node: ReactNode) => (
    <ErrorBoundary what={what} source={P} fill>
      {node}
    </ErrorBoundary>
  );
  return (
    <Bento areas={areas.page}>
      <Cell area="head">{guard('JJAVA', <Head d={d} w={w} setW={setW} failure={poll.failure} />)}</Cell>
      <Cell area="use">{guard('USE', <UsePanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="lat">{guard('LATENCY', <LatencyPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="sets">{guard('QUESTION SETS', <SetsPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="jev">{guard('JEV API', <JevPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="thr">{guard('THRESHOLDS', <ThresholdsPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="lane">{guard('THE LANE', <LanePanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="prior">{guard('PRIORS', <PriorsPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="inj">{guard('INJECTOR', <InjectorPanel data={d} stale={poll.stale} fill />)}</Cell>
    </Bento>
  );
}
