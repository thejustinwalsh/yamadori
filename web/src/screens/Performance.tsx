// 速度 SOKUDO -- performance: tokens per second per model from real traffic,
// by time and by context depth; both GPUs' utilisation, memory, power and
// temperature over the window; the GPU gates' arms beside them at the same
// depths; the model swaps. Replaces the benchmark page (SENTEI, retired
// 2026-09-30). /dash/api/perf (mcp/dash_perf.py) reads records only; the
// last ten minutes live are KŌGŌSEI's (/dash/api/power/series).
import * as stylex from '@stylexjs/stylex';
import type { ReactNode } from 'react';
import {
  MODEL_SLOT,
  part,
  perfPath,
  shortCard,
  spreadText,
  tpsText,
  type GateFile,
  type ModelPerf,
  type Perf,
} from '../api/stats';
import { SERIES_PATH } from '../api/powerSeries';
import { usePoll } from '../api/usePoll';
import { clock, n } from '../format';
import { colors, space } from '../tokens/tokens.stylex';
import { Bento, Cell } from '../ui/Bento';
import { MQ } from '../ui/breakpoints.stylex';
import { LineChart, type Serie } from '../ui/Charts';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { KogoseiLive } from '../ui/KogoseiPanel';
import { Panel } from '../ui/Panel';
import { Chip, Label, layout, Stat } from '../ui/primitives';
import { Counts, KV, Note, SectionError, useWindow, WindowPicker } from '../ui/StatsParts';
import { PollState, StateView } from '../ui/StateView';
import { Table } from '../ui/Table';
import { text } from '../ui/text';

const areas = stylex.create({
  page: {
    gridTemplateAreas: {
      default: '"head" "speed" "ctx" "gpu" "live" "gates" "swaps" "left"',
      [MQ.tablet]: '"head head head head head head" "speed speed speed speed speed speed" "ctx ctx ctx ctx ctx ctx" "gpu gpu gpu gpu gpu gpu" "live live live swaps swaps swaps" "gates gates gates gates gates gates" "left left left left left left"',
      [MQ.desktop]:
        '"head head head head head head head head head head head head" "speed speed speed speed speed speed speed ctx ctx ctx ctx ctx" "gpu gpu gpu gpu gpu gpu gpu gpu live live live live" "gates gates gates gates gates gates gates gates swaps swaps swaps swaps" "gates gates gates gates gates gates gates gates left left left left"',
    },
  },
});

const s = stylex.create({
  head: { display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: space.spaceSm },
  cards: { display: 'grid', gap: space.spaceSm, gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))' },
  card: { display: 'flex', flexDirection: 'column', gap: space.spaceXs, padding: space.spaceXs, backgroundColor: colors.surfaceContainerLowest, borderWidth: 1, borderStyle: 'solid', borderColor: `color-mix(in srgb, ${colors.outlineVariant} 30%, transparent)`, minWidth: 0 },
  asIs: { textTransform: 'none' },
  summary: { cursor: 'pointer', overflowWrap: 'anywhere' },
});

type D = { data: Perf | null; stale?: boolean };

const slotOf = (model: string, i: number) => MODEL_SLOT[model] ?? Math.min(3, i);

function Head({ d, w, setW, failure }: { d: Perf | null; w: ReturnType<typeof useWindow>[0]; setW: ReturnType<typeof useWindow>[1]; failure: Parameters<typeof PollState>[0]['failure'] }) {
  const ms = part(d?.models) ?? [];
  const st = d?.sources.stats;
  return (
    <Panel kanji="速度" title="SOKUDO · PERFORMANCE" tag={d ? `${n(d.sources.generations)} GENERATIONS · ${w.toUpperCase()}` : '—'} tagTone="moss" edge="moss"
      sub="speed as llama-server reports it (timings), per model and depth, from real traffic; both cards; the gates' arms at the same depths. Nothing here asks a model.">
      <div {...stylex.props(s.head)}>
        <WindowPicker value={w} onChange={setW} />
        {d ? <Label>READ {clock(d.at)} · BUCKET {Math.round(d.window.bucket_s / 60)} MIN</Label> : null}
      </div>
      {!d ? (
        <PollState path={perfPath(w)} failure={failure} />
      ) : (
        <div {...stylex.props(layout.grid4)}>
          {ms.slice(0, 3).map((m) => (
            <Stat key={m.model} label={`${m.model} · DECODE`} value={`${tpsText(m.decode_tps.p50)} tok/s`} sub={`p90 ${tpsText(m.decode_tps.p90)} · n ${n(m.decode_tps.n)} · prefill p50 ${tpsText(m.prompt_tps.p50)}`} tone="moss" />
          ))}
          <Stat label="STATS STORE" value={st?.enabled ? 'RECORDING' : 'NOT HERE'} sub={`${st?.enabled ? '' : 'this process does not record · '}${n(st?.written ?? 0)} written · ${n(st?.dropped ?? 0)} dropped`} tone={st?.enabled ? 'moss' : 'rose'} />
        </div>
      )}
    </Panel>
  );
}

function SpeedPanel({ data, stale }: D) {
  const ms = part(data?.models);
  const t = data?.window.buckets ?? [];
  const b = data?.window.bucket_s ?? 3600;
  const pick = (f: (m: ModelPerf) => (number | null)[]): Serie[] => (ms ?? []).slice(0, 4).map((m, i) => ({ name: m.model, values: f(m), slot: slotOf(m.model, i) }));
  return (
    <Panel kanji="流" title="TOK/S PER MODEL · OVER TIME" tag={ms ? `${ms.length} MODELS` : '—'} tagTone="cyan" stale={stale} fill
      sub="per bucket, the median over that bucket's generations (a one-token decider read has no decode rate and is counted apart)">
      {!data ? null : !ms ? (
        <SectionError what="models" x={data.models} />
      ) : !ms.length ? (
        <StateView kind="empty" title="no generation recorded in this window" detail="The history starts at the deploy that ships mcp/stats_store.py; the proxy, the worker and the tools API record into it." />
      ) : (
        <>
          <LineChart t={t} bucketS={b} unit="tok/s" label="decode · median" series={pick((m) => m.series.decode_p50)} height={110} />
          <LineChart t={t} bucketS={b} unit="tok/s" label="prefill · median" series={pick((m) => m.series.prompt_p50)} height={80} />
          <Table
            rows={ms}
            rowKey={(m) => m.model}
            columns={[
              { key: 'm', head: 'model', cell: (m) => m.model },
              { key: 'g', head: 'generations', num: true, cell: (m) => n(m.generations) },
              { key: 'd', head: 'decode tok/s', cell: (m) => spreadText(m.decode_tps, tpsText) },
              { key: 'p', head: 'prefill tok/s', cell: (m) => spreadText(m.prompt_tps, tpsText) },
              { key: 'dr', head: 'decider read prefill', cell: (m) => spreadText(m.decider_read_prompt_tps, tpsText) },
              { key: 'r', head: 'roles', cell: (m) => <Counts m={m.roles} /> },
              { key: 't', head: 'tokens', cell: (m) => `${n(m.tokens.completion)} out · ${n(m.tokens.processed)} prefilled · ${n(m.tokens.reused)} reused` },
            ]}
          />
        </>
      )}
    </Panel>
  );
}

/** Real traffic by depth beside the gates' arms at the same depths. */
function CtxPanel({ data, stale }: D) {
  const ms = part(data?.models);
  const bins = data?.ctx_bins ?? [];
  const rows = (ms ?? []).flatMap((m) => m.by_ctx.map((c) => ({ model: m.model, ...c })));
  return (
    <Panel kanji="深" title="TOK/S BY CONTEXT DEPTH" tag={bins.length ? `${bins.length} BINS` : '—'} tagTone="cyan" stale={stale} fill
      sub="bins at the gates' measurement depths (4K / 8K / 32K / 64K / 128K), so traffic reads beside the gate's arms below">
      {!data ? null : !ms ? (
        <SectionError what="models" x={data.models} />
      ) : !rows.length ? (
        <StateView kind="empty" title="no generation recorded in this window" />
      ) : (
        <Table
          rows={rows}
          rowKey={(r) => `${r.model}-${r.bin}`}
          columns={[
            { key: 'm', head: 'model', cell: (r) => r.model },
            { key: 'b', head: 'context', cell: (r) => r.bin },
            { key: 'n', head: 'n', num: true, cell: (r) => n(r.n) },
            { key: 'd', head: 'decode p50 · p90', num: true, cell: (r) => `${tpsText(r.decode.p50)} · ${tpsText(r.decode.p90)}` },
            { key: 'p', head: 'prefill p50', num: true, cell: (r) => tpsText(r.prompt.p50) },
          ]}
        />
      )}
    </Panel>
  );
}

function GpuPanel({ data, stale }: D) {
  const g = part(data?.gpus);
  return (
    <Panel kanji="鉄" title="BOTH GPUS · OVER TIME" tag={g ? `${g.cards.length} CARDS · ${Math.round(g.bucket_s / 60)} MIN POINTS` : '—'} tagTone="cyan" stale={stale} fill
      sub="the power sampler's 1 s nvidia-smi reads, one row per card per minute (mcp/stats_store.py), folded per window. One chart per measure: no second axis.">
      {!data ? null : !g ? (
        <SectionError what="gpus" x={data.gpus} />
      ) : !g.cards.length ? (
        <StateView kind="empty" title="no GPU minute recorded in this window" detail="Recorded by the proxy's power sampler since the deploy that ships mcp/stats_store.py." />
      ) : (
        <div {...stylex.props(s.cards)}>
          {g.cards.map((c) => {
            const t = c.series.t;
            const name = `${shortCard(c.name, c.idx)}${c.main ? ' · MAIN' : ''}`;
            return (
              <div key={c.idx} {...stylex.props(s.card)}>
                <div {...stylex.props(layout.between)}>
                  <span {...stylex.props(text.labelMd, text.primary)}>GPU{c.idx} · {name}</span>
                  <Chip tone="muted">{c.total_mib ? `${n(c.total_mib)} MiB` : '—'}</Chip>
                </div>
                <LineChart t={t} bucketS={g.bucket_s} unit="%" label="utilisation · mean and max" max={100} height={60}
                  series={[{ name: 'mean', values: c.series.util_avg, slot: 0 }, { name: 'max', values: c.series.util_max, slot: 1 }]} />
                <LineChart t={t} bucketS={g.bucket_s} unit="MiB" label="VRAM used · max" max={c.total_mib ?? undefined} height={50} fmt={(x) => n(Math.round(x))}
                  series={[{ name: 'used', values: c.series.used_max, slot: 1 }]} />
                <LineChart t={t} bucketS={g.bucket_s} unit="W" label="power · mean" height={50} series={[{ name: 'watts', values: c.series.watts_avg, slot: 2 }]} />
                <LineChart t={t} bucketS={g.bucket_s} unit="°C" label="temperature · max" height={50} series={[{ name: 'temp', values: c.series.temp_max, slot: 3 }]} />
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

function SwapsPanel({ data, stale }: D) {
  const sw = part(data?.swaps);
  return (
    <Panel kanji="換" title="MODEL SWAPS" tag={sw ? `${sw.rows.length} IN THE WINDOW` : '—'} tagTone="rose" stale={stale} fill
      sub="max_mode.wait_ready: one main model on the 5060 Ti at a time; the load's seconds are llama-swap's start of the model">
      {!data ? null : !sw ? (
        <SectionError what="swaps" x={data.swaps} />
      ) : (
        <>
          {Object.entries(sw.load_s).map(([to, x]) => (
            <KV key={to} k={`LOAD → ${to}`} v={spreadText(x, (v) => (v == null ? '—' : `${v} s`))} />
          ))}
          {sw.rows.length ? (
            <Table
              tall
              rows={sw.rows}
              rowKey={(r, i) => `${r.ts}-${i}`}
              bad={(r) => !r.ok || r.left_loaded.length > 0}
              columns={[
                { key: 't', head: 'at', cell: (r) => clock(r.ts) },
                { key: 'f', head: 'from → to', cell: (r) => `${r.from.join(', ') || 'nothing'} → ${r.to ?? '—'}` },
                { key: 'l', head: 'load s', num: true, cell: (r) => r.load_s ?? '—' },
                { key: 'o', head: 'ok', cell: (r) => (r.ok ? <Chip tone="moss">OK</Chip> : <Chip tone="crimson">{r.how ?? 'FAILED'}</Chip>) },
                { key: 'x', head: 'still loaded', cell: (r) => r.left_loaded.join(', ') || '—' },
              ]}
            />
          ) : (
            <StateView kind="empty" title="no swap in this window" />
          )}
          <KV k="TIER → MODEL" v={JSON.stringify((sw.table as { tiers?: unknown }).tiers ?? sw.table)} />
        </>
      )}
    </Panel>
  );
}

function GateCard({ g }: { g: GateFile }) {
  const speed = g.groups.filter((x) => x.decode.n || x.prompt.n);
  const passes = Object.entries(g.pass);
  return (
    <details open={g.in_progress || undefined}>
      <summary {...stylex.props(text.labelMd, text.primary, s.summary)}>
        {g.kind.toUpperCase()} · <span {...stylex.props(s.asIs)}>{g.source}</span> · {g.started ?? '—'}
        {g.in_progress ? ' · IN PROGRESS' : ''} · {speed.length} RUN SETS
      </summary>
      <div {...stylex.props(layout.stack)}>
        {passes.length ? (
          <div {...stylex.props(layout.rowWrap)}>
            {passes.map(([k, v]) => (
              <Chip key={k} tone={v ? 'moss' : 'crimson'}>
                <span {...stylex.props(s.asIs)}>{k}</span> {v ? 'PASS' : 'FAIL'}
              </Chip>
            ))}
          </div>
        ) : null}
        {speed.length ? (
          <Table
            rows={speed}
            rowKey={(x) => `${x.arm}/${x.what}`}
            columns={[
              { key: 'a', head: 'arm', cell: (x) => <span {...stylex.props(s.asIs)}>{x.arm}</span> },
              { key: 'w', head: 'measure', cell: (x) => <span {...stylex.props(s.asIs)}>{x.what}</span> },
              { key: 'n', head: 'n', num: true, cell: (x) => x.n },
              { key: 'd', head: 'decode median [min–max]', num: true, cell: (x) => (x.decode.n ? `${tpsText(x.decode.median)} [${tpsText(x.decode.min)}–${tpsText(x.decode.max)}]` : '—') },
              { key: 'p', head: 'prefill median', num: true, cell: (x) => tpsText(x.prompt.median) },
              { key: 'c', head: 'prompt tokens', num: true, cell: (x) => n(x.prompt_n) },
            ]}
          />
        ) : (
          <StateView kind="empty" title="no speed runs in this file" />
        )}
      </div>
    </details>
  );
}

function GatesPanel({ data, stale }: D) {
  const g = part(data?.gates);
  return (
    <Panel kanji="関" title="GATES · ARMS BY DEPTH" tag={g ? `${g.files.length} RESULT FILES` : '—'} tagTone="muted" stale={stale} fill
      sub="each gate's runs as measured (llama-server timings), with n: the Flash-Next gate, the Mirai S gate, bench/results/kv_rank. Each run is one; repeat before citing a difference (docs/PROTOCOL.md).">
      {!data ? null : !g ? (
        <SectionError what="gates" x={data.gates} />
      ) : !g.files.length ? (
        <StateView kind="empty" title="no gate result found" detail={g.globs.map((x) => x.glob).join(' · ')} />
      ) : (
        <div {...stylex.props(layout.stackSm)}>
          {g.files.map((f) => (
            <GateCard key={`${f.kind}-${f.source}-${f.file}`} g={f} />
          ))}
          {g.errors.map((e) => (
            <StateView key={e.file} kind="error" title={`${e.file} could not be read`} detail={e.error} />
          ))}
        </div>
      )}
    </Panel>
  );
}

function LeftOutPanel({ data }: D) {
  const lo = data?.left_out ?? [];
  return (
    <Panel kanji="欠" title="LEFT OUT, AND WHY" tag={`${lo.length}`} tagTone="muted" fill>
      <div {...stylex.props(layout.stack)}>
        {lo.map((x) => (
          <div key={x.graph}>
            <Label>{x.graph}</Label>
            <Note>{x.why}</Note>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export function PerformanceScreen() {
  const [w, setW] = useWindow('yamadori_perf_window');
  const poll = usePoll<Perf>(perfPath(w), 30000);
  const d = poll.data;
  const P = perfPath(w);
  const guard = (what: string, node: ReactNode, source = P) => (
    <ErrorBoundary what={what} source={source} fill>
      {node}
    </ErrorBoundary>
  );
  return (
    <Bento areas={areas.page}>
      <Cell area="head">{guard('SOKUDO', <Head d={d} w={w} setW={setW} failure={poll.failure} />)}</Cell>
      <Cell area="speed">{guard('TOK/S PER MODEL', <SpeedPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="ctx">{guard('TOK/S BY DEPTH', <CtxPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="gpu">{guard('BOTH GPUS', <GpuPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="live">{guard('KŌGŌSEI · WATTS PER TOKEN', <KogoseiLive fill />, SERIES_PATH)}</Cell>
      <Cell area="gates">{guard('GATES', <GatesPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="swaps">{guard('MODEL SWAPS', <SwapsPanel data={d} stale={poll.stale} />)}</Cell>
      <Cell area="left">{guard('LEFT OUT', <LeftOutPanel data={d} />)}</Cell>
    </Bento>
  );
}
