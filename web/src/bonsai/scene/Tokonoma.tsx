import * as stylex from '@stylexjs/stylex';
import { lazy, Suspense, useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import type { DatasetsOverview, Pulse, Vitals } from '../../api/types';
import { useNow, usePoll } from '../../api/usePoll';
import { colors, space } from '../../tokens/tokens.stylex';
import { Panel } from '../../ui/Panel';
import { ErrorBoundary } from '../../ui/ErrorBoundary';
import { Chip, Label } from '../../ui/primitives';
import { StateView } from '../../ui/StateView';
import { text } from '../../ui/text';
import { kvSplit } from '../../api/kv';
import { LiveBus, slotsOf } from '../live';
import { CHANNELS, inertChannels, treeParams, treeState } from '../mapping';
import { hex32 } from '../prng';
import { growSkeleton, limbSeeds } from '../skeleton';
import type { TreeLayer } from './Scene';
import { GpuCard, KvCard, QueueCard, RateCard, SeedCard, SlotsCard, StrataCard, telemetry, ToolsCard } from './Telemetry';

const BonsaiCanvas = lazy(() => import('./Scene').then((m) => ({ default: m.BonsaiCanvas })));

/** The fast subset of vitals (mcp/vitals.py pulse()), polled every second. */
const PULSE = '/dash/api/vitals/pulse';
const PULSE_MS = 1000;
/** Seconds of decode-rate history in the DECODE sparkline. */
const HISTORY = 90;

// prefers-reduced-motion stills the scene; it never removes the tree.
const RM = '(prefers-reduced-motion: reduce)';
function subscribeMotion(cb: () => void) {
  const m = window.matchMedia(RM);
  m.addEventListener('change', cb);
  return () => m.removeEventListener('change', cb);
}
const readMotion = () => window.matchMedia(RM).matches;

/**
 * The genome when no seed has ever been reported. Not a data value: the
 * caption says UNSEEDED, and this constant is the FNV-1a of the empty
 * string, i.e. "no word".
 */
const UNSEEDED = 0x811c9dc5;

const s = stylex.create({
  stage: {
    position: 'relative',
    // Grows with its bento cell so the hero tile meets the column beside it.
    minHeight: { default: '560px', '@media (max-width: 767px)': '420px' },
    flexGrow: 1,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 35%, transparent)`,
    overflow: 'hidden',
  },
  vignette: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    backgroundImage: `radial-gradient(ellipse at 50% 45%, transparent 45%, color-mix(in srgb, ${colors.surfaceContainerLowest} 85%, transparent) 100%)`,
  },
  scan: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    opacity: 0.35,
    backgroundImage: `repeating-linear-gradient(to bottom, transparent 0, transparent 2px, color-mix(in srgb, ${colors.surfaceContainerLowest} 40%, transparent) 3px)`,
  },
  corner: { position: 'absolute', width: '18px', height: '18px', borderColor: colors.primaryContainer, borderStyle: 'solid', opacity: 0.7 },
  tl: { top: '8px', left: '8px', borderTopWidth: 1, borderLeftWidth: 1, borderRightWidth: 0, borderBottomWidth: 0 },
  tr: { top: '8px', right: '8px', borderTopWidth: 1, borderRightWidth: 1, borderLeftWidth: 0, borderBottomWidth: 0 },
  bl: { bottom: '8px', left: '8px', borderBottomWidth: 1, borderLeftWidth: 1, borderRightWidth: 0, borderTopWidth: 0 },
  br: { bottom: '8px', right: '8px', borderBottomWidth: 1, borderRightWidth: 1, borderLeftWidth: 0, borderTopWidth: 0 },
  capTop: { position: 'absolute', top: '14px', left: '16px', right: '16px', display: 'flex', justifyContent: 'space-between', gap: space.spaceSm, pointerEvents: 'none' },
  capBottom: {
    position: 'absolute',
    bottom: '14px',
    left: '16px',
    right: '16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    gap: space.spaceSm,
    pointerEvents: 'none',
  },
  specimen: { margin: 0, color: colors.primary, textShadow: `0 0 20px color-mix(in srgb, ${colors.primaryContainer} 40%, transparent)` },
  kanji: { color: colors.primaryContainer, fontSize: '28px', lineHeight: '32px', margin: 0 },
  legend: { display: 'flex', flexDirection: 'column', gap: '2px', alignItems: 'flex-end', textAlign: 'right' },
  legendRow: { color: colors.onSurfaceVariant },
  rate: { margin: 0, color: colors.primaryContainer, textShadow: `0 0 16px color-mix(in srgb, ${colors.primaryContainer} 45%, transparent)` },
  rateIdle: { color: colors.outline, textShadow: 'none' },
  unit: { color: colors.outline, marginInlineStart: '6px' },
  details: { color: colors.onSurfaceVariant },
  chanList: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.4fr)', columnGap: space.spaceSm, rowGap: '2px', marginTop: space.spaceXs },
  dead: { color: colors.outlineVariant },
  live: { color: colors.primaryContainer },
});

export function Tokonoma({ vitals, datasets, fill }: { vitals: Vitals | null; datasets: DatasetsOverview | null; fill?: boolean }) {
  const still = useSyncExternalStore(subscribeMotion, readMotion, () => false);
  // A server that predates the pulse answers 404: stop asking, and run on
  // the 5 s snapshot instead (it carries the same fields once restarted).
  const [gone, setGone] = useState(false);
  const pulseProbe = usePoll<Pulse>(gone ? null : PULSE, PULSE_MS);
  const missing = pulseProbe.failure?.kind === 'http' && pulseProbe.failure.status === 404;
  useEffect(() => {
    if (missing) setGone(true);
  }, [missing]);
  const now = useNow(1000);
  const pulse = gone ? null : pulseProbe.data;
  const since = (gone ? null : pulseProbe.receivedAt) ?? now;
  const [bus] = useState(() => new LiveBus());
  useEffect(() => {
    bus.push(pulse, vitals);
  }, [bus, pulse, vitals]);
  const slots = slotsOf(pulse, vitals);
  const tps = slots ? slots.slots.reduce((a, x) => a + (x.tps || 0), 0) : null;
  const busy = slots ? slots.slots.filter((x) => x.state !== 'idle').length : null;
  const [history, setHistory] = useState<number[]>([]);
  useEffect(() => {
    if (tps !== null) setHistory((h) => [...h.slice(-(HISTORY - 1)), tps]);
  }, [pulse, tps]);

  const gpus = pulse?.gpus ?? vitals?.gpus ?? [];

  const state = useMemo(() => treeState(vitals, datasets), [vitals, datasets]);
  const seed = state.seed ?? UNSEEDED;
  const sk = useMemo(() => growSkeleton(limbSeeds(seed)), [seed]);
  const params = useMemo(() => treeParams(state, sk), [state, sk]);
  const inert = useMemo(() => new Set(inertChannels(state)), [state]);

  // New genome -> new tree. The old one retracts while the new one grows:
  // topology is never interpolated (BONSAI-VIZ §4).
  const [layers, setLayers] = useState<TreeLayer[]>([]);
  useEffect(() => {
    const key = hex32(seed);
    setLayers((prev) => {
      const others = prev.filter((l) => l.key !== key).map((l) => ({ ...l, dying: true }));
      return [...others, { key, sk, target: params, dying: false }];
    });
  }, [seed, sk, params]);

  const word = vitals?.seed?.word ?? null;
  const kv = kvSplit(vitals?.context);
  const hasSeedField = vitals !== null && 'seed' in vitals;

  return (
    <Panel
      kanji="床の間"
      title="TOKONOMA"
      tag={vitals ? 'LIVE SPECIMEN' : 'NO SNAPSHOT'}
      tagTone={vitals ? 'moss' : 'muted'}
      flag={`GENOME ${hex32(seed)}`}
      edge="moss"
      fill={fill}
    >
      <div {...stylex.props(s.stage)}>
        <ErrorBoundary what="the 3D tree" bare>
          <Suspense fallback={<StateView kind="loading" title="loading the renderer" />}>
            <BonsaiCanvas
              layers={layers}
              scene={params.scene}
              onLayerGone={(key) => setLayers((prev) => prev.filter((l) => !(l.key === key && l.dying)))}
              bus={bus}
              still={still}
            />
          </Suspense>
        </ErrorBoundary>
        <span {...stylex.props(s.vignette)} />
        <span {...stylex.props(s.scan)} />
        <span {...stylex.props(s.corner, s.tl)} />
        <span {...stylex.props(s.corner, s.tr)} />
        <span {...stylex.props(s.corner, s.bl)} />
        <span {...stylex.props(s.corner, s.br)} />
        <div {...stylex.props(s.capTop)}>
          <div>
            <p {...stylex.props(text.kanji, s.kanji)}>盆栽</p>
            <Label>SPECIMEN · {hex32(seed)}</Label>
          </div>
          <div {...stylex.props(s.legend)}>
            <p {...stylex.props(text.headlineSm, text.num, s.rate, !busy && s.rateIdle)}>
              {tps === null ? '—' : tps.toFixed(1)}
              <span {...stylex.props(text.labelXs, s.unit)}>TOK/S</span>
            </p>
            <span {...stylex.props(text.labelXs, s.legendRow)}>
              {busy === null ? 'SLOTS NOT REPORTED' : busy === 0 ? 'ALL SLOTS IDLE' : `${busy} SLOT${busy === 1 ? '' : 'S'} WORKING`}
            </span>
          </div>
        </div>
        <div {...stylex.props(s.capBottom)}>
          <div>
            <p {...stylex.props(text.headlineLg, s.specimen)}>{word ?? 'unseeded'}</p>
            <Label>{word ? 'concept seed' : hasSeedField ? 'no seed yet' : 'seed not reported'}</Label>
          </div>
          <div {...stylex.props(s.legend)}>
            <span {...stylex.props(text.labelXs, s.legendRow)}>
              MAIN LIMB ← KV MAIN {state.mainShare === null ? '—' : `${(state.mainShare * 100).toFixed(0)}%`}
            </span>
            <span {...stylex.props(text.labelXs, s.legendRow)}>
              THINKING LIMB ← KV DEEP THINKING {state.thinkingShare === null ? '—' : `${(state.thinkingShare * 100).toFixed(0)}%${kv && kv.helpers > 1 ? ` ×${kv.helpers}` : ''}`}
            </span>
            <span {...stylex.props(text.labelXs, s.legendRow)}>
              ARCS ← GPU0 UTIL {state.activity === null ? '—' : `${Math.round(state.activity * 100)}%`}
            </span>
          </div>
        </div>
      </div>
      <div {...stylex.props(telemetry.grid)}>
        <SlotsCard slots={pulse?.slots ?? vitals?.slots} lanes={pulse?.lanes ?? vitals?.lanes} context={pulse?.context ?? vitals?.context} since={since} />
        <ToolsCard tools={pulse?.tools ?? vitals?.tools} since={since} now={now} />
        <RateCard history={history} current={tps} />
        {gpus.map((g) => (
          <GpuCard key={g.index} g={g} />
        ))}
        <KvCard context={pulse?.context ?? vitals?.context} />
        <StrataCard strata={pulse?.strata ?? vitals?.strata} />
        <QueueCard queue={pulse?.queue ?? vitals?.queue} />
        <SeedCard seed={pulse?.seed ?? vitals?.seed} />
      </div>
      <details {...stylex.props(text.labelXs, s.details)}>
        <summary>
          CHANNELS · {CHANNELS.length - inert.size} LIVE · {inert.size} INERT
        </summary>
        <div {...stylex.props(s.chanList)}>
          {CHANNELS.map((c) => (
            <span key={c.name} style={{ display: 'contents' }}>
              <span {...stylex.props(inert.has(c.name) ? s.dead : s.live)}>
                {inert.has(c.name) ? '○' : '●'} {c.name}
              </span>
              <span>{c.source ? `${c.api}.${c.source}` : <Chip tone="muted">INERT</Chip>}</span>
            </span>
          ))}
        </div>
      </details>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {state.fanout === null && <Chip tone="muted">FAN-OUT · INERT</Chip>}
        <Chip tone="muted">FOLIAGE · INERT</Chip>
        {state.alarms ? <Chip tone="crimson">{state.alarms} CRIMSON CAP{state.alarms === 1 ? '' : 'S'}</Chip> : null}
        {state.errored ? <Chip tone="rose">{state.errored} SHARI · ERRORED JOBS</Chip> : null}
      </div>
    </Panel>
  );
}
