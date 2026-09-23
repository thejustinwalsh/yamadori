// The live readouts around the tokonoma: what the model and its tools are
// doing this second, from /dash/api/vitals/pulse (mcp/vitals.py pulse()).
// Counters that grow between polls (context held while decoding, a running
// tool's elapsed time) are interpolated at their measured rate, so nothing
// jumps once a second; interpolation stops a few seconds past the last poll
// rather than inventing progress that was never measured.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useRef } from 'react';
import { kvSplit } from '../../api/kv';
import type { ContextPool, Gpu, JobQueue, Lanes, Seed, Slot, Slots, Strata, ToolActivity } from '../../api/types';
import { ago, gib, n } from '../../format';
import { colors, space } from '../../tokens/tokens.stylex';
import { Label, Meter, SplitBar, type Tone } from '../../ui/primitives';
import { SeedBits } from '../../ui/SeedReadout';
import { windowLabel } from '../../ui/Strata';
import { text } from '../../ui/text';
import { hex32 } from '../prng';

/** Interpolation never runs further than this past the last measurement. */
const HOLD_S = 3;

const flash = stylex.keyframes({
  '0%': { opacity: 0.9 },
  '100%': { opacity: 0 },
});

const s = stylex.create({
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: space.spaceSm,
  },
  wide: { gridColumn: { default: 'span 1', '@media (min-width: 560px)': 'span 2' } },
  card: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    gap: space.spaceXs,
    padding: space.spaceSm,
    minWidth: 0,
    backgroundColor: colors.surfaceContainerLowest,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: `color-mix(in srgb, ${colors.outlineVariant} 45%, transparent)`,
    overflow: 'hidden',
  },
  head: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: space.spaceXs, minWidth: 0 },
  flash: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    boxShadow: `inset 0 0 0 1px ${colors.tertiaryContainer}, inset 0 0 18px color-mix(in srgb, ${colors.tertiaryContainer} 35%, transparent)`,
    opacity: 0,
    animationName: { default: flash, '@media (prefers-reduced-motion: reduce)': 'none' },
    animationDuration: '1.4s',
    animationTimingFunction: 'ease-out',
  },
  flashMoss: {
    boxShadow: `inset 0 0 0 1px ${colors.primaryContainer}, inset 0 0 18px color-mix(in srgb, ${colors.primaryContainer} 35%, transparent)`,
  },
  big: { margin: 0, color: colors.primary, whiteSpace: 'nowrap' },
  unit: { color: colors.outline, marginInlineStart: '6px' },
  dim: { color: colors.outline },
  soft: { color: colors.onSurfaceVariant },
  moss: { color: colors.primaryContainer },
  cyan: { color: colors.tertiaryContainer },
  slotRow: {
    display: 'grid',
    gridTemplateColumns: '28px 64px minmax(0, 1fr) auto',
    alignItems: 'center',
    columnGap: space.spaceXs,
  },
  toolRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto auto',
    columnGap: space.spaceSm,
    alignItems: 'baseline',
  },
  name: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0, textTransform: 'none' },
  spark: { width: '100%', height: '38px', display: 'block' },
  split3: { display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: space.spaceXs },
  word: { margin: 0, color: colors.primaryContainer, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
});

/** A number that grows at `rate`/s from `base`, measured at `since` (ms). */
function Ticker({ base, rate, since, format }: { base: number; rate: number; since: number; format: (x: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const paint = () => {
      const dt = Math.min(HOLD_S, Math.max(0, (Date.now() - since) / 1000));
      el.textContent = format(base + rate * dt);
    };
    paint();
    if (!rate) return;
    const id = window.setInterval(paint, 100);
    return () => window.clearInterval(id);
  }, [base, rate, since, format]);
  return <span ref={ref} {...stylex.props(text.num)} />;
}

const fmtInt = (x: number) => n(Math.round(x));
const fmtSec = (x: number) => (x < 10 ? `${x.toFixed(1)} s` : `${Math.round(x)} s`);
const fmtMs = (ms: number | null | undefined) =>
  typeof ms === 'number' && Number.isFinite(ms) ? (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`) : '—';

function Card({ label, right, wide, flashKey, tone, children }: {
  label: string;
  right?: React.ReactNode;
  wide?: boolean;
  /** changes when the card's subject fires an event: the border flashes once */
  flashKey?: string | number | null;
  tone?: 'moss' | 'cyan';
  children: React.ReactNode;
}) {
  return (
    <div {...stylex.props(s.card, wide && s.wide)}>
      {flashKey !== undefined && flashKey !== null && (
        <span key={flashKey} aria-hidden {...stylex.props(s.flash, tone === 'moss' && s.flashMoss)} />
      )}
      <div {...stylex.props(s.head)}>
        <Label>{label}</Label>
        {right !== undefined && <span {...stylex.props(text.labelXs, s.dim)}>{right}</span>}
      </div>
      {children}
    </div>
  );
}

const STATE_TONE: Record<Slot['state'], 'moss' | 'cyan' | 'dim'> = { decode: 'moss', prefill: 'cyan', idle: 'dim' };

export function SlotsCard({ slots, lanes, context, since }: {
  slots: Slots | null | undefined;
  lanes: Lanes | null | undefined;
  context: ContextPool | null | undefined;
  since: number;
}) {
  const kv = kvSplit(context);
  const busy = slots?.ok ? slots.slots.filter((x) => x.state !== 'idle').length : null;
  const laneText =
    lanes && lanes.in_proxy
      ? `MAIN ${lanes.main}/${lanes.main_lanes ?? '—'} · DEEP THINKING ${lanes.helper}/${lanes.helper_lanes ?? '—'}`
      : busy === null
        ? undefined
        : `${busy} BUSY`;
  return (
    <Card label="SLOTS · LLAMA-SERVER" right={laneText} wide flashKey={busy ? `b${busy}` : null} tone="moss">
      {!slots ? (
        <span {...stylex.props(text.labelXs, s.dim)}>slot state not reported by this server</span>
      ) : !slots.ok ? (
        <span {...stylex.props(text.labelXs, s.dim)}>/slots not answering · {slots.error ?? 'no reason given'}</span>
      ) : (
        slots.slots.map((x) => {
          const tone = STATE_TONE[x.state];
          const rate = x.state === 'decode' ? x.tps : x.state === 'prefill' ? x.pps : 0;
          return (
            <div key={x.id} {...stylex.props(text.labelXs, s.slotRow)}>
              <span {...stylex.props(s.dim)}>S{x.id}</span>
              <span {...stylex.props(s[tone])}>{x.state.toUpperCase()}</span>
              <Meter
                value={x.state === 'idle' || !kv ? (x.state === 'idle' ? 0 : null) : Math.min(1, x.ctx / kv.main)}
                tone={x.state === 'prefill' ? 'cyan' : 'moss'}
                label={`slot ${x.id} holds ${x.ctx} tokens`}
              />
              <span {...stylex.props(s.soft)}>
                {x.state === 'idle' ? (
                  '—'
                ) : (
                  <>
                    <Ticker base={x.ctx} rate={x.state === 'decode' ? x.tps : x.pps} since={since} format={fmtInt} /> CTX ·{' '}
                    {rate ? `${rate.toFixed(rate < 100 ? 1 : 0)} TOK/S` : '…'}
                  </>
                )}
              </span>
            </div>
          );
        })
      )}
    </Card>
  );
}

export function ToolsCard({ tools, since, now }: { tools: ToolActivity | null | undefined; since: number; now: number }) {
  if (!tools) {
    return (
      <Card label="TOOLS" wide>
        <span {...stylex.props(text.labelXs, s.dim)}>tool activity not reported by this server</span>
      </Card>
    );
  }
  const win = windowLabel(tools.window_seconds);
  const run = tools.running;
  const last = tools.recent[0];
  const t = now / 1000;
  return (
    <Card
      label="TOOLS"
      right={`${win} · ${tools.calls_window} CALLS · ${tools.turns_window} REQ`}
      wide
      flashKey={tools.last?.id ?? null}
      tone="cyan"
    >
      <div {...stylex.props(s.toolRow)}>
        {run ? (
          <>
            <span {...stylex.props(text.titleMd, s.cyan, s.name)}>▶ {run.name}</span>
            <span {...stylex.props(text.labelXs, s.soft)}>
              <Ticker base={Math.max(0, since / 1000 - run.at)} rate={1} since={since} format={fmtSec} />
            </span>
            <span {...stylex.props(text.labelXs, s.cyan)}>RUNNING</span>
          </>
        ) : last ? (
          <>
            <span {...stylex.props(text.titleMd, s.moss, s.name)}>{last.name}</span>
            <span {...stylex.props(text.labelXs, s.soft, text.num)}>{fmtMs(last.ms)}</span>
            <span {...stylex.props(text.labelXs, s.dim)}>{ago(t - last.at)} AGO</span>
          </>
        ) : (
          <span {...stylex.props(text.labelXs, s.dim)}>no tool call logged yet</span>
        )}
      </div>
      {tools.recent.slice(run ? 0 : 1, 6).map((r) => (
        <div key={r.id} {...stylex.props(text.labelXs, s.toolRow)}>
          <span {...stylex.props(s.soft, s.name)}>
            {r.name}
            {r.empty ? <span {...stylex.props(s.dim)}> · empty</span> : null}
          </span>
          <span {...stylex.props(s.soft, text.num)}>{fmtMs(r.ms)}</span>
          <span {...stylex.props(s.dim, text.num)}>{ago(t - r.at)}</span>
        </div>
      ))}
      {tools.last_turn && (
        <span {...stylex.props(text.labelXs, s.dim)}>
          LAST REQUEST {ago(t - tools.last_turn.at)} AGO{tools.last_turn.open ? ' · ANSWERING' : ''}
        </span>
      )}
    </Card>
  );
}

/** Decode tokens/s across all slots, over the last polls. */
export function RateCard({ history, current }: { history: number[]; current: number | null }) {
  const max = Math.max(60, ...history);
  const W = 90;
  const H = 36;
  const pts = history.map((v, i) => `${((i + W - history.length) / (W - 1)) * W},${H - 2 - (v / max) * (H - 4)}`).join(' ');
  return (
    <Card label="DECODE" right={`${history.length} S`}>
      <p {...stylex.props(text.headlineSm, text.num, s.big)}>
        {current === null ? '—' : current.toFixed(1)}
        <span {...stylex.props(text.labelXs, s.unit)}>TOK/S</span>
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" {...stylex.props(s.spark)} role="img" aria-label="decode tokens per second, last 90 seconds">
        <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1.2" vectorEffect="non-scaling-stroke" {...stylex.props(s.moss)} />
      </svg>
    </Card>
  );
}

export function GpuCard({ g }: { g: Gpu }) {
  const tone: Tone = g.tight ? 'crimson' : 'moss';
  return (
    <Card label={`GPU${g.index} · ${g.name.replace(/^NVIDIA (GeForce )?/, '')}`}>
      <p {...stylex.props(text.headlineSm, text.num, s.big)}>
        {g.util}
        <span {...stylex.props(text.labelXs, s.unit)}>% UTIL</span>
      </p>
      <Meter value={g.util / 100} tone="cyan" label={`GPU${g.index} utilisation ${g.util}%`} />
      <Meter value={g.pct / 100} tone={tone} label={`GPU${g.index} ${g.used_mib} of ${g.total_mib} MiB used`} />
      <span {...stylex.props(text.labelXs, s.soft, text.num)}>
        VRAM {gib(g.used_mib)} / {gib(g.total_mib)} GIB · {n(g.free_mib)} MIB FREE
      </span>
    </Card>
  );
}

export function KvCard({ context }: { context: ContextPool | null | undefined }) {
  const kv = kvSplit(context);
  return (
    <Card label="KV POOL" right={kv ? `${n(kv.pool)} TOK` : undefined}>
      {!kv ? (
        <span {...stylex.props(text.labelXs, s.dim)}>no context budget</span>
      ) : (
        <>
          <SplitBar
            label={`main ${kv.main}, ${kv.helpers} deep thinking of ${kv.helper}, reserve ${kv.reserve}`}
            parts={[
              { value: kv.main, tone: 'moss' },
              ...Array.from({ length: kv.helpers }, () => ({ value: kv.helper, tone: 'cyan' as const })),
              ...(kv.reserve > 0 ? [{ value: kv.reserve, tone: 'hatch' as const }] : []),
            ]}
          />
          <span {...stylex.props(text.labelXs, s.soft, text.num)}>
            MAIN {n(kv.main)} · DEEP THINKING {kv.helpers}×{n(kv.helper)}
          </span>
        </>
      )}
    </Card>
  );
}

export function StrataCard({ strata }: { strata: Strata | null | undefined }) {
  return (
    <Card label="TIER STRATA" right={strata ? windowLabel(strata.window_seconds) : undefined}>
      {!strata ? (
        <span {...stylex.props(text.labelXs, s.dim)}>not reported</span>
      ) : (
        <>
          <div {...stylex.props(s.split3, text.num)}>
            <span {...stylex.props(text.titleMd, s.moss)}>{n(strata.taproot)}</span>
            <span {...stylex.props(text.titleMd, s.cyan)}>{n(strata.branch)}</span>
            <span {...stylex.props(text.titleMd, s.soft)}>{n(strata.shoot)}</span>
          </div>
          <div {...stylex.props(s.split3, text.labelXs, s.dim)}>
            <span>TAPROOT</span>
            <span>BRANCH</span>
            <span>SHOOT</span>
          </div>
          <span {...stylex.props(text.labelXs, s.dim)}>{n(strata.searches)} SEARCHES</span>
        </>
      )}
    </Card>
  );
}

export function QueueCard({ queue }: { queue: JobQueue | null | undefined }) {
  const st = queue?.states ?? {};
  return (
    <Card label="JOB QUEUE" right={queue?.oldest_queued_age != null ? `OLDEST ${ago(queue.oldest_queued_age)}` : undefined}>
      {!queue ? (
        <span {...stylex.props(text.labelXs, s.dim)}>not reported</span>
      ) : (
        <>
          <div {...stylex.props(s.split3, text.num)}>
            <span {...stylex.props(text.titleMd, s.soft)}>{n(st.queued ?? 0)}</span>
            <span {...stylex.props(text.titleMd, s.moss)}>{n(st.running ?? 0)}</span>
            <span {...stylex.props(text.titleMd, (st.errored ?? 0) > 0 ? s.cyan : s.dim)}>{n(st.errored ?? 0)}</span>
          </div>
          <div {...stylex.props(s.split3, text.labelXs, s.dim)}>
            <span>QUEUED</span>
            <span>RUNNING</span>
            <span>ERRORED</span>
          </div>
        </>
      )}
    </Card>
  );
}

export function SeedCard({ seed }: { seed: Seed | null | undefined }) {
  return (
    <Card label="CONCEPT SEED" right={seed ? hex32(seed.u32) : undefined} flashKey={seed?.at ?? null} tone="moss">
      {!seed ? (
        <span {...stylex.props(text.labelXs, s.dim)}>no seed drawn yet</span>
      ) : (
        <>
          <p {...stylex.props(text.titleMd, s.word)}>{seed.word}</p>
          <SeedBits u32={seed.u32} />
        </>
      )}
    </Card>
  );
}

export const telemetry = s;
