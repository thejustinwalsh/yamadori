// The live readouts around the tokonoma: what the model and its tools are
// doing this second, from /dash/api/vitals/pulse (mcp/vitals.py pulse()).
// Counters that grow between polls (context held while decoding, a running
// tool's elapsed time) are interpolated at their measured rate, so nothing
// jumps once a second; interpolation stops a few seconds past the last poll
// rather than inventing progress that was never measured.
import * as stylex from '@stylexjs/stylex';
import { useEffect, useRef } from 'react';
import { kvNames, kvSplit } from '../../api/kv';
import { cardModel, laneView, slotDetail, type LaneCard, type LaneModel, type LaneSlot, type LaneView } from '../../api/lanes';
import type { ContextPool, Gpu, JobQueue, Lanes, Seed, Slot, Slots, Strata, ToolActivity } from '../../api/types';
import { ago, gib, n } from '../../format';
import { colors, space } from '../../tokens/tokens.stylex';
import { Label, Meter, SplitBar, type Tone } from '../../ui/primitives';
import { KvLive } from '../../ui/KvLive';
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
  // The first, second and fourth columns are as wide as their text (max-content), never a fixed width a label can
  // outgrow (operator, 2026-10-05: "slot name overflow"); the bar takes the rest.
  slotRow: {
    display: 'grid',
    gridTemplateColumns: 'max-content 9ch minmax(24px, 1fr) max-content',
    alignItems: 'center',
    columnGap: space.spaceXs,
  },
  slotBlock: { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 },
  modelBlock: { display: 'flex', flexDirection: 'column', gap: space.spaceXs, minWidth: 0 },
  modelHead: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: space.spaceXs,
    minWidth: 0,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: `color-mix(in srgb, ${colors.outlineVariant} 35%, transparent)`,
    paddingTop: space.spaceXs,
  },
  nowrap: { whiteSpace: 'nowrap' },
  stateCol: { minWidth: '9ch' },
  cells: { textAlign: 'right' },
  detail: { overflowWrap: 'anywhere', paddingInlineStart: space.spaceXs },
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

export function Card({ label, right, rightTitle, wide, flashKey, tone, children }: {
  label: string;
  right?: React.ReactNode;
  rightTitle?: string;
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
        {right !== undefined && (
          <span title={rightTitle} {...stylex.props(text.labelXs, s.dim, s.nowrap)}>
            {right}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

const STATE_TONE: Record<Slot['state'], 'moss' | 'cyan' | 'dim'> = { decode: 'moss', prefill: 'cyan', idle: 'dim' };

/**
 * One slot: its bar and the cells it holds against its window, then (wrapping freely) what it is for, which
 * conversation is in it, how long it has been idle, what is left of its hold and its rate.
 * Labels never wrap: the first, second and last columns are as wide as their text.
 */
function SlotRow({ x, window, since }: { x: LaneSlot; window: number; since: number }) {
  const tone = STATE_TONE[x.state];
  const rate = x.state === 'decode' ? x.tps : x.state === 'prefill' ? x.pps : 0;
  const held = x.ctx > 0;
  return (
    <div {...stylex.props(s.slotBlock)}>
      <div {...stylex.props(text.labelXs, s.slotRow)}>
        <span {...stylex.props(s.dim, s.nowrap)}>S{x.id}</span>
        <span {...stylex.props(s[tone], s.nowrap, s.stateCol)}>{x.state.toUpperCase()}</span>
        <Meter
          value={x.state === 'idle' && !held ? 0 : window > 0 ? Math.min(1, x.ctx / window) : null}
          tone={x.state === 'prefill' ? 'cyan' : 'moss'}
          label={`slot ${x.id} holds ${x.ctx} of ${window} tokens`}
        />
        <span
          {...stylex.props(s.soft, s.nowrap, s.cells)}
          title={x.state === 'idle' ? "tokens of the slot's last request, which llama-server still holds for it (/slots)" : 'tokens the slot holds now (prompt + decoded)'}
        >
          {x.state === 'idle' ? n(x.ctx) : <Ticker base={x.ctx} rate={rate} since={since} format={fmtInt} />} / {n(window)}
        </span>
      </div>
      <span {...stylex.props(text.labelXs, s.dim, s.detail)}>{slotDetail(x).join(' · ')}</span>
    </div>
  );
}

/** A slot's window: the conversation window the tier table enforces on the main card, else the slot's own n_ctx. */
function windowOf(x: LaneSlot, m: LaneModel, kv: ReturnType<typeof kvSplit>): number {
  if (m.main && kv && kv.model === m.model) return x.role === 'child' ? kv.helper : kv.main;
  return x.n_ctx;
}

/** One model's slots under a heading: the card, the model, what llama-swap says it is. */
function ModelSlots({ card, m, kv, since }: { card: LaneCard; m: LaneModel; kv: ReturnType<typeof kvSplit>; since: number }) {
  const where = card.key === 'main' ? '5060 TI' : card.key === 'a4000' ? 'A4000' : (card.name ?? card.key);
  return (
    <div {...stylex.props(s.modelBlock)}>
      <div {...stylex.props(text.labelXs, s.modelHead)}>
        <span {...stylex.props(s.soft, s.nowrap)}>
          {where} · {m.model.toUpperCase()}
        </span>
        <span {...stylex.props(m.state === 'ready' ? s.moss : s.cyan, s.nowrap)}>{m.state.toUpperCase()}</span>
      </div>
      {m.state !== 'ready' ? (
        <span {...stylex.props(text.labelXs, s.dim)}>llama-swap says {m.state}; its slots are read once it is ready</span>
      ) : !m.slots ? (
        <span {...stylex.props(text.labelXs, s.dim)}>no slots (not a chat server)</span>
      ) : !m.slots.ok ? (
        <span {...stylex.props(text.labelXs, s.dim)}>/slots not read · {m.slots.error ?? 'no reason given'}</span>
      ) : (
        m.slots.slots.map((x) => <SlotRow key={x.id} x={x} since={since} window={windowOf(x, m, kv)} />)
      )}
    </div>
  );
}

export function SlotsCard({ slots, lanes, context, since, view }: {
  slots: Slots | null | undefined;
  lanes: Lanes | null | undefined;
  context: ContextPool | null | undefined;
  since: number;
  /** what runs where (mcp/lane_view.py); absent on a server that predates it: the main card's slots alone */
  view?: LaneView | null;
}) {
  const kv = kvSplit(context);
  const lv = laneView(view);
  // every loaded model that has slots to show, or is on its way to having them
  const llama = (lv?.cards ?? []).flatMap((c) => c.models.filter((m) => m.slots !== null || m.state !== 'ready').map((m) => ({ c, m })));
  const busy = lv
    ? llama.reduce((a, { m }) => a + (m.slots?.slots.filter((x) => x.state !== 'idle').length ?? 0), 0)
    : slots?.ok
      ? slots.slots.filter((x) => x.state !== 'idle').length
      : null;
  // The count in the header is ADMISSION's requests in flight over its main lanes (mcp/admission.py), not slots.
  const laneText = lanes && lanes.in_proxy ? `REQUESTS ${lanes.main}/${lanes.main_lanes ?? '—'}` : busy === null ? undefined : `${busy} BUSY`;
  const model = lv ? cardModel(lv.cards.find((c) => c.key === 'main'))?.model : slots?.model;
  return (
    <Card
      label={`SLOTS · ${model ? model.toUpperCase() : 'LLAMA-SERVER'}`}
      right={laneText}
      rightTitle={lanes && lanes.in_proxy ? 'requests the proxy is serving now / the requests it admits at once (admission). Slots are listed below.' : undefined}
      wide
      flashKey={busy ? `b${busy}` : null}
      tone="moss"
    >
      {lv ? (
        <>
          {llama.length === 0 && <span {...stylex.props(text.labelXs, s.dim)}>no llama-server model is loaded (llama-swap loads one on the next request)</span>}
          {llama.map(({ c, m }) => (
            <ModelSlots key={`${c.key}/${m.model}`} card={c} m={m} kv={kv} since={since} />
          ))}
          {lv.cards.flatMap((c) =>
            c.expected.map((e) => (
              <span key={e.model} {...stylex.props(text.labelXs, s.dim, s.detail)} title={e.why}>
                A4000 · {e.model.toUpperCase()} · NOT LOADED · {e.for}
              </span>
            )),
          )}
        </>
      ) : !slots ? (
        <span {...stylex.props(text.labelXs, s.dim)}>slot state not reported by this server</span>
      ) : !slots.ok ? (
        <span {...stylex.props(text.labelXs, s.dim)}>
          {slots.off_card ? '' : '/slots not answering · '}
          {slots.error ?? 'no reason given'}
        </span>
      ) : (
        slots.slots.map((x) => <SlotRow key={x.id} x={x} since={since} window={kv ? (x.role === 'child' ? kv.helper : kv.main) : x.n_ctx} />)
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
        VRAM {gib(g.used_mib)} / {gib(g.total_mib)} GIB · {n(g.free_mib)} MIB FREE{g.floor_mib != null ? ` / ${n(g.floor_mib)} FLOOR` : ''}
      </span>
    </Card>
  );
}

export function KvCard({ context, view }: { context: ContextPool | null | undefined; view?: LaneView | null }) {
  const kv = kvSplit(context);
  const nm = kv ? kvNames(kv) : null;
  return (
    <Card label="KV POOL" right={kv ? `${n(kv.pool)} TOK` : undefined}>
      {!kv || !nm ? (
        <span {...stylex.props(text.labelXs, s.dim)}>no context budget</span>
      ) : (
        <>
          <SplitBar
            label={`${nm.main.toLowerCase()} ${kv.main}, ${nm.helper.toLowerCase()} ${kv.helper}, ${nm.reserve.toLowerCase()} ${kv.reserve}`}
            parts={[
              { value: kv.main, tone: 'moss' },
              ...Array.from({ length: kv.helpers }, () => ({ value: kv.helper, tone: 'cyan' as const })),
              ...(kv.reserve > 0 ? [{ value: kv.reserve, tone: 'hatch' as const }] : []),
            ]}
          />
          <span {...stylex.props(text.labelXs, s.soft, text.num)}>
            {kv.layout === 'cap' ? `MAIN ${n(kv.main)} · CHILD ${n(kv.helper)} · 2ND ${n(kv.reserve)}` : `MAIN ${n(kv.main)} · ${nm.helperRole.toUpperCase()} ${kv.helpers}×${n(kv.helper)}`}
          </span>
          <KvLive view={view} noLane={kv.noLane} stacked />
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
