// What runs where: the lane view (mcp/lane_view.py, carried as `cards` on /dash/api/vitals and
// /dash/api/vitals/pulse). Types plus the pure readings the panels share, so the words are testable.
//
//   cards   one per GPU: the models llama-swap has loaded on it, each with its own slots
//   jobs    the worker's GPU job lanes (gpu: the main card, gpu_a4000: the A4000)
//   holds   what holds what, each in plain words with its reason and since when
//   swap    the swap in progress and the last swaps
//
// Nothing here is invented: every number is the server's, with the unit it came in.
import type { Slot } from './types';

/** The conversation pinned to a slot (mcp/slots.py occupants()); null when none is, or outside the proxy process. */
export type SlotConv = {
  /** first 8 characters of the conversation's key */
  conv: string;
  busy: boolean;
  /** seconds since its latest activity; null when this process has not seen any (a pin restored at a restart) */
  idle_s: number | null;
  /** the one-conversation hold keeps the card for it (a card's slot 0 only; null elsewhere) */
  held: boolean | null;
  /** seconds of the hold left: 0 once it has ended, null while a request is in flight or the slot has no hold */
  rest_s: number | null;
  hold_s?: number;
};

export type LaneSlot = Slot & { use?: string | null; conv?: SlotConv | null };

export type LaneSlots = {
  ok: boolean;
  slots: LaneSlot[];
  ms?: number;
  error?: string;
  decoding?: number;
  prefilling?: number;
  tps?: number;
};

export type LaneModel = {
  model: string;
  /** llama-swap's state: ready, starting, stopping */
  state: string;
  port: number | null;
  gguf: string | null;
  ttl?: number | null;
  card: 'main' | 'a4000' | string;
  main: boolean;
  /** null: the model has no slots to read (an embedder, an image model) or is not ready */
  slots: LaneSlots | null;
  /** requests the proxy holds in flight on it (max_mode leases) */
  inflight: number;
  inflight_since: number | null;
  last_use: number | null;
};

export type LaneExpected = { model: string; for: string; why: string };

export type LaneCard = {
  key: 'main' | 'a4000' | string;
  index: number | null;
  name: string | null;
  uuid: string | null;
  util?: number | null;
  used_mib?: number | null;
  total_mib?: number | null;
  free_mib?: number | null;
  models: LaneModel[];
  expected: LaneExpected[];
};

export type JobRun = { id: string; queue: string; dataset: string | null; stage: string | null; started: number | null; heartbeat: number | null };

export type JobLane = {
  lane: 'gpu' | 'gpu_a4000' | string;
  card: string;
  label: string;
  paused: { lane: string; by: string | null; why: string | null; since: number | null; until: number | null } | null;
  running: JobRun[];
  queued: number | null;
  error: string | null;
};

export type Hold = {
  kind: 'job_lane_paused' | 'model_lease' | 'swap' | 'conversation' | 'gpu_room_lease' | 'model_state' | 'unknown' | string;
  card: string;
  what: string;
  by: string | null;
  why: string | null;
  since: number | null;
  until: number | null;
  since_basis?: string;
};

export type SwapNow = { to: string; from: string[]; phase: 'waiting' | 'loading' | string; since: number };
export type SwapDone = { at: number; from: string[]; to: string | null; load_s: number | null; ok: boolean; how: string | null; left_loaded: string[] };

export type LaneView =
  | {
      at: number;
      in_proxy: boolean;
      running_read: boolean;
      cards: LaneCard[];
      jobs: JobLane[];
      holds: Hold[];
      swap: { now: SwapNow | null; last: SwapDone[] };
      error?: undefined;
    }
  | { error: string };

export function laneView(x: LaneView | null | undefined): Extract<LaneView, { cards: LaneCard[] }> | null {
  return x && !('error' in x && x.error) && Array.isArray((x as { cards?: unknown }).cards) ? (x as Extract<LaneView, { cards: LaneCard[] }>) : null;
}

const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

/** "12 s", "3 min", "2 h", "5 d": one coarse unit, for an age or a remaining time. */
export function span(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 90) return `${s} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  if (s < 172800) return `${Math.round(s / 3600)} h`;
  return `${Math.round(s / 86400)} d`;
}

/** "since 41 s ago" / "until in 5 d" for a hold's two ends, in the viewer's time. */
export function holdTimes(h: { since: number | null; until: number | null; since_basis?: string }, now: number): string {
  const out: string[] = [];
  if (finite(h.since)) out.push(`since ${span(now - h.since)} ago${h.since_basis ? ` (${h.since_basis})` : ''}`);
  if (finite(h.until)) out.push(h.until > now ? `until in ${span(h.until - now)}` : 'until now');
  return out.join(' · ');
}

/** The one-line status of a slot, in the units the server sent. */
export function slotLine(x: Pick<Slot, 'state' | 'ctx' | 'tps' | 'pps'>): string {
  if (x.state === 'idle') return 'IDLE';
  const rate = x.state === 'decode' ? x.tps : x.pps;
  const r = rate ? `${rate.toFixed(rate < 100 ? 1 : 0)} ${x.state === 'decode' ? 'TOK/S' : 'PREFILL TOK/S'}` : '…';
  return `${x.state.toUpperCase()} · ${x.ctx.toLocaleString('en-US')} CTX · ${r}`;
}

/** A model's slots in one phrase: "S0 DECODE · 26,000 CTX · 12.3 TOK/S", or why there are none to show. */
export function modelStatus(m: LaneModel): string {
  if (m.state !== 'ready') return m.state.toUpperCase();
  if (!m.slots) return 'READY · no slots (not a chat server)';
  if (!m.slots.ok) return `READY · /slots not read: ${m.slots.error ?? 'no reason given'}`;
  if (!m.slots.slots.length) return 'READY · no slots reported';
  return m.slots.slots.map((s) => `S${s.id} ${slotLine(s)}`).join(' | ');
}

/** The model on a card: the ready one first. */
export function cardModel(c: LaneCard | undefined): LaneModel | null {
  const ms = (c?.models ?? []).filter((m) => m.slots !== undefined);
  return ms.find((m) => m.state === 'ready' && m.main) ?? ms.find((m) => m.main) ?? null;
}

export type KvLive = { part: 'main' | 'child' | 'second'; tone: 'moss' | 'cyan' | 'muted'; text: string };

/**
 * What runs in each part of the KV bargraph, from the lane view: MAIN is the conversation slot of the model on
 * the main card, CHILD the jjava lane when the card has one (`-np 2`), SECOND CONVERSATION the other card's
 * conversation slot (bonsai-a4000 slot 0). `noLane`: the card runs one conversation alone (-np 1: the pool has no
 * child), jjava and side calls are on bonsai-a4000 slot 1.
 */
export function kvLive(view: ReturnType<typeof laneView>, noLane: boolean): KvLive[] {
  const out: KvLive[] = [];
  const main = view?.cards.find((c) => c.key === 'main');
  const a4 = view?.cards.find((c) => c.key === 'a4000');
  const mm = cardModel(main);
  const tone = (s: LaneSlot | undefined): KvLive['tone'] => (!s || s.state === 'idle' ? 'muted' : s.state === 'prefill' ? 'cyan' : 'moss');
  if (!view) return out;
  if (!mm) {
    out.push({ part: 'main', tone: 'muted', text: 'no main model is loaded' });
  } else {
    const conv = mm.slots?.slots.find((s) => s.role !== 'child');
    out.push({ part: 'main', tone: tone(conv), text: `${mm.model} · ${mm.state !== 'ready' ? mm.state.toUpperCase() : conv ? `S${conv.id} ${slotLine(conv)}` : 'no slot read'}${mm.inflight ? ` · ${mm.inflight} in flight` : ''}` });
  }
  const bonsaiA4 = a4?.models.find((m) => m.model === 'bonsai-a4000');
  const lane = bonsaiA4?.slots?.slots.find((s) => s.use?.startsWith('jjava'));
  if (noLane) {
    out.push({
      part: 'child',
      tone: tone(lane),
      text: `no lane on this card (${mm ? mm.model : 'the model'} runs alone, -np 1) · jjava and side calls: ${
        bonsaiA4 ? (lane ? `bonsai-a4000 S${lane.id} ${slotLine(lane)}` : `bonsai-a4000 ${bonsaiA4.state}`) : 'bonsai-a4000 is not loaded'
      }`,
    });
  } else {
    const l = mm?.slots?.slots.find((s) => s.role === 'child');
    out.push({ part: 'child', tone: tone(l), text: l ? `${mm?.model} S${l.id} ${slotLine(l)}` : mm ? `${mm.model}: no lane slot read` : 'no main model is loaded' });
  }
  const second = bonsaiA4?.slots?.slots.find((s) => s.use?.startsWith('second conversation'));
  out.push({
    part: 'second',
    tone: tone(second),
    text: bonsaiA4
      ? second
        ? `bonsai-a4000 S${second.id} ${slotLine(second)}${second.use ? ` · ${second.use.replace('second conversation (the other card)', 'the other card')}` : ''}`
        : `bonsai-a4000 ${bonsaiA4.state}: no slot read`
      : 'bonsai-a4000 is not loaded (llama-swap loads it on the first request for it)',
  });
  return out;
}


/** What a slot is for, in a few words (the server's `use`, else its role). */
export function slotName(x: Pick<LaneSlot, 'use' | 'role' | 'primary'>): string {
  const u = x.use ?? '';
  if (u.startsWith('jjava lane') || x.role === 'child') return 'JJAVA LANE';
  if (u.startsWith('jjava')) return 'JJAVA · SIDE CALLS';
  if (u.startsWith('second conversation')) return 'SECOND CONVERSATION';
  if (u.startsWith('image reads')) return 'IMAGE READS';
  if (x.primary || u.includes('primary')) return 'PRIMARY CONVERSATION';
  return 'CONVERSATION';
}

/**
 * The words under a slot's bar, each part only when the server said it: what the slot is for, the conversation
 * in it, how long it has been idle and what is left of its hold, the rate while it works. A conversation slot
 * with no cells and no conversation is EMPTY.
 */
export function slotDetail(x: LaneSlot): string[] {
  const out = [slotName(x)];
  const c = x.conv ?? null;
  if (c) out.push(`CONV ${c.conv}`);
  else if (x.role !== 'child' && !x.use?.startsWith('jjava') && x.ctx === 0 && x.state === 'idle') out.push('EMPTY');
  if (x.state === 'idle') {
    if (c && c.idle_s !== null) out.push(`IDLE ${span(c.idle_s)}`);
    if (c && c.held === true && c.rest_s !== null) out.push(`HOLD ${c.rest_s} S LEFT`);
    if (c && c.held === false) out.push(`HOLD OVER${c.hold_s ? ` (${c.hold_s} S)` : ''} · FREE TO TAKE`);
  } else {
    const rate = x.state === 'decode' ? x.tps : x.pps;
    out.push(x.state === 'decode' ? 'DECODING' : 'PREFILLING');
    out.push(rate ? `${rate.toFixed(rate < 100 ? 1 : 0)} ${x.state === 'decode' ? 'TOK/S' : 'PREFILL TOK/S'}` : 'RATE …');
  }
  return out;
}
