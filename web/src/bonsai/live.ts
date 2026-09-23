// Live state -> motion (the tokonoma breathing with the model).
//
// The genome and the spring-driven params (mapping.ts) say what the tree IS.
// This module says what it is DOING this second, from /dash/api/vitals/pulse
// (mcp/vitals.py pulse(), polled ~1 s) or, on a server without the pulse,
// from the 5 s vitals snapshot:
//
//   activity   decode rate across llama-server slots (else GPU0 utilisation)
//   busy       slots in prefill or decode: main plus deep-thinking helpers
//   wind       sway amplitude: calm when idle, rising with busy slots
//   gust       gust strength: only while more than nothing is running
//   glow       ground and ring brightness, following activity
//
// and which EVENTS happened between two polls: a request arrived, a tool was
// called, a concept seed was drawn. Events fire ring pulses; the first poll
// only sets the baseline, so opening the page is not a burst of fake events.
//
// Everything except LiveBus is pure. LiveBus integrates time and eases toward
// the targets so motion is continuous between polls.
import type { Pulse, Slots, Vitals } from '../api/types';

export type LiveTargets = {
  activity: number;
  busy: number;
  wind: number;
  gust: number;
  glow: number;
  /** false: no signal at all, so the scene shows the idle breath */
  measured: boolean;
};

/**
 * Decode tokens/s read as full activity. One slot decoded at 42 tok/s on the
 * 5060 Ti (pulse sample, 2026-09-22, n=1 run of ~3 reads): full-scale is set
 * a little above a single stream, so two decoding slots saturate it.
 */
export const TPS_FULL = 60;

export const IDLE: LiveTargets = { activity: 0, busy: 0, wind: 0.08, gust: 0, glow: 0.12, measured: false };

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);

export function slotsOf(p: Pulse | null | undefined, v: Vitals | null | undefined): Slots | null {
  const s = p?.slots ?? v?.slots;
  return s && s.ok && Array.isArray(s.slots) ? s : null;
}

export function liveTargets(p: Pulse | null | undefined, v: Vitals | null | undefined): LiveTargets {
  const slots = slotsOf(p, v);
  const gpus = p?.gpus ?? v?.gpus;
  const g0 = Array.isArray(gpus) ? gpus.find((g) => g && g.index === 0) : undefined;
  const util = g0 && finite(g0.util) ? clamp01(g0.util / 100) : null;
  if (!slots && util === null) return IDLE;
  let activity: number;
  let busy: number;
  if (slots) {
    busy = slots.slots.filter((s) => s.state !== 'idle').length;
    const tps = slots.slots.reduce((a, s) => a + (finite(s.tps) ? s.tps : 0), 0);
    const prefill = slots.slots.some((s) => s.state === 'prefill');
    // A slot can be busy between two reads before a rate exists: it still
    // counts as work, just not as full work.
    activity = clamp01(Math.max(tps / TPS_FULL, prefill ? 0.55 : 0, busy > 0 ? 0.3 : 0));
  } else {
    busy = util! > 0.25 ? 1 : 0;
    activity = util!;
  }
  return {
    activity,
    busy,
    wind: busy === 0 ? IDLE.wind : clamp01(0.35 + 0.2 * (busy - 1) + 0.3 * activity),
    gust: busy === 0 ? 0 : clamp01(0.3 * busy),
    glow: IDLE.glow + (1 - IDLE.glow) * activity,
    measured: true,
  };
}

// ------------------------------------------------------------------ events

export type EventKind = 'request' | 'tool' | 'seed';
export type Marks = { turn: number | null; tool: number | null; seed: number | null; busy: number };

export function marksOf(p: Pulse | null | undefined, v?: Vitals | null): Marks {
  const tools = p?.tools ?? v?.tools ?? null;
  const seed = p?.seed ?? v?.seed ?? null;
  const slots = slotsOf(p, v);
  return {
    turn: tools?.last_turn?.id ?? null,
    tool: tools?.running?.id ?? tools?.last?.id ?? null,
    seed: seed && finite(seed.at) ? seed.at : null,
    busy: slots ? slots.slots.filter((s) => s.state !== 'idle').length : 0,
  };
}

/** What happened between two polls. The first poll is a baseline: none. */
export function newEvents(prev: Marks | null, next: Marks): EventKind[] {
  if (!prev) return [];
  const out: EventKind[] = [];
  // A request is a new corpus turn, or a slot going from idle to busy (the
  // benchmark and internal generation do not log turns, but they take slots).
  if ((next.turn !== null && next.turn !== prev.turn) || next.busy > prev.busy) out.push('request');
  if (next.tool !== null && next.tool !== prev.tool) out.push('tool');
  if (next.seed !== null && next.seed !== prev.seed) out.push('seed');
  return out;
}

// -------------------------------------------------------------------- wind

export type WindState = { amp: number; phase: number; gust: number; gphase: number };

/** World units of sway at the crown for amp = 1. */
export const WIND_REACH = 0.085;
/** Below this height nothing moves: the nebari and its weld stay put. */
export const WIND_FLOOR = 0.3;
const WIND_DIR = [0.83, 0, 0.56] as const; // from behind-left, across the key light

export function gustEnvelope(gphase: number): number {
  const s = Math.max(0, Math.sin(gphase));
  return s * s * s * (0.55 + 0.45 * Math.sin(gphase * 0.37 + 1.1));
}

/**
 * Sway offset for a point at rest position (x, y, z), written to out[o..o+2].
 * One continuous field for bark, foliage and tip caps alike, so pads stay on
 * their branches. Zero at and below WIND_FLOOR.
 */
export function windOffset(x: number, y: number, z: number, w: WindState, out: Float32Array | number[], o = 0) {
  const h = clamp01((y - WIND_FLOOR) / 2.4);
  const reach = h * h * (1 + 0.3 * clamp01((Math.hypot(x, z) - 0.6) / 1.5));
  if (reach === 0 || w.amp === 0) {
    out[o] = 0;
    out[o + 1] = 0;
    out[o + 2] = 0;
    return;
  }
  const wave = 0.65 * Math.sin(w.phase + x * 0.9 + z * 0.6) + 0.2 * Math.sin(w.phase * 2.3 + y * 1.7 + x);
  const d = WIND_REACH * w.amp * reach * (wave + 1.5 * w.gust * gustEnvelope(w.gphase));
  const cross = WIND_REACH * w.amp * reach * 0.25 * Math.sin(w.phase * 1.7 + z * 2.1);
  out[o] = d * WIND_DIR[0] - cross * WIND_DIR[2];
  out[o + 1] = -0.18 * Math.abs(d);
  out[o + 2] = d * WIND_DIR[2] + cross * WIND_DIR[0];
}

// --------------------------------------------------------------------- bus

/** Exponential approach: frame-rate independent easing toward a target. */
export const approach = (cur: number, target: number, dt: number, tau: number) =>
  cur + (target - cur) * (1 - Math.exp(-dt / Math.max(tau, 1e-3)));

/** How long a ring pulse lasts, seconds. */
export const PULSE_S = 3.2;

/**
 * Owned outside React (like TreeRig): the page pushes each poll in, the
 * scene ticks it every frame. Methods only, so no component assigns into a
 * hook value.
 */
export class LiveBus {
  target: LiveTargets = IDLE;
  activity = 0;
  wind = IDLE.wind;
  gust = 0;
  glow = IDLE.glow;
  /** bus clock, seconds */
  t = 0;
  phase = 0;
  gphase = 0;
  /** bus-clock time each event kind last fired */
  readonly fired: Record<EventKind, number> = { request: -100, tool: -100, seed: -100 };
  private marks: Marks | null = null;
  private listeners = new Set<() => void>();

  push(p: Pulse | null | undefined, v: Vitals | null | undefined) {
    this.target = liveTargets(p, v);
    if (p || v) {
      const m = marksOf(p, v);
      for (const k of newEvents(this.marks, m)) this.fired[k] = this.t;
      this.marks = m;
    }
    this.listeners.forEach((f) => f());
  }

  listen(f: () => void) {
    this.listeners.add(f);
    return () => void this.listeners.delete(f);
  }

  tick(dt: number, still: boolean) {
    const h = Math.min(dt, 0.1);
    this.t += h;
    this.activity = approach(this.activity, this.target.activity, h, still ? 2.5 : 0.8);
    this.glow = approach(this.glow, this.target.glow, h, still ? 3 : 1.2);
    this.wind = approach(this.wind, still ? 0 : this.target.wind, h, 2.5);
    this.gust = approach(this.gust, still ? 0 : this.target.gust, h, 2.0);
    this.phase += h * (0.7 + 1.6 * this.wind);
    this.gphase += h * (0.22 + 0.8 * this.gust);
  }

  /** Ground and ring brightness this frame: a slow breath around the eased glow. */
  glowNow(still: boolean): number {
    return still ? this.glow : this.glow * (0.8 + 0.2 * Math.sin((this.t * Math.PI * 2) / 6.5));
  }

  windState(still: boolean): WindState {
    return { amp: still ? 0 : this.wind, phase: this.phase, gust: this.gust, gphase: this.gphase };
  }

  /** Something is visibly changing: a pulse in flight or a value still easing. */
  animating(): boolean {
    const last = Math.max(this.fired.request, this.fired.tool, this.fired.seed);
    return (
      this.t - last < PULSE_S ||
      Math.abs(this.glow - this.target.glow) > 2e-3 ||
      Math.abs(this.activity - this.target.activity) > 2e-3
    );
  }

  /** Busy enough to deserve every frame (otherwise the scene idles at a lower rate). */
  lively(): boolean {
    return this.target.busy > 0 || this.animating() || this.wind > 0.2;
  }
}
