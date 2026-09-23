// Phase 2: the pure mapping, state -> params (design/BONSAI-VIZ.md §2, §4).
//
// Two pure functions and a registry:
//
//   treeState(vitals, datasets)   what was measured, normalised, or null
//   treeParams(state, skeleton)   one parameter vector per branch + scene
//   CHANNELS                      every visual channel and the key it reads
//
// No clock, no RNG, no I/O, no mutation of inputs. Tests enforce all four.
//
// THE RULE (§1): every channel is wired to a signal we already collect, or
// it is INERT -- a fixed value that looks inert (matte, unlit, unanimated),
// never noise. A channel with no source is listed as such in CHANNELS and
// shown as such in the tokonoma caption.
import type { DatasetsOverview, Vitals } from '../api/types';
import { LIMB, type Skeleton } from './skeleton';

// ---------------------------------------------------------------- channels

export type Channel = {
  name: string;
  /** Dotted path into the /dash/api payload, or null: no source exists yet. */
  source: string | null;
  /** Which payload `source` is read from. 'pulse' is /dash/api/vitals/pulse (live.ts). */
  api: 'vitals' | 'datasets' | 'pulse' | null;
  reads: string;
};

/**
 * Deviations from BONSAI-VIZ §2, stated rather than hidden:
 *  - "Bioluminescent arcs <- token throughput": tok/s is not in any payload.
 *    The arcs read GPU0 utilisation instead -- the card the model runs on is
 *    computing or it is not. Real, and named as what it is.
 *  - "Foliage density <- hints above the floor": no payload carries hint
 *    counts, so foliage is INERT: matte moss, unlit.
 */
export const CHANNELS: Channel[] = [
  { name: 'genome', source: 'seed.u32', api: 'vitals', reads: 'the concept seed word, as the PRNG seed of every limb' },
  { name: 'limb.main.girth', source: 'context.main', api: 'vitals', reads: 'main KV budget share of the pool' },
  { name: 'limb.thinking.girth', source: 'context.helper', api: 'vitals', reads: 'deep-thinking KV budget share of the pool' },
  { name: 'reserve.sapwood', source: 'context.reserve', api: 'vitals', reads: 'unclaimed reserve share (trunk girth)' },
  { name: 'arcs.activity', source: 'gpus.util', api: 'vitals', reads: 'GPU0 utilisation: the model card at work' },
  { name: 'slab.haze', source: 'gpus.pct', api: 'vitals', reads: 'highest GPU memory use' },
  { name: 'slab.alarm', source: 'gpus.tight', api: 'vitals', reads: 'any GPU below the free-memory floor' },
  { name: 'traces', source: 'endpoints.ok', api: 'vitals', reads: 'one ground trace per probed endpoint' },
  { name: 'traces.ports', source: 'listeners.conflict', api: 'vitals', reads: 'one ground trace per watched port; crimson on a conflict' },
  { name: 'caps.crimson', source: 'warnings', api: 'vitals', reads: 'warnings, endpoints down, port conflicts' },
  { name: 'shari.errored', source: 'queue.states.errored', api: 'datasets', reads: 'errored jobs, bleached to deadwood' },
  { name: 'fanout', source: null, api: null, reads: 'fan-out arity, chosen and culled samples' },
  { name: 'foliage', source: null, api: null, reads: 'hints above the similarity floor' },
  { name: 'nebari.spread', source: null, api: null, reads: 'corpus breadth (chunks, defs, roots)' },
  { name: 'sway', source: 'slots.slots.state', api: 'pulse', reads: 'busy llama-server slots (main and deep thinking): wind and gusts' },
  { name: 'ground.glow', source: 'slots.slots.tps', api: 'pulse', reads: 'decode tokens/s across slots: slab rings and traces brighten' },
  { name: 'rings.request', source: 'tools.last_turn.id', api: 'pulse', reads: 'a request arrived (or a slot woke): a green pulse' },
  { name: 'rings.tool', source: 'tools.last.id', api: 'pulse', reads: 'a tool was called: a cyan pulse' },
  { name: 'rings.seed', source: 'seed.at', api: 'pulse', reads: 'a concept seed was drawn: a pale pulse' },
  { name: 'moss', source: null, api: null, reads: 'index freshness' },
];

// ------------------------------------------------------------------- state

/** Everything normalised to plain numbers; null means "not measured". */
export type TreeState = {
  seed: number | null;
  mainShare: number | null;
  thinkingShare: number | null;
  reserveShare: number | null;
  activity: number | null;
  memPressure: number | null;
  anyTight: boolean | null;
  endpoints: { name: string; ok: boolean }[] | null;
  alarms: number | null;
  errored: number | null;
  fanout: { arity: number; chosen: number } | null;
};

const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
export const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function treeState(vitals: Vitals | null | undefined, datasets?: DatasetsOverview | null): TreeState {
  const v = (vitals ?? {}) as Partial<Vitals>;
  const ctx = v.context && 'pool' in v.context && finite(v.context.pool) && v.context.pool > 0 ? v.context : null;
  const gpus = Array.isArray(v.gpus) ? v.gpus.filter((g) => g && finite(g.pct)) : [];
  const eps = Array.isArray(v.endpoints) ? v.endpoints : null;
  const lis = Array.isArray(v.listeners) ? v.listeners : [];
  const warnings = Array.isArray(v.warnings) ? v.warnings.length : null;
  const share = (x: unknown) => (ctx && finite(x) ? clamp01(x / ctx.pool) : null);
  const gpu0 = gpus.find((g) => g.index === 0) ?? null;
  const errored = datasets?.queue?.states?.errored;
  return {
    seed: v.seed && finite(v.seed.u32) ? v.seed.u32 >>> 0 : null,
    mainShare: share(ctx?.main),
    thinkingShare: share(ctx?.helper),
    reserveShare: share(ctx?.reserve),
    activity: gpu0 && finite(gpu0.util) ? clamp01(gpu0.util / 100) : null,
    memPressure: gpus.length ? clamp01(Math.max(...gpus.map((g) => g.pct)) / 100) : null,
    anyTight: gpus.length ? gpus.some((g) => g.tight === true) : null,
    endpoints: eps
      ? [
          ...eps.map((e) => ({ name: String(e?.name ?? '?'), ok: e?.ok === true })),
          ...lis.map((l) => ({ name: `:${l?.port} ${l?.role}`, ok: l?.conflict !== true })),
        ]
      : null,
    alarms:
      warnings === null && !eps
        ? null
        : (warnings ?? 0) + (eps ? eps.filter((e) => e?.ok !== true).length : 0) + lis.filter((l) => l?.conflict).length,
    errored: finite(errored) ? Math.max(0, Math.floor(errored)) : null,
    // No payload carries fan-out yet. INERT: a single trunk, no fork.
    fanout: null,
  };
}

// ------------------------------------------------------------------ params

/** The per-branch parameter vector. Every field is in [0, 1] except girth. */
export type BranchParams = {
  /** fraction of the genome length grown */
  growth: number;
  /** multiplier on the genome radius, [GIRTH_MIN, GIRTH_MAX] */
  girth: number;
  /** foliage pad presence */
  foliage: number;
  /** foliage is lit by a real signal (1) or inert matte (0) */
  foliageLive: number;
  /** emissive arc intensity */
  emissive: number;
  /** deadwood: bone albedo, high roughness */
  bleach: number;
  /** crimson cap on the tip */
  cap: number;
  /** channel has no source: draw desaturated */
  inert: number;
};

export type SceneParams = {
  haze: number;
  alarm: number;
  activity: number;
  traces: { name: string; ok: boolean }[];
};

export type TreeParams = { branches: BranchParams[]; scene: SceneParams };

export const GIRTH_MIN = 0.35;
export const GIRTH_MAX = 1.6;

/** Inert values: fixed, matte, unanimated. What a dead channel looks like. */
export const INERT = {
  girth: 0.7,
  growth: 1,
  foliage: 1,
  activity: 0,
  haze: 0,
} as const;

const girthOf = (share: number | null, lo = 0.55, span = 1.0) =>
  share === null ? INERT.girth : Math.min(GIRTH_MAX, Math.max(GIRTH_MIN, lo + span * share));

export function treeParams(state: TreeState, sk: Skeleton): TreeParams {
  const fan = state.fanout;
  const arity = fan ? Math.max(1, Math.min(3, Math.floor(fan.arity))) : 1;
  const chosen = fan ? Math.max(0, Math.min(arity - 1, Math.floor(fan.chosen))) : 0;
  const activity = state.activity ?? INERT.activity;
  const alarms = state.alarms ?? 0;
  const errored = state.errored ?? 0;

  // Culled-by-error deadwood: the first `errored` twigs, in genome order, of
  // the main limb. Deterministic, capped, and never removed (§4: death is
  // retraction and bleaching, not deletion).
  const twigs = sk.branches.filter((b) => b.kind === 'twig' && b.limb === LIMB.main).map((b) => b.id);
  const shari = new Set(twigs.slice(0, Math.min(errored, twigs.length)));
  // Crimson caps: one twig tip per alarm, across all limbs, capped.
  const allTips = sk.branches.filter((b) => b.kind === 'twig').map((b) => b.id);
  const capped = new Set(allTips.filter((_, i) => i % 5 === 0).slice(0, Math.min(alarms, 12)));

  const branches = sk.branches.map((b): BranchParams => {
    let growth = 1;
    let girth = 1;
    let inert = 0;
    let bleach = 0;

    // Which fan-out sample this limb slot is. main = sample 0.
    const sample = b.limb === LIMB.main ? 0 : b.limb === LIMB.fanout1 ? 1 : b.limb === LIMB.fanout2 ? 2 : -1;
    if (sample > 0) {
      if (sample >= arity) growth = 0; // slot unused at this arity: retracted
      else if (sample !== chosen) bleach = 1; // culled: bleached, stays
    } else if (sample === 0 && arity > 1 && chosen !== 0) {
      bleach = 1;
    }

    if (b.limb === LIMB.main) {
      girth = girthOf(state.mainShare, 0.45, 1.1);
      if (state.mainShare === null) inert = 1;
    } else if (b.limb === LIMB.thinking) {
      girth = girthOf(state.thinkingShare, 0.45, 1.6);
      if (state.thinkingShare === null) inert = 1;
    } else if (b.kind === 'trunk') {
      // Sapwood: the trunk is fattest when nothing is unclaimed.
      girth = state.reserveShare === null ? INERT.girth : 1.15 - 0.5 * state.reserveShare;
      if (state.reserveShare === null) inert = 1;
    } else if (b.kind === 'root') {
      inert = 1; // nebari.spread has no source
      girth = INERT.girth + 0.3;
    }
    if (shari.has(b.id)) bleach = 1;

    return {
      growth: clamp01(growth),
      girth: Math.min(GIRTH_MAX, Math.max(GIRTH_MIN, girth)),
      foliage: b.pad && growth > 0 && bleach < 1 ? INERT.foliage : 0,
      foliageLive: 0, // foliage has no source yet
      // Arcs thread the limbs, branches and twigs; never the trunk or roots.
      emissive: bleach >= 1 ? 0 : clamp01(activity * (b.kind === 'twig' ? 1 : b.kind === 'branch' ? 0.6 : b.kind === 'limb' ? 0.35 : 0)),
      bleach: clamp01(bleach),
      cap: capped.has(b.id) ? 1 : 0,
      inert: clamp01(inert),
    };
  });

  return {
    branches,
    scene: {
      haze: state.memPressure ?? INERT.haze,
      alarm: state.anyTight ? 1 : 0,
      activity,
      traces: state.endpoints ?? [],
    },
  };
}

// ----------------------------------------------------------- interpolation

export function lerpBranch(a: BranchParams, b: BranchParams, t: number): BranchParams {
  const u = clamp01(t);
  const l = (x: number, y: number) => x + (y - x) * u;
  return {
    growth: l(a.growth, b.growth),
    girth: l(a.girth, b.girth),
    foliage: l(a.foliage, b.foliage),
    foliageLive: l(a.foliageLive, b.foliageLive),
    emissive: l(a.emissive, b.emissive),
    bleach: l(a.bleach, b.bleach),
    cap: l(a.cap, b.cap),
    inert: l(a.inert, b.inert),
  };
}

/** Which channels are inert under this state -- for the tokonoma caption. */
export function inertChannels(state: TreeState): string[] {
  const out = CHANNELS.filter((c) => c.source === null).map((c) => c.name);
  if (state.seed === null) out.unshift('genome');
  if (state.mainShare === null) out.push('limb.main.girth');
  if (state.thinkingShare === null) out.push('limb.thinking.girth');
  if (state.activity === null) out.push('arcs.activity');
  if (state.memPressure === null) out.push('slab.haze');
  if (state.errored === null) out.push('shari.errored');
  return out;
}
