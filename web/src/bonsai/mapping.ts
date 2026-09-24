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
 *  - "Bioluminescent arcs <- token throughput": tok/s is not in the vitals
 *    payload. The arcs read GPU0 utilisation instead -- the card the model
 *    runs on is computing or it is not. Real, and named as what it is.
 *  - "Foliage density <- hints above the floor": the floor is applied before
 *    anything reaches the payload, so foliage reads what recall actually
 *    INJECTED: items per task turn over the last 30 min (x_yamadori.skills
 *    on YAMADORI_RECALL=skills, x_yamadori.hints on =hints; the path is
 *    labelled). 3 items per turn is full foliage -- a choice, not a measure.
 *  - "Nebari <- corpus breadth": the breadth is the indexes the server
 *    holds (package indexes, the bound code index, repository indexes:
 *    chunks + defs), on a log scale, not the corpus log.
 *  - fanout, foliage, nebari and moss come from vitals.tree
 *    (mcp/tree_sources.py), which the proxy fills from its own per-request
 *    records (mcp/recent_turns.py) and index counts cached for a minute.
 *    Absent on a server that predates it: those channels are INERT.
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
  { name: 'fanout', source: 'tree.recent.fanout', api: 'vitals', reads: 'the last fan-out (x_yamadori.fanout): arity, delivered, culled; held 120 s, retracts over 60 s' },
  { name: 'foliage', source: 'tree.recent.foliage', api: 'vitals', reads: 'recall injected per task turn, last 30 min (x_yamadori.skills / .hints, per YAMADORI_RECALL): pad density' },
  { name: 'nebari.spread', source: 'tree.nebari.spread', api: 'vitals', reads: 'index breadth: package + code + repo indexes, log10(chunks + defs): root reach and girth' },
  { name: 'sway', source: 'slots.slots.state', api: 'pulse', reads: 'busy llama-server slots (main and deep thinking): wind and gusts' },
  { name: 'ground.glow', source: 'slots.slots.tps', api: 'pulse', reads: 'decode tokens/s across slots: slab rings and traces brighten' },
  { name: 'rings.request', source: 'tools.last_turn.id', api: 'pulse', reads: 'a request arrived (or a slot woke): a green pulse' },
  { name: 'rings.tool', source: 'tools.last.id', api: 'pulse', reads: 'a tool was called: a cyan pulse' },
  { name: 'rings.seed', source: 'seed.at', api: 'pulse', reads: 'a concept seed was drawn: a pale pulse' },
  { name: 'moss', source: 'tree.moss.value', api: 'vitals', reads: 'index staleness: package indexes, code index, last skill arm (recipes on the hints path); fresh = bare bark' },
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
  /** fade: 1 while held, falling to 0 as the fork retracts to arity 1 */
  fanout: { arity: number; chosen: number; fade?: number } | null;
  /** index breadth, [0, 1] (log scale) */
  rootSpread: number | null;
  /** recall injected per recent task turn, [0, 1]; path is YAMADORI_RECALL */
  foliage: { density: number; path: string } | null;
  /** index staleness, [0, 1] */
  moss: number | null;
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
  // The model's card by UUID (vitals marks it `main`); index 0 only for an
  // older server that does not send the flag.
  const gpu0 = gpus.find((g) => g.main === true) ?? gpus.find((g) => g.main === undefined && g.index === 0) ?? null;
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
    ...treeSources(v.tree),
  };
}

const pick = (o: unknown, ...keys: string[]): unknown => {
  let cur: unknown = o;
  for (const k of keys) cur = cur && typeof cur === 'object' ? (cur as Record<string, unknown>)[k] : undefined;
  return cur;
};

/** The vitals.tree fields (mcp/tree_sources.py). An absent tree, or a field
 *  without a number: that channel is null, so INERT. */
function treeSources(tree: unknown): Pick<TreeState, 'fanout' | 'rootSpread' | 'foliage' | 'moss'> {
  const spread = pick(tree, 'nebari', 'spread');
  const moss = pick(tree, 'moss', 'value');
  const recent = pick(tree, 'recent');
  const live = !!recent && typeof recent === 'object' && !('error' in (recent as object));
  const f = live ? pick(recent, 'fanout') : null;
  const arity = pick(f, 'arity');
  const chosen = pick(f, 'chosen');
  const fadeRaw = pick(f, 'fade');
  const fade = finite(fadeRaw) ? clamp01(fadeRaw) : 1;
  const density = live ? pick(recent, 'foliage', 'density') : null;
  // The recall path: the turns' own, else the live YAMADORI_RECALL (moss reads it).
  const path = live ? (pick(recent, 'foliage', 'path') ?? pick(tree, 'moss', 'recall')) : null;
  return {
    fanout: finite(arity) && arity >= 2 && fade > 0 ? { arity: Math.floor(arity), chosen: finite(chosen) ? Math.floor(chosen) : 0, fade } : null,
    rootSpread: finite(spread) ? clamp01(spread) : null,
    // A recent block with no task turn in the window is measured: nothing
    // was recalled, so the pads are sparse, not inert.
    foliage: live ? { density: finite(density) ? clamp01(density) : 0, path: typeof path === 'string' ? path : '—' } : null,
    moss: finite(moss) ? clamp01(moss) : null,
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
  /** index staleness in [0, 1]; 0 when inert (no moss is drawn for unknown) */
  moss: number;
  /** foliage is driven by recall (1) or inert (0) */
  foliageLive: number;
  /** recall density, [0, 1]; 0 when inert */
  foliageDensity: number;
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
  // A held fork is full grown; a retracting one shrinks toward arity 1.
  const fade = fan ? clamp01(fan.fade ?? 1) : 1;
  const fol = state.foliage;
  // Live pads: sparse at no recall, full at 3 items per turn.
  const padSize = fol ? 0.35 + 0.65 * fol.density : INERT.foliage;
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
      else {
        growth = fade;
        if (sample !== chosen) bleach = 1; // culled: bleached, stays
      }
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
      // Nebari: the breadth of what the server knows spreads and thickens
      // the roots. Unknown: the fixed roots, drawn as dormant bark.
      if (state.rootSpread === null) {
        inert = 1;
        girth = INERT.girth + 0.3;
      } else {
        growth = 0.55 + 0.45 * state.rootSpread;
        girth = 0.8 + 0.5 * state.rootSpread;
      }
    }
    if (shari.has(b.id)) bleach = 1;

    return {
      growth: clamp01(growth),
      girth: Math.min(GIRTH_MAX, Math.max(GIRTH_MIN, girth)),
      foliage: b.pad && growth > 0 && bleach < 1 ? padSize : 0,
      foliageLive: fol ? 1 : 0,
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
      moss: state.moss ?? 0,
      foliageLive: fol ? 1 : 0,
      foliageDensity: fol ? fol.density : 0,
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
  if (state.rootSpread === null) out.push('nebari.spread');
  if (state.foliage === null) out.push('foliage');
  if (state.moss === null) out.push('moss');
  // fanout: null both when nothing fanned out lately (measured: arity 1)
  // and when the recent block is absent (foliage null too); only that is inert.
  if (state.foliage === null) out.push('fanout');
  return out;
}
