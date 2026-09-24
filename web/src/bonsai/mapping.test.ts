import { afterEach, describe, expect, it, vi } from 'vitest';
import type { DatasetsOverview, Vitals } from '../api/types';
import liveDatasets from './__fixtures__/datasets.live.json';
import livePulse from './__fixtures__/pulse.live.json';
import liveVitals from './__fixtures__/vitals.live.json';
import {
  CHANNELS,
  GIRTH_MAX,
  GIRTH_MIN,
  INERT,
  inertChannels,
  lerpBranch,
  treeParams,
  treeState,
  type BranchParams,
  type TreeState,
} from './mapping';
import { poseFrames } from './pose';
import { fnv1a32 } from './prng';
import { growSkeleton, LIMB, limbSeeds } from './skeleton';
import { ParamSprings, stepSpring } from './spring';

// vitals.live.json: a live /dash/api/vitals snapshot (2026-09-22, seed
// "inicialmente"), with `context` and `strata` replaced by what the new
// mcp/budget.py budgets(147456) and mcp/vitals.py strata() return, because the
// running proxy predates both fields. The seed is the live one. `tree` is
// mcp/tree_sources.py snapshot() as read on this machine on 2026-09-24 (19
// package indexes, the code index and 2 repo indexes; 162,107 chunks + defs;
// no request since the reading process started, so recent.fanout and
// recent.foliage are null), added because the running proxy predates it.
const VITALS = liveVitals as unknown as Vitals;
const SEED = VITALS.seed!;
const DATASETS = liveDatasets as unknown as DatasetsOverview;
const SK = growSkeleton(limbSeeds(SEED.u32));

function deepFreeze<T>(o: T): T {
  if (o && typeof o === 'object') {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
}

// Adversarial states (§7): empty index, all endpoints down, pool exhausted,
// garbage numbers, missing everything.
const ADVERSARIAL: unknown[] = [
  null,
  undefined,
  {},
  { gpus: [], endpoints: [], listeners: [], warnings: [] },
  { ...VITALS, endpoints: VITALS.endpoints.map((e) => ({ ...e, ok: false, code: 0 })) },
  { ...VITALS, context: { pool: 0, main: 0, helper: 0, reserve: 0, gib: 0 } },
  { ...VITALS, context: { pool: 100, main: 100, helper: 0, reserve: 0, gib: 1 } },
  { ...VITALS, context: { error: 'URLError: refused' } },
  { ...VITALS, gpus: [{ index: 0, name: 'x', used_mib: 1, total_mib: 1, free_mib: 0, pct: 250, tight: true, util: 900 }] },
  { ...VITALS, gpus: [{ index: 0, name: 'x', used_mib: 1, total_mib: 1, free_mib: 0, pct: -5, tight: false, util: Number.NaN }] },
  { ...VITALS, warnings: Array.from({ length: 500 }, (_, i) => `w${i}`) },
  { ...VITALS, seed: null },
  { ...VITALS, seed: { word: 'x', u32: Number.NaN } },
];

const inDomain = (p: BranchParams) => {
  for (const [k, v] of Object.entries(p)) {
    expect(Number.isFinite(v), k).toBe(true);
    if (k === 'girth') {
      expect(v).toBeGreaterThanOrEqual(GIRTH_MIN);
      expect(v).toBeLessThanOrEqual(GIRTH_MAX);
    } else {
      expect(v, k).toBeGreaterThanOrEqual(0);
      expect(v, k).toBeLessThanOrEqual(1);
    }
  }
};

afterEach(() => vi.restoreAllMocks());

describe('mapping purity (§7)', () => {
  it('has no clock, no RNG: both functions run with them booby-trapped', () => {
    vi.spyOn(Math, 'random').mockImplementation(() => {
      throw new Error('Math.random called');
    });
    vi.spyOn(Date, 'now').mockImplementation(() => {
      throw new Error('Date.now called');
    });
    vi.spyOn(performance, 'now').mockImplementation(() => {
      throw new Error('performance.now called');
    });
    expect(() => treeParams(treeState(VITALS, DATASETS), SK)).not.toThrow();
  });

  it('does not mutate its inputs (deep-frozen inputs)', () => {
    const v = deepFreeze(structuredClone(VITALS));
    const d = deepFreeze(structuredClone(DATASETS));
    const sk = deepFreeze(structuredClone(SK));
    expect(() => treeParams(treeState(v, d), sk)).not.toThrow();
  });

  it('same input -> identical output', () => {
    const a = treeParams(treeState(VITALS, DATASETS), SK);
    const b = treeParams(treeState(structuredClone(VITALS), structuredClone(DATASETS)), SK);
    expect(a).toEqual(b);
  });
});

describe('mapping range (§7)', () => {
  it.each(ADVERSARIAL.map((v, i) => [i, v]))('adversarial state #%i stays in domain', (_i, v) => {
    const p = treeParams(treeState(v as Vitals), SK);
    expect(p.branches).toHaveLength(SK.branches.length);
    p.branches.forEach(inDomain);
    expect(p.scene.haze).toBeGreaterThanOrEqual(0);
    expect(p.scene.haze).toBeLessThanOrEqual(1);
    expect(p.scene.activity).toBeGreaterThanOrEqual(0);
    expect(p.scene.activity).toBeLessThanOrEqual(1);
    for (const f of poseFrames(SK, p.branches)) {
      for (const n of [...f.start, ...f.end, f.r0, f.r1]) expect(Number.isFinite(n)).toBe(true);
    }
  });

  it('interpolation stays in range at every t', () => {
    const states = ADVERSARIAL.map((v) => treeParams(treeState(v as Vitals), SK).branches);
    for (let s = 0; s + 1 < states.length; s++) {
      for (const t of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
        states[s]!.forEach((a, i) => inDomain(lerpBranch(a, states[s + 1]![i]!, t)));
      }
    }
  });
});

describe('inert channels emit inert values, never noise (§1, §7)', () => {
  it('a snapshot with nothing in it produces the INERT constants', () => {
    const st = treeState(null);
    expect(st).toEqual({
      seed: null, mainShare: null, thinkingShare: null, reserveShare: null, activity: null,
      memPressure: null, anyTight: null, endpoints: null, alarms: null, errored: null, fanout: null,
      rootSpread: null, foliage: null, moss: null,
    } satisfies TreeState);
    const p = treeParams(st, SK);
    const main = p.branches.filter((_, i) => SK.branches[i]!.limb === LIMB.main);
    for (const b of main) {
      expect(b.girth).toBe(INERT.girth);
      expect(b.inert).toBe(1);
      expect(b.emissive).toBe(0);
    }
    expect(p.scene.haze).toBe(INERT.haze);
    expect(p.scene.activity).toBe(INERT.activity);
  });

  it('foliage is never lit without a recall record', () => {
    for (const v of ADVERSARIAL) {
      const t = (v as Vitals | null)?.tree;
      const live = !!(t && (t as { recent?: unknown }).recent);
      for (const b of treeParams(treeState(v as Vitals), SK).branches) expect(b.foliageLive).toBe(live ? 1 : 0);
    }
    const { tree: _t, ...noTree } = VITALS;
    for (const b of treeParams(treeState(noTree as Vitals), SK).branches) expect(b.foliageLive).toBe(0);
  });

  it('inertChannels() names every sourceless channel', () => {
    const names = inertChannels(treeState(VITALS));
    for (const c of CHANNELS.filter((c) => c.source === null)) expect(names).toContain(c.name);
    expect(inertChannels(treeState(null))).toContain('genome');
  });
});

describe('no fabrication (§7)', () => {
  it('the live seed hashes to the u32 the server sent (Python encode() == TS fnv1a32)', () => {
    expect(fnv1a32(SEED.word)).toBe(SEED.u32);
  });

  const resolve = (obj: unknown, dotted: string): unknown => {
    let cur: unknown[] = [obj];
    for (const part of dotted.split('.')) {
      cur = cur.flatMap((c) => (Array.isArray(c) ? c : [c])).map((c) => (c as Record<string, unknown>)?.[part]);
    }
    return cur.find((x) => x !== undefined);
  };

  it('every sourced channel names a key that exists in a real payload', () => {
    for (const c of CHANNELS) {
      if (c.source === null) continue;
      const payload = c.api === 'datasets' ? DATASETS : c.api === 'pulse' ? livePulse : VITALS;
      expect(resolve(payload, c.source), `${c.name} <- ${c.api}.${c.source}`).not.toBeUndefined();
    }
  });

  it('live snapshot drives the limbs from the real KV split', () => {
    const st = treeState(VITALS);
    const ctx = (liveVitals as unknown as Vitals).context as { pool: number; main: number; helper: number };
    expect(st.mainShare).toBeCloseTo(ctx.main / ctx.pool, 9);
    expect(st.thinkingShare).toBeCloseTo(ctx.helper / ctx.pool, 9);
    const p = treeParams(st, SK);
    const girthOf = (limb: number) => p.branches[SK.branches.findIndex((b) => b.limb === limb && b.kind === 'limb')]!.girth;
    expect(girthOf(LIMB.main)).toBeGreaterThan(girthOf(LIMB.thinking)); // this snapshot predates the 5/8 + 3/8 split: 73,728 main vs 36,864 per deep-thinking context; either way main is the thicker limb
  });
});

describe('topology stability and fan-out arity (§7)', () => {
  it('branch count is constant across every state', () => {
    for (const v of ADVERSARIAL) expect(treeParams(treeState(v as Vitals), SK).branches).toHaveLength(SK.branches.length);
  });

  it.each([1, 2, 3])('fan-out %i: exactly N limbs grown; culled ones bleach, never vanish', (n) => {
    const st: TreeState = { ...treeState(VITALS), fanout: { arity: n, chosen: 0 } };
    const p = treeParams(st, SK);
    const limbIdx = (limb: number) => SK.branches.findIndex((b) => b.kind === 'limb' && b.limb === limb);
    const sampleSlots = [LIMB.main, LIMB.fanout1, LIMB.fanout2];
    const grown = sampleSlots.filter((l) => p.branches[limbIdx(l)]!.growth > 0);
    expect(grown).toHaveLength(n);
    for (const l of grown.slice(1)) expect(p.branches[limbIdx(l)]!.bleach).toBe(1);
    expect(p.branches[limbIdx(LIMB.main)]!.bleach).toBe(0);
  });

  it('with no fan-out signal the tree is a single trunk (arity 1)', () => {
    const p = treeParams(treeState(VITALS), SK);
    for (const b of SK.branches.filter((x) => x.limb === LIMB.fanout1 || x.limb === LIMB.fanout2)) {
      expect(p.branches[b.id]!.growth).toBe(0);
    }
  });

  it('errored jobs bleach twigs to shari, capped, never removed', () => {
    const d = structuredClone(DATASETS);
    d.queue.states.errored = 4;
    const p = treeParams(treeState(VITALS, d), SK);
    expect(p.branches.filter((b) => b.bleach === 1)).toHaveLength(4);
    d.queue.states.errored = 10_000;
    const q = treeParams(treeState(VITALS, d), SK);
    expect(q.branches).toHaveLength(SK.branches.length);
  });
});

describe('springs', () => {
  it('critically damped: converges without overshoot', () => {
    let s = { x: 0, v: 0 };
    let max = 0;
    for (let i = 0; i < 600; i++) {
      s = stepSpring(s, 1, 4, 1 / 60);
      max = Math.max(max, s.x);
    }
    expect(s.x).toBeCloseTo(1, 4);
    expect(max).toBeLessThanOrEqual(1 + 1e-9);
  });

  it('ParamSprings reports settled, which is when rendering stops', () => {
    const a = treeParams(treeState(null), SK).branches;
    const b = treeParams(treeState(VITALS), SK).branches;
    const sp = new ParamSprings(a);
    let frames = 0;
    while (sp.step(b, 1 / 60) && frames < 60 * 30) frames++;
    expect(frames).toBeLessThan(60 * 30);
    b.forEach((t, i) => expect(sp.get(i).girth).toBeCloseTo(t.girth, 2));
  });
});

// ------------------------------------------------ the four tree sources

const withTree = (tree: unknown): Vitals => ({ ...VITALS, tree: tree as Vitals['tree'] });
const { tree: _dropped, ...BARE } = VITALS;
const NO_TREE = BARE as Vitals;
const roots = (p: ReturnType<typeof treeParams>) => p.branches.filter((_, i) => SK.branches[i]!.kind === 'root');
const limbIdx = (limb: number) => SK.branches.findIndex((b) => b.kind === 'limb' && b.limb === limb);

describe('nebari.spread <- index breadth (vitals.tree.nebari.spread)', () => {
  it('the live fixture spreads and thickens the roots, and they are not inert', () => {
    const st = treeState(VITALS);
    expect(st.rootSpread).toBeCloseTo(0.7366, 3);
    for (const r of roots(treeParams(st, SK))) {
      expect(r.inert).toBe(0);
      expect(r.growth).toBeCloseTo(0.55 + 0.45 * 0.7366, 3);
      expect(r.girth).toBeCloseTo(0.8 + 0.5 * 0.7366, 3);
    }
    expect(inertChannels(st)).not.toContain('nebari.spread');
  });
  it('more breadth, longer and thicker roots', () => {
    const lo = roots(treeParams(treeState(withTree({ nebari: { spread: 0.1 } })), SK))[0]!;
    const hi = roots(treeParams(treeState(withTree({ nebari: { spread: 0.9 } })), SK))[0]!;
    expect(hi.growth).toBeGreaterThan(lo.growth);
    expect(hi.girth).toBeGreaterThan(lo.girth);
  });
  it('absent: the old fixed roots, inert (dormant bark), and named inert', () => {
    for (const st of [treeState(NO_TREE), treeState(withTree({ nebari: { spread: null } }))]) {
      expect(st.rootSpread).toBeNull();
      for (const r of roots(treeParams(st, SK))) {
        expect(r.inert).toBe(1);
        expect(r.growth).toBe(1);
        expect(r.girth).toBe(INERT.girth + 0.3);
      }
      expect(inertChannels(st)).toContain('nebari.spread');
    }
  });
});

describe('fanout <- the last x_yamadori.fanout (vitals.tree.recent.fanout)', () => {
  const tree = (fanout: unknown) => withTree({ ...VITALS.tree, recent: { fanout, foliage: null } });
  it('arity 3, candidate 2 delivered: three limbs, the other two bleached', () => {
    const st = treeState(tree({ arity: 3, chosen: 2, culled: [0, 1], age_s: 5, fade: 1 }));
    expect(st.fanout).toEqual({ arity: 3, chosen: 2, fade: 1 });
    const p = treeParams(st, SK);
    expect(p.branches[limbIdx(LIMB.fanout2)]!.bleach).toBe(0);
    expect(p.branches[limbIdx(LIMB.fanout1)]!.bleach).toBe(1);
    expect(p.branches[limbIdx(LIMB.main)]!.bleach).toBe(1);
    expect(p.branches[limbIdx(LIMB.fanout2)]!.growth).toBe(1);
  });
  it('after the hold the fork retracts with the fade, then is arity 1', () => {
    const half = treeParams(treeState(tree({ arity: 2, chosen: 0, fade: 0.5 })), SK);
    expect(half.branches[limbIdx(LIMB.fanout1)]!.growth).toBe(0.5);
    const gone = treeState(tree({ arity: 2, chosen: 0, fade: 0 }));
    expect(gone.fanout).toBeNull();
    expect(treeParams(gone, SK).branches[limbIdx(LIMB.fanout1)]!.growth).toBe(0);
  });
  it('no recent fan-out is measured arity 1, not inert; no recent block is inert', () => {
    expect(treeState(VITALS).fanout).toBeNull();
    expect(inertChannels(treeState(VITALS))).not.toContain('fanout');
    expect(inertChannels(treeState(NO_TREE))).toContain('fanout');
  });
});

describe('foliage <- recall injected lately (vitals.tree.recent.foliage)', () => {
  const tree = (foliage: unknown) => withTree({ ...VITALS.tree, recent: { fanout: null, foliage } });
  const pads = (p: ReturnType<typeof treeParams>) => p.branches.filter((b) => b.foliage > 0);
  it('density sizes the pads and lights them; the recall path is kept', () => {
    const full = treeParams(treeState(tree({ density: 1, path: 'skills' })), SK);
    const bare = treeParams(treeState(tree({ density: 0, path: 'hints' })), SK);
    expect(pads(full)[0]!.foliage).toBeCloseTo(1, 6);
    expect(pads(bare)[0]!.foliage).toBeCloseTo(0.35, 6);
    expect(full.scene.foliageLive).toBe(1);
    expect(full.scene.foliageDensity).toBe(1);
    expect(treeState(tree({ density: 0.4, path: 'skills' })).foliage).toEqual({ density: 0.4, path: 'skills' });
  });
  it('a recent block with no turns in the window is sparse and live, not inert', () => {
    const st = treeState(VITALS);
    expect(st.foliage).toEqual({ density: 0, path: 'hints' }); // the live YAMADORI_RECALL, from tree.moss.recall
    expect(inertChannels(st)).not.toContain('foliage');
  });
  it('absent: full matte pads, unlit, named inert', () => {
    const st = treeState(NO_TREE);
    expect(st.foliage).toBeNull();
    const p = treeParams(st, SK);
    expect(pads(p)[0]!.foliage).toBe(INERT.foliage);
    expect(p.scene.foliageLive).toBe(0);
    expect(inertChannels(st)).toContain('foliage');
  });
});

describe('moss <- index staleness (vitals.tree.moss.value)', () => {
  it('the live fixture carries its value to the scene', () => {
    const p = treeParams(treeState(VITALS), SK);
    expect(p.scene.moss).toBeCloseTo(0.2921, 3);
  });
  it('clamped into [0, 1]', () => {
    expect(treeState(withTree({ moss: { value: 7 } })).moss).toBe(1);
    expect(treeState(withTree({ moss: { value: -1 } })).moss).toBe(0);
  });
  it('absent: no moss drawn (0), named inert', () => {
    const st = treeState(NO_TREE);
    expect(st.moss).toBeNull();
    expect(treeParams(st, SK).scene.moss).toBe(0);
    expect(inertChannels(st)).toContain('moss');
    expect(inertChannels(treeState(withTree({ moss: { value: 'stale' } })))).toContain('moss');
  });
});

describe('the model card is chosen by the main flag, not by index', () => {
  const gpu = (index: number, util: number, main?: boolean) => ({
    index, name: `gpu${index}`, used_mib: 1, total_mib: 2, free_mib: 1, pct: 50, tight: false, util,
    ...(main === undefined ? {} : { main }),
  });
  it('reads the flagged card even when it is not index 0', () => {
    const v = { ...(VITALS as Vitals), gpus: [gpu(0, 0, false), gpu(1, 80, true)] } as Vitals;
    expect(treeState(v, DATASETS).activity).toBeCloseTo(0.8);
  });
  it('falls back to index 0 only when the server sends no flag', () => {
    const v = { ...(VITALS as Vitals), gpus: [gpu(0, 40), gpu(1, 90)] } as Vitals;
    expect(treeState(v, DATASETS).activity).toBeCloseTo(0.4);
  });
});
