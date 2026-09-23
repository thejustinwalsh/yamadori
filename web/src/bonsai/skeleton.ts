// Phase 1: the superset skeleton (design/BONSAI-VIZ.md §4).
//
// Grow once, from the seeds. Every branch that could ever exist for these
// seeds exists now; state only animates each branch's parameter vector, so
// nothing is ever added or removed and nothing can pop.
//
// This module is the GENOME: lengths, radii and angles drawn from the PRNG
// using integer and basic float arithmetic only -- no trig, no Math.random,
// no clock -- so a skeleton is byte-identical across runs, machines and
// JavaScript engines. World positions (which need trig) are computed later,
// in pose.ts, and are not part of the determinism contract.
import { mulberry32, substream } from './prng';

export const SKELETON_VERSION = 1;

/** high/max tiers fan out 3 (/dash/api/tiers); the superset holds that many. */
export const MAX_FANOUT = 3;

export type Kind = 'root' | 'trunk' | 'limb' | 'branch' | 'twig';

/**
 * Limb slots. The trunk forks into the MAIN limb (the conversation's KV
 * budget) and the THINKING limb (the deep-thinking budget). Fan-out samples
 * are limbs from the same fork: sample 0 IS the main limb, samples 1..N-1
 * are the FANOUT slots.
 */
export const LIMB = { none: -1, main: 0, thinking: 1, fanout1: 2, fanout2: 3 } as const;
export const LIMB_COUNT = 4;

export type Branch = {
  id: number;
  /** -1: grows from the base (roots, first trunk segment). */
  parent: number;
  kind: Kind;
  limb: number;
  depth: number;
  /** 0..1, where along the parent this branch starts. */
  attach: number;
  length: number;
  radius: number;
  /** end radius / start radius */
  taper: number;
  /** radians away from the parent's axis */
  pitch: number;
  /** radians around the parent's axis */
  yaw: number;
  /** 0..1, how strongly this branch levels out toward horizontal */
  flatten: number;
  /** ends in a foliage pad */
  pad: boolean;
  /** pad size multiplier, meaningful when pad is true */
  padSize: number;
};

export type Skeleton = {
  version: number;
  /** limb seeds, one u32 per limb slot, in LIMB order */
  seeds: number[];
  branches: Branch[];
};

// Every number drawn goes through q(): quantised to 1e-6 so the serialised
// form cannot differ in the 17th digit between two arithmetic orderings.
const q = (x: number) => Math.round(x * 1e6) / 1e6;
const TAU = 6.283185;

type Rng = () => number;
const range = (r: Rng, lo: number, hi: number) => lo + (hi - lo) * r();

/**
 * Limb seeds from the known concept words. `seeds[0]` is the main seed (the
 * vitals `seed.u32`). Slots with no word of their own get a substream of the
 * main seed -- deterministic, and labelled as derived by the caller.
 */
export function limbSeeds(main: number, fanoutSeeds: number[] = []): number[] {
  return [
    main >>> 0,
    substream(main, 1),
    (fanoutSeeds[0] ?? substream(main, 2)) >>> 0,
    (fanoutSeeds[1] ?? substream(main, 3)) >>> 0,
  ];
}

export function growSkeleton(seeds: number[]): Skeleton {
  if (seeds.length !== LIMB_COUNT) throw new Error(`growSkeleton wants ${LIMB_COUNT} limb seeds`);
  const branches: Branch[] = [];
  const add = (b: Omit<Branch, 'id'>) => {
    const id = branches.length;
    branches.push({
      ...b,
      id,
      attach: q(b.attach),
      length: q(b.length),
      radius: q(b.radius),
      taper: q(b.taper),
      pitch: q(b.pitch),
      yaw: q(b.yaw),
      flatten: q(b.flatten),
      padSize: q(b.padSize),
    });
    return id;
  };

  // --- nebari and trunk: from the main seed's own substream -------------
  const base = mulberry32(substream(seeds[0]!, 7));
  const ROOTS = 7;
  const yaw0 = range(base, 0, TAU);
  for (let i = 0; i < ROOTS; i++) {
    add({
      parent: -1, kind: 'root', limb: LIMB.none, depth: 0, attach: 0,
      length: range(base, 0.55, 1.05),
      radius: range(base, 0.11, 0.2),
      taper: range(base, 0.12, 0.22),
      pitch: range(base, 1.62, 1.8), // just past horizontal: roots grip the slab
      yaw: yaw0 + (i * TAU) / ROOTS + range(base, -0.25, 0.25),
      flatten: 0, pad: false, padSize: 0,
    });
  }

  // Informal upright (moyogi): a short, heavily tapered trunk that snakes --
  // each segment bends back across the one below it.
  const SEGMENTS = 5;
  const trunk: number[] = [];
  const trunkR: number[] = [];
  let parent = -1;
  let radius = range(base, 0.3, 0.36);
  const lean = range(base, 0, TAU);
  for (let s = 0; s < SEGMENTS; s++) {
    const length = range(base, 0.36, 0.48) * (1 - s * 0.1);
    const taper = range(base, 0.72, 0.8);
    parent = add({
      parent, kind: 'trunk', limb: LIMB.none, depth: 0, attach: 1,
      length, radius, taper,
      pitch: s === 0 ? range(base, 0.16, 0.26) : range(base, 0.34, 0.5),
      yaw: s === 0 ? lean : s % 2 === 0 ? range(base, -0.4, 0.4) : 3.141593 + range(base, -0.4, 0.4),
      flatten: 0,
      // The apex ends in the crown pad.
      pad: s === SEGMENTS - 1, padSize: s === SEGMENTS - 1 ? range(base, 1.0, 1.25) : 0,
    });
    trunk.push(parent);
    trunkR.push(radius);
    radius *= taper;
  }

  // --- limbs: each from its own seed -------------------------------------
  // The fork sits low on the trunk, as a bonsai's first branch does. Main
  // leaves one side, thinking the other side a segment higher; fan-out
  // samples are siblings of main from the same fork.
  const forkYaw = range(base, 0, TAU);
  const slots = [
    { seg: 1, attach: 0.85, yaw: 0, r: 0.62 },
    { seg: 2, attach: 0.8, yaw: 3.141593, r: 0.66 },
    { seg: 1, attach: 0.85, yaw: 1.4, r: 0.52 },
    { seg: 1, attach: 0.85, yaw: -1.4, r: 0.52 },
  ];
  for (let limb = 0; limb < LIMB_COUNT; limb++) {
    const r = mulberry32(seeds[limb]!);
    const slot = slots[limb]!;
    growLimb(add, r, trunk[slot.seg]!, slot.attach, limb, trunkR[slot.seg]! * slot.r, forkYaw + slot.yaw);
  }

  return { version: SKELETON_VERSION, seeds: seeds.map((s) => s >>> 0), branches };
}

function growLimb(
  add: (b: Omit<Branch, 'id'>) => number,
  r: Rng,
  from: number,
  attach: number,
  limb: number,
  radius: number,
  yaw: number,
) {
  // The limb: leaves the trunk rising, then levels out and reaches.
  const limbId = add({
    parent: from, kind: 'limb', limb, depth: 0, attach,
    length: range(r, 1.35, 1.8),
    radius,
    taper: range(r, 0.42, 0.55),
    pitch: range(r, 0.85, 1.15),
    yaw: yaw + range(r, -0.35, 0.35),
    flatten: range(r, 0.55, 0.75),
    pad: true, padSize: range(r, 1.0, 1.3),
  });

  // Branches along the limb, alternating sides and levelling into tiers.
  const nBranches = 3 + Math.floor(r() * 2); // 3..4
  for (let i = 0; i < nBranches; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const branchId = add({
      parent: limbId, kind: 'branch', limb, depth: 1,
      attach: 0.28 + (0.62 * (i + r() * 0.5)) / nBranches,
      length: range(r, 0.55, 0.85) * (1 - i * 0.1),
      radius: radius * range(r, 0.36, 0.48),
      taper: range(r, 0.45, 0.6),
      pitch: range(r, 1.0, 1.4),
      yaw: side * range(r, 1.3, 1.9),
      flatten: range(r, 0.75, 0.92),
      // Pads sit on the twigs, not the branch: the structure stays visible.
      pad: false, padSize: 0,
    });
    const nTwigs = 2 + Math.floor(r() * 2); // 2..3
    for (let j = 0; j < nTwigs; j++) {
      add({
        parent: branchId, kind: 'twig', limb, depth: 2,
        attach: range(r, 0.4, 0.95),
        length: range(r, 0.26, 0.46),
        radius: radius * range(r, 0.14, 0.2),
        taper: range(r, 0.35, 0.5),
        pitch: range(r, 0.7, 1.2),
        yaw: (j - (nTwigs - 1) / 2) * range(r, 1.2, 1.8) + range(r, -0.3, 0.3),
        flatten: range(r, 0.7, 0.95),
        pad: true, padSize: range(r, 0.5, 0.75),
      });
    }
  }
}

/**
 * Canonical serialisation: fixed key order, numbers already quantised. This
 * string's hash is what the determinism test compares.
 */
export function serializeSkeleton(sk: Skeleton): string {
  const keys: (keyof Branch)[] = [
    'id', 'parent', 'kind', 'limb', 'depth', 'attach', 'length', 'radius', 'taper',
    'pitch', 'yaw', 'flatten', 'pad', 'padSize',
  ];
  const rows = sk.branches.map((b) => '[' + keys.map((k) => JSON.stringify(b[k])).join(',') + ']');
  return `{"version":${sk.version},"seeds":[${sk.seeds.join(',')}],"branches":[${rows.join(',')}]}`;
}
