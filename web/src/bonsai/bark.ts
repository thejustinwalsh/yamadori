// The bark as ONE merged tube mesh over the fixed superset topology.
//
// BONSAI-VIZ §4 asks for "one instanced draw". Instanced cylinders cannot
// taper per instance here: three's NodeMaterial applies `positionNode` after
// the instance matrix, so a per-instance taper would act in instanced space.
// A merged mesh keeps what §4 actually cares about -- a fixed topology, one
// draw call, nothing ever added or removed -- and lets branches curve, flare
// at the root and join smoothly. Vertex COUNT never changes; only positions
// and the per-vertex state move, and only while a spring is settling.
//
// Pure given its inputs: deterministic bark texture comes from the branch id,
// never from an RNG.
import type { BranchParams } from './mapping';
import type { Frame } from './pose';
import type { Skeleton } from './skeleton';

export const RADIAL = 10;
export const RINGS = 7;
const VPB = (RADIAL + 1) * RINGS; // vertices per branch (seam duplicated for uv)

export type BarkBuffers = {
  position: Float32Array;
  normal: Float32Array;
  uv: Float32Array;
  /** per vertex: emissive, bleach, inert, depth-ish (0 trunk .. 1 twig) */
  state: Float32Array;
  index: Uint32Array;
  vertexCount: number;
};

export function allocateBark(sk: Skeleton): BarkBuffers {
  const n = sk.branches.length;
  const vertexCount = n * VPB;
  const index = new Uint32Array(n * RADIAL * (RINGS - 1) * 6);
  let o = 0;
  for (let b = 0; b < n; b++) {
    const base = b * VPB;
    for (let k = 0; k < RINGS - 1; k++) {
      for (let a = 0; a < RADIAL; a++) {
        const i0 = base + k * (RADIAL + 1) + a;
        const i1 = i0 + 1;
        const i2 = i0 + RADIAL + 1;
        const i3 = i2 + 1;
        // Counter-clockwise seen from OUTSIDE the tube. The ring sweeps
        // counter-clockwise around the branch axis (normal = side*cos +
        // (dir x side)*sin), so going AROUND first and then UP gives an
        // outward face: (p1 - p0) x (p2 - p0) = tangent x dir = +radial.
        // It was (i0, i2, i1), which faced inward: back-face culling then
        // showed the inside of the far wall and the trunk looked inside-out
        // under the outward vertex normals. Asserted in bark.test.ts.
        index[o++] = i0;
        index[o++] = i1;
        index[o++] = i2;
        index[o++] = i1;
        index[o++] = i3;
        index[o++] = i2;
      }
    }
  }
  return {
    position: new Float32Array(vertexCount * 3),
    normal: new Float32Array(vertexCount * 3),
    uv: new Float32Array(vertexCount * 2),
    state: new Float32Array(vertexCount * 4),
    index,
    vertexCount,
  };
}

const KIND_DEPTH = { root: 0, trunk: 0, limb: 0.35, branch: 0.7, twig: 1 } as const;

/**
 * A trunk segment that continues another from its end. The trunk is a chain
 * of such segments, each bent back across the one below (moyogi); their
 * joints are welded (see writeTube).
 */
const continues = (sk: Skeleton, b: Skeleton['branches'][number]) =>
  b.kind === 'trunk' && b.parent >= 0 && b.attach >= 1 && sk.branches[b.parent]!.kind === 'trunk';

export function writeBark(buf: BarkBuffers, sk: Skeleton, frames: Frame[], params: ArrayLike<BranchParams>) {
  // Trunk joints. Each segment's rings are square to its own axis, so at a
  // bend the parent's last ring and the child's first ring were two circles
  // tilted 20-30 degrees apart: a wedge-shaped crack on the outside of every
  // bend, widest where the flare meets the trunk above it. Now the parent's
  // last ring is mitred onto the plane bisecting the two axes, the child's
  // first ring IS that ring (same vertices, same normals), and the child's
  // rings are turned (`spin`) so vertex a sits under vertex a: no twist.
  const n = sk.branches.length;
  const spin = new Float64Array(n);
  const child = new Int32Array(n).fill(-1);
  for (const b of sk.branches) if (continues(sk, b)) child[b.parent] = b.id;
  // Roots last: each one is welded onto trunk vertices written in this pass.
  // Parents come before children in genome order, so a joint's parent ring
  // is always written when its child reads it.
  for (const b of sk.branches) {
    if (b.kind === 'root') continue;
    const f = frames[b.id]!;
    let weld = -1;
    if (continues(sk, b)) {
      weld = b.parent;
      const pf = frames[b.parent]!;
      const ps = spin[b.parent]!;
      const [pax, pay, paz] = pf.side;
      const [pdx, pdy, pdz] = pf.dir;
      // the parent's ring-start direction, after its own spin
      const pbx = pdy * paz - pdz * pay;
      const pby = pdz * pax - pdx * paz;
      const pbz = pdx * pay - pdy * pax;
      const ex = pax * Math.cos(ps) + pbx * Math.sin(ps);
      const ey = pay * Math.cos(ps) + pby * Math.sin(ps);
      const ez = paz * Math.cos(ps) + pbz * Math.sin(ps);
      const [ax, ay, az] = f.side;
      const [dx, dy, dz] = f.dir;
      const bx = dy * az - dz * ay;
      const by = dz * ax - dx * az;
      const bz = dx * ay - dy * ax;
      spin[b.id] = Math.atan2(ex * bx + ey * by + ez * bz, ex * ax + ey * ay + ez * az);
    }
    const c = child[b.id]!;
    writeTube(buf, b, f, params[b.id]!, spin[b.id]!, c >= 0 ? frames[c]!.dir : null, weld);
  }
  const base = sk.branches.find((b) => b.kind === 'trunk' && b.parent === -1);
  const roots = sk.branches.filter((b) => b.kind === 'root');
  if (!base) {
    for (const b of roots) writeTube(buf, b, frames[b.id]!, params[b.id]!);
    return;
  }
  const cells = rootCells(roots, frames, frames[base.id]!);
  for (const b of roots) writeRoot(buf, b, frames[b.id]!, params[b.id]!, base.id, frames[base.id]!, cells.get(b.id)!);
}

function writeTube(
  buf: BarkBuffers,
  b: Skeleton['branches'][number],
  f: Frame,
  p: BranchParams,
  spin = 0,
  /** the continuing child's axis: mitre the last ring onto the bisecting plane */
  next: Frame['dir'] | null = null,
  /** the parent whose last ring this tube's first ring is welded to, or -1 */
  weld = -1,
) {
  const { position, normal, uv, state } = buf;
  // Mitre plane normal at the far joint: the bisector of the two axes.
  let mx0 = 0;
  let my0 = 0;
  let mz0 = 0;
  if (next) {
    mx0 = f.dir[0] + next[0];
    my0 = f.dir[1] + next[1];
    mz0 = f.dir[2] + next[2];
    const ml = Math.hypot(mx0, my0, mz0) || 1;
    mx0 /= ml;
    my0 /= ml;
    mz0 /= ml;
  }
  const along = f.dir[0] * mx0 + f.dir[1] * my0 + f.dir[2] * mz0;
  {
    const [sx, sy, sz] = f.start;
    const [dx, dy, dz] = f.dir;
    const [ax, ay, az] = f.side;
    // binormal = dir x side
    const bx = dy * az - dz * ay;
    const by = dz * ax - dx * az;
    const bz = dx * ay - dy * ax;
    // Gravity sag for everything above the trunk: a bonsai's pads weigh down
    // their branches. Zero at the joint, full at the tip.
    const sag = b.kind === 'trunk' || b.kind === 'root' ? 0 : f.length * 0.1 * (1 - b.flatten * 0.4);
    const flare = b.kind === 'trunk' && b.parent === -1;
    const depth = KIND_DEPTH[b.kind];
    for (let k = 0; k < RINGS; k++) {
      const t = k / (RINGS - 1);
      const cx = sx + dx * f.length * t;
      const cy = sy + dy * f.length * t - sag * t * t;
      const cz = sz + dz * f.length * t;
      let r = f.r0 + (f.r1 - f.r0) * t;
      // Radius slope along the axis, for the flare's normals: the nebari
      // skirt is nearly horizontal at the foot and must light like it.
      let slope = f.length > 1e-6 ? (f.r1 - f.r0) / f.length : 0;
      if (flare) {
        const fl = 1 + 1.1 * (1 - t) ** 4; // nebari: the trunk flares into its roots
        slope = f.length > 1e-6 ? (slope * fl * f.length - r * 4.4 * (1 - t) ** 3) / f.length : 0;
        r *= fl;
      }
      if (b.kind === 'root') r *= 1 - 0.35 * t;
      // The flare's lowest rings are levelled onto the slab: the first trunk
      // segment leans, and a ring square to a leaning axis lifts off the
      // ground on one side and opens a gap under the trunk.
      const level = flare ? (1 - t) ** 6 : 0;
      const mitre = next !== null && k === RINGS - 1 && along > 1e-3;
      for (let a = 0; a <= RADIAL; a++) {
        const ang = (a / RADIAL) * Math.PI * 2 + spin;
        const v = b.id * VPB + k * (RADIAL + 1) + a;
        uv[v * 2] = a / RADIAL;
        uv[v * 2 + 1] = t * Math.max(f.length, 1e-3) * 4;
        state[v * 4] = p.emissive;
        state[v * 4 + 1] = p.bleach;
        state[v * 4 + 2] = p.inert;
        state[v * 4 + 3] = depth;
        if (weld >= 0 && k === 0) {
          // The joint ring is the parent's mitred last ring, shared exactly.
          const w = weld * VPB + (RINGS - 1) * (RADIAL + 1) + a;
          for (let i = 0; i < 3; i++) {
            position[v * 3 + i] = position[w * 3 + i]!;
            normal[v * 3 + i] = normal[w * 3 + i]!;
          }
          continue;
        }
        // Deterministic bark ridges, keyed on the branch id.
        const ridge = 1 + 0.07 * Math.sin(ang * 3 + b.id * 1.7 + t * 5) + 0.04 * Math.sin(ang * 7 - b.id);
        const c = Math.cos(ang);
        const s = Math.sin(ang);
        const nx = ax * c + bx * s;
        const ny = ay * c + by * s;
        const nz = az * c + bz * s;
        let px = cx + nx * r * ridge;
        let py = cy + ny * r * ridge;
        let pz = cz + nz * r * ridge;
        if (mitre) {
          // Slide along the axis onto the bisecting plane through the joint.
          const off = ((px - cx) * mx0 + (py - cy) * my0 + (pz - cz) * mz0) / along;
          px -= dx * off;
          py -= dy * off;
          pz -= dz * off;
        }
        position[v * 3] = px;
        position[v * 3 + 1] = py + (sy - FOOT_SINK - py) * level;
        position[v * 3 + 2] = pz;
        // Surface normal of a tube whose radius changes along its axis:
        // the radial direction tilted back along the axis by the slope.
        const mx = nx - dx * slope;
        const my = ny - dy * slope;
        const mz = nz - dz * slope;
        const ml = Math.hypot(mx, my, mz) || 1;
        normal[v * 3] = mx / ml;
        normal[v * 3 + 1] = my / ml;
        normal[v * 3 + 2] = mz / ml;
      }
    }
  }
}

// ------------------------------------------------------------------ nebari
//
// The roots used to be separate tubes starting on the trunk's AXIS: each one
// pierced the flare, and where the two surfaces crossed there was a crack
// visible from most orbits. Now every root is welded on. The flare's grid has
// RADIAL columns; each root owns one cell of it, one column wide and
// FOOT_RINGS rings tall, and its first ring IS that cell's boundary loop --
// the same ten trunk vertices, the same positions, the same normals. Over the
// next rings the loop is carried outward and morphed into the root's own
// round cross-section, so the root grows out of the flare the way a nebari
// buttress does. Nothing here changes topology: vertex count is fixed.

/** Rings of the flare a root's footprint spans. 2 * (1 + FOOT_RINGS) = RADIAL. */
const FOOT_RINGS = RADIAL / 2 - 1;
/** The foot of the flare sits this far into the slab, so the underside of
 *  every root, where it folds against the flare, is under the slab top. */
export const FOOT_SINK = 0.03;
/** How far along its length a root takes to become round. */
const WELD_T = 0.45;
/** Root length used outside the flare, as a share of the genome length. */
const ROOT_REACH = 0.75;
/** How far below its own heading a root tip sinks: roots dive into the soil. */
const ROOT_DIVE = 0.16;

/** Which flare cell each root grows from: nearest free cell to its heading. */
export function rootCells(roots: Skeleton['branches'], frames: Frame[], trunk: Frame): Map<number, number> {
  const [ax, ay, az] = trunk.side;
  const [dx, dy, dz] = trunk.dir;
  const bx = dy * az - dz * ay;
  const by = dz * ax - dx * az;
  const bz = dx * ay - dy * ax;
  const cellW = (Math.PI * 2) / RADIAL;
  const want = roots.map((b) => {
    const [x, y, z] = frames[b.id]!.dir;
    let ang = Math.atan2(x * bx + y * by + z * bz, x * ax + y * ay + z * az);
    if (ang < 0) ang += Math.PI * 2;
    return { id: b.id, ang };
  });
  const taken = new Set<number>();
  const out = new Map<number, number>();
  // Deterministic order: the root closest to a cell centre claims first.
  const off = (ang: number) => Math.abs(ang / cellW - (Math.floor(ang / cellW) + 0.5));
  for (const w of [...want].sort((p, q) => off(p.ang) - off(q.ang) || p.id - q.id)) {
    const home = Math.floor(w.ang / cellW) % RADIAL;
    const right = w.ang / cellW - home > 0.5;
    for (let step = 0; step < RADIAL; step++) {
      // home, then the nearer neighbour, then the farther, widening
      const d = step === 0 ? 0 : Math.ceil(step / 2) * ((step % 2 === 1) === right ? 1 : -1);
      const c = (((home + d) % RADIAL) + RADIAL) % RADIAL;
      if (!taken.has(c)) {
        taken.add(c);
        out.set(w.id, c);
        break;
      }
    }
  }
  return out;
}

const smooth = (e0: number, e1: number, x: number) => {
  const u = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return u * u * (3 - 2 * u);
};

function writeRoot(
  buf: BarkBuffers,
  b: Skeleton['branches'][number],
  f: Frame,
  p: BranchParams,
  trunkId: number,
  trunk: Frame,
  cell: number,
) {
  const { position: P, normal: N, uv, state } = buf;
  const RING = RADIAL + 1;
  const tv = (col: number, ring: number) => trunkId * VPB + ring * RING + (col % RADIAL);

  // The cell's boundary loop on the flare: along the foot, up one side,
  // back across the top, down the other.
  const c0 = cell;
  const c1 = cell + 1;
  const loop: number[] = [tv(c0, 0), tv(c1, 0)];
  for (let k = 1; k <= FOOT_RINGS; k++) loop.push(tv(c1, k));
  loop.push(tv(c0, FOOT_RINGS));
  for (let k = FOOT_RINGS - 1; k >= 1; k--) loop.push(tv(c0, k));

  let ox = 0;
  let oy = 0;
  let oz = 0;
  for (const v of loop) {
    ox += P[v * 3]!;
    oy += P[v * 3 + 1]!;
    oz += P[v * 3 + 2]!;
  }
  ox /= loop.length;
  oy /= loop.length;
  oz /= loop.length;

  // Heading: out through the middle of the cell, level, keeping the root's
  // own dip toward the slab.
  const [ax, ay, az] = trunk.side;
  const [tdx, tdy, tdz] = trunk.dir;
  const bx = tdy * az - tdz * ay;
  const bz = tdx * ay - tdy * ax;
  const mid = ((cell + 0.5) / RADIAL) * Math.PI * 2;
  let hx = ax * Math.cos(mid) + bx * Math.sin(mid);
  let hz = az * Math.cos(mid) + bz * Math.sin(mid);
  const hl = Math.hypot(hx, hz) || 1;
  hx /= hl;
  hz /= hl;
  // The round part starts out at the foot of the flare, not at the
  // footprint's middle: started inside, its underside would double back
  // under the flare and fold.
  const [tsx, , tsz] = trunk.start;
  const foot = (v: number) => Math.hypot(P[v * 3]! - tsx, P[v * 3 + 2]! - tsz);
  const push = Math.max(0, (foot(loop[0]!) + foot(loop[1]!)) / 2 - Math.hypot(ox - tsx, oz - tsz));
  const rx = ox + hx * push;
  const rz = oz + hz * push;
  const dip = Math.max(-0.5, Math.min(0, f.dir[1]));
  const flat = Math.sqrt(1 - dip * dip);
  const dx = hx * flat;
  const dy = dip;
  const dz = hz * flat;
  // side = level perpendicular to the heading; up = dir x side.
  const sx = -hz;
  const sz = hx;
  const ux = dy * sz;
  const uy = dz * sx - dx * sz;
  const uz = -dy * sx;

  // Orient the loop counter-clockwise about the heading (the winding the
  // index buffer assumes) and start it where the round ring starts, so the
  // morph never twists.
  const rel = loop.map((v) => [P[v * 3]! - ox, P[v * 3 + 1]! - oy, P[v * 3 + 2]! - oz] as const);
  let turn = 0;
  for (let i = 0; i < rel.length; i++) {
    const [x0, y0, z0] = rel[i]!;
    const [x1, y1, z1] = rel[(i + 1) % rel.length]!;
    turn += (y0 * z1 - z0 * y1) * dx + (z0 * x1 - x0 * z1) * dy + (x0 * y1 - y0 * x1) * dz;
  }
  if (turn < 0) {
    loop.reverse();
    rel.reverse();
  }
  const phi = rel.map(([x, y, z]) => Math.atan2(x * ux + y * uy + z * uz, x * sx + z * sz));
  let start = 0;
  for (let i = 1; i < phi.length; i++) if (Math.abs(phi[i]!) < Math.abs(phi[start]!)) start = i;
  // The round ring is evenly spaced, starting at the loop's first vertex:
  // footprint angles bunch at the foot, where the flare is nearly flat, and
  // bunched angles fold the round part of the root over itself.
  const order: number[] = [];
  const ang: number[] = [];
  for (let i = 0; i <= loop.length; i++) {
    order.push(loop[(start + i) % loop.length]!);
    ang.push(phi[start]! + (i / loop.length) * Math.PI * 2);
  }

  const len = f.length * ROOT_REACH;
  const depth = KIND_DEPTH.root;
  for (let k = 0; k < RINGS; k++) {
    const t = k / (RINGS - 1);
    const w = smooth(0, WELD_T, t);
    const r = (f.r0 + (f.r1 - f.r0) * t) * (1 - 0.35 * t);
    const cx = rx + dx * len * t;
    const cy = oy + dy * len * t - ROOT_DIVE * t * t;
    const cz = rz + dz * len * t;
    for (let a = 0; a <= RADIAL; a++) {
      const fv = order[a]!;
      const v = b.id * VPB + k * RING + a;
      const c = Math.cos(ang[a]!);
      const s = Math.sin(ang[a]!);
      const nx = sx * c + ux * s;
      const ny = uy * s;
      const nz = sz * c + uz * s;
      const ridge = 1 + 0.07 * Math.sin(ang[a]! * 3 + b.id * 1.7 + t * 5);
      // carried footprint -> round cross-section
      const ex = P[fv * 3]! + dx * len * t;
      const ey = P[fv * 3 + 1]! + dy * len * t - ROOT_DIVE * t * t;
      const ez = P[fv * 3 + 2]! + dz * len * t;
      P[v * 3] = ex + (cx + nx * r * ridge - ex) * w;
      P[v * 3 + 1] = ey + (cy + ny * r * ridge - ey) * w;
      P[v * 3 + 2] = ez + (cz + nz * r * ridge - ez) * w;
      const mx = N[fv * 3]! * (1 - w) + nx * w;
      const my = N[fv * 3 + 1]! * (1 - w) + ny * w;
      const mz = N[fv * 3 + 2]! * (1 - w) + nz * w;
      const ml = Math.hypot(mx, my, mz) || 1;
      N[v * 3] = mx / ml;
      N[v * 3 + 1] = my / ml;
      N[v * 3 + 2] = mz / ml;
      uv[v * 2] = a / RADIAL;
      uv[v * 2 + 1] = t * Math.max(len, 1e-3) * 4;
      state[v * 4] = p.emissive;
      state[v * 4 + 1] = p.bleach;
      state[v * 4 + 2] = p.inert;
      state[v * 4 + 3] = depth;
    }
  }
}

/** Where each foliage pad sits and how big it is, per frame. */
export function padBlobs(sk: Skeleton, frames: Frame[], params: ArrayLike<BranchParams>, out: Float32Array): number {
  // out: per blob [x, y, z, scale]; BLOBS per pad; returns blob count
  let i = 0;
  for (const b of sk.branches) {
    if (!b.pad) continue;
    const f = frames[b.id]!;
    const p = params[b.id]!;
    const size = b.padSize * 0.36 * p.foliage * Math.min(1, p.growth * 1.5);
    const sag = b.kind === 'trunk' ? 0 : f.length * 0.1 * (1 - b.flatten * 0.4);
    const [ex, ey, ez] = [f.end[0], f.end[1] - sag, f.end[2]];
    for (let j = 0; j < BLOBS; j++) {
      // A flattened cloud: a ring of blobs plus a crown, fixed per branch id.
      const ang = j * 2.39996 + b.id * 0.61; // golden angle
      const rad = j === 0 ? 0 : 0.55 + 0.35 * ((j * 7 + b.id) % 5) / 5;
      out[i * 4] = ex + Math.cos(ang) * rad * size * 1.25;
      out[i * 4 + 1] = ey + (j === 0 ? 0.22 : 0.05 - 0.1 * (((j + b.id) % 3) / 3)) * size;
      out[i * 4 + 2] = ez + Math.sin(ang) * rad * size * 1.25;
      out[i * 4 + 3] = size * (j === 0 ? 0.95 : 0.62 + 0.12 * ((j + b.id) % 4) / 4);
      i++;
    }
  }
  return i;
}

export const BLOBS = 9;

export function padCount(sk: Skeleton): number {
  return sk.branches.filter((b) => b.pad).length;
}
