import { describe, expect, it } from 'vitest';
import type { Vitals } from '../api/types';
import liveVitals from './__fixtures__/vitals.live.json';
import { allocateBark, FOOT_SINK, rootCells, writeBark } from './bark';
import { treeParams, treeState } from './mapping';
import { poseFrames } from './pose';
import { fnv1a32 } from './prng';
import { growSkeleton, limbSeeds } from './skeleton';

// The trunk rendered inside-out: triangles were wound (i0, i2, i1), which
// faces INWARD for a ring that sweeps counter-clockwise around the branch
// axis. Back-face culling then drew the inside of the far wall under outward
// vertex normals. Every triangle must face the way its vertex normals point.
describe('bark winding', () => {
  for (const word of ['harbor', 'glacier', 'mettendo', '盆栽', '']) {
    it(`every bark triangle faces outward (${JSON.stringify(word)})`, () => {
      const sk = growSkeleton(limbSeeds(fnv1a32(word)));
      const p = treeParams(treeState(liveVitals as unknown as Vitals), sk);
      const buf = allocateBark(sk);
      writeBark(buf, sk, poseFrames(sk, p.branches), p.branches);
      const P = buf.position;
      const N = buf.normal;
      let inward = 0;
      let checked = 0;
      let underside = 0;
      const roots = new Set(sk.branches.filter((b) => b.kind === 'root').map((b) => b.id));
      for (let t = 0; t < buf.index.length; t += 3) {
        const [a, b, c] = [buf.index[t]!, buf.index[t + 1]!, buf.index[t + 2]!];
        const e1 = [P[b * 3]! - P[a * 3]!, P[b * 3 + 1]! - P[a * 3 + 1]!, P[b * 3 + 2]! - P[a * 3 + 2]!];
        const e2 = [P[c * 3]! - P[a * 3]!, P[c * 3 + 1]! - P[a * 3 + 1]!, P[c * 3 + 2]! - P[a * 3 + 2]!];
        const fn = [e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!];
        const area = Math.hypot(fn[0]!, fn[1]!, fn[2]!);
        if (area < 1e-9) continue; // degenerate sliver at a zero-radius tip
        const vn = [0, 1, 2].map((k) => N[a * 3 + k]! + N[b * 3 + k]! + N[c * 3 + k]!);
        if (fn[0]! * vn[0]! + fn[1]! * vn[1]! + fn[2]! * vn[2]! < 0) {
          // A root's underside, in the first two rings of the weld, folds
          // against the flare it grows from: its faces point DOWN, at the
          // slab, while its first ring carries the flare's upward normals.
          // The camera never goes below the slab (maxPolarAngle < pi/2), so
          // these are always culled back faces and their normals are never
          // shaded. Anything else facing against its normals is a bug.
          const ring = Math.floor((a % 77) / 11);
          if (roots.has(Math.floor(a / 77)) && ring <= 1 && fn[1]! / area < -0.3) underside++;
          else inward++;
        }
        checked++;
      }
      expect(checked).toBeGreaterThan(1000);
      expect(inward).toBe(0);
      expect(underside).toBeLessThanOrEqual(roots.size * 4);
    });
  }
});

// The nebari split: roots were separate tubes that started at the trunk's
// axis and pierced the flare, so from most orbits a crack showed where the
// two surfaces crossed. Each root now begins ON the trunk: its first ring is
// made of trunk-flare vertices, and its normals there are the trunk's own.
describe('nebari weld', () => {
  const RING = 11; // RADIAL + 1
  const VPB = RING * 7;
  for (const word of ['harbor', 'glacier', 'mettendo', '盆栽', '', 'veniva']) {
    it(`every root starts on the trunk, with the trunk's normals (${JSON.stringify(word)})`, () => {
      const sk = growSkeleton(limbSeeds(fnv1a32(word)));
      const p = treeParams(treeState(liveVitals as unknown as Vitals), sk);
      const buf = allocateBark(sk);
      writeBark(buf, sk, poseFrames(sk, p.branches), p.branches);
      const P = buf.position;
      const N = buf.normal;
      const trunk = sk.branches.find((b) => b.kind === 'trunk' && b.parent === -1)!;
      const roots = sk.branches.filter((b) => b.kind === 'root');
      expect(roots.length).toBeGreaterThan(0);
      let maxGap = 0;
      let minDot = 1;
      let maxBend = 0;
      for (const r of roots) {
        for (let a = 0; a < RING; a++) {
          const v = r.id * VPB + a;
          let best = Infinity;
          let bestT = -1;
          for (let t = trunk.id * VPB; t < (trunk.id + 1) * VPB; t++) {
            const d = Math.hypot(P[v * 3]! - P[t * 3]!, P[v * 3 + 1]! - P[t * 3 + 1]!, P[v * 3 + 2]! - P[t * 3 + 2]!);
            if (d < best) [best, bestT] = [d, t];
          }
          maxGap = Math.max(maxGap, best);
          const dot = N[v * 3]! * N[bestT * 3]! + N[v * 3 + 1]! * N[bestT * 3 + 1]! + N[v * 3 + 2]! * N[bestT * 3 + 2]!;
          minDot = Math.min(minDot, dot);
          // No crease one ring out: the root's second ring turns its
          // normals away from the first by a bounded angle.
          const w = v + RING;
          const bend = N[v * 3]! * N[w * 3]! + N[v * 3 + 1]! * N[w * 3 + 1]! + N[v * 3 + 2]! * N[w * 3 + 2]!;
          maxBend = Math.max(maxBend, Math.acos(Math.max(-1, Math.min(1, bend))));
        }
      }
      // eslint-disable-next-line no-console
      console.log(`nebari ${JSON.stringify(word)}: max gap ${maxGap.toExponential(2)}, min normal dot ${minDot.toFixed(4)}, max ring-1 bend ${((maxBend * 180) / Math.PI).toFixed(1)} deg`);
      expect(maxGap).toBeLessThan(1e-5);
      expect(minDot).toBeGreaterThan(0.999);
      expect(maxBend).toBeLessThan((75 * Math.PI) / 180);
    });
  }

  it('holds while the tree is still growing, and the foot sits level in the slab', () => {
    const sk = growSkeleton(limbSeeds(fnv1a32('harbor')));
    const full = treeParams(treeState(liveVitals as unknown as Vitals), sk).branches;
    const half = full.map((b) => ({ ...b, growth: b.growth * 0.3 }));
    const buf = allocateBark(sk);
    writeBark(buf, sk, poseFrames(sk, half), half);
    const P = buf.position;
    const trunk = sk.branches.find((b) => b.kind === 'trunk' && b.parent === -1)!;
    // The first trunk segment leans; its foot ring must not lift off the slab.
    for (let a = 0; a < RING; a++) expect(P[(trunk.id * VPB + a) * 3 + 1]!).toBeCloseTo(-FOOT_SINK, 6);
    for (const r of sk.branches.filter((b) => b.kind === 'root')) {
      for (let a = 0; a < RING; a++) {
        const v = r.id * VPB + a;
        let best = Infinity;
        for (let t = trunk.id * VPB; t < (trunk.id + 1) * VPB; t++) {
          best = Math.min(best, Math.hypot(P[v * 3]! - P[t * 3]!, P[v * 3 + 1]! - P[t * 3 + 1]!, P[v * 3 + 2]! - P[t * 3 + 2]!));
        }
        expect(best).toBeLessThan(1e-5);
      }
    }
  });

  it('welds every trunk joint: the child starts on the parent ring, untwisted', () => {
    for (const word of ['harbor', 'glacier', 'mettendo', '盆栽', '', 'veniva']) {
      const sk = growSkeleton(limbSeeds(fnv1a32(word)));
      const p = treeParams(treeState(liveVitals as unknown as Vitals), sk).branches;
      const buf = allocateBark(sk);
      writeBark(buf, sk, poseFrames(sk, p), p);
      const P = buf.position;
      const d = (i: number, j: number) => Math.hypot(P[i * 3]! - P[j * 3]!, P[i * 3 + 1]! - P[j * 3 + 1]!, P[i * 3 + 2]! - P[j * 3 + 2]!);
      let joints = 0;
      let maxGap = 0;
      let twisted = 0;
      for (const b of sk.branches) {
        if (b.kind !== 'trunk' || b.parent < 0 || sk.branches[b.parent]!.kind !== 'trunk') continue;
        joints++;
        for (let a = 0; a < RING; a++) {
          const v = b.id * VPB + a;
          const w = b.parent * VPB + 6 * RING + a;
          maxGap = Math.max(maxGap, d(v, w));
          // untwisted: ring-1 vertex a is the nearest ring-1 vertex to ring-0 vertex a
          let best = -1;
          let bestD = Infinity;
          for (let q = 0; q < RING - 1; q++) {
            const dd = d(v, b.id * VPB + RING + q);
            if (dd < bestD) [best, bestD] = [q, dd];
          }
          if (best !== a % (RING - 1)) twisted++;
        }
      }
      expect(joints).toBe(4);
      expect(maxGap).toBeLessThan(1e-6);
      expect(twisted).toBe(0);
    }
  });

  it('gives every root its own flare cell', () => {
    for (const word of ['harbor', 'glacier', 'mettendo', '盆栽', '', 'veniva']) {
      const sk = growSkeleton(limbSeeds(fnv1a32(word)));
      const p = treeParams(treeState(liveVitals as unknown as Vitals), sk).branches;
      const frames = poseFrames(sk, p);
      const trunk = sk.branches.find((b) => b.kind === 'trunk' && b.parent === -1)!;
      const roots = sk.branches.filter((b) => b.kind === 'root');
      const cells = rootCells(roots, frames, frames[trunk.id]!);
      expect(new Set(cells.values()).size).toBe(roots.length);
    }
  });
});

// Dormant roots on a live trunk: the root's inert value (aState.z) used to
// start at 1 on its first ring, so the shader's inert look began exactly at
// the flare and drew a seam round the base. The weld now carries the trunk's
// value out and reaches the root's own at the end of the morph.
describe('nebari weld: inert blends from the trunk', () => {
  it('ring 0 matches the trunk, the far rings are the root, and it never jumps back', () => {
    const { tree: _t, ...noTree } = liveVitals as unknown as Vitals;
    const sk = growSkeleton(limbSeeds(fnv1a32('harbor')));
    const p = treeParams(treeState(noTree as Vitals), sk);
    const buf = allocateBark(sk);
    writeBark(buf, sk, poseFrames(sk, p.branches), p.branches);
    const VPB = buf.position.length / 3 / sk.branches.length;
    const RING = 11;
    const rings = VPB / RING;
    const trunk = sk.branches.find((b) => b.kind === 'trunk' && b.parent === -1)!;
    expect(p.branches[trunk.id]!.inert).toBe(0);
    for (const r of sk.branches.filter((b) => b.kind === 'root')) {
      expect(p.branches[r.id]!.inert).toBe(1);
      const z = (ring: number) => buf.state[(r.id * VPB + ring * RING + 3) * 4 + 2]!;
      expect(z(0)).toBeCloseTo(0, 6);
      expect(z(rings - 1)).toBeCloseTo(1, 6);
      for (let k = 1; k < rings; k++) expect(z(k)).toBeGreaterThanOrEqual(z(k - 1) - 1e-9);
    }
  });
});
