// Skeleton + params -> world-space frames. Pure (trig lives here, which is
// why the skeleton itself does not). Used by the scene to rebuild the merged
// bark mesh while springs settle, and by tests to check topology stability.
import type { BranchParams } from './mapping';
import type { Skeleton } from './skeleton';

export type Vec3 = [number, number, number];

export type Frame = {
  start: Vec3;
  end: Vec3;
  /** unit axis */
  dir: Vec3;
  /** unit vector perpendicular to dir, used to orient rings */
  side: Vec3;
  r0: number;
  r1: number;
  length: number;
};

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** Rodrigues: rotate v about unit axis k by angle a. */
function rotate(v: Vec3, k: Vec3, a: number): Vec3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
}

const UP: Vec3 = [0, 1, 0];

export const TRUNK_BASE: Vec3 = [0, 0, 0];

export function poseFrames(sk: Skeleton, params: ArrayLike<BranchParams>): Frame[] {
  const frames: Frame[] = [];
  for (const b of sk.branches) {
    const p = params[b.id]!;
    let pDir: Vec3 = UP;
    let pSide: Vec3 = [1, 0, 0];
    let start: Vec3 = TRUNK_BASE;
    let pR = 0;
    if (b.parent >= 0) {
      const f = frames[b.parent]!;
      pDir = f.dir;
      pSide = f.side;
      start = add(f.start, mul(f.dir, f.length * b.attach));
      pR = f.r0 + (f.r1 - f.r0) * b.attach;
    }
    // Tilt away from the parent axis by `pitch`, around the parent's side
    // vector spun by `yaw` about the parent axis.
    const axis = rotate(pSide, pDir, b.yaw);
    let dir = rotate(pDir, axis, b.pitch);
    // Bonsai branches level out: flatten pulls the direction toward the
    // horizontal plane, keeping its compass heading.
    if (b.flatten > 0) {
      dir = norm([dir[0], dir[1] * (1 - 0.85 * b.flatten) + 0.04, dir[2]]);
    }
    dir = norm(dir);
    let side = norm(cross(dir, Math.abs(dir[1]) > 0.95 ? [1, 0, 0] : UP));
    if (!Number.isFinite(side[0])) side = [1, 0, 0];

    const length = Math.max(0, b.length * p.growth);
    // A child never out-thicks the point of its parent it grows from.
    let r0 = b.radius * p.girth;
    if (pR > 0) r0 = Math.min(r0, pR * 0.92);
    // Retracting branches thin as they shorten and reach a point at zero:
    // retracted, never deleted (§4).
    r0 *= Math.min(1, p.growth * 3);
    const r1 = r0 * b.taper;
    frames.push({ start, end: add(start, mul(dir, length)), dir, side, r0, r1, length });
  }
  return frames;
}
