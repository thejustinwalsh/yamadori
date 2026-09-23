// Critically damped springs, per channel (design/BONSAI-VIZ.md §4): throughput
// arcs snap, trunk girth crawls. Pure step function, so settling is testable
// and the renderer knows exactly when to stop drawing (frameloop="demand").
import type { BranchParams } from './mapping';

export type SpringState = { x: number; v: number };

/** Exact critically-damped step toward `target` with angular frequency w. */
export function stepSpring(s: SpringState, target: number, w: number, dt: number): SpringState {
  const d = s.x - target;
  const e = Math.exp(-w * dt);
  const x = target + (d + (s.v + w * d) * dt) * e;
  const v = (s.v - (s.v + w * d) * w * dt) * e;
  return { x, v };
}

/** Per-channel stiffness (rad/s). Higher snaps, lower crawls. */
export const STIFFNESS: Record<keyof BranchParams, number> = {
  growth: 1.2,
  girth: 0.8,
  foliage: 1.5,
  foliageLive: 2,
  emissive: 7,
  bleach: 1.0,
  cap: 5,
  inert: 2,
};

const KEYS = Object.keys(STIFFNESS) as (keyof BranchParams)[];
const EPS = 1e-3;

/**
 * Advance every channel of every branch one step, in place on flat arrays
 * (value, velocity), toward `targets`. Returns true while anything moves.
 */
export class ParamSprings {
  readonly n: number;
  readonly x: Float32Array;
  readonly v: Float32Array;

  constructor(initial: BranchParams[]) {
    this.n = initial.length;
    this.x = new Float32Array(this.n * KEYS.length);
    this.v = new Float32Array(this.n * KEYS.length);
    initial.forEach((p, i) => KEYS.forEach((k, j) => (this.x[i * KEYS.length + j] = p[k])));
  }

  step(targets: BranchParams[], dt: number): boolean {
    let moving = false;
    const h = Math.min(dt, 1 / 20); // a long stall must not overshoot
    for (let i = 0; i < this.n; i++) {
      const t = targets[i];
      if (!t) continue;
      for (let j = 0; j < KEYS.length; j++) {
        const k = KEYS[j]!;
        const idx = i * KEYS.length + j;
        const s = stepSpring({ x: this.x[idx]!, v: this.v[idx]! }, t[k], STIFFNESS[k], h);
        if (Math.abs(s.x - t[k]) < EPS && Math.abs(s.v) < EPS) {
          this.x[idx] = t[k];
          this.v[idx] = 0;
        } else {
          this.x[idx] = s.x;
          this.v[idx] = s.v;
          moving = true;
        }
      }
    }
    return moving;
  }

  get(i: number): BranchParams {
    const o = i * KEYS.length;
    const out = {} as BranchParams;
    KEYS.forEach((k, j) => (out[k] = this.x[o + j]!));
    return out;
  }
}
