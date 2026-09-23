// Counts in a JS closure variable; the shader variable is never touched, so a simulation reading hitCount sees 0.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const hitCount = tgpu.privateVar(d.u32, 0);
let hits = 0;

export function recordHits(samples: number[], threshold: number): void {
  for (const s of samples) if (s > threshold) hits++;
}

export function countHitsOnCpu(samples: number[], threshold: number): number {
  hits = 0;
  recordHits(samples, threshold);
  return hits;
}
