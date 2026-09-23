// Uses `.value` to access the variable (older typegpu); 0.12.5 variables expose their in-shader value as `.$`.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const hitCount = tgpu.privateVar(d.u32, 0);

export function recordHits(samples: number[], threshold: number): void {
  for (const s of samples) if (s > threshold) hitCount.value += 1;
}

export function countHitsOnCpu(samples: number[], threshold: number): number {
  return tgpu['~unstable'].simulate(() => {
    recordHits(samples, threshold);
    return hitCount.value;
  }).value;
}
