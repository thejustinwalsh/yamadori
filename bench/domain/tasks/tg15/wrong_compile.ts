// Calls tgpu.simulate as the JSDoc example writes it; in 0.12.5 simulate lives only under tgpu['~unstable'].
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const hitCount = tgpu.privateVar(d.u32, 0);

export function recordHits(samples: number[], threshold: number): void {
  for (const s of samples) if (s > threshold) hitCount.$ += 1;
}

export function countHitsOnCpu(samples: number[], threshold: number): number {
  return tgpu.simulate(() => {
    recordHits(samples, threshold);
    return hitCount.$;
  }).value;
}
