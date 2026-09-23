import { tgpu, d } from 'typegpu';

export const hitCount = tgpu.privateVar(d.u32, 0);

export const recordHits = (samples: number[], threshold: number): void => {
  samples.filter((s) => s > threshold).forEach(() => {
    hitCount.$ = hitCount.$ + 1;
  });
};

// Read the final value out of the simulation's recorded private-variable state.
export const countHitsOnCpu = (samples: number[], threshold: number): number => {
  const { simulate } = tgpu['~unstable'];
  const result = simulate(() => {
    hitCount.$ = 0;
    recordHits(samples, threshold);
  });
  return result.privateVars[0]![0]![0]!.get(hitCount) as number;
};
