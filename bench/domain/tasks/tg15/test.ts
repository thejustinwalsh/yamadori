import assert from 'node:assert/strict';
import tgpu, { type TgpuVar } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { countHitsOnCpu, hitCount, recordHits } from './solution.ts';

type _v = Expect<Equal<typeof hitCount, TgpuVar<'private', d.U32>>>;
type _c = Expect<Equal<ReturnType<typeof countHitsOnCpu>, number>>;
export const _neg = () => {
  // @ts-expect-error -- a private variable is not workgroup-scoped
  const _w: TgpuVar<'workgroup'> = hitCount;
};

const samples = [0.1, 0.9, 0.5, 0.7, 0.6, 0.61];
assert.equal(countHitsOnCpu(samples, 0.6), 3, 'strictly greater than the threshold');
assert.equal(countHitsOnCpu(samples, 0.6), 3, 'each call starts from 0');
assert.equal(countHitsOnCpu([], 0), 0);
assert.equal(countHitsOnCpu([5, 6, 7], 0), 3);

// recordHits must drive the shader variable itself: run it in our own simulation.
const sim = tgpu['~unstable'].simulate(() => {
  hitCount.$ = 0;
  recordHits([1, 2, 3, 4], 2.5);
  return hitCount.$;
});
assert.equal(sim.value, 2, 'recordHits increments hitCount.$ inside a simulation');

// ...and outside a simulation the variable is not accessible, so recordHits must fail there.
assert.throws(() => recordHits([1], 0), /inaccessible|simulat|variable/i, 'recordHits outside a simulation touches the GPU variable');
