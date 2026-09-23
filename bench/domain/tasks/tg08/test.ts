import assert from 'node:assert/strict';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { greenChannelRange, Instance, Instances } from './solution.ts';

type R = ReturnType<typeof greenChannelRange>;
type _r = Expect<Equal<{ [K in keyof R]: R[K] }, { offset: number; contiguous: number }>>;
type _i = Expect<Equal<d.Infer<typeof Instance>, { transform: d.m4x4f; color: d.v3f; id: number; velocity: d.v3f }>>;
export const _neg = () => {
  // @ts-expect-error -- index is a number
  greenChannelRange('3');
};

assert.equal(d.sizeOf(Instance), 96);
assert.equal(d.sizeOf(Instances), 96 * 64);

for (const [i, offset] of [[0, 68], [1, 164], [2, 260], [63, 6116]] as const) {
  const r = greenChannelRange(i);
  assert.equal(r.offset, offset, `offset of Instances[${i}].color.y`);
  assert.equal(r.contiguous, 24, `contiguous bytes from Instances[${i}].color.y (color.y, color.z, id, velocity; then 4 bytes padding)`);
}
