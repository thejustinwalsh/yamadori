import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { chunk } from './solution.ts';

const ro: readonly number[] = Object.freeze([1, 2, 3, 4, 5]);
const r = chunk(ro, 2);
type _t = Expect<Equal<typeof r, number[][]>>;
type _s = Expect<Equal<ReturnType<typeof chunk<string>>, string[][]>>;

assert.deepEqual(r, [[1, 2], [3, 4], [5]]);
assert.deepEqual(chunk([1, 2, 3, 4], 2), [[1, 2], [3, 4]]);
assert.deepEqual(chunk([], 3), []);
assert.deepEqual(chunk(['a'], 5), [['a']]);
for (const bad of [0, -1, 1.5, Number.NaN]) {
  assert.throws(() => chunk([1], bad), RangeError, `size ${bad}`);
}
