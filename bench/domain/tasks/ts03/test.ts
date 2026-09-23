import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { assertIsDefined } from './solution.ts';

function narrowsNull(x: string | null): string {
  assertIsDefined(x);
  type _n = Expect<Equal<typeof x, string>>;
  return x;
}
function narrowsBoth(x: number | null | undefined): number {
  assertIsDefined(x, 'x is required');
  type _n = Expect<Equal<typeof x, number>>;
  return x;
}
function narrowsProperty(o: { id?: number }): number {
  assertIsDefined(o.id);
  return o.id;
}
function keepsUnion(x: 'a' | 'b' | undefined): 'a' | 'b' {
  assertIsDefined(x);
  type _n = Expect<Equal<typeof x, 'a' | 'b'>>;
  return x;
}

for (const ok of [0, '', false, Number.NaN, [], {}, 0n]) {
  assert.doesNotThrow(() => assertIsDefined(ok), `defined value ${String(ok)} must pass`);
}
assert.equal(narrowsNull('s'), 's');
assert.equal(narrowsBoth(0), 0);
assert.equal(narrowsProperty({ id: 7 }), 7);
assert.equal(keepsUnion('b'), 'b');

assert.equal(assertIsDefined(1), undefined);
assert.throws(() => assertIsDefined(null), TypeError);
assert.throws(() => assertIsDefined(undefined), TypeError);
assert.throws(() => narrowsBoth(undefined), { name: 'TypeError', message: 'x is required' });
assert.throws(() => narrowsProperty({}), TypeError);
