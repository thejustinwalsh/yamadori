import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { memoize } from './solution.ts';

class Calc {
  calls = 0;
  constructor(readonly factor: number) {}

  @memoize
  times(x: number): number {
    this.calls++;
    return x * this.factor;
  }

  @memoize
  describe(o: { id: number } | undefined): string {
    this.calls++;
    return `#${o?.id}`;
  }
}

type _sig = Expect<Equal<Parameters<Calc['times']>, [x: number]>>;
type _ret = Expect<Equal<ReturnType<Calc['times']>, number>>;

const a = new Calc(2);
const b = new Calc(3);
assert.equal(a.times(5), 10);
assert.equal(a.times(5), 10);
assert.equal(a.calls, 1, 'second call with the same argument is cached');
assert.equal(a.times(6), 12);
assert.equal(a.calls, 2);
assert.equal(b.times(5), 15, 'instances do not share a cache');
assert.equal(b.calls, 1);

assert.equal(a.times(Number.NaN), Number.NaN);
assert.equal(a.times(Number.NaN), Number.NaN);
assert.equal(a.calls, 3, 'NaN is one key (SameValueZero)');

const o1 = { id: 1 };
const o2 = { id: 1 };
const c = new Calc(1);
assert.equal(c.describe(o1), '#1');
assert.equal(c.describe(o1), '#1');
assert.equal(c.calls, 1, 'same object -> cached');
assert.equal(c.describe(o2), '#1');
assert.equal(c.calls, 2, 'a different object with equal contents is a different key');
assert.equal(c.describe(undefined), '#undefined');
assert.equal(c.describe(undefined), '#undefined');
assert.equal(c.calls, 3, 'undefined is a key too');

// A cached value computed with one instance's `this` is never served to another instance.
const d = new Calc(10);
assert.equal(d.times(4), 40);
assert.equal(d.times(4), 40);
assert.equal(d.calls, 1);
assert.equal(a.times(4), 8);

class Misuse {
  // @ts-expect-error -- memoize only applies to single-argument methods
  @memoize
  add(x: number, y: number): number {
    return x + y;
  }
}
void Misuse;
