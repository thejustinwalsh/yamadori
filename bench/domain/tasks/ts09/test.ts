import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { partial } from './solution.ts';

const fmt = (n: number, unit: string, upper: boolean): string => {
  const s = `${n}${unit}`;
  return upper ? s.toUpperCase() : s;
};

const withN = partial(fmt, 3);
type _p1 = Expect<Equal<Parameters<typeof withN>, [unit: string, upper: boolean]>>;
type _r1 = Expect<Equal<ReturnType<typeof withN>, string>>;
assert.equal(withN('kb', true), '3KB');
assert.equal(withN('mb', false), '3mb');

const withTwo = partial(fmt, 7, 'px');
type _p2 = Expect<Equal<Parameters<typeof withTwo>, [upper: boolean]>>;
assert.equal(withTwo(false), '7px');

const all = partial(fmt, 1, 'x', true);
type _p3 = Expect<Equal<Parameters<typeof all>, []>>;
assert.equal(all(), '1X');

const none = partial(fmt);
type _p4 = Expect<Equal<Parameters<typeof none>, [n: number, unit: string, upper: boolean]>>;
assert.equal(none(2, 'g', false), '2g');

const calls: unknown[][] = [];
const rec = (a: string, b: number, c: { k: string }) => { calls.push([a, b, c]); return b * 2; };
const obj = { k: 'v' };
const recA = partial(rec, 'a');
type _r2 = Expect<Equal<ReturnType<typeof recA>, number>>;
assert.equal(recA(4, obj), 8);
assert.equal(recA(5, obj), 10);
assert.deepEqual(calls, [['a', 4, obj], ['a', 5, obj]]);
assert.equal(calls[0]![2], obj);

function compileOnly(): void {
  // @ts-expect-error -- first argument of fmt is a number
  partial(fmt, 'x');
  // @ts-expect-error -- too few remaining arguments
  withN('kb');
  // @ts-expect-error -- wrong type for a remaining argument
  withTwo('yes');
  // @ts-expect-error -- too many pre-supplied arguments
  partial(fmt, 1, 'x', true, 4);
}
void compileOnly;
