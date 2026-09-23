import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { partition } from './solution.ts';

const mixed: readonly (string | number)[] = Object.freeze(['a', 1, 'b', 2, 'c']);
const [strs, nums] = partition(mixed, (x): x is string => typeof x === 'string');
type _s = Expect<Equal<typeof strs, string[]>>;
type _n = Expect<Equal<typeof nums, number[]>>;
assert.deepEqual(strs, ['a', 'b', 'c']);
assert.deepEqual(nums, [1, 2]);
assert.deepEqual(mixed, ['a', 1, 'b', 2, 'c']);

type Ev = { kind: 'click'; x: number } | { kind: 'key'; code: string } | { kind: 'scroll'; dy: number };
const evs: Ev[] = [{ kind: 'click', x: 1 }, { kind: 'key', code: 'A' }, { kind: 'scroll', dy: 3 }, { kind: 'click', x: 2 }];
const isClick = (e: Ev): e is Extract<Ev, { kind: 'click' }> => e.kind === 'click';
const [clicks, rest] = partition(evs, isClick);
type _c = Expect<Equal<typeof clicks, { kind: 'click'; x: number }[]>>;
type _r = Expect<Equal<typeof rest, ({ kind: 'key'; code: string } | { kind: 'scroll'; dy: number })[]>>;
assert.deepEqual(clicks.map((c) => c.x), [1, 2]);
assert.deepEqual(rest.map((e) => e.kind), ['key', 'scroll']);

const [big, small] = partition([1, 5, 10, 3], (x: number) => x > 4);
type _b = Expect<Equal<typeof big, number[]>>;
type _sm = Expect<Equal<typeof small, number[]>>;
assert.deepEqual(big, [5, 10]);
assert.deepEqual(small, [1, 3]);

const [even, odd] = partition(['a', 'b', 'c', 'd', 'e'], (_: string, i: number) => i % 2 === 0);
assert.deepEqual(even, ['a', 'c', 'e']);
assert.deepEqual(odd, ['b', 'd']);

const [none, all] = partition([] as number[], () => true);
assert.deepEqual(none, []);
assert.deepEqual(all, []);
const [a1, a2] = partition(mixed, () => true);
assert.notEqual(a1, mixed as unknown);
assert.deepEqual(a2, []);

function compileOnly(): void {
  // @ts-expect-error -- the predicate must accept the element type
  partition([1, 2], (x: string) => x.length > 0);
}
void compileOnly;
