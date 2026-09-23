import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { zip } from './solution.ts';

type Elem<I> = I extends Iterable<infer X> ? X : never;

const z = zip([1, 2, 3], ['a', 'b', 'c'], new Set([true, false]));
type _z = Expect<Equal<Elem<typeof z>, [number, string, boolean]>>;
assert.deepEqual([...z], [[1, 'a', true], [2, 'b', false]]);

const m = zip(new Map([['k', 1]]), 'xy');
type _m = Expect<Equal<Elem<typeof m>, [[string, number], string]>>;
assert.deepEqual([...m], [[['k', 1], 'x']]);

const single = zip([10, 20]);
type _s = Expect<Equal<Elem<typeof single>, [number]>>;
assert.deepEqual([...single], [[10], [20]]);

assert.deepEqual([...zip()], []);
assert.deepEqual([...zip([], [1, 2])], []);

const ro: readonly string[] = ['r'];
const withRo = zip(ro, [1]);
type _ro = Expect<Equal<Elem<typeof withRo>, [string, number]>>;

// The result is an iterator as well as an iterable.
const it = zip([1], [2]);
assert.deepEqual(it.next(), { done: false, value: [1, 2] });
assert.equal(it.next().done, true);

// Lazy: works on infinite inputs, and pulls only what it needs.
let pulled = 0;
function* naturals() {
  for (let n = 0; ; n++) { pulled++; yield n; }
}
const lazy = zip(naturals(), ['a', 'b']);
assert.equal(pulled, 0, 'nothing is pulled before iteration starts');
assert.deepEqual([...lazy], [[0, 'a'], [1, 'b']]);
assert.ok(pulled <= 3, `pulled ${pulled} values from an infinite source`);

// When one input ends, the others that are still open are closed (their `finally` runs).
const closed: string[] = [];
function* tracked(name: string, n: number) {
  try {
    for (let i = 0; i < n; i++) yield `${name}${i}`;
  } finally {
    closed.push(name);
  }
}
assert.deepEqual([...zip(tracked('long', 10), tracked('short', 1))], [['long0', 'short0']]);
assert.ok(closed.includes('long'), 'the unfinished input must be closed');

// When the consumer stops early, every input is closed.
closed.length = 0;
for (const [a] of zip(tracked('p', 5), tracked('q', 5))) {
  assert.equal(a, 'p0');
  break;
}
assert.deepEqual(closed.toSorted(), ['p', 'q']);

function compileOnly(): void {
  // @ts-expect-error -- rows are exact tuples
  const bad: [string, string] = [...zip([1], ['a'])][0]!;
  // @ts-expect-error -- only iterables are accepted
  zip([1], 5);
  void bad;
}
void compileOnly;
