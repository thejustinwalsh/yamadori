import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { snapshot } from './solution.ts';

type State = {
  when: Date;
  pattern: RegExp;
  tags: Set<string>;
  scores: Map<string, number[]>;
  bytes: Uint8Array;
  nested: { list: { id: number }[] };
  self?: State;
};

const shared = { id: 1 };
const original: State = {
  when: new Date('2026-09-22T10:00:00Z'),
  pattern: /ab+c/gi,
  tags: new Set(['a', 'b']),
  scores: new Map([['ann', [1, 2]]]),
  bytes: new Uint8Array([1, 2, 3]),
  nested: { list: [shared, shared, { id: 2 }] },
};
original.self = original;

const copy = snapshot(original);
type _t = Expect<Equal<typeof copy, State>>;

assert.notEqual(copy, original);
assert.ok(copy.when instanceof Date, 'Date stays a Date');
assert.equal(copy.when.getTime(), original.when.getTime());
assert.notEqual(copy.when, original.when);
assert.ok(copy.pattern instanceof RegExp);
assert.equal(copy.pattern.source, 'ab+c');
assert.equal(copy.pattern.flags, 'gi');
assert.ok(copy.tags instanceof Set);
assert.deepEqual([...copy.tags], ['a', 'b']);
assert.ok(copy.scores instanceof Map);
assert.deepEqual(copy.scores.get('ann'), [1, 2]);
assert.notEqual(copy.scores.get('ann'), original.scores.get('ann'));
assert.ok(copy.bytes instanceof Uint8Array);
assert.deepEqual([...copy.bytes], [1, 2, 3]);

assert.equal(copy.self, copy, 'cycles are preserved, pointing into the copy');
assert.equal(copy.nested.list[0], copy.nested.list[1], 'shared references stay shared');
assert.notEqual(copy.nested.list[0], shared);

// Mutating the original afterwards does not affect the snapshot (and vice versa).
original.when.setFullYear(2000);
original.tags.add('c');
original.scores.get('ann')!.push(3);
original.bytes[0] = 99;
shared.id = 42;
assert.equal(copy.when.getUTCFullYear(), 2026);
assert.deepEqual([...copy.tags], ['a', 'b']);
assert.deepEqual(copy.scores.get('ann'), [1, 2]);
assert.equal(copy.bytes[0], 1);
assert.equal(copy.nested.list[0]!.id, 1);
copy.nested.list.push({ id: 3 });
assert.equal(original.nested.list.length, 3);

assert.equal(snapshot(5), 5);
assert.equal(snapshot(null), null);
const arr = snapshot([1, [2, [3]]] as const);
assert.deepEqual(arr, [1, [2, [3]]]);

// Functions cannot be snapshotted.
assert.throws(() => snapshot({ f() { return 1; } }));
