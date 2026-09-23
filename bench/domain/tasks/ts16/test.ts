import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { openAll, Resource } from './solution.ts';

// Resource: logs open/close, disposes once, works with `using`.
const log: string[] = [];
{
  using r = new Resource('db', log);
  assert.equal(r.name, 'db');
  assert.equal(r.disposed, false);
  assert.deepEqual(log, ['open:db']);
}
assert.deepEqual(log, ['open:db', 'close:db']);
const twice = new Resource('x', log);
twice[Symbol.dispose]();
twice[Symbol.dispose]();
assert.equal(twice.disposed, true);
assert.deepEqual(log.slice(2), ['open:x', 'close:x'], 'disposing twice closes once');

// openAll: opens in order, disposes in reverse order at scope exit, even on an exception.
log.length = 0;
const open = (n: string) => new Resource(n, log);
{
  using group = openAll(['a', 'b', 'c'], open);
  type _g = Expect<Equal<typeof group extends Disposable ? true : false, true>>;
  assert.deepEqual(log, ['open:a', 'open:b', 'open:c']);
}
assert.deepEqual(log, ['open:a', 'open:b', 'open:c', 'close:c', 'close:b', 'close:a']);

log.length = 0;
assert.throws(() => {
  using g = openAll(['p', 'q'], open);
  void g;
  throw new Error('body failed');
}, /body failed/);
assert.deepEqual(log, ['open:p', 'open:q', 'close:q', 'close:p']);

// Disposing the group twice disposes each resource once.
log.length = 0;
const g2 = openAll(['m', 'n'], open);
g2[Symbol.dispose]();
g2[Symbol.dispose]();
assert.deepEqual(log, ['open:m', 'open:n', 'close:n', 'close:m']);

// A failing open: what was already opened is disposed in reverse, and the original error propagates.
log.length = 0;
const openErr = new Error('cannot open c');
assert.throws(
  () => openAll(['a', 'b', 'c', 'd'], (n) => { if (n === 'c') throw openErr; return open(n); }),
  (e) => e === openErr,
);
assert.deepEqual(log, ['open:a', 'open:b', 'close:b', 'close:a']);

// A failing disposal does not stop the others, and its error is rethrown afterwards.
log.length = 0;
const disposeErr = new Error('close failed');
const flaky = (n: string): Disposable => {
  const r = open(n);
  if (n !== 'y') return r;
  return { [Symbol.dispose]() { r[Symbol.dispose](); throw disposeErr; } };
};
const g3 = openAll(['x', 'y', 'z'], flaky);
assert.throws(() => g3[Symbol.dispose](), (e) => e === disposeErr);
assert.deepEqual(log, ['open:x', 'open:y', 'open:z', 'close:z', 'close:y', 'close:x']);

// Empty input is fine.
{ using empty = openAll([], open); void empty; }
