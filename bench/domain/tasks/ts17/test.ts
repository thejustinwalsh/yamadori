import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { createPool } from './solution.ts';

const flush = () => new Promise<void>((r) => setTimeout(r, 5));
const keepAlive = setInterval(() => {}, 1000); // pending acquires hold no timers of their own

const pool = createPool(['a', 'b']);
assert.equal(pool.available, 2);
assert.equal(pool.waiting, 0);

const events: string[] = [];
let w1: Promise<{ value: string } & AsyncDisposable> | undefined;
let w2: typeof w1;
let first = '';
let second = '';
{
  await using l1 = await pool.acquire();
  await using l2 = await pool.acquire();
  type _v = Expect<Equal<typeof l1.value, string>>;
  first = l1.value;
  second = l2.value;
  assert.deepEqual(new Set([first, second]), new Set(['a', 'b']), 'two distinct items are leased');
  assert.equal(pool.available, 0);

  w1 = pool.acquire();
  w2 = pool.acquire();
  w1.then((l) => events.push(`w1:${l.value}`));
  w2.then((l) => events.push(`w2:${l.value}`));
  await flush();
  assert.equal(pool.waiting, 2);
  assert.deepEqual(events, [], 'acquire waits while the pool is exhausted');
}
// Leaving the block disposes l2 then l1; the waiters are served first-come-first-served.
await flush();
assert.deepEqual(events, [`w1:${second}`, `w2:${first}`]);
assert.equal(pool.waiting, 0);
assert.equal(pool.available, 0, 'released items go straight to waiters');

const lw1 = await w1!;
const lw2 = await w2!;
await lw1[Symbol.asyncDispose]();
assert.equal(pool.available, 1);
await lw1[Symbol.asyncDispose]();
assert.equal(pool.available, 1, 'disposing a lease twice releases once');
await lw2[Symbol.asyncDispose]();
assert.equal(pool.available, 2);

// A stale lease must not release an item that has since been re-leased.
const small = createPool([1]);
const x = await small.acquire();
await x[Symbol.asyncDispose]();
const y = await small.acquire();
type _n = Expect<Equal<typeof y.value, number>>;
await x[Symbol.asyncDispose]();
assert.equal(small.available, 0, 'a second dispose of an old lease freed an item that is in use');
let got = false;
const z = small.acquire().then((l) => { got = true; return l; });
await flush();
assert.equal(got, false);
await y[Symbol.asyncDispose]();
assert.equal((await z).value, 1);

// An exception inside the block still releases.
const p3 = createPool([{ id: 1 }]);
await assert.rejects(async () => {
  await using l = await p3.acquire();
  assert.equal(p3.available, 0);
  throw new Error('work failed');
}, /work failed/);
assert.equal(p3.available, 1);

clearInterval(keepAlive);
