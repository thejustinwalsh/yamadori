import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { AsyncQueue } from './solution.ts';

type Outcome = { settled: true; value: unknown } | { settled: false };
// How `p` settles within `ms`; the grader's timer is cleared as soon as it settles.
function within(p: Promise<unknown>, ms = 2000): Promise<Outcome> {
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve({ settled: false }), ms);
    p.then((value) => { clearTimeout(t); resolve({ settled: true, value }); },
           (value) => { clearTimeout(t); resolve({ settled: true, value }); });
  });
}
const tick = () => new Promise<void>((r) => setTimeout(r, 5));

// Buffered items come out in FIFO order; size counts unconsumed items.
const q = new AsyncQueue<number>();
q.push(1);
q.push(2);
assert.equal(q.size, 2);
const it = q[Symbol.asyncIterator]();
assert.deepEqual(await it.next(), { done: false, value: 1 });
assert.equal(q.size, 1);

// Consumers wait for items, and several pending next() calls are served in call order.
const q2 = new AsyncQueue<string>();
const it2 = q2[Symbol.asyncIterator]();
const n1 = it2.next();
const n2 = it2.next();
await tick();
q2.push('a');
q2.push('b');
assert.deepEqual(await within(Promise.all([n1, n2])), {
  settled: true,
  value: [{ done: false, value: 'a' }, { done: false, value: 'b' }],
}, 'both pending next() calls must resolve, in order');

// close(): remaining items are still delivered, then iteration ends; push after close throws.
const q3 = new AsyncQueue<string>();
q3.push('x');
q3.push('y');
q3.close();
assert.throws(() => q3.push('z'));
const seen: string[] = [];
const loop = (async () => {
  for await (const s of q3) {
    type _s = Expect<Equal<typeof s, string>>;
    seen.push(s);
  }
})();
assert.deepEqual(await within(loop), { settled: true, value: undefined });
assert.deepEqual(seen, ['x', 'y'], 'items pushed before close are not lost');

// A consumer waiting on an empty queue is released by close().
const q4 = new AsyncQueue<number>();
const got: number[] = [];
const consumer = (async () => { for await (const n of q4) got.push(n); return 'finished'; })();
await tick();
q4.push(7);
await tick();
q4.close();
assert.deepEqual(await within(consumer), { settled: true, value: 'finished' }, 'close() must end a waiting for-await loop');
assert.deepEqual(got, [7]);

// Producer and consumer interleaved.
const q5 = new AsyncQueue<number>();
const total = (async () => { let s = 0; for await (const n of q5) s += n; return s; })();
for (let i = 1; i <= 5; i++) { q5.push(i); if (i % 2) await tick(); }
q5.close();
assert.deepEqual(await within(total), { settled: true, value: 15 });
