import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { withTimeout } from './solution.ts';

const never = () => new Promise<never>(() => {});

type Outcome = { status: 'fulfilled'; value: unknown } | { status: 'rejected'; reason: unknown } | { status: 'pending' };
// How `p` settles within `limit` ms. The grader's own timer is cleared as soon as `p` settles.
function outcome(p: Promise<unknown>, limit: number): Promise<Outcome> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve({ status: 'pending' }), limit);
    p.then(
      (value) => { clearTimeout(timer); resolve({ status: 'fulfilled', value }); },
      (reason) => { clearTimeout(timer); resolve({ status: 'rejected', reason }); },
    );
  });
}

// Resolves with the task's value. The long timeout must not keep the process alive afterwards:
// this script is expected to exit well within the grader's 30 s limit.
const p = withTimeout(async () => 'x', 60_000);
type _p = Expect<Equal<typeof p, Promise<string>>>;
assert.deepEqual(await outcome(p, 3000), { status: 'fulfilled', value: 'x' });

// The task's own rejection passes through unchanged.
const boom = new Error('boom');
assert.deepEqual(await outcome(withTimeout(async () => { throw boom; }, 60_000), 3000), { status: 'rejected', reason: boom });

// A synchronous throw from the task becomes a rejection, not a throw from withTimeout.
const sync = new Error('sync');
let syncResult: Promise<unknown> | undefined;
assert.doesNotThrow(() => { syncResult = withTimeout(() => { throw sync; }, 60_000); });
assert.deepEqual(await outcome(syncResult!, 3000), { status: 'rejected', reason: sync });

// Timeout: rejects promptly even though the task ignores its signal; the task's signal is aborted.
let seen: AbortSignal | undefined;
const t = await outcome(withTimeout((signal) => { seen = signal; return never(); }, 30), 3000);
assert.equal(t.status, 'rejected', 'a task that ignores its signal must still time out (within 3 s for ms=30)');
assert.equal((t as { reason: Error }).reason.name, 'TimeoutError');
assert.ok(seen instanceof AbortSignal, 'task receives an AbortSignal');
assert.equal(seen!.aborted, true, "the task's signal is aborted on timeout");
assert.equal(seen!.reason, (t as { reason: unknown }).reason, "the rejection is the task signal's reason");

// External abort: rejects with exactly the caller's reason, promptly, and aborts the task's signal.
const ctl = new AbortController();
const why = new Error('user cancelled');
let seen2: AbortSignal | undefined;
const pending = withTimeout((signal) => { seen2 = signal; return never(); }, 60_000, { signal: ctl.signal });
setTimeout(() => ctl.abort(why), 20);
assert.deepEqual(await outcome(pending, 3000), { status: 'rejected', reason: why }, 'external abort must reject with its reason');
assert.equal(seen2!.aborted, true);
assert.equal(seen2!.reason, why);

// Already-aborted signal: rejects with its reason and never calls the task.
const pre = new AbortController();
const reason = new Error('already');
pre.abort(reason);
let called = false;
const early = withTimeout(async () => { called = true; return 1; }, 60_000, { signal: pre.signal });
assert.deepEqual(await outcome(early, 3000), { status: 'rejected', reason });
assert.equal(called, false, 'task must not run when the signal is already aborted');

// A task that finishes before the deadline wins, and its signal is not aborted by the call itself.
let seen3: AbortSignal | undefined;
const fast = withTimeout(async (signal) => { seen3 = signal; await new Promise((r) => setTimeout(r, 10)); return 5; }, 5_000);
assert.deepEqual(await outcome(fast, 3000), { status: 'fulfilled', value: 5 });
assert.equal(seen3!.aborted, false);
