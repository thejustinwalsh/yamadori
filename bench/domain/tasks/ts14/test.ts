import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { TypedEmitter } from './solution.ts';

// Declared as an interface on purpose: interfaces have no implicit index signature.
interface ChatEvents {
  message: [text: string, from: number];
  typing: [from: number];
  close: [];
}

const e = new TypedEmitter<ChatEvents>();
const log: string[] = [];

assert.equal(e.emit('close'), false, 'emit with no listeners returns false');

const offMsg = e.on('message', (text, from) => {
  type _t = Expect<Equal<typeof text, string>>;
  type _f = Expect<Equal<typeof from, number>>;
  log.push(`A:${text}:${from}`);
});
type _off = Expect<Equal<typeof offMsg, () => void>>;
e.on('message', (text) => log.push(`B:${text}`));
assert.equal(e.emit('message', 'hi', 1), true);
assert.deepEqual(log, ['A:hi:1', 'B:hi'], 'listeners run synchronously in registration order');

offMsg();
offMsg(); // idempotent
e.emit('message', 'again', 2);
assert.deepEqual(log.slice(2), ['B:again']);
assert.equal(e.listenerCount('message'), 1);

// once: runs a single time, and removing it mid-emit must not skip the next listener.
log.length = 0;
e.once('typing', (from) => log.push(`once:${from}`));
e.on('typing', (from) => log.push(`on:${from}`));
e.emit('typing', 7);
e.emit('typing', 8);
assert.deepEqual(log, ['once:7', 'on:7', 'on:8']);
assert.equal(e.listenerCount('typing'), 1);

// A listener added during emit does not run in that emit; one removed during emit still runs (snapshot).
log.length = 0;
const e2 = new TypedEmitter<ChatEvents>();
let offSecond = () => {};
e2.on('close', () => {
  log.push('first');
  e2.on('close', () => log.push('late'));
  offSecond();
});
offSecond = e2.on('close', () => log.push('second'));
e2.emit('close');
assert.deepEqual(log, ['first', 'second']);
log.length = 0;
e2.emit('close');
assert.deepEqual(log, ['first', 'late']);

// The same function registered twice is called twice, and each unsubscribe removes one registration.
const e3 = new TypedEmitter<ChatEvents>();
let hits = 0;
const fn = () => { hits++; };
const off1 = e3.on('close', fn);
e3.on('close', fn);
e3.emit('close');
assert.equal(hits, 2);
off1();
e3.emit('close');
assert.equal(hits, 3);
assert.equal(e3.listenerCount('close'), 1);

// A once-listener can be cancelled before it fires.
const e4 = new TypedEmitter<{ ping: [n: number] }>();
const cancel = e4.once('ping', () => assert.fail('cancelled once-listener ran'));
cancel();
assert.equal(e4.emit('ping', 1), false);

function compileOnly(): void {
  // @ts-expect-error -- missing the `from` argument
  e.emit('message', 'hi');
  // @ts-expect-error -- unknown event
  e.emit('nope');
  // @ts-expect-error -- wrong argument type
  e.emit('typing', 'x');
  // @ts-expect-error -- listener expects an argument the event does not provide
  e.on('close', (x: string) => log.push(x));
  // @ts-expect-error -- unknown event
  e.on('opened', () => {});
}
void compileOnly;
