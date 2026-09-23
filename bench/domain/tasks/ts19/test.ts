import assert from 'node:assert/strict';
import type { Equal, Expect } from './assert_types.ts';
import { createMachine } from './solution.ts';

const m = createMachine({ states: ['idle', 'running', 'done'], initial: 'idle' });
type _cur = Expect<Equal<typeof m.current, 'idle' | 'running' | 'done'>>;
type _go = Expect<Equal<Parameters<typeof m.go>, [to: 'idle' | 'running' | 'done']>>;
assert.equal(m.current, 'idle');
m.go('running');
assert.equal(m.current, 'running');
m.go('done');
assert.equal(m.current, 'done');
assert.deepEqual([...m.states], ['idle', 'running', 'done']);
assert.throws(() => (m.go as (to: string) => void)('crashed'), RangeError);
assert.equal(m.current, 'done');

const single = createMachine({ states: ['only'], initial: 'only' });
type _single = Expect<Equal<typeof single.current, 'only'>>;
assert.equal(single.current, 'only');

const states = ['a', 'b'] as const;
const fromConst = createMachine({ states, initial: 'b' });
type _fc = Expect<Equal<typeof fromConst.current, 'a' | 'b'>>;
assert.equal(fromConst.current, 'b');

function compileOnly(): void {
  // @ts-expect-error -- 'stopped' is not one of the states
  createMachine({ states: ['idle', 'running'], initial: 'stopped' });
  // @ts-expect-error -- go() only accepts known states
  m.go('paused');
  // @ts-expect-error -- initial is required
  createMachine({ states: ['x'] });
}
void compileOnly;
