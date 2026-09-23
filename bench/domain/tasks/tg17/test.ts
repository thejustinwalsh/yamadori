import assert from 'node:assert/strict';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { Particle, setVelocity } from './solution.ts';

type _p = Expect<Equal<d.Infer<typeof Particle>, { position: d.v3f; velocity: d.v3f; color: d.v4f; age: number }>>;
type _s = Expect<Equal<Parameters<typeof setVelocity>, [buffer: ArrayBuffer, count: number, index: number, velocity: d.v3f]>>;
export const _neg = () => {
  // @ts-expect-error -- velocity is a vec3f
  setVelocity(new ArrayBuffer(64), 1, 0, d.vec4f());
};

assert.equal(d.sizeOf(Particle), 64);

// Fill with a sentinel so any stray write -- padding included -- is visible.
function run(count: number, index: number, v: d.v3f) {
  const buf = new ArrayBuffer(count * 64);
  new Uint8Array(buf).fill(0xab);
  setVelocity(buf, count, index, v);
  const bytes = new Uint8Array(buf);
  const changed: number[] = [];
  bytes.forEach((b, i) => { if (b !== 0xab) changed.push(i); });
  return { buf, changed };
}

for (const [count, index] of [[3, 1], [4, 0], [4, 3]] as const) {
  const v = d.vec3f(1.5, -2.25, 1e3);
  const { buf, changed } = run(count, index, v);
  const start = index * 64 + 16;
  const dv = new DataView(buf);
  assert.deepEqual([dv.getFloat32(start, true), dv.getFloat32(start + 4, true), dv.getFloat32(start + 8, true)],
    [1.5, -2.25, 1000], `particles[${index}].velocity at byte ${start} (count ${count})`);
  assert.ok(changed.every((i) => i >= start && i < start + 12),
    `only bytes ${start}..${start + 11} may change (count ${count}, index ${index}); changed: ${changed.slice(0, 20).join(',')}`);
}
