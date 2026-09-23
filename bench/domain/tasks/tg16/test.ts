import assert from 'node:assert/strict';
import { readFromArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { packParticles, Particle } from './solution.ts';

type _p = Expect<Equal<d.Infer<typeof Particle>, { position: d.v3f; velocity: d.v3f; color: d.v4f; age: number }>>;
type _f = Expect<Equal<ReturnType<typeof packParticles>, ArrayBuffer>>;
export const _neg = () => {
  // @ts-expect-error -- takes particle values, not a typed array
  packParticles(new Float32Array(16));
};

assert.equal(d.sizeOf(Particle), 64);

const ps: d.Infer<typeof Particle>[] = [
  { position: d.vec3f(1, 2, 3), velocity: d.vec3f(4, 5, 6), color: d.vec4f(0.1, 0.2, 0.3, 0.4), age: 7 },
  { position: d.vec3f(-1, -2, -3), velocity: d.vec3f(-4, -5, -6), color: d.vec4f(1, 1, 0, 1), age: 8.5 },
  { position: d.vec3f(9, 10, 11), velocity: d.vec3f(12, 13, 14), color: d.vec4f(0, 0, 0, 0), age: 0.25 },
];
const buf = packParticles(ps);
assert.ok(buf instanceof ArrayBuffer, 'returns an ArrayBuffer');
assert.equal(buf.byteLength, 3 * 64, 'byteLength of array<Particle, 3>');

const dv = new DataView(buf);
const f = (o: number) => dv.getFloat32(o, true);
const near = (a: number, b: number, what: string) => assert.ok(Math.abs(a - b) < 1e-6, `${what}: ${a} vs ${b}`);
ps.forEach((p, i) => {
  const b = i * 64;
  [0, 1, 2].forEach((k) => near(f(b + 0 + 4 * k), p.position[k]!, `p${i}.position[${k}] @${b + 4 * k}`));
  [0, 1, 2].forEach((k) => near(f(b + 16 + 4 * k), p.velocity[k]!, `p${i}.velocity[${k}] @${b + 16 + 4 * k}`));
  [0, 1, 2, 3].forEach((k) => near(f(b + 32 + 4 * k), p.color[k]!, `p${i}.color[${k}] @${b + 32 + 4 * k}`));
  near(f(b + 48), p.age, `p${i}.age @${b + 48}`);
});

// Round-trip through typegpu's own reader.
const back = readFromArrayBuffer(buf, d.arrayOf(Particle, 3));
near(back[2]!.velocity.y, 13, 'round trip');

