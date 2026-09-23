import assert from 'node:assert/strict';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { Particle, particleByteSize, type ParticleValue } from './solution.ts';

type _v = Expect<Equal<ParticleValue, { position: d.v3f; velocity: d.v3f; mass: number }>>;
type _s = Expect<Equal<typeof Particle, d.WgslStruct<{ position: d.Vec3f; velocity: d.Vec3f; mass: d.F32 }>>>;

assert.equal(particleByteSize, 32, 'vec3f aligns to 16: 0,16,28 -> 32');
assert.equal(d.sizeOf(Particle), 32);
assert.equal(d.alignmentOf(Particle), 16);
const wgsl = tgpu.resolve([Particle]);
assert.match(wgsl, /position: vec3f,\s*velocity: vec3f,\s*mass: f32/);
