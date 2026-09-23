// Alternative correct answer: namespace from a destructured unstable API, strict names requested explicitly.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');
export const speed = tgpu.fn([Particle], d.f32)`(p: Particle) -> f32 { return length(p.vel); }`.$name('speed');
export const physicsStep = tgpu.fn([Particle, d.f32], Particle)`(p: Particle, dt: f32) -> Particle {
  var q = p;
  q.pos += q.vel * dt * speed(p);
  return q;
}`.$uses({ speed }).$name('physicsStep');
export const renderColor = tgpu.fn([Particle], d.vec4f)`(p: Particle) -> vec4f {
  return vec4f(vec3f(speed(p)), 1.0);
}`.$uses({ speed }).$name('renderColor');

import { tgpu as t } from 'typegpu';

export const buildChunks = (): [string, string] => {
  const { namespace } = t['~unstable'];
  const ns = namespace({ names: 'strict' });
  return [physicsStep, renderColor].map((item) => t.resolve([item], { names: ns })) as [string, string];
};
