// Calls tgpu.namespace(); in 0.12.5 namespace is only available as tgpu['~unstable'].namespace.
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

export function buildChunks(): [string, string] {
  const names = tgpu.namespace();
  return [tgpu.resolve([physicsStep], { names }), tgpu.resolve([renderColor], { names })];
}
