// Uses the free functions asUniform/asMutable from older typegpu; 0.12.5 does not export them (use root.createUniform/createMutable or buffer.as(...)).
import tgpu, { asMutable, asUniform, type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';

export const SimParams = d.struct({ dt: d.f32, gravity: d.vec3f }).$name('SimParams');
export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');
export const Counters = d.struct({ alive: d.atomic(d.u32), spawned: d.atomic(d.u32) }).$name('Counters');

export function createSimBuffers(root: TgpuRoot) {
  const params = asUniform(root.createBuffer(SimParams, { dt: 1 / 60, gravity: d.vec3f(0, -9.81, 0) }).$usage('uniform'));
  const particles = asMutable(root.createBuffer(d.arrayOf(Particle, 1024)).$usage('storage'));
  const counters = root.createBuffer(Counters).$usage('storage');
  const quad = root
    .createBuffer(d.arrayOf(d.vec2f, 6), [
      d.vec2f(-1, -1), d.vec2f(1, -1), d.vec2f(-1, 1),
      d.vec2f(-1, 1), d.vec2f(1, -1), d.vec2f(1, 1),
    ])
    .$usage('vertex');
  return { params, particles, counters, quad };
}
void tgpu;
