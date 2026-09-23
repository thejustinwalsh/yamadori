import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';

export const SimParams = d.struct({ dt: d.f32, gravity: d.vec3f }).$name('SimParams');
export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');
export const Counters = d.struct({ alive: d.atomic(d.u32), spawned: d.atomic(d.u32) }).$name('Counters');

export function createSimBuffers(root: TgpuRoot) {
  const params = root.createUniform(SimParams, { dt: 1 / 60, gravity: d.vec3f(0, -9.81, 0) });
  const particles = root.createMutable(d.arrayOf(Particle, 1024));
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
