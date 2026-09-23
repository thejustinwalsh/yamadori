import { d, type TgpuRoot } from 'typegpu';

export const SimParams = d.struct({ dt: d.f32, gravity: d.vec3f }).$name('SimParams');
export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');
export const Counters = d.struct({ alive: d.atomic(d.u32), spawned: d.atomic(d.u32) }).$name('Counters');

const corners: [number, number][] = [[-1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1], [1, 1]];

// Raw buffers first, then fixed views of them.
export const createSimBuffers = (root: TgpuRoot) => ({
  params: root
    .createBuffer(SimParams, { dt: 1 / 60, gravity: d.vec3f(0, -9.81, 0) })
    .$usage('uniform')
    .as('uniform'),
  particles: root.createBuffer(d.arrayOf(Particle, 1024)).$usage('storage').as('mutable'),
  counters: root.createBuffer(Counters).$usage('storage'),
  quad: root.createBuffer(d.arrayOf(d.vec2f, 6), corners.map(([x, y]) => d.vec2f(x, y))).$usage('vertex'),
});
