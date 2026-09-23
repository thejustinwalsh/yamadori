// Pulls the bindings out of layout.$ eagerly at module scope; outside shader generation that access throws in 0.12.5.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');
export const SimParams = d.struct({ dt: d.f32, count: d.u32 }).$name('SimParams');

export const simLayout = tgpu.bindGroupLayout({
  params: { uniform: SimParams },
  particles: { storage: (n: number) => d.arrayOf(Particle, n), access: 'mutable' },
});

export const integrate = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [64],
})`{
  let i = gid.x;
  if (i >= params.count) { return; }
  particles[i].pos += particles[i].vel * params.dt;
}`
  .$uses({ params: simLayout.$.params, particles: simLayout.$.particles })
  .$name('integrate');
