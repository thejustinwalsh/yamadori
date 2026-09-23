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
  let i = in.gid.x;
  if (i >= layout.$.params.count) {
    return;
  }
  layout.$.particles[i].pos += layout.$.particles[i].vel * layout.$.params.dt;
}`
  .$uses({ layout: simLayout })
  .$name('integrate');
