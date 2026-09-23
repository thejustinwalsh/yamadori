import { tgpu, d } from 'typegpu';

export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');
export const SimParams = d.struct({ dt: d.f32, count: d.u32 }).$name('SimParams');

export const simLayout = tgpu.bindGroupLayout({
  params: { uniform: SimParams },
  particles: { storage: d.arrayOf(Particle), access: 'mutable' },
});

const body = /* wgsl */ `{
  let idx = globalId.x;
  if (idx >= L.$.params.count) { return; }
  let p = L.$.particles[idx];
  L.$.particles[idx].pos = p.pos + p.vel * L.$.params.dt;
}`;

export const integrate = tgpu
  .computeFn({ in: { globalId: d.builtin.globalInvocationId }, workgroupSize: [64, 1, 1] })(body)
  .$uses({ L: simLayout });
integrate.$name('integrate');
