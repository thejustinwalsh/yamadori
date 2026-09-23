// Writes a full WGSL parameter list in the string (older typegpu style); 0.12.5 expects only the body for entry functions, so the header is emitted twice.
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
})`(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= layout.$.params.count) { return; }
  layout.$.particles[i].pos += layout.$.particles[i].vel * layout.$.params.dt;
}`
  .$uses({ layout: simLayout })
  .$name('integrate');
