// Leaves the particle storage at the default read-only access, so shaders cannot write it (var<storage, read>).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const Camera = d.struct({ viewProj: d.mat4x4f, position: d.vec3f }).$name('Camera');
export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');

export const sceneLayout = tgpu
  .bindGroupLayout({
    camera: { uniform: Camera },
    particles: { storage: (n: number) => d.arrayOf(Particle, n) },
    lights: { storage: (n: number) => d.arrayOf(d.vec4f, n) },
    albedo: { texture: d.texture2d(d.f32) },
    albedoSampler: { sampler: 'filtering' },
  })
  .$idx(2);
