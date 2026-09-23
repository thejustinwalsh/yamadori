// Uses the older string form for the texture entry ({ texture: 'float' }); 0.12.5 takes a texture schema such as d.texture2d(d.f32).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const Camera = d.struct({ viewProj: d.mat4x4f, position: d.vec3f }).$name('Camera');
export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');

export const sceneLayout = tgpu
  .bindGroupLayout({
    camera: { uniform: Camera },
    particles: { storage: (n: number) => d.arrayOf(Particle, n), access: 'mutable' },
    lights: { storage: (n: number) => d.arrayOf(d.vec4f, n) },
    albedo: { texture: 'float', viewDimension: '2d' },
    albedoSampler: { sampler: 'filtering' },
  })
  .$idx(2);
