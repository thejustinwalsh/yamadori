import { tgpu, d } from 'typegpu';

export const Camera = d.struct({ viewProj: d.mat4x4f, position: d.vec3f }).$name('Camera');
export const Particle = d.struct({ pos: d.vec3f, vel: d.vec3f }).$name('Particle');

// Unsized arrayOf is itself a (count) => array constructor; readonly is the default access.
const layout = tgpu.bindGroupLayout({
  camera: { uniform: Camera },
  particles: { storage: d.arrayOf(Particle), access: 'mutable' },
  lights: { storage: d.arrayOf(d.vec4f) },
  albedo: { texture: d.texture2d(), visibility: ['fragment'] },
  albedoSampler: { sampler: 'filtering', visibility: ['fragment'] },
});
layout.$idx(2);
export const sceneLayout = layout;
