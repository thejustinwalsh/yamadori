// Pads by hand with extra fields instead of WGSL @size/@align attributes -- the JS value gains padding keys.
import * as d from 'typegpu/data';

export const LightUniform = d.struct({
  color: d.vec3f,
  intensity: d.f32,
  _pad0: d.vec3f,
  direction: d.vec3f,
  _pad1: d.f32,
  castsShadow: d.u32,
});

export const lightUniformSize: number = d.sizeOf(LightUniform);
