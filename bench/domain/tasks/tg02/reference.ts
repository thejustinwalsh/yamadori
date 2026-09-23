import * as d from 'typegpu/data';

export const LightUniform = d.struct({
  color: d.vec3f,
  intensity: d.size(16, d.f32),
  direction: d.vec3f,
  castsShadow: d.align(16, d.u32),
});

export const lightUniformSize: number = d.sizeOf(LightUniform);
