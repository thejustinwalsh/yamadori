// Swaps the two attributes: aligns intensity to 16 and sizes castsShadow to 16, so the offsets come out 16/32/44.
import * as d from 'typegpu/data';

export const LightUniform = d.struct({
  color: d.vec3f,
  intensity: d.align(16, d.f32),
  direction: d.vec3f,
  castsShadow: d.size(16, d.u32),
});

export const lightUniformSize: number = d.sizeOf(LightUniform);
