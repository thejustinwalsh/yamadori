// Passes the attribute arguments in the wrong order: d.size / d.align take (bytes, schema), not (schema, bytes).
import * as d from 'typegpu/data';

export const LightUniform = d.struct({
  color: d.vec3f,
  intensity: d.size(d.f32, 16),
  direction: d.vec3f,
  castsShadow: d.align(d.u32, 16),
});

export const lightUniformSize: number = d.sizeOf(LightUniform);
