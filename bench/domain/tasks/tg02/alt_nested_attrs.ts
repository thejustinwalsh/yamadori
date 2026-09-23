import tgpu from 'typegpu';
import { align, f32, size, sizeOf, struct, u32, vec3f } from 'typegpu/data';

// Attributes composed differently (redundant explicit align/size on the same members).
const Light = struct({
  color: vec3f,
  intensity: size(16, align(4, f32)),
  direction: vec3f,
  castsShadow: align(16, size(4, u32)),
});

export const LightUniform = Light;
export const lightUniformSize = sizeOf(Light);
void tgpu;
