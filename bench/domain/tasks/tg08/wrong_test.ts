// Hand-computed: uses the unpadded struct size (92) as the array stride and counts only the rest of the vec3f (8 bytes) as contiguous.
import * as d from 'typegpu/data';

export const Instance = d.struct({ transform: d.mat4x4f, color: d.vec3f, id: d.u32, velocity: d.vec3f });
export const Instances = d.arrayOf(Instance, 64);

export function greenChannelRange(index: number): { offset: number; contiguous: number } {
  const stride = 64 + 12 + 4 + 12;
  return { offset: index * stride + 64 + 4, contiguous: 8 };
}
