import * as d from 'typegpu/data';

export const Instance = d.struct({ transform: d.mat4x4f, color: d.vec3f, id: d.u32, velocity: d.vec3f });
export const Instances = d.arrayOf(Instance, 64);

export function greenChannelRange(index: number): { offset: number; contiguous: number } {
  const { offset, contiguous } = d.memoryLayoutOf(Instances, (all) => all[index]!.color.y);
  return { offset, contiguous };
}
