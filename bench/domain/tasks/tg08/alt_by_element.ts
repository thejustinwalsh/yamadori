import * as d from 'typegpu/data';

export const Instance = d.struct({ transform: d.mat4x4f, color: d.vec3f, id: d.u32, velocity: d.vec3f });
export const Instances = d.arrayOf(Instance, 64);

// Offset inside one element, plus the element stride (sizeOf includes trailing padding).
const inElement = d.memoryLayoutOf(Instance, (inst) => inst.color.y);

export function greenChannelRange(index: number) {
  return { offset: index * d.sizeOf(Instance) + inElement.offset, contiguous: inElement.contiguous };
}
