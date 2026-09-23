// Measures inside a single Instance and forgets the array element offset.
import * as d from 'typegpu/data';

export const Instance = d.struct({ transform: d.mat4x4f, color: d.vec3f, id: d.u32, velocity: d.vec3f });
export const Instances = d.arrayOf(Instance, 64);

export function greenChannelRange(_index: number): { offset: number; contiguous: number } {
  return d.memoryLayoutOf(Instance, (inst) => inst.color.y);
}
