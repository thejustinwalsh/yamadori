// Passes the WebGPU feature name ('shader-f16'); enableExtensions takes WGSL extension names ('f16').
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const HalfVertex = d.struct({ position: d.vec3h, uv: d.vec2h, weight: d.f16 }).$name('HalfVertex');

export const halfWeight = tgpu.fn([HalfVertex], d.f16)`(v: HalfVertex) -> f16 { return v.weight; }`
  .$uses({ HalfVertex })
  .$name('halfWeight');

export function buildHalfShader(): string {
  return tgpu.resolve([halfWeight], { enableExtensions: ['shader-f16'] });
}
