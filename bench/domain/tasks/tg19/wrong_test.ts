// Resolves without enabling the f16 extension, so the WGSL lacks `enable f16;` and fails to compile on a real device.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const HalfVertex = d.struct({ position: d.vec3h, uv: d.vec2h, weight: d.f16 }).$name('HalfVertex');

export const halfWeight = tgpu.fn([HalfVertex], d.f16)`(v: HalfVertex) -> f16 { return v.weight; }`
  .$uses({ HalfVertex })
  .$name('halfWeight');

export function buildHalfShader(): string {
  return tgpu.resolve([halfWeight]);
}
