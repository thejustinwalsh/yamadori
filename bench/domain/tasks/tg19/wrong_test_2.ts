// Uses full-precision vec3f/vec2f/f32 fields instead of the half-precision f16 types.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const HalfVertex = d.struct({ position: d.vec3f, uv: d.vec2f, weight: d.f32 }).$name('HalfVertex');

export const halfWeight = tgpu.fn([HalfVertex], d.f32)`(v: HalfVertex) -> f32 { return v.weight; }`
  .$uses({ HalfVertex })
  .$name('halfWeight');

export function buildHalfShader(): string {
  return tgpu.resolve([halfWeight], { enableExtensions: ['f16'] });
}
