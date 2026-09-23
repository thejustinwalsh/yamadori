import { tgpu, d } from 'typegpu';

export const HalfVertex = d.struct({ position: d.vec3h, uv: d.vec2h, weight: d.f16 });
HalfVertex.$name('HalfVertex');

export const halfWeight = tgpu.fn([HalfVertex], d.f16)('(v: HalfVertex) -> f16 { return v.weight; }');
halfWeight.$name('halfWeight');

// resolveWithContext takes the same options and also reports what was used.
export const buildHalfShader = (): string =>
  tgpu.resolveWithContext([halfWeight], { enableExtensions: ['f16'] }).code;
