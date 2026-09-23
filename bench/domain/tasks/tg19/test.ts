import assert from 'node:assert/strict';
import type { TgpuFn } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { buildHalfShader, HalfVertex, halfWeight } from './solution.ts';

type _v = Expect<Equal<d.Infer<typeof HalfVertex>, { position: d.v3h; uv: d.v2h; weight: number }>>;
type _s = Expect<Equal<typeof HalfVertex, d.WgslStruct<{ position: d.Vec3h; uv: d.Vec2h; weight: d.F16 }>>>;
const _f: TgpuFn<(...args: [typeof HalfVertex]) => d.F16> = halfWeight;
export const _neg = () => {
  // @ts-expect-error -- weight is f16, not f32
  const _x: d.WgslStruct<{ position: d.Vec3h; uv: d.Vec2h; weight: d.F32 }> = HalfVertex;
};

assert.equal(d.sizeOf(HalfVertex), 16, 'vec3h@0, vec2h@8, f16@12 -> 16');

const wgsl = buildHalfShader();
const statements = wgsl.replace(/\/\/[^\n]*/g, '').trim();
assert.match(statements, /^enable\s+f16\s*;/, 'the shader must start with `enable f16;`\n' + wgsl);
assert.equal(wgsl.match(/\benable\s+f16\b/g)?.length, 1, 'exactly one enable directive\n' + wgsl);
assert.match(wgsl, /struct\s+HalfVertex\s*\{\s*position\s*:\s*vec3h\s*,\s*uv\s*:\s*vec2h\s*,\s*weight\s*:\s*f16\s*,?\s*\}/, wgsl);
assert.match(wgsl, /fn\s+halfWeight\s*\(\s*v\s*:\s*HalfVertex\s*\)\s*->\s*f16\s*\{\s*return\s+v\.weight\s*;\s*\}/, wgsl);
assert.ok(wgsl.indexOf('struct HalfVertex') < wgsl.indexOf('fn halfWeight'), 'struct is declared before use');
assert.match(wgsl.trim(), /\}$/, 'nothing but declarations: the shader ends with the function body\n' + wgsl);
