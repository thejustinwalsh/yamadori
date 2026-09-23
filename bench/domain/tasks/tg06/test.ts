import assert from 'node:assert/strict';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { hasNoPadding } from './solution.ts';

type _sig = Expect<Equal<ReturnType<typeof hasNoPadding>, boolean>>;
// Type-only negative case: never executed.
export const _neg = () => {
  // @ts-expect-error -- takes a schema, not a value
  hasNoPadding(d.vec3f(1, 2, 3).x);
};

// [name, schema, expected] -- expected values follow WGSL layout rules and
// typegpu 0.12.5's own layout (vec3f aligns to 16, mat3x3f columns are
// vec3f-strided, unstruct/disarray are packed).
const cases: [string, d.AnyData, boolean][] = [
  ['f32', d.f32, true],
  ['vec3f alone', d.vec3f, true],
  ['struct {vec3f, f32}', d.struct({ a: d.vec3f, b: d.f32 }), true],
  ['struct {f32, vec3f}: gap before vec3f', d.struct({ a: d.f32, b: d.vec3f }), false],
  ['struct {vec2f, f32}: trailing pad', d.struct({ a: d.vec2f, b: d.f32 }), false],
  ['array<vec3f, 4>: 16-byte stride', d.arrayOf(d.vec3f, 4), false],
  ['array<vec4f, 4>', d.arrayOf(d.vec4f, 4), true],
  ['nested struct with inner gap', d.struct({ inner: d.struct({ a: d.f32, b: d.vec3f }) }), false],
  ['nested packed', d.struct({ inner: d.struct({ a: d.vec3f, b: d.f32 }), c: d.vec4f }), true],
  ['@size(16) f32', d.struct({ a: d.size(16, d.f32) }), false],
  ['@align(16) second member', d.struct({ a: d.f32, b: d.align(16, d.f32) }), false],
  ['array of packed structs', d.arrayOf(d.struct({ a: d.vec3f, b: d.f32 }), 3), true],
  ['array of padded structs', d.arrayOf(d.struct({ a: d.vec2f, b: d.f32 }), 2), false],
  ['mat3x3f', d.mat3x3f, false],
  ['mat4x4f', d.mat4x4f, true],
  ['unstruct {f32, vec3f}', d.unstruct({ a: d.f32, b: d.vec3f }), true],
  ['disarray<vec3f, 3>', d.disarrayOf(d.vec3f, 3), true],
  ['unstruct of vertex formats', d.unstruct({ p: d.float32x3, c: d.unorm8x4 }), true],
  ['struct {f16, f32}', d.struct({ a: d.f16, b: d.f32 }), false],
  ['array<f16, 3>', d.arrayOf(d.f16, 3), true],
];

for (const [name, schema, expected] of cases) {
  assert.equal(hasNoPadding(schema), expected, name);
}
