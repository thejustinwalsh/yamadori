import assert from 'node:assert/strict';
import tgpu, { type TgpuConst, type TgpuVar } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { GAUSS_WEIGHTS } from './solution.ts';

type _t = Expect<Equal<typeof GAUSS_WEIGHTS, TgpuConst<d.WgslArray<d.F32>>>>;
// @ts-expect-error -- a module constant is not a variable
const _notVar: TgpuVar = GAUSS_WEIGHTS;

const wgsl = tgpu.resolve([GAUSS_WEIGHTS]);
const m = /\bconst\s+(\w+)\s*:\s*array<\s*f32\s*,\s*5\s*>\s*=\s*array<\s*f32\s*,\s*5\s*>\s*\(([^)]*)\)/.exec(wgsl);
assert.ok(m, 'expected a `const NAME: array<f32, 5> = array<f32, 5>(...)` declaration, got:\n' + wgsl);
assert.equal(m[1], 'GAUSS_WEIGHTS', 'WGSL identifier of the constant');
const values = m[2].split(',').map((s) => Number.parseFloat(s.trim()));
const expected = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216];
assert.equal(values.length, 5);
values.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-6, `weight ${i}: ${v}`));
assert.doesNotMatch(wgsl, /var\s*</, 'must not be a variable');

// Used from a WGSL function, it is referenced by name and declared exactly once.
const weight = tgpu.fn([d.u32], d.f32)`(i: u32) -> f32 { return W[i]; }`.$uses({ W: GAUSS_WEIGHTS });
const fnWgsl = tgpu.resolve([weight]);
assert.match(fnWgsl, /return\s+GAUSS_WEIGHTS\s*\[\s*i\s*\]/, fnWgsl);
assert.equal(fnWgsl.match(/\bconst\s+GAUSS_WEIGHTS\b/g)?.length, 1, 'declared once');
